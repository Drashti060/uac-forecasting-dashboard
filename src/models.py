"""Forecasting models, strict walk-forward validation and metrics.

Model zoo (all share the ``BaseForecaster`` interface)
-------------------------------------------------------
* Benchmarks      : ``Naive`` (persistence), ``MA7``, ``SeasonalNaive7``
* Statistical     : ``SARIMAX`` (weekly seasonal), ``Prophet``
* Machine learning: ``LightGBM`` / ``XGBoost`` (falls back to sklearn
  ``HistGBM`` if neither is installed) and ``RandomForest``. These use the
  *direct* multi-horizon strategy: one model per horizon h, trained to predict
  ``y[t+h] - y[t]``.
* ``Ensemble``    : weighted mean of SARIMAX and the primary GBM.

Walk-forward protocol (no leakage)
----------------------------------
For each forecast origin ``T`` (stepping forward through time) the model sees
only rows ``<= T``. Direct models additionally drop the last ``h`` rows whose
labels would lie beyond ``T``. Prediction intervals for models without native
ones come from an *expanding conformal-style residual calibrator* that only
uses residuals whose target date is ``<= T``.
"""
from __future__ import annotations

import importlib.util
import logging
import warnings
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import config
from .features import make_supervised

log = logging.getLogger(__name__)

BAND_COLS = ["lo80", "hi80", "lo95", "hi95"]
LEVELS = {80: (0.10, 0.90), 95: (0.025, 0.975)}


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


GBM_KIND = "lightgbm" if _has("lightgbm") else ("xgboost" if _has("xgboost") else "hist")
GBM_NAME = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "hist": "HistGBM"}[GBM_KIND]


# ===================================================================== output
def _frame(origin: pd.Timestamp, horizons: Iterable[int], yhat, bands: dict | None = None) -> pd.DataFrame:
    hs = [int(h) for h in horizons]
    df = pd.DataFrame(
        {"date": [origin + pd.Timedelta(days=h) for h in hs], "yhat": np.asarray(yhat, dtype=float)},
        index=pd.Index(hs, name="h"),
    )
    for c in BAND_COLS:
        df[c] = np.nan
    for k, v in (bands or {}).items():
        df[k] = np.asarray(v, dtype=float)
    return df


class BaseForecaster:
    """fit(feat, horizons) then predict() -> DataFrame indexed by h."""

    name = "base"

    def fit(self, feat: pd.DataFrame, horizons: Iterable[int]) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self) -> pd.DataFrame:
        raise NotImplementedError


# ================================================================= benchmarks
class NaivePersistence(BaseForecaster):
    name = "Naive"

    def fit(self, feat, horizons, target=config.TARGET):
        self.origin, self.horizons = feat.index[-1], tuple(horizons)
        self.last = float(feat[target].iloc[-1])
        return self

    def predict(self):
        return _frame(self.origin, self.horizons, [self.last] * len(self.horizons))


class MovingAverage(BaseForecaster):
    def __init__(self, window: int = 7):
        self.window, self.name = window, f"MA{window}"

    def fit(self, feat, horizons, target=config.TARGET):
        self.origin, self.horizons = feat.index[-1], tuple(horizons)
        self.mean = float(feat[target].iloc[-self.window:].mean())
        return self

    def predict(self):
        return _frame(self.origin, self.horizons, [self.mean] * len(self.horizons))


class SeasonalNaive(BaseForecaster):
    """Value from the same weekday in the most recent observed week."""

    def __init__(self, period: int = 7):
        self.period, self.name = period, f"SeasonalNaive{period}"

    def fit(self, feat, horizons, target=config.TARGET):
        self.origin, self.horizons = feat.index[-1], tuple(horizons)
        self.y = feat[target].to_numpy(dtype=float)
        return self

    def predict(self):
        p = self.period
        yhat = [self.y[-1 + (h - p * int(np.ceil(h / p)))] for h in self.horizons]
        return _frame(self.origin, self.horizons, yhat)


# ============================================================== statistical
class SARIMAXModel(BaseForecaster):
    """Univariate SARIMAX with a weekly seasonal component (native 80/95% bands)."""

    name = "SARIMAX"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, config.SEASONAL_PERIOD), maxiter: int = 60):
        self.order, self.seasonal_order, self.maxiter = order, seasonal_order, maxiter

    def fit(self, feat, horizons, target=config.TARGET):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self.origin, self.horizons = feat.index[-1], tuple(horizons)
        y = feat[target].astype(float).asfreq("D")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(
                y, order=self.order, seasonal_order=self.seasonal_order, trend="n",
                enforce_stationarity=False, enforce_invertibility=False,
            )
            self.res = mod.fit(disp=False, maxiter=self.maxiter)
        return self

    def predict(self):
        steps = max(self.horizons)
        fc = self.res.get_forecast(steps=steps)
        pos = [h - 1 for h in self.horizons]
        mean = np.asarray(fc.predicted_mean)[pos]
        ci80 = np.asarray(fc.conf_int(alpha=0.20))[pos]
        ci95 = np.asarray(fc.conf_int(alpha=0.05))[pos]
        bands = {"lo80": ci80[:, 0], "hi80": ci80[:, 1], "lo95": ci95[:, 0], "hi95": ci95[:, 1]}
        return _frame(self.origin, self.horizons, mean, bands)


class ProphetModel(BaseForecaster):
    """Prophet with weekly seasonality + US holidays; bands from posterior samples."""

    name = "Prophet"

    def fit(self, feat, horizons, target=config.TARGET):
        from prophet import Prophet

        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        logging.getLogger("prophet").setLevel(logging.WARNING)
        self.origin, self.horizons = feat.index[-1], tuple(horizons)
        df = pd.DataFrame({"ds": feat.index, "y": feat[target].to_numpy(dtype=float)})
        m = Prophet(
            weekly_seasonality=True, yearly_seasonality="auto", daily_seasonality=False,
            interval_width=0.95, changepoint_prior_scale=0.1,
        )
        m.add_country_holidays(country_name="US")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df)
        self.m = m
        return self

    def predict(self):
        future = pd.DataFrame({"ds": [self.origin + pd.Timedelta(days=h) for h in self.horizons]})
        yhat = self.m.predict(future)["yhat"].to_numpy()
        draws = np.asarray(self.m.predictive_samples(future)["yhat"])  # (n_dates, n_samples)
        q = lambda p: np.quantile(draws, p, axis=1)  # noqa: E731
        bands = {"lo80": q(0.10), "hi80": q(0.90), "lo95": q(0.025), "hi95": q(0.975)}
        return _frame(self.origin, self.horizons, yhat, bands)


# ======================================================================= ML
def make_regressor(kind: str, seed: int = config.SEED):
    if kind == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=15,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
            random_state=seed, verbose=-1,
        )
    if kind == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.7, reg_lambda=1.0, random_state=seed, n_jobs=-1,
        )
    if kind == "hist":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=15,
            l2_regularization=1.0, random_state=seed,
        )
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=seed,
        )
    raise ValueError(kind)


class DirectRegressorForecaster(BaseForecaster):
    """One regressor per horizon (direct strategy), predicting the census *change*."""

    def __init__(self, name: str, kind: str, delta: bool = True, min_train_rows: int = 60):
        self.name, self.kind, self.delta, self.min_train_rows = name, kind, delta, min_train_rows

    def fit(self, feat, horizons, target=config.TARGET):
        self.origin, self.horizons = feat.index[-1], tuple(horizons)
        self.y_last = float(feat[target].iloc[-1])
        self.models, self.x_last = {}, {}
        for h in self.horizons:
            X, y = make_supervised(feat, h, target=target, delta=self.delta)
            ok = y.notna() & X.notna().all(axis=1)
            if ok.sum() < self.min_train_rows:
                raise ValueError(f"{self.name}: only {int(ok.sum())} usable rows for h={h}")
            est = make_regressor(self.kind)
            est.fit(X[ok], y[ok])
            xl = X.iloc[[-1]].copy()
            self.x_last[h] = xl.fillna(X[ok].median())
            self.models[h] = est
        return self

    def predict(self):
        base = self.y_last if self.delta else 0.0
        yhat = [base + float(self.models[h].predict(self.x_last[h])[0]) for h in self.horizons]
        return _frame(self.origin, self.horizons, yhat)


# =========================================================== interval calibration
class ResidualCalibrator:
    """Empirical (conformal-style) interval offsets learned from past forecast errors.

    Residual = y_true - yhat, kept per horizon. Quantile offsets are linearly
    interpolated across horizons and anchored at 0 for h=0, so a smooth fan
    can be drawn for any h in 1..30 even though backtests only score 7/14/30.
    Beyond the largest calibrated horizon the offsets are scaled by sqrt(h/h_max).
    """

    def __init__(self, residuals: dict[int, np.ndarray], min_obs: int = config.CALIBRATION_MIN_OBS):
        self.h: list[int] = []
        self.off: dict[int, list[tuple[float, float]]] = {80: [], 95: []}
        for h in sorted(residuals):
            r = np.asarray(residuals[h], dtype=float)
            r = r[np.isfinite(r)]
            if len(r) < min_obs:
                continue
            self.h.append(int(h))
            for lvl, (a, b) in LEVELS.items():
                lo, hi = np.quantile(r, [a, b])
                self.off[lvl].append((float(lo), float(hi)))

    @property
    def ready(self) -> bool:
        return len(self.h) > 0

    def offsets(self, h: float, level: int) -> tuple[float, float]:
        if not self.ready:
            return (np.nan, np.nan)
        xs = [0.0] + [float(x) for x in self.h]
        lo = [0.0] + [o[0] for o in self.off[level]]
        hi = [0.0] + [o[1] for o in self.off[level]]
        scale = np.sqrt(h / xs[-1]) if h > xs[-1] else 1.0
        return float(np.interp(h, xs, lo) * scale), float(np.interp(h, xs, hi) * scale)

    def apply(self, fc: pd.DataFrame) -> pd.DataFrame:
        out = fc.copy()
        for h in out.index:
            for lvl in (80, 95):
                lo, hi = self.offsets(h, lvl)
                out.loc[h, f"lo{lvl}"] = out.loc[h, "yhat"] + lo
                out.loc[h, f"hi{lvl}"] = out.loc[h, "yhat"] + hi
        return out

    @classmethod
    def from_frame(cls, pred: pd.DataFrame, min_obs: int = config.CALIBRATION_MIN_OBS):
        res = {int(h): (g["y_true"] - g["yhat"]).to_numpy() for h, g in pred.groupby("h")}
        return cls(res, min_obs=min_obs)

    @classmethod
    def from_history(cls, y: pd.Series, horizons: Iterable[int] = config.HORIZONS,
                     lookback: int = 365, min_obs: int = config.CALIBRATION_MIN_OBS):
        """Uncalibrated fallback: scale of historical h-day census changes (naive-error scale)."""
        res = {}
        for h in horizons:
            d = (y.shift(-h) - y).dropna().tail(lookback)
            res[int(h)] = (d - d.mean()).to_numpy()
        return cls(res, min_obs=min_obs)


def sanitize_bands(fc: pd.DataFrame) -> pd.DataFrame:
    """Nest 95% around 80% around yhat and floor at zero (a census can't be negative)."""
    out = fc.copy()
    out["lo80"] = np.minimum(out["lo80"], out["yhat"])
    out["hi80"] = np.maximum(out["hi80"], out["yhat"])
    out["lo95"] = np.minimum(out["lo95"], out["lo80"])
    out["hi95"] = np.maximum(out["hi95"], out["hi80"])
    for c in BAND_COLS:
        out[c] = out[c].clip(lower=0)
    return out


# ============================================================ model registry
def build_factories(include_slow: bool = True) -> dict[str, Callable[[], BaseForecaster]]:
    """Name -> zero-arg constructor for every model that can run in this environment."""
    f: dict[str, Callable[[], BaseForecaster]] = {
        "Naive": NaivePersistence,
        "MA7": lambda: MovingAverage(7),
        "SeasonalNaive7": lambda: SeasonalNaive(7),
    }
    if _has("statsmodels"):
        f["SARIMAX"] = SARIMAXModel
    else:
        log.warning("statsmodels not installed - SARIMAX skipped")
    if include_slow and _has("prophet"):
        f["Prophet"] = ProphetModel
    elif include_slow:
        log.warning("prophet not installed - Prophet skipped")
    f[GBM_NAME] = lambda: DirectRegressorForecaster(GBM_NAME, GBM_KIND)
    if _has("xgboost") and GBM_NAME != "XGBoost":
        f["XGBoost"] = lambda: DirectRegressorForecaster("XGBoost", "xgboost")
    f["RandomForest"] = lambda: DirectRegressorForecaster("RandomForest", "rf")
    return f


def ensemble_members(available: Iterable[str]) -> list[str]:
    avail = set(available)
    members = [m for m in ("SARIMAX", GBM_NAME) if m in avail]
    return members if len(members) >= 2 else [m for m in (GBM_NAME, "RandomForest") if m in avail]


# ========================================================== walk-forward CV
def walk_forward_backtest(
    feat: pd.DataFrame,
    factories: dict[str, Callable[[], BaseForecaster]],
    horizons: tuple[int, ...] = config.HORIZONS,
    initial_train: int | None = None,
    step: int = config.WF_STEP_DAYS,
    max_folds: int | None = config.WF_MAX_FOLDS,
    target: str = config.TARGET,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Expanding-window walk-forward evaluation. Returns a long prediction frame.

    Columns: model, origin, h, target_date, yhat, y_true, lo80..hi95.
    Only folds whose *every* horizon has an observed outcome are scored.
    """
    n, H = len(feat), max(horizons)
    if initial_train is None:
        initial_train = min(config.INITIAL_TRAIN_DAYS, max(120, n // 2))
    positions = list(range(initial_train - 1, n - H, step))
    if max_folds:
        positions = positions[-max_folds:]
    if not positions:
        raise ValueError(f"Series too short ({n} days) for initial_train={initial_train}, H={H}")

    actual = feat[target]
    rows = []
    for k, i in enumerate(positions, 1):
        train = feat.iloc[: i + 1]                      # <-- nothing after the origin is visible
        origin = train.index[-1]
        for name, make in factories.items():
            try:
                model = make().fit(train, horizons)
                p = model.predict().reset_index()
            except Exception as exc:                    # keep the backtest alive
                log.warning("fold %s model %s failed: %s", origin.date(), name, exc)
                continue
            p.insert(0, "model", name)
            p.insert(1, "origin", origin)
            rows.append(p.rename(columns={"date": "target_date"}))
        if progress:
            progress(k, len(positions))

    pred = pd.concat(rows, ignore_index=True)
    pred["y_true"] = actual.reindex(pred["target_date"]).to_numpy()
    pred = add_ensemble(pred)
    pred = apply_expanding_calibration(pred)
    return pred


def add_ensemble(pred: pd.DataFrame, name: str = "Ensemble", weights: dict[str, float] | None = None) -> pd.DataFrame:
    members = ensemble_members(pred["model"].unique())
    if len(members) < 2:
        return pred
    w = weights or {m: 1.0 / len(members) for m in members}
    sub = pred[pred["model"].isin(members)].copy()
    sub["w"] = sub["model"].map(w)
    sub["wy"] = sub["yhat"] * sub["w"]
    g = sub.groupby(["origin", "h", "target_date"]).agg(
        wy=("wy", "sum"), w=("w", "sum"), y_true=("y_true", "first"), n=("model", "nunique")
    ).reset_index()
    g = g[g["n"] == len(members)]
    ens = pd.DataFrame({
        "model": name, "origin": g["origin"], "h": g["h"], "target_date": g["target_date"],
        "yhat": g["wy"] / g["w"], "y_true": g["y_true"],
    })
    for c in BAND_COLS:
        ens[c] = np.nan
    return pd.concat([pred, ens[pred.columns]], ignore_index=True)


def apply_expanding_calibration(pred: pd.DataFrame, min_obs: int = config.CALIBRATION_MIN_OBS) -> pd.DataFrame:
    """Fill bands for models lacking native intervals using only *already-known* residuals.

    A residual for target date D is usable at origin T only if D <= T. Early
    folds therefore have no bands until enough history accumulates.
    """
    pred = pred.reset_index(drop=True).copy()
    needs = [m for m in pred["model"].unique() if pred.loc[pred["model"] == m, "lo95"].isna().all()]
    for m in needs:
        sub = pred[pred["model"] == m]
        for origin, grp in sub.groupby("origin"):
            past = sub[sub["target_date"] <= origin]
            cal = ResidualCalibrator.from_frame(past, min_obs=min_obs) if len(past) else None
            if cal is None or not cal.ready:
                continue
            for ix, row in grp.iterrows():
                for lvl in (80, 95):
                    lo, hi = cal.offsets(row["h"], lvl)
                    pred.loc[ix, f"lo{lvl}"] = max(row["yhat"] + lo, 0.0)
                    pred.loc[ix, f"hi{lvl}"] = row["yhat"] + hi
    return pred


def calibrators_from_backtest(pred: pd.DataFrame) -> dict[str, ResidualCalibrator]:
    """Final calibrators (all residuals) used for the live forecast."""
    return {m: ResidualCalibrator.from_frame(g) for m, g in pred.groupby("model")}


# ================================================================== metrics
def _row_metrics(g: pd.DataFrame) -> dict:
    e = g["yhat"] - g["y_true"]
    out = {
        "n": int(len(g)),
        "MAE": float(e.abs().mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "MAPE_%": float((e.abs() / g["y_true"].abs().replace(0, np.nan)).mean() * 100),
        "Bias": float(e.mean()),
    }
    for lvl in (80, 95):
        b = g.dropna(subset=[f"lo{lvl}", f"hi{lvl}"])
        out[f"Cover{lvl}_%"] = (
            float(((b["y_true"] >= b[f"lo{lvl}"]) & (b["y_true"] <= b[f"hi{lvl}"])).mean() * 100)
            if len(b) else np.nan
        )
        out[f"Width{lvl}"] = float((b[f"hi{lvl}"] - b[f"lo{lvl}"]).mean()) if len(b) else np.nan
    return out


def compute_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE / MAPE / bias / interval coverage by model and horizon (+ 'All')."""
    rows = []
    for (m, h), g in pred.dropna(subset=["y_true"]).groupby(["model", "h"]):
        rows.append({"model": m, "h": str(h), **_row_metrics(g)})
    for m, g in pred.dropna(subset=["y_true"]).groupby("model"):
        rows.append({"model": m, "h": "All", **_row_metrics(g)})
    out = pd.DataFrame(rows)
    naive = out[out["model"] == "Naive"].set_index("h")["MAE"]
    out["Skill_vs_Naive_%"] = out.apply(
        lambda r: (1 - r["MAE"] / naive[r["h"]]) * 100 if r["h"] in naive.index and naive[r["h"]] > 0 else np.nan,
        axis=1,
    )
    order = {"7": 0, "14": 1, "30": 2, "All": 3}
    out["_o"] = out["h"].map(order).fillna(9)
    return out.sort_values(["_o", "MAE"]).drop(columns="_o").reset_index(drop=True)


def best_model(metrics: pd.DataFrame, exclude: tuple[str, ...] = ("Naive", "MA7", "SeasonalNaive7")) -> str:
    """Lowest 'All'-horizon MAE among non-benchmark models."""
    m = metrics[(metrics["h"] == "All") & (~metrics["model"].isin(exclude))]
    return str(m.sort_values("MAE").iloc[0]["model"]) if len(m) else "Naive"


# ============================================================ live forecasting
def fit_point_forecast(feat: pd.DataFrame, model_name: str,
                       horizons: Iterable[int] = range(1, config.PATH_HORIZON + 1)) -> pd.DataFrame:
    factories = build_factories()
    if model_name not in factories:
        raise KeyError(f"Model '{model_name}' unavailable here. Options: {list(factories)}")
    return factories[model_name]().fit(feat, tuple(horizons)).predict()


def finalize_forecast(raw: pd.DataFrame, model_name: str, feat: pd.DataFrame,
                      calibrators: dict[str, ResidualCalibrator] | None = None) -> pd.DataFrame:
    """Return a forecast with 80/95% bands (native if present, else calibrated)."""
    if raw[BAND_COLS].notna().all().all():
        return sanitize_bands(raw)
    cal = (calibrators or {}).get(model_name)
    if cal is None or not cal.ready:
        cal = ResidualCalibrator.from_history(feat[config.TARGET])
        raw = raw.copy()
        raw.attrs["interval_source"] = "uncalibrated (history-based fallback)"
    else:
        raw = raw.copy()
        raw.attrs["interval_source"] = "calibrated on walk-forward residuals"
    out = sanitize_bands(cal.apply(raw))
    out.attrs.update(raw.attrs)
    return out


def fit_and_forecast(feat: pd.DataFrame, model_names: Iterable[str],
                     horizons: Iterable[int] = range(1, config.PATH_HORIZON + 1),
                     calibrators: dict[str, ResidualCalibrator] | None = None) -> dict[str, pd.DataFrame]:
    names = list(model_names)
    horizons = tuple(horizons)
    members = ensemble_members(build_factories().keys())
    need = {m for n in names for m in (members if n == "Ensemble" else [n])}
    raws = {m: fit_point_forecast(feat, m, horizons) for m in need}
    out = {}
    for n in names:
        if n == "Ensemble":
            raw = raws[members[0]].copy()
            raw["yhat"] = np.mean([raws[m]["yhat"].to_numpy() for m in members], axis=0)
            raw[BAND_COLS] = np.nan
        else:
            raw = raws[n]
        out[n] = finalize_forecast(raw, n, feat, calibrators)
    return out
