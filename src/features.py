"""Feature engineering.

Design rule (no leakage)
------------------------
Every feature in row ``t`` is a function of data observed on or before day
``t`` (the *forecast origin*). Lags use ``shift(k)`` with k >= 0, rolling
windows are trailing, calendar features are deterministic. The only features
that look ahead are the ``tgt_*`` calendar columns, which describe the
*target date* ``t+h`` - day-of-week and holidays are known in advance, so
this is legitimate and is what lets a direct model learn weekday effects.

``tests/test_pipeline.py`` verifies the rule by rebuilding features from a
truncated series and checking the overlap is identical.

Lag 0 is the raw column itself (``hhs_care``, ``transferred`` ...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import config

EXCLUDE_FROM_X = {"is_imputed"}


# ------------------------------------------------------------------ calendar
def calendar_features(index: pd.DatetimeIndex, prefix: str = "cal_") -> pd.DataFrame:
    """Deterministic calendar features for any DatetimeIndex (past or future)."""
    idx = pd.DatetimeIndex(index)
    dates = idx.normalize()
    hol = USFederalHolidayCalendar().holidays(
        start=dates.min() - pd.Timedelta(days=7), end=dates.max() + pd.Timedelta(days=7)
    )
    is_hol = np.asarray(dates.isin(hol))
    near = np.zeros(len(idx), dtype=bool)
    for k in (-2, -1, 1, 2):
        near |= np.asarray((dates + pd.Timedelta(days=k)).isin(hol))

    dow = np.asarray(idx.dayofweek)
    month = np.asarray(idx.month)
    out = pd.DataFrame(
        {
            f"{prefix}dow": dow,
            f"{prefix}is_weekend": (dow >= 5).astype(int),
            f"{prefix}month": month,
            f"{prefix}quarter": np.asarray(idx.quarter),
            f"{prefix}week": idx.isocalendar().week.to_numpy().astype(int),
            f"{prefix}dow_sin": np.sin(2 * np.pi * dow / 7),
            f"{prefix}dow_cos": np.cos(2 * np.pi * dow / 7),
            f"{prefix}month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            f"{prefix}month_cos": np.cos(2 * np.pi * (month - 1) / 12),
            f"{prefix}is_holiday": is_hol.astype(int),
            f"{prefix}near_holiday": (near & ~is_hol).astype(int),
        },
        index=idx,
    )
    return out


# ------------------------------------------------------------------ features
def build_features(df: pd.DataFrame, target: str = config.TARGET) -> pd.DataFrame:
    """Clean daily frame -> engineered feature frame (same index)."""
    f = df.copy()
    if "is_imputed" not in f:
        f["is_imputed"] = False

    # --- system net-pressure signal:  Delta_Net = Transfers In - Discharges Out
    f["net_pressure"] = f["transferred"] - f["discharged"]

    series_cols = [target, "apprehended", "cbp_custody", "transferred", "discharged", "net_pressure"]
    new: dict[str, pd.Series] = {}

    for c in series_cols:
        s = f[c]
        for k in config.LAGS:
            new[f"{c}_lag{k}"] = s.shift(k)
        for w in config.WINDOWS:
            r = s.rolling(w, min_periods=w)
            new[f"{c}_roll{w}_mean"] = r.mean()
            new[f"{c}_roll{w}_std"] = r.std()

    # --- cumulative net pressure: how many beds the pipeline has "added" lately
    for w in config.WINDOWS:
        new[f"net_pressure_sum{w}"] = f["net_pressure"].rolling(w, min_periods=w).sum()

    # --- sponsor-placement bottleneck signals
    for w in config.WINDOWS:
        dis = f["discharged"].rolling(w, min_periods=w).sum()
        tra = f["transferred"].rolling(w, min_periods=w).sum()
        new[f"discharge_transfer_ratio_{w}"] = (dis / tra.replace(0, np.nan)).clip(0, 5)
    census7 = f[target].rolling(7, min_periods=7).mean()
    dis7 = f["discharged"].rolling(7, min_periods=7).mean()
    new["implied_los_days"] = (census7 / dis7.replace(0, np.nan)).clip(0, 365)  # Little's law proxy

    # --- census momentum + "other flows" residual (census change not explained by net pressure)
    new[f"{target}_diff1"] = f[target].diff(1)
    new[f"{target}_diff7"] = f[target].diff(7)
    new[f"{target}_diff14"] = f[target].diff(14)
    new[f"{target}_vs_ma7"] = f[target] - census7
    new["flow_residual_7"] = f[target].diff(7) - new["net_pressure_sum7"]

    out = pd.concat([f, pd.DataFrame(new, index=f.index), calendar_features(f.index, "cal_")], axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns(feat: pd.DataFrame) -> list[str]:
    return [c for c in feat.columns if c not in EXCLUDE_FROM_X]


# ------------------------------------------------------------- supervision
def make_supervised(
    feat: pd.DataFrame,
    horizon: int,
    target: str = config.TARGET,
    delta: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Direct-strategy design matrix for a single horizon ``h``.

    Row ``t`` -> features known at ``t`` (+ calendar of ``t+h``); label is
    ``y[t+h]`` (or ``y[t+h] - y[t]`` when ``delta=True``).

    The last ``h`` rows have NaN labels (their future is unobserved). Callers
    must slice ``feat`` to the forecast origin *before* calling this, which
    guarantees no label peeks past the origin.

    ``delta=True`` is recommended for tree models: they cannot extrapolate
    levels, but the change in census is far more stationary.
    """
    X = feat[feature_columns(feat)].copy()
    tgt_cal = calendar_features(feat.index + pd.Timedelta(days=horizon), prefix="tgt_")
    tgt_cal.index = feat.index
    X = pd.concat([X, tgt_cal], axis=1)
    y_future = feat[target].shift(-horizon)
    y = (y_future - feat[target]) if delta else y_future
    y.name = f"y_h{horizon}"
    return X, y
