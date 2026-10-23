"""Run with `pytest -q` (or `python -m tests.test_pipeline` if pytest is absent)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_synthetic_data import simulate, to_raw_csv  # noqa: E402
from src import config, data_pipeline as dp, features as fe, models as md, risk_engine as rk  # noqa: E402

_CACHE: dict = {}


def clean_df() -> pd.DataFrame:
    if "df" not in _CACHE:
        p = Path(tempfile.mkdtemp()) / "raw.csv"
        to_raw_csv(simulate(end="2025-06-30"), p)
        _CACHE["df"] = dp.load_clean_data(p)
    return _CACHE["df"]


# ------------------------------------------------------------------ cleaning
def test_cleaning_produces_complete_numeric_daily_series():
    d = clean_df()
    assert d.index.freq is not None and d.index.is_monotonic_increasing
    assert (d.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert d[list(config.RAW_COLUMNS)].notna().all().all()
    assert all(pd.api.types.is_float_dtype(d[c]) for c in config.RAW_COLUMNS)
    assert d["is_imputed"].any() and d.attrs["n_imputed_days"] > 0


def test_comma_numbers_and_junk_rows():
    raw = pd.DataFrame({
        "Date": ["January 3, 2025", "January 1, 2025", "* footnote", None],
        "Children apprehended and placed in CBP custody*": ["10", "8", "", None],
        "Children in CBP custody": ["100", "90", "", None],
        "Children transferred out of CBP custody": ["12", "9", "", None],
        "Children in HHS Care": ["2,484", "2,400", "", None],
        "Children discharged from HHS Care": ["20", "18", "", None],
    })
    d = dp.clean_dataframe(raw)
    assert len(d) == 3                                   # Jan 1..3 filled daily
    assert d.loc["2025-01-03", "hhs_care"] == 2484
    assert d.loc["2025-01-02", "hhs_care"] == 2442       # linear interpolation
    assert bool(d.loc["2025-01-02", "is_imputed"])


def test_causal_fill_never_uses_future_values():
    raw = pd.DataFrame({
        "Date": ["January 3, 2025", "January 1, 2025"],
        "Children apprehended and placed in CBP custody*": ["10", "8"],
        "Children in CBP custody": ["100", "90"],
        "Children transferred out of CBP custody": ["12", "9"],
        "Children in HHS Care": ["2,500", "2,400"],
        "Children discharged from HHS Care": ["20", "18"],
    })
    d = dp.clean_dataframe(raw, causal=True)
    assert d.loc["2025-01-02", "hhs_care"] == 2400        # forward-filled, not interpolated


def test_decomposition_reconstructs_series():
    d = clean_df()
    dec = dp.decompose_series(d[config.TARGET])
    recon = dec["trend"] + dec["seasonal"] + dec["resid"]
    assert np.allclose(recon, dec["observed"])


# ------------------------------------------------------------------- features
def test_features_have_no_lookahead():
    d = clean_df()
    full = fe.build_features(d)
    cut = 600
    part = fe.build_features(d.iloc[:cut])
    a = full.iloc[:cut].drop(columns=["is_imputed"])
    b = part.drop(columns=["is_imputed"])
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)


def test_expected_feature_columns_exist():
    f = fe.build_features(clean_df())
    for k in (1, 3, 7, 14, 30):
        assert f"hhs_care_lag{k}" in f
    for w in (7, 14, 30):
        assert {f"hhs_care_roll{w}_mean", f"hhs_care_roll{w}_std", f"net_pressure_sum{w}"} <= set(f)
    assert np.allclose(f["net_pressure"], f["transferred"] - f["discharged"])
    assert {"cal_dow", "cal_is_weekend", "cal_month", "cal_is_holiday"} <= set(f)


def test_supervised_labels_align_with_future():
    f = fe.build_features(clean_df())
    X, y = fe.make_supervised(f, 7, delta=False)
    assert y.iloc[100] == f["hhs_care"].iloc[107]
    assert y.iloc[-7:].isna().all()
    Xd, yd = fe.make_supervised(f, 7, delta=True)
    assert yd.iloc[100] == f["hhs_care"].iloc[107] - f["hhs_care"].iloc[100]


# ---------------------------------------------------------------- walk-forward
def test_direct_model_ignores_data_after_origin():
    """Predictions at origin T must be identical whether or not the future exists."""
    f = fe.build_features(clean_df())
    T = 500
    m1 = md.DirectRegressorForecaster("t", "hist").fit(f.iloc[: T + 1], (7, 14))
    p1 = m1.predict()["yhat"].to_numpy()
    f2 = f.copy()
    num = f2.select_dtypes("number").columns
    f2.loc[f2.index[T + 1:], num] = 1e6          # wildly corrupt everything after the origin
    m2 = md.DirectRegressorForecaster("t", "hist").fit(f2.iloc[: T + 1], (7, 14))
    assert np.allclose(p1, m2.predict()["yhat"].to_numpy())


def test_walk_forward_end_to_end_and_metrics():
    f = fe.build_features(clean_df())
    fac = {k: v for k, v in md.build_factories(include_slow=False).items() if k in ("Naive", "MA7", "SeasonalNaive7", md.GBM_NAME)}
    pred = md.walk_forward_backtest(f, fac, initial_train=365, step=30, max_folds=10)
    assert pred["origin"].nunique() == 10
    assert (pred["target_date"] > pred["origin"]).all()
    # every y_true must equal the actual observed series value on target_date
    assert np.allclose(pred["y_true"], f["hhs_care"].reindex(pred["target_date"]).to_numpy())
    met = md.compute_metrics(pred)
    assert {"MAE", "RMSE", "MAPE_%"} <= set(met.columns)
    assert set(met["h"]) >= {"7", "14", "30", "All"}
    bands = pred.dropna(subset=["lo95"])
    assert (bands["lo95"] <= bands["lo80"] + 1e-9).all() and (bands["hi80"] <= bands["hi95"] + 1e-9).all()


def test_expanding_calibration_uses_only_known_residuals():
    f = fe.build_features(clean_df())
    fac = {"Naive": md.NaivePersistence}
    pred = md.walk_forward_backtest(f, fac, initial_train=365, step=7, max_folds=20)
    first_banded = pred.dropna(subset=["lo95"])["origin"].min()
    past = pred[(pred["model"] == "Naive") & (pred["target_date"] <= first_banded)]
    assert past.groupby("h").size().max() >= config.CALIBRATION_MIN_OBS


# ------------------------------------------------------------------ risk engine
def _fc(level=2500.0, sd=40.0, H=30, drift=0.0):
    h = np.arange(1, H + 1)
    s = sd * np.sqrt(h / H)
    mid = level + drift * h
    return pd.DataFrame({"yhat": mid, "lo80": mid - 1.2816 * s, "hi80": mid + 1.2816 * s,
                         "lo95": mid - 1.96 * s, "hi95": mid + 1.96 * s}, index=pd.Index(h, name="h"))


def test_breach_probability_matches_normal_cdf():
    from scipy.stats import norm
    fc = _fc()
    r = rk.compute_risk(fc, threshold=2540.0, current_census=2500.0, horizon=30, n_sims=40_000)
    expected = 1 - norm.cdf((2540 - 2500) / 40.0)
    assert abs(r.p_breach_at_horizon - expected) < 0.01
    assert r.p_breach_any_within_horizon >= r.p_breach_at_horizon


def test_lead_time_and_status():
    fc = _fc(level=2400, drift=5.0)
    r = rk.compute_risk(fc, threshold=2500.0, current_census=2400.0, horizon=30)
    assert r.lead_time_expected == 21                      # 2400 + 5h > 2500 first at h=21
    assert r.status in {"AMBER", "RED"}
    r2 = rk.compute_risk(_fc(level=2000), threshold=3000.0, current_census=2000.0)
    assert r2.status == "GREEN" and r2.lead_time_expected is None
    r3 = rk.compute_risk(_fc(level=3100), threshold=3000.0, current_census=3100.0)
    assert r3.already_breached and r3.lead_time_expected == 0 and r3.status == "RED"


def test_probability_monotone_in_threshold():
    fc = _fc()
    ps = [rk.compute_risk(fc, t, 2500.0).p_breach_any_within_horizon for t in (2450, 2500, 2550, 2600)]
    assert all(a >= b for a, b in zip(ps, ps[1:]))


def test_bottleneck_levels():
    d = pd.DataFrame({"transferred": [100.0] * 30, "discharged": [60.0] * 30, "hhs_care": np.linspace(2000, 2300, 30)})
    a = rk.sponsor_bottleneck_alert(d, threshold=2600)
    assert a["level"] == "CRITICAL" and a["net_pressure_per_day"] == 40
    d["discharged"] = 100.0
    assert rk.sponsor_bottleneck_alert(d)["level"] == "OK"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(fns)} tests passed")
