"""UAC Program capacity-forecast dashboard.

    streamlit run app.py

Tabs
----
1. Forecast & Capacity Risk   - fan chart (80/95% bands), breach probability, surge lead time
2. Pipeline & Bottleneck      - transfers vs discharges, net pressure, sponsor-placement alerts
3. Model Benchmarks           - walk-forward metrics from ``scripts/train.py``
4. Data & Decomposition       - cleaning report and STL components
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import config, data_pipeline as dp, features as fe, models as md, risk_engine as rk

st.set_page_config(page_title="UAC Capacity Forecast", layout="wide")

STATUS_FN = {"GREEN": st.success, "AMBER": st.warning, "RED": st.error}
BN_FN = {"OK": st.success, "WARNING": st.warning, "CRITICAL": st.error}
C_HIST, C_FC, C_THR = "#1f3b73", "#e07b00", "#c0392b"


# ------------------------------------------------------------------ caching
@st.cache_data(show_spinner="Cleaning data and engineering features...")
def load_data(upload: bytes | None, path: str, causal: bool):
    src = io.BytesIO(upload) if upload else path
    clean = dp.load_clean_data(src, causal=causal)
    return clean, fe.build_features(clean)


@st.cache_data(show_spinner="Fitting model (first run only)...")
def point_forecast(feat: pd.DataFrame, model_name: str) -> pd.DataFrame:
    return md.fit_point_forecast(feat, model_name)


@st.cache_data
def load_backtest(mtime: float) -> pd.DataFrame:
    return pd.read_csv(config.ARTIFACT_DIR / "backtest_predictions.csv", parse_dates=["origin", "target_date"])


def backtest_path() -> Path:
    return config.ARTIFACT_DIR / "backtest_predictions.csv"


def get_forecast(feat: pd.DataFrame, model_name: str, calibrators) -> pd.DataFrame:
    if model_name == "Ensemble":
        members = md.ensemble_members(md.build_factories().keys())
        raw = point_forecast(feat, members[0]).copy()
        raw["yhat"] = np.mean([point_forecast(feat, m)["yhat"].to_numpy() for m in members], axis=0)
        raw[md.BAND_COLS] = np.nan
        raw.attrs["members"] = members
    else:
        raw = point_forecast(feat, model_name)
    return md.finalize_forecast(raw, model_name, feat, calibrators)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Controls")
st.sidebar.subheader("Data")
upload = st.sidebar.file_uploader("HHS UAC CSV", type="csv")
use_demo = st.sidebar.checkbox("Use synthetic demo data", value=False,
                               help="Simulated numbers for trying the app - not real.")
causal = st.sidebar.checkbox("Causal gap-fill (forward-fill only)", value=False,
                             help="Off = linear interpolation inside reporting gaps.")

data_path = config.DEFAULT_DATA_PATH
if use_demo:
    data_path = config.DATA_DIR / "synthetic_uac.csv"
    if not data_path.exists():
        from scripts.make_synthetic_data import main as _make

        _saved = sys.argv
        sys.argv = [sys.argv[0], "--out", str(data_path)]
        _make()
        sys.argv = _saved

if upload is None and not Path(data_path).exists():
    st.title("UAC Program - Capacity Forecast")
    st.info(f"Upload the HHS CSV in the sidebar, or place it at `{config.DEFAULT_DATA_PATH}`. "
            "You can also tick **Use synthetic demo data** to explore the app.")
    st.stop()

clean, feat = load_data(upload.getvalue() if upload else None, str(data_path), causal)
census_s = feat[config.TARGET]
census = float(census_s.iloc[-1])
last_date = feat.index[-1]

st.sidebar.subheader("Forecast")
horizon = st.sidebar.radio("Horizon (days)", list(config.HORIZONS), index=1, horizontal=True)

factories = md.build_factories()
options = [m for m in ("SARIMAX", md.GBM_NAME) if m in factories]
if len(md.ensemble_members(factories.keys())) >= 2:
    options.append("Ensemble")
options += [m for m in factories if m not in options]
if "SARIMAX" not in factories:
    st.sidebar.caption("SARIMAX unavailable (install statsmodels).")
model_name = st.sidebar.selectbox("Model", options, index=0)
levels = st.sidebar.multiselect("Confidence bands", [80, 95], default=[80, 95])
show_naive = st.sidebar.checkbox("Overlay naive baseline", value=False)
history_days = st.sidebar.select_slider("History shown (days)", [60, 120, 180, 365, 730], value=180)

st.sidebar.subheader("Capacity")
dflt = rk.default_threshold(census_s)
lo_t = float(max(0, round(census_s.min() * 0.9 / 50) * 50))
hi_t = float(round(max(census_s.max(), dflt) * 1.6 / 50) * 50)
threshold = st.sidebar.slider("Nominal capacity threshold (children)", lo_t, hi_t, float(dflt), step=25.0,
                              help="Set to the operating capacity you want to monitor.")

st.sidebar.subheader("Sponsor-placement alerts")
bn_window = st.sidebar.select_slider("Look-back window (days)", [7, 14, 30], value=7)
ratio_warn = st.sidebar.slider("Warn if discharges/transfers <", 0.50, 1.20, float(config.RATIO_WARN), 0.01)
ratio_crit = st.sidebar.slider("Critical if discharges/transfers <", 0.40, 1.00, float(config.RATIO_CRIT), 0.01)
net_warn = st.sidebar.slider("Material net inflow (children/day)", 0.0, 200.0, float(config.NET_PRESSURE_WARN), 5.0)
ratio_crit = min(ratio_crit, ratio_warn)

# --------------------------------------------------------------- forecasting
bt_ok = backtest_path().exists()
bt = load_backtest(backtest_path().stat().st_mtime) if bt_ok else None
calibrators = md.calibrators_from_backtest(bt) if bt_ok else None

try:
    fc = get_forecast(feat, model_name, calibrators)
except Exception as exc:  # pragma: no cover - UI guard
    st.error(f"Could not fit **{model_name}**: {exc}")
    st.stop()

risk = rk.compute_risk(fc, threshold, census, horizon=horizon)
bn = rk.sponsor_bottleneck_alert(feat, window=bn_window, ratio_warn=ratio_warn,
                                 ratio_crit=ratio_crit, net_warn=net_warn, threshold=threshold)
interval_src = fc.attrs.get("interval_source", "native model intervals")

# ------------------------------------------------------------------- header
st.title("UAC Program - Capacity Forecast & Risk")
st.caption(f"Data through **{last_date:%b %d, %Y}** · model **{model_name}** · intervals: {interval_src}")
if not bt_ok:
    st.warning("No walk-forward backtest found - bands use a history-based fallback and are **uncalibrated**. "
               "Run `python scripts/train.py` (or use the Benchmarks tab) to calibrate.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Forecast & Capacity Risk", "Pipeline & Bottleneck", "Model Benchmarks", "Data & Decomposition"])


# ---------------------------------------------------------------- tab 1
def _band(fig, x, lo, hi, name, color):
    fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]), y=list(hi) + list(lo[::-1]), fill="toself",
                             fillcolor=color, line=dict(width=0), hoverinfo="skip", name=name))


with tab1:
    row = fc.loc[horizon]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Current census", f"{census:,.0f}")
    k2.metric(f"Forecast +{horizon}d", f"{row['yhat']:,.0f}", f"{row['yhat'] - census:+,.0f}")
    k3.metric(f"Range (80%) +{horizon}d", f"{row['lo80']:,.0f} - {row['hi80']:,.0f}")
    k4.metric("P(breach within horizon)", f"{risk.p_breach_any_within_horizon:.0%}")
    lead = ("Already above" if risk.already_breached else
            (f"{risk.lead_time_expected:.0f} days" if risk.lead_time_expected is not None else "> 30 days"))
    k5.metric("Surge lead time", lead)

    STATUS_FN[risk.status](
        f"**{risk.status}** - {risk.p_breach_any_within_horizon:.0%} chance census exceeds "
        f"{threshold:,.0f} on at least one day in the next {horizon} days "
        f"(peak expected {risk.expected_peak:,.0f}; 95% upper {risk.peak_hi95:,.0f})." + " " + " ".join(risk.notes))

    view = fc.loc[:horizon]
    hist = census_s.iloc[-history_days:]
    x_fc = [last_date] + list(view["date"])
    fig = go.Figure()
    for lvl, col in ((95, "rgba(224,123,0,0.15)"), (80, "rgba(224,123,0,0.30)")):
        if lvl in levels:
            lo = [census] + list(view[f"lo{lvl}"])
            hi = [census] + list(view[f"hi{lvl}"])
            _band(fig, x_fc, lo, hi, f"{lvl}% band", col)
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Observed", line=dict(color=C_HIST, width=2)))
    fig.add_trace(go.Scatter(x=x_fc, y=[census] + list(view["yhat"]), name=f"{model_name} forecast",
                             line=dict(color=C_FC, width=2, dash="dash")))
    if show_naive:
        fig.add_trace(go.Scatter(x=x_fc, y=[census] * len(x_fc), name="Naive (flat)",
                                 line=dict(color="#7f8c8d", width=1, dash="dot")))
    fig.add_hline(y=threshold, line_dash="dash", line_color=C_THR,
                  annotation_text=f"Capacity threshold {threshold:,.0f}", annotation_position="top left")
    ymin = min(hist.min(), view["lo95"].min(), threshold) * 0.97
    ymax = max(hist.max(), view["hi95"].max(), threshold) * 1.03
    fig.add_trace(go.Scatter(x=[last_date, last_date], y=[ymin, ymax], mode="lines", name="Forecast origin",
                             line=dict(color="#95a5a6", width=1), showlegend=False, hoverinfo="skip"))
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis=dict(title="Children in HHS care", range=[ymin, ymax]),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        p = risk.p_breach_by_day.loc[:horizon]
        fig2 = go.Figure(go.Bar(x=view["date"], y=p.values, marker_color=[
            C_THR if v >= config.RISK_RED else ("#e0a800" if v >= config.RISK_AMBER else "#2e8b57") for v in p.values]))
        fig2.add_hline(y=config.RISK_AMBER, line_dash="dot", line_color="#e0a800")
        fig2.add_hline(y=config.RISK_RED, line_dash="dot", line_color=C_THR)
        fig2.update_layout(title="Probability census exceeds threshold, by day", height=300,
                           margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(range=[0, 1], tickformat=".0%"))
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.markdown("**Surge lead time**")
        lt = pd.DataFrame({
            "Scenario": ["Expected path (median)", "10% chance breached by", "50% chance breached by"],
            "Days": [rk_ if rk_ is not None else None for rk_ in
                     (risk.lead_time_expected, risk.lead_time_p10, risk.lead_time_p50)],
        })
        lt["Days"] = lt["Days"].map(lambda v: "none in 30d" if v is None else f"{v:.0f}")
        st.table(lt.set_index("Scenario"))
        beds80 = rk.beds_at_confidence(fc, horizon, 80)
        st.markdown(f"**Plan-to level (80% upper) at +{horizon}d:** {beds80:,.0f} beds "
                    f"({beds80 - row['yhat']:+,.0f} above the point forecast)")
    st.download_button("Download forecast (CSV)", fc.reset_index().to_csv(index=False),
                       file_name=f"forecast_{model_name}_{last_date:%Y%m%d}.csv", mime="text/csv")

# ---------------------------------------------------------------- tab 2
with tab2:
    BN_FN[bn["level"]](f"**Sponsor-placement pipeline: {bn['level']}** - {bn['message']}")
    a, b, c, d = st.columns(4)
    a.metric(f"Discharges / transfers ({bn_window}d)", f"{bn['discharge_transfer_ratio']:.0%}")
    b.metric("Net inflow (children/day)", f"{bn['net_pressure_per_day']:+.1f}")
    c.metric("Implied length of care (days)", f"{bn['los_days_now']:.1f}",
             f"{bn['los_change_pct']:+.0f}%" if np.isfinite(bn['los_change_pct']) else None, delta_color="inverse")
    d.metric("Days to threshold at current net rate",
             "n/a" if bn["days_to_threshold_at_current_rate"] is None else f"{bn['days_to_threshold_at_current_rate']:.0f}")

    tail = feat.iloc[-history_days:]
    f3 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                       subplot_titles=("Transfers in vs. discharges out (7-day mean)",
                                       "Net pressure = transfers - discharges (7-day mean)",
                                       "Discharges / transfers ratio (7-day)"))
    f3.add_trace(go.Scatter(x=tail.index, y=tail["transferred"].rolling(7).mean(), name="Transfers in",
                            line=dict(color=C_HIST)), row=1, col=1)
    f3.add_trace(go.Scatter(x=tail.index, y=tail["discharged"].rolling(7).mean(), name="Discharges out",
                            line=dict(color="#2e8b57")), row=1, col=1)
    npm = tail["net_pressure"].rolling(7).mean()
    f3.add_trace(go.Bar(x=tail.index, y=npm, name="Net pressure",
                        marker_color=[C_THR if v > net_warn else "#95a5a6" for v in npm.fillna(0)]), row=2, col=1)
    f3.add_hline(y=net_warn, line_dash="dot", line_color=C_THR, row=2, col=1)
    f3.add_trace(go.Scatter(x=tail.index, y=tail["discharge_transfer_ratio_7"], name="Ratio",
                            line=dict(color="#6c3483")), row=3, col=1)
    f3.add_hline(y=ratio_warn, line_dash="dot", line_color="#e0a800", row=3, col=1)
    f3.add_hline(y=ratio_crit, line_dash="dot", line_color=C_THR, row=3, col=1)
    f3.add_hline(y=1.0, line_color="#bdc3c7", row=3, col=1)
    f3.update_layout(height=700, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    st.plotly_chart(f3, use_container_width=True)
    st.caption("A ratio below 100% means children are entering HHS care faster than sponsors are vetted and "
               "placed, so the census grows even when border inflow is flat.")

# ---------------------------------------------------------------- tab 3
with tab3:
    mpath = config.ARTIFACT_DIR / "metrics.csv"
    if not mpath.exists():
        st.info("No backtest metrics yet. Run `python scripts/train.py`, or run a lighter backtest here.")
        if st.button("Run light backtest now (benchmarks + GBM, a few minutes)"):
            fac = {k: v for k, v in md.build_factories(include_slow=False).items()
                   if k in ("Naive", "MA7", "SeasonalNaive7", md.GBM_NAME)}
            with st.spinner("Walk-forward backtest running..."):
                pred = md.walk_forward_backtest(feat, fac, step=21, max_folds=12)
                config.ARTIFACT_DIR.mkdir(exist_ok=True)
                pred.to_csv(backtest_path(), index=False)
                md.compute_metrics(pred).to_csv(mpath, index=False)
            st.cache_data.clear()
            st.rerun()
    else:
        met = pd.read_csv(mpath, dtype={"h": str})
        st.markdown("**Walk-forward (expanding window) out-of-sample metrics** - lower is better; "
                    "coverage should be near 80 / 95.")
        st.dataframe(met.round(2), use_container_width=True, hide_index=True)
        piv = met[met["h"].isin(["7", "14", "30"])]
        fig4 = go.Figure()
        for h, col in zip(["7", "14", "30"], ["#1f3b73", "#e07b00", "#c0392b"]):
            s = piv[piv["h"] == h].sort_values("MAE")
            fig4.add_trace(go.Bar(x=s["model"], y=s["MAE"], name=f"h={h}", marker_color=col))
        fig4.update_layout(barmode="group", height=360, title="MAE by model and horizon (children)",
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig4, use_container_width=True)
        if bt is not None:
            hh = st.selectbox("Forecast vs actual - horizon", [7, 14, 30], index=1, key="bt_h")
            mm = st.selectbox("Model", sorted(bt["model"].unique()),
                              index=sorted(bt["model"].unique()).index(model_name)
                              if model_name in set(bt["model"]) else 0, key="bt_m")
            s = bt[(bt["model"] == mm) & (bt["h"] == hh)].sort_values("target_date")
            fig5 = go.Figure()
            if s["lo95"].notna().any():
                sb = s.dropna(subset=["lo95"])
                _band(fig5, list(sb["target_date"]), list(sb["lo95"]), list(sb["hi95"]), "95%", "rgba(224,123,0,0.15)")
            fig5.add_trace(go.Scatter(x=s["target_date"], y=s["y_true"], name="Actual", line=dict(color=C_HIST)))
            fig5.add_trace(go.Scatter(x=s["target_date"], y=s["yhat"], name="Forecast", line=dict(color=C_FC, dash="dash")))
            fig5.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig5, use_container_width=True)

# ---------------------------------------------------------------- tab 4
with tab4:
    at = clean.attrs
    a, b, c, d = st.columns(4)
    a.metric("Raw rows", f"{at.get('n_raw_rows', 0):,}")
    b.metric("Reported days", f"{at.get('n_reported_days', 0):,}")
    c.metric("Calendar days", f"{at.get('n_calendar_days', 0):,}")
    d.metric("Imputed days", f"{at.get('n_imputed_days', 0):,}")
    dec = dp.decompose_series(census_s.iloc[-max(history_days, 120):])
    st.caption(f"Decomposition method: {dec.attrs.get('method')} · weekly seasonal strength "
               f"{dp.seasonal_strength(dec):.2f} (0 = none, 1 = pure seasonality)")
    f5 = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                       subplot_titles=("Observed", "Trend", "Weekly seasonal", "Residual"))
    for i, col in enumerate(["observed", "trend", "seasonal", "resid"], 1):
        f5.add_trace(go.Scatter(x=dec.index, y=dec[col], line=dict(color=C_HIST)), row=i, col=1)
    f5.update_layout(height=640, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(f5, use_container_width=True)
    st.download_button("Download cleaned daily data (CSV)", clean.reset_index().to_csv(index=False),
                       file_name="uac_clean_daily.csv", mime="text/csv")
