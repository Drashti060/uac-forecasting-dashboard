"""Render reports/executive_summary.md from backtest metrics and live risk outputs.

Every number in the report is computed, never typed in, so the briefing can be
regenerated whenever the data refreshes:  ``python scripts/train.py``.
"""
from __future__ import annotations

import math
from datetime import date

import pandas as pd

from . import config
from .risk_engine import RiskSummary


def _n(x, d=0) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:,.{d}f}"


def _lead(x) -> str:
    return "none within 30 days" if x is None else ("already above" if x == 0 else f"{int(x)} days")


def render_executive_summary(
    *,
    feat: pd.DataFrame,
    metrics: pd.DataFrame,
    model: str,
    forecast: pd.DataFrame,
    risks: dict[int, RiskSummary],
    bottleneck: dict,
    threshold: float,
    models_run: list[str],
    n_folds: int,
    interval_source: str,
    data_note: str | None = None,
) -> str:
    last = feat.index[-1]
    census = float(feat["hhs_care"].iloc[-1])
    m = metrics[metrics["model"] == model].set_index("h")
    nv = metrics[metrics["model"] == "Naive"].set_index("h")
    r14 = risks[14]
    cpm, cpw = config.CHILDREN_PER_CASE_MANAGER, config.CHILDREN_PER_YOUTH_CARE_WORKER

    L: list[str] = []
    A = L.append
    A("# Executive Briefing - UAC Program Census Outlook")
    A(f"*Prepared {date.today():%B %d, %Y} · data through {last:%B %d, %Y} · model: **{model}***\n")
    if data_note:
        A(f"> **{data_note}**\n")

    # ------------------------------------------------------------ BLUF
    A("## 1. Bottom line")
    A(f"- **Current census (children in HHS care):** {_n(census)}. "
      f"Planning threshold used here: **{_n(threshold)}** (edit in the dashboard).")
    for h in (7, 14, 30):
        r, f = risks[h], forecast.loc[h]
        A(f"- **+{h} days:** expected **{_n(f['yhat'])}** (80% range {_n(f['lo80'])}-{_n(f['hi80'])}); "
          f"chance of exceeding the threshold on or before that day: **{r.p_breach_any_within_horizon:.0%}**.")
    A(f"- **Overall status: {r14.status}** (14-day view). Surge lead time - expected path: "
      f"**{_lead(r14.lead_time_expected)}**; earliest plausible (10% scenario): **{_lead(r14.lead_time_p10)}**.")
    A(f"- **Sponsor-placement pipeline: {bottleneck['level']}.** {bottleneck['message']}\n")

    # --------------------------------------------------------- accuracy
    A("## 2. How much can we trust the forecast?")
    A(f"Rolling-origin backtest: {n_folds} forecast origins, each using **only data available at the time**.\n")
    A("| Horizon | Typical miss (MAE, children) | % error (MAPE) | Improvement vs. 'no change' | 80% band hit-rate | 95% band hit-rate |")
    A("|---|---:|---:|---:|---:|---:|")
    for h in ("7", "14", "30"):
        if h in m.index:
            A(f"| {h} days | ±{_n(m.loc[h, 'MAE'])} | {_n(m.loc[h, 'MAPE_%'], 1)}% | "
              f"{_n(m.loc[h, 'Skill_vs_Naive_%'], 0)}% | {_n(m.loc[h, 'Cover80_%'], 0)}% | {_n(m.loc[h, 'Cover95_%'], 0)}% |")
    A(f"\n*Reading the table:* at 14 days the forecast is typically off by about **{_n(m.loc['14', 'MAE']) if '14' in m.index else 'n/a'} children** "
      f"(naive 'census stays flat' misses by {_n(nv.loc['14', 'MAE']) if '14' in nv.index else 'n/a'}). "
      f"Band hit-rates should sit near their nominal 80% / 95%; large gaps mean the bands are over- or under-confident. "
      f"Intervals: {interval_source}.")
    if n_folds < 30:
        A(f"\n> Only {n_folds} forecast origins were scored, so hit-rates are noisy (each origin moves them by several points). "
          "Increase `--max-folds` / lower `--step` for firmer estimates.")
    A("")

    # --------------------------------------------------------- staffing
    A("## 3. Turning forecast error into staffing decisions")
    A("Plan to the **upper 80% bound**, not the point forecast. The gap between them is the capacity buffer that "
      "covers forecast error four times out of five.\n")
    A(f"| Horizon | Point forecast | Plan-to level (80% upper) | Buffer beds | Extra case managers (1:{cpm}) | Extra waking-shift care workers (1:{cpw}) |")
    A("|---|---:|---:|---:|---:|---:|")
    for h in (7, 14, 30):
        f = forecast.loc[h]
        buf = max(f["hi80"] - f["yhat"], 0)
        A(f"| {h} days | {_n(f['yhat'])} | {_n(f['hi80'])} | {_n(buf)} | {_n(math.ceil(buf / cpm))} | {_n(math.ceil(buf / cpw))} |")
    A(f"\n> Staffing ratios ({cpm} children per case manager, {cpw} per care worker per waking shift) are "
      f"**illustrative defaults** in `src/config.py`; replace them with current program standards.\n")
    A("**Rule of thumb from the backtest:** every ±100 children of forecast error ≈ "
      f"{math.ceil(100 / cpm)} case managers and {math.ceil(100 / cpw)} care-worker positions per waking shift.\n")

    # --------------------------------------------------------- triggers
    A("## 4. Decision triggers")
    A("| Signal | Threshold | Action |")
    A("|---|---|---|")
    A(f"| P(exceed threshold within horizon) | < {config.RISK_AMBER:.0%} | **GREEN** - maintain staffing |")
    A(f"| | {config.RISK_AMBER:.0%}-{config.RISK_RED:.0%} | **AMBER** - pre-authorise surge staffing / identify overflow beds |")
    A(f"| | > {config.RISK_RED:.0%} or already above | **RED** - activate surge plan; open contingency capacity |")
    A(f"| 7-day discharges ÷ transfers | < {config.RATIO_WARN:.0%} | Review sponsor-vetting backlog |")
    A(f"| | < {config.RATIO_CRIT:.0%} with net inflow > {config.NET_PRESSURE_WARN:.0f}/day | Escalate: add sponsor-assessment / case-management capacity |\n")

    # ------------------------------------------------------- bottleneck
    A("## 5. Sponsor-placement bottleneck")
    bn = bottleneck
    A(f"- Last {bn['window_days']} days: discharges cover **{_n(bn['discharge_transfer_ratio'] * 100)}%** of transfers in; "
      f"net inflow **{_n(bn['net_pressure_per_day'], 1)}/day**.")
    chg = bn["los_change_pct"]
    trend = ("lengthening stays mean beds are turning over more slowly" if chg == chg and chg > 5
             else "stable or shortening stays" if chg == chg else "trend unavailable")
    A(f"- Implied length of care: **{_n(bn['los_days_now'], 1)} days** "
      f"({_n(chg, 0)}% vs. two weeks earlier) - {trend}.")
    if bn["days_to_threshold_at_current_rate"] is not None:
        A(f"- At the current net-inflow rate the threshold would be reached in about "
          f"**{_n(bn['days_to_threshold_at_current_rate'], 0)} days** (flow-based cross-check on the model).")
    A("")

    # ------------------------------------------------------- caveats
    A("## 6. Caveats")
    A("- Forecasts are statistical projections of reported census; they cannot anticipate policy changes, "
      "enforcement shifts or reporting changes. Re-run after any such event.")
    A("- Non-reporting days (weekends/holidays) are interpolated; the backtest scores against these filled values.")
    A(f"- Models evaluated: {', '.join(models_run)}. Selection of **{model}** is by lowest average MAE across the 7/14/30-day horizons, "
      "so its reported error is slightly optimistic; confirm on fresh data.")
    A("- Breach probabilities are Monte-Carlo estimates built from the forecast bands; they inherit any band mis-calibration shown in Section 2.")
    return "\n".join(L) + "\n"
