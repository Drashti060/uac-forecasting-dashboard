"""Capacity-risk analytics built on top of a forecast with 80/95% bands.

Monte-Carlo approach
--------------------
A forecast gives, for each day h, five quantiles of the census:
2.5%, 10%, 50% (yhat), 90%, 97.5%. We turn them into a full marginal
distribution per day by interpolating the quantile function on the probit
scale (with linear extrapolation in the tails). To get *paths* (needed for
"does the census exceed capacity at any point?") we correlate the days with a
Gaussian copula whose correlation structure is that of a random walk,
``corr(i, j) = sqrt(min(i,j)/max(i,j))`` - errors of a stock variable such as
census accumulate, so neighbouring days are highly dependent.

Outputs
-------
* Capacity-breach probability  : per day, at the horizon, and "any day <= h".
* Surge lead time              : days until the census first exceeds the
  threshold - from the median path and from the simulated first-passage
  distribution.
* Sponsor-placement bottleneck : discharge/transfer ratio, net pressure and
  implied length-of-care trend (flow-based, model-free).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config

KNOT_COLS = ["lo95", "lo80", "yhat", "hi80", "hi95"]
KNOT_Z = norm.ppf([0.025, 0.10, 0.50, 0.90, 0.975])


# --------------------------------------------------------------- simulation
def _corr_matrix(H: int, kind: str = "brownian", rho: float = 0.95) -> np.ndarray:
    i = np.arange(1, H + 1)
    if kind == "brownian":
        return np.sqrt(np.minimum.outer(i, i) / np.maximum.outer(i, i))
    if kind == "ar1":
        return rho ** np.abs(np.subtract.outer(i, i))
    raise ValueError(kind)


def _interp_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    y = np.interp(x, xp, fp)
    lo_slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
    hi_slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    y = np.where(x < xp[0], fp[0] + lo_slope * (x - xp[0]), y)
    y = np.where(x > xp[-1], fp[-1] + hi_slope * (x - xp[-1]), y)
    return y


def simulate_paths(
    forecast: pd.DataFrame,
    n_sims: int = config.N_SIMS,
    seed: int = config.SEED,
    dependence: str = "brownian",
    rho: float = 0.95,
) -> np.ndarray:
    """Return an (n_sims, H) array of simulated census paths for days 1..H."""
    if forecast[KNOT_COLS].isna().any().any():
        raise ValueError("forecast needs yhat plus 80% and 95% bands")
    H = len(forecast)
    knots = np.maximum.accumulate(forecast[KNOT_COLS].to_numpy(float), axis=1)
    L = np.linalg.cholesky(_corr_matrix(H, dependence, rho) + 1e-10 * np.eye(H))
    Z = np.random.default_rng(seed).standard_normal((n_sims, H)) @ L.T
    paths = np.empty_like(Z)
    for j in range(H):
        paths[:, j] = _interp_extrap(Z[:, j], KNOT_Z, knots[j])
    return np.clip(paths, 0, None)


# --------------------------------------------------------------------- risk
@dataclass
class RiskSummary:
    threshold: float
    current_census: float
    horizon: int
    p_breach_by_day: pd.Series                # P(census_h > C), index = h
    p_breach_at_horizon: float
    p_breach_any_within_horizon: float
    already_breached: bool
    lead_time_expected: float | None          # first day the *median* path exceeds C
    lead_time_p10: float | None               # day by which P(breach so far) reaches 10%
    lead_time_p50: float | None               # ... reaches 50%
    expected_peak: float = np.nan
    peak_hi95: float = np.nan
    status: str = "GREEN"
    notes: list[str] = field(default_factory=list)


def _first_crossing_quantile(first: np.ndarray, q: float) -> float | None:
    v = float(np.percentile(first, q, method="lower"))
    return None if np.isinf(v) else v


def compute_risk(
    forecast: pd.DataFrame,
    threshold: float,
    current_census: float,
    horizon: int | None = None,
    n_sims: int = config.N_SIMS,
    seed: int = config.SEED,
) -> RiskSummary:
    """Breach probability and lead time for a forecast covering days 1..H."""
    horizon = int(horizon or forecast.index.max())
    paths = simulate_paths(forecast, n_sims=n_sims, seed=seed)
    breach = paths > threshold
    p_day = pd.Series(breach.mean(axis=0), index=forecast.index, name="p_breach")

    h_idx = int(np.searchsorted(forecast.index.to_numpy(), horizon, side="right"))
    p_any = float(breach[:, :h_idx].any(axis=1).mean())
    p_at = float(p_day.loc[horizon])

    already = current_census > threshold
    first = np.where(breach.any(axis=1), breach.argmax(axis=1) + 1, np.inf).astype(float)
    over = np.where(forecast["yhat"].to_numpy() > threshold)[0]
    expected = 0.0 if already else (float(forecast.index[over[0]]) if len(over) else None)
    p10 = 0.0 if already else _first_crossing_quantile(first, 10)
    p50 = 0.0 if already else _first_crossing_quantile(first, 50)

    window = forecast.iloc[:h_idx]
    if already or p_any >= config.RISK_RED:
        status = "RED"
    elif p_any >= config.RISK_AMBER:
        status = "AMBER"
    else:
        status = "GREEN"

    notes = []
    if already:
        notes.append("Census is already above the nominal threshold.")
    elif p10 is not None:
        notes.append(f"There is a 10% chance the threshold is crossed within {int(p10)} day(s).")
    return RiskSummary(
        threshold=float(threshold), current_census=float(current_census), horizon=horizon,
        p_breach_by_day=p_day, p_breach_at_horizon=p_at, p_breach_any_within_horizon=p_any,
        already_breached=bool(already), lead_time_expected=expected,
        lead_time_p10=p10, lead_time_p50=p50,
        expected_peak=float(window["yhat"].max()), peak_hi95=float(window["hi95"].max()),
        status=status, notes=notes,
    )


def default_threshold(census: pd.Series, lookback: int = 90, headroom: float = 1.05, round_to: int = 50) -> float:
    """A sensible starting capacity: recent peak + 5%, rounded. Replace with real capacity."""
    return float(round(census.tail(lookback).max() * headroom / round_to) * round_to)


def beds_at_confidence(forecast: pd.DataFrame, horizon: int, level: int = 80) -> float:
    """Upper band at ``horizon`` = beds to plan for at the chosen confidence."""
    return float(forecast.loc[horizon, f"hi{level}"])


# ---------------------------------------------------- sponsor bottleneck
def sponsor_bottleneck_alert(
    feat: pd.DataFrame,
    window: int = 7,
    ratio_warn: float = config.RATIO_WARN,
    ratio_crit: float = config.RATIO_CRIT,
    net_warn: float = config.NET_PRESSURE_WARN,
    threshold: float | None = None,
) -> dict:
    """Flow-based check that discharges (sponsor placements) are keeping up with transfers in.

    * ratio   = discharges / transfers over the last ``window`` days (<1 = backlog building)
    * net     = mean daily (transfers - discharges)
    * LOS     = census / discharge rate (Little's law) now vs ``2*window`` days earlier
    """
    last = feat.iloc[-window:]
    tra, dis = float(last["transferred"].sum()), float(last["discharged"].sum())
    ratio = dis / tra if tra > 0 else np.nan
    net = (tra - dis) / window

    def los(sl: pd.DataFrame) -> float:
        d = sl["discharged"].mean()
        return float(sl["hhs_care"].mean() / d) if d > 0 else np.nan

    los_now = los(last)
    prev = feat.iloc[-3 * window:-2 * window] if len(feat) >= 3 * window else feat.iloc[:window]
    los_prev = los(prev)
    los_change = (los_now / los_prev - 1) * 100 if los_prev and np.isfinite(los_prev) else np.nan

    if np.isfinite(ratio) and ratio < ratio_crit and net > net_warn:
        level = "CRITICAL"
    elif np.isfinite(ratio) and ratio < ratio_warn and net > 0:
        level = "WARNING"
    else:
        level = "OK"

    days_to_threshold = None
    if threshold is not None and net > 0:
        gap = threshold - float(feat["hhs_care"].iloc[-1])
        days_to_threshold = 0.0 if gap <= 0 else float(gap / net)

    msg = {
        "CRITICAL": f"Discharges are covering only {ratio:.0%} of transfers in; backlog is growing ~{net:.0f}/day.",
        "WARNING": f"Discharges cover {ratio:.0%} of transfers in; backlog is edging up (~{net:.0f}/day).",
        "OK": "Sponsor placements are keeping pace with transfers into HHS care.",
    }[level]
    return {
        "level": level, "message": msg, "discharge_transfer_ratio": ratio,
        "net_pressure_per_day": net, "los_days_now": los_now, "los_days_prev": los_prev,
        "los_change_pct": los_change, "days_to_threshold_at_current_rate": days_to_threshold,
        "window_days": window,
    }
