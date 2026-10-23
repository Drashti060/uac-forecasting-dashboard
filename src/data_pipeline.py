"""Data ingestion and cleaning for the HHS Unaccompanied Alien Children dataset.

Steps
-----
1. Read the raw CSV as strings (so nothing is silently mangled).
2. Normalise the verbose column headers to short canonical names.
3. Drop empty / trailing / footnote rows and parse ``Date``.
4. Convert comma-formatted numbers ("2,484") to numeric.
5. Re-index to a complete daily calendar and fill non-reporting days
   (linear interpolation inside gaps, forward-fill at the tail).
6. Optionally decompose the target with STL.

Leakage note
------------
Linear interpolation of an *interior* gap uses the value on the far side of the
gap. That is fine for descriptive work, but if you want a strictly causal
series pass ``causal=True`` (forward-fill only). Imputed days are always
flagged in the ``is_imputed`` column so models/evaluation can inspect them.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ columns
def canonical_name(col: str) -> str | None:
    """Map a verbose raw header to its canonical short name (or None)."""
    c = re.sub(r"[^a-z ]", " ", str(col).lower())
    c = " ".join(c.split())
    if c == "date" or c.startswith("date"):
        return "date"
    if "apprehended" in c:
        return "apprehended"
    if "transferred" in c:
        return "transferred"
    if "discharged" in c:
        return "discharged"
    if "hhs" in c:
        return "hhs_care"
    if "cbp" in c:
        return "cbp_custody"
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping: dict[str, str] = {}
    for col in df.columns:
        name = canonical_name(col)
        if name and name not in mapping.values():
            mapping[col] = name
    out = df.rename(columns=mapping)[list(mapping.values())]
    missing = {"date", *config.RAW_COLUMNS} - set(out.columns)
    if missing:
        raise ValueError(
            f"Could not locate required columns {sorted(missing)}. "
            f"Headers found: {list(df.columns)}"
        )
    return out[["date", *config.RAW_COLUMNS]]


# ------------------------------------------------------------------ cleaning
_JUNK = re.compile(r"[^0-9.\-]")


def _to_number(v):
    """'2,484' -> 2484.0 ; None / blanks / junk -> NaN (pure-Python so it never depends on Arrow's regex engine)."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return np.nan
    txt = _JUNK.sub("", str(v))
    try:
        return float(txt)
    except ValueError:
        return np.nan


def clean_numeric(s: pd.Series) -> pd.Series:
    """'2,484' -> 2484.0 ; blanks / junk -> NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")
    return pd.Series([_to_number(v) for v in s.astype(object)], index=s.index, dtype="float64")


def parse_dates(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    d = pd.to_datetime(s, format=config.DATE_FORMAT, errors="coerce")
    if d.isna().any():  # try a permissive parse for stragglers
        d2 = pd.to_datetime(s[d.isna()], errors="coerce", format="mixed")
        d = d.fillna(d2)
    return d


def fill_gaps(df: pd.DataFrame, causal: bool = False) -> pd.DataFrame:
    """Reindex to daily frequency and fill non-reporting days.

    ``is_imputed`` is True for any day that had at least one missing value
    before filling.
    """
    df = df.asfreq("D")
    cols = list(config.RAW_COLUMNS)
    imputed = df[cols].isna().any(axis=1)

    # Warn about long gaps (they make interpolation unreliable).
    gap_id = (~imputed).cumsum()
    gap_len = imputed.groupby(gap_id).sum()
    if (gap_len > config.MAX_GAP_WARN_DAYS).any():
        log.warning(
            "%d reporting gap(s) longer than %d days - interpolation there is low-confidence.",
            int((gap_len > config.MAX_GAP_WARN_DAYS).sum()),
            config.MAX_GAP_WARN_DAYS,
        )

    filled = df[cols].copy()
    if not causal:
        filled = filled.interpolate(method="linear", limit_area="inside")
    filled = filled.ffill()                # tail / causal mode
    filled = filled.dropna(how="any")      # leading rows before every column has started
    filled[cols] = filled[cols].clip(lower=0)
    filled["is_imputed"] = imputed.reindex(filled.index).astype(bool)
    return filled


def clean_dataframe(raw: pd.DataFrame, causal: bool = False) -> pd.DataFrame:
    """Raw string frame -> clean daily frame (index=DatetimeIndex, freq='D')."""
    n_raw = len(raw)
    df = standardize_columns(raw)
    df["date"] = parse_dates(df["date"])
    for c in config.RAW_COLUMNS:
        df[c] = clean_numeric(df[c])

    df = df.dropna(subset=["date"])                                  # footnotes / blank tails
    df = df.dropna(subset=list(config.RAW_COLUMNS), how="all")       # date but no data
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
    n_reported = len(df)
    df = df.set_index("date")
    df.index.name = "date"

    out = fill_gaps(df, causal=causal)
    out.attrs.update(
        n_raw_rows=n_raw,
        n_reported_days=n_reported,
        n_calendar_days=len(out),
        n_imputed_days=int(out["is_imputed"].sum()),
        causal_fill=causal,
    )
    return out


def load_clean_data(source: Any = config.DEFAULT_DATA_PATH, causal: bool = False) -> pd.DataFrame:
    """Read ``source`` (path or file-like) and return the clean daily frame."""
    raw = pd.read_csv(source, dtype=str, encoding="utf-8-sig", skipinitialspace=True)
    return clean_dataframe(raw, causal=causal)


# ----------------------------------------------------------- decomposition
def decompose_series(
    series: pd.Series, period: int = config.SEASONAL_PERIOD, robust: bool = True
) -> pd.DataFrame:
    """STL decomposition -> observed / trend / seasonal / resid.

    Falls back to a classical moving-average decomposition when statsmodels is
    unavailable (``result.attrs['method']`` tells you which one ran).

    Descriptive tool only - STL uses the full series, so its components must
    NOT be used as model features in a walk-forward evaluation.
    """
    s = series.astype("float64")
    try:
        from statsmodels.tsa.seasonal import STL

        res = STL(s, period=period, robust=robust).fit()
        out = pd.DataFrame(
            {"observed": s, "trend": res.trend, "seasonal": res.seasonal, "resid": res.resid}
        )
        out.attrs["method"] = "STL"
        return out
    except ImportError:
        trend = s.rolling(period, center=True, min_periods=1).mean()
        detr = s - trend
        pos = np.arange(len(s)) % period
        seas = detr.groupby(pos).transform("mean")
        seas = seas - seas.mean()
        out = pd.DataFrame(
            {"observed": s, "trend": trend, "seasonal": seas, "resid": s - trend - seas}
        )
        out.attrs["method"] = "classical (statsmodels not installed)"
        return out


def seasonal_strength(decomp: pd.DataFrame) -> float:
    """Hyndman's seasonal strength  F_s = max(0, 1 - Var(R)/Var(S+R))."""
    r, s = decomp["resid"], decomp["seasonal"]
    denom = float(np.var(s + r))
    return float(max(0.0, 1.0 - np.var(r) / denom)) if denom > 0 else 0.0


if __name__ == "__main__":  # quick manual check:  python -m src.data_pipeline
    logging.basicConfig(level=logging.INFO)
    d = load_clean_data()
    print(d.tail())
    print(d.attrs)
