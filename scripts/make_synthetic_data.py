"""Generate a SYNTHETIC file shaped like the HHS UAC CSV (for tests and demos only).

It reproduces the raw-file quirks the pipeline must handle:
  * dates as 'December 21, 2025', newest first
  * thousands separators in numbers ("2,484")
  * weekend / random missing rows
  * blank trailing records and a footnote row

The numbers are simulated from a simple stock-and-flow model and carry NO
information about the real program. Never draw conclusions from them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLS = [
    "Date",
    "Children apprehended and placed in CBP custody*",
    "Children in CBP custody",
    "Children transferred out of CBP custody",
    "Children in HHS Care",
    "Children discharged from HHS Care",
]


def simulate(start="2023-03-01", end="2025-12-21", seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    n = len(days)
    doy = days.dayofyear.to_numpy()
    dow = days.dayofweek.to_numpy()

    season = 1 + 0.35 * np.sin(2 * np.pi * (doy - 120) / 365)
    surge = np.where((days >= "2024-05-01") & (days <= "2024-08-15"), 1.7, 1.0)
    los = np.where((days >= "2024-09-15") & (days <= "2024-12-01"), 46.0, 32.0)  # placement bottleneck
    wk_dis = np.where(dow >= 5, 0.45, 1.0)

    A = rng.poisson(70 * season * surge)
    cbp, hhs = 180.0, 2300.0
    rows = []
    for t in range(n):
        T = int(max(0, rng.poisson(max(0.4 * cbp, 1))))
        D = int(max(0, rng.poisson(hhs / los[t] * wk_dis[t] * (1.0 if dow[t] < 5 else 1.0))))
        cbp = max(0.0, cbp + A[t] - T)
        hhs = max(0.0, hhs + T - D + rng.normal(0, 2))
        rows.append((days[t], A[t], round(cbp), T, round(hhs), D))
    return pd.DataFrame(rows, columns=["date", "apprehended", "cbp", "transferred", "hhs", "discharged"])


def to_raw_csv(df: pd.DataFrame, path: Path, seed=11) -> None:
    rng = np.random.default_rng(seed)
    keep = (df["date"].dt.dayofweek < 5) | (rng.random(len(df)) < 0.25)   # mostly weekdays
    keep &= rng.random(len(df)) > 0.03                                    # a few random holes
    d = df[keep].sort_values("date", ascending=False)
    fmt = lambda x: f"{int(x):,}"  # noqa: E731
    out = pd.DataFrame({
        COLS[0]: [f"{x:%B} {x.day}, {x.year}" for x in d["date"]],
        COLS[1]: d["apprehended"].map(fmt).to_numpy(),
        COLS[2]: d["cbp"].map(fmt).to_numpy(),
        COLS[3]: d["transferred"].map(fmt).to_numpy(),
        COLS[4]: d["hhs"].map(fmt).to_numpy(),
        COLS[5]: d["discharged"].map(fmt).to_numpy(),
    })
    foot = pd.DataFrame([["* Synthetic data - footnote row", "", "", "", "", ""]], columns=COLS)
    blanks = pd.DataFrame([[""] * 6] * 4, columns=COLS)
    pd.concat([out, foot, blanks], ignore_index=True).to_csv(path, index=False)


def main() -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "synthetic_uac.csv"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    to_raw_csv(simulate(seed=a.seed), p)
    print(f"wrote {p}")
    return p


if __name__ == "__main__":
    main()
