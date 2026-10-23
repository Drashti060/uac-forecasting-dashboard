"""Central configuration for the UAC forecasting system.

Every tunable lives here so the pipeline, models, risk engine, dashboard and
report generator all agree on the same defaults.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_DATA_PATH = DATA_DIR / "HHS_Unaccompanied_Alien_Children_Program.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

# ------------------------------------------------------------------- schema
# Canonical short names used everywhere after cleaning.
TARGET = "hhs_care"                      # active census in HHS shelters
RAW_COLUMNS = ("apprehended", "cbp_custody", "transferred", "hhs_care", "discharged")
DATE_FORMAT = "%B %d, %Y"                # e.g. "December 21, 2025"

# --------------------------------------------------------------- data prep
MAX_GAP_WARN_DAYS = 14                   # warn if a reporting gap is longer than this
SEASONAL_PERIOD = 7                      # weekly cycle

# ----------------------------------------------------------------- features
LAGS = (1, 3, 7, 14, 30)                 # lag 0 is the raw (current) column itself
WINDOWS = (7, 14, 30)

# ------------------------------------------------------------------ modeling
HORIZONS = (7, 14, 30)                   # benchmark horizons
PATH_HORIZON = 30                        # dashboard fan chart length
SEED = 42

# Walk-forward (expanding window) cross-validation
INITIAL_TRAIN_DAYS = 365
WF_STEP_DAYS = 14
WF_MAX_FOLDS = 26                        # keep the most recent N origins (None = all)
CALIBRATION_MIN_OBS = 8                  # min past residuals before an interval is issued

# ---------------------------------------------------------------- risk engine
N_SIMS = 10_000
RISK_AMBER = 0.20                        # P(breach within horizon) triggering AMBER
RISK_RED = 0.50                          # ... triggering RED
RATIO_WARN = 0.95                        # 7d discharges / 7d transfers below this = warning
RATIO_CRIT = 0.85                        # ... below this = critical
NET_PRESSURE_WARN = 20.0                 # children/day of net inflow considered material

# ------------------------------------------------- staffing translation (EDIT!)
# ILLUSTRATIVE planning ratios only. Replace with the program's current
# minimum-standards before using the executive report for real decisions.
CHILDREN_PER_CASE_MANAGER = 12
CHILDREN_PER_YOUTH_CARE_WORKER = 8       # per waking-hours shift position
