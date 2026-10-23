"""End-to-end run: clean -> features -> walk-forward backtest -> artifacts -> report.

    python scripts/train.py --data data/HHS_Unaccompanied_Alien_Children_Program.csv
    python scripts/train.py --synthetic            # demo on simulated data

Outputs
-------
artifacts/backtest_predictions.csv   every out-of-sample forecast (long format)
artifacts/metrics.csv                MAE/RMSE/MAPE/coverage by model and horizon
reports/executive_summary.md         regenerated briefing
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import config, data_pipeline as dp, features as fe, models as md, reporting, risk_engine as rk  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DEFAULT_DATA_PATH))
    ap.add_argument("--synthetic", action="store_true", help="generate and use simulated data (demo only)")
    ap.add_argument("--step", type=int, default=config.WF_STEP_DAYS, help="days between forecast origins")
    ap.add_argument("--max-folds", type=int, default=config.WF_MAX_FOLDS)
    ap.add_argument("--initial-train", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None, help="capacity threshold (default: recent peak +5%%)")
    ap.add_argument("--fast", action="store_true", help="skip Prophet (slow)")
    ap.add_argument("--causal", action="store_true", help="forward-fill only (no interpolation across gaps)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_note = None
    if a.synthetic:
        from scripts.make_synthetic_data import main as make

        sys.argv = [sys.argv[0]]
        a.data = str(make())
        data_note = ("ILLUSTRATIVE OUTPUT - generated from SYNTHETIC data to demonstrate the report format. "
                     "Re-run `python scripts/train.py` on the real HHS file before using any number below.")

    # 1) data + features
    clean = dp.load_clean_data(a.data, causal=a.causal)
    feat = fe.build_features(clean)
    print(f"cleaned: {clean.attrs}")

    # 2) walk-forward backtest
    factories = md.build_factories(include_slow=not a.fast)
    print("models:", list(factories))
    pred = md.walk_forward_backtest(
        feat, factories, initial_train=a.initial_train, step=a.step, max_folds=a.max_folds,
        progress=lambda k, n: print(f"  fold {k}/{n}", end="\r"),
    )
    print()
    metrics = md.compute_metrics(pred)
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    pred.to_csv(config.ARTIFACT_DIR / "backtest_predictions.csv", index=False)
    metrics.to_csv(config.ARTIFACT_DIR / "metrics.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(metrics.round(2).to_string(index=False))

    # 3) live forecast with the best model
    best = md.best_model(metrics)
    cal = md.calibrators_from_backtest(pred)
    fc = md.fit_and_forecast(feat, [best], calibrators=cal)[best]
    threshold = a.threshold or rk.default_threshold(feat[config.TARGET])
    census = float(feat[config.TARGET].iloc[-1])
    risks = {h: rk.compute_risk(fc, threshold, census, horizon=h) for h in config.HORIZONS}
    bottleneck = rk.sponsor_bottleneck_alert(feat, threshold=threshold)

    # 4) executive report
    md_text = reporting.render_executive_summary(
        feat=feat, metrics=metrics, model=best, forecast=fc, risks=risks, bottleneck=bottleneck,
        threshold=threshold, models_run=sorted(pred["model"].unique()),
        n_folds=int(pred["origin"].nunique()),
        interval_source=fc.attrs.get("interval_source", "native model intervals"),
        data_note=data_note,
    )
    config.REPORT_DIR.mkdir(exist_ok=True)
    out = config.REPORT_DIR / "executive_summary.md"
    out.write_text(md_text, encoding="utf-8")
    print(f"best model: {best} | report -> {out}")


if __name__ == "__main__":
    main()
