# UAC Program Predictive Intelligence System

Forecasts the HHS "Children in HHS Care" census, quantifies capacity-breach risk and
flags sponsor-placement bottlenecks.

```
src/config.py         all tunables (horizons, folds, risk thresholds, staffing ratios)
src/data_pipeline.py  clean CSV -> complete daily series, gap fill, STL
src/features.py       lags, rolling stats, net pressure, bottleneck signals, calendar
src/models.py         benchmarks, SARIMAX, Prophet, LightGBM/XGB, RF, ensemble,
                      walk-forward CV, interval calibration, metrics
src/risk_engine.py    Monte-Carlo breach probability, surge lead time, bottleneck alerts
src/reporting.py      renders reports/executive_summary.md from computed results
app.py                Streamlit dashboard
scripts/train.py      backtest + artifacts + report;  scripts/make_synthetic_data.py
tests/                leakage, cleaning, risk-engine tests
```

## Run
```bash
pip install -r requirements.txt
cp /path/to/HHS_Unaccompanied_Alien_Children_Program.csv data/
python scripts/train.py            # backtest, metrics, calibrated bands, executive report
streamlit run app.py
python -m tests.test_pipeline      # or: pytest -q
```
`--fast` skips Prophet; `--step`/`--max-folds` control backtest cost; `--causal` forward-fills only.

## Design notes
* **No leakage:** features use only data <= origin; direct models drop rows whose label lies
  beyond the origin; interval calibration only uses residuals already observed at the origin.
* **Direct multi-horizon GBM/RF** predict the census *change* (trees can't extrapolate levels).
* **Intervals:** SARIMAX/Prophet native; others via expanding residual quantiles.
* **Risk:** paths are sampled from the forecast quantiles with random-walk dependence.
* Staffing ratios in `config.py` are illustrative: replace with program standards.
* `reports/executive_summary.md` as shipped is from SYNTHETIC data; regenerate on real data.
