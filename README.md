UAC Program Predictive Intelligence System
A Streamlit dashboard for exploring Unaccompanied Alien Children (UAC) program capacity forecasts, capacity-breach risk, and sponsor-placement bottlenecks. It combines historical time-series data, forecasting models, uncertainty intervals, and operational risk indicators in one interactive interface.
Important: The included synthetic dataset and reports are for demonstration only. They are not official HHS data and should not be used to make real-world operational decisions. Use only data you are authorized to access.

Features
- Forecast & Capacity Risk: Forecast the daily census, view uncertainty bands, and estimate the probability of exceeding a selected capacity threshold.
- Pipeline & Bottleneck: Explore transfers, discharges, net pressure, and sponsor-placement bottleneck alerts.
- Model Benchmarks: Review walk-forward backtest predictions and evaluation metrics when artifacts are available.
- Data & Decomposition: Inspect data cleaning results and time-series decomposition.
- Interactive controls: Select forecast horizon and model, confidence bands, history window, capacity threshold, and alert settings.
- CSV input: Upload a compatible HHS UAC CSV through the dashboard sidebar.
- Synthetic demo mode: Try the dashboard without supplying a real dataset.
Technology Stack
- Python
- Streamlit
- Pandas and NumPy
- Plotly
- Statsmodels / SARIMAX
- Scikit-learn
- LightGBM and XGBoost (optional model backends)
- Prophet (optional forecasting backend)

Project Structure
uac_forecasting/
├── app.py                         # Streamlit dashboard entry point
├── requirements.txt               # Python dependencies
├── data/
│   └── synthetic_uac.csv           # Synthetic demonstration dataset
├── artifacts/
│   ├── backtest_predictions.csv   # Walk-forward backtest predictions
│   └── metrics.csv                # Model evaluation metrics
├── reports/
│   └── executive_summary.md       # Example executive report
├── scripts/
│   ├── make_synthetic_data.py     # Generate demo data
│   └── train.py                   # Run model backtests and generate artifacts
├── src/
│   ├── config.py                  # Settings and thresholds
│   ├── data_pipeline.py           # Data loading and cleaning
│   ├── features.py                # Feature engineering
│   ├── models.py                  # Forecasting models and evaluation
│   ├── risk_engine.py              # Capacity risk and bottleneck logic
│   └── reporting.py                # Report generation
└── tests/
    └── test_pipeline.py            # Pipeline and risk-engine tests

    Use Real Data
1. Obtain a CSV dataset that you are authorized to use and that matches the columns expected by the project's data pipeline.
2. Open the app and use HHS UAC CSV in the sidebar to upload the file.
3. Review the cleaning output and confirm the date range and values are appropriate before interpreting forecasts.
Do not commit confidential, restricted, or personally identifiable information to a public repository. Even for a private repository, follow your organization's data-handling requirements.
