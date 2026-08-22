# Institutional Research, Backtesting & Market Replay Lab

Independent Streamlit research application complementary to the live
`institutional-market-dashboard`.

## Phase 1 scope

- Dataset health and schema validation
- Point-in-time India VIX feature engineering
- 60/252-observation VIX percentiles
- VIX change Z-score and shock/crush detection
- NIFTY-return versus VIX-change correlation
- NIFTY–VIX confirmation/divergence classification
- VIX-adjusted risk multiplier
- Baseline forward-return study
- Candle-by-candle market replay foundation

All rolling features use current and prior observations only. Synthetic demonstration data is
provided strictly to verify the interface and must not be treated as trading evidence.

## Required CSV columns

```text
timestamp,nifty_close,india_vix_close
```

See `data/sample_schema.csv`.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Test

```bash
pytest -q
```

## Streamlit Community Cloud

1. Create a new GitHub repository, for example `institutional-research-lab`.
2. Upload every file and folder from this project.
3. In Streamlit Community Cloud, select the repository and set the main file to `app.py`.
4. Deploy. Phase 1 does not require secrets or a Zerodha access token.

## Next build phases

1. Historical DuckDB/Parquet market store
2. RF + Sector RS + Futures OI + Options signal import
3. True NSE-session 11:15 validation and forward outcomes
4. Setup-level backtester with costs, MAE, MFE and expectancy
5. Market Profile/Volume Profile replay
6. Portfolio backtesting and Monte Carlo analysis
7. Walk-forward validation and model-drift reports

