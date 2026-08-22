# Institutional Research, Backtesting & Market Replay Lab

Independent Streamlit research application complementary to the live
`institutional-market-dashboard`.

## Phase 2 scope

- Dataset health and schema validation
- Point-in-time India VIX feature engineering
- 60/252-observation VIX percentiles
- VIX change Z-score and shock/crush detection
- NIFTY-return versus VIX-change correlation
- NIFTY–VIX confirmation/divergence classification
- VIX-adjusted risk multiplier
- Baseline forward-return study
- Candle-by-candle market replay foundation
- Multi-file historical CSV import bridge
- Common NIFTY/India VIX column-name standardisation
- Incremental timestamp merge and deduplication
- Weekday NSE-session filtering (09:15–15:30)
- Date-range research controls and session-quality report
- Actual 11:15 extraction with returns to 12:00, 13:30 and 15:15
- Close-path MFE/MAE and VIX-regime outcome summary
- Downloadable merged history and 11:15 outcome datasets

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
4. Deploy. Phase 2 does not require secrets or a second Zerodha login.

## Updating the live Phase 1 deployment

Upload the Phase 2 project contents to the same GitHub repository and choose **Add files →
Upload files**. GitHub will replace changed files and add `core/history.py` plus
`tests/test_history.py`. Streamlit Community Cloud redeploys automatically after the commit.

## Next build phases

1. Durable historical DuckDB/Parquet or cloud database store
2. RF + Sector RS + Futures OI + Options signal import
3. Setup-level backtester with costs, high/low MAE, MFE and expectancy
5. Market Profile/Volume Profile replay
6. Portfolio backtesting and Monte Carlo analysis
7. Walk-forward validation and model-drift reports
