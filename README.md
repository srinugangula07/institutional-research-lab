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

## Phase 3B — F&O Stock Research Engine

- Dedicated all-F&O long-format CSV importer
- Stock+timestamp uniqueness and OHLC integrity validation
- Corrected 11:15 decision entry using the completed 10:45 candle
- Completed-candle checkpoints at 11:45, 13:15 and 15:15
- True high/low MFE and MAE
- Stock return relative to NIFTY
- Directional India VIX beta and correlation
- Defensive and high-risk VIX stock scores
- Sector–VIX sensitivity matrix

## Phase 3C — Point-in-Time RF + RS + VIX Backtest

- Rotation Factor calculated from four completed morning candles
- Stock RS versus NIFTY from 09:45 to the 11:15 decision gate
- Cross-sectional Sector RS
- Daily RF, Stock RS and Sector RS percentiles
- Provisional institutional directional score: RF 40%, Stock RS 35%, Sector RS 25%
- True rolling 60-session VIX percentile used only as a risk multiplier
- Long/short oriented returns, win rates, expectancy and sector attribution
- Adjustable minimum score threshold and downloadable trade-level evidence

## Phase 3D — Walk-Forward Calibration

- Separate long and short threshold sensitivity
- RF/Stock RS/Sector RS weight-grid testing
- Round-trip transaction-cost and slippage deductions
- Minimum trade-count gates
- Expanding training windows with strictly later test windows
- Out-of-sample win rate and net expectancy
- Production gate that fails automatically when OOS expectancy is non-positive

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
## Futures confirmation layer

Upload the cumulative `all_fno_futures_research_single_file.csv` together with
the ALL F&O stock research file, then open **Futures Confirmation**. Invalid
weekend/out-of-window captures are rejected. The module unlocks descriptive
results after 20 matched sessions and keeps calibration conclusions locked until
at least 60 matched sessions are available.
## Cross-sectional portfolio research

The **Cross-Sectional Portfolio** module tests equal-weight top-score longs
against bottom-score shorts with sector concentration caps, round-trip costs,
membership churn, VIX-regime attribution and an untouched holdout period. A
negative holdout expectancy or Sharpe fails the portfolio gate.
## Robustness and stress testing

The **Robustness & Stress Tests** module runs five-session block bootstraps,
basket-size and cost grids, and leave-one-sector-out dependency checks. The
robustness gate requires a positive lower confidence bound, at least 95%
bootstrap probability of positive expectancy, broad parameter stability and
broad sector-exclusion stability.
