from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.demo import make_demo_data
from core.fno_research import (
    build_stock_1115_outcomes,
    fno_quality_tables,
    prepare_fno_history,
    sector_vix_sensitivity,
    stock_vix_sensitivity,
    validate_fno_history,
)
from core.history import (
    build_1115_outcomes,
    merge_history,
    normalise_columns,
    outcome_summary,
    restrict_nse_session,
    session_quality,
)
from core.research import correlation_table, forward_return_study
from core.point_in_time import build_1115_rf_rs_features, signal_backtest_summary
from core.walk_forward import calibration_surface, expanding_walk_forward, threshold_sensitivity
from core.vix import enrich_vix_features, validate_market_data, vix_risk_multiplier
from core.futures_confirmation import (
    build_futures_confirmation,
    futures_confirmation_summary,
    prepare_futures_snapshots,
    validate_futures_snapshots,
)
from core.portfolio_research import (
    build_cross_sectional_portfolio,
    portfolio_summary,
    regime_summary,
)
from core.robustness import block_bootstrap, leave_one_sector_out, parameter_stability
from core.information_coefficient import (
    daily_information_coefficients,
    ic_regime_summary,
    information_coefficient_summary,
    quintile_spread,
)


st.set_page_config(
    page_title="Institutional Research Lab",
    page_icon="🔬",
    layout="wide",
)

st.title("Institutional Research, Backtesting & Market Replay Lab")
st.caption("Point-in-time research engine • India VIX regime intelligence • No-look-ahead design")

with st.sidebar:
    st.header("Research controls")
    uploaded = st.file_uploader(
        "Upload historical CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one combined file or separate/incremental NIFTY and India VIX files.",
    )
    fno_uploaded = st.file_uploader(
        "Upload ALL F&O research CSV",
        type=["csv"],
        key="all_fno_research_upload",
        help="Upload all_fno_stock_research_single_file.csv for stock-level research.",
    )
    futures_uploaded = st.file_uploader(
        "Upload cumulative Futures Research CSV",
        type=["csv"],
        key="futures_research_upload",
        help="Upload all_fno_futures_research_single_file.csv from the live dashboard.",
    )
    use_demo = st.toggle("Use demonstration dataset", value=not uploaded)
    page = st.radio(
        "Module",
        [
            "Research Overview",
            "Historical Import Bridge",
            "Data Health",
            "India VIX Intelligence",
            "VIX–Index Correlation",
            "11:15 Validation",
            "Market Replay",
            "Setup Backtester",
            "F&O Data Health",
            "Stock 11:15 Outcomes",
            "Stock VIX Sensitivity",
            "Sector–VIX Matrix",
            "RF + RS + VIX Signals",
            "RF + RS + VIX Backtest",
            "Calibration Sensitivity",
            "Walk-Forward Validation",
            "Cross-Sectional Portfolio",
            "Robustness & Stress Tests",
            "Information Coefficient & Decay",
            "Futures Confirmation",
        ],
    )


@st.cache_data(show_spinner=False)
def load_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


@st.cache_data(show_spinner="Building point-in-time F&O research outcomes...")
def load_fno_research(raw: bytes):
    source = pd.read_csv(io.BytesIO(raw))
    problems = validate_fno_history(source)
    if problems:
        return source, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), problems
    prepared = prepare_fno_history(source)
    outcomes = build_stock_1115_outcomes(prepared)
    stock_quality, session_quality = fno_quality_tables(prepared)
    return prepared, outcomes, stock_quality, session_quality, []


@st.cache_data(show_spinner="Calculating point-in-time RF, RS and VIX signals...")
def load_phase3c_signals(raw: bytes):
    source = pd.read_csv(io.BytesIO(raw))
    problems = validate_fno_history(source)
    if problems:
        return pd.DataFrame(), problems
    prepared = prepare_fno_history(source)
    outcomes = build_stock_1115_outcomes(prepared)
    return build_1115_rf_rs_features(prepared, outcomes), []


is_demo = False
if uploaded:
    raw_df = None
    for item in uploaded:
        incoming = normalise_columns(load_csv(item.getvalue()))
        raw_df = merge_history(raw_df, incoming)
elif use_demo:
    raw_df = make_demo_data()
    is_demo = True
    st.info("Demonstration mode uses synthetic data. Do not treat its statistics as trading evidence.")
else:
    st.warning("Upload a CSV or enable the demonstration dataset.")
    st.stop()

problems = validate_market_data(raw_df)
if problems:
    st.error("The dataset cannot be analysed yet.")
    for problem in problems:
        st.write(f"• {problem}")
    st.stop()

session_raw = restrict_nse_session(raw_df)
df = enrich_vix_features(session_raw)
if df.empty:
    st.error("No weekday observations between 09:15 and 15:30 remain after NSE-session filtering.")
    st.stop()

with st.sidebar:
    min_date = df["timestamp"].dt.date.min()
    max_date = df["timestamp"].dt.date.max()
    date_range = st.date_input("Research date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)].copy()

latest = df.iloc[-1]

if page == "Research Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("India VIX", f"{latest['india_vix_close']:.2f}", f"{latest['vix_change_pct']:.2f}%")
    c3.metric("VIX regime", latest["vix_regime"])
    c4.metric("Risk multiplier", f"{vix_risk_multiplier(latest['vix_regime']):.2f}×")

    st.subheader("Research pipeline")
    st.write(
        "Data Health → India VIX Regime → Correlation → 11:15 Validation → "
        "Setup Backtest → Market Replay → Walk-Forward Validation"
    )
    st.success("Phase 2 is operational: historical import, NSE sessions, VIX intelligence and 11:15 outcomes.")

elif page == "Historical Import Bridge":
    st.subheader("Historical Database & Import Bridge")
    st.write(
        "Upload combined or separate CSV exports. The bridge standardises common column names, "
        "joins matching timestamps, removes duplicates and keeps NSE-session observations."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Imported files", len(uploaded) if uploaded else 0)
    c2.metric("Historical rows", f"{len(df):,}")
    c3.metric("Sessions", df["session_date"].nunique())
    c4.metric("11:15 sessions", len(build_1115_outcomes(df)))
    st.code("timestamp,nifty_close,india_vix_close", language="text")
    st.download_button(
        "Download merged research data pack",
        df.to_csv(index=False).encode(),
        "institutional_research_history.csv",
        "text/csv",
    )
    st.caption(
        "Streamlit's local filesystem is not permanent. Download this merged data pack after every "
        "incremental import; durable cloud database storage will be added in a later phase."
    )

elif page == "Data Health":
    st.subheader("Dataset health")
    st.success("Required columns, timestamps and positive price values passed validation.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Start", str(df["timestamp"].min()))
    c2.metric("End", str(df["timestamp"].max()))
    c3.metric("Duplicate timestamps", int(df["timestamp"].duplicated().sum()))
    quality = session_quality(df)
    complete_pct = quality["complete"].mean() * 100 if not quality.empty else 0
    st.metric("Complete sessions", f"{complete_pct:.1f}%")
    st.dataframe(quality.tail(30), use_container_width=True)
    st.subheader("Latest observations")
    st.dataframe(df.tail(20), use_container_width=True)

elif page == "India VIX Intelligence":
    st.subheader("India VIX regime intelligence")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["india_vix_close"], name="India VIX"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vix_sma_20"], name="20-bar mean"))
    fig.update_layout(height=430, xaxis_title=None, yaxis_title="VIX")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        df[[
            "timestamp", "india_vix_close", "vix_change_pct", "vix_percentile_60",
            "vix_change_zscore_20", "vix_regime", "price_vix_state",
        ]].tail(50),
        use_container_width=True,
    )

elif page == "VIX–Index Correlation":
    st.subheader("NIFTY return versus India VIX change")
    corr = correlation_table(df, [5, 20, 60, 252])
    st.dataframe(corr.style.format({"NIFTY–VIX correlation": "{:.3f}"}), use_container_width=True)
    st.scatter_chart(df, x="vix_change_pct", y="nifty_return_pct", use_container_width=True)
    st.caption("Correlation is calculated on percentage changes, not raw index levels.")

elif page == "11:15 Validation":
    st.subheader("11:15 forward-return research")
    if is_demo:
        st.warning("Demonstration outcomes are synthetic and are only for interface verification.")
    outcomes = build_1115_outcomes(df)
    if outcomes.empty:
        st.info("No 11:15 observations exist in the current dataset.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Valid sessions", len(outcomes))
        c2.metric("Average MFE", f"{outcomes['mfe_close_pct'].mean():.2f}%")
        c3.metric("Average MAE", f"{outcomes['mae_close_pct'].mean():.2f}%")
        st.subheader("VIX-regime outcome summary")
        st.dataframe(outcome_summary(outcomes), use_container_width=True)
        st.subheader("Session-level outcomes")
        st.dataframe(outcomes, use_container_width=True)
        st.download_button(
            "Download 11:15 outcome dataset",
            outcomes.to_csv(index=False).encode(),
            "phase2_1115_vix_outcomes.csv",
            "text/csv",
        )

elif page == "Market Replay":
    st.subheader("Point-in-time market replay")
    replay_index = st.slider("Visible observation", 1, len(df), min(len(df), 100))
    visible = df.iloc[:replay_index]
    fig = go.Figure(go.Scatter(x=visible["timestamp"], y=visible["nifty_close"], name="NIFTY"))
    fig.update_layout(height=450, xaxis_title=None, yaxis_title="NIFTY")
    st.plotly_chart(fig, use_container_width=True)
    replay_latest = visible.iloc[-1]
    st.write(
        f"Current point-in-time state: **{replay_latest['price_vix_state']}** | "
        f"VIX regime: **{replay_latest['vix_regime']}**"
    )

elif page == "Setup Backtester":
    st.subheader("VIX-regime baseline backtest")
    horizons = st.multiselect("Forward horizons (bars)", [1, 2, 4, 8, 13], default=[1, 2, 4])
    if horizons:
        result = forward_return_study(df, "vix_regime", horizons)
        st.dataframe(result, use_container_width=True)
        st.download_button(
            "Download research result",
            result.to_csv(index=False).encode(),
            "vix_regime_backtest.csv",
            "text/csv",
        )
    st.caption("Next phase will join RF, Sector RS, Futures OI, Options and auction-structure signals.")

elif page == "F&O Data Health":
    st.subheader("All NSE F&O historical data health")
    if fno_uploaded is None:
        st.info("Upload `all_fno_stock_research_single_file.csv` in the sidebar.")
    else:
        fno, outcomes, stock_quality, session_quality, fno_problems = load_fno_research(
            fno_uploaded.getvalue()
        )
        if fno_problems:
            st.error("F&O dataset validation failed.")
            for problem in fno_problems:
                st.write(f"• {problem}")
        else:
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Rows", f"{len(fno):,}")
            d2.metric("F&O Stocks", fno["Stock"].nunique())
            d3.metric("Sectors", fno["Sector"].nunique())
            d4.metric("Sessions", fno["session_date"].nunique())
            st.success("Stock+timestamp uniqueness, OHLC integrity and market context passed validation.")
            st.subheader("Session coverage")
            st.dataframe(session_quality, use_container_width=True, hide_index=True)
            st.subheader("Stock coverage")
            st.dataframe(stock_quality, use_container_width=True, hide_index=True)

elif page == "Stock 11:15 Outcomes":
    st.subheader("Point-in-time stock outcomes at the 11:15 decision gate")
    st.caption(
        "No-look-ahead correction: entry uses the 10:45 candle close, which becomes known at "
        "11:15. Checkpoints use completed 30-minute candles at 11:45, 13:15 and 15:15."
    )
    if fno_uploaded is None:
        st.info("Upload the ALL F&O research CSV in the sidebar.")
    else:
        fno, outcomes, _, _, fno_problems = load_fno_research(fno_uploaded.getvalue())
        if fno_problems:
            st.error("Correct the F&O input file before analysis: " + " | ".join(fno_problems))
        elif outcomes.empty:
            st.warning("No valid 10:45 decision bars were found.")
        else:
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Stock-Sessions", f"{len(outcomes):,}")
            o2.metric("Stocks", outcomes["Stock"].nunique())
            o3.metric("Average MFE", f"{outcomes['mfe_high_pct'].mean():.2f}%")
            o4.metric("Average MAE", f"{outcomes['mae_low_pct'].mean():.2f}%")
            sectors = ["ALL"] + sorted(outcomes["Sector"].dropna().unique().tolist())
            selected_sector = st.selectbox("Sector filter", sectors, key="outcome_sector")
            filtered = outcomes if selected_sector == "ALL" else outcomes[outcomes["Sector"] == selected_sector]
            selected_stock = st.selectbox(
                "Stock filter",
                ["ALL"] + sorted(filtered["Stock"].unique().tolist()),
                key="outcome_stock",
            )
            if selected_stock != "ALL":
                filtered = filtered[filtered["Stock"] == selected_stock]
            st.dataframe(filtered, use_container_width=True, hide_index=True)
            st.download_button(
                "Download corrected stock 11:15 outcomes",
                filtered.to_csv(index=False).encode(),
                "phase3b_fno_stock_1115_outcomes.csv",
                "text/csv",
            )

elif page == "Stock VIX Sensitivity":
    st.subheader("Stock-level India VIX sensitivity rankings")
    if fno_uploaded is None:
        st.info("Upload the ALL F&O research CSV in the sidebar.")
    else:
        _, outcomes, _, _, fno_problems = load_fno_research(fno_uploaded.getvalue())
        if fno_problems:
            st.error("Correct the F&O input file before analysis: " + " | ".join(fno_problems))
        else:
            minimum_sessions = st.slider("Minimum valid sessions", 20, 80, 50, 5)
            sensitivity = stock_vix_sensitivity(outcomes, minimum_sessions)
            if sensitivity.empty:
                st.warning("No stocks passed the selected session threshold.")
            else:
                st.caption(
                    "VIX beta measures directional sensitivity to the VIX change from the 11:15 "
                    "decision gate to 15:15. Negative beta indicates risk-off sensitivity."
                )
                left, right = st.columns(2)
                with left:
                    st.markdown("#### Most defensive when VIX rises")
                    st.dataframe(
                        sensitivity.sort_values("VIX_Defensive_Score", ascending=False).head(25),
                        use_container_width=True,
                        hide_index=True,
                    )
                with right:
                    st.markdown("#### Highest VIX risk")
                    st.dataframe(
                        sensitivity.sort_values("VIX_Risk_Score", ascending=False).head(25),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.download_button(
                    "Download complete stock VIX ranking",
                    sensitivity.to_csv(index=False).encode(),
                    "phase3b_stock_vix_sensitivity.csv",
                    "text/csv",
                )

elif page == "Sector–VIX Matrix":
    st.subheader("Sector–India VIX sensitivity matrix")
    if fno_uploaded is None:
        st.info("Upload the ALL F&O research CSV in the sidebar.")
    else:
        _, outcomes, _, _, fno_problems = load_fno_research(fno_uploaded.getvalue())
        if fno_problems:
            st.error("Correct the F&O input file before analysis: " + " | ".join(fno_problems))
        else:
            sector_matrix = sector_vix_sensitivity(outcomes, minimum_sessions=30)
            if sector_matrix.empty:
                st.warning("Insufficient sector observations.")
            else:
                st.dataframe(sector_matrix, use_container_width=True, hide_index=True)
                st.bar_chart(
                    sector_matrix.set_index("Sector")[["VIX_Beta"]],
                    use_container_width=True,
                )
                st.download_button(
                    "Download sector–VIX matrix",
                    sector_matrix.to_csv(index=False).encode(),
                    "phase3b_sector_vix_matrix.csv",
                    "text/csv",
                )

elif page == "RF + RS + VIX Signals":
    st.subheader("Point-in-time RF + Stock RS + Sector RS + India VIX signals")
    st.caption(
        "All directional inputs use only the 09:15, 09:45, 10:15 and 10:45 candles "
        "completed by the 11:15 decision gate. India VIX modifies risk, not direction."
    )
    if fno_uploaded is None:
        st.info("Upload the ALL F&O research CSV in the sidebar.")
    else:
        signals, signal_problems = load_phase3c_signals(fno_uploaded.getvalue())
        if signal_problems:
            st.error(" | ".join(signal_problems))
        elif signals.empty:
            st.warning("No point-in-time signals were created.")
        else:
            dates = sorted(signals["session_date"].unique(), reverse=True)
            signal_date = st.selectbox("Research session", dates, key="phase3c_signal_date")
            day = signals[signals["session_date"] == signal_date].copy()
            long_df = day[day["Signal"] == "LONG"].nlargest(20, "Institutional_Directional_Score")
            short_df = day[day["Signal"] == "SHORT"].nsmallest(20, "Institutional_Directional_Score")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Stocks", len(day))
            s2.metric("Long signals", len(day[day["Signal"] == "LONG"]))
            s3.metric("Short signals", len(day[day["Signal"] == "SHORT"]))
            s4.metric("VIX risk multiplier", f"{day['VIX_Risk_Multiplier'].iloc[0]:.2f}×")
            left, right = st.columns(2)
            display_columns = [
                "Stock", "Sector", "RF_1115", "Stock_RS_1115_%", "Sector_RS_1115_%",
                "Institutional_Directional_Score", "Signal", "VIX_60D_Percentile",
                "VIX_Risk_Multiplier", "return_1515_pct", "relative_return_1515_pct",
            ]
            with left:
                st.markdown("#### Top point-in-time longs")
                st.dataframe(long_df[display_columns], use_container_width=True, hide_index=True)
            with right:
                st.markdown("#### Top point-in-time shorts")
                st.dataframe(short_df[display_columns], use_container_width=True, hide_index=True)
            st.download_button(
                "Download complete Phase 3C signal dataset",
                signals.to_csv(index=False).encode(),
                "phase3c_rf_rs_vix_signals.csv",
                "text/csv",
            )

elif page == "RF + RS + VIX Backtest":
    st.subheader("RF + RS + VIX point-in-time backtest")
    if fno_uploaded is None:
        st.info("Upload the ALL F&O research CSV in the sidebar.")
    else:
        signals, signal_problems = load_phase3c_signals(fno_uploaded.getvalue())
        if signal_problems:
            st.error(" | ".join(signal_problems))
        else:
            summary = signal_backtest_summary(signals)
            st.dataframe(summary, use_container_width=True, hide_index=True)
            active = signals[signals["Signal"].isin(["LONG", "SHORT"])].copy()
            threshold = st.slider("Minimum absolute institutional score", 40, 80, 40, 5)
            active = active[active["Institutional_Directional_Score"].abs() >= threshold]
            if active.empty:
                st.warning("No trades pass this score threshold.")
            else:
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Signals", f"{len(active):,}")
                b2.metric("Sessions", active["session_date"].nunique())
                b3.metric("Win rate", f"{(active['Oriented_Return_1515_%'] > 0).mean() * 100:.1f}%")
                b4.metric("Expectancy", f"{active['Oriented_Return_1515_%'].mean():.3f}%")
                by_sector = active.groupby(["Sector", "Signal"], as_index=False).agg(
                    Trades=("Stock", "size"),
                    Win_Rate=("Oriented_Return_1515_%", lambda x: (x > 0).mean() * 100),
                    Expectancy=("Oriented_Return_1515_%", "mean"),
                    Relative_Edge=("relative_return_1515_pct", "mean"),
                )
                st.subheader("Sector attribution")
                st.dataframe(by_sector, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download filtered Phase 3C backtest trades",
                    active.to_csv(index=False).encode(),
                    "phase3c_rf_rs_vix_backtest_trades.csv",
                    "text/csv",
                )

elif page == "Calibration Sensitivity":
    st.subheader("Phase 3D — Threshold, weight and transaction-cost sensitivity")
    if fno_uploaded is None:
        st.info("Upload the ALL F&O research CSV in the sidebar.")
    else:
        signals, signal_problems = load_phase3c_signals(fno_uploaded.getvalue())
        if signal_problems:
            st.error(" | ".join(signal_problems))
        else:
            cost_bps = st.slider("Round-trip cost and slippage (bps)", 0, 30, 10, 1)
            sensitivity = threshold_sensitivity(signals, cost_bps=cost_bps)
            st.markdown("#### Existing 40/35/25 weights")
            st.dataframe(sensitivity, use_container_width=True, hide_index=True)
            with st.spinner("Testing institutional weight and long/short threshold combinations..."):
                surface = calibration_surface(signals, cost_bps=cost_bps, minimum_trades_per_side=100)
            st.markdown("#### Top in-sample combinations — research only")
            st.warning(
                "These are in-sample results and must not become production settings unless "
                "they also survive the Walk-Forward Validation module."
            )
            st.dataframe(surface.head(50), use_container_width=True, hide_index=True)
            st.download_button(
                "Download Phase 3D calibration surface",
                surface.to_csv(index=False).encode(),
                "phase3d_calibration_surface.csv",
                "text/csv",
            )

elif page == "Walk-Forward Validation":
    st.subheader("Phase 3D — Expanding walk-forward validation")
    st.caption("Each test window is evaluated using settings chosen only from earlier sessions.")
    if fno_uploaded is None:
        st.info("Upload the ALL F&O research CSV in the sidebar.")
    else:
        signals, signal_problems = load_phase3c_signals(fno_uploaded.getvalue())
        if signal_problems:
            st.error(" | ".join(signal_problems))
        else:
            w1, w2, w3 = st.columns(3)
            with w1:
                wf_cost = st.slider("Cost/slippage (bps)", 0, 30, 10, 1, key="wf_cost")
            with w2:
                train_sessions = st.slider("Initial training sessions", 40, 60, 50, 5)
            with w3:
                test_sessions = st.slider("Test-window sessions", 5, 15, 10, 5)
            with st.spinner("Running expanding walk-forward calibration..."):
                folds, trades = expanding_walk_forward(
                    signals, wf_cost, train_sessions, test_sessions
                )
            if folds.empty or trades.empty:
                st.warning("Insufficient sessions for the selected walk-forward configuration.")
            else:
                w1, w2, w3, w4 = st.columns(4)
                w1.metric("Folds", len(folds))
                w2.metric("Out-of-sample trades", f"{len(trades):,}")
                w3.metric("OOS net win rate", f"{(trades['Net_Oriented_Return_%'] > 0).mean() * 100:.1f}%")
                w4.metric("OOS net expectancy", f"{trades['Net_Oriented_Return_%'].mean():.3f}%")
                oos_expectancy = trades["Net_Oriented_Return_%"].mean()
                if oos_expectancy <= 0:
                    st.error(
                        "PRODUCTION GATE: FAILED. Out-of-sample expectancy is not positive after "
                        "costs. Keep RF + RS as a ranking layer; do not trade this model standalone."
                    )
                else:
                    st.success(
                        "Research gate passed provisionally. Require a larger sample and additional "
                        "confirmation layers before production use."
                    )
                st.dataframe(folds, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download walk-forward folds",
                    folds.to_csv(index=False).encode(),
                    "phase3d_walk_forward_folds.csv",
                    "text/csv",
                )
                st.download_button(
                    "Download out-of-sample trades",
                    trades.to_csv(index=False).encode(),
                    "phase3d_walk_forward_oos_trades.csv",
                    "text/csv",
                )

elif page == "Cross-Sectional Portfolio":
    st.subheader("Cross-Sectional Long/Short Portfolio Research")
    st.caption(
        "Tests RF + RS as a daily ranking spread: equal-weight top-score longs versus "
        "bottom-score shorts, with sector caps, transaction costs and an untouched holdout."
    )
    if fno_uploaded is None:
        st.info("Upload the ALL F&O research CSV in the sidebar.")
    else:
        signals, signal_problems = load_phase3c_signals(fno_uploaded.getvalue())
        if signal_problems:
            st.error(" | ".join(signal_problems))
        else:
            p1,p2,p3,p4 = st.columns(4)
            with p1:
                basket_size = st.select_slider("Stocks per side", options=[5,10,15,20,25], value=20)
            with p2:
                sector_cap = st.select_slider("Maximum per sector", options=[1,2,3,4,5], value=3)
            with p3:
                portfolio_cost = st.slider("Round-trip cost/slippage (bps)",0,30,10,1,key="portfolio_cost")
            with p4:
                holdout = st.slider("Untouched holdout sessions",10,30,20,5)
            daily, holdings = build_cross_sectional_portfolio(
                signals, basket_size=basket_size, max_per_sector=sector_cap,
                cost_bps=portfolio_cost,
            )
            summary = portfolio_summary(daily, holdout_sessions=holdout)
            if daily.empty:
                st.warning("No sessions contain enough stocks after sector-cap controls.")
            else:
                st.markdown("#### Portfolio validation")
                st.dataframe(summary,use_container_width=True,hide_index=True)
                holdout_row=summary[summary["Sample"].eq("HOLDOUT")]
                holdout_expectancy=holdout_row["Net_Expectancy_%"].iloc[0]
                holdout_sharpe=holdout_row["Annualised_Sharpe"].iloc[0]
                if len(daily) < 60:
                    st.warning("SAMPLE GATE: fewer than 60 portfolio sessions; conclusions are preliminary.")
                elif holdout_expectancy <= 0 or pd.isna(holdout_sharpe) or holdout_sharpe <= 0:
                    st.error(
                        "PORTFOLIO GATE: FAILED. The top-minus-bottom ranking spread does not "
                        "retain positive holdout performance after costs."
                    )
                else:
                    st.success(
                        "PORTFOLIO GATE: PROVISIONALLY PASSED. Continue robustness and "
                        "walk-forward testing before production use."
                    )
                st.line_chart(daily.set_index("session_date")["Equity_Index"])
                st.markdown("#### India VIX regime attribution")
                st.dataframe(regime_summary(daily),use_container_width=True,hide_index=True)
                c1,c2=st.columns(2)
                with c1:
                    st.download_button(
                        "Download daily portfolio results",daily.to_csv(index=False).encode(),
                        "phase4_cross_sectional_portfolio_daily.csv","text/csv",
                    )
                with c2:
                    st.download_button(
                        "Download portfolio holdings",holdings.to_csv(index=False).encode(),
                        "phase4_cross_sectional_portfolio_holdings.csv","text/csv",
                    )

elif page == "Robustness & Stress Tests":
    st.subheader("Robustness, Bootstrap & Dependency Stress Tests")
    st.caption(
        "Challenges the ranking portfolio across costs, basket breadth, clustered session "
        "resampling and sector exclusions. This module searches for fragility, not the best backtest."
    )
    if fno_uploaded is None:
        st.info("Upload the ALL F&O research CSV in the sidebar.")
    else:
        signals, signal_problems = load_phase3c_signals(fno_uploaded.getvalue())
        if signal_problems:
            st.error(" | ".join(signal_problems))
        else:
            r1,r2,r3,r4 = st.columns(4)
            with r1:
                robust_basket=st.select_slider("Bootstrap basket per side",[5,10,15,20,25],value=20)
            with r2:
                robust_sector_cap=st.select_slider("Sector cap",[1,2,3,4,5],value=3,key="robust_cap")
            with r3:
                robust_cost=st.slider("Cost/slippage (bps)",0,30,10,1,key="robust_cost")
            with r4:
                simulations=st.select_slider("Bootstrap simulations",[250,500,1000,2000],value=1000)
            with st.spinner("Running parameter, bootstrap and sector-dependency stress tests..."):
                daily,_=build_cross_sectional_portfolio(
                    signals,robust_basket,robust_sector_cap,robust_cost
                )
                bootstrap_summary,bootstrap_paths=block_bootstrap(
                    daily,simulations=simulations,block_size=5,seed=42
                )
                stability=parameter_stability(signals,sector_cap=robust_sector_cap)
                sector_stress=leave_one_sector_out(
                    signals,robust_basket,robust_sector_cap,robust_cost
                )
            if bootstrap_summary.empty:
                st.warning("At least 20 complete portfolio sessions are required.")
            else:
                st.markdown("#### Block-bootstrap evidence")
                st.dataframe(bootstrap_summary,use_container_width=True,hide_index=True)
                probability=bootstrap_summary.iloc[0]["Probability_Positive_Expectancy_%"]
                lower=bootstrap_summary.iloc[0]["Expectancy_5th_%"]
                stable_positive=(
                    not stability.empty
                    and stability.groupby("Basket_Size_Per_Side")["Positive_After_Costs"].any().mean() >= 0.8
                )
                sector_positive=(
                    not sector_stress.empty and sector_stress["Positive_After_Exclusion"].mean() >= 0.8
                )
                if probability < 95 or lower <= 0 or not stable_positive or not sector_positive:
                    st.error(
                        "ROBUSTNESS GATE: FAILED. Positive expectancy is not supported across "
                        "resampling, parameter neighborhoods and sector exclusions."
                    )
                else:
                    st.success("ROBUSTNESS GATE: PROVISIONALLY PASSED across all required tests.")
                st.markdown("#### Basket-size and cost stability")
                st.dataframe(stability,use_container_width=True,hide_index=True)
                st.markdown("#### Leave-one-sector-out dependency")
                st.dataframe(sector_stress,use_container_width=True,hide_index=True)
                c1,c2,c3=st.columns(3)
                with c1:
                    st.download_button("Download bootstrap paths",bootstrap_paths.to_csv(index=False).encode(),"phase4_bootstrap_paths.csv","text/csv")
                with c2:
                    st.download_button("Download parameter stability",stability.to_csv(index=False).encode(),"phase4_parameter_stability.csv","text/csv")
                with c3:
                    st.download_button("Download sector stress",sector_stress.to_csv(index=False).encode(),"phase4_sector_dependency.csv","text/csv")

elif page == "Information Coefficient & Decay":
    st.subheader("Information Coefficient & Signal Decay")
    st.caption(
        "Measures cross-sectional Spearman rank correlation between the 11:15 score and "
        "subsequent stock return relative to NIFTY. This tests ranking quality without "
        "forcing arbitrary trade thresholds."
    )
    if fno_uploaded is None:
        st.info("Upload the ALL F&O research CSV in the sidebar.")
    else:
        signals,signal_problems=load_phase3c_signals(fno_uploaded.getvalue())
        if signal_problems:
            st.error(" | ".join(signal_problems))
        else:
            ic_holdout=st.slider("Untouched IC holdout sessions",10,30,20,5)
            daily_ic=daily_information_coefficients(signals)
            summary=information_coefficient_summary(daily_ic,ic_holdout)
            st.markdown("#### Horizon decay and holdout stability")
            st.dataframe(summary,use_container_width=True,hide_index=True)
            holdout_1515=summary[
                summary["Sample"].eq("HOLDOUT") & summary["Horizon"].eq("15:15")
            ]
            if holdout_1515.empty:
                st.warning("Insufficient holdout sessions for the 15:15 IC gate.")
            else:
                row=holdout_1515.iloc[0]
                if row["Mean_Rank_IC"] < 0.03 or row["Positive_IC_Rate_%"] < 55 or row["IC_t_Statistic"] < 2:
                    st.error(
                        "RANKING IC GATE: FAILED. Holdout ranking strength, consistency or "
                        "statistical evidence is below the institutional threshold."
                    )
                else:
                    st.success("RANKING IC GATE: PROVISIONALLY PASSED in the untouched holdout.")
            chart=daily_ic.pivot(index="session_date",columns="Horizon",values="Rolling_20D_IC")
            st.line_chart(chart)
            st.markdown("#### Score-quintile monotonicity")
            horizon=st.selectbox("Quintile outcome horizon",["11:45","13:15","15:15"],index=2)
            quintiles=quintile_spread(signals,horizon)
            quintile_summary=quintiles.groupby("Score_Quintile",as_index=False).agg(
                Mean_Relative_Return=("Mean_Relative_Return_%","mean"),
                Sessions=("session_date","nunique"),
            )
            st.bar_chart(quintile_summary.set_index("Score_Quintile")["Mean_Relative_Return"])
            st.dataframe(quintile_summary,use_container_width=True,hide_index=True)
            st.markdown("#### India VIX regime IC")
            st.dataframe(ic_regime_summary(daily_ic),use_container_width=True,hide_index=True)
            c1,c2=st.columns(2)
            with c1:
                st.download_button("Download daily IC",daily_ic.to_csv(index=False).encode(),"phase5_daily_information_coefficient.csv","text/csv")
            with c2:
                st.download_button("Download quintile study",quintiles.to_csv(index=False).encode(),"phase5_score_quintile_study.csv","text/csv")

elif page == "Futures Confirmation":
    st.subheader("Futures OI + Basis Confirmation Layer")
    st.caption(
        "Tests whether true futures positioning improves RF + RS ranking. Futures confirms or "
        "rejects a stock signal; it does not create direction by itself."
    )
    if fno_uploaded is None or futures_uploaded is None:
        st.info("Upload both the ALL F&O stock research CSV and cumulative Futures Research CSV.")
    else:
        signals, signal_problems = load_phase3c_signals(fno_uploaded.getvalue())
        futures_source = load_csv(futures_uploaded.getvalue())
        futures_problems = validate_futures_snapshots(futures_source)
        problems = signal_problems + futures_problems
        if problems:
            st.error("Dataset rejected: " + " | ".join(problems))
        else:
            futures = prepare_futures_snapshots(futures_source, slot="11:15")
            matched = build_futures_confirmation(signals, futures)
            collected_sessions = futures["session_date"].nunique()
            matched_sessions = matched["session_date"].nunique() if not matched.empty else 0
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Futures sessions", collected_sessions)
            c2.metric("Futures stock-rows", f"{len(futures):,}")
            c3.metric("Matched sessions", matched_sessions)
            c4.metric("Matched stock-rows", f"{len(matched):,}")
            if matched.empty:
                st.warning(
                    "No date overlap exists yet. Continue daily Futures capture and refresh the "
                    "ALL F&O stock research file so both datasets contain the same sessions."
                )
            elif matched_sessions < 20:
                st.warning(
                    f"COLLECTION GATE: {matched_sessions}/20 matched sessions. Data health may be "
                    "reviewed, but expectancy and production conclusions remain locked."
                )
                st.dataframe(
                    matched[["session_date", "Stock", "Signal", "Positioning", "Futures_Alignment",
                             "OI_Change_%", "Basis_%", "Basis_State"]].tail(300),
                    use_container_width=True, hide_index=True,
                )
            else:
                cost_bps = st.slider("Round-trip cost/slippage (bps)", 0, 30, 10, 1, key="futures_cost")
                summary = futures_confirmation_summary(matched, cost_bps)
                st.dataframe(summary, use_container_width=True, hide_index=True)
                confirmed = matched[matched["Futures_Confirmed"]].copy()
                net = confirmed["Oriented_Return_1515_%"].mean() - cost_bps / 100.0 if not confirmed.empty else float("nan")
                if matched_sessions < 60:
                    st.warning(
                        "PRELIMINARY ONLY: 20 sessions unlock descriptive analysis; require at "
                        "least 60 matched sessions before threshold calibration."
                    )
                elif pd.isna(net) or net <= 0:
                    st.error("FUTURES CONFIRMATION GATE: FAILED after estimated costs.")
                else:
                    st.success("Futures confirmation shows provisional positive expectancy after costs.")
                st.download_button(
                    "Download futures-confirmed research dataset",
                    matched.to_csv(index=False).encode(),
                    "phase4_futures_confirmation_dataset.csv",
                    "text/csv",
                )
