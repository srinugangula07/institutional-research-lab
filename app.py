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
from core.vix import enrich_vix_features, validate_market_data, vix_risk_multiplier


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
