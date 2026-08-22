from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.demo import make_demo_data
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
    uploaded = st.file_uploader("Upload NIFTY + India VIX CSV", type=["csv"])
    use_demo = st.toggle("Use demonstration dataset", value=uploaded is None)
    page = st.radio(
        "Module",
        [
            "Research Overview",
            "Data Health",
            "India VIX Intelligence",
            "VIX–Index Correlation",
            "11:15 Validation",
            "Market Replay",
            "Setup Backtester",
        ],
    )


@st.cache_data(show_spinner=False)
def load_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


if uploaded is not None:
    raw_df = load_csv(uploaded.getvalue())
elif use_demo:
    raw_df = make_demo_data()
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

df = enrich_vix_features(raw_df)
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
    st.success("Phase 1 foundation is operational: schema validation and point-in-time VIX features.")

elif page == "Data Health":
    st.subheader("Dataset health")
    st.success("Required columns, timestamps and positive price values passed validation.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Start", str(df["timestamp"].min()))
    c2.metric("End", str(df["timestamp"].max()))
    c3.metric("Duplicate timestamps", int(df["timestamp"].duplicated().sum()))
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
    st.warning(
        "The demonstration dataset is not restricted to actual NSE sessions. Upload timestamped "
        "market data to perform a valid 11:15 study."
    )
    at_1115 = df[(df["timestamp"].dt.hour == 11) & (df["timestamp"].dt.minute == 15)].copy()
    if at_1115.empty:
        st.info("No 11:15 observations exist in the current dataset.")
    else:
        result = forward_return_study(at_1115, "vix_regime", [1, 2, 4])
        st.dataframe(result, use_container_width=True)

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

