from __future__ import annotations

import pandas as pd


def build_liquidity_capacity(signals, participation_rate_pct=1.0):
    out = signals.copy()
    out["Morning_Traded_Value_Rs"] = (
        pd.to_numeric(out["Morning_Volume"], errors="coerce")
        * pd.to_numeric(out["entry_close"], errors="coerce")
    )
    out["Estimated_One_Side_Capacity_Rs"] = (
        out["Morning_Traded_Value_Rs"] * float(participation_rate_pct) / 100.0
    )
    out["Liquidity_Percentile"] = out.groupby("session_date")["Morning_Traded_Value_Rs"].rank(pct=True) * 100
    out["Liquidity_Bucket"] = pd.cut(
        out["Liquidity_Percentile"], [0, 20, 50, 80, 100],
        labels=["BOTTOM 20%", "LOWER MID", "UPPER MID", "TOP 20%"], include_lowest=True,
    )
    return out


def liquidity_performance(capacity, cost_bps=10):
    active = capacity[capacity["Signal"].isin(["LONG", "SHORT"])].copy()
    active["Net_Oriented_Return_%"] = active["Oriented_Return_1515_%"] - float(cost_bps) / 100.0
    return active.groupby("Liquidity_Bucket", observed=False, as_index=False).agg(
        Trades=("Stock", "size"),
        Stocks=("Stock", "nunique"),
        Median_Capacity_Rs=("Estimated_One_Side_Capacity_Rs", "median"),
        Win_Rate=("Net_Oriented_Return_%", lambda x: (x > 0).mean() * 100),
        Net_Expectancy=("Net_Oriented_Return_%", "mean"),
    )


def latest_capacity_table(capacity):
    latest = max(capacity["session_date"])
    columns = [
        "session_date", "Stock", "Sector", "Signal", "Institutional_Directional_Score",
        "Morning_Traded_Value_Rs", "Liquidity_Percentile", "Liquidity_Bucket",
        "Estimated_One_Side_Capacity_Rs",
    ]
    return capacity[capacity["session_date"].eq(latest)][columns].sort_values(
        "Estimated_One_Side_Capacity_Rs", ascending=False
    )
