from __future__ import annotations

import numpy as np
import pandas as pd


def _select_sector_capped(frame, ascending, basket_size, max_per_sector):
    ranked = frame.sort_values(
        ["Institutional_Directional_Score", "Stock"],
        ascending=[ascending, True],
    )
    selected = []
    sector_counts = {}
    for row in ranked.itertuples(index=False):
        sector = str(getattr(row, "Sector", "UNKNOWN"))
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected.append(row)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) >= basket_size:
            break
    return pd.DataFrame(selected, columns=ranked.columns)


def build_cross_sectional_portfolio(
    signals,
    basket_size=20,
    max_per_sector=3,
    cost_bps=10,
):
    """Build daily market-neutral top/bottom baskets from point-in-time scores."""
    source = signals.copy()
    source["session_date"] = pd.to_datetime(source["session_date"]).dt.date
    source["Institutional_Directional_Score"] = pd.to_numeric(
        source["Institutional_Directional_Score"], errors="coerce"
    )
    source["return_1515_pct"] = pd.to_numeric(source["return_1515_pct"], errors="coerce")
    source = source.dropna(subset=["Institutional_Directional_Score", "return_1515_pct"])
    daily_rows = []
    holdings = []
    previous_long, previous_short = set(), set()
    cost_pct = float(cost_bps) / 100.0

    for session_date, day in source.groupby("session_date", sort=True):
        long_book = _select_sector_capped(day, False, basket_size, max_per_sector)
        short_book = _select_sector_capped(day, True, basket_size, max_per_sector)
        if len(long_book) < basket_size or len(short_book) < basket_size:
            continue
        long_names = set(long_book["Stock"])
        short_names = set(short_book["Stock"])
        long_return = long_book["return_1515_pct"].mean()
        short_return = -short_book["return_1515_pct"].mean()
        gross = 0.5 * long_return + 0.5 * short_return
        net = gross - cost_pct
        long_churn = 1.0 if not previous_long else 1 - len(long_names & previous_long) / basket_size
        short_churn = 1.0 if not previous_short else 1 - len(short_names & previous_short) / basket_size
        membership_churn = 0.5 * (long_churn + short_churn)
        vix_pct = pd.to_numeric(day["VIX_60D_Percentile"], errors="coerce").median()
        regime = "UNAVAILABLE"
        if pd.notna(vix_pct):
            regime = "LOW" if vix_pct < 33 else "HIGH" if vix_pct >= 67 else "NORMAL"
        daily_rows.append({
            "session_date": session_date,
            "Long_Return_%": long_return,
            "Short_Return_%": short_return,
            "Gross_Long_Short_%": gross,
            "Cost_%": cost_pct,
            "Net_Long_Short_%": net,
            "Membership_Churn_%": membership_churn * 100,
            "VIX_Regime": regime,
            "Long_Count": len(long_book),
            "Short_Count": len(short_book),
        })
        for side, book in (("LONG", long_book), ("SHORT", short_book)):
            oriented = book["return_1515_pct"] if side == "LONG" else -book["return_1515_pct"]
            for (_, row), ret in zip(book.iterrows(), oriented):
                holdings.append({
                    "session_date": session_date,
                    "Side": side,
                    "Stock": row["Stock"],
                    "Sector": row.get("Sector", "UNKNOWN"),
                    "Score": row["Institutional_Directional_Score"],
                    "Stock_Return_%": row["return_1515_pct"],
                    "Oriented_Return_%": ret,
                })
        previous_long, previous_short = long_names, short_names

    daily = pd.DataFrame(daily_rows)
    if not daily.empty:
        daily["Equity_Index"] = (1 + daily["Net_Long_Short_%"] / 100).cumprod() * 100
        daily["Equity_Peak"] = daily["Equity_Index"].cummax()
        daily["Drawdown_%"] = (daily["Equity_Index"] / daily["Equity_Peak"] - 1) * 100
    return daily, pd.DataFrame(holdings)


def portfolio_summary(daily, holdout_sessions=20):
    if daily is None or daily.empty:
        return pd.DataFrame()
    ordered = daily.sort_values("session_date").copy()
    split = max(len(ordered) - int(holdout_sessions), 0)
    ordered["Sample"] = np.where(np.arange(len(ordered)) < split, "DEVELOPMENT", "HOLDOUT")
    rows = []
    for sample, frame in [("ALL", ordered)] + list(ordered.groupby("Sample", sort=False)):
        returns = frame["Net_Long_Short_%"]
        std = returns.std(ddof=1)
        sharpe = returns.mean() / std * np.sqrt(252) if pd.notna(std) and std > 0 else np.nan
        equity = (1 + returns / 100).cumprod() * 100
        drawdown = (equity / equity.cummax() - 1) * 100
        rows.append({
            "Sample": sample,
            "Sessions": len(frame),
            "Net_Win_Rate_%": (returns > 0).mean() * 100,
            "Net_Expectancy_%": returns.mean(),
            "Annualised_Sharpe": sharpe,
            "Cumulative_Return_%": equity.iloc[-1] - 100,
            "Maximum_Drawdown_%": drawdown.min(),
            "Average_Membership_Churn_%": frame["Membership_Churn_%"].mean(),
        })
    return pd.DataFrame(rows)


def regime_summary(daily):
    if daily is None or daily.empty:
        return pd.DataFrame()
    return daily.groupby("VIX_Regime", as_index=False).agg(
        Sessions=("session_date", "size"),
        Win_Rate=("Net_Long_Short_%", lambda x: (x > 0).mean() * 100),
        Net_Expectancy=("Net_Long_Short_%", "mean"),
        Worst_Day=("Net_Long_Short_%", "min"),
        Best_Day=("Net_Long_Short_%", "max"),
    ).sort_values("Net_Expectancy", ascending=False)
