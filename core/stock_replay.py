from __future__ import annotations

import numpy as np
import pandas as pd


OUTCOME_COLUMNS = {
    "11:45": ("return_1145_pct", "relative_return_1145_pct"),
    "13:15": ("return_1315_pct", "relative_return_1315_pct"),
    "15:15": ("return_1515_pct", "relative_return_1515_pct"),
}


def build_stock_replay_dataset(signals, auction_features):
    signal_columns = [
        "Stock", "Sector", "session_date", "RF_1115", "RF_Percentile",
        "Stock_RS_1115_%", "Stock_RS_Percentile", "Sector_RS_1115_%",
        "Sector_RS_Percentile", "Institutional_Directional_Score", "Signal",
        "VIX_1115", "VIX_60D_Percentile", "VIX_Risk_Multiplier",
        "entry_close", "return_1145_pct", "relative_return_1145_pct",
        "return_1315_pct", "relative_return_1315_pct", "return_1515_pct",
        "relative_return_1515_pct", "mfe_high_pct", "mae_low_pct",
    ]
    auction_columns = [
        "Stock", "session_date", "Gap_%", "IB_Range_%", "IB_Location",
        "IB_State", "Morning_Return_%", "Morning_Volume_Ratio", "Auction_Score",
    ]
    left = signals[signal_columns].copy()
    right = auction_features[auction_columns].copy()
    left["session_date"] = pd.to_datetime(left["session_date"]).dt.date
    right["session_date"] = pd.to_datetime(right["session_date"]).dt.date
    result = left.merge(right, on=["Stock", "session_date"], how="inner", validate="one_to_one")
    result["PreOutcome_Alignment"] = np.select(
        [
            (result["Institutional_Directional_Score"] >= 40) & (result["Auction_Score"] >= 70),
            (result["Institutional_Directional_Score"] <= -40) & (result["Auction_Score"] <= 30),
            (result["Institutional_Directional_Score"] >= 40) & (result["Auction_Score"] <= 30),
            (result["Institutional_Directional_Score"] <= -40) & (result["Auction_Score"] >= 70),
        ],
        ["BULLISH ALIGNMENT", "BEARISH ALIGNMENT", "BULLISH CONFLICT", "BEARISH CONFLICT"],
        default="NEUTRAL",
    )
    return result.sort_values(
        ["session_date", "Institutional_Directional_Score"], ascending=[True, False]
    ).reset_index(drop=True)


def replay_session(replay_data, session_date, checkpoint="15:15", reveal_outcome=False):
    selected_date = pd.Timestamp(session_date).date()
    day = replay_data[replay_data["session_date"].eq(selected_date)].copy()
    if day.empty:
        return day
    pre_outcome = [
        "Stock", "Sector", "entry_close", "VIX_1115", "VIX_60D_Percentile",
        "RF_1115", "RF_Percentile", "Stock_RS_1115_%", "Stock_RS_Percentile",
        "Sector_RS_1115_%", "Sector_RS_Percentile", "Institutional_Directional_Score",
        "Signal", "Gap_%", "IB_Range_%", "IB_Location", "IB_State",
        "Morning_Return_%", "Morning_Volume_Ratio", "Auction_Score", "PreOutcome_Alignment",
    ]
    if not reveal_outcome:
        return day[pre_outcome].sort_values("Institutional_Directional_Score", ascending=False)
    absolute, relative = OUTCOME_COLUMNS[checkpoint]
    revealed = day[pre_outcome + [absolute, relative, "mfe_high_pct", "mae_low_pct"]].copy()
    revealed["Outcome_Direction"] = np.select(
        [revealed[relative] > 0, revealed[relative] < 0],
        ["OUTPERFORMED NIFTY", "UNDERPERFORMED NIFTY"], default="FLAT",
    )
    return revealed.sort_values("Institutional_Directional_Score", ascending=False)


def replay_sector_breadth(replay_data, session_date):
    selected_date = pd.Timestamp(session_date).date()
    day = replay_data[replay_data["session_date"].eq(selected_date)].copy()
    if day.empty:
        return pd.DataFrame()
    rows = []
    for sector, group in day.groupby("Sector"):
        rows.append({
            "Sector": sector,
            "Stocks": len(group),
            "Bullish_RF_RS": int((group["Institutional_Directional_Score"] >= 40).sum()),
            "Bearish_RF_RS": int((group["Institutional_Directional_Score"] <= -40).sum()),
            "Above_IB": int(group["IB_State"].eq("ABOVE IB").sum()),
            "Below_IB": int(group["IB_State"].eq("BELOW IB").sum()),
            "Mean_RF_RS_Score": group["Institutional_Directional_Score"].mean(),
            "Mean_Auction_Score": group["Auction_Score"].mean(),
        })
    return pd.DataFrame(rows).sort_values("Mean_RF_RS_Score", ascending=False)
