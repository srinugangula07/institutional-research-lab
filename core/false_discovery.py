from __future__ import annotations

import numpy as np
import pandas as pd


def _rank_ic(left, right):
    left_rank = pd.Series(left).rank(method="average")
    right_rank = pd.Series(right).rank(method="average")
    if left_rank.nunique() < 2 or right_rank.nunique() < 2:
        return np.nan
    return left_rank.corr(right_rank)


def candidate_daily_ic(signals, auction, sector_neutral, horizon="15:15"):
    outcome = f"relative_return_{horizon.replace(':','')}_pct"
    base_columns = [
        "Stock", "session_date", outcome, "RF_Percentile", "Stock_RS_Percentile",
        "Sector_RS_Percentile", "Institutional_Directional_Score",
    ]
    source = signals[base_columns].copy()
    auction_keep = [
        "Stock", "session_date", "Gap_%", "IB_Location", "Morning_Return_%",
        "Range_Extension_Score_%", "Volume_Confirmed_Direction", "Auction_Score",
    ]
    source = source.merge(auction[auction_keep], on=["Stock", "session_date"], how="left")
    residual_outcome = f"Sector_Residual_Return_{horizon.replace(':','')}_%"
    residual_keep = [
        "Stock", "session_date", "RF_Within_Sector", "Stock_RS_Within_Sector",
        "Composite_Within_Sector", residual_outcome,
    ]
    source = source.merge(sector_neutral[residual_keep], on=["Stock", "session_date"], how="left")
    candidates = {
        "RF": ("RF_Percentile", outcome),
        "Stock RS": ("Stock_RS_Percentile", outcome),
        "Sector RS": ("Sector_RS_Percentile", outcome),
        "RF+RS Composite": ("Institutional_Directional_Score", outcome),
        "Gap": ("Gap_%", outcome),
        "IB Location": ("IB_Location", outcome),
        "Morning Return": ("Morning_Return_%", outcome),
        "Range Extension": ("Range_Extension_Score_%", outcome),
        "Volume Direction": ("Volume_Confirmed_Direction", outcome),
        "Auction Composite": ("Auction_Score", outcome),
        "RF Within Sector": ("RF_Within_Sector", residual_outcome),
        "Stock RS Within Sector": ("Stock_RS_Within_Sector", residual_outcome),
        "Composite Within Sector": ("Composite_Within_Sector", residual_outcome),
    }
    rows = []
    for session_date, day in source.groupby("session_date", sort=True):
        for name, (feature, target) in candidates.items():
            pair = day[[feature, target]].apply(pd.to_numeric, errors="coerce").dropna()
            rows.append({
                "session_date": session_date, "Candidate": name,
                "Rank_IC": _rank_ic(pair[feature], pair[target]) if len(pair) >= 20 else np.nan,
            })
    return pd.DataFrame(rows)


def false_discovery_control(daily_ic, permutations=2000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for candidate, group in daily_ic.groupby("Candidate"):
        values = group["Rank_IC"].dropna().to_numpy()
        observed = values.mean()
        signs = rng.choice([-1.0, 1.0], size=(int(permutations), len(values)))
        null_means = (signs * values).mean(axis=1)
        p_value = (1 + (np.abs(null_means) >= abs(observed)).sum()) / (int(permutations) + 1)
        rows.append({
            "Candidate": candidate, "Sessions": len(values), "Mean_Rank_IC": observed,
            "Positive_IC_Rate_%": (values > 0).mean() * 100, "Raw_P_Value": p_value,
        })
    result = pd.DataFrame(rows).sort_values("Raw_P_Value").reset_index(drop=True)
    m = len(result)
    adjusted = result["Raw_P_Value"] * m / (result.index + 1)
    result["BH_FDR_Q_Value"] = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    result["Survives_5pct_FDR"] = result["BH_FDR_Q_Value"] <= 0.05
    return result
