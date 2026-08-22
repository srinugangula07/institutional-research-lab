from __future__ import annotations

import numpy as np
import pandas as pd


HORIZONS = {
    "11:45": "return_1145_pct",
    "13:15": "return_1315_pct",
    "15:15": "return_1515_pct",
}

SECTOR_NEUTRAL_FEATURES = [
    "RF_Within_Sector",
    "Stock_RS_Within_Sector",
    "Composite_Within_Sector",
]


def _rank_ic(left, right):
    left_rank = pd.Series(left).rank(method="average")
    right_rank = pd.Series(right).rank(method="average")
    if left_rank.nunique() < 2 or right_rank.nunique() < 2:
        return np.nan
    return left_rank.corr(right_rank)


def build_sector_neutral_dataset(signals, minimum_sector_stocks=3):
    out = signals.copy()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.date
    group_keys = ["session_date", "Sector"]
    out["Sector_Stock_Count"] = out.groupby(group_keys)["Stock"].transform("size")
    out = out[out["Sector_Stock_Count"] >= int(minimum_sector_stocks)].copy()
    out["RF_Within_Sector"] = out.groupby(group_keys)["RF_1115"].rank(pct=True) * 100
    out["Stock_RS_Within_Sector"] = out.groupby(group_keys)["Stock_RS_1115_%"].rank(pct=True) * 100
    out["Composite_Within_Sector"] = out.groupby(group_keys)[
        "Institutional_Directional_Score"
    ].rank(pct=True) * 100
    for horizon, return_column in HORIZONS.items():
        sector_mean = out.groupby(group_keys)[return_column].transform("mean")
        out[f"Sector_Residual_Return_{horizon.replace(':','')}_%"] = (
            pd.to_numeric(out[return_column], errors="coerce") - sector_mean
        )
    return out.sort_values(["session_date", "Sector", "Composite_Within_Sector"])


def sector_neutral_daily_ic(dataset, horizon="15:15"):
    outcome = f"Sector_Residual_Return_{horizon.replace(':','')}_%"
    rows = []
    for session_date, day in dataset.groupby("session_date", sort=True):
        for feature in SECTOR_NEUTRAL_FEATURES:
            pair = day[[feature, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
            rows.append({
                "session_date": session_date,
                "Horizon": horizon,
                "Feature": feature,
                "Rank_IC": _rank_ic(pair[feature], pair[outcome]) if len(pair) >= 20 else np.nan,
                "Stocks": len(pair),
                "Sectors": day["Sector"].nunique(),
            })
    return pd.DataFrame(rows)


def sector_neutral_summary(daily_ic, holdout_sessions=20):
    dates = sorted(daily_ic["session_date"].unique())
    holdout_dates = set(dates[-int(holdout_sessions):])
    rows = []
    for sample, frame in [
        ("ALL", daily_ic),
        ("HOLDOUT", daily_ic[daily_ic["session_date"].isin(holdout_dates)]),
    ]:
        for feature, group in frame.groupby("Feature", sort=False):
            values = group["Rank_IC"].dropna()
            std = values.std(ddof=1)
            rows.append({
                "Sample": sample,
                "Feature": feature,
                "Sessions": len(values),
                "Mean_Rank_IC": values.mean(),
                "Median_Rank_IC": values.median(),
                "Positive_IC_Rate_%": (values > 0).mean() * 100,
                "IC_t_Statistic": values.mean() / (std / np.sqrt(len(values)))
                if pd.notna(std) and std > 0 and len(values) > 1 else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["Sample", "Mean_Rank_IC"], ascending=[True, False])


def sector_residual_attribution(dataset, horizon="15:15"):
    outcome = f"Sector_Residual_Return_{horizon.replace(':','')}_%"
    rows = []
    for sector, group in dataset.groupby("Sector"):
        pair = group[["Composite_Within_Sector", outcome]].dropna()
        rows.append({
            "Sector": sector,
            "Stocks": group["Stock"].nunique(),
            "Sessions": group["session_date"].nunique(),
            "Composite_Residual_IC": _rank_ic(pair.iloc[:, 0], pair.iloc[:, 1])
            if len(pair) >= 30 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("Composite_Residual_IC", ascending=False)
