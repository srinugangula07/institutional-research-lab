from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd


FEATURES = [
    "Gap_%", "IB_Location", "Morning_Return_%", "Range_Extension_Score_%",
    "Volume_Confirmed_Direction", "Auction_Score",
]

HORIZONS = {
    "11:45": "relative_return_1145_pct",
    "13:15": "relative_return_1315_pct",
    "15:15": "relative_return_1515_pct",
}


def _rank_ic(left, right):
    left_rank = pd.Series(left).rank(method="average")
    right_rank = pd.Series(right).rank(method="average")
    if left_rank.nunique() < 2 or right_rank.nunique() < 2:
        return np.nan
    return left_rank.corr(right_rank)


def build_morning_auction_features(history, outcomes):
    source = history.copy().sort_values(["Stock", "timestamp"])
    source["session_date"] = pd.to_datetime(source["timestamp"]).dt.date
    session_close = source.groupby(["Stock", "session_date"], as_index=False).agg(
        Session_Close=("close", "last")
    )
    session_close["Previous_Close"] = session_close.groupby("Stock")["Session_Close"].shift(1)
    rows = []
    required_times = {time(9, 15), time(9, 45), time(10, 15), time(10, 45)}
    for (stock, session_date), day in source.groupby(["Stock", "session_date"], sort=False):
        morning = day[day["timestamp"].dt.time.isin(required_times)].sort_values("timestamp")
        if len(morning) != 4:
            continue
        open_price = float(morning.iloc[0]["open"])
        gate_close = float(morning.iloc[-1]["close"])
        ib = morning.iloc[:2]
        ib_high = float(ib["high"].max())
        ib_low = float(ib["low"].min())
        ib_range = ib_high - ib_low
        morning_high = float(morning["high"].max())
        morning_low = float(morning["low"].min())
        rows.append({
            "Stock": stock,
            "Sector": morning.iloc[-1]["Sector"],
            "session_date": session_date,
            "Open": open_price,
            "IB_High": ib_high,
            "IB_Low": ib_low,
            "Gate_Close": gate_close,
            "Morning_Volume": float(morning["volume"].sum()),
            "IB_Range_%": ib_range / open_price * 100 if open_price else np.nan,
            "IB_Location": (gate_close - ib_low) / ib_range if ib_range > 0 else 0.5,
            "Morning_Return_%": (gate_close / open_price - 1) * 100 if open_price else np.nan,
            "Upper_Extension_%": max(morning_high - ib_high, 0) / open_price * 100 if open_price else np.nan,
            "Lower_Extension_%": max(ib_low - morning_low, 0) / open_price * 100 if open_price else np.nan,
            "IB_State": "ABOVE IB" if gate_close > ib_high else "BELOW IB" if gate_close < ib_low else "INSIDE IB",
        })
    features = pd.DataFrame(rows)
    if features.empty:
        return features
    features = features.merge(
        session_close[["Stock", "session_date", "Previous_Close"]],
        on=["Stock", "session_date"], how="left",
    )
    features["Gap_%"] = (features["Open"] / features["Previous_Close"] - 1) * 100
    features["Range_Extension_Score_%"] = features["Upper_Extension_%"] - features["Lower_Extension_%"]
    features = features.sort_values(["Stock", "session_date"])
    features["Prior_20D_Median_Morning_Volume"] = features.groupby("Stock")["Morning_Volume"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=10).median()
    )
    features["Morning_Volume_Ratio"] = (
        features["Morning_Volume"] / features["Prior_20D_Median_Morning_Volume"]
    )
    direction = np.sign(features["Morning_Return_%"])
    features["Volume_Confirmed_Direction"] = direction * np.log1p(
        features["Morning_Volume_Ratio"].clip(lower=0)
    )
    percentile_inputs = {
        "Gap_%": 0.15,
        "IB_Location": 0.30,
        "Morning_Return_%": 0.25,
        "Range_Extension_Score_%": 0.15,
        "Volume_Confirmed_Direction": 0.15,
    }
    score = pd.Series(0.0, index=features.index)
    for column, weight in percentile_inputs.items():
        percentile = features.groupby("session_date")[column].rank(pct=True) * 100
        score = score + weight * percentile.fillna(50)
    features["Auction_Score"] = score
    return features.merge(
        outcomes,
        on=["Stock", "Sector", "session_date"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["session_date", "Auction_Score"], ascending=[True, False])


def auction_feature_ic(feature_data, horizon="15:15"):
    outcome = HORIZONS[horizon]
    rows = []
    for session_date, day in feature_data.groupby("session_date", sort=True):
        for feature in FEATURES:
            pair = day[[feature, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
            rows.append({
                "session_date": session_date,
                "Horizon": horizon,
                "Feature": feature,
                "Rank_IC": _rank_ic(pair[feature], pair[outcome]) if len(pair) >= 20 else np.nan,
                "Stocks": len(pair),
            })
    return pd.DataFrame(rows)


def auction_ic_summary(daily_ic, holdout_sessions=20):
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
