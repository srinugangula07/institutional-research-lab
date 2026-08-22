from __future__ import annotations

import numpy as np
import pandas as pd


def _rotation_factor(morning: pd.DataFrame) -> int:
    high_rotation = np.sign(morning["high"].diff()).fillna(0)
    low_rotation = np.sign(morning["low"].diff()).fillna(0)
    return int((high_rotation + low_rotation).sum())


def build_1115_rf_rs_features(history: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Build RF/RS features using only completed candles available at 11:15."""
    rows = []
    cutoff = pd.to_datetime(history["timestamp"]).dt.time <= pd.Timestamp("10:45").time()
    morning_history = history[cutoff].copy()
    for (stock, session_date), day in morning_history.groupby(["Stock", "session_date"], sort=False):
        day = day.sort_values("timestamp")
        morning = day[day["timestamp"].dt.time.isin([
            pd.Timestamp("09:15").time(), pd.Timestamp("09:45").time(),
            pd.Timestamp("10:15").time(), pd.Timestamp("10:45").time(),
        ])]
        if len(morning) < 4:
            continue
        first = morning.iloc[0]
        last = morning.iloc[-1]
        stock_momentum = (last["close"] / first["close"] - 1) * 100
        nifty_momentum = (last["nifty_close"] / first["nifty_close"] - 1) * 100
        rows.append({
            "Stock": stock,
            "Sector": last["Sector"],
            "session_date": session_date,
            "RF_1115": _rotation_factor(morning),
            "Stock_Momentum_0945_1115_%": stock_momentum,
            "NIFTY_Momentum_0945_1115_%": nifty_momentum,
            "Stock_RS_1115_%": stock_momentum - nifty_momentum,
            "VIX_1115": last["india_vix_close"],
            "Morning_Volume": morning["volume"].sum(),
        })
    features = pd.DataFrame(rows)
    if features.empty:
        return features

    sector = features.groupby(["session_date", "Sector"], as_index=False).agg(
        **{"Sector_RS_1115_%": ("Stock_RS_1115_%", "mean")}
    )
    features = features.merge(sector, on=["session_date", "Sector"], how="left")
    for source, target in (
        ("RF_1115", "RF_Percentile"),
        ("Stock_RS_1115_%", "Stock_RS_Percentile"),
        ("Sector_RS_1115_%", "Sector_RS_Percentile"),
    ):
        features[target] = features.groupby("session_date")[source].rank(pct=True) * 100

    features["Institutional_Directional_Score"] = (
        0.40 * (features["RF_Percentile"] * 2 - 100)
        + 0.35 * (features["Stock_RS_Percentile"] * 2 - 100)
        + 0.25 * (features["Sector_RS_Percentile"] * 2 - 100)
    ).round(1)

    daily_vix = features.groupby("session_date", as_index=False)["VIX_1115"].first()
    daily_vix["VIX_60D_Percentile"] = daily_vix["VIX_1115"].rolling(60, min_periods=20).apply(
        lambda values: pd.Series(values).rank(pct=True).iloc[-1] * 100,
        raw=False,
    )
    features = features.merge(daily_vix, on=["session_date", "VIX_1115"], how="left")
    features["VIX_Risk_Multiplier"] = np.select(
        [
            features["VIX_60D_Percentile"] > 80,
            features["VIX_60D_Percentile"] > 60,
            features["VIX_60D_Percentile"] > 40,
        ],
        [0.60, 0.75, 0.90],
        default=1.00,
    )
    features["Signal"] = np.select(
        [
            features["Institutional_Directional_Score"] >= 40,
            features["Institutional_Directional_Score"] <= -40,
        ],
        ["LONG", "SHORT"],
        default="NEUTRAL",
    )
    result = features.merge(outcomes, on=["Stock", "Sector", "session_date"], how="inner")
    result["Oriented_Return_1515_%"] = np.select(
        [result["Signal"] == "LONG", result["Signal"] == "SHORT"],
        [result["return_1515_pct"], -result["return_1515_pct"]],
        default=np.nan,
    )
    result["Risk_Adjusted_Oriented_Return_%"] = (
        result["Oriented_Return_1515_%"] * result["VIX_Risk_Multiplier"]
    )
    return result.sort_values(["session_date", "Institutional_Directional_Score"], ascending=[True, False])


def signal_backtest_summary(signals: pd.DataFrame) -> pd.DataFrame:
    active = signals[signals["Signal"].isin(["LONG", "SHORT"])].dropna(
        subset=["Oriented_Return_1515_%"]
    )
    rows = []
    for signal, group in active.groupby("Signal"):
        returns = group["Oriented_Return_1515_%"]
        rows.append({
            "Signal": signal,
            "Trades": len(group),
            "Sessions": group["session_date"].nunique(),
            "Win_Rate_%": (returns > 0).mean() * 100,
            "Average_Return_%": returns.mean(),
            "Median_Return_%": returns.median(),
            "Average_MFE_%": group["mfe_high_pct"].mean() if signal == "LONG" else -group["mae_low_pct"].mean(),
            "Average_MAE_%": group["mae_low_pct"].mean() if signal == "LONG" else -group["mfe_high_pct"].mean(),
            "Average_Risk_Adjusted_Return_%": group["Risk_Adjusted_Oriented_Return_%"].mean(),
        })
    return pd.DataFrame(rows)
