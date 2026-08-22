from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd


REQUIRED_FNO_COLUMNS = {
    "timestamp", "Stock", "Sector", "open", "high", "low", "close", "volume",
    "nifty_close", "india_vix_close",
}


def _population_beta(asset: pd.Series, factor: pd.Series) -> float:
    variance = factor.var(ddof=0)
    if not np.isfinite(variance) or variance <= 0:
        return np.nan
    covariance = ((asset - asset.mean()) * (factor - factor.mean())).mean()
    return covariance / variance


def validate_fno_history(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    missing = REQUIRED_FNO_COLUMNS.difference(df.columns)
    if missing:
        return ["Missing required columns: " + ", ".join(sorted(missing))]
    if df.empty:
        return ["F&O dataset contains no rows."]

    timestamp = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamp.isna().any():
        problems.append(f"Invalid timestamps: {int(timestamp.isna().sum())}")
    duplicates = df.assign(_timestamp=timestamp).duplicated(["Stock", "_timestamp"]).sum()
    if duplicates:
        problems.append(f"Duplicate Stock+timestamp rows: {int(duplicates)}")

    numeric = ["open", "high", "low", "close", "volume", "nifty_close", "india_vix_close"]
    converted = {column: pd.to_numeric(df[column], errors="coerce") for column in numeric}
    for column, values in converted.items():
        if values.isna().any():
            problems.append(f"Invalid or missing {column}: {int(values.isna().sum())}")
    if (converted["close"] <= 0).any():
        problems.append(f"Non-positive close values: {int((converted['close'] <= 0).sum())}")
    invalid_ohlc = (
        (converted["high"] < pd.concat([converted["open"], converted["close"], converted["low"]], axis=1).max(axis=1))
        | (converted["low"] > pd.concat([converted["open"], converted["close"], converted["high"]], axis=1).min(axis=1))
    )
    if invalid_ohlc.any():
        problems.append(f"Invalid OHLC relationships: {int(invalid_ohlc.sum())}")
    return problems


def prepare_fno_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["Stock"] = out["Stock"].astype(str).str.upper().str.strip()
    out["Sector"] = out["Sector"].fillna("UNKNOWN").astype(str).str.strip()
    for column in ("open", "high", "low", "close", "volume", "nifty_close", "india_vix_close"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["timestamp", "Stock", "close", "nifty_close", "india_vix_close"])
    out = out.drop_duplicates(["Stock", "timestamp"], keep="last")
    out = out.sort_values(["Stock", "timestamp"]).reset_index(drop=True)
    out["session_date"] = out["timestamp"].dt.date
    return out


def fno_quality_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_quality = df.groupby(["Stock", "Sector"], as_index=False).agg(
        Rows=("timestamp", "size"),
        Sessions=("session_date", "nunique"),
        First=("timestamp", "min"),
        Last=("timestamp", "max"),
        Missing_Volume=("volume", lambda x: int(x.isna().sum())),
    )
    session_quality = df.groupby("session_date", as_index=False).agg(
        Stocks=("Stock", "nunique"),
        Rows=("timestamp", "size"),
        First_Bar=("timestamp", "min"),
        Last_Bar=("timestamp", "max"),
    )
    return stock_quality, session_quality


def _row_at_start(day: pd.DataFrame, bar_start: time) -> pd.Series | None:
    rows = day[day["timestamp"].dt.time == bar_start]
    return None if rows.empty else rows.iloc[0]


def build_stock_1115_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Use the 10:45 bar close, known at 11:15, to prevent look-ahead bias."""
    rows: list[dict] = []
    for (stock, session_date), day in df.groupby(["Stock", "session_date"], sort=False):
        day = day.sort_values("timestamp")
        decision_bar = _row_at_start(day, time(10, 45))
        if decision_bar is None or decision_bar["close"] <= 0:
            continue
        entry = float(decision_bar["close"])
        nifty_entry = float(decision_bar["nifty_close"])
        vix_entry = float(decision_bar["india_vix_close"])
        record = {
            "Stock": stock,
            "Sector": decision_bar["Sector"],
            "session_date": session_date,
            "decision_time": "11:15",
            "source_bar_start": decision_bar["timestamp"],
            "entry_close": entry,
            "entry_nifty": nifty_entry,
            "entry_vix": vix_entry,
        }

        checkpoints = {
            "1145": time(11, 15),
            "1315": time(12, 45),
            "1515": time(14, 45),
        }
        for label, bar_start in checkpoints.items():
            target = _row_at_start(day, bar_start)
            if target is None:
                record[f"return_{label}_pct"] = np.nan
                record[f"nifty_return_{label}_pct"] = np.nan
                record[f"relative_return_{label}_pct"] = np.nan
                if label == "1515":
                    record["vix_change_1515_pct"] = np.nan
                continue
            stock_return = (float(target["close"]) / entry - 1) * 100
            nifty_return = (float(target["nifty_close"]) / nifty_entry - 1) * 100
            record[f"return_{label}_pct"] = stock_return
            record[f"nifty_return_{label}_pct"] = nifty_return
            record[f"relative_return_{label}_pct"] = stock_return - nifty_return
            if label == "1515":
                record["vix_change_1515_pct"] = (
                    float(target["india_vix_close"]) / vix_entry - 1
                ) * 100

        future = day[day["timestamp"] >= pd.Timestamp.combine(pd.Timestamp(session_date), time(11, 15))]
        record["mfe_high_pct"] = (future["high"].max() / entry - 1) * 100 if not future.empty else np.nan
        record["mae_low_pct"] = (future["low"].min() / entry - 1) * 100 if not future.empty else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def stock_vix_sensitivity(outcomes: pd.DataFrame, minimum_sessions: int = 30) -> pd.DataFrame:
    rows = []
    clean = outcomes.dropna(subset=["return_1515_pct", "vix_change_1515_pct"])
    for (stock, sector), group in clean.groupby(["Stock", "Sector"]):
        if len(group) < minimum_sessions:
            continue
        stock_return = group["return_1515_pct"]
        relative = group["relative_return_1515_pct"]
        vix_change = group["vix_change_1515_pct"]
        beta = _population_beta(stock_return, vix_change)
        up_vix = group[vix_change > 0]
        down_vix = group[vix_change < 0]
        rows.append({
            "Stock": stock,
            "Sector": sector,
            "Sessions": len(group),
            "Avg_Return_1515_%": stock_return.mean(),
            "Positive_Close_%": (stock_return > 0).mean() * 100,
            "Avg_Relative_Return_%": relative.mean(),
            "Avg_Absolute_Move_%": stock_return.abs().mean(),
            "Avg_MFE_%": group["mfe_high_pct"].mean(),
            "Avg_MAE_%": group["mae_low_pct"].mean(),
            "VIX_Beta": beta,
            "VIX_Return_Correlation": stock_return.corr(vix_change),
            "Avg_Return_When_VIX_Rises_%": up_vix["return_1515_pct"].mean(),
            "Avg_Return_When_VIX_Falls_%": down_vix["return_1515_pct"].mean(),
            "Downside_Hit_When_VIX_Rises_%": (up_vix["return_1515_pct"] < 0).mean() * 100 if len(up_vix) else np.nan,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["VIX_Defensive_Score"] = (
        result["Avg_Return_When_VIX_Rises_%"].rank(pct=True) * 50
        + result["VIX_Beta"].rank(pct=True) * 30
        + result["Avg_Relative_Return_%"].rank(pct=True) * 20
    ).round(1)
    result["VIX_Risk_Score"] = (
        (-result["Avg_Return_When_VIX_Rises_%"]).rank(pct=True) * 45
        + (-result["VIX_Beta"]).rank(pct=True) * 35
        + result["Avg_Absolute_Move_%"].rank(pct=True) * 20
    ).round(1)
    return result.sort_values("VIX_Defensive_Score", ascending=False).reset_index(drop=True)


def sector_vix_sensitivity(outcomes: pd.DataFrame, minimum_sessions: int = 30) -> pd.DataFrame:
    clean = outcomes.dropna(subset=["return_1515_pct", "vix_change_1515_pct"])
    rows = []
    for sector, group in clean.groupby("Sector"):
        daily = group.groupby("session_date", as_index=False).agg(
            sector_return=("return_1515_pct", "mean"),
            relative_return=("relative_return_1515_pct", "mean"),
            vix_change=("vix_change_1515_pct", "first"),
        )
        if len(daily) < minimum_sessions:
            continue
        rows.append({
            "Sector": sector,
            "Stocks": group["Stock"].nunique(),
            "Sessions": len(daily),
            "Avg_Return_1515_%": daily["sector_return"].mean(),
            "Avg_Relative_Return_%": daily["relative_return"].mean(),
            "VIX_Beta": _population_beta(daily["sector_return"], daily["vix_change"]),
            "VIX_Return_Correlation": daily["sector_return"].corr(daily["vix_change"]),
            "Avg_When_VIX_Rises_%": daily.loc[daily["vix_change"] > 0, "sector_return"].mean(),
            "Avg_When_VIX_Falls_%": daily.loc[daily["vix_change"] < 0, "sector_return"].mean(),
        })
    result = pd.DataFrame(rows)
    return result.sort_values("VIX_Beta", ascending=False).reset_index(drop=True) if not result.empty else result
