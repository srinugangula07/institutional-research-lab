from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd


ALIASES = {
    "timestamp": ["timestamp", "datetime", "date_time", "date", "time"],
    "nifty_close": ["nifty_close", "nifty", "nifty50", "nifty_50", "index_close"],
    "india_vix_close": ["india_vix_close", "india_vix", "indiavix", "vix", "vix_close"],
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common dashboard export column names to the canonical research schema."""
    out = df.copy()
    lookup = {str(column).strip().lower(): column for column in out.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                rename[lookup[alias]] = canonical
                break
    return out.rename(columns=rename)


def merge_history(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    """Append incremental history with latest-upload-wins timestamp deduplication."""
    frames = [normalise_columns(incoming)]
    if existing is not None and not existing.empty:
        frames.insert(0, normalise_columns(existing))
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    value_columns = [column for column in out.columns if column != "timestamp"]
    out = out.groupby("timestamp", as_index=False)[value_columns].agg(
        lambda values: values.dropna().iloc[-1] if not values.dropna().empty else np.nan
    )
    return out.sort_values("timestamp").reset_index(drop=True)


def restrict_nse_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep weekday observations between 09:15 and 15:30 India-market time."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    valid_day = out["timestamp"].dt.dayofweek < 5
    clock = out["timestamp"].dt.time
    valid_time = clock.between(time(9, 15), time(15, 30))
    out = out[valid_day & valid_time].copy()
    out["session_date"] = out["timestamp"].dt.date
    return out.reset_index(drop=True)


def session_quality(df: pd.DataFrame, expected_bars: int | None = None) -> pd.DataFrame:
    grouped = df.groupby("session_date").agg(
        bars=("timestamp", "size"),
        first_bar=("timestamp", "min"),
        last_bar=("timestamp", "max"),
        missing_nifty=("nifty_close", lambda x: int(x.isna().sum())),
        missing_vix=("india_vix_close", lambda x: int(x.isna().sum())),
    )
    grouped = grouped.reset_index()
    if expected_bars is None and not grouped.empty:
        expected_bars = int(grouped["bars"].mode().iloc[0])
    grouped["expected_bars"] = expected_bars or 0
    grouped["complete"] = (
        (grouped["bars"] >= grouped["expected_bars"])
        & (grouped["missing_nifty"] == 0)
        & (grouped["missing_vix"] == 0)
    )
    return grouped


def _first_at_or_after(day: pd.DataFrame, target: time) -> pd.Series | None:
    candidates = day[day["timestamp"].dt.time >= target]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def build_1115_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Create one point-in-time 11:15 observation per session and same-day outcomes."""
    rows: list[dict] = []
    for session_date, day in df.groupby("session_date", sort=True):
        day = day.sort_values("timestamp")
        entry_rows = day[
            (day["timestamp"].dt.hour == 11) & (day["timestamp"].dt.minute == 15)
        ]
        if entry_rows.empty:
            continue
        entry = entry_rows.iloc[0]
        entry_price = entry["nifty_close"]
        if pd.isna(entry_price) or entry_price <= 0:
            continue
        record = {
            "session_date": session_date,
            "entry_timestamp": entry["timestamp"],
            "entry_nifty": entry_price,
            "entry_vix": entry["india_vix_close"],
            "vix_regime": entry.get("vix_regime"),
            "price_vix_state": entry.get("price_vix_state"),
            "vix_percentile_60": entry.get("vix_percentile_60"),
            "vix_change_zscore_20": entry.get("vix_change_zscore_20"),
        }
        for label, target in (("1200", time(12, 0)), ("1330", time(13, 30)), ("1515", time(15, 15))):
            target_row = _first_at_or_after(day, target)
            record[f"return_{label}_pct"] = (
                (target_row["nifty_close"] / entry_price - 1) * 100
                if target_row is not None and pd.notna(target_row["nifty_close"])
                else np.nan
            )
        after_entry = day[day["timestamp"] >= entry["timestamp"]]
        path = after_entry["nifty_close"].dropna()
        record["mfe_close_pct"] = (path.max() / entry_price - 1) * 100 if not path.empty else np.nan
        record["mae_close_pct"] = (path.min() / entry_price - 1) * 100 if not path.empty else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def outcome_summary(outcomes: pd.DataFrame, group_column: str = "vix_regime") -> pd.DataFrame:
    if outcomes.empty or group_column not in outcomes.columns:
        return pd.DataFrame()
    metrics = []
    for group, data in outcomes.groupby(group_column, dropna=False):
        row = {"Regime": group, "Sessions": len(data)}
        for label in ("1200", "1330", "1515"):
            values = data[f"return_{label}_pct"].dropna()
            row[f"Avg {label} %"] = values.mean()
            row[f"Win {label} %"] = (values > 0).mean() * 100 if len(values) else np.nan
        row["Avg MFE %"] = data["mfe_close_pct"].mean()
        row["Avg MAE %"] = data["mae_close_pct"].mean()
        metrics.append(row)
    return pd.DataFrame(metrics)
