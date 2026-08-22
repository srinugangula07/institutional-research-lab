from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "nifty_close", "india_vix_close"}


def validate_market_data(df: pd.DataFrame) -> list[str]:
    """Return human-readable data-quality problems; an empty list means usable."""
    problems: list[str] = []
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        problems.append(f"Missing required columns: {', '.join(sorted(missing))}")
        return problems

    if df.empty:
        problems.append("Dataset contains no rows.")
        return problems

    parsed = pd.to_datetime(df["timestamp"], errors="coerce")
    if parsed.isna().any():
        problems.append(f"Invalid timestamps: {int(parsed.isna().sum())}")
    if parsed.duplicated().any():
        problems.append(f"Duplicate timestamps: {int(parsed.duplicated().sum())}")

    for column in ("nifty_close", "india_vix_close"):
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            problems.append(f"Invalid or missing {column}: {int(values.isna().sum())}")
        if (values <= 0).any():
            problems.append(f"Non-positive {column}: {int((values <= 0).sum())}")
    return problems


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(10, window // 4)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
        raw=False,
    )


def _regime(row: pd.Series) -> str:
    pct = row.get("vix_percentile_60")
    change = row.get("vix_change_pct")
    zscore = row.get("vix_change_zscore_20")
    if pd.isna(pct) or pd.isna(change):
        return "WARM-UP"
    if pd.notna(zscore) and zscore >= 2:
        return "VIX SHOCK"
    if pd.notna(zscore) and zscore <= -2:
        return "VIX CRUSH"
    level = "HIGH" if pct >= 60 else "LOW" if pct <= 40 else "NORMAL"
    direction = "RISING" if change > 0.20 else "FALLING" if change < -0.20 else "STABLE"
    return f"{level} & {direction}"


def _price_vix_state(row: pd.Series) -> str:
    n = row.get("nifty_return_pct")
    v = row.get("vix_change_pct")
    if pd.isna(n) or pd.isna(v):
        return "WARM-UP"
    if n > 0 and v < 0:
        return "BULLISH CONFIRMATION"
    if n < 0 and v > 0:
        return "BEARISH CONFIRMATION"
    if n > 0 and v > 0:
        return "UNSTABLE RALLY"
    if n < 0 and v < 0:
        return "CONTROLLED DECLINE"
    return "NEUTRAL"


def enrich_vix_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time India VIX features without using future information."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    out["nifty_close"] = pd.to_numeric(out["nifty_close"], errors="coerce")
    out["india_vix_close"] = pd.to_numeric(out["india_vix_close"], errors="coerce")

    out["nifty_return_pct"] = out["nifty_close"].pct_change() * 100
    out["vix_change_pct"] = out["india_vix_close"].pct_change() * 100
    out["vix_sma_20"] = out["india_vix_close"].rolling(20, min_periods=5).mean()
    out["vix_percentile_60"] = _rolling_percentile(out["india_vix_close"], 60)
    out["vix_percentile_252"] = _rolling_percentile(out["india_vix_close"], 252)

    mean20 = out["vix_change_pct"].rolling(20, min_periods=10).mean()
    std20 = out["vix_change_pct"].rolling(20, min_periods=10).std(ddof=0).replace(0, np.nan)
    out["vix_change_zscore_20"] = (out["vix_change_pct"] - mean20) / std20
    out["nifty_vix_corr_20"] = out["nifty_return_pct"].rolling(20, min_periods=10).corr(
        out["vix_change_pct"]
    )
    out["vix_regime"] = out.apply(_regime, axis=1)
    out["price_vix_state"] = out.apply(_price_vix_state, axis=1)
    return out


def vix_risk_multiplier(regime: str) -> float:
    regime = str(regime).upper()
    if "SHOCK" in regime:
        return 0.25
    if "HIGH & RISING" in regime:
        return 0.50
    if "RISING" in regime:
        return 0.70
    if "NORMAL" in regime:
        return 0.90
    return 1.00

