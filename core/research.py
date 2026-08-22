from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_table(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    rows = []
    clean = df.dropna(subset=["nifty_return_pct", "vix_change_pct"])
    for window in windows:
        sample = clean.tail(window)
        rows.append(
            {
                "Window": window,
                "Observations": len(sample),
                "NIFTY–VIX correlation": sample["nifty_return_pct"].corr(sample["vix_change_pct"]),
            }
        )
    return pd.DataFrame(rows)


def forward_return_study(
    df: pd.DataFrame,
    signal_column: str,
    horizons: list[int],
) -> pd.DataFrame:
    """Aggregate point-in-time signal outcomes at multiple row horizons."""
    work = df.copy()
    results = []
    for horizon in horizons:
        work[f"forward_{horizon}"] = work["nifty_close"].shift(-horizon) / work["nifty_close"] - 1
        grouped = work.dropna(subset=[signal_column, f"forward_{horizon}"]).groupby(signal_column)
        for signal, group in grouped:
            returns = group[f"forward_{horizon}"] * 100
            results.append(
                {
                    "Signal": signal,
                    "Horizon (bars)": horizon,
                    "Trades": len(returns),
                    "Win rate %": (returns > 0).mean() * 100,
                    "Average return %": returns.mean(),
                    "Median return %": returns.median(),
                    "Return volatility %": returns.std(ddof=0),
                    "t-stat": returns.mean() / (returns.std(ddof=1) / np.sqrt(len(returns)))
                    if len(returns) > 1 and returns.std(ddof=1) > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(results)

