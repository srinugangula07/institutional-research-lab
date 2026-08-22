from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_data(rows: int = 320, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic data for UI verification only—not research evidence."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=max(1, int(np.ceil(rows / 13))))
    timestamps = []
    for day in days:
        timestamps.extend(pd.date_range(day + pd.Timedelta(hours=9, minutes=15), periods=13, freq="30min"))
    timestamp = pd.DatetimeIndex(timestamps[:rows])
    vix_change = rng.normal(0, 0.018, rows)
    nifty_change = -0.30 * vix_change + rng.normal(0.00015, 0.0045, rows)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "nifty_close": 24000 * np.cumprod(1 + nifty_change),
            "india_vix_close": 14 * np.cumprod(1 + vix_change),
        }
    )
