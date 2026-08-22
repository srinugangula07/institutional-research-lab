from __future__ import annotations

import numpy as np
import pandas as pd

from core.portfolio_research import build_cross_sectional_portfolio


def block_bootstrap(daily, simulations=1000, block_size=5, seed=42):
    """Preserve short serial dependence while resampling daily net returns."""
    values = pd.to_numeric(daily["Net_Long_Short_%"], errors="coerce").dropna().to_numpy()
    n = len(values)
    if n < max(20, block_size * 2):
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    max_start = max(n - block_size + 1, 1)
    for simulation in range(1, simulations + 1):
        sampled = []
        while len(sampled) < n:
            start = int(rng.integers(0, max_start))
            sampled.extend(values[start:start + block_size])
        path = np.asarray(sampled[:n], dtype=float)
        equity = np.cumprod(1 + path / 100.0) * 100
        drawdown = (equity / np.maximum.accumulate(equity) - 1) * 100
        rows.append({
            "Simulation": simulation,
            "Expectancy_%": path.mean(),
            "Cumulative_Return_%": equity[-1] - 100,
            "Maximum_Drawdown_%": drawdown.min(),
            "Win_Rate_%": (path > 0).mean() * 100,
        })
    paths = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "Simulations": simulations,
        "Sessions_Per_Path": n,
        "Probability_Positive_Expectancy_%": (paths["Expectancy_%"] > 0).mean() * 100,
        "Expectancy_5th_%": paths["Expectancy_%"].quantile(0.05),
        "Expectancy_Median_%": paths["Expectancy_%"].median(),
        "Expectancy_95th_%": paths["Expectancy_%"].quantile(0.95),
        "Cumulative_Return_5th_%": paths["Cumulative_Return_%"].quantile(0.05),
        "Median_Maximum_Drawdown_%": paths["Maximum_Drawdown_%"].median(),
        "Drawdown_5th_Worst_%": paths["Maximum_Drawdown_%"].quantile(0.05),
    }])
    return summary, paths


def parameter_stability(signals, sector_cap=3):
    rows = []
    for basket_size in [5, 10, 15, 20, 25]:
        for cost_bps in [0, 5, 10, 15, 20, 30]:
            daily, _ = build_cross_sectional_portfolio(
                signals, basket_size=basket_size, max_per_sector=sector_cap,
                cost_bps=cost_bps,
            )
            if daily.empty:
                continue
            ret = daily["Net_Long_Short_%"]
            rows.append({
                "Basket_Size_Per_Side": basket_size,
                "Sector_Cap": sector_cap,
                "Cost_bps": cost_bps,
                "Sessions": len(daily),
                "Win_Rate_%": (ret > 0).mean() * 100,
                "Net_Expectancy_%": ret.mean(),
                "Cumulative_Return_%": (1 + ret / 100).prod() * 100 - 100,
                "Positive_After_Costs": ret.mean() > 0,
            })
    return pd.DataFrame(rows)


def leave_one_sector_out(signals, basket_size=20, sector_cap=3, cost_bps=10):
    rows = []
    baseline, _ = build_cross_sectional_portfolio(
        signals, basket_size=basket_size, max_per_sector=sector_cap, cost_bps=cost_bps
    )
    baseline_expectancy = baseline["Net_Long_Short_%"].mean() if not baseline.empty else np.nan
    for sector in sorted(signals["Sector"].dropna().astype(str).unique()):
        reduced = signals[signals["Sector"].astype(str).ne(sector)]
        daily, _ = build_cross_sectional_portfolio(
            reduced, basket_size=basket_size, max_per_sector=sector_cap, cost_bps=cost_bps
        )
        if daily.empty:
            continue
        expectancy = daily["Net_Long_Short_%"].mean()
        rows.append({
            "Excluded_Sector": sector,
            "Sessions": len(daily),
            "Net_Expectancy_%": expectancy,
            "Change_vs_Baseline_pp": expectancy - baseline_expectancy,
            "Positive_After_Exclusion": expectancy > 0,
        })
    return pd.DataFrame(rows).sort_values("Change_vs_Baseline_pp")
