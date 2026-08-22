from __future__ import annotations

import numpy as np
import pandas as pd


def build_excursion_diagnostics(signals, cost_bps=10):
    active = signals[signals["Signal"].isin(["LONG", "SHORT"])].copy()
    side = np.where(active["Signal"].eq("LONG"), 1.0, -1.0)
    active["Oriented_1145_%"] = side * pd.to_numeric(active["return_1145_pct"], errors="coerce")
    active["Oriented_1315_%"] = side * pd.to_numeric(active["return_1315_pct"], errors="coerce")
    active["Oriented_1515_%"] = side * pd.to_numeric(active["return_1515_pct"], errors="coerce")
    active["Favorable_Excursion_%"] = np.where(
        active["Signal"].eq("LONG"), active["mfe_high_pct"], -active["mae_low_pct"]
    )
    active["Adverse_Excursion_%"] = np.where(
        active["Signal"].eq("LONG"), active["mae_low_pct"], -active["mfe_high_pct"]
    )
    active["Net_1515_%"] = active["Oriented_1515_%"] - float(cost_bps) / 100.0
    active["Capture_Efficiency_%"] = np.where(
        active["Favorable_Excursion_%"] > 0,
        active["Oriented_1515_%"] / active["Favorable_Excursion_%"] * 100,
        np.nan,
    )
    active["Score_Band"] = pd.cut(
        active["Institutional_Directional_Score"].abs(),
        [0, 40, 60, 80, 101], labels=["<40", "40–60", "60–80", "80–100"], include_lowest=True,
    )
    return active


def excursion_summary(active):
    rows = []
    for signal, group in active.groupby("Signal"):
        for checkpoint in ["Oriented_1145_%", "Oriented_1315_%", "Oriented_1515_%"]:
            values = group[checkpoint].dropna()
            rows.append({
                "Signal": signal,
                "Checkpoint": checkpoint.replace("Oriented_", "").replace("_%", ""),
                "Trades": len(values),
                "Win_Rate_%": (values > 0).mean() * 100,
                "Expectancy_%": values.mean(),
                "Median_%": values.median(),
            })
    return pd.DataFrame(rows)


def excursion_distribution(active):
    return active.groupby(["Signal", "Score_Band"], observed=False, as_index=False).agg(
        Trades=("Stock", "size"),
        Median_Favorable_Excursion=("Favorable_Excursion_%", "median"),
        Median_Adverse_Excursion=("Adverse_Excursion_%", "median"),
        Median_Capture_Efficiency=("Capture_Efficiency_%", "median"),
        Net_1515_Expectancy=("Net_1515_%", "mean"),
    )
