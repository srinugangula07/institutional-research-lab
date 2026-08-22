from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_WEIGHTS = {
    "Full 40/35/25": (0.40, 0.35, 0.25),
    "RF only": (1.00, 0.00, 0.00),
    "Stock RS only": (0.00, 1.00, 0.00),
    "Sector RS only": (0.00, 0.00, 1.00),
    "Without RF": (0.00, 0.583333, 0.416667),
    "Without Stock RS": (0.615385, 0.00, 0.384615),
    "Without Sector RS": (0.533333, 0.466667, 0.00),
}

HORIZON_COLUMNS = {
    "11:45": "relative_return_1145_pct",
    "13:15": "relative_return_1315_pct",
    "15:15": "relative_return_1515_pct",
}


def add_ablation_scores(signals):
    out = signals.copy()
    rf = pd.to_numeric(out["RF_Percentile"], errors="coerce")
    stock = pd.to_numeric(out["Stock_RS_Percentile"], errors="coerce")
    sector = pd.to_numeric(out["Sector_RS_Percentile"], errors="coerce")
    for name, (rf_weight, stock_weight, sector_weight) in MODEL_WEIGHTS.items():
        out[name] = rf_weight * rf + stock_weight * stock + sector_weight * sector
    return out


def daily_ablation_ic(signals, horizon="15:15"):
    source = add_ablation_scores(signals)
    source["session_date"] = pd.to_datetime(source["session_date"]).dt.date
    outcome = HORIZON_COLUMNS[horizon]
    rows = []
    for session_date, day in source.groupby("session_date", sort=True):
        for model in MODEL_WEIGHTS:
            pair = day[[model, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
            ic = (
                pair[model].corr(pair[outcome], method="spearman")
                if len(pair) >= 20 and pair[model].nunique() > 1 and pair[outcome].nunique() > 1
                else np.nan
            )
            rows.append({
                "session_date": session_date,
                "Horizon": horizon,
                "Model": model,
                "Rank_IC": ic,
                "Stocks": len(pair),
            })
    return pd.DataFrame(rows)


def ablation_summary(daily_ic, holdout_sessions=20):
    if daily_ic is None or daily_ic.empty:
        return pd.DataFrame()
    dates = sorted(daily_ic["session_date"].unique())
    holdout_dates = set(dates[-int(holdout_sessions):])
    rows = []
    for sample, frame in [
        ("ALL", daily_ic),
        ("HOLDOUT", daily_ic[daily_ic["session_date"].isin(holdout_dates)]),
    ]:
        for model, group in frame.groupby("Model", sort=False):
            values = group["Rank_IC"].dropna()
            std = values.std(ddof=1)
            rows.append({
                "Sample": sample,
                "Model": model,
                "Sessions": len(values),
                "Mean_Rank_IC": values.mean(),
                "Median_Rank_IC": values.median(),
                "Positive_IC_Rate_%": (values > 0).mean() * 100,
                "IC_t_Statistic": values.mean() / (std / np.sqrt(len(values)))
                if pd.notna(std) and std > 0 and len(values) > 1 else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["Sample", "Mean_Rank_IC"], ascending=[True, False])


def removal_impact(summary):
    holdout = summary[summary["Sample"].eq("HOLDOUT")].set_index("Model")
    full = holdout.loc["Full 40/35/25", "Mean_Rank_IC"]
    mapping = {
        "RF": "Without RF",
        "Stock RS": "Without Stock RS",
        "Sector RS": "Without Sector RS",
    }
    rows = []
    for feature, removed_model in mapping.items():
        without = holdout.loc[removed_model, "Mean_Rank_IC"]
        impact = full - without
        rows.append({
            "Feature": feature,
            "Full_Holdout_IC": full,
            "IC_Without_Feature": without,
            "Marginal_IC_Contribution": impact,
            "Diagnosis": "HELPS" if impact > 0.01 else "HURTS" if impact < -0.01 else "NEUTRAL",
        })
    return pd.DataFrame(rows).sort_values("Marginal_IC_Contribution", ascending=False)


def component_correlation(signals):
    columns = ["RF_Percentile", "Stock_RS_Percentile", "Sector_RS_Percentile"]
    return signals[columns].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
