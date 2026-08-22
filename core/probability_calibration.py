from __future__ import annotations

import numpy as np
import pandas as pd


SCORE_BINS = [-101, -80, -60, -40, -20, 0, 20, 40, 60, 80, 101]
SCORE_LABELS = ["-100:-80", "-80:-60", "-60:-40", "-40:-20", "-20:0", "0:20", "20:40", "40:60", "60:80", "80:100"]


def probability_calibration(signals, holdout_sessions=20, horizon="15:15"):
    outcome = f"relative_return_{horizon.replace(':','')}_pct"
    source = signals.copy()
    source["session_date"] = pd.to_datetime(source["session_date"]).dt.date
    source["Score"] = pd.to_numeric(source["Institutional_Directional_Score"], errors="coerce")
    source["Outcome"] = (pd.to_numeric(source[outcome], errors="coerce") > 0).astype(float)
    source = source.dropna(subset=["Score", outcome])
    source["Score_Bin"] = pd.cut(source["Score"], SCORE_BINS, labels=SCORE_LABELS, include_lowest=True)
    dates = sorted(source["session_date"].unique())
    holdout_dates = set(dates[-int(holdout_sessions):])
    development = source[~source["session_date"].isin(holdout_dates)].copy()
    holdout = source[source["session_date"].isin(holdout_dates)].copy()
    dev_curve = development.groupby("Score_Bin", observed=False, as_index=False).agg(
        Development_Rows=("Stock", "size"),
        Development_Outperform_Probability=("Outcome", "mean"),
        Mean_Score=("Score", "mean"),
    )
    global_probability = development["Outcome"].mean()
    dev_curve["Development_Outperform_Probability"] = dev_curve[
        "Development_Outperform_Probability"
    ].fillna(global_probability)
    holdout = holdout.merge(
        dev_curve[["Score_Bin", "Development_Outperform_Probability"]],
        on="Score_Bin", how="left",
    )
    holdout["Predicted_Probability"] = holdout["Development_Outperform_Probability"].fillna(global_probability)
    holdout["Squared_Error"] = (holdout["Predicted_Probability"] - holdout["Outcome"]) ** 2
    holdout_curve = holdout.groupby("Score_Bin", observed=False, as_index=False).agg(
        Holdout_Rows=("Stock", "size"),
        Predicted_Probability=("Predicted_Probability", "mean"),
        Actual_Outperform_Probability=("Outcome", "mean"),
        Brier_Score=("Squared_Error", "mean"),
    )
    curve = dev_curve.merge(holdout_curve, on="Score_Bin", how="left")
    valid = curve["Holdout_Rows"].fillna(0) > 0
    weights = curve.loc[valid, "Holdout_Rows"] / curve.loc[valid, "Holdout_Rows"].sum()
    ece = (
        weights * (
            curve.loc[valid, "Predicted_Probability"] - curve.loc[valid, "Actual_Outperform_Probability"]
        ).abs()
    ).sum()
    ordered_actual = curve.loc[valid, "Actual_Outperform_Probability"].dropna()
    monotonic = bool(ordered_actual.is_monotonic_increasing) if len(ordered_actual) >= 3 else False
    metrics = pd.DataFrame([{
        "Development_Sessions": development["session_date"].nunique(),
        "Holdout_Sessions": holdout["session_date"].nunique(),
        "Holdout_Rows": len(holdout),
        "Holdout_Brier_Score": holdout["Squared_Error"].mean(),
        "Naive_50pct_Brier": ((0.5 - holdout["Outcome"]) ** 2).mean(),
        "Expected_Calibration_Error": ece,
        "Holdout_Monotonic": monotonic,
        "Probability_Edge_vs_50pct": holdout["Outcome"].mean() - 0.5,
    }])
    return metrics, curve, holdout
