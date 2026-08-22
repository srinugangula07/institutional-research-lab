from __future__ import annotations

import numpy as np
import pandas as pd


HORIZONS = {
    "11:45": "relative_return_1145_pct",
    "13:15": "relative_return_1315_pct",
    "15:15": "relative_return_1515_pct",
}


def daily_information_coefficients(signals):
    source = signals.copy()
    source["session_date"] = pd.to_datetime(source["session_date"]).dt.date
    source["Institutional_Directional_Score"] = pd.to_numeric(
        source["Institutional_Directional_Score"], errors="coerce"
    )
    rows = []
    for session_date, day in source.groupby("session_date", sort=True):
        vix_values = pd.to_numeric(day["VIX_60D_Percentile"], errors="coerce").dropna()
        vix_pct = vix_values.median() if not vix_values.empty else np.nan
        regime = "UNAVAILABLE"
        if pd.notna(vix_pct):
            regime = "LOW" if vix_pct < 33 else "HIGH" if vix_pct >= 67 else "NORMAL"
        for horizon, column in HORIZONS.items():
            pair = day[["Institutional_Directional_Score", column]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            ic = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman") if len(pair) >= 20 else np.nan
            rows.append({
                "session_date": session_date,
                "Horizon": horizon,
                "Rank_IC": ic,
                "Stocks": len(pair),
                "VIX_Regime": regime,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Rolling_20D_IC"] = out.groupby("Horizon")["Rank_IC"].transform(
            lambda values: values.rolling(20, min_periods=10).mean()
        )
    return out


def information_coefficient_summary(daily_ic, holdout_sessions=20):
    if daily_ic is None or daily_ic.empty:
        return pd.DataFrame()
    dates = sorted(daily_ic["session_date"].unique())
    holdout_dates = set(dates[-int(holdout_sessions):])
    frames = [("ALL", daily_ic), ("HOLDOUT", daily_ic[daily_ic["session_date"].isin(holdout_dates)])]
    rows = []
    for sample, data in frames:
        for horizon, frame in data.groupby("Horizon", sort=False):
            values = frame["Rank_IC"].dropna()
            std = values.std(ddof=1)
            rows.append({
                "Sample": sample,
                "Horizon": horizon,
                "Sessions": len(values),
                "Mean_Rank_IC": values.mean(),
                "Median_Rank_IC": values.median(),
                "Positive_IC_Rate_%": (values > 0).mean() * 100,
                "IC_Information_Ratio": values.mean() / std * np.sqrt(252)
                if pd.notna(std) and std > 0 else np.nan,
                "IC_t_Statistic": values.mean() / (std / np.sqrt(len(values)))
                if pd.notna(std) and std > 0 and len(values) > 1 else np.nan,
            })
    return pd.DataFrame(rows)


def quintile_spread(signals, horizon="15:15"):
    return_column = HORIZONS[horizon]
    source = signals.copy()
    source["session_date"] = pd.to_datetime(source["session_date"]).dt.date
    rows = []
    for session_date, day in source.groupby("session_date", sort=True):
        work = day[["Institutional_Directional_Score", return_column]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        if len(work) < 50:
            continue
        work["Score_Quintile"] = pd.qcut(
            work["Institutional_Directional_Score"].rank(method="first"),
            5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        means = work.groupby("Score_Quintile")[return_column].mean()
        for quintile, value in means.items():
            rows.append({
                "session_date": session_date,
                "Horizon": horizon,
                "Score_Quintile": quintile,
                "Mean_Relative_Return_%": value,
                "Q5_minus_Q1_%": means.get(5, np.nan) - means.get(1, np.nan),
            })
    return pd.DataFrame(rows)


def ic_regime_summary(daily_ic):
    if daily_ic is None or daily_ic.empty:
        return pd.DataFrame()
    valid = daily_ic[daily_ic["VIX_Regime"].ne("UNAVAILABLE")].copy()
    return valid.groupby(["VIX_Regime", "Horizon"], as_index=False).agg(
        Sessions=("Rank_IC", "count"),
        Mean_Rank_IC=("Rank_IC", "mean"),
        Positive_IC_Rate=("Rank_IC", lambda x: (x > 0).mean() * 100),
    )
