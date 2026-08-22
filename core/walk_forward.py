from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def _score(data: pd.DataFrame, weights: tuple[int, int, int]) -> pd.Series:
    rf, stock_rs, sector_rs = weights
    return (
        rf / 100 * (data["RF_Percentile"] * 2 - 100)
        + stock_rs / 100 * (data["Stock_RS_Percentile"] * 2 - 100)
        + sector_rs / 100 * (data["Sector_RS_Percentile"] * 2 - 100)
    )


def _trades(
    data: pd.DataFrame,
    weights: tuple[int, int, int],
    long_threshold: int,
    short_threshold: int,
    cost_bps: float,
) -> pd.DataFrame:
    score = _score(data, weights)
    side = np.select([score >= long_threshold, score <= -short_threshold], [1, -1], default=0)
    out = data.loc[side != 0].copy()
    out["Calibrated_Score"] = score.loc[side != 0]
    out["Calibrated_Signal"] = np.where(np.asarray(side)[side != 0] == 1, "LONG", "SHORT")
    out["Gross_Oriented_Return_%"] = np.where(
        out["Calibrated_Signal"] == "LONG", out["return_1515_pct"], -out["return_1515_pct"]
    )
    out["Net_Oriented_Return_%"] = out["Gross_Oriented_Return_%"] - float(cost_bps) / 100
    return out


def threshold_sensitivity(
    signals: pd.DataFrame,
    cost_bps: float = 10,
    weights: tuple[int, int, int] = (40, 35, 25),
    thresholds: tuple[int, ...] = (40, 50, 60, 70, 80),
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        trades = _trades(signals, weights, threshold, threshold, cost_bps)
        for side, group in trades.groupby("Calibrated_Signal"):
            net = group["Net_Oriented_Return_%"]
            rows.append({
                "Threshold": threshold,
                "Signal": side,
                "Trades": len(group),
                "Sessions": group["session_date"].nunique(),
                "Gross_Expectancy_%": group["Gross_Oriented_Return_%"].mean(),
                "Net_Expectancy_%": net.mean(),
                "Net_Win_Rate_%": (net > 0).mean() * 100,
                "Median_Net_Return_%": net.median(),
            })
    return pd.DataFrame(rows)


def _weight_grid(step: int = 10):
    for rf in range(20, 61, step):
        for stock_rs in range(20, 61, step):
            sector_rs = 100 - rf - stock_rs
            if 10 <= sector_rs <= 50:
                yield rf, stock_rs, sector_rs


def calibration_surface(
    signals: pd.DataFrame,
    cost_bps: float = 10,
    minimum_trades_per_side: int = 100,
) -> pd.DataFrame:
    rows = []
    for weights, long_threshold, short_threshold in itertools.product(
        _weight_grid(), (40, 50, 60, 70), (40, 50, 60, 70)
    ):
        trades = _trades(signals, weights, long_threshold, short_threshold, cost_bps)
        longs = trades[trades["Calibrated_Signal"] == "LONG"]
        shorts = trades[trades["Calibrated_Signal"] == "SHORT"]
        if len(longs) < minimum_trades_per_side or len(shorts) < minimum_trades_per_side:
            continue
        rows.append({
            "RF_Weight": weights[0],
            "Stock_RS_Weight": weights[1],
            "Sector_RS_Weight": weights[2],
            "Long_Threshold": long_threshold,
            "Short_Threshold": short_threshold,
            "Trades": len(trades),
            "Long_Trades": len(longs),
            "Short_Trades": len(shorts),
            "Long_Net_Expectancy_%": longs["Net_Oriented_Return_%"].mean(),
            "Short_Net_Expectancy_%": shorts["Net_Oriented_Return_%"].mean(),
            "Combined_Net_Expectancy_%": trades["Net_Oriented_Return_%"].mean(),
            "Net_Win_Rate_%": (trades["Net_Oriented_Return_%"] > 0).mean() * 100,
        })
    result = pd.DataFrame(rows)
    return result.sort_values("Combined_Net_Expectancy_%", ascending=False).reset_index(drop=True)


def expanding_walk_forward(
    signals: pd.DataFrame,
    cost_bps: float = 10,
    initial_train_sessions: int = 50,
    test_sessions: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = signals.copy()
    data["session_date"] = pd.to_datetime(data["session_date"]).dt.date
    sessions = sorted(data["session_date"].unique())
    folds = []
    test_trades = []
    train_end = initial_train_sessions
    fold_number = 1
    while train_end < len(sessions):
        test_end = min(train_end + test_sessions, len(sessions))
        if test_end - train_end < 3:
            break
        train_dates = sessions[:train_end]
        test_dates = sessions[train_end:test_end]
        train = data[data["session_date"].isin(train_dates)]
        test = data[data["session_date"].isin(test_dates)]
        surface = calibration_surface(train, cost_bps, minimum_trades_per_side=50)
        if surface.empty:
            break
        best = surface.iloc[0]
        weights = (int(best.RF_Weight), int(best.Stock_RS_Weight), int(best.Sector_RS_Weight))
        trades = _trades(
            test, weights, int(best.Long_Threshold), int(best.Short_Threshold), cost_bps
        )
        trades["Fold"] = fold_number
        test_trades.append(trades)
        folds.append({
            "Fold": fold_number,
            "Train_Start": train_dates[0],
            "Train_End": train_dates[-1],
            "Test_Start": test_dates[0],
            "Test_End": test_dates[-1],
            "RF_Weight": weights[0],
            "Stock_RS_Weight": weights[1],
            "Sector_RS_Weight": weights[2],
            "Long_Threshold": int(best.Long_Threshold),
            "Short_Threshold": int(best.Short_Threshold),
            "Test_Trades": len(trades),
            "Test_Net_Win_Rate_%": (trades["Net_Oriented_Return_%"] > 0).mean() * 100,
            "Test_Net_Expectancy_%": trades["Net_Oriented_Return_%"].mean(),
        })
        train_end = test_end
        fold_number += 1
    return pd.DataFrame(folds), pd.concat(test_trades, ignore_index=True) if test_trades else pd.DataFrame()

