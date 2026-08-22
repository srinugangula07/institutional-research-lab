import pandas as pd

from core.walk_forward import calibration_surface, expanding_walk_forward, threshold_sensitivity


def make_signals():
    rows = []
    for day_number, day in enumerate(pd.bdate_range("2026-01-01", periods=60)):
        for stock_number in range(20):
            percentile = (stock_number + 1) / 20 * 100
            rows.append({
                "session_date": day.date(),
                "Stock": f"S{stock_number:02d}",
                "RF_Percentile": percentile,
                "Stock_RS_Percentile": percentile,
                "Sector_RS_Percentile": percentile,
                "return_1515_pct": (percentile - 50) / 100 + (day_number % 3 - 1) * 0.01,
            })
    return pd.DataFrame(rows)


def test_sensitivity_and_walk_forward():
    signals = make_signals()
    assert len(threshold_sensitivity(signals, cost_bps=5)) == 10
    assert not calibration_surface(signals, cost_bps=5, minimum_trades_per_side=20).empty
    folds, trades = expanding_walk_forward(
        signals, cost_bps=5, initial_train_sessions=40, test_sessions=10
    )
    assert not folds.empty
    assert not trades.empty
    assert set(trades["Calibrated_Signal"]) == {"LONG", "SHORT"}

