import pandas as pd

from core.fno_research import build_stock_1115_outcomes, prepare_fno_history
from core.point_in_time import build_1115_rf_rs_features, signal_backtest_summary
from tests.test_fno_research import make_stock_history


def test_point_in_time_rf_rs_signals():
    history = prepare_fno_history(make_stock_history(days=35))
    outcomes = build_stock_1115_outcomes(history)
    signals = build_1115_rf_rs_features(history, outcomes)
    assert len(signals) == 70
    assert signals["source_bar_start"].dt.strftime("%H:%M").eq("10:45").all()
    assert signals[["RF_1115", "Stock_RS_1115_%", "Sector_RS_1115_%"]].notna().all().all()
    summary = signal_backtest_summary(signals)
    assert set(summary["Signal"]).issubset({"LONG", "SHORT"})

