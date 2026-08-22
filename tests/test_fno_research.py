import pandas as pd

from core.fno_research import (
    build_stock_1115_outcomes,
    prepare_fno_history,
    sector_vix_sensitivity,
    stock_vix_sensitivity,
    validate_fno_history,
)


def make_stock_history(days=35):
    rows = []
    for stock, sector, offset in (("AAA", "BANK", 0), ("BBB", "IT", 10)):
        for day_number, day in enumerate(pd.bdate_range("2026-01-01", periods=days)):
            for bar, timestamp in enumerate(
                pd.date_range(day + pd.Timedelta(hours=9, minutes=15), periods=13, freq="30min")
            ):
                price = 100 + offset + day_number * 0.1 + bar * 0.05
                rows.append({
                    "timestamp": timestamp,
                    "Stock": stock,
                    "Sector": sector,
                    "open": price,
                    "high": price + 0.2,
                    "low": price - 0.2,
                    "close": price + 0.05,
                    "volume": 1000 + bar,
                    "nifty_close": 25000 + day_number + bar,
                    "india_vix_close": 14 + ((day_number % 5) - 2) * 0.1 + bar * 0.01,
                })
    return pd.DataFrame(rows)


def test_fno_point_in_time_outcomes_and_rankings():
    raw = make_stock_history()
    assert validate_fno_history(raw) == []
    prepared = prepare_fno_history(raw)
    outcomes = build_stock_1115_outcomes(prepared)
    assert len(outcomes) == 70
    assert outcomes["source_bar_start"].dt.strftime("%H:%M").eq("10:45").all()
    assert outcomes[["return_1145_pct", "return_1315_pct", "return_1515_pct"]].notna().all().all()
    assert len(stock_vix_sensitivity(outcomes, minimum_sessions=30)) == 2
    assert len(sector_vix_sensitivity(outcomes, minimum_sessions=30)) == 2

