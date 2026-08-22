import pandas as pd

from core.portfolio_research import build_cross_sectional_portfolio, portfolio_summary


def make_signals():
    rows = []
    for day in pd.date_range("2026-01-01", periods=25, freq="B"):
        for i in range(8):
            rows.append({
                "session_date": day, "Stock": f"S{i}", "Sector": f"SEC{i % 4}",
                "Institutional_Directional_Score": 100 - i * 10,
                "return_1515_pct": (3.5 - i) * 0.1,
                "VIX_60D_Percentile": 50,
            })
    return pd.DataFrame(rows)


def test_market_neutral_basket_and_holdout():
    daily, holdings = build_cross_sectional_portfolio(
        make_signals(), basket_size=2, max_per_sector=1, cost_bps=10
    )
    assert len(daily) == 25
    assert len(holdings) == 100
    assert (daily["Net_Long_Short_%"] > 0).all()
    summary = portfolio_summary(daily, holdout_sessions=5)
    assert summary.loc[summary["Sample"].eq("HOLDOUT"), "Sessions"].iloc[0] == 5
