from core.robustness import block_bootstrap, parameter_stability
from tests.test_portfolio_research import make_signals
from core.portfolio_research import build_cross_sectional_portfolio


def test_bootstrap_is_deterministic_and_complete():
    daily, _ = build_cross_sectional_portfolio(make_signals(), 2, 1, 10)
    summary, paths = block_bootstrap(daily, simulations=100, block_size=5, seed=7)
    assert len(summary) == 1
    assert len(paths) == 100
    assert summary.iloc[0]["Probability_Positive_Expectancy_%"] == 100


def test_parameter_surface_has_cost_stress():
    surface = parameter_stability(make_signals(), sector_cap=3)
    assert set(surface["Cost_bps"]) == {0, 5, 10, 15, 20, 30}
    assert len(surface) > 0
