import pandas as pd

from core.demo import make_demo_data
from core.vix import enrich_vix_features, validate_market_data, vix_risk_multiplier


def test_demo_data_is_valid():
    assert validate_market_data(make_demo_data()) == []


def test_features_are_created():
    result = enrich_vix_features(make_demo_data())
    assert {"vix_regime", "vix_percentile_60", "nifty_vix_corr_20"}.issubset(result.columns)
    assert result["vix_regime"].notna().all()


def test_validation_catches_missing_columns():
    problems = validate_market_data(pd.DataFrame({"timestamp": ["2026-01-01"]}))
    assert problems


def test_shock_reduces_risk():
    assert vix_risk_multiplier("VIX SHOCK") == 0.25

