import pandas as pd

from core.demo import make_demo_data
from core.history import (
    build_1115_outcomes,
    merge_history,
    normalise_columns,
    outcome_summary,
    restrict_nse_session,
    session_quality,
)
from core.vix import enrich_vix_features


def test_aliases_and_incremental_merge():
    first = pd.DataFrame({"datetime": ["2026-08-21 09:15"], "nifty": [25000]})
    second = pd.DataFrame({"timestamp": ["2026-08-21 09:15"], "vix": [12.5]})
    merged = merge_history(normalise_columns(first), normalise_columns(second))
    assert len(merged) == 1
    assert merged.loc[0, "nifty_close"] == 25000
    assert merged.loc[0, "india_vix_close"] == 12.5


def test_nse_sessions_and_1115_outcomes():
    history = restrict_nse_session(make_demo_data())
    enriched = enrich_vix_features(history)
    outcomes = build_1115_outcomes(enriched)
    assert not outcomes.empty
    assert {"return_1200_pct", "return_1330_pct", "return_1515_pct"}.issubset(outcomes.columns)
    assert not outcome_summary(outcomes).empty
    assert not session_quality(enriched).empty

