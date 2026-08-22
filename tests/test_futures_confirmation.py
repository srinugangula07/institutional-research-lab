import pandas as pd

from core.futures_confirmation import build_futures_confirmation, validate_futures_snapshots


def test_alignment_labels():
    signals = pd.DataFrame([
        {"Stock": "A", "session_date": "2026-08-24", "Signal": "LONG", "Oriented_Return_1515_%": 0.2},
        {"Stock": "B", "session_date": "2026-08-24", "Signal": "SHORT", "Oriented_Return_1515_%": 0.3},
    ])
    futures = pd.DataFrame([
        {"Stock": "A", "session_date": "2026-08-24", "Future_Symbol": "AFUT", "Future_Expiry": "2026-08-25", "Future_Price_Change_%": 1, "OI_Change_%": 2, "Basis_%": 0.1, "Basis_Change_pp": 0.1, "Basis_State": "EXPANDING", "Positioning": "LONG BUILDUP"},
        {"Stock": "B", "session_date": "2026-08-24", "Future_Symbol": "BFUT", "Future_Expiry": "2026-08-25", "Future_Price_Change_%": -1, "OI_Change_%": 2, "Basis_%": -0.1, "Basis_Change_pp": -0.1, "Basis_State": "CONTRACTING", "Positioning": "SHORT BUILDUP"},
    ])
    out = build_futures_confirmation(signals, futures)
    assert out["Futures_Confirmed"].all()


def test_weekend_snapshot_rejected():
    required = {
        "Captured_At": "2026-08-22T22:05:00+05:30", "Snapshot_Date": "2026-08-22",
        "Snapshot_Slot": "11:15", "Stock": "A", "Future_Symbol": "AFUT",
        "Future_Expiry": "2026-08-25", "Spot_Price": 100, "Future_Price": 101,
        "Future_Price_Change_%": 1, "Future_OI": 1000, "Previous_Future_OI": 900,
        "OI_Change": 100, "OI_Change_%": 11.1, "Basis_%": 1,
        "Previous_Basis_%": 0.5, "Basis_Change_pp": 0.5, "Basis_State": "EXPANDING",
        "Positioning": "LONG BUILDUP", "Data_Quality": "COMPLETE",
    }
    problems = validate_futures_snapshots(pd.DataFrame([required]))
    assert any("weekend" in problem for problem in problems)
