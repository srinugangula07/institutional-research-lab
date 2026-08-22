from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_FUTURES_COLUMNS = {
    "Captured_At", "Snapshot_Date", "Snapshot_Slot", "Stock", "Future_Symbol",
    "Future_Expiry", "Spot_Price", "Future_Price", "Future_Price_Change_%",
    "Future_OI", "Previous_Future_OI", "OI_Change", "OI_Change_%", "Basis_%",
    "Previous_Basis_%", "Basis_Change_pp", "Basis_State", "Positioning", "Data_Quality",
}


def _capture_slot(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Kolkata")
    else:
        ts = ts.tz_convert("Asia/Kolkata")
    if ts.weekday() >= 5:
        return None
    minute = ts.hour * 60 + ts.minute
    if 11 * 60 + 5 <= minute <= 11 * 60 + 25:
        return "11:15"
    if 15 * 60 + 10 <= minute <= 15 * 60 + 25:
        return "15:15"
    return None


def validate_futures_snapshots(source):
    problems = []
    missing = sorted(REQUIRED_FUTURES_COLUMNS.difference(source.columns))
    if missing:
        return ["Missing futures columns: " + ", ".join(missing)]
    if source.empty:
        return ["Futures research file is empty"]
    if source.duplicated(["Snapshot_Date", "Snapshot_Slot", "Stock", "Future_Symbol"]).any():
        problems.append("Duplicate stock/contract snapshot keys found")
    if not source["Data_Quality"].astype(str).eq("COMPLETE").all():
        problems.append("Only COMPLETE futures observations are accepted")
    captured = pd.to_datetime(source["Captured_At"], errors="coerce")
    if captured.isna().any():
        problems.append("Invalid Captured_At timestamps found")
    else:
        derived = source["Captured_At"].map(_capture_slot)
        invalid = derived.isna() | derived.astype(str).ne(source["Snapshot_Slot"].astype(str))
        if invalid.any():
            problems.append(
                f"{int(invalid.sum())} observations are weekend, out-of-window or incorrectly labelled"
            )
    numeric = [
        "Spot_Price", "Future_Price", "Future_OI", "Previous_Future_OI",
        "OI_Change_%", "Basis_%", "Basis_Change_pp",
    ]
    converted = source[numeric].apply(pd.to_numeric, errors="coerce")
    if converted.isna().any().any():
        problems.append("Missing or non-numeric price/OI/basis values found")
    if (converted[["Spot_Price", "Future_Price", "Future_OI", "Previous_Future_OI"]] <= 0).any().any():
        problems.append("Non-positive price or OI values found")
    return problems


def prepare_futures_snapshots(source, slot="11:15"):
    out = source.copy()
    out["Stock"] = out["Stock"].astype(str).str.upper().str.strip()
    out["session_date"] = pd.to_datetime(out["Snapshot_Date"]).dt.date
    out = out[out["Snapshot_Slot"].astype(str).eq(str(slot))].copy()
    return out.sort_values(["session_date", "Stock"]).reset_index(drop=True)


def build_futures_confirmation(signals, futures):
    sig = signals.copy()
    fut = futures.copy()
    sig["Stock"] = sig["Stock"].astype(str).str.upper().str.strip()
    sig["session_date"] = pd.to_datetime(sig["session_date"]).dt.date
    fut["session_date"] = pd.to_datetime(fut["session_date"]).dt.date
    keep = [
        "Stock", "session_date", "Future_Symbol", "Future_Expiry", "Future_Price_Change_%",
        "OI_Change_%", "Basis_%", "Basis_Change_pp", "Basis_State", "Positioning",
    ]
    out = sig.merge(fut[keep], on=["Stock", "session_date"], how="inner", validate="one_to_one")
    if out.empty:
        return out
    out["Futures_Alignment"] = np.select(
        [
            out["Signal"].eq("LONG") & out["Positioning"].eq("LONG BUILDUP"),
            out["Signal"].eq("SHORT") & out["Positioning"].eq("SHORT BUILDUP"),
            out["Signal"].eq("LONG") & out["Positioning"].eq("SHORT BUILDUP"),
            out["Signal"].eq("SHORT") & out["Positioning"].eq("LONG BUILDUP"),
            out["Signal"].eq("LONG") & out["Positioning"].eq("SHORT COVERING"),
            out["Signal"].eq("SHORT") & out["Positioning"].eq("LONG UNWINDING"),
        ],
        [
            "CONFIRMED LONG", "CONFIRMED SHORT", "REJECTED LONG", "REJECTED SHORT",
            "WEAK LONG", "WEAK SHORT",
        ],
        default="UNCONFIRMED",
    )
    out["Futures_Confirmed"] = out["Futures_Alignment"].isin(
        ["CONFIRMED LONG", "CONFIRMED SHORT"]
    )
    return out


def futures_confirmation_summary(matched, cost_bps=10):
    if matched is None or matched.empty:
        return pd.DataFrame()
    out = matched.copy()
    out["Net_Oriented_Return_%"] = (
        pd.to_numeric(out["Oriented_Return_1515_%"], errors="coerce") - cost_bps / 100.0
    )
    return out.groupby("Futures_Alignment", as_index=False).agg(
        Trades=("Stock", "size"),
        Sessions=("session_date", "nunique"),
        Gross_Win_Rate=("Oriented_Return_1515_%", lambda x: (x > 0).mean() * 100),
        Gross_Expectancy=("Oriented_Return_1515_%", "mean"),
        Net_Win_Rate=("Net_Oriented_Return_%", lambda x: (x > 0).mean() * 100),
        Net_Expectancy=("Net_Oriented_Return_%", "mean"),
    ).sort_values("Net_Expectancy", ascending=False)
