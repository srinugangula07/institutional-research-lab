import datetime as dt
import numpy as np
import pandas as pd


DEFAULT_CAPTURE_SLOTS = [
    "09:30",
    "10:00",
    "11:15",
    "12:30",
    "14:00",
    "15:15",
]


def _norm_stock(df):
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "Stock" not in out.columns:
        for c in ["stock", "Symbol", "Underlying"]:
            if c in out.columns:
                out = out.rename(columns={c: "Stock"})
                break

    if "Stock" not in out.columns:
        return pd.DataFrame()

    out["Stock"] = out["Stock"].astype(str).str.upper().str.strip()
    return out


def _dedupe(df, cols):
    if df is None or df.empty:
        return pd.DataFrame()

    keep = ["Stock"] + [c for c in cols if c in df.columns]
    return df[keep].drop_duplicates("Stock")


def build_research_snapshot(
    phase7d=None,
    phase7e=None,
    phase7g=None,
    gated_trades=None,
    admission_log=None,
):
    """
    Build one wide per-stock research snapshot from the latest live state.

    The output is designed for later backtesting / sensitivity analysis:
      Phase-7 institutional context
      live timing state
      transition state
      portfolio admission state
      current paper-trade state
    """
    p7d = _norm_stock(phase7d)
    p7e = _norm_stock(phase7e)
    p7g = _norm_stock(phase7g)
    trades = _norm_stock(gated_trades)
    adm = _norm_stock(admission_log)

    frames = []

    if not p7d.empty:
        frames.append(
            _dedupe(
                p7d,
                [
                    "Sector",
                    "P7D_Institutional_Score",
                    "P7D_Final_Action",
                    "P7D_Adjusted_Conviction",
                    "P7D_Conviction_Grade",
                    "P7D_Conflict_Penalty",
                    "P7D_Soft_Conflict_Flags",
                    "P7D_Hard_Veto_Flags",
                    "P7_RF_Score",
                    "P7_Sector_RS_Score",
                    "P7_Stock_RS_Score",
                    "P7_Futures_Score",
                    "P7_Options_Score",
                    "Institutional_Signal",
                    "Lightweight_Heavy_Alignment",
                    "Gamma_Regime",
                    "Zero_Gamma_Level",
                ],
            )
        )

    if not p7e.empty:
        frames.append(
            _dedupe(
                p7e,
                [
                    "P7E_Direction",
                    "P7E_Timing_Score",
                    "P7E_Participation_State",
                    "P7E_Entry_State",
                    "Live_RF",
                    "RVOL_Same_Time",
                    "RVOL_Baseline_Sessions",
                    "LTP",
                    "VWAP",
                    "IB_High",
                    "IB_Low",
                    "Open",
                    "Day_High",
                    "Day_Low",
                    "P7E_Why",
                    "Live_Data_Status",
                ],
            )
        )

    if not p7g.empty:
        frames.append(
            _dedupe(
                p7g,
                [
                    "P7G_Previous_State",
                    "P7G_Transition",
                    "P7G_State_Changed",
                    "P7G_Snapshot_Time",
                ],
            )
        )

    if not trades.empty:
        trade_cols = [
            "Paper_Status",
            "Paper_Side",
            "Entry_Price",
            "Stop_Price",
            "Target_Price",
            "Quantity",
            "Risk_Budget",
            "Open_Time",
            "Exit_Time",
            "Exit_Price",
            "Exit_Reason",
            "Realized_PnL",
            "Realized_R",
        ]
        frames.append(_dedupe(trades, trade_cols))

    if not adm.empty:
        # Keep latest admission decision per stock.
        if "Timestamp" in adm.columns:
            adm["Timestamp"] = pd.to_datetime(adm["Timestamp"], errors="coerce")
            adm = adm.sort_values("Timestamp").drop_duplicates("Stock", keep="last")

        admission_cols = [
            "Decision",
            "Reason",
            "Open_Positions_After",
            "Portfolio_Heat_After",
            "Sector_Positions_After",
        ]
        frames.append(_dedupe(adm, admission_cols))

    if not frames:
        return pd.DataFrame()

    out = frames[0]

    for f in frames[1:]:
        out = out.merge(f, on="Stock", how="outer")

    out["Research_Snapshot_Time"] = pd.Timestamp.now()
    return out


def due_capture_slot(
    now=None,
    slots=None,
    tolerance_minutes=7,
):
    """
    Return the scheduled slot due around 'now'.

    This does not create a background scheduler. Streamlit must rerun while the
    clock is within the tolerance window. De-duplication is handled by DB key.
    """
    if now is None:
        now = dt.datetime.now()

    if slots is None:
        slots = DEFAULT_CAPTURE_SLOTS

    today = now.date()

    candidates = []

    for slot in slots:
        hh, mm = [int(x) for x in slot.split(":")]
        target = dt.datetime.combine(today, dt.time(hh, mm))
        delta = abs((now - target).total_seconds()) / 60.0

        if delta <= float(tolerance_minutes):
            candidates.append((delta, slot, target))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return {
        "slot": candidates[0][1],
        "target_time": candidates[0][2],
        "delta_minutes": round(candidates[0][0], 2),
    }


def capture_key_for_slot(slot, now=None, prefix="RESEARCH"):
    if now is None:
        now = dt.datetime.now()

    return f"{prefix}:{now.date().isoformat()}:{slot}"
