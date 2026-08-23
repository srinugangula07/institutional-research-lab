import datetime as dt
import time
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
APP_BUILD = "PHASE9_20DAY_HISTORICAL_VALIDATION_V1"
from analytics.journal_db import init_journal_db, append_snapshot, append_snapshot_once, snapshot_counts, read_snapshots, database_size_bytes, export_db_bytes, restore_db_bytes, clear_journal_db, capture_registry
from analytics.research_dataset import build_research_snapshot, due_capture_slot, capture_key_for_slot, DEFAULT_CAPTURE_SLOTS
from analytics.live_feed import build_phase7f_live_feed

from data.kite_session import get_kite, exchange_request_token
from data.instruments import get_fno_equity_universe, get_nearest_futures_map
from analytics.rf import calculate_intraday_rf, daily_rf_summary, stock_rf_features, add_rf_rank_scores
from analytics.relative_strength import calculate_rs_features, percentile_scores, sector_scores, alignment_label
from analytics.auction_profile import auction_features, auction_direction_score
from analytics.market_profile import market_profile_features, add_profile_score
from analytics.institutional_setup import build_institutional_setup, component_correlation
from analytics.candidate_scanner import build_candidate_scanner
from analytics.execution_1115 import intraday_execution_features, rank_execution, add_trade_plan, rank_trade_plan
from analytics.risk_engine import add_position_sizing, portfolio_summary
from analytics.futures_positioning import (
    futures_features,
    add_futures_score,
    add_futures_conviction,
    add_institutional_alignment,
)

from analytics.options_positioning import build_option_chain
from analytics.options_greeks import add_greeks_and_gex, approximate_gamma_flip, gamma_map, add_vanna_charm_exposure
from analytics.options_regime import build_options_regime
from analytics.options_scanner import build_fno_options_universe, stock_options_universe, lightweight_stock_options_scan, build_heavy_analysis_queue
from analytics.heavy_options import run_heavy_queue_sample, score_heavy_stock_options, run_heavy_queue_batch
from analytics.master_score import build_master_institutional_score, build_phase7_core_funnel

def normalize_lightweight_options_scores(df):
    """Cross-sectional normalization kept locally as a deployment hotfix."""
    if df is None or df.empty:
        return pd.DataFrame()

    out=df.copy()

    pcr=pd.to_numeric(out["PCR_OI"],errors="coerce")
    out["PCR_Pct"]=pcr.rank(pct=True,method="average")*100

    put_conc=pd.to_numeric(out["Put_OI_Concentration_%"],errors="coerce")
    call_conc=pd.to_numeric(out["Call_OI_Concentration_%"],errors="coerce")
    out["Concentration_Imbalance"]=put_conc-call_conc
    out["Concentration_Pct"]=out["Concentration_Imbalance"].rank(
        pct=True,method="average"
    )*100

    mp=pd.to_numeric(out["Spot_vs_MaxPain_%"],errors="coerce")
    out["MaxPain_Pct"]=mp.rank(pct=True,method="average")*100

    spot=pd.to_numeric(out["Spot"],errors="coerce")
    call_wall=pd.to_numeric(out["Max_Call_OI_Strike"],errors="coerce")
    put_wall=pd.to_numeric(out["Max_Put_OI_Strike"],errors="coerce")

    call_dist=(call_wall/spot-1.0)*100
    put_dist=(spot/put_wall-1.0)*100
    geometry_raw=put_dist-call_dist

    out["Wall_Geometry_Raw"]=geometry_raw
    out["Wall_Geometry_Pct"]=geometry_raw.rank(
        pct=True,method="average"
    )*100

    liquidity=pd.to_numeric(out["Liquidity_Score"],errors="coerce").fillna(50)
    coverage=pd.to_numeric(out["Chain_Coverage_%"],errors="coerce").fillna(50)
    out["Quality_Score"]=(0.70*liquidity+0.30*coverage).clip(0,100)

    for c in ["PCR_Pct","Concentration_Pct","MaxPain_Pct","Wall_Geometry_Pct"]:
        out[c]=pd.to_numeric(out[c],errors="coerce").fillna(50)

    out["Normalized_Options_Score"]=(
        0.25*out["PCR_Pct"]+
        0.20*out["Concentration_Pct"]+
        0.20*out["MaxPain_Pct"]+
        0.20*out["Wall_Geometry_Pct"]+
        0.15*out["Quality_Score"]
    ).round(1)

    out["Normalized_Options_Bias"]=np.select(
        [
            out["Normalized_Options_Score"]>=65,
            out["Normalized_Options_Score"]<=35
        ],
        ["BULLISH","BEARISH"],
        default="NEUTRAL"
    )

    out["Options_CrossSection_Rank"]=out["Normalized_Options_Score"].rank(
        ascending=False,method="dense"
    ).astype(int)

    return out.sort_values(
        ["Normalized_Options_Score","Quality_Score"],
        ascending=False
    ).reset_index(drop=True)


def build_phase7c_confirmed_ranking(core_df, heavy_options_df):
    """Phase 7C local deployment-safe merge/scoring helper."""
    if core_df is None or core_df.empty:
        return pd.DataFrame()
    if heavy_options_df is None or heavy_options_df.empty:
        return pd.DataFrame()

    core=core_df.copy()
    opt=heavy_options_df.copy()

    core["Stock"]=core["Stock"].astype(str).str.upper().str.strip()
    opt["Stock"]=opt["Stock"].astype(str).str.upper().str.strip()

    if "P7_Options_Score" not in opt.columns:
        if "Heavy_Options_Score" in opt.columns:
            opt["P7_Options_Score"]=pd.to_numeric(
                opt["Heavy_Options_Score"],errors="coerce"
            )
        else:
            opt["P7_Options_Score"]=np.nan

    keep_opt=["Stock","P7_Options_Score"]
    for c in [
        "Heavy_Options_Score","Heavy_Options_Bias",
        "Lightweight_Heavy_Alignment","Heavy_Conviction",
        "Heavy_Data_Quality","Status","Gamma_Regime",
        "Zero_Gamma_Level","Dealer_Vanna_1vol",
        "Dealer_Charm_1day","Final_Options_Classification",
        "Phase7_Options_Bias","Phase7_Options_Strength",
        "Options_Rejection_Flags"
    ]:
        if c in opt.columns:
            keep_opt.append(c)

    opt=opt[keep_opt].drop_duplicates("Stock")
    m=core.merge(opt,on="Stock",how="inner")

    required=[
        "P7_RF_Score","P7_Sector_RS_Score",
        "P7_Stock_RS_Score","P7_Futures_Score",
        "P7_Options_Score"
    ]

    for c in required:
        m[c]=pd.to_numeric(m[c],errors="coerce")

    m["P7C_Data_Complete"]=m[required].notna().all(axis=1)

    m["Final_Phase7_Score"]=np.nan
    ok=m["P7C_Data_Complete"]

    m.loc[ok,"Final_Phase7_Score"]=(
        0.30*m.loc[ok,"P7_RF_Score"]+
        0.20*m.loc[ok,"P7_Sector_RS_Score"]+
        0.20*m.loc[ok,"P7_Stock_RS_Score"]+
        0.15*m.loc[ok,"P7_Futures_Score"]+
        0.15*m.loc[ok,"P7_Options_Score"]
    ).round(1)

    decisions=[]
    vetoes=[]
    confirmations=[]

    for _,r in m.iterrows():
        score=r["Final_Phase7_Score"]
        veto=[]

        if not bool(r["P7C_Data_Complete"]):
            veto.append("INCOMPLETE_DATA")

        if str(r.get("Heavy_Data_Quality","")) not in ["","nan","PASS"]:
            veto.append("OPTIONS_DATA_QUALITY")

        if str(r.get("Status","")) not in ["","nan","OK"]:
            veto.append("OPTIONS_ENGINE_ERROR")

        opt_class=str(r.get("Final_Options_Classification",""))
        opt_align=str(r.get("Lightweight_Heavy_Alignment",""))
        fut_sig=str(r.get("Institutional_Signal",""))

        if pd.notna(score):
            if score>=70 and "SHORT" in opt_class:
                veto.append("OPTIONS_OPPOSE_LONG")
            if score<=30 and "LONG" in opt_class:
                veto.append("OPTIONS_OPPOSE_SHORT")
            if score>=70 and "SHORT" in fut_sig:
                veto.append("FUTURES_OPPOSE_LONG")
            if score<=30 and "LONG" in fut_sig:
                veto.append("FUTURES_OPPOSE_SHORT")

        if veto:
            decision="AVOID"
        elif score>=75:
            decision="LONG"
        elif score>=62:
            decision="LONG WATCH"
        elif score<=25:
            decision="SHORT"
        elif score<=38:
            decision="SHORT WATCH"
        else:
            decision="NEUTRAL"

        if "CONFIRMED BULLISH" in opt_align:
            confirm="OPTIONS CONFIRM LONG"
        elif "CONFIRMED BEARISH" in opt_align:
            confirm="OPTIONS CONFIRM SHORT"
        elif "MIXED" in opt_align:
            confirm="OPTIONS MIXED"
        elif "REJECTED" in opt_align:
            confirm="OPTIONS REJECT"
        else:
            confirm="OPTIONS UNCLASSIFIED"

        decisions.append(decision)
        vetoes.append(" | ".join(veto))
        confirmations.append(confirm)

    m["Final_Phase7_Decision"]=decisions
    m["Final_Phase7_Veto_Flags"]=vetoes
    m["Options_Confirmation_State"]=confirmations

    m["Final_Phase7_Conviction"]=np.where(
        m["P7C_Data_Complete"],
        ((m["Final_Phase7_Score"]-50).abs()*2).clip(0,100),
        np.nan
    )
    m["Final_Phase7_Conviction"]=pd.to_numeric(
        m["Final_Phase7_Conviction"],errors="coerce"
    ).round(1)

    m=m.sort_values(
        ["P7C_Data_Complete","Final_Phase7_Score","Stock"],
        ascending=[False,False,True],
        na_position="last"
    ).reset_index(drop=True)

    m["Final_Phase7_Rank"]=np.where(
        m["P7C_Data_Complete"],
        m["Final_Phase7_Score"].rank(
            ascending=False,method="min"
        ),
        np.nan
    )

    return m


def build_final_options_confirmation_ranking(heavy_df):
    """Deployment-safe local Phase 6F.4D ranking helper."""
    if heavy_df is None or heavy_df.empty:
        return pd.DataFrame()

    df=heavy_df.copy()

    score=pd.to_numeric(df["Heavy_Options_Score"],errors="coerce")
    conviction=pd.to_numeric(
        df.get("Heavy_Conviction",pd.Series(0,index=df.index)),
        errors="coerce"
    ).fillna(0)
    quality=df.get(
        "Heavy_Data_Quality",pd.Series("",index=df.index)
    ).astype(str)
    status=df.get(
        "Status",pd.Series("",index=df.index)
    ).astype(str)
    alignment=df.get(
        "Lightweight_Heavy_Alignment",
        pd.Series("",index=df.index)
    ).astype(str)

    classifications=[]
    phase7_bias=[]
    rejection_flags=[]

    for i in df.index:
        s=score.loc[i]
        c=conviction.loc[i]
        q_ok=quality.loc[i]=="PASS"
        st_ok=status.loc[i]=="OK"
        a=alignment.loc[i]

        flags=[]
        if not q_ok:
            flags.append("DATA_QUALITY")
        if not st_ok:
            flags.append("ENGINE_ERROR")
        if "REJECTED" in a:
            flags.append("LIGHT_HEAVY_CONFLICT")

        if (not q_ok) or (not st_ok) or pd.isna(s):
            cls="NEUTRAL"
            bias=0
        elif s>=65 and a.startswith("CONFIRMED") and c>=50:
            cls="CONFIRMED LONG"
            bias=2
        elif s>=55:
            cls="CONDITIONAL LONG"
            bias=1
        elif s<=35 and a.startswith("CONFIRMED") and c>=50:
            cls="CONFIRMED SHORT"
            bias=-2
        elif s<=45:
            cls="CONDITIONAL SHORT"
            bias=-1
        else:
            cls="NEUTRAL"
            bias=0

        classifications.append(cls)
        phase7_bias.append(bias)
        rejection_flags.append(" | ".join(flags) if flags else "")

    df["Final_Options_Classification"]=classifications
    df["Phase7_Options_Bias"]=phase7_bias
    df["Options_Rejection_Flags"]=rejection_flags
    df["Phase7_Options_Strength"]=(
        (score-50.0).abs()*2.0
    ).clip(0,100).round(1)

    order={
        "CONFIRMED LONG":0,
        "CONDITIONAL LONG":1,
        "NEUTRAL":2,
        "CONDITIONAL SHORT":3,
        "CONFIRMED SHORT":4,
    }

    df["_class_order"]=df[
        "Final_Options_Classification"
    ].map(order).fillna(9)

    df=df.sort_values(
        ["_class_order","Heavy_Options_Score","Heavy_Conviction"],
        ascending=[True,False,False]
    ).drop(columns=["_class_order"]).reset_index(drop=True)

    df["Final_Options_Rank"]=range(1,len(df)+1)

    return df


def build_phase7d_conflict_conviction(phase7c_df):
    """
    Phase 7D decision refinement.

    Preserves Final_Phase7_Score unchanged.
    Separates:
      HARD VETO  -> invalid/unreliable setup
      SOFT CONFLICT -> valid setup with opposing evidence and conviction penalty

    Penalties are explicit and auditable.
    """
    if phase7c_df is None or phase7c_df.empty:
        return pd.DataFrame()

    df=phase7c_df.copy()

    score=pd.to_numeric(df["Final_Phase7_Score"],errors="coerce")
    base_conv=((score-50).abs()*2).clip(0,100)

    hard_flags=[]
    soft_flags=[]
    penalties=[]
    final_conv=[]
    grades=[]
    actions=[]

    for i,r in df.iterrows():
        s=score.loc[i]
        hard=[]
        soft=[]
        penalty=0.0

        # --------------------
        # HARD VETOES
        # --------------------
        if not bool(r.get("P7C_Data_Complete",False)):
            hard.append("INCOMPLETE_DATA")

        if str(r.get("Heavy_Data_Quality","")) not in ["","nan","PASS"]:
            hard.append("OPTIONS_DATA_QUALITY")

        if str(r.get("Status","")) not in ["","nan","OK"]:
            hard.append("OPTIONS_ENGINE_ERROR")

        # Explicit Phase-6 rejection remains hard.
        align=str(r.get("Lightweight_Heavy_Alignment",""))
        if "REJECTED" in align:
            hard.append("OPTIONS_REJECTED")

        # --------------------
        # SOFT CONFLICTS
        # --------------------
        fut_sig=str(r.get("Institutional_Signal","")).upper()
        opt_class=str(r.get("Final_Options_Classification","")).upper()

        long_setup=pd.notna(s) and s>=62
        short_setup=pd.notna(s) and s<=38

        if long_setup and "SHORT" in fut_sig:
            soft.append("FUTURES_OPPOSE_LONG")
            penalty += 15

        if short_setup and "LONG" in fut_sig:
            soft.append("FUTURES_OPPOSE_SHORT")
            penalty += 15

        if long_setup and "SHORT" in opt_class:
            soft.append("OPTIONS_OPPOSE_LONG")
            penalty += 20

        if short_setup and "LONG" in opt_class:
            soft.append("OPTIONS_OPPOSE_SHORT")
            penalty += 20

        # Mixed options is uncertainty, not directional rejection.
        if "MIXED" in align:
            soft.append("OPTIONS_MIXED")
            penalty += 7.5

        # Cap total soft penalty.
        penalty=min(penalty,40.0)

        bc=float(base_conv.loc[i]) if pd.notna(base_conv.loc[i]) else 0.0
        fc=max(0.0,bc-penalty)

        if hard:
            grade="HARD VETO"
            action="AVOID"
            fc=0.0
        else:
            if fc>=70:
                grade="HIGH CONVICTION"
            elif fc>=50:
                grade="CONFIRMED"
            elif fc>=30:
                grade="CONDITIONAL"
            else:
                grade="LOW CONVICTION"

            if pd.isna(s):
                action="AVOID"
            elif s>=75:
                action="LONG" if fc>=50 else "LONG WATCH"
            elif s>=62:
                action="LONG WATCH"
            elif s<=25:
                action="SHORT" if fc>=50 else "SHORT WATCH"
            elif s<=38:
                action="SHORT WATCH"
            else:
                action="NEUTRAL"

        hard_flags.append(" | ".join(hard))
        soft_flags.append(" | ".join(soft))
        penalties.append(round(penalty,1))
        final_conv.append(round(fc,1))
        grades.append(grade)
        actions.append(action)

    df["P7D_Hard_Veto_Flags"]=hard_flags
    df["P7D_Soft_Conflict_Flags"]=soft_flags
    df["P7D_Conflict_Penalty"]=penalties
    df["P7D_Adjusted_Conviction"]=final_conv
    df["P7D_Conviction_Grade"]=grades
    df["P7D_Final_Action"]=actions

    # Score is intentionally NOT modified.
    df["P7D_Institutional_Score"]=score

    # Rank tradeable candidates first, then by score/conviction.
    action_order={
        "LONG":0,
        "LONG WATCH":1,
        "NEUTRAL":2,
        "SHORT WATCH":3,
        "SHORT":4,
        "AVOID":5,
    }
    df["_action_order"]=df["P7D_Final_Action"].map(action_order).fillna(9)

    df=df.sort_values(
        ["_action_order","P7D_Adjusted_Conviction","P7D_Institutional_Score"],
        ascending=[True,False,False],
        na_position="last"
    ).drop(columns=["_action_order"]).reset_index(drop=True)

    df["P7D_Rank"]=range(1,len(df)+1)

    return df


def build_phase7e_live_entry_gate(
    phase7d_df,
    live_df,
):
    """
    Phase 7F.2 timing gate.

    READY now requires ALL of:
      1) Phase 7D candidate is tradeable
      2) Live RF confirms direction
      3) Same-time RVOL >= 0.75x
      4) At least 3 timing evidence fields
      5) Timing score >= 75%
      6) No hard intraday invalidation

    Low participation causes WAIT, not INVALIDATED.
    """
    if phase7d_df is None or phase7d_df.empty:
        return pd.DataFrame()
    if live_df is None or live_df.empty:
        return pd.DataFrame()

    p=phase7d_df.copy()
    l=live_df.copy()

    p["Stock"]=p["Stock"].astype(str).str.upper().str.strip()
    l["Stock"]=l["Stock"].astype(str).str.upper().str.strip()

    m=p.merge(l,on="Stock",how="left",suffixes=("","_Live"))

    num_cols=[
        "Live_RF","LTP","VWAP","Day_Volume",
        "Avg_Volume_Same_Time","RVOL_Same_Time",
        "RVOL_Baseline_Sessions","IB_High","IB_Low",
        "Open","Day_High","Day_Low"
    ]

    for c in num_cols:
        if c in m.columns:
            m[c]=pd.to_numeric(m[c],errors="coerce")

    states=[]
    timing_scores=[]
    reasons=[]
    directions=[]
    participation_states=[]

    for _,r in m.iterrows():
        action=str(r.get("P7D_Final_Action",""))

        direction=(
            "LONG"
            if action.startswith("LONG")
            else "SHORT"
            if action.startswith("SHORT")
            else "NONE"
        )

        directions.append(direction)

        if action=="AVOID" or direction=="NONE":
            states.append("INVALIDATED")
            timing_scores.append(0.0)
            reasons.append("Phase 7D candidate is not tradeable")
            participation_states.append("NOT APPLICABLE")
            continue

        evidence=0
        passed=0
        fail_hard=False
        rf_confirmed=False
        participation_ok=False
        why=[]

        rf=r.get("Live_RF",np.nan)
        ltp=r.get("LTP",np.nan)
        vwap=r.get("VWAP",np.nan)
        vol=r.get("Day_Volume",np.nan)
        avgvol=r.get("Avg_Volume_Same_Time",np.nan)
        rvol=r.get("RVOL_Same_Time",np.nan)
        ibh=r.get("IB_High",np.nan)
        ibl=r.get("IB_Low",np.nan)

        # 1) RF confirmation - mandatory.
        if pd.notna(rf):
            evidence+=1

            if (
                (direction=="LONG" and rf>0)
                or (direction=="SHORT" and rf<0)
            ):
                passed+=1
                rf_confirmed=True
                why.append("RF confirms")
            elif (
                (direction=="LONG" and rf<=-4)
                or (direction=="SHORT" and rf>=4)
            ):
                fail_hard=True
                why.append("RF strongly opposes")
            else:
                why.append("RF not confirmed")
        else:
            why.append("RF unavailable")

        # 2) VWAP location.
        if pd.notna(ltp) and pd.notna(vwap) and vwap!=0:
            evidence+=1

            if (
                (direction=="LONG" and ltp>vwap)
                or (direction=="SHORT" and ltp<vwap)
            ):
                passed+=1
                why.append("VWAP confirms")
            else:
                why.append("VWAP not confirmed")

        # 3) True same-time RVOL - mandatory minimum for READY.
        if pd.isna(rvol):
            if (
                pd.notna(vol)
                and pd.notna(avgvol)
                and avgvol>0
            ):
                rvol=vol/avgvol

        participation_state="UNAVAILABLE"

        if pd.notna(rvol):
            evidence+=1

            if rvol>=1.25:
                passed+=1
                participation_ok=True
                participation_state="STRONG"
                why.append(f"Strong RVOL ({rvol:.2f}x)")
            elif rvol>=1.00:
                passed+=1
                participation_ok=True
                participation_state="CONFIRMED"
                why.append(f"Confirmed RVOL ({rvol:.2f}x)")
            elif rvol>=0.75:
                participation_ok=True
                participation_state="ACCEPTABLE"
                why.append(f"Acceptable RVOL ({rvol:.2f}x)")
            else:
                participation_state="POOR"
                why.append(f"Poor RVOL ({rvol:.2f}x)")
        else:
            why.append("RVOL unavailable")

        # 4) Initial Balance structure.
        if pd.notna(ltp) and pd.notna(ibh) and pd.notna(ibl):
            evidence+=1

            if direction=="LONG":
                if ltp>ibh:
                    passed+=1
                    why.append("Above IB high")
                elif ltp<ibl:
                    fail_hard=True
                    why.append("Below IB low")
                else:
                    why.append("Inside IB")
            else:
                if ltp<ibl:
                    passed+=1
                    why.append("Below IB low")
                elif ltp>ibh:
                    fail_hard=True
                    why.append("Above IB high")
                else:
                    why.append("Inside IB")

        timing=(passed/evidence*100.0) if evidence else 0.0

        if fail_hard:
            state="INVALIDATED"
        elif not rf_confirmed:
            state="WAIT"
            why.append("READY blocked: RF confirmation mandatory")
        elif not participation_ok:
            state="WAIT"
            why.append("READY blocked: RVOL must be >=0.75x")
        elif evidence<3:
            state="WAIT"
            why.append("Insufficient live timing evidence")
        elif timing>=75:
            state="READY"
        else:
            state="WAIT"

        states.append(state)
        timing_scores.append(round(timing,1))
        reasons.append(" | ".join(why))
        participation_states.append(participation_state)

    m["P7E_Direction"]=directions
    m["P7E_Timing_Score"]=timing_scores
    m["P7E_Participation_State"]=participation_states
    m["P7E_Entry_State"]=states
    m["P7E_Why"]=reasons

    order={"READY":0,"WAIT":1,"INVALIDATED":2}

    m["_entry_order"]=m["P7E_Entry_State"].map(order).fillna(9)

    m=m.sort_values(
        [
            "_entry_order",
            "P7D_Adjusted_Conviction",
            "P7D_Institutional_Score"
        ],
        ascending=[True,False,False],
        na_position="last"
    ).drop(columns=["_entry_order"]).reset_index(drop=True)

    return m

def build_phase7g_state_transitions(current_gate, previous_gate=None):
    """
    Phase 7G candidate monitor.

    Compares the latest Phase 7F.2 entry-gate snapshot with the previous one and
    classifies each stock's state transition:
      NEW
      UNCHANGED
      WAIT -> READY
      READY -> WAIT
      READY -> INVALIDATED
      WAIT -> INVALIDATED
      INVALIDATED -> WAIT
      INVALIDATED -> READY
      other state changes

    Returns:
      current_enriched, transition_log
    """
    if current_gate is None or current_gate.empty:
        return pd.DataFrame(), pd.DataFrame()

    cur = current_gate.copy()
    cur["Stock"] = cur["Stock"].astype(str).str.upper().str.strip()

    keep = [
        "Stock",
        "P7E_Entry_State",
        "P7E_Timing_Score",
        "P7E_Participation_State",
        "P7D_Final_Action",
        "P7D_Institutional_Score",
        "P7D_Adjusted_Conviction",
        "Live_RF",
        "RVOL_Same_Time",
        "LTP",
        "VWAP",
        "IB_High",
        "IB_Low",
        "P7E_Why",
        "Live_Data_Status",
    ]

    for c in keep:
        if c not in cur.columns:
            cur[c] = np.nan

    now_ts = pd.Timestamp.now()
    cur["P7G_Snapshot_Time"] = now_ts

    if previous_gate is None or previous_gate.empty:
        cur["P7G_Previous_State"] = "NONE"
        cur["P7G_Transition"] = "NEW"
        cur["P7G_State_Changed"] = True

        log = cur.copy()
        log["P7G_New_State"] = log["P7E_Entry_State"]
        log["P7G_Transition_Time"] = now_ts

        log_cols = [
            "P7G_Transition_Time","Stock","P7G_Previous_State",
            "P7G_New_State","P7G_Transition",
            "P7D_Final_Action","P7D_Institutional_Score",
            "P7D_Adjusted_Conviction","P7E_Timing_Score",
            "P7E_Participation_State","Live_RF","RVOL_Same_Time",
            "LTP","VWAP","IB_High","IB_Low","P7E_Why"
        ]

        return cur, log[[c for c in log_cols if c in log.columns]]

    prev = previous_gate.copy()
    prev["Stock"] = prev["Stock"].astype(str).str.upper().str.strip()

    prev_state = prev[["Stock","P7E_Entry_State"]].rename(
        columns={"P7E_Entry_State":"P7G_Previous_State"}
    )

    cur = cur.merge(prev_state, on="Stock", how="left")
    cur["P7G_Previous_State"] = cur["P7G_Previous_State"].fillna("NONE")

    transitions = []
    changed = []

    for _, r in cur.iterrows():
        old = str(r.get("P7G_Previous_State","NONE"))
        new = str(r.get("P7E_Entry_State",""))

        if old == "NONE":
            tr = "NEW"
            ch = True
        elif old == new:
            tr = "UNCHANGED"
            ch = False
        else:
            tr = f"{old} -> {new}"
            ch = True

        transitions.append(tr)
        changed.append(ch)

    cur["P7G_Transition"] = transitions
    cur["P7G_State_Changed"] = changed

    changed_df = cur[cur["P7G_State_Changed"]].copy()
    changed_df["P7G_New_State"] = changed_df["P7E_Entry_State"]
    changed_df["P7G_Transition_Time"] = now_ts

    log_cols = [
        "P7G_Transition_Time","Stock","P7G_Previous_State",
        "P7G_New_State","P7G_Transition",
        "P7D_Final_Action","P7D_Institutional_Score",
        "P7D_Adjusted_Conviction","P7E_Timing_Score",
        "P7E_Participation_State","Live_RF","RVOL_Same_Time",
        "LTP","VWAP","IB_High","IB_Low","P7E_Why"
    ]

    log = changed_df[[c for c in log_cols if c in changed_df.columns]].copy()

    return cur, log


def build_phase7h_paper_trades(
    monitored_df,
    existing_trades=None,
    risk_per_trade=1000.0,
    target_r_multiple=2.0,
):
    """
    Phase 7H paper-trading engine.

    Opens a simulated trade only when:
      - Phase 7G / 7F.2 state is READY
      - Phase 7D action is LONG/LONG WATCH/SHORT/SHORT WATCH
      - no active paper trade already exists for the stock

    Entry:
      current LTP

    Stop:
      LONG  -> min(VWAP, IB_Low) when available
      SHORT -> max(VWAP, IB_High) when available

    Target:
      Entry +/- target_r_multiple * initial risk

    Quantity:
      floor(risk_per_trade / risk_per_share)

    No broker orders are placed.
    """
    if monitored_df is None or monitored_df.empty:
        return pd.DataFrame()

    cur = monitored_df.copy()
    cur["Stock"] = cur["Stock"].astype(str).str.upper().str.strip()

    if existing_trades is None or existing_trades.empty:
        trades = pd.DataFrame()
    else:
        trades = existing_trades.copy()

    active_stocks = set()
    if not trades.empty and "Paper_Status" in trades.columns:
        active_stocks = set(
            trades.loc[
                trades["Paper_Status"] == "OPEN",
                "Stock"
            ].astype(str).str.upper().str.strip()
        )

    new_rows = []

    for _, r in cur.iterrows():
        stock = str(r.get("Stock","")).upper().strip()
        state = str(r.get("P7E_Entry_State",""))
        action = str(r.get("P7D_Final_Action",""))
        ltp = pd.to_numeric(pd.Series([r.get("LTP")]), errors="coerce").iloc[0]
        vwap = pd.to_numeric(pd.Series([r.get("VWAP")]), errors="coerce").iloc[0]
        ibh = pd.to_numeric(pd.Series([r.get("IB_High")]), errors="coerce").iloc[0]
        ibl = pd.to_numeric(pd.Series([r.get("IB_Low")]), errors="coerce").iloc[0]

        if (
            state != "READY"
            or stock in active_stocks
            or pd.isna(ltp)
        ):
            continue

        if action.startswith("LONG"):
            side = "LONG"

            stop_candidates = [
                x for x in [vwap, ibl]
                if pd.notna(x) and x < ltp
            ]

            if not stop_candidates:
                continue

            stop = min(stop_candidates)
            risk_per_share = ltp - stop

            if risk_per_share <= 0:
                continue

            target = ltp + float(target_r_multiple) * risk_per_share

        elif action.startswith("SHORT"):
            side = "SHORT"

            stop_candidates = [
                x for x in [vwap, ibh]
                if pd.notna(x) and x > ltp
            ]

            if not stop_candidates:
                continue

            stop = max(stop_candidates)
            risk_per_share = stop - ltp

            if risk_per_share <= 0:
                continue

            target = ltp - float(target_r_multiple) * risk_per_share

        else:
            continue

        qty = int(float(risk_per_trade) // risk_per_share)

        if qty < 1:
            continue

        new_rows.append({
            "Trade_ID": f'{stock}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}',
            "Open_Time": pd.Timestamp.now(),
            "Stock": stock,
            "Paper_Side": side,
            "Entry_Price": round(float(ltp), 2),
            "Stop_Price": round(float(stop), 2),
            "Target_Price": round(float(target), 2),
            "Initial_Risk_Per_Share": round(float(risk_per_share), 4),
            "Risk_Budget": round(float(risk_per_trade), 2),
            "Quantity": qty,
            "Target_R_Multiple": float(target_r_multiple),
            "P7D_Score": r.get("P7D_Institutional_Score"),
            "P7D_Conviction": r.get("P7D_Adjusted_Conviction"),
            "P7E_Timing_Score": r.get("P7E_Timing_Score"),
            "Live_RF": r.get("Live_RF"),
            "RVOL_Same_Time": r.get("RVOL_Same_Time"),
            "Paper_Status": "OPEN",
            "Exit_Time": pd.NaT,
            "Exit_Price": np.nan,
            "Exit_Reason": "",
            "Realized_PnL": np.nan,
            "Realized_R": np.nan,
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        trades = pd.concat([trades, new_df], ignore_index=True)

    return trades


def update_phase7h_paper_trades(
    trades_df,
    monitored_df,
):
    """
    Mark open paper trades exited if:
      - stop is hit
      - target is hit
      - Phase 7E becomes INVALIDATED

    Uses current LTP only; no intrabar sequencing assumption is made.
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    trades = trades_df.copy()
    live = monitored_df.copy()

    live["Stock"] = live["Stock"].astype(str).str.upper().str.strip()

    live_map = live.set_index("Stock").to_dict("index")

    for i, r in trades.iterrows():
        if str(r.get("Paper_Status","")) != "OPEN":
            continue

        stock = str(r.get("Stock","")).upper().strip()

        if stock not in live_map:
            continue

        lv = live_map[stock]

        ltp = pd.to_numeric(
            pd.Series([lv.get("LTP")]),
            errors="coerce"
        ).iloc[0]

        if pd.isna(ltp):
            continue

        side = str(r.get("Paper_Side",""))
        stop = float(r.get("Stop_Price"))
        target = float(r.get("Target_Price"))
        entry = float(r.get("Entry_Price"))
        qty = int(r.get("Quantity"))
        state = str(lv.get("P7E_Entry_State",""))

        exit_reason = None
        exit_price = None

        if side == "LONG":
            if ltp <= stop:
                exit_reason = "STOP"
                exit_price = stop
            elif ltp >= target:
                exit_reason = "TARGET"
                exit_price = target
            elif state == "INVALIDATED":
                exit_reason = "SETUP INVALIDATED"
                exit_price = float(ltp)

            if exit_reason:
                pnl = (exit_price - entry) * qty
                risk_total = (entry - stop) * qty

        elif side == "SHORT":
            if ltp >= stop:
                exit_reason = "STOP"
                exit_price = stop
            elif ltp <= target:
                exit_reason = "TARGET"
                exit_price = target
            elif state == "INVALIDATED":
                exit_reason = "SETUP INVALIDATED"
                exit_price = float(ltp)

            if exit_reason:
                pnl = (entry - exit_price) * qty
                risk_total = (stop - entry) * qty

        else:
            continue

        if exit_reason:
            trades.at[i, "Paper_Status"] = "CLOSED"
            trades.at[i, "Exit_Time"] = pd.Timestamp.now()
            trades.at[i, "Exit_Price"] = round(float(exit_price), 2)
            trades.at[i, "Exit_Reason"] = exit_reason
            trades.at[i, "Realized_PnL"] = round(float(pnl), 2)
            trades.at[i, "Realized_R"] = round(
                float(pnl / risk_total),
                3
            ) if risk_total > 0 else np.nan

    return trades


def apply_phase7h1_portfolio_risk_gate(
    monitored_df,
    existing_trades=None,
    risk_per_trade=1000.0,
    target_r_multiple=2.0,
    max_open_positions=5,
    max_portfolio_heat=5000.0,
    max_sector_positions=2,
):
    """
    Phase 7H.1 portfolio-risk gate for PAPER TRADES ONLY.

    Admission order:
      1) READY candidates only
      2) highest adjusted conviction first
      3) max concurrent open positions
      4) max total open risk / portfolio heat
      5) max open positions per sector
      6) no duplicate active stock

    Returns:
      updated_trades, admission_log
    """
    if monitored_df is None or monitored_df.empty:
        return (
            existing_trades.copy()
            if existing_trades is not None
            else pd.DataFrame(),
            pd.DataFrame(),
        )

    cur=monitored_df.copy()
    cur["Stock"]=cur["Stock"].astype(str).str.upper().str.strip()

    if existing_trades is None or existing_trades.empty:
        trades=pd.DataFrame()
    else:
        trades=existing_trades.copy()

    # Existing open portfolio state.
    if trades.empty:
        open_trades=pd.DataFrame()
    else:
        open_trades=trades[
            trades["Paper_Status"]=="OPEN"
        ].copy()

    active_stocks=set(
        open_trades["Stock"].astype(str).str.upper().str.strip()
    ) if not open_trades.empty else set()

    open_count=len(open_trades)

    if not open_trades.empty:
        open_heat=pd.to_numeric(
            open_trades.get("Risk_Budget",pd.Series(dtype=float)),
            errors="coerce"
        ).fillna(0).sum()
    else:
        open_heat=0.0

    sector_counts={}
    if not open_trades.empty and "Sector" in open_trades.columns:
        sector_counts=(
            open_trades["Sector"]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
            .to_dict()
        )

    # Highest-conviction READY candidates first.
    ready=cur[
        cur["P7E_Entry_State"]=="READY"
    ].copy()

    if "P7D_Adjusted_Conviction" in ready.columns:
        ready=ready.sort_values(
            ["P7D_Adjusted_Conviction","P7D_Institutional_Score"],
            ascending=[False,False],
            na_position="last"
        )

    admitted_rows=[]
    admission_log=[]

    for _,r in ready.iterrows():
        stock=str(r.get("Stock","")).upper().strip()
        sector=str(r.get("Sector","UNKNOWN"))
        action=str(r.get("P7D_Final_Action",""))
        ltp=pd.to_numeric(
            pd.Series([r.get("LTP")]),errors="coerce"
        ).iloc[0]
        vwap=pd.to_numeric(
            pd.Series([r.get("VWAP")]),errors="coerce"
        ).iloc[0]
        ibh=pd.to_numeric(
            pd.Series([r.get("IB_High")]),errors="coerce"
        ).iloc[0]
        ibl=pd.to_numeric(
            pd.Series([r.get("IB_Low")]),errors="coerce"
        ).iloc[0]

        decision="REJECT"
        reason=""

        if stock in active_stocks:
            reason="DUPLICATE_ACTIVE_STOCK"

        elif open_count >= int(max_open_positions):
            reason="MAX_OPEN_POSITIONS"

        elif open_heat + float(risk_per_trade) > float(max_portfolio_heat):
            reason="MAX_PORTFOLIO_HEAT"

        elif sector_counts.get(sector,0) >= int(max_sector_positions):
            reason="MAX_SECTOR_POSITIONS"

        elif pd.isna(ltp):
            reason="LTP_MISSING"

        else:
            if action.startswith("LONG"):
                side="LONG"
                stop_candidates=[
                    x for x in [vwap,ibl]
                    if pd.notna(x) and x < ltp
                ]
                if not stop_candidates:
                    reason="NO_VALID_LONG_STOP"
                else:
                    stop=min(stop_candidates)
                    risk_per_share=ltp-stop
                    target=ltp+float(target_r_multiple)*risk_per_share

            elif action.startswith("SHORT"):
                side="SHORT"
                stop_candidates=[
                    x for x in [vwap,ibh]
                    if pd.notna(x) and x > ltp
                ]
                if not stop_candidates:
                    reason="NO_VALID_SHORT_STOP"
                else:
                    stop=max(stop_candidates)
                    risk_per_share=stop-ltp
                    target=ltp-float(target_r_multiple)*risk_per_share

            else:
                reason="NON_TRADEABLE_ACTION"

            if reason=="":
                qty=int(float(risk_per_trade)//risk_per_share)

                if qty < 1:
                    reason="QTY_ZERO"
                else:
                    actual_risk=risk_per_share*qty

                    # Use ACTUAL risk for portfolio heat.
                    if open_heat + actual_risk > float(max_portfolio_heat):
                        reason="MAX_PORTFOLIO_HEAT"
                    else:
                        decision="ADMIT"

                        trade={
                            "Trade_ID":f'{stock}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}',
                            "Open_Time":pd.Timestamp.now(),
                            "Stock":stock,
                            "Sector":sector,
                            "Paper_Side":side,
                            "Entry_Price":round(float(ltp),2),
                            "Stop_Price":round(float(stop),2),
                            "Target_Price":round(float(target),2),
                            "Initial_Risk_Per_Share":round(float(risk_per_share),4),
                            "Risk_Budget":round(float(actual_risk),2),
                            "Quantity":qty,
                            "Target_R_Multiple":float(target_r_multiple),
                            "P7D_Score":r.get("P7D_Institutional_Score"),
                            "P7D_Conviction":r.get("P7D_Adjusted_Conviction"),
                            "P7E_Timing_Score":r.get("P7E_Timing_Score"),
                            "Live_RF":r.get("Live_RF"),
                            "RVOL_Same_Time":r.get("RVOL_Same_Time"),
                            "Paper_Status":"OPEN",
                            "Exit_Time":pd.NaT,
                            "Exit_Price":np.nan,
                            "Exit_Reason":"",
                            "Realized_PnL":np.nan,
                            "Realized_R":np.nan,
                        }

                        admitted_rows.append(trade)

                        active_stocks.add(stock)
                        open_count+=1
                        open_heat+=actual_risk
                        sector_counts[sector]=sector_counts.get(sector,0)+1

        admission_log.append({
            "Timestamp":pd.Timestamp.now(),
            "Stock":stock,
            "Sector":sector,
            "P7D_Final_Action":action,
            "P7D_Adjusted_Conviction":r.get("P7D_Adjusted_Conviction"),
            "P7E_Timing_Score":r.get("P7E_Timing_Score"),
            "RVOL_Same_Time":r.get("RVOL_Same_Time"),
            "Decision":decision,
            "Reason":reason if reason else "ADMITTED",
            "Open_Positions_After":open_count,
            "Portfolio_Heat_After":round(float(open_heat),2),
            "Sector_Positions_After":sector_counts.get(sector,0),
        })

    if admitted_rows:
        new_df=pd.DataFrame(admitted_rows)
        trades=pd.concat([trades,new_df],ignore_index=True)

    return trades,pd.DataFrame(admission_log)


def _p8c_num(s):
    return pd.to_numeric(s, errors="coerce")


def summarize_closed_trades(trades):
    if trades is None or trades.empty:
        return pd.DataFrame()

    df=trades.copy()

    if "Paper_Status" in df.columns:
        df=df[df["Paper_Status"].astype(str)=="CLOSED"].copy()

    if "Realized_R" not in df.columns:
        return pd.DataFrame()

    df["Realized_R"]=_p8c_num(df["Realized_R"])
    df=df[df["Realized_R"].notna()].copy()

    if df.empty:
        return pd.DataFrame()

    r=df["Realized_R"]
    wins=r[r>0]
    losses=r[r<0]

    gross_profit=wins.sum()
    gross_loss=abs(losses.sum())

    equity=r.cumsum()
    running_max=equity.cummax()
    drawdown=equity-running_max

    return pd.DataFrame([{
        "Trades":len(df),
        "Wins":int((r>0).sum()),
        "Losses":int((r<0).sum()),
        "Flat":int((r==0).sum()),
        "Win_Rate_%":round((r>0).mean()*100,2),
        "Avg_R":round(r.mean(),3),
        "Median_R":round(r.median(),3),
        "Expectancy_R":round(r.mean(),3),
        "Profit_Factor":(
            round(gross_profit/gross_loss,3)
            if gross_loss>0 else np.nan
        ),
        "Total_R":round(r.sum(),3),
        "Best_R":round(r.max(),3),
        "Worst_R":round(r.min(),3),
        "Max_Drawdown_R":round(drawdown.min(),3),
    }])


def performance_by_group(trades, group_col):
    if (
        trades is None
        or trades.empty
        or group_col not in trades.columns
    ):
        return pd.DataFrame()

    df=trades.copy()

    if "Paper_Status" in df.columns:
        df=df[df["Paper_Status"].astype(str)=="CLOSED"].copy()

    if "Realized_R" not in df.columns:
        return pd.DataFrame()

    df["Realized_R"]=_p8c_num(df["Realized_R"])
    df=df[df["Realized_R"].notna()].copy()

    if df.empty:
        return pd.DataFrame()

    rows=[]

    for key,g in df.groupby(group_col,dropna=False):
        r=g["Realized_R"]
        wins=r[r>0]
        losses=r[r<0]
        gross_profit=wins.sum()
        gross_loss=abs(losses.sum())

        rows.append({
            group_col:key,
            "Trades":len(g),
            "Win_Rate_%":round((r>0).mean()*100,2),
            "Avg_R":round(r.mean(),3),
            "Expectancy_R":round(r.mean(),3),
            "Profit_Factor":(
                round(gross_profit/gross_loss,3)
                if gross_loss>0 else np.nan
            ),
            "Total_R":round(r.sum(),3),
        })

    return pd.DataFrame(rows).sort_values(
        ["Expectancy_R","Trades"],
        ascending=[False,False]
    ).reset_index(drop=True)


def add_research_bands(df):
    if df is None or df.empty:
        return pd.DataFrame()

    out=df.copy()

    if "P7D_Institutional_Score" in out.columns:
        s=_p8c_num(out["P7D_Institutional_Score"])
        out["Score_Band"]=pd.cut(
            s,
            bins=[-np.inf,38,62,75,np.inf],
            labels=["<=38","38-62","62-75",">=75"]
        )

    if "P7D_Adjusted_Conviction" in out.columns:
        c=_p8c_num(out["P7D_Adjusted_Conviction"])
        out["Conviction_Band"]=pd.cut(
            c,
            bins=[-np.inf,30,50,70,np.inf],
            labels=["LOW","CONDITIONAL","CONFIRMED","HIGH"]
        )

    if "RVOL_Same_Time" in out.columns:
        rv=_p8c_num(out["RVOL_Same_Time"])
        out["RVOL_Band"]=pd.cut(
            rv,
            bins=[-np.inf,0.75,1.0,1.25,np.inf],
            labels=["<0.75","0.75-1.00","1.00-1.25",">=1.25"]
        )

    if "P7E_Timing_Score" in out.columns:
        t=_p8c_num(out["P7E_Timing_Score"])
        out["Timing_Band"]=pd.cut(
            t,
            bins=[-np.inf,50,75,90,np.inf],
            labels=["<50","50-75","75-90",">=90"]
        )

    return out


def build_expectancy_dashboard(trades):
    if trades is None or trades.empty:
        return {}

    df=add_research_bands(trades)

    tables={"overall":summarize_closed_trades(df)}

    for col in [
        "Paper_Side",
        "Sector",
        "Score_Band",
        "Conviction_Band",
        "RVOL_Band",
        "Timing_Band",
        "P7D_Final_Action",
        "P7E_Participation_State",
    ]:
        if col in df.columns:
            tables[col]=performance_by_group(df,col)

    return tables


def merge_trade_context(trades, research_snapshots):
    if trades is None or trades.empty:
        return pd.DataFrame()

    t=trades.copy()

    if research_snapshots is None or research_snapshots.empty:
        return add_research_bands(t)

    r=research_snapshots.copy()

    if "Stock" not in r.columns:
        return add_research_bands(t)

    r["Stock"]=r["Stock"].astype(str).str.upper().str.strip()
    t["Stock"]=t["Stock"].astype(str).str.upper().str.strip()

    if "Research_Snapshot_Time" in r.columns:
        r["Research_Snapshot_Time"]=pd.to_datetime(
            r["Research_Snapshot_Time"],
            errors="coerce"
        )
        r=r.sort_values("Research_Snapshot_Time")

    r=r.drop_duplicates("Stock",keep="last")

    context_cols=[
        c for c in [
            "Stock",
            "Sector",
            "P7D_Institutional_Score",
            "P7D_Adjusted_Conviction",
            "P7D_Conviction_Grade",
            "P7_RF_Score",
            "P7_Sector_RS_Score",
            "P7_Stock_RS_Score",
            "P7_Futures_Score",
            "P7_Options_Score",
            "P7E_Timing_Score",
            "P7E_Participation_State",
            "P7E_Entry_State",
            "Live_RF",
            "RVOL_Same_Time",
            "Gamma_Regime",
        ]
        if c in r.columns and (c=="Stock" or c not in t.columns)
    ]

    out=t.merge(
        r[context_cols],
        on="Stock",
        how="left"
    )

    return add_research_bands(out)


def build_phase8e_entry_context_dataset(
    trades_df,
    research_df,
    max_lookback_minutes=60,
):
    """
    Phase 8E:
    Match each paper trade to the nearest research snapshot at or BEFORE entry.

    This fixes the Phase 8C limitation where the latest research row per stock
    could be attached even if it occurred after the trade was opened.

    Returns:
      one enriched row per trade with exact entry-time research context.
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    t=trades_df.copy()

    if "Stock" not in t.columns:
        return pd.DataFrame()

    t["Stock"]=t["Stock"].astype(str).str.upper().str.strip()

    if "Open_Time" not in t.columns:
        return t

    t["Open_Time"]=pd.to_datetime(
        t["Open_Time"],
        errors="coerce"
    )

    if research_df is None or research_df.empty:
        return t

    r=research_df.copy()

    if "Stock" not in r.columns:
        return t

    r["Stock"]=r["Stock"].astype(str).str.upper().str.strip()

    time_col=None
    for c in [
        "Research_Snapshot_Time",
        "snapshot_time",
        "P7G_Snapshot_Time"
    ]:
        if c in r.columns:
            time_col=c
            break

    if time_col is None:
        return t

    r["_Research_Time"]=pd.to_datetime(
        r[time_col],
        errors="coerce"
    )

    r=r[r["_Research_Time"].notna()].copy()
    t=t[t["Open_Time"].notna()].copy()

    if r.empty or t.empty:
        return t

    context_cols=[
        c for c in [
            "Sector",
            "P7D_Institutional_Score",
            "P7D_Final_Action",
            "P7D_Adjusted_Conviction",
            "P7D_Conviction_Grade",
            "P7D_Conflict_Penalty",
            "P7_RF_Score",
            "P7_Sector_RS_Score",
            "P7_Stock_RS_Score",
            "P7_Futures_Score",
            "P7_Options_Score",
            "Institutional_Signal",
            "Lightweight_Heavy_Alignment",
            "Gamma_Regime",
            "Zero_Gamma_Level",
            "P7E_Timing_Score",
            "P7E_Participation_State",
            "P7E_Entry_State",
            "Live_RF",
            "RVOL_Same_Time",
            "LTP",
            "VWAP",
            "IB_High",
            "IB_Low",
            "P7G_Transition",
            "Decision",
            "Reason",
        ]
        if c in r.columns
    ]

    rows=[]

    for _,tr in t.iterrows():
        stock=tr["Stock"]
        entry_time=tr["Open_Time"]

        candidates=r[
            (r["Stock"]==stock)
            & (r["_Research_Time"]<=entry_time)
        ].copy()

        if candidates.empty:
            row=tr.to_dict()
            row["Entry_Context_Status"]="NO_PRIOR_SNAPSHOT"
            row["Entry_Context_Age_Min"]=np.nan
            rows.append(row)
            continue

        candidates=candidates.sort_values(
            "_Research_Time",
            ascending=False
        )

        ctx=candidates.iloc[0]
        age_min=(
            entry_time-ctx["_Research_Time"]
        ).total_seconds()/60.0

        row=tr.to_dict()

        if age_min > float(max_lookback_minutes):
            row["Entry_Context_Status"]="STALE_PRIOR_SNAPSHOT"
        else:
            row["Entry_Context_Status"]="MATCHED"

        row["Entry_Context_Time"]=ctx["_Research_Time"]
        row["Entry_Context_Age_Min"]=round(
            max(0.0,float(age_min)),
            2
        )

        for c in context_cols:
            key=(
                c
                if c not in row
                else f"EntryCtx_{c}"
            )
            row[key]=ctx.get(c)

        rows.append(row)

    out=pd.DataFrame(rows)

    # Stable research bands from ENTRY context.
    score_col=(
        "EntryCtx_P7D_Institutional_Score"
        if "EntryCtx_P7D_Institutional_Score" in out.columns
        else "P7D_Institutional_Score"
        if "P7D_Institutional_Score" in out.columns
        else None
    )

    conviction_col=(
        "EntryCtx_P7D_Adjusted_Conviction"
        if "EntryCtx_P7D_Adjusted_Conviction" in out.columns
        else "P7D_Adjusted_Conviction"
        if "P7D_Adjusted_Conviction" in out.columns
        else None
    )

    rvol_col=(
        "EntryCtx_RVOL_Same_Time"
        if "EntryCtx_RVOL_Same_Time" in out.columns
        else "RVOL_Same_Time"
        if "RVOL_Same_Time" in out.columns
        else None
    )

    timing_col=(
        "EntryCtx_P7E_Timing_Score"
        if "EntryCtx_P7E_Timing_Score" in out.columns
        else "P7E_Timing_Score"
        if "P7E_Timing_Score" in out.columns
        else None
    )

    if score_col:
        s=pd.to_numeric(out[score_col],errors="coerce")
        out["Entry_Score_Band"]=pd.cut(
            s,
            bins=[-np.inf,38,62,75,np.inf],
            labels=["<=38","38-62","62-75",">=75"]
        )

    if conviction_col:
        c=pd.to_numeric(
            out[conviction_col],
            errors="coerce"
        )
        out["Entry_Conviction_Band"]=pd.cut(
            c,
            bins=[-np.inf,30,50,70,np.inf],
            labels=[
                "LOW","CONDITIONAL","CONFIRMED","HIGH"
            ]
        )

    if rvol_col:
        rv=pd.to_numeric(out[rvol_col],errors="coerce")
        out["Entry_RVOL_Band"]=pd.cut(
            rv,
            bins=[-np.inf,0.75,1.0,1.25,np.inf],
            labels=[
                "<0.75","0.75-1.00","1.00-1.25",">=1.25"
            ]
        )

    if timing_col:
        ts=pd.to_numeric(out[timing_col],errors="coerce")
        out["Entry_Timing_Band"]=pd.cut(
            ts,
            bins=[-np.inf,50,75,90,np.inf],
            labels=["<50","50-75","75-90",">=90"]
        )

    return out


def build_phase8e_equity_curve(enriched_trades):
    """
    Build realized R and realized P&L equity curves from CLOSED trades.
    """
    if enriched_trades is None or enriched_trades.empty:
        return pd.DataFrame(), pd.DataFrame()

    df=enriched_trades.copy()

    if "Paper_Status" in df.columns:
        df=df[
            df["Paper_Status"].astype(str)=="CLOSED"
        ].copy()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "Exit_Time" in df.columns:
        df["Exit_Time"]=pd.to_datetime(
            df["Exit_Time"],
            errors="coerce"
        )
        df=df.sort_values(
            ["Exit_Time","Stock"],
            na_position="last"
        )

    df["Realized_R"]=pd.to_numeric(
        df.get("Realized_R"),
        errors="coerce"
    )
    df["Realized_PnL"]=pd.to_numeric(
        df.get("Realized_PnL"),
        errors="coerce"
    )

    df=df[
        df["Realized_R"].notna()
    ].copy()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["Trade_Number"]=range(1,len(df)+1)
    df["Cumulative_R"]=df["Realized_R"].cumsum()
    df["Peak_R"]=df["Cumulative_R"].cummax()
    df["Drawdown_R"]=df["Cumulative_R"]-df["Peak_R"]

    df["Cumulative_PnL"]=df[
        "Realized_PnL"
    ].fillna(0).cumsum()
    df["Peak_PnL"]=df["Cumulative_PnL"].cummax()
    df["Drawdown_PnL"]=(
        df["Cumulative_PnL"]-df["Peak_PnL"]
    )

    summary=pd.DataFrame([{
        "Closed_Trades":len(df),
        "Total_R":round(df["Realized_R"].sum(),3),
        "Average_R":round(df["Realized_R"].mean(),3),
        "Win_Rate_%":round(
            (df["Realized_R"]>0).mean()*100,
            2
        ),
        "Max_Drawdown_R":round(
            df["Drawdown_R"].min(),
            3
        ),
        "Total_Realized_PnL":round(
            df["Realized_PnL"].fillna(0).sum(),
            2
        ),
        "Max_Drawdown_PnL":round(
            df["Drawdown_PnL"].min(),
            2
        ),
    }])

    return df.reset_index(drop=True),summary


def build_phase8f_daily_session_report(
    research_df,
    transition_df=None,
    paper_df=None,
    session_date=None,
):
    """
    Phase 8F daily session summary.

    Uses the automatically journaled Phase 8D datasets to build:
      - one latest row per stock for the selected session
      - session-level KPI summary
      - transition counts
      - paper-trade counts / realized performance when available
    """
    if session_date is None:
        session_date = pd.Timestamp.now().date()

    research = (
        research_df.copy()
        if research_df is not None
        else pd.DataFrame()
    )

    transitions = (
        transition_df.copy()
        if transition_df is not None
        else pd.DataFrame()
    )

    paper = (
        paper_df.copy()
        if paper_df is not None
        else pd.DataFrame()
    )

    def _filter_day(df):
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()

        time_col = None
        for c in [
            "snapshot_time",
            "Research_Snapshot_Time",
            "P7G_Transition_Time",
            "Open_Time",
        ]:
            if c in out.columns:
                time_col = c
                break

        if time_col is None:
            return out

        out["_Session_Time"] = pd.to_datetime(
            out[time_col],
            errors="coerce"
        )

        return out[
            out["_Session_Time"].dt.date == session_date
        ].copy()

    research_day = _filter_day(research)
    transitions_day = _filter_day(transitions)
    paper_day = _filter_day(paper)

    latest = pd.DataFrame()

    if not research_day.empty and "Stock" in research_day.columns:
        research_day["Stock"] = (
            research_day["Stock"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        sort_col = (
            "_Session_Time"
            if "_Session_Time" in research_day.columns
            else None
        )

        if sort_col:
            research_day = research_day.sort_values(sort_col)

        latest = research_day.drop_duplicates(
            "Stock",
            keep="last"
        ).copy()

    ready = 0
    wait = 0
    invalid = 0

    if not latest.empty and "P7E_Entry_State" in latest.columns:
        states = latest["P7E_Entry_State"].astype(str).value_counts()
        ready = int(states.get("READY", 0))
        wait = int(states.get("WAIT", 0))
        invalid = int(states.get("INVALIDATED", 0))

    longs = 0
    shorts = 0

    if not latest.empty and "P7D_Final_Action" in latest.columns:
        actions = latest["P7D_Final_Action"].astype(str)
        longs = int(actions.str.startswith("LONG").sum())
        shorts = int(actions.str.startswith("SHORT").sum())

    transition_changes = 0
    wait_ready = 0
    ready_wait = 0
    ready_invalid = 0

    if not transitions_day.empty and "P7G_Transition" in transitions_day.columns:
        tr = transitions_day["P7G_Transition"].astype(str)
        transition_changes = int((tr != "UNCHANGED").sum())
        wait_ready = int((tr == "WAIT -> READY").sum())
        ready_wait = int((tr == "READY -> WAIT").sum())
        ready_invalid = int((tr == "READY -> INVALIDATED").sum())

    paper_open = 0
    paper_closed = 0
    realized_pnl = 0.0
    realized_r = 0.0

    if not paper_day.empty and "Paper_Status" in paper_day.columns:
        ps = paper_day["Paper_Status"].astype(str)
        paper_open = int((ps == "OPEN").sum())
        paper_closed = int((ps == "CLOSED").sum())

        if "Realized_PnL" in paper_day.columns:
            realized_pnl = float(
                pd.to_numeric(
                    paper_day["Realized_PnL"],
                    errors="coerce"
                ).fillna(0).sum()
            )

        if "Realized_R" in paper_day.columns:
            realized_r = float(
                pd.to_numeric(
                    paper_day["Realized_R"],
                    errors="coerce"
                ).fillna(0).sum()
            )

    summary = pd.DataFrame([{
        "Session_Date": str(session_date),
        "Latest_Stocks": len(latest),
        "READY": ready,
        "WAIT": wait,
        "INVALIDATED": invalid,
        "Long_Bias_Stocks": longs,
        "Short_Bias_Stocks": shorts,
        "State_Changes": transition_changes,
        "WAIT_to_READY": wait_ready,
        "READY_to_WAIT": ready_wait,
        "READY_to_INVALIDATED": ready_invalid,
        "Paper_Open": paper_open,
        "Paper_Closed": paper_closed,
        "Realized_PnL": round(realized_pnl, 2),
        "Realized_R": round(realized_r, 3),
    }])

    return summary, latest, transitions_day, paper_day



def _p9_normalize_side(row):
    """Infer LONG/SHORT from the historical snapshot without using future data."""
    for c in ["Side", "Paper_Side", "P7E_Direction", "Scanner_Group", "P7D_Final_Action", "Institutional_Signal"]:
        if c in row.index and pd.notna(row.get(c)):
            v=str(row.get(c)).upper().strip()
            if any(k in v for k in ["LONG", "BULL"]):
                return "LONG"
            if any(k in v for k in ["SHORT", "BEAR"]):
                return "SHORT"
    return ""


def _p9_prepare_signals(df, start_date=None, end_date=None, cutoff="11:15"):
    if df is None or df.empty:
        return pd.DataFrame(), "No historical signal rows found."

    x=df.copy()
    if "Stock" not in x.columns:
        for c in ["stock","Symbol","Underlying"]:
            if c in x.columns:
                x=x.rename(columns={c:"Stock"})
                break
    if "Stock" not in x.columns:
        return pd.DataFrame(), "Historical dataset must contain Stock/Symbol/Underlying."

    time_col=None
    for c in ["Research_Snapshot_Time","P7G_Snapshot_Time","Timestamp","Open_Time","datetime","date","Date","Trading_Date","Session_Date"]:
        if c in x.columns:
            parsed=pd.to_datetime(x[c],errors="coerce")
            if parsed.notna().any():
                x["_P9_Time"]=parsed
                time_col=c
                break
    if time_col is None:
        return pd.DataFrame(), "Historical dataset needs a date/time column."

    x=x[x["_P9_Time"].notna()].copy()
    x["Stock"]=x["Stock"].astype(str).str.upper().str.strip()
    x["_P9_Date"]=x["_P9_Time"].dt.date

    if start_date is not None:
        x=x[x["_P9_Date"]>=start_date]
    if end_date is not None:
        x=x[x["_P9_Date"]<=end_date]

    cutoff_time=pd.Timestamp(cutoff).time()
    # When true timestamps exist, select the latest snapshot available at/before 11:15.
    has_clock=(x["_P9_Time"].dt.hour!=0).any() or (x["_P9_Time"].dt.minute!=0).any()
    if has_clock:
        x=x[x["_P9_Time"].dt.time<=cutoff_time].copy()
        x=x.sort_values("_P9_Time").drop_duplicates(["_P9_Date","Stock"],keep="last")
    else:
        x=x.sort_values("_P9_Time").drop_duplicates(["_P9_Date","Stock"],keep="last")

    x["P9_Side"]=x.apply(_p9_normalize_side,axis=1)
    x=x[x["P9_Side"].isin(["LONG","SHORT"])].copy()
    return x, "OK"


def _p9_forward_metrics(raw5, session_date, side):
    if raw5 is None or raw5.empty:
        return None
    x=raw5.copy()
    x["date"]=pd.to_datetime(x["date"],errors="coerce")
    x=x[x["date"].dt.date==session_date].sort_values("date").copy()
    if x.empty:
        return None

    # 5-minute candles are start-stamped. 11:10-11:15 is the last fully completed bar at 11:15.
    pre=x[x["date"].dt.time<=pd.Timestamp("11:10").time()].copy()
    post=x[x["date"].dt.time>=pd.Timestamp("11:15").time()].copy()
    if pre.empty or post.empty:
        return None

    entry=float(pre.iloc[-1]["close"])
    if not np.isfinite(entry) or entry<=0:
        return None

    def close_at_or_before(t):
        z=x[x["date"].dt.time<=pd.Timestamp(t).time()]
        return float(z.iloc[-1]["close"]) if not z.empty else np.nan

    c30=close_at_or_before("11:40")   # 11:40-11:45 close = +30m
    c60=close_at_or_before("12:10")   # 12:10-12:15 close = +60m
    c1430=close_at_or_before("14:25") # 14:25-14:30 close
    cclose=close_at_or_before("15:25")# 15:25-15:30 close

    sign=1.0 if side=="LONG" else -1.0
    dret=lambda px: ((float(px)/entry)-1.0)*100.0*sign if pd.notna(px) else np.nan

    if side=="LONG":
        mfe=(float(post["high"].max())/entry-1.0)*100.0
        mae=(float(post["low"].min())/entry-1.0)*100.0
    else:
        mfe=(entry/float(post["low"].min())-1.0)*100.0
        mae=(entry/float(post["high"].max())-1.0)*100.0

    return {
        "P9_Entry_1115":round(entry,4),
        "P9_Ret_30m_%":round(dret(c30),4),
        "P9_Ret_60m_%":round(dret(c60),4),
        "P9_Ret_1430_%":round(dret(c1430),4),
        "P9_Ret_Close_%":round(dret(cclose),4),
        "P9_MFE_%":round(float(mfe),4),
        "P9_MAE_%":round(float(mae),4),
        "P9_Close_Hit":bool(pd.notna(cclose) and dret(cclose)>0),
    }


def _p9_summary(results):
    if results is None or results.empty:
        return pd.DataFrame()
    r=results.copy()
    ret=pd.to_numeric(r.get("P9_Ret_Close_%"),errors="coerce").dropna()
    pos=ret[ret>0].sum()
    neg=abs(ret[ret<0].sum())
    pf=(pos/neg) if neg>0 else np.nan
    days=int(pd.Series(r["_P9_Date"]).nunique()) if "_P9_Date" in r.columns else 0
    hit=float((ret>0).mean()*100) if len(ret) else np.nan
    avg=float(ret.mean()) if len(ret) else np.nan
    med=float(ret.median()) if len(ret) else np.nan
    mfe=float(pd.to_numeric(r.get("P9_MFE_%"),errors="coerce").mean())
    mae=float(pd.to_numeric(r.get("P9_MAE_%"),errors="coerce").mean())
    preliminary=(len(ret)>=50 and days>=10 and hit>=52 and avg>0 and (pd.isna(pf) or pf>=1.10) and mfe>abs(mae))
    return pd.DataFrame([{
        "Trading_Days":days,
        "Signals":int(len(ret)),
        "Close_Hit_Rate_%":round(hit,2) if pd.notna(hit) else np.nan,
        "Avg_Directional_Close_%":round(avg,4) if pd.notna(avg) else np.nan,
        "Median_Directional_Close_%":round(med,4) if pd.notna(med) else np.nan,
        "Profit_Factor_Returns":round(float(pf),3) if pd.notna(pf) else np.nan,
        "Avg_MFE_%":round(mfe,4) if pd.notna(mfe) else np.nan,
        "Avg_MAE_%":round(mae,4) if pd.notna(mae) else np.nan,
        "Preliminary_Engineering_Gate":"PASS" if preliminary else "REVIEW",
    }])

st.set_page_config(page_title="Institutional Market Intelligence",layout="wide")
st.title("Institutional Market Intelligence")
st.caption("Phase 2E — RF + RS + Futures Institutional Alignment")

if "access_token" not in st.session_state:
    st.session_state.access_token = None

kite = get_kite()

with st.sidebar:
    st.header("Kite Session")
    st.link_button("1. Open Zerodha Login",kite.login_url())
    request_token = st.text_input("2. Paste request_token")

    if st.button("Generate Access Token",type="primary"):
        if not request_token:
            st.error("Enter request_token first.")
        else:
            try:
                st.session_state.access_token = exchange_request_token(kite,request_token.strip())
                st.success("Kite session connected.")
            except Exception as e:
                st.error(f"Kite login failed: {e}")

    if st.session_state.access_token:
        st.success("Connected")
        if st.button("Disconnect"):
            st.session_state.access_token = None
            st.rerun()

tabs = st.tabs([
    "Market Regime",
    "RF Scanner",
    "Sector + Stock RS",
    "Futures OI + Basis",
    "Options",
    "Auction Levels",
    "Market Profile",
    "Institutional Setup",
    "Top 20 Scanner",
    "11:15 Execution",
    "Risk & Position Size",
    "System",
    "Phase 7 Master",
    "Phase 8 Journal",
    "Phase 8 Analytics",
    "20-Day Validation"
])

with tabs[0]:
    st.subheader("Market Regime")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("NIFTY","Pending")
    c2.metric("BANKNIFTY","Pending")
    c3.metric("India VIX","Pending")
    c4.metric("Regime","BUILDING")

with tabs[1]:
    st.subheader("Full NSE F&O Rotation Factor Scanner")
    if not st.session_state.access_token:
        st.warning("Connect Kite first.")
    else:
        kite.set_access_token(st.session_state.access_token)

        if st.button("Load F&O Universe",key="rf_load"):
            st.session_state["fno_universe"] = get_fno_equity_universe(kite)

        if "fno_universe" in st.session_state:
            universe = st.session_state["fno_universe"]
            st.metric("F&O Stocks Mapped",len(universe))

            c1,c2 = st.columns(2)
            with c1:
                scan_size = st.selectbox("Stocks to scan",[25,50,100,"ALL"],index=3)
            with c2:
                history_days = st.slider("RF History Days",15,75,45)

            if st.button("Run RF Scan",type="primary"):
                work = universe if scan_size=="ALL" else universe.head(int(scan_size))
                rows=[]
                rf_errors=[]
                progress=st.progress(0)
                status=st.empty()
                total=len(work)
                to_date=dt.datetime.now()
                from_date=to_date-dt.timedelta(days=history_days)

                for n,row in enumerate(work.itertuples(index=False),start=1):
                    status.write(f"RF: {row.Stock} — {n}/{total}")
                    success=False
                    last_error="Unknown error"

                    for attempt in range(1,3):
                        try:
                            raw=pd.DataFrame(kite.historical_data(
                                int(row.instrument_token),from_date,to_date,"30minute",oi=False
                            ))
                            if raw.empty:
                                last_error="No 30-minute historical data returned"
                            else:
                                feat=stock_rf_features(
                                    daily_rf_summary(calculate_intraday_rf(raw))
                                )
                                if feat:
                                    feat["Stock"]=row.Stock
                                    feat["Instrument_Token"]=int(row.instrument_token)
                                    rows.append(feat)
                                    success=True
                                    break
                                last_error="RF feature calculation returned no result"
                        except Exception as e:
                            last_error=str(e)
                        time.sleep(0.75)

                    if not success:
                        rf_errors.append([row.Stock,last_error])

                    progress.progress(n/total)
                    time.sleep(0.38)

                status.empty()
                if rows:
                    st.session_state["rf_result"]=add_rf_rank_scores(pd.DataFrame(rows))
                    st.session_state["rf_requested"]=total
                    st.session_state["rf_errors"]=pd.DataFrame(
                        rf_errors,columns=["Stock","Error"]
                    )

        if "rf_result" in st.session_state:
            r=st.session_state["rf_result"].copy()

            rf_requested=int(st.session_state.get("rf_requested",len(r)))
            rf_success=len(r)
            rf_failed=max(rf_requested-rf_success,0)
            rf_coverage=(rf_success/rf_requested*100) if rf_requested else 0

            c1,c2,c3,c4=st.columns(4)
            c1.metric("RF Requested",rf_requested)
            c2.metric("RF Successful",rf_success)
            c3.metric("RF Failed",rf_failed)
            c4.metric("RF Coverage",f"{rf_coverage:.1f}%")

            if rf_coverage < 95:
                st.error("RF coverage is below 95%. Final institutional signals are blocked until coverage is adequate.")
            else:
                st.success("RF coverage validation passed.")

            if "rf_errors" in st.session_state and not st.session_state["rf_errors"].empty:
                with st.expander("RF Scan Errors"):
                    st.dataframe(st.session_state["rf_errors"],use_container_width=True,hide_index=True)
            bull=r.sort_values("RF_Strength_Score",ascending=False).head(20).reset_index(drop=True)
            bull.insert(0,"Rank",range(1,len(bull)+1))
            bear=r.sort_values("RF_Strength_Score",ascending=True).head(20).reset_index(drop=True)
            bear.insert(0,"Rank",range(1,len(bear)+1))

            st.markdown("### Top 20 Bullish RF")
            st.dataframe(bull[["Rank","Stock","RF_Strength_Score","Latest_RF","Avg_RF_5D","RF_Acceleration"]],use_container_width=True,hide_index=True)

            st.markdown("### Top 20 Bearish RF")
            st.dataframe(bear[["Rank","Stock","RF_Strength_Score","Latest_RF","Avg_RF_5D","RF_Acceleration"]],use_container_width=True,hide_index=True)

with tabs[2]:
    st.subheader("Sector Rotation + Stock Relative Strength")

    if not st.session_state.access_token:
        st.warning("Connect Kite first.")
    else:
        kite.set_access_token(st.session_state.access_token)

        if st.button("Load F&O Universe for RS",key="rs_load"):
            st.session_state["fno_universe"]=get_fno_equity_universe(kite)

        if "fno_universe" in st.session_state:
            universe=st.session_state["fno_universe"].copy()
            sector_map=pd.read_csv(Path(__file__).parent/"data"/"sector_map.csv")
            universe=universe.merge(sector_map,on="Stock",how="left")
            universe["Sector"]=universe["Sector"].fillna("UNKNOWN")

            mapped=(universe["Sector"]!="UNKNOWN").sum()
            sector_total=len(universe)
            sector_coverage=(mapped/sector_total*100) if sector_total else 0
            st.caption(f"Sector mapping coverage: {mapped}/{sector_total} ({sector_coverage:.1f}%)")

            if sector_coverage < 98:
                st.warning("Some F&O stocks are missing sector mapping.")
                with st.expander("Missing Sector Mapping"):
                    st.dataframe(
                        universe.loc[universe["Sector"]=="UNKNOWN",["Stock"]],
                        use_container_width=True,hide_index=True
                    )

            c1,c2=st.columns(2)
            with c1:
                rs_scan_size=st.selectbox("RS stocks to scan",[25,50,100,"ALL"],index=3)
            with c2:
                rs_days=st.slider("RS calendar history days",60,150,100)

            if st.button("Run Stock + Sector RS",type="primary"):
                work=universe if rs_scan_size=="ALL" else universe.head(int(rs_scan_size))
                rows=[]
                to_date=dt.datetime.now()
                from_date=to_date-dt.timedelta(days=rs_days)

                nifty=pd.DataFrame(kite.historical_data(
                    256265,from_date,to_date,"day",oi=False
                ))

                progress=st.progress(0)
                status=st.empty()
                total=len(work)

                for n,row in enumerate(work.itertuples(index=False),start=1):
                    status.write(f"RS: {row.Stock} — {n}/{total}")
                    try:
                        raw=pd.DataFrame(kite.historical_data(
                            int(row.instrument_token),from_date,to_date,"day",oi=False
                        ))
                        feat=calculate_rs_features(raw,nifty)
                        if feat:
                            feat["Stock"]=row.Stock
                            feat["Sector"]=row.Sector
                            feat["Instrument_Token"]=int(row.instrument_token)
                            rows.append(feat)
                    except Exception:
                        pass

                    progress.progress(n/total)
                    time.sleep(0.38)

                status.empty()

                if rows:
                    stock_rs=percentile_scores(pd.DataFrame(rows))
                    sector_rs=sector_scores(stock_rs)

                    if not sector_rs.empty:
                        stock_rs=stock_rs.merge(
                            sector_rs[["Sector","Sector_RS_Score","Sector_RS_Acceleration","Sector_Rank"]],
                            on="Sector",how="left"
                        )

                    stock_rs["Alignment"]=stock_rs.apply(
                        lambda r: alignment_label(r["Stock_RS_Score"],r["Sector_RS_Score"]),
                        axis=1
                    )

                    if "rf_result" in st.session_state:
                        rfcols=st.session_state["rf_result"][[
                            "Stock","RF_Strength_Score","Latest_RF","Avg_RF_5D","RF_Acceleration","Latest_%Change"
                        ]]
                        stock_rs=stock_rs.merge(rfcols,on="Stock",how="left")

                        rf_pct=stock_rs["RF_Strength_Score"].rank(pct=True)*100
                        stock_rs["RF_RS_Alignment_Score"]=(
                            0.45*stock_rs["Stock_RS_Score"]
                            +0.30*stock_rs["Sector_RS_Score"]
                            +0.25*rf_pct
                        ).round(1)
                    else:
                        stock_rs["RF_RS_Alignment_Score"]=(
                            0.60*stock_rs["Stock_RS_Score"]
                            +0.40*stock_rs["Sector_RS_Score"].fillna(50)
                        ).round(1)

                    st.session_state["stock_rs_result"]=stock_rs
                    st.session_state["sector_rs_result"]=sector_rs

        if "sector_rs_result" in st.session_state:
            st.markdown("### Sector Ranking")
            st.dataframe(st.session_state["sector_rs_result"],use_container_width=True,hide_index=True)

        if "stock_rs_result" in st.session_state:
            sr=st.session_state["stock_rs_result"].copy()
            bull=sr.sort_values("RF_RS_Alignment_Score",ascending=False).head(20).reset_index(drop=True)
            bull.insert(0,"Rank",range(1,len(bull)+1))
            bear=sr.sort_values("RF_RS_Alignment_Score",ascending=True).head(20).reset_index(drop=True)
            bear.insert(0,"Rank",range(1,len(bear)+1))

            st.markdown("### Top 20 Bullish RS Alignment")
            st.dataframe(bull[["Rank","Stock","Sector","RF_RS_Alignment_Score","Stock_RS_Score","Sector_RS_Score","RS_5D","RS_4W","RS_Acceleration"]],use_container_width=True,hide_index=True)

            st.markdown("### Top 20 Bearish RS Alignment")
            st.dataframe(bear[["Rank","Stock","Sector","RF_RS_Alignment_Score","Stock_RS_Score","Sector_RS_Score","RS_5D","RS_4W","RS_Acceleration"]],use_container_width=True,hide_index=True)

with tabs[3]:
    st.subheader("Futures OI + Basis Engine")
    st.caption("Nearest-expiry stock futures. Price/OI classification + basis + RF/RS merge + institutional alignment.")

    if not st.session_state.access_token:
        st.warning("Connect Kite first.")
    else:
        kite.set_access_token(st.session_state.access_token)

        if st.button("Load Futures Universe"):
            cash=get_fno_equity_universe(kite)
            fut=get_nearest_futures_map(kite)
            combo=cash.merge(fut,on="Stock",how="inner")
            st.session_state["futures_universe"]=combo

        if "futures_universe" in st.session_state:
            combo=st.session_state["futures_universe"]
            st.metric("Nearest Futures Mapped",len(combo))

            c1,c2=st.columns(2)
            with c1:
                fut_scan_size=st.selectbox("Futures stocks to scan",[25,50,100,"ALL"],index=0)
            with c2:
                hist_days=st.slider("Futures OI history days",5,20,8)

            st.info("Run RF ALL and RS ALL first. Then run Futures ALL in the same session.")

            if st.button("Run Futures OI + Basis Scan",type="primary"):
                rf_ready=False
                rs_ready=False

                if "rf_result" in st.session_state:
                    rf_requested=int(st.session_state.get("rf_requested",len(st.session_state["rf_result"])))
                    rf_success=len(st.session_state["rf_result"])
                    rf_ready=rf_requested > 0 and (rf_success/rf_requested) >= 0.95

                if "stock_rs_result" in st.session_state:
                    rs_df=st.session_state["stock_rs_result"]
                    if len(rs_df) > 0:
                        rs_complete=rs_df[
                            ["Stock_RS_Score","Sector_RS_Score","RF_RS_Alignment_Score"]
                        ].notna().all(axis=1).sum()
                        rs_ready=(rs_complete/len(rs_df)) >= 0.95

                if not rf_ready or not rs_ready:
                    st.error(
                        "Institutional Futures scan blocked: RF and RF+RS coverage must both be at least 95%. "
                        "Run RF ALL, resolve failures, then run RS ALL."
                    )
                    st.stop()

                work=combo if fut_scan_size=="ALL" else combo.head(int(fut_scan_size))
                rows=[]
                errors=[]

                cash_keys=[f"NSE:{x}" for x in work["Stock"].tolist()]
                try:
                    cash_quotes=kite.quote(cash_keys)
                except Exception:
                    cash_quotes={}

                to_date=dt.datetime.now()
                from_date=to_date-dt.timedelta(days=hist_days)

                progress=st.progress(0)
                status=st.empty()
                total=len(work)

                for n,row in enumerate(work.itertuples(index=False),start=1):
                    status.write(f"Futures: {row.Stock} — {n}/{total}")

                    try:
                        spot_quote=cash_quotes.get(f"NSE:{row.Stock}",{})
                        spot=float(spot_quote.get("last_price",0) or 0)

                        hist=pd.DataFrame(kite.historical_data(
                            int(row.Future_Token),
                            from_date,
                            to_date,
                            "day",
                            oi=True
                        ))

                        feat=futures_features(spot,hist)
                        if feat:
                            feat["Stock"]=row.Stock
                            feat["Future_Symbol"]=row.Future_Symbol
                            feat["Future_Expiry"]=row.Future_Expiry
                            feat["Lot_Size"]=row.Lot_Size
                            feat["Spot_Price"]=round(spot,2)
                            rows.append(feat)
                        else:
                            errors.append([row.Stock,"Insufficient futures history"])
                    except Exception as e:
                        errors.append([row.Stock,str(e)])

                    progress.progress(n/total)
                    time.sleep(0.38)

                status.empty()

                if rows:
                    fdf=add_futures_score(pd.DataFrame(rows))
                    fdf=add_futures_conviction(fdf)

                    if "stock_rs_result" in st.session_state:
                        rscols=st.session_state["stock_rs_result"][[
                            "Stock","Stock_RS_Score","Sector_RS_Score","RF_RS_Alignment_Score"
                        ]].copy()
                        fdf=fdf.merge(rscols,on="Stock",how="left")

                    if "rf_result" in st.session_state:
                        rfcols=st.session_state["rf_result"][[
                            "Stock","Latest_RF","RF_Strength_Score"
                        ]].copy()
                        fdf=fdf.merge(rfcols,on="Stock",how="left")

                    if "RF_RS_Alignment_Score" in fdf.columns:
                        fdf["RF_RS_Futures_Score"]=(
                            0.65*fdf["RF_RS_Alignment_Score"].fillna(50)
                            +0.35*fdf["Futures_Score"].fillna(50)
                        ).round(1)
                    else:
                        fdf["RF_RS_Futures_Score"]=fdf["Futures_Score"]

                    fdf=add_institutional_alignment(fdf)
                    st.session_state["futures_result"]=fdf

                st.session_state["futures_errors"]=pd.DataFrame(errors,columns=["Stock","Error"])

        if "futures_result" in st.session_state:
            fdf=st.session_state["futures_result"].copy()

            st.markdown("### Institutional Long Candidates")
            long_candidates=fdf[
                fdf["Institutional_Signal"].isin(["CONFIRMED LONG","EARLY LONG"])
            ].sort_values(
                ["Institutional_Conviction","Institutional_Direction_Score"],
                ascending=False
            )
            st.dataframe(long_candidates,use_container_width=True,hide_index=True)

            st.markdown("### Institutional Short Candidates")
            short_candidates=fdf[
                fdf["Institutional_Signal"].isin(["CONFIRMED SHORT","EARLY SHORT"])
            ].sort_values(
                ["Institutional_Conviction","Institutional_Direction_Score"],
                ascending=[False,True]
            )
            st.dataframe(short_candidates,use_container_width=True,hide_index=True)

            st.markdown("### Divergence / Avoid")
            divergence=fdf[
                fdf["Institutional_Signal"].isin([
                    "BULLISH DIVERGENCE","BEARISH DIVERGENCE",
                    "SHORT COVERING RALLY","LONG UNWINDING"
                ])
            ].sort_values("Institutional_Conviction",ascending=False)
            st.dataframe(divergence,use_container_width=True,hide_index=True)

            st.markdown("### Full Futures Positioning")
            st.dataframe(
                fdf.sort_values(
                    ["Institutional_Conviction","Institutional_Direction_Score"],
                    ascending=[False,False]
                ),
                use_container_width=True,hide_index=True
            )

            st.download_button(
                "Download Futures Positioning CSV",
                data=fdf.to_csv(index=False).encode("utf-8"),
                file_name="futures_positioning.csv",
                mime="text/csv"
            )

        if "futures_errors" in st.session_state and not st.session_state["futures_errors"].empty:
            with st.expander("Futures scan errors"):
                st.dataframe(st.session_state["futures_errors"],use_container_width=True,hide_index=True)

with tabs[4]:
    st.subheader("F&O Stock Options Scanner — Phase 6F.1")

    # SPEED FIX:
    # Do NOT download the full NFO instrument master on every Streamlit rerun.
    # Streamlit executes code inside all tabs during a rerun, even when the
    # Options tab is not open. Load once per session and reuse the result.
    if "fno_stock_options_universe" not in st.session_state:
        if st.button("Load F&O Options Universe", key="load_fno_options_universe"):
            try:
                with st.spinner("Loading NFO instrument master once..."):
                    _inst_master=pd.DataFrame(kite.instruments("NFO"))
                    _fno_universe=build_fno_options_universe(_inst_master)
                    _stock_universe=stock_options_universe(_inst_master)

                    st.session_state["fno_options_universe"]=_fno_universe
                    st.session_state["fno_stock_options_universe"]=_stock_universe
                    st.session_state["nfo_instrument_master"]=_inst_master

                st.success("F&O options universe loaded and cached for this session.")
            except Exception as e:
                st.warning(f"F&O options universe unavailable: {e}")
        else:
            st.info(
                "F&O options universe is not loaded yet. "
                "Load it only when you need the Options scanner."
            )

    if "fno_stock_options_universe" in st.session_state:
        _stock_universe=st.session_state["fno_stock_options_universe"]
        _fno_universe=st.session_state.get(
            "fno_options_universe",
            pd.DataFrame()
        )

        u1,u2,u3=st.columns(3)
        u1.metric("Option Underlyings",len(_fno_universe))
        u2.metric("F&O Stocks",len(_stock_universe))
        u3.metric(
            "Index Underlyings",
            int((_fno_universe["Underlying_Type"]=="INDEX").sum())
            if not _fno_universe.empty else 0
        )

        with st.expander("View All F&O Stock Option Underlyings"):
            st.dataframe(
                _stock_universe,
                use_container_width=True,
                hide_index=True
            )

        if not _stock_universe.empty:
            st.download_button(
                "Download F&O Options Universe CSV",
                data=_stock_universe.to_csv(index=False).encode("utf-8"),
                file_name="fno_stock_options_universe.csv",
                mime="text/csv"
            )

    st.subheader("All F&O Lightweight Options Scanner — Phase 6F.2")

    if "fno_stock_options_universe" in st.session_state:
        scan_universe=st.session_state["fno_stock_options_universe"].copy()

        s1,s2=st.columns(2)
        with s1:
            scan_size=st.selectbox(
                "Stocks to scan",
                [25,50,100,"ALL"],
                index=0,
                key="fno_opt_scan_size"
            )
        with s2:
            wings=st.slider(
                "Strikes each side of ATM",
                min_value=5,
                max_value=12,
                value=8,
                step=1,
                key="fno_opt_wings"
            )

        st.info("Start with 25 for validation. Then run ALL.")

        if st.button("Run Lightweight Options Scan",type="primary"):
            work=scan_universe if scan_size=="ALL" else scan_universe.head(int(scan_size))

            with st.spinner(f"Scanning {len(work)} option underlyings..."):
                try:
                    result=lightweight_stock_options_scan(
                        kite,
                        work,
                        strikes_each_side=wings
                    )
                    result=normalize_lightweight_options_scores(result)
                    st.session_state["fno_light_options_result"]=result
                except Exception as e:
                    st.error(f"Lightweight options scan failed: {e}")

        if "fno_light_options_result" in st.session_state:
            lf=st.session_state["fno_light_options_result"].copy()

            if not lf.empty:
                bull=lf[
                    (lf["Normalized_Options_Bias"]=="BULLISH")
                    & (lf["Quality_Score"]>=50)
                ].sort_values(
                    ["Normalized_Options_Score","Quality_Score"],
                    ascending=False
                ).head(20).reset_index(drop=True)
                bull.insert(0,"Rank",range(1,len(bull)+1))

                bear=lf[
                    (lf["Normalized_Options_Bias"]=="BEARISH")
                    & (lf["Quality_Score"]>=50)
                ].sort_values(
                    ["Normalized_Options_Score","Quality_Score"],
                    ascending=[True,False]
                ).head(20).reset_index(drop=True)
                bear.insert(0,"Rank",range(1,len(bear)+1))

                c1,c2,c3,c4=st.columns(4)
                c1.metric("Scanned",len(lf))
                c2.metric("Bullish",int((lf["Normalized_Options_Bias"]=="BULLISH").sum()))
                c3.metric("Bearish",int((lf["Normalized_Options_Bias"]=="BEARISH").sum()))
                c4.metric("Quality >=50",int((lf["Quality_Score"]>=50).sum()))

                st.markdown("### Cross-Sectional Top Bullish")
                st.dataframe(bull,use_container_width=True,hide_index=True)

                st.markdown("### Cross-Sectional Top Bearish")
                st.dataframe(bear,use_container_width=True,hide_index=True)

                st.markdown("### Full Normalized Lightweight Options Scan")
                st.dataframe(lf,use_container_width=True,hide_index=True)

                st.download_button(
                    "Download Lightweight Options Scan CSV",
                    data=lf.to_csv(index=False).encode("utf-8"),
                    file_name="fno_light_options_scan.csv",
                    mime="text/csv"
                )

                st.subheader("Heavy Analysis Queue — Phase 6F.3")

                q1,q2,q3=st.columns(3)
                with q1:
                    bullish_n=st.number_input(
                        "Bullish queue size",
                        min_value=5,
                        max_value=50,
                        value=20,
                        step=5,
                        key="opt_heavy_bull_n"
                    )
                with q2:
                    bearish_n=st.number_input(
                        "Bearish queue size",
                        min_value=5,
                        max_value=50,
                        value=20,
                        step=5,
                        key="opt_heavy_bear_n"
                    )
                with q3:
                    min_quality=st.number_input(
                        "Minimum Quality Score",
                        min_value=0.0,
                        max_value=100.0,
                        value=60.0,
                        step=5.0,
                        key="opt_heavy_min_quality"
                    )

                bull_q,bear_q,heavy_q=build_heavy_analysis_queue(
                    lf,
                    bullish_n=int(bullish_n),
                    bearish_n=int(bearish_n),
                    min_quality=float(min_quality),
                )

                st.session_state["options_heavy_bull_queue"]=bull_q
                st.session_state["options_heavy_bear_queue"]=bear_q
                st.session_state["options_heavy_queue"]=heavy_q

                h1,h2,h3=st.columns(3)
                h1.metric("Bullish Queue",len(bull_q))
                h2.metric("Bearish Queue",len(bear_q))
                h3.metric("Total Heavy Queue",len(heavy_q))

                queue_cols=[
                    "Queue_Priority","Stock","Queue_Side",
                    "Normalized_Options_Score","Options_CrossSection_Rank",
                    "Quality_Score","PCR_OI","Max_Pain",
                    "Max_Call_OI_Strike","Max_Put_OI_Strike",
                    "Total_OI","Total_Volume","Lot_Size"
                ]

                st.markdown("### Bullish Heavy-Analysis Queue")
                st.dataframe(
                    bull_q[[c for c in queue_cols if c in bull_q.columns]],
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### Bearish Heavy-Analysis Queue")
                st.dataframe(
                    bear_q[[c for c in queue_cols if c in bear_q.columns]],
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Heavy Analysis Queue CSV",
                    data=heavy_q.to_csv(index=False).encode("utf-8"),
                    file_name="fno_options_heavy_queue.csv",
                    mime="text/csv"
                )

                st.subheader("Heavy Stock Options Engine — Phase 6F.4A")
                st.info(
                    "Validation run: 3 highest-priority bullish + "
                    "2 highest-priority bearish stocks. Full 40 comes only after validation."
                )

                hv1,hv2=st.columns(2)
                with hv1:
                    heavy_wings=st.slider(
                        "Heavy-analysis strikes each side ATM",
                        8,15,12,1,key="heavy_opt_wings"
                    )
                with hv2:
                    heavy_rate=st.number_input(
                        "Risk-free rate %",
                        min_value=0.0,max_value=20.0,value=6.5,step=0.1,
                        key="heavy_opt_rate"
                    )

                if st.button("Run 5-Stock Heavy Validation",type="primary"):
                    with st.spinner("Running IV, Greeks, GEX, Zero Gamma, Vanna and Charm..."):
                        heavy_result,heavy_details=run_heavy_queue_sample(
                            kite,
                            heavy_q,
                            bullish_n=3,
                            bearish_n=2,
                            strikes_each_side=int(heavy_wings),
                            risk_free_rate_pct=float(heavy_rate),
                        )

                        heavy_result=score_heavy_stock_options(heavy_result)

                        st.session_state["heavy_options_5_result"]=heavy_result
                        st.session_state["heavy_options_5_details"]=heavy_details

                if "heavy_options_5_result" in st.session_state:
                    hr=st.session_state["heavy_options_5_result"].copy()

                    if not hr.empty:
                        ok=int((hr["Status"]=="OK").sum()) if "Status" in hr.columns else 0
                        review=int((hr.get("Heavy_Data_Quality",pd.Series(dtype=str))=="REVIEW").sum())

                        z1,z2,z3=st.columns(3)
                        z1.metric("Heavy Stocks Tested",len(hr))
                        z2.metric("Successful",ok)
                        z3.metric("Quality Review",review)

                        display_cols=[
                            "Stock","Queue_Side","Lightweight_Score",
                            "Heavy_Options_Score","Heavy_Options_Bias",
                            "Lightweight_Heavy_Alignment","Heavy_Conviction",
                            "Gamma_Location_Score","Vanna_Charm_Context_Score",
                            "Spot","Expiry","IV_Coverage_%",
                            "Net_GEX_1pct","Gamma_Regime","Zero_Gamma_Level",
                            "Call_Gamma_Wall","Put_Gamma_Wall","Net_Gamma_Wall",
                            "Dealer_Vanna_1vol","Dealer_Charm_1day",
                            "Vanna_Wall","Charm_Wall","Heavy_Data_Quality",
                            "Heavy_Why","Status","Error"
                        ]

                        if "Lightweight_Heavy_Alignment" in hr.columns:
                            a1,a2,a3=st.columns(3)
                            a1.metric(
                                "Confirmed",
                                int(hr["Lightweight_Heavy_Alignment"].str.startswith("CONFIRMED").sum())
                            )
                            a2.metric(
                                "Mixed",
                                int(hr["Lightweight_Heavy_Alignment"].str.startswith("MIXED").sum())
                            )
                            a3.metric(
                                "Rejected",
                                int((hr["Lightweight_Heavy_Alignment"]=="REJECTED / CONTRADICTED").sum())
                            )

                        st.dataframe(
                            hr[[c for c in display_cols if c in hr.columns]],
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download 5-Stock Heavy Scoring CSV",
                            data=hr.to_csv(index=False).encode("utf-8"),
                            file_name="fno_heavy_options_5_scored.csv",
                            mime="text/csv"
                        )

                st.subheader("10-Stock Heavy Scale Test — Phase 6F.4C")
                st.caption(
                    "Runs the validated heavy engine on 5 bullish + "
                    "5 bearish queue names before scaling to all 40."
                )

                if st.button("Run 10-Stock Heavy Test", type="primary"):
                    with st.spinner(
                        "Running heavy options analysis on 10 stocks..."
                    ):
                        hr10, hd10 = run_heavy_queue_batch(
                            kite,
                            heavy_q,
                            bullish_n=5,
                            bearish_n=5,
                            strikes_each_side=int(heavy_wings),
                            risk_free_rate_pct=float(heavy_rate),
                        )

                        st.session_state["heavy_options_10_result"] = hr10
                        st.session_state["heavy_options_10_details"] = hd10

                if "heavy_options_10_result" in st.session_state:
                    hr10 = st.session_state[
                        "heavy_options_10_result"
                    ].copy()

                    if not hr10.empty:
                        ok10 = int(
                            (hr10["Status"] == "OK").sum()
                        ) if "Status" in hr10.columns else 0

                        pass10 = int(
                            (hr10.get(
                                "Heavy_Data_Quality",
                                pd.Series(dtype=str)
                            ) == "PASS").sum()
                        )

                        confirmed10 = int(
                            hr10.get(
                                "Lightweight_Heavy_Alignment",
                                pd.Series(dtype=str)
                            ).astype(str).str.startswith("CONFIRMED").sum()
                        )

                        t1,t2,t3,t4 = st.columns(4)
                        t1.metric("Stocks Tested", len(hr10))
                        t2.metric("Successful", ok10)
                        t3.metric("Quality PASS", pass10)
                        t4.metric("Confirmed", confirmed10)

                        display_cols10 = [
                            "Stock","Queue_Side","Lightweight_Score",
                            "Heavy_Options_Score","Heavy_Options_Bias",
                            "Lightweight_Heavy_Alignment","Heavy_Conviction",
                            "IV_Coverage_%","Gamma_Regime",
                            "Zero_Gamma_Level","Dealer_Vanna_1vol",
                            "Dealer_Charm_1day","Heavy_Data_Quality",
                            "Status","Error"
                        ]

                        st.dataframe(
                            hr10[[
                                c for c in display_cols10
                                if c in hr10.columns
                            ]],
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download 10-Stock Heavy Test CSV",
                            data=hr10.to_csv(index=False).encode("utf-8"),
                            file_name="fno_heavy_options_10_scored.csv",
                            mime="text/csv"
                        )

                st.subheader("Full 40-Stock Heavy Run — Phase 6F.4C")
                st.caption(
                    "Memory-optimized full run: 20 bullish + 20 bearish. "
                    "The NFO instrument master is loaded once and only compact "
                    "per-stock summaries are retained."
                )

                if st.button("Run Full 40-Stock Heavy Analysis", type="primary"):
                    p40=st.progress(0)
                    s40=st.empty()

                    def _heavy40_progress(i,total,stock):
                        s40.write(f"Heavy options: {stock} — {i}/{total}")
                        p40.progress(i/total)

                    try:
                        hr40, _ = run_heavy_queue_batch(
                            kite,
                            heavy_q,
                            bullish_n=20,
                            bearish_n=20,
                            strikes_each_side=int(heavy_wings),
                            risk_free_rate_pct=float(heavy_rate),
                            keep_details=False,
                            progress_callback=_heavy40_progress,
                        )

                        # Memory fix: store ONLY the compact 40-row summary.
                        st.session_state["heavy_options_40_result"] = hr40
                        st.session_state.pop("heavy_options_40_details",None)

                    finally:
                        s40.empty()

                if "heavy_options_40_result" in st.session_state:
                    hr40 = st.session_state[
                        "heavy_options_40_result"
                    ].copy()

                    if not hr40.empty:
                        ok40 = int(
                            (hr40["Status"] == "OK").sum()
                        ) if "Status" in hr40.columns else 0

                        pass40 = int(
                            (
                                hr40.get(
                                    "Heavy_Data_Quality",
                                    pd.Series(dtype=str)
                                ) == "PASS"
                            ).sum()
                        )

                        align40 = hr40.get(
                            "Lightweight_Heavy_Alignment",
                            pd.Series(dtype=str)
                        ).astype(str)

                        confirmed40 = int(
                            align40.str.startswith("CONFIRMED").sum()
                        )
                        mixed40 = int(
                            align40.str.startswith("MIXED").sum()
                        )
                        rejected40 = int(
                            (align40 == "REJECTED / CONTRADICTED").sum()
                        )

                        f1,f2,f3,f4,f5 = st.columns(5)
                        f1.metric("Stocks Tested", len(hr40))
                        f2.metric("Successful", ok40)
                        f3.metric("Quality PASS", pass40)
                        f4.metric("Confirmed", confirmed40)
                        f5.metric("Mixed / Rejected", mixed40 + rejected40)

                        display_cols40 = [
                            "Stock","Queue_Side","Lightweight_Score",
                            "Heavy_Options_Score","Heavy_Options_Bias",
                            "Lightweight_Heavy_Alignment","Heavy_Conviction",
                            "Gamma_Location_Score",
                            "Vanna_Charm_Context_Score",
                            "Spot","IV_Coverage_%","Gamma_Regime",
                            "Zero_Gamma_Level","Call_Gamma_Wall",
                            "Put_Gamma_Wall","Net_Gamma_Wall",
                            "Dealer_Vanna_1vol","Dealer_Charm_1day",
                            "Vanna_Wall","Charm_Wall",
                            "Heavy_Data_Quality","Heavy_Why",
                            "Status","Error"
                        ]

                        st.dataframe(
                            hr40[[
                                c for c in display_cols40
                                if c in hr40.columns
                            ]],
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download Full 40-Stock Heavy Analysis CSV",
                            data=hr40.to_csv(index=False).encode("utf-8"),
                            file_name="fno_heavy_options_40_scored.csv",
                            mime="text/csv"
                        )

                        st.subheader("Final Options Confirmation Ranking — Phase 6F.4D")

                        final_opt=build_final_options_confirmation_ranking(hr40)
                        st.session_state["final_options_confirmation"]=final_opt

                        class_counts=final_opt[
                            "Final_Options_Classification"
                        ].value_counts()

                        r1,r2,r3,r4,r5=st.columns(5)
                        r1.metric("Confirmed Long",int(class_counts.get("CONFIRMED LONG",0)))
                        r2.metric("Conditional Long",int(class_counts.get("CONDITIONAL LONG",0)))
                        r3.metric("Neutral",int(class_counts.get("NEUTRAL",0)))
                        r4.metric("Conditional Short",int(class_counts.get("CONDITIONAL SHORT",0)))
                        r5.metric("Confirmed Short",int(class_counts.get("CONFIRMED SHORT",0)))

                        final_cols=[
                            "Final_Options_Rank","Stock","Queue_Side",
                            "Final_Options_Classification",
                            "Phase7_Options_Bias","Phase7_Options_Strength",
                            "Lightweight_Score","Heavy_Options_Score",
                            "Heavy_Conviction","Lightweight_Heavy_Alignment",
                            "Gamma_Regime","Zero_Gamma_Level",
                            "Gamma_Location_Score",
                            "Vanna_Charm_Context_Score",
                            "Dealer_Vanna_1vol","Dealer_Charm_1day",
                            "IV_Coverage_%","Heavy_Data_Quality",
                            "Options_Rejection_Flags","Heavy_Why"
                        ]

                        st.dataframe(
                            final_opt[[c for c in final_cols if c in final_opt.columns]],
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download Final Options Confirmation CSV",
                            data=final_opt.to_csv(index=False).encode("utf-8"),
                            file_name="final_options_confirmation_ranking.csv",
                            mime="text/csv"
                        )

    st.subheader("Options Positioning — Phase 6E.1")
    st.caption("Nearest-expiry NIFTY/BANKNIFTY with user-captured OI baseline. OI change, buildup/unwinding and Modified Max Pain are calculated only from that baseline — no day-low proxy.")

    if not st.session_state.access_token:
        st.warning("Connect Kite first.")
    else:
        kite.set_access_token(st.session_state.access_token)
        c1,c2=st.columns(2)
        with c1:
            option_underlying=st.selectbox("Underlying",["NIFTY","BANKNIFTY"])
        with c2:
            option_wings=st.slider("Strikes each side of ATM",5,30,15,5)

        g1,g2=st.columns(2)
        with g1:
            risk_free_rate=st.number_input("Risk-free Rate (%)",0.0,20.0,6.5,0.1)
        with g2:
            dividend_yield=st.number_input("Dividend Yield (%)",0.0,10.0,0.0,0.1)

        g3,g4=st.columns(2)
        with g3:
            gamma_sweep_pct=st.slider(
                "Gamma Map Spot Sweep (%)",
                min_value=2.0,
                max_value=10.0,
                value=5.0,
                step=1.0
            )
        with g4:
            gamma_sweep_steps=st.selectbox(
                "Gamma Map Resolution",
                [41,61,81,101],
                index=2
            )

        b1,b2=st.columns(2)

        with b1:
            if st.button("Capture OI Baseline",type="primary"):
                try:
                    with st.spinner("Capturing OI baseline..."):
                        base_chain,base_summary=build_option_chain(
                            kite,
                            option_underlying,
                            option_wings
                        )

                    if base_chain.empty:
                        st.error("No option-chain data returned.")
                    else:
                        base_summary["Baseline_Time"]=dt.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        st.session_state["option_oi_baseline"]=base_chain.copy()
                        st.session_state["option_oi_baseline_summary"]=base_summary.copy()

                        st.success(
                            f'OI baseline captured for {option_underlying} '
                            f'at {base_summary["Baseline_Time"]}.'
                        )
                except Exception as e:
                    st.error(f"Baseline capture failed: {e}")

        with b2:
            if st.button("Refresh Option Chain"):
                try:
                    baseline=st.session_state.get("option_oi_baseline")
                    baseline_summary=st.session_state.get(
                        "option_oi_baseline_summary"
                    )

                    with st.spinner("Loading live option chain..."):
                        oc,os=build_option_chain(
                            kite,
                            option_underlying,
                            option_wings,
                            baseline_chain=baseline,
                            baseline_summary=baseline_summary
                        )

                    if oc.empty:
                        st.error("No option-chain data returned.")
                    else:
                        st.session_state["option_chain_result"]=oc
                        st.session_state["option_summary"]=os
                except Exception as e:
                    st.error(f"Options scan failed: {e}")

        if "option_oi_baseline_summary" in st.session_state:
            bs=st.session_state["option_oi_baseline_summary"]
            st.info(
                f'OI Baseline: {bs.get("Underlying")} | '
                f'Expiry {bs.get("Expiry")} | '
                f'{bs.get("Baseline_Time","time unavailable")}'
            )

            if st.button("Clear OI Baseline"):
                st.session_state.pop("option_oi_baseline",None)
                st.session_state.pop("option_oi_baseline_summary",None)
                st.session_state.pop("option_chain_result",None)
                st.session_state.pop("option_summary",None)
                st.rerun()

        if "option_summary" in st.session_state:
            s=st.session_state["option_summary"]

            # Hotfix: Streamlit may preserve an older Phase 6A/6B summary
            # across a code deploy. New Phase 6B.1 fields are therefore read
            # defensively until the user captures/refreshes a new baseline.
            if "OI_Baseline_Status" not in s:
                s["OI_Baseline_Status"]="NOT CAPTURED"
            if "Dynamic_OI_Bias" not in s:
                s["Dynamic_OI_Bias"]="BASELINE REQUIRED"
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Spot",f'{s["Spot"]:,.2f}')
            c2.metric("ATM",f'{s["ATM_Strike"]:,.0f}')
            c3.metric("PCR (OI)",f'{s["PCR_OI"]:.3f}')
            c4.metric("PCR (OI Change)",(
                f'{s.get("PCR_OI_Change", np.nan):.3f}'
                if pd.notna(s.get("PCR_OI_Change", np.nan)) else "BASELINE REQUIRED"
            ))

            c5,c6,c7,c8=st.columns(4)
            c5.metric("Max Pain",f'{s["Max_Pain"]:,.0f}')
            c6.metric("Modified Max Pain",(
                f'{s.get("Modified_Max_Pain", np.nan):,.0f}'
                if pd.notna(s.get("Modified_Max_Pain", np.nan)) else "BASELINE REQUIRED"
            ))
            c7.metric("Max Call OI",f'{s["Max_Call_OI_Strike"]:,.0f}')
            c8.metric("Max Put OI",f'{s["Max_Put_OI_Strike"]:,.0f}')

            c9,c10,c11,c12=st.columns(4)
            c9.metric("Max Call OI Change",(
                f'{s.get("Max_Call_OI_Change_Strike", np.nan):,.0f}'
                if pd.notna(s.get("Max_Call_OI_Change_Strike", np.nan)) else "NA"
            ))
            c10.metric("Max Put OI Change",(
                f'{s.get("Max_Put_OI_Change_Strike", np.nan):,.0f}'
                if pd.notna(s.get("Max_Put_OI_Change_Strike", np.nan)) else "NA"
            ))
            c11.metric("Options Bias",s["Options_Bias"])
            c12.metric("Dynamic OI Bias",s.get("Dynamic_OI_Bias", "BASELINE REQUIRED"))

            mod_dist=(
                f'{s.get("Spot_vs_ModifiedMaxPain_%", np.nan):.2f}%'
                if pd.notna(s.get("Spot_vs_ModifiedMaxPain_%", np.nan))
                else "BASELINE REQUIRED"
            )
            st.caption(
                f'Nearest expiry: {s["Expiry"]} | '
                f'OI baseline: {s.get("OI_Baseline_Status","NOT CAPTURED")} | '
                f'Spot vs Max Pain: {s["Spot_vs_MaxPain_%"]:.2f}% | '
                f'Spot vs Modified Max Pain: {mod_dist}'
            )

        if "option_chain_result" in st.session_state:
            oc=st.session_state["option_chain_result"].copy()
            if "option_summary" in st.session_state:
                s=st.session_state["option_summary"]
                try:
                    gc,gs=add_greeks_and_gex(
                        oc,
                        s["Spot"],
                        s["Expiry"],
                        risk_free_rate,
                        dividend_yield
                    )

                    gm,gms=gamma_map(
                        gc,
                        spot=s["Spot"],
                        expiry=s["Expiry"],
                        risk_free_rate_pct=risk_free_rate,
                        dividend_yield_pct=dividend_yield,
                        sweep_pct=gamma_sweep_pct,
                        steps=gamma_sweep_steps,
                    )

                    gs.update(gms)

                    vc,vcs=add_vanna_charm_exposure(
                        gc,
                        spot=s["Spot"],
                        expiry=s["Expiry"],
                        risk_free_rate_pct=risk_free_rate,
                        dividend_yield_pct=dividend_yield,
                    )

                    st.session_state["option_greeks_result"]=gc
                    st.session_state["option_greeks_summary"]=gs
                    st.session_state["gamma_map_result"]=gm
                    st.session_state["vanna_charm_result"]=vc
                    st.session_state["vanna_charm_summary"]=vcs

                    regime=build_options_regime(
                        option_summary=s,
                        gex_summary=gs,
                        vanna_charm_summary=vcs,
                        option_chain=oc,
                    )
                    st.session_state["options_regime_result"]=regime

                except Exception as e:
                    st.warning(f"Greeks/GEX calculation unavailable: {e}")
            st.markdown("### Option Chain")
            st.dataframe(
                oc[[
                    "CE_OI","CE_OI_Change","CE_OI_Change_%",
                    "CE_Positioning","CE_Price_Change_%",
                    "CE_Volume","CE_LTP",
                    "Strike",
                    "PE_LTP","PE_Volume","PE_Price_Change_%",
                    "PE_Positioning","PE_OI_Change_%",
                    "PE_OI_Change","PE_OI"
                ]],
                use_container_width=True,hide_index=True
            )

            if "options_regime_result" in st.session_state:
                rg=st.session_state["options_regime_result"]

                st.markdown("### Institutional Options Regime")

                r1,r2,r3,r4=st.columns(4)
                r1.metric("Options Direction",f'{rg["Options_Direction_Score"]:.1f}')
                r2.metric("Bias",rg["Options_Bias"])
                r3.metric("Volatility Regime",rg["Volatility_Behavior"])
                r4.metric("Conviction",f'{rg["Options_Conviction"]:.1f}')

                st.dataframe(
                    pd.DataFrame([
                        ["Static OI Structure",rg["Static_Structure_Score"]],
                        ["Dynamic OI",rg["Dynamic_OI_Score"]],
                        ["Gamma",rg["Gamma_Score"]],
                        ["Vanna + Charm",rg["Vanna_Charm_Score"]],
                    ],columns=["Component","Score_0_100"]),
                    use_container_width=True,
                    hide_index=True
                )

                q1,q2,q3,q4=st.columns(4)
                q1.metric(
                    "ΔOI Data Quality",
                    f'{rg["Dynamic_OI_Quality_Score"]:.0f}/100'
                )
                q2.metric(
                    "Dynamic OI Weight",
                    f'{rg["Dynamic_OI_Weight"]*100:.0f}%'
                )
                q3.metric(
                    "ΔOI Breadth",
                    f'{rg["NonZero_OI_Breadth_%"]:.1f}%'
                )
                q4.metric(
                    "Baseline Age",
                    (
                        f'{rg["Baseline_Elapsed_Min"]:.1f} min'
                        if pd.notna(rg["Baseline_Elapsed_Min"])
                        else "NA"
                    )
                )

                if not rg["Dynamic_OI_Active"]:
                    st.info(
                        "Dynamic OI excluded from the regime score: "
                        + rg["Dynamic_OI_Quality_Reason"]
                    )

                if rg["Conflicts"]!="NONE":
                    st.warning(f'Options conflicts: {rg["Conflicts"]}')
                else:
                    st.success("No major cross-component options conflict.")

                st.write("**Why:**",rg["Why"])

                levels=pd.DataFrame([
                    ["Max Pain",rg["Max_Pain"]],
                    ["Modified Max Pain",rg["Modified_Max_Pain"]],
                    ["Call OI Wall",rg["Call_OI_Wall"]],
                    ["Put OI Wall",rg["Put_OI_Wall"]],
                    ["Zero Gamma",rg["Zero_Gamma_Level"]],
                    ["Call Gamma Wall",rg["Call_Gamma_Wall"]],
                    ["Put Gamma Wall",rg["Put_Gamma_Wall"]],
                    ["Net Gamma Wall",rg["Net_Gamma_Wall"]],
                    ["Vanna Wall",rg["Vanna_Wall"]],
                    ["Charm Wall",rg["Charm_Wall"]],
                ],columns=["Institutional_Level","Price"])

                st.markdown("### Options Institutional Level Map")
                st.dataframe(
                    levels,
                    use_container_width=True,
                    hide_index=True
                )

            if "option_greeks_summary" in st.session_state:
                gs=st.session_state["option_greeks_summary"]
                st.markdown("### IV + Greeks + GEX")
                z1,z2,z3,z4=st.columns(4)
                z1.metric("Current Net GEX / 1% Move",(
                    f'{gs.get("Current_Net_GEX_1pct",gs.get("Net_GEX_1pct",0)):,.0f}'
                ))
                z2.metric("Gamma Regime",gs.get("Gamma_Regime",gs.get("Gamma_State","UNKNOWN")))
                z3.metric("Zero-Gamma Level",(
                    f'{gs["Zero_Gamma_Level"]:,.0f}'
                    if pd.notna(gs.get("Zero_Gamma_Level",np.nan)) else "NA"
                ))
                z4.metric("Spot vs Zero-Gamma",(
                    f'{gs["Spot_vs_ZeroGamma_%"]:.2f}%'
                    if pd.notna(gs.get("Spot_vs_ZeroGamma_%",np.nan)) else "NA"
                ))

                w1,w2,w3,w4=st.columns(4)
                w1.metric("Call Gamma Wall",(
                    f'{gs["Call_Gamma_Wall"]:,.0f}'
                    if pd.notna(gs.get("Call_Gamma_Wall",np.nan)) else "NA"
                ))
                w2.metric("Put Gamma Wall",(
                    f'{gs["Put_Gamma_Wall"]:,.0f}'
                    if pd.notna(gs.get("Put_Gamma_Wall",np.nan)) else "NA"
                ))
                w3.metric("Net Gamma Wall",(
                    f'{gs["Net_Gamma_Wall"]:,.0f}'
                    if pd.notna(gs.get("Net_Gamma_Wall",np.nan)) else "NA"
                ))
                w4.metric("ATM GEX Concentration",(
                    f'{gs["ATM_GEX_Concentration_%"]:.1f}%'
                    if pd.notna(gs.get("ATM_GEX_Concentration_%",np.nan)) else "NA"
                ))
                gc=st.session_state["option_greeks_result"]
                st.dataframe(gc[["Strike","CE_IV_%","CE_Delta","CE_Gamma","CE_Vega","CE_Theta","CE_GEX_1pct","PE_IV_%","PE_Delta","PE_Gamma","PE_Vega","PE_Theta","PE_GEX_1pct","Net_GEX_1pct"]],use_container_width=True,hide_index=True)
                st.caption(
                    "GEX: Gamma × OI × Lot Size × Spot² × 1%. "
                    "Calls positive, puts negative; dealer sign is a modelling assumption."
                )

                if "gamma_map_result" in st.session_state:
                    gm=st.session_state["gamma_map_result"].copy()

                    st.markdown("### Spot-Sweep Gamma Map")
                    st.line_chart(
                        gm.set_index("Scenario_Spot")[["Net_GEX_1pct"]]
                    )

                    st.dataframe(
                        gm,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Zero-Gamma is estimated by sweeping hypothetical NIFTY spot "
                        "and recomputing aggregate signed gamma exposure while holding "
                        "current strike IVs constant."
                    )

            if "vanna_charm_summary" in st.session_state:
                vcs=st.session_state["vanna_charm_summary"]

                st.markdown("### Vanna + Charm Dealer-Flow Proxy")

                v1,v2,v3,v4=st.columns(4)
                v1.metric(
                    "Dealer Vanna Hedge / +1 vol pt",
                    f'{vcs["Dealer_Net_Vanna_Hedge_Proxy_1vol"]:,.0f}'
                )
                v2.metric(
                    "Dealer Charm Hedge / 1 day",
                    f'{vcs["Dealer_Net_Charm_Hedge_Proxy_1day"]:,.0f}'
                )
                v3.metric(
                    "Vanna Wall",
                    f'{vcs["Vanna_Wall"]:,.0f}'
                    if pd.notna(vcs["Vanna_Wall"]) else "NA"
                )
                v4.metric(
                    "Charm Wall",
                    f'{vcs["Charm_Wall"]:,.0f}'
                    if pd.notna(vcs["Charm_Wall"]) else "NA"
                )

                st.write("**Vanna interpretation:**",vcs["Vanna_Hedge_Bias"])
                st.write("**Charm interpretation:**",vcs["Charm_Hedge_Bias"])

                vc=st.session_state["vanna_charm_result"].copy()

                st.dataframe(
                    vc[[
                        "Strike",
                        "CE_Vanna","CE_Vanna_Exposure_1vol",
                        "PE_Vanna","PE_Vanna_Exposure_1vol",
                        "Dealer_Vanna_Hedge_Proxy_1vol",
                        "CE_Charm_Day","CE_Charm_Exposure_1day",
                        "PE_Charm_Day","PE_Charm_Exposure_1day",
                        "Dealer_Charm_Hedge_Proxy_1day",
                    ]],
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Vanna/Charm dealer-flow signs are model proxies. "
                    "They assume dealers hold the opposite side of aggregate option-holder OI; "
                    "actual dealer inventory is not observable from NSE OI alone."
                )

            with st.expander("OI Change Source Diagnostics"):
                st.dataframe(
                    oc[[
                        "Strike",
                        "CE_OI_Base","CE_OI_Change_Source",
                        "PE_OI_Base","PE_OI_Change_Source"
                    ]],
                    use_container_width=True,hide_index=True
                )
            st.download_button(
                "Download Option Chain CSV",
                data=oc.to_csv(index=False).encode("utf-8"),
                file_name="option_chain.csv",mime="text/csv"
            )

            if "options_regime_result" in st.session_state:
                st.download_button(
                    "Download Options Regime CSV",
                    data=pd.DataFrame([
                        st.session_state["options_regime_result"]
                    ]).to_csv(index=False).encode("utf-8"),
                    file_name="options_regime.csv",
                    mime="text/csv"
                )

            if "option_greeks_result" in st.session_state:
                st.download_button(
                    "Download Options Greeks CSV",
                    data=st.session_state["option_greeks_result"].to_csv(index=False).encode("utf-8"),
                    file_name="options_greeks.csv",
                    mime="text/csv"
                )

                st.download_button(
                    "Download GEX Summary CSV",
                    data=pd.DataFrame([
                        st.session_state["option_greeks_summary"]
                    ]).to_csv(index=False).encode("utf-8"),
                    file_name="gex_summary.csv",
                    mime="text/csv"
                )

                if "gamma_map_result" in st.session_state:
                    st.download_button(
                        "Download Gamma Map CSV",
                        data=st.session_state["gamma_map_result"].to_csv(index=False).encode("utf-8"),
                        file_name="gamma_map.csv",
                        mime="text/csv"
                    )

                if "vanna_charm_result" in st.session_state:
                    st.download_button(
                        "Download Vanna Charm CSV",
                        data=st.session_state["vanna_charm_result"].to_csv(index=False).encode("utf-8"),
                        file_name="vanna_charm.csv",
                        mime="text/csv"
                    )

                    st.download_button(
                        "Download Vanna Charm Summary CSV",
                        data=pd.DataFrame([
                            st.session_state["vanna_charm_summary"]
                        ]).to_csv(index=False).encode("utf-8"),
                        file_name="vanna_charm_summary.csv",
                        mime="text/csv"
                    )

            if "option_oi_baseline" in st.session_state:
                baseline_export=st.session_state["option_oi_baseline"].copy()
                st.download_button(
                    "Download OI Baseline CSV",
                    data=baseline_export.to_csv(index=False).encode("utf-8"),
                    file_name="option_oi_baseline.csv",
                    mime="text/csv"
                )
            if "option_summary" in st.session_state:
                osdf=pd.DataFrame([st.session_state["option_summary"]])
                st.download_button(
                    "Download Options Summary CSV",
                    data=osdf.to_csv(index=False).encode("utf-8"),
                    file_name="options_summary.csv",mime="text/csv"
                )

with tabs[5]:
    st.subheader("Auction Levels — Phase 3A")
    st.caption(
        "30-minute auction context: previous-day references, Initial Balance, "
        "range extension and current auction state."
    )

    if not st.session_state.access_token:
        st.warning("Connect Kite first.")
    else:
        kite.set_access_token(st.session_state.access_token)

        if st.button("Load Auction Universe",key="auction_load"):
            st.session_state["auction_universe"]=get_fno_equity_universe(kite)

        if "auction_universe" in st.session_state:
            auction_universe=st.session_state["auction_universe"].copy()
            st.metric("Auction Stocks Mapped",len(auction_universe))

            c1,c2=st.columns(2)
            with c1:
                auction_scan_size=st.selectbox(
                    "Auction stocks to scan",
                    [25,50,100,"ALL"],
                    index=0
                )
            with c2:
                auction_history_days=st.slider(
                    "Auction history days",
                    5,20,8
                )

            st.info(
                "Validate 25 stocks first. Initial Balance uses the first two "
                "30-minute NSE bars: 09:15 and 09:45."
            )

            if st.button("Run Auction Levels Scan",type="primary"):
                work=(
                    auction_universe
                    if auction_scan_size=="ALL"
                    else auction_universe.head(int(auction_scan_size))
                )

                rows=[]
                errors=[]
                total=len(work)
                progress=st.progress(0)
                status=st.empty()

                to_date=dt.datetime.now()
                from_date=to_date-dt.timedelta(days=auction_history_days)

                for n,row in enumerate(work.itertuples(index=False),start=1):
                    status.write(f"Auction: {row.Stock} — {n}/{total}")

                    try:
                        raw=pd.DataFrame(
                            kite.historical_data(
                                int(row.instrument_token),
                                from_date,
                                to_date,
                                "30minute",
                                oi=False
                            )
                        )

                        feat=auction_features(raw)

                        if feat:
                            feat["Stock"]=row.Stock
                            feat["Instrument_Token"]=int(row.instrument_token)
                            rows.append(feat)
                        else:
                            errors.append([
                                row.Stock,
                                "Insufficient current/previous session data"
                            ])

                    except Exception as e:
                        errors.append([row.Stock,str(e)])

                    progress.progress(n/total)
                    time.sleep(0.38)

                status.empty()

                if rows:
                    auction_df=auction_direction_score(pd.DataFrame(rows))
                    st.session_state["auction_result"]=auction_df

                st.session_state["auction_errors"]=pd.DataFrame(
                    errors,
                    columns=["Stock","Error"]
                )

        if "auction_result" in st.session_state:
            adf=st.session_state["auction_result"].copy()

            st.markdown("### Upside Auction")
            upside=adf[
                adf["Auction_State"].isin([
                    "UPSIDE RANGE EXTENSION",
                    "TWO-SIDED EXPANSION - BULLISH CLOSE"
                ])
            ].sort_values(
                ["Auction_Direction_Score","RE_Up_%IB"],
                ascending=False
            )
            st.dataframe(
                upside[[
                    "Stock","Auction_State","Current_Close",
                    "IB_High","IB_Low","IB_Range",
                    "RE_Up_%IB","Current_vs_IB",
                    "Open_Location","Auction_Direction_Score"
                ]],
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Downside Auction")
            downside=adf[
                adf["Auction_State"].isin([
                    "DOWNSIDE RANGE EXTENSION",
                    "TWO-SIDED EXPANSION - BEARISH CLOSE"
                ])
            ].sort_values(
                ["Auction_Direction_Score","RE_Down_%IB"],
                ascending=[True,False]
            )
            st.dataframe(
                downside[[
                    "Stock","Auction_State","Current_Close",
                    "IB_High","IB_Low","IB_Range",
                    "RE_Down_%IB","Current_vs_IB",
                    "Open_Location","Auction_Direction_Score"
                ]],
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Full Auction Levels")
            st.dataframe(
                adf.sort_values(
                    ["Auction_Direction_Score","Stock"],
                    ascending=[False,True]
                ),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Auction Levels CSV",
                data=adf.to_csv(index=False).encode("utf-8"),
                file_name="auction_levels.csv",
                mime="text/csv"
            )

        if (
            "auction_errors" in st.session_state and
            not st.session_state["auction_errors"].empty
        ):
            with st.expander("Auction Scan Errors"):
                st.dataframe(
                    st.session_state["auction_errors"],
                    use_container_width=True,
                    hide_index=True
                )


with tabs[6]:
    st.subheader("Market Profile — Phase 3B.1")
    st.caption(
        "30-minute TPO profile + 70% value area + refined profile-shape logic + approximate Volume Profile. "
        "TPO levels are derived from candle ranges; volume-at-price is estimated "
        "from 30-minute OHLCV bars, not tick-by-tick trades."
    )

    if not st.session_state.access_token:
        st.warning("Connect Kite first.")
    else:
        kite.set_access_token(st.session_state.access_token)

        if st.button("Load Market Profile Universe",key="mp_load"):
            st.session_state["mp_universe"]=get_fno_equity_universe(kite)

        if "mp_universe" in st.session_state:
            mp_universe=st.session_state["mp_universe"].copy()
            st.metric("Profile Stocks Mapped",len(mp_universe))

            c1,c2,c3=st.columns(3)
            with c1:
                mp_scan_size=st.selectbox(
                    "Profile stocks to scan",
                    [25,50,100,"ALL"],
                    index=0
                )
            with c2:
                mp_history_days=st.slider(
                    "Profile history days",
                    5,20,10
                )
            with c3:
                mp_rows=st.selectbox(
                    "Target profile rows",
                    [40,60,80,100],
                    index=2
                )

            st.info(
                "Validate 25 stocks first. Value Area = 70%. "
                "Target profile rows controls price-row resolution."
            )

            if st.button("Run Market Profile Scan",type="primary"):
                work=(
                    mp_universe
                    if mp_scan_size=="ALL"
                    else mp_universe.head(int(mp_scan_size))
                )

                rows=[]
                errors=[]
                total=len(work)
                progress=st.progress(0)
                status=st.empty()

                to_date=dt.datetime.now()
                from_date=to_date-dt.timedelta(days=mp_history_days)

                for n,row in enumerate(work.itertuples(index=False),start=1):
                    status.write(f"Market Profile: {row.Stock} — {n}/{total}")

                    try:
                        raw=pd.DataFrame(
                            kite.historical_data(
                                int(row.instrument_token),
                                from_date,
                                to_date,
                                "30minute",
                                oi=False
                            )
                        )

                        feat=market_profile_features(
                            raw,
                            target_rows=int(mp_rows)
                        )

                        if feat:
                            feat["Stock"]=row.Stock
                            feat["Instrument_Token"]=int(row.instrument_token)
                            rows.append(feat)
                        else:
                            errors.append([
                                row.Stock,
                                "Insufficient profile history"
                            ])

                    except Exception as e:
                        errors.append([row.Stock,str(e)])

                    progress.progress(n/total)
                    time.sleep(0.38)

                status.empty()

                if rows:
                    mp_df=add_profile_score(pd.DataFrame(rows))
                    st.session_state["market_profile_result"]=mp_df

                st.session_state["market_profile_errors"]=pd.DataFrame(
                    errors,
                    columns=["Stock","Error"]
                )

        if "market_profile_result" in st.session_state:
            mpdf=st.session_state["market_profile_result"].copy()

            bullish=mpdf[
                mpdf["Profile_Direction_Score"] >= 65
            ].sort_values(
                ["Profile_Direction_Score","Value_Overlap_%"],
                ascending=[False,True]
            )

            bearish=mpdf[
                mpdf["Profile_Direction_Score"] <= 35
            ].sort_values(
                ["Profile_Direction_Score","Value_Overlap_%"],
                ascending=[True,True]
            )

            st.markdown("### Bullish Value Migration / Acceptance")
            st.dataframe(
                bullish[[
                    "Stock","Profile_Bias","Profile_Direction_Score",
                    "TPO_POC","TPO_VAH","TPO_VAL",
                    "Value_Migration","POC_Migration","Close_vs_Value",
                    "Profile_Shape","Value_Overlap_%","Latest_Naked_POC"
                ]],
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Bearish Value Migration / Acceptance")
            st.dataframe(
                bearish[[
                    "Stock","Profile_Bias","Profile_Direction_Score",
                    "TPO_POC","TPO_VAH","TPO_VAL",
                    "Value_Migration","POC_Migration","Close_vs_Value",
                    "Profile_Shape","Value_Overlap_%","Latest_Naked_POC"
                ]],
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Full Market Profile")
            st.dataframe(
                mpdf.sort_values(
                    ["Profile_Direction_Score","Stock"],
                    ascending=[False,True]
                ),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Market Profile CSV",
                data=mpdf.to_csv(index=False).encode("utf-8"),
                file_name="market_profile.csv",
                mime="text/csv"
            )

        if (
            "market_profile_errors" in st.session_state and
            not st.session_state["market_profile_errors"].empty
        ):
            with st.expander("Market Profile Scan Errors"):
                st.dataframe(
                    st.session_state["market_profile_errors"],
                    use_container_width=True,
                    hide_index=True
                )


with tabs[7]:
    st.subheader("Institutional Setup Engine — Phase 4.2")
    st.caption(
        "RF + Stock RS + Sector RS + Futures + Auction + Market Profile. "
        "Direction, Conviction, Location and Risk are scored separately. ""RF is percentile-normalized and Futures uses standalone positioning to avoid RF/RS double-counting."
    )

    required_session_keys=[
        "rf_result",
        "stock_rs_result",
        "futures_result",
        "auction_result",
        "market_profile_result",
    ]

    missing_keys=[
        key for key in required_session_keys
        if key not in st.session_state
    ]

    if missing_keys:
        st.warning(
            "Run ALL modules in the same Streamlit session: "
            "RF → RS → Futures → Auction → Market Profile."
        )
        st.write("Missing:",", ".join(missing_keys))
    else:
        if st.button("Build Institutional Setup",type="primary"):
            setup_df=build_institutional_setup(
                st.session_state["rf_result"],
                st.session_state["stock_rs_result"],
                st.session_state["futures_result"],
                st.session_state["auction_result"],
                st.session_state["market_profile_result"],
            )
            st.session_state["institutional_setup_result"]=setup_df

        if "institutional_setup_result" in st.session_state:
            sdf=st.session_state["institutional_setup_result"].copy()

            complete=int(sdf["Data_Complete"].sum())
            total=len(sdf)
            coverage=(complete/total*100) if total else 0
            actionable=int(
                sdf["Institutional_Setup"].isin(
                    ["A+ LONG","A LONG","A+ SHORT","A SHORT"]
                ).sum()
            )

            c1,c2,c3,c4=st.columns(4)
            c1.metric("Setup Universe",total)
            c2.metric("Complete Stocks",complete)
            c3.metric("Coverage",f"{coverage:.1f}%")
            c4.metric("Actionable",actionable)

            if coverage < 95:
                st.error(
                    "Institutional Setup coverage is below 95%. "
                    "Do not use final grades until missing module data is resolved."
                )
            else:
                st.success("Institutional Setup coverage validation passed.")

            st.markdown("### Scale Validation")
            scale_df=pd.DataFrame([
                ["RF Raw","RF_Raw_Score"],
                ["RF Normalized","RF_Normalized_Score"],
                ["Stock RS","Stock_RS_Score"],
                ["Sector RS","Sector_RS_Score"],
                ["Futures Direction","Direction_Futures"],
                ["Auction","Auction_Direction_Score"],
                ["Profile","Profile_Direction_Score"],
            ],columns=["Component","Column"])

            scale_rows=[]
            for item in scale_df.itertuples(index=False):
                if item.Column in sdf.columns:
                    s=pd.to_numeric(sdf[item.Column],errors="coerce")
                    scale_rows.append([
                        item.Component,
                        round(float(s.min()),1) if s.notna().any() else None,
                        round(float(s.median()),1) if s.notna().any() else None,
                        round(float(s.max()),1) if s.notna().any() else None,
                    ])

            st.dataframe(
                pd.DataFrame(
                    scale_rows,
                    columns=["Component","Min","Median","Max"]
                ),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Phase 4.2 Direction Weights")
            st.dataframe(
                pd.DataFrame([
                    ["RF Normalized",15],
                    ["Stock RS",15],
                    ["Sector RS",10],
                    ["Futures Positioning",20],
                    ["Auction Direction",15],
                    ["Market Profile",25],
                ],columns=["Component","Weight_%"]),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Component Correlation")
            corr=component_correlation(sdf)
            if corr.empty:
                st.info("Correlation matrix unavailable.")
            else:
                st.dataframe(corr,use_container_width=True)

                corr_abs=corr.abs().copy()
                for c in corr_abs.columns:
                    corr_abs.loc[c,c]=0

                max_corr=float(corr_abs.max().max())
                if max_corr >= 0.80:
                    st.warning(
                        f"High component correlation detected: {max_corr:.2f}. "
                        "Review for possible information overlap."
                    )
                else:
                    st.success(
                        f"No extreme component duplication detected. "
                        f"Maximum absolute correlation: {max_corr:.2f}"
                    )

            st.markdown("### A+ / A Long")
            longs=sdf[
                sdf["Institutional_Setup"].isin(["A+ LONG","A LONG"])
            ].sort_values(
                ["Institutional_Setup_Score","Setup_Conviction"],
                ascending=False
            )
            st.dataframe(
                longs[[
                    "Stock","Institutional_Setup",
                    "Institutional_Setup_Score","Setup_Direction_Score",
                    "Setup_Conviction","Setup_Location_Quality",
                    "Setup_Risk_Quality","Module_Agreement_%",
                    "RF_Strength_Score","Stock_RS_Score","Sector_RS_Score",
                    "Positioning","Auction_State","Profile_Bias",
                    "Close_vs_Value","Current_vs_IB"
                ]],
                use_container_width=True,hide_index=True
            )

            st.markdown("### A+ / A Short")
            shorts=sdf[
                sdf["Institutional_Setup"].isin(["A+ SHORT","A SHORT"])
            ].sort_values(
                ["Institutional_Setup_Score","Setup_Conviction"],
                ascending=False
            )
            st.dataframe(
                shorts[[
                    "Stock","Institutional_Setup",
                    "Institutional_Setup_Score","Setup_Direction_Score",
                    "Setup_Conviction","Setup_Location_Quality",
                    "Setup_Risk_Quality","Module_Agreement_%",
                    "RF_Strength_Score","Stock_RS_Score","Sector_RS_Score",
                    "Positioning","Auction_State","Profile_Bias",
                    "Close_vs_Value","Current_vs_IB"
                ]],
                use_container_width=True,hide_index=True
            )

            st.markdown("### Watchlist")
            watch=sdf[
                sdf["Institutional_Setup"].isin(["WATCH LONG","WATCH SHORT"])
            ].sort_values(
                ["Institutional_Setup_Score","Setup_Conviction"],
                ascending=False
            )
            st.dataframe(watch,use_container_width=True,hide_index=True)

            st.markdown("### Full Institutional Setup")
            st.dataframe(
                sdf.sort_values(
                    ["Institutional_Setup_Score","Setup_Conviction"],
                    ascending=False
                ),
                use_container_width=True,hide_index=True
            )

            st.download_button(
                "Download Institutional Setup CSV",
                data=sdf.to_csv(index=False).encode("utf-8"),
                file_name="institutional_setup.csv",
                mime="text/csv"
            )


with tabs[8]:
    st.subheader("Top 20 Institutional Candidate Scanner — Phase 4.3.1")
    st.caption(
        "Operational shortlist generated from the validated Phase 4.2 setup engine. "
        "Candidate Score ranks quality; Setup Type separates fresh institutional buildup from covering/unwinding. ""Adjusted Score gives a modest continuation preference to fresh buildup + auction/value acceptance."
    )

    if "institutional_setup_result" not in st.session_state:
        st.warning(
            "Build the Institutional Setup first. "
            "Run RF → RS → Futures → Auction → Market Profile → Institutional Setup."
        )
    else:
        setup_df=st.session_state["institutional_setup_result"].copy()

        bull20,bear20,scanner_full=build_candidate_scanner(
            setup_df,
            top_n=20
        )

        st.session_state["bullish_top20"]=bull20
        st.session_state["bearish_top20"]=bear20
        st.session_state["candidate_scanner_result"]=scanner_full

        display_cols=[
            "Rank","Stock","Institutional_Setup","Setup_Type",
            "Candidate_Score_Adjusted","Candidate_Score","Continuation_Bonus",
            "Institutional_Setup_Score","Setup_Direction_Score",
            "Setup_Conviction","Setup_Location_Quality",
            "Setup_Risk_Quality","Module_Agreement_%",
            "Positioning","Auction_State","Profile_Bias",
            "Close_vs_Value","Current_vs_IB",
            "Why_Selected"
        ]

        st.markdown("### Top 20 Institutional Bullish")
        st.dataframe(
            bull20[[c for c in display_cols if c in bull20.columns]],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Top 20 Institutional Bearish")
        st.dataframe(
            bear20[[c for c in display_cols if c in bear20.columns]],
            use_container_width=True,
            hide_index=True
        )

        c1,c2=st.columns(2)

        with c1:
            st.download_button(
                "Download Top 20 Bullish CSV",
                data=bull20.to_csv(index=False).encode("utf-8"),
                file_name="top20_institutional_bullish.csv",
                mime="text/csv"
            )

        with c2:
            st.download_button(
                "Download Top 20 Bearish CSV",
                data=bear20.to_csv(index=False).encode("utf-8"),
                file_name="top20_institutional_bearish.csv",
                mime="text/csv"
            )

        st.download_button(
            "Download Combined Top 40 Scanner CSV",
            data=scanner_full.to_csv(index=False).encode("utf-8"),
            file_name="institutional_top40_scanner.csv",
            mime="text/csv"
        )


with tabs[9]:
    st.subheader("11:15 AM Execution Engine — Phase 5D")
    st.caption(
        "Strict confirmation/invalidation layer for the Phase 4 Top-40 universe. "
        "TRADE requires structural acceptance plus VWAP/rotation confirmation. Phase 5D adds a volatility buffer from recent 30-minute ATR to create an Effective Stop, Effective Risk and Effective R:R. Uses fully completed 30-minute candles available by ~11:15: VWAP, "
        "developing IB, early rotation, profile location and volume participation."
    )

    if not st.session_state.access_token:
        st.warning("Connect Kite first.")
    elif "candidate_scanner_result" not in st.session_state:
        st.warning("Build Institutional Setup and open Top 20 Scanner first.")
    elif "market_profile_result" not in st.session_state:
        st.warning("Market Profile result is required.")
    else:
        kite.set_access_token(st.session_state.access_token)

        candidates=st.session_state["candidate_scanner_result"].copy()
        profile_df=st.session_state["market_profile_result"].copy()

        st.metric("Preselected Candidates",len(candidates))

        if st.button("Run 11:15 Execution Scan",type="primary"):
            rows=[]
            errors=[]
            progress=st.progress(0)
            status=st.empty()
            total=len(candidates)

            # Pull enough history for current + prior session.
            to_date=dt.datetime.now()
            from_date=to_date-dt.timedelta(days=5)

            universe=get_fno_equity_universe(kite)
            token_map=dict(zip(universe["Stock"],universe["instrument_token"]))

            for n,row in enumerate(candidates.itertuples(index=False),start=1):
                stock=row.Stock
                side="BULLISH" if row.Scanner_Group=="TOP BULLISH" else "BEARISH"
                status.write(f"11:15: {stock} — {n}/{total}")

                try:
                    token=token_map.get(stock)
                    if token is None:
                        raise ValueError("Instrument token not found")

                    raw=pd.DataFrame(kite.historical_data(
                        int(token),from_date,to_date,"30minute",oi=False
                    ))

                    # Anti-lookahead guard for a true 11:15 decision.
                    # Kite 30-minute candles are start-stamped, so the 10:45 bar
                    # is the last bar fully completed at 11:15. Preserve full prior
                    # sessions for previous-day high/low context.
                    if not raw.empty and "date" in raw.columns:
                        _d=pd.to_datetime(raw["date"],errors="coerce")
                        _last_session=_d.dt.date.max()
                        _is_last=_d.dt.date.eq(_last_session)
                        _after_cutoff=_d.dt.time > pd.Timestamp("10:45").time()
                        raw=raw.loc[~(_is_last & _after_cutoff)].copy()

                    prow=profile_df[profile_df["Stock"]==stock]
                    profile_row=prow.iloc[0].to_dict() if not prow.empty else {}

                    feat=intraday_execution_features(
                        raw,
                        side=side,
                        profile_row=profile_row
                    )

                    if feat:
                        base=row._asdict()
                        base.update(feat)
                        rows.append(base)
                    else:
                        errors.append([stock,"Insufficient intraday history"])
                except Exception as e:
                    errors.append([stock,str(e)])

                progress.progress(n/total)
                time.sleep(0.38)

            status.empty()

            if rows:
                edf=rank_execution(pd.DataFrame(rows))
                edf=add_trade_plan(edf)
                edf=rank_trade_plan(edf)
                st.session_state["execution_1115_result"]=edf

            st.session_state["execution_1115_errors"]=pd.DataFrame(
                errors,columns=["Stock","Error"]
            )

        if "execution_1115_result" in st.session_state:
            edf=st.session_state["execution_1115_result"].copy()

            trades=edf[edf["Trade_Action"]=="TRADE NOW"].copy()
            waits=edf[
                edf["Trade_Action"].isin([
                    "WAIT FOR PULLBACK",
                    "WAIT FOR BREAKOUT",
                    "WAIT - POOR R:R",
                    "WAIT - STOP TOO WIDE",
                    "WAIT - TARGET STRUCTURE",
                    "TOO EXTENDED"
                ])
            ].copy()
            rejects=edf[edf["Trade_Action"]=="REJECT"].copy()

            c1,c2,c3,c4=st.columns(4)
            c1.metric("Scanned",len(edf))
            c2.metric("TRADE NOW",len(trades))
            c3.metric("WAIT",len(waits))
            c4.metric("REJECT",len(rejects))

            show=[
                "Stock","Side","Setup_Type","Candidate_Score_Adjusted",
                "Execution_Score","Execution_Decision","Entry_Quality",
                "Trade_Action","Trade_Plan_Grade",
                "Entry_Zone_Low","Entry_Zone_High","Entry_Trigger",
                "Structural_Stop","Volatility_Buffer","Effective_Stop","T1","T2","Target_Hierarchy_OK","Risk_%","Effective_Risk_%","RR_T1","RR_T2","Effective_RR_T1","Effective_RR_T2",
                "Extension_%","Structural_Acceptance","Opening_Type",
                "Price_1115","ATR_30m","ATR_30m_%","VWAP_1115","Current_vs_Developing_IB",
                "Current_vs_Profile_Value","Early_Rotation_Balance",
                "Early_Volume_Ratio","Confirmation"
            ]

            st.markdown("### Highest-Quality Executions")
            highq=trades[
                trades["Trade_Plan_Grade"].isin(["A+","A"])
            ].copy()
            st.dataframe(
                highq[[c for c in show if c in highq.columns]],
                use_container_width=True,hide_index=True
            )

            st.markdown("### Trade Now")
            st.dataframe(
                trades[[c for c in show if c in trades.columns]],
                use_container_width=True,hide_index=True
            )

            st.markdown("### Wait / Pullback / R:R Filter")
            st.dataframe(
                waits[[c for c in show if c in waits.columns]],
                use_container_width=True,hide_index=True
            )

            with st.expander("Rejected Candidates"):
                st.dataframe(
                    rejects[[c for c in show if c in rejects.columns]],
                    use_container_width=True,hide_index=True
                )

            st.download_button(
                "Download 11:15 Execution CSV",
                data=edf.to_csv(index=False).encode("utf-8"),
                file_name="execution_1115.csv",
                mime="text/csv"
            )

        if "execution_1115_errors" in st.session_state and not st.session_state["execution_1115_errors"].empty:
            with st.expander("11:15 Scan Errors"):
                st.dataframe(
                    st.session_state["execution_1115_errors"],
                    use_container_width=True,hide_index=True
                )


with tabs[10]:
    st.subheader("Risk & Position Sizing — Phase 5E.2")
    st.caption(
        "Sizes only TRADE NOW candidates using Effective Stop distance. FUTURES mode now uses risk per contract + margin per contract instead of cash-style notional exposure. "
        "Portfolio controls cap risk per trade, total portfolio risk, "
        "position value and simultaneous positions."
    )

    if "execution_1115_result" not in st.session_state:
        st.warning("Run the 11:15 Execution Scan first.")
    else:
        edf=st.session_state["execution_1115_result"].copy()

        instrument_mode=st.radio(
            "Instrument Mode",
            ["CASH","FUTURES"],
            horizontal=True
        )

        margin_source="ESTIMATED"
        estimated_margin_pct=15.0
        available_futures_margin=1000000.0

        if instrument_mode=="FUTURES":
            cma,cmb=st.columns(2)
            with cma:
                available_futures_margin=st.number_input(
                    "Available Futures Margin (₹)",
                    min_value=10000.0,
                    value=1000000.0,
                    step=50000.0
                )
            with cmb:
                estimated_margin_pct=st.number_input(
                    "Estimated Margin per Contract (% of Notional)",
                    min_value=5.0,
                    max_value=50.0,
                    value=15.0,
                    step=1.0
                )
            st.caption(
                "Phase 5E.2 uses estimated futures margin for sizing. "
                "Later, paper/live execution can replace this with broker margin API values."
            )

        c1,c2,c3=st.columns(3)
        with c1:
            capital=st.number_input(
                "Trading Capital (₹)",
                min_value=10000.0,
                value=1000000.0,
                step=50000.0
            )
        with c2:
            risk_per_trade=st.number_input(
                "Risk per Trade (%)",
                min_value=0.10,
                max_value=2.00,
                value=0.50,
                step=0.10
            )
        with c3:
            max_portfolio_risk=st.number_input(
                "Max Total Portfolio Risk (%)",
                min_value=0.50,
                max_value=5.00,
                value=2.00,
                step=0.25
            )

        c4,c5=st.columns(2)
        with c4:
            max_position_value=st.number_input(
                "Max Position Value / Stock (%)",
                min_value=5.0,
                max_value=100.0,
                value=20.0,
                step=5.0
            )
        with c5:
            max_positions=st.number_input(
                "Max Simultaneous Positions",
                min_value=1,
                max_value=20,
                value=5,
                step=1
            )

        if st.button("Build Risk Plan",type="primary"):
            lot_size_map={}
            margin_per_contract_map={}

            if instrument_mode=="FUTURES":
                source=None

                if "futures_universe" in st.session_state:
                    source=st.session_state["futures_universe"].copy()
                elif "futures_result" in st.session_state:
                    source=st.session_state["futures_result"].copy()

                if source is None or "Lot_Size" not in source.columns:
                    st.error(
                        "Futures lot sizes are not available. "
                        "Run Load Futures Universe / Futures scan first."
                    )
                    st.stop()

                lot_size_map=dict(zip(source["Stock"],source["Lot_Size"]))

                # Estimate one-contract margin from current entry/notional.
                # This is intentionally explicit and visible until we wire
                # broker margin API values in the paper-trading phase.
                for erow in edf.itertuples(index=False):
                    stock=erow.Stock
                    lot=lot_size_map.get(stock)
                    entry=getattr(erow,"Entry_Trigger",None)

                    try:
                        if lot and entry:
                            notional=float(entry)*float(lot)
                            margin_per_contract_map[stock]=(
                                notional*float(estimated_margin_pct)/100.0
                            )
                    except Exception:
                        pass

            rdf=add_position_sizing(
                edf,
                capital=capital,
                risk_per_trade_pct=risk_per_trade,
                max_total_portfolio_risk_pct=max_portfolio_risk,
                max_position_value_pct=max_position_value,
                max_positions=max_positions,
                instrument_mode=instrument_mode,
                lot_size_map=lot_size_map,
                margin_per_contract_map=margin_per_contract_map,
                available_futures_margin_rs=(
                    available_futures_margin
                    if instrument_mode=="FUTURES"
                    else None
                ),
            )
            st.session_state["risk_plan_result"]=rdf

        if "risk_plan_result" in st.session_state:
            rdf=st.session_state["risk_plan_result"].copy()
            summary=portfolio_summary(rdf)

            c1,c2,c3,c4=st.columns(4)
            c1.metric("Approved Positions",summary["Approved_Positions"])
            c2.metric("Total Risk (₹)",f'{summary["Total_Actual_Risk_Rs"]:,.0f}')
            c3.metric("Total Risk (%)",f'{summary["Total_Actual_Risk_%"]:.2f}%')

            if rdf["Instrument_Mode_Setting"].iloc[0]=="FUTURES":
                c4.metric("Margin Usage (%)",f'{summary["Margin_Usage_%"]:.1f}%')
            else:
                c4.metric("Total Exposure (%)",f'{summary["Total_Exposure_%"]:.1f}%')

            approved=rdf[rdf["Position_Approved"]==True].copy()
            blocked=rdf[
                (rdf["Trade_Action"]=="TRADE NOW") &
                (rdf["Position_Approved"]==False)
            ].copy()

            show=[
                "Stock","Side","Setup_Type","Trade_Plan_Grade",
                "Execution_Score","Trade_Action","Instrument_Mode",
                "Lot_Size","Contracts","Qty",
                "Entry_Trigger","Effective_Stop","T1","T2",
                "Effective_Risk_%","Effective_RR_T2",
                "Risk_Per_Contract_Rs","Margin_Per_Contract_Rs",
                "Contracts_By_Risk","Contracts_By_Margin",
                "Qty_By_Risk","Qty_By_Exposure","Sizing_Constraint",
                "Margin_Required_Rs","Allowed_Risk_Rs","Actual_Risk_Rs",
                "Actual_Risk_%","Position_Value_Rs",
                "Capital_Exposure_%","Position_Block_Reason",
                "Portfolio_Risk_After_%"
            ]

            st.markdown("### Sizing Constraint Summary")
            constraints=(
                rdf[rdf["Trade_Action"]=="TRADE NOW"]["Sizing_Constraint"]
                .value_counts(dropna=False)
                .rename_axis("Constraint")
                .reset_index(name="Stocks")
            )
            st.dataframe(
                constraints,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Approved Positions")
            st.dataframe(
                approved[[c for c in show if c in approved.columns]],
                use_container_width=True,
                hide_index=True
            )

            if not blocked.empty:
                st.markdown("### Trade Signals Blocked by Portfolio Limits")
                st.dataframe(
                    blocked[[c for c in show if c in blocked.columns]],
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("### Full Risk Plan")
            st.dataframe(
                rdf,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Risk Plan CSV",
                data=rdf.to_csv(index=False).encode("utf-8"),
                file_name="risk_plan.csv",
                mime="text/csv"
            )

with tabs[11]:
    st.subheader("Build Status")
    st.dataframe(pd.DataFrame([
        ["1","Cloud dashboard + Kite","DONE"],
        ["2A","Daily RF","DONE"],
        ["2B","Full F&O RF","DONE"],
        ["2C","Stock RS + Sector RS","DONE"],
        ["2D","Futures OI + Basis","DONE"],
        ["2E","Institutional Alignment Signal","DONE"],
        ["3A","Auction Levels + Initial Balance","DONE"],
        ["3B","Market/Volume Profile","DONE"],
        ["3B.1","Profile Shape + Single Prints","DONE"],
        ["4","Institutional Setup Engine","DONE"],
        ["4.1","RF Normalization + Scale Validation","DONE"],
        ["4.2","Weight + Correlation Calibration","DONE"],
        ["4.3","Top 20 Institutional Scanner","DONE"],
        ["4.3.1","Fresh Buildup vs Covering/Unwinding","DONE"],
        ["5A","11:15 Execution Confirmation","DONE"],
        ["5B","Strict Execution + Entry Quality","DONE"],
        ["5C","Entry Zone + Stop + Targets + R:R","DONE"],
        ["5C.1","Target Hierarchy + R:R Validation","DONE"],
        ["5D","Volatility-Adjusted Stop + Effective R:R","DONE"],
        ["5E","Risk + Position Sizing","DONE"],
        ["5E.1","Cash/Futures Sizing + Constraint Diagnostics","DONE"],
        ["5E.2","Futures Margin-Aware Sizing","DONE"],
        ["6A","Options Chain + PCR + Max Pain","DONE"],
        ["6B","OI Change + Positioning + Modified Max Pain","DONE"],
        ["6B.1","Captured OI Baseline + True Intraday ΔOI","CURRENT"],
        ["4","Options + Greeks","QUEUED"],
        ["5","Institutional Score + 11:15 scan","QUEUED"],
    ],columns=["Phase","Module","Status"]),use_container_width=True,hide_index=True)


with tabs[12]:
    st.subheader("Master Institutional Decision Engine — Phase 7A")
    st.caption(
        "RF 30% + Sector RS 20% + Stock RS 20% + Futures 15% + Options 15%. "
        "Incomplete data and directional conflicts are explicitly gated."
    )

    st.markdown("## One-Click Full Institutional Run")
    st.caption(
        "Runs the complete sequence in one window: "
        "RF ALL → Stock + Sector RS ALL → Futures ALL → Phase-7 Top/Bottom 40 "
        "→ Heavy Options full 40 → Final Phase-7C/7D ranking."
    )

    with st.expander("One-Click Run Settings", expanded=False):
        oc1,oc2,oc3,oc4=st.columns(4)
        with oc1:
            one_rf_days=st.number_input(
                "RF history days",
                min_value=15,
                max_value=75,
                value=45,
                step=5,
                key="oneclick_rf_days"
            )
        with oc2:
            one_rs_days=st.number_input(
                "RS calendar days",
                min_value=60,
                max_value=150,
                value=100,
                step=10,
                key="oneclick_rs_days"
            )
        with oc3:
            one_fut_days=st.number_input(
                "Futures OI days",
                min_value=5,
                max_value=20,
                value=8,
                step=1,
                key="oneclick_fut_days"
            )
        with oc4:
            one_heavy_wings=st.number_input(
                "Options strikes each side ATM",
                min_value=8,
                max_value=15,
                value=12,
                step=1,
                key="oneclick_heavy_wings"
            )

        one_rate=st.number_input(
            "Options risk-free rate %",
            min_value=0.0,
            max_value=20.0,
            value=6.5,
            step=0.1,
            key="oneclick_rate"
        )

        st.markdown("#### Live Timing Settings")
        lc1,lc2,lc3=st.columns(3)

        with lc1:
            one_live_interval=st.selectbox(
                "Live candle interval",
                ["5minute","15minute"],
                index=0,
                key="oneclick_live_interval"
            )

        with lc2:
            one_ib_minutes=st.selectbox(
                "Initial Balance duration",
                [30,45,60],
                index=2,
                key="oneclick_ib_minutes"
            )

        with lc3:
            one_rvol_sessions=st.selectbox(
                "RVOL lookback sessions",
                [10,15,20],
                index=2,
                key="oneclick_rvol_sessions"
            )

    st.markdown("## Fast Live Mode")
    st.caption(
        "Reuses the latest validated Phase-7D institutional baseline and refreshes "
        "only the live execution layer: Live RF + true RVOL + VWAP + IB + "
        "READY/WAIT/INVALIDATED + Phase-7G transitions."
    )

    cache_time=st.session_state.get("full_refresh_time")
    if cache_time is not None:
        cache_time=pd.Timestamp(cache_time)
        cache_age_min=max(
            0.0,
            (pd.Timestamp.now()-cache_time).total_seconds()/60.0
        )
        fc1,fc2=st.columns(2)
        fc1.metric(
            "Cached Phase-7D Baseline",
            cache_time.strftime("%H:%M:%S")
        )
        fc2.metric(
            "Baseline Age",
            f"{cache_age_min:.1f} min"
        )

        if cache_age_min > 60:
            st.warning(
                "Cached institutional baseline is older than 60 minutes. "
                "Use FULL REFRESH before relying on new trade candidates."
            )
        elif cache_age_min > 30:
            st.info(
                "Baseline is over 30 minutes old. Fast Live Mode is suitable for "
                "timing/monitoring existing candidates, but consider a Full Refresh "
                "before adding new positions."
            )
    else:
        st.info(
            "No cached Phase-7D baseline exists yet. Run the FULL REFRESH once first."
        )

    fast_col1,fast_col2=st.columns(2)
    with fast_col1:
        fast_interval=st.selectbox(
            "Fast mode candle interval",
            ["5minute","15minute"],
            index=0,
            key="fast_live_interval"
        )
    with fast_col2:
        fast_ib=st.selectbox(
            "Fast mode IB duration",
            [30,45,60],
            index=2,
            key="fast_live_ib"
        )

    fast_rvol=st.selectbox(
        "Fast mode RVOL lookback",
        [10,15,20],
        index=2,
        key="fast_live_rvol"
    )

    st.markdown("### Automatic Fast Refresh + Alerts")
    auto1,auto2=st.columns(2)
    with auto1:
        auto_fast_enabled=st.toggle(
            "Enable automatic fast refresh",
            value=False,
            key="auto_fast_enabled"
        )
    with auto2:
        auto_fast_minutes=st.selectbox(
            "Refresh every",
            [2,5,10],
            index=1,
            key="auto_fast_minutes"
        )

    st.caption(
        "Automatic mode refreshes only the cached Phase-7D candidates. "
        "It does not rerun RF/RS/Futures/Heavy Options."
    )

    st.markdown("#### Automatic Paper-Trading Loop")
    st.caption(
        "Simulation only — no Zerodha orders are placed. "
        "When enabled, each auto refresh updates OPEN paper trades first, "
        "then admits new READY candidates through the validated Phase 7H.1 risk gate."
    )

    ap1,ap2,ap3,ap4=st.columns(4)

    with ap1:
        auto_paper_enabled=st.toggle(
            "Enable auto paper trades",
            value=False,
            key="auto_paper_enabled"
        )

    with ap2:
        auto_paper_risk=st.number_input(
            "Paper risk / trade (₹)",
            min_value=100.0,
            max_value=100000.0,
            value=1000.0,
            step=100.0,
            key="auto_paper_risk"
        )

    with ap3:
        auto_paper_max_positions=st.number_input(
            "Max paper positions",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="auto_paper_max_positions"
        )

    with ap4:
        auto_paper_max_heat=st.number_input(
            "Max paper heat (₹)",
            min_value=500.0,
            max_value=100000.0,
            value=5000.0,
            step=500.0,
            key="auto_paper_max_heat"
        )

    ap5,ap6=st.columns(2)

    with ap5:
        auto_paper_sector_limit=st.number_input(
            "Max positions / sector",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            key="auto_paper_sector_limit"
        )

    with ap6:
        auto_paper_rr=st.number_input(
            "Paper target R",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            key="auto_paper_rr"
        )

    if "phase7h1_gated_trades" not in st.session_state:
        st.session_state["phase7h1_gated_trades"]=pd.DataFrame()

    if st.button(
        "Reset Auto Paper Portfolio",
        key="reset_auto_paper_portfolio"
    ):
        st.session_state["phase7h1_gated_trades"]=pd.DataFrame()
        st.session_state.pop("auto_paper_admission_log",None)
        st.success("Automatic paper portfolio reset to 0 positions / ₹0 heat.")

    def _run_fast_live_cycle():
        if "phase7d_final_ranking" not in st.session_state:
            return None, "No cached Phase-7D ranking. Run FULL REFRESH first."

        p7d_fast=st.session_state[
            "phase7d_final_ranking"
        ].copy()

        fast_candidates=p7d_fast[
            p7d_fast["P7D_Final_Action"].isin(
                ["LONG","LONG WATCH","SHORT","SHORT WATCH"]
            )
        ][["Stock"]].drop_duplicates().copy()

        if fast_candidates.empty:
            return None, "No tradeable Phase-7D candidates."

        live_fast=build_phase7f_live_feed(
            kite,
            fast_candidates,
            interval=fast_interval,
            ib_minutes=int(fast_ib),
            rvol_lookback_sessions=int(fast_rvol),
        )

        st.session_state["phase7f_live_feed"]=live_fast

        gate_fast=build_phase7e_live_entry_gate(
            p7d_fast,
            live_fast,
        )

        st.session_state[
            "phase7f_auto_entry_gate"
        ]=gate_fast

        previous_snapshot=st.session_state.get(
            "phase7g_previous_gate_snapshot"
        )

        monitored_fast,new_log_fast=(
            build_phase7g_state_transitions(
                gate_fast,
                previous_snapshot
            )
        )

        st.session_state[
            "phase7g_current_monitor"
        ]=monitored_fast

        if "phase7g_transition_log" not in st.session_state:
            st.session_state[
                "phase7g_transition_log"
            ]=pd.DataFrame()

        if new_log_fast is not None and not new_log_fast.empty:
            st.session_state[
                "phase7g_transition_log"
            ]=pd.concat(
                [
                    st.session_state[
                        "phase7g_transition_log"
                    ],
                    new_log_fast
                ],
                ignore_index=True
            )

        st.session_state[
            "phase7g_previous_gate_snapshot"
        ]=gate_fast.copy()

        st.session_state[
            "fast_live_refresh_time"
        ]=pd.Timestamp.now()

        return monitored_fast, None

    # Streamlit fragments refresh only this small live section, not the whole app.
    if auto_fast_enabled:
        if hasattr(st,"fragment"):
            @st.fragment(run_every=f"{int(auto_fast_minutes)}m")
            def _auto_fast_fragment():
                if not st.session_state.access_token:
                    st.warning("Connect Kite to use automatic fast refresh.")
                    return

                try:
                    kite.set_access_token(
                        st.session_state.access_token
                    )

                    monitored_auto,auto_err=_run_fast_live_cycle()

                    if auto_err:
                        st.warning(auto_err)
                        return

                    gate_auto=st.session_state.get(
                        "phase7f_auto_entry_gate",
                        pd.DataFrame()
                    )

                    if gate_auto is not None and not gate_auto.empty:
                        ac=gate_auto["P7E_Entry_State"].value_counts()

                        ar1,ar2,ar3,ar4=st.columns(4)
                        ar1.metric("AUTO READY",int(ac.get("READY",0)))
                        ar2.metric("AUTO WAIT",int(ac.get("WAIT",0)))
                        ar3.metric(
                            "AUTO INVALIDATED",
                            int(ac.get("INVALIDATED",0))
                        )
                        ar4.metric(
                            "Last Auto Refresh",
                            pd.Timestamp.now().strftime("%H:%M:%S")
                        )

                    if monitored_auto is not None and not monitored_auto.empty:
                        if auto_paper_enabled:
                            try:
                                gated_auto=st.session_state.get(
                                    "phase7h1_gated_trades",
                                    pd.DataFrame()
                                ).copy()

                                # 1) Update existing OPEN paper trades from latest LTP/state.
                                gated_auto=update_phase7h_paper_trades(
                                    gated_auto,
                                    monitored_auto
                                )

                                # 2) Admit new READY candidates through Phase 7H.1.
                                gated_auto,auto_admission=(
                                    apply_phase7h1_portfolio_risk_gate(
                                        monitored_auto,
                                        existing_trades=gated_auto,
                                        risk_per_trade=float(auto_paper_risk),
                                        target_r_multiple=float(auto_paper_rr),
                                        max_open_positions=int(
                                            auto_paper_max_positions
                                        ),
                                        max_portfolio_heat=float(
                                            auto_paper_max_heat
                                        ),
                                        max_sector_positions=int(
                                            auto_paper_sector_limit
                                        ),
                                    )
                                )

                                st.session_state[
                                    "phase7h1_gated_trades"
                                ]=gated_auto
                                st.session_state[
                                    "auto_paper_admission_log"
                                ]=auto_admission

                                open_auto=int(
                                    (
                                        gated_auto["Paper_Status"]=="OPEN"
                                    ).sum()
                                ) if (
                                    not gated_auto.empty
                                    and "Paper_Status" in gated_auto.columns
                                ) else 0

                                closed_auto=int(
                                    (
                                        gated_auto["Paper_Status"]=="CLOSED"
                                    ).sum()
                                ) if (
                                    not gated_auto.empty
                                    and "Paper_Status" in gated_auto.columns
                                ) else 0

                                heat_auto=pd.to_numeric(
                                    gated_auto.loc[
                                        gated_auto["Paper_Status"]=="OPEN",
                                        "Risk_Budget"
                                    ],
                                    errors="coerce"
                                ).fillna(0).sum() if (
                                    not gated_auto.empty
                                    and "Paper_Status" in gated_auto.columns
                                    and "Risk_Budget" in gated_auto.columns
                                ) else 0.0

                                realized_auto=pd.to_numeric(
                                    gated_auto.get(
                                        "Realized_PnL",
                                        pd.Series(dtype=float)
                                    ),
                                    errors="coerce"
                                ).fillna(0).sum() if not gated_auto.empty else 0.0

                                pr1,pr2,pr3,pr4=st.columns(4)
                                pr1.metric(
                                    "AUTO PAPER OPEN",
                                    open_auto
                                )
                                pr2.metric(
                                    "AUTO PAPER CLOSED",
                                    closed_auto
                                )
                                pr3.metric(
                                    "AUTO PAPER HEAT",
                                    f"₹{heat_auto:,.0f}"
                                )
                                pr4.metric(
                                    "AUTO REALIZED P&L",
                                    f"₹{realized_auto:,.0f}"
                                )

                                if (
                                    auto_admission is not None
                                    and not auto_admission.empty
                                ):
                                    admitted_now=int(
                                        (
                                            auto_admission["Decision"]
                                            =="ADMIT"
                                        ).sum()
                                    )

                                    if admitted_now:
                                        st.success(
                                            f"{admitted_now} new paper trade(s) "
                                            "admitted by Phase 7H.1."
                                        )

                            except Exception as e:
                                st.error(
                                    f"Automatic paper-trading loop failed: {e}"
                                )

                        # -------------------------------------------------
                        # PHASE 8D — AUTOMATIC JOURNAL BRIDGE
                        # -------------------------------------------------
                        try:
                            research_auto=build_research_snapshot(
                                phase7d=st.session_state.get(
                                    "phase7d_final_ranking"
                                ),
                                phase7e=st.session_state.get(
                                    "phase7f_auto_entry_gate"
                                ),
                                phase7g=st.session_state.get(
                                    "phase7g_current_monitor"
                                ),
                                gated_trades=st.session_state.get(
                                    "phase7h1_gated_trades"
                                ),
                                admission_log=st.session_state.get(
                                    "auto_paper_admission_log",
                                    st.session_state.get(
                                        "phase7h1_admission_log"
                                    )
                                ),
                            )

                            auto_ts=pd.Timestamp.now()
                            cycle_key=(
                                "AUTO_LIVE:"
                                + auto_ts.strftime("%Y-%m-%d:%H:%M")
                            )

                            # Save the wide research state once per minute.
                            n_research,status_research=append_snapshot_once(
                                "PHASE8D_AUTO_RESEARCH",
                                research_auto,
                                capture_key=cycle_key+":RESEARCH",
                                snapshot_time=auto_ts,
                            )

                            # Save state transitions only when there are new ones.
                            if (
                                new_log_fast is not None
                                and not new_log_fast.empty
                            ):
                                append_snapshot_once(
                                    "PHASE8D_AUTO_TRANSITIONS",
                                    new_log_fast,
                                    capture_key=cycle_key+":TRANSITIONS",
                                    snapshot_time=auto_ts,
                                )

                            # Save current paper portfolio snapshot if enabled.
                            if auto_paper_enabled:
                                auto_port=st.session_state.get(
                                    "phase7h1_gated_trades",
                                    pd.DataFrame()
                                )

                                if (
                                    auto_port is not None
                                    and not auto_port.empty
                                ):
                                    append_snapshot_once(
                                        "PHASE8D_AUTO_PAPER",
                                        auto_port,
                                        capture_key=cycle_key+":PAPER",
                                        snapshot_time=auto_ts,
                                    )

                            st.session_state[
                                "phase8d_last_auto_journal_time"
                            ]=auto_ts

                        except Exception as e:
                            st.warning(
                                f"Automatic journal save skipped: {e}"
                            )

                        alert_mask=monitored_auto[
                            "P7G_Transition"
                        ].isin([
                            "WAIT -> READY",
                            "READY -> WAIT",
                            "READY -> INVALIDATED",
                            "INVALIDATED -> READY",
                        ])

                        alerts=monitored_auto[
                            alert_mask
                        ].copy()

                        if not alerts.empty:
                            st.warning(
                                f"{len(alerts)} meaningful state change(s) detected."
                            )

                            alert_cols=[
                                "Stock","P7D_Final_Action",
                                "P7G_Previous_State",
                                "P7E_Entry_State",
                                "P7G_Transition",
                                "P7E_Timing_Score",
                                "Live_RF","RVOL_Same_Time",
                                "LTP","VWAP","IB_High","IB_Low",
                                "P7E_Why"
                            ]

                            st.dataframe(
                                alerts[[
                                    c for c in alert_cols
                                    if c in alerts.columns
                                ]],
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.success(
                                "Auto monitor active — no meaningful state change."
                            )

                except Exception as e:
                    st.error(f"Automatic Fast Refresh failed: {e}")

            _auto_fast_fragment()

            if auto_paper_enabled:
                auto_portfolio=st.session_state.get(
                    "phase7h1_gated_trades",
                    pd.DataFrame()
                ).copy()

                if not auto_portfolio.empty:
                    st.markdown("#### Automatic Paper Portfolio")

                    auto_cols=[
                        "Trade_ID","Open_Time","Stock","Sector",
                        "Paper_Side","Entry_Price","Stop_Price",
                        "Target_Price","Quantity","Risk_Budget",
                        "P7D_Score","P7D_Conviction",
                        "P7E_Timing_Score","Live_RF",
                        "RVOL_Same_Time","Paper_Status",
                        "Exit_Time","Exit_Price","Exit_Reason",
                        "Realized_PnL","Realized_R"
                    ]

                    st.dataframe(
                        auto_portfolio[[
                            c for c in auto_cols
                            if c in auto_portfolio.columns
                        ]],
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Download Automatic Paper Portfolio",
                        data=auto_portfolio.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="auto_phase7h1_paper_portfolio.csv",
                        mime="text/csv"
                    )

            if "phase8d_last_auto_journal_time" in st.session_state:
                st.caption(
                    "Last automatic journal save: "
                    + pd.Timestamp(
                        st.session_state[
                            "phase8d_last_auto_journal_time"
                        ]
                    ).strftime("%Y-%m-%d %H:%M:%S")
                )
        else:
            st.warning(
                "This Streamlit version does not support timed fragments. "
                "Use manual FAST LIVE REFRESH or upgrade Streamlit."
            )

    if not st.session_state.access_token:
        st.warning("Connect Kite first before using Full or Fast Live Mode.")
    else:
        kite.set_access_token(st.session_state.access_token)

        if st.button(
            "FAST LIVE REFRESH",
            type="primary",
            key="fast_live_refresh"
        ):
            fast_status=st.empty()

            try:
                fast_status.write(
                    "Refreshing live timing for cached Phase-7D candidates..."
                )

                monitored_fast,fast_err=_run_fast_live_cycle()

                if fast_err:
                    st.error(fast_err)
                else:
                    gate_fast=st.session_state[
                        "phase7f_auto_entry_gate"
                    ].copy()

                    sc=gate_fast[
                        "P7E_Entry_State"
                    ].value_counts()

                    st.success(
                        "FAST LIVE REFRESH COMPLETE ✅ — "
                        f"READY {int(sc.get('READY',0))} | "
                        f"WAIT {int(sc.get('WAIT',0))} | "
                        f"INVALIDATED {int(sc.get('INVALIDATED',0))}"
                    )

            except Exception as e:
                st.error(f"Fast Live Refresh failed: {e}")

            finally:
                fast_status.empty()

        st.divider()
        st.markdown("## Full Institutional Refresh")
        st.caption(
            "Use this heavier refresh when you want to rebuild RF, RS, Futures, "
            "Options 40 and Phase-7D from scratch."
        )

        if st.button(
            "FULL REFRESH: RF → RS → FUTURES → OPTIONS 40 → LIVE TIMING",
            type="secondary",
            key="oneclick_full_institutional_run"
        ):
            overall=st.progress(0)
            stage=st.empty()
            detail=st.empty()

            try:
                # -------------------------------------------------
                # STAGE 0 — Common F&O cash universe
                # -------------------------------------------------
                stage.write("Stage 0/5 — Loading F&O universe...")
                universe=get_fno_equity_universe(kite).copy()

                if universe.empty:
                    raise RuntimeError("F&O cash universe is empty.")

                st.session_state["fno_universe"]=universe.copy()
                overall.progress(0.03)

                # -------------------------------------------------
                # STAGE 1 — RF ALL
                # -------------------------------------------------
                stage.write("Stage 1/5 — RF ALL")
                rf_rows=[]
                rf_errors=[]

                to_date=dt.datetime.now()
                from_date=to_date-dt.timedelta(days=int(one_rf_days))
                total=len(universe)

                for n,row in enumerate(universe.itertuples(index=False),start=1):
                    detail.write(f"RF ALL: {row.Stock} — {n}/{total}")
                    success=False
                    last_error="Unknown error"

                    for attempt in range(1,3):
                        try:
                            raw=pd.DataFrame(
                                kite.historical_data(
                                    int(row.instrument_token),
                                    from_date,
                                    to_date,
                                    "30minute",
                                    oi=False
                                )
                            )

                            if raw.empty:
                                last_error="No 30-minute historical data returned"
                            else:
                                feat=stock_rf_features(
                                    daily_rf_summary(
                                        calculate_intraday_rf(raw)
                                    )
                                )

                                if feat:
                                    feat["Stock"]=row.Stock
                                    feat["Instrument_Token"]=int(
                                        row.instrument_token
                                    )
                                    rf_rows.append(feat)
                                    success=True
                                    break

                                last_error="RF feature calculation returned no result"

                        except Exception as e:
                            last_error=str(e)

                        time.sleep(0.75)

                    if not success:
                        rf_errors.append([row.Stock,last_error])

                    overall.progress(
                        min(0.03 + 0.22*(n/total),0.25)
                    )
                    time.sleep(0.38)

                if not rf_rows:
                    raise RuntimeError("RF ALL returned zero successful stocks.")

                rf_result=add_rf_rank_scores(pd.DataFrame(rf_rows))
                st.session_state["rf_result"]=rf_result
                st.session_state["rf_requested"]=total
                st.session_state["rf_errors"]=pd.DataFrame(
                    rf_errors,
                    columns=["Stock","Error"]
                )

                rf_coverage=len(rf_result)/total if total else 0

                if rf_coverage < 0.95:
                    raise RuntimeError(
                        f"RF coverage only {rf_coverage*100:.1f}%. "
                        "Need at least 95% before continuing."
                    )

                # -------------------------------------------------
                # STAGE 2 — STOCK + SECTOR RS ALL
                # -------------------------------------------------
                stage.write("Stage 2/5 — Stock + Sector RS ALL")

                sector_map=pd.read_csv(
                    Path(__file__).parent/"data"/"sector_map.csv"
                )

                rs_universe=universe.merge(
                    sector_map,
                    on="Stock",
                    how="left"
                )
                rs_universe["Sector"]=rs_universe[
                    "Sector"
                ].fillna("UNKNOWN")

                to_date=dt.datetime.now()
                from_date=to_date-dt.timedelta(days=int(one_rs_days))

                nifty=pd.DataFrame(
                    kite.historical_data(
                        256265,
                        from_date,
                        to_date,
                        "day",
                        oi=False
                    )
                )

                rs_rows=[]
                total_rs=len(rs_universe)

                for n,row in enumerate(
                    rs_universe.itertuples(index=False),
                    start=1
                ):
                    detail.write(
                        f"Stock + Sector RS ALL: {row.Stock} — "
                        f"{n}/{total_rs}"
                    )

                    try:
                        raw=pd.DataFrame(
                            kite.historical_data(
                                int(row.instrument_token),
                                from_date,
                                to_date,
                                "day",
                                oi=False
                            )
                        )

                        feat=calculate_rs_features(raw,nifty)

                        if feat:
                            feat["Stock"]=row.Stock
                            feat["Sector"]=row.Sector
                            feat["Instrument_Token"]=int(
                                row.instrument_token
                            )
                            rs_rows.append(feat)

                    except Exception:
                        pass

                    overall.progress(
                        min(0.25 + 0.20*(n/total_rs),0.45)
                    )
                    time.sleep(0.38)

                if not rs_rows:
                    raise RuntimeError(
                        "Stock + Sector RS ALL returned zero successful stocks."
                    )

                stock_rs=percentile_scores(pd.DataFrame(rs_rows))
                sector_rs=sector_scores(stock_rs)

                if not sector_rs.empty:
                    stock_rs=stock_rs.merge(
                        sector_rs[[
                            "Sector",
                            "Sector_RS_Score",
                            "Sector_RS_Acceleration",
                            "Sector_Rank"
                        ]],
                        on="Sector",
                        how="left"
                    )

                stock_rs["Alignment"]=stock_rs.apply(
                    lambda r: alignment_label(
                        r["Stock_RS_Score"],
                        r["Sector_RS_Score"]
                    ),
                    axis=1
                )

                rfcols=rf_result[[
                    "Stock",
                    "RF_Strength_Score",
                    "Latest_RF",
                    "Avg_RF_5D",
                    "RF_Acceleration",
                    "Latest_%Change"
                ]].copy()

                stock_rs=stock_rs.merge(
                    rfcols,
                    on="Stock",
                    how="left"
                )

                rf_pct=stock_rs[
                    "RF_Strength_Score"
                ].rank(pct=True)*100

                stock_rs["RF_RS_Alignment_Score"]=(
                    0.45*stock_rs["Stock_RS_Score"]
                    +0.30*stock_rs["Sector_RS_Score"]
                    +0.25*rf_pct
                ).round(1)

                st.session_state["stock_rs_result"]=stock_rs
                st.session_state["sector_rs_result"]=sector_rs

                rs_complete=stock_rs[[
                    "Stock_RS_Score",
                    "Sector_RS_Score",
                    "RF_RS_Alignment_Score"
                ]].notna().all(axis=1).mean()

                if rs_complete < 0.95:
                    raise RuntimeError(
                        f"RS complete coverage only {rs_complete*100:.1f}%. "
                        "Need at least 95% before continuing."
                    )

                # -------------------------------------------------
                # STAGE 3 — FUTURES ALL
                # -------------------------------------------------
                stage.write("Stage 3/5 — Futures ALL")

                fut=get_nearest_futures_map(kite)
                combo=universe.merge(fut,on="Stock",how="inner")
                st.session_state["futures_universe"]=combo.copy()

                cash_keys=[
                    f"NSE:{x}" for x in combo["Stock"].tolist()
                ]

                try:
                    cash_quotes=kite.quote(cash_keys)
                except Exception:
                    cash_quotes={}

                to_date=dt.datetime.now()
                from_date=to_date-dt.timedelta(
                    days=int(one_fut_days)
                )

                fut_rows=[]
                fut_errors=[]
                total_fut=len(combo)

                for n,row in enumerate(
                    combo.itertuples(index=False),
                    start=1
                ):
                    detail.write(
                        f"Futures ALL: {row.Stock} — {n}/{total_fut}"
                    )

                    try:
                        spot_quote=cash_quotes.get(
                            f"NSE:{row.Stock}",
                            {}
                        )
                        spot=float(
                            spot_quote.get("last_price",0) or 0
                        )

                        hist=pd.DataFrame(
                            kite.historical_data(
                                int(row.Future_Token),
                                from_date,
                                to_date,
                                "day",
                                oi=True
                            )
                        )

                        feat=futures_features(spot,hist)

                        if feat:
                            feat["Stock"]=row.Stock
                            feat["Future_Symbol"]=row.Future_Symbol
                            feat["Future_Expiry"]=row.Future_Expiry
                            feat["Lot_Size"]=row.Lot_Size
                            feat["Spot_Price"]=round(spot,2)
                            fut_rows.append(feat)
                        else:
                            fut_errors.append([
                                row.Stock,
                                "Insufficient futures history"
                            ])

                    except Exception as e:
                        fut_errors.append([row.Stock,str(e)])

                    overall.progress(
                        min(0.45 + 0.18*(n/total_fut),0.63)
                    )
                    time.sleep(0.38)

                if not fut_rows:
                    raise RuntimeError(
                        "Futures ALL returned zero successful stocks."
                    )

                fdf=add_futures_score(pd.DataFrame(fut_rows))
                fdf=add_futures_conviction(fdf)

                rscols=stock_rs[[
                    "Stock",
                    "Stock_RS_Score",
                    "Sector_RS_Score",
                    "RF_RS_Alignment_Score"
                ]].copy()

                fdf=fdf.merge(
                    rscols,
                    on="Stock",
                    how="left"
                )

                rfcols2=rf_result[[
                    "Stock",
                    "Latest_RF",
                    "RF_Strength_Score"
                ]].copy()

                fdf=fdf.merge(
                    rfcols2,
                    on="Stock",
                    how="left"
                )

                fdf["RF_RS_Futures_Score"]=(
                    0.65*fdf[
                        "RF_RS_Alignment_Score"
                    ].fillna(50)
                    +0.35*fdf[
                        "Futures_Score"
                    ].fillna(50)
                ).round(1)

                fdf=add_institutional_alignment(fdf)

                st.session_state["futures_result"]=fdf
                st.session_state["futures_errors"]=pd.DataFrame(
                    fut_errors,
                    columns=["Stock","Error"]
                )

                # -------------------------------------------------
                # STAGE 4 — PHASE 7 CORE TOP/BOTTOM 40
                # -------------------------------------------------
                stage.write(
                    "Stage 4/5 — Building Phase-7 Top 20 Bullish + "
                    "Bottom 20 Bearish queue"
                )

                core,bull_core,bear_core,opt_queue=(
                    build_phase7_core_funnel(
                        rf_result,
                        stock_rs,
                        fdf,
                        bullish_n=20,
                        bearish_n=20,
                    )
                )

                st.session_state["phase7_core_ranking"]=core
                st.session_state["phase7_core_bullish"]=bull_core
                st.session_state["phase7_core_bearish"]=bear_core
                st.session_state["phase7_options_queue"]=opt_queue

                overall.progress(0.67)

                # -------------------------------------------------
                # STAGE 5 — OPTIONS FULL 40
                # -------------------------------------------------
                stage.write("Stage 5/5 — Heavy Options full 40")

                # Build nearest option-expiry map only once here.
                nfo_master=pd.DataFrame(kite.instruments("NFO"))
                fno_opt_universe=build_fno_options_universe(
                    nfo_master
                )
                stock_opt_universe=stock_options_universe(
                    nfo_master
                )

                st.session_state[
                    "fno_options_universe"
                ]=fno_opt_universe
                st.session_state[
                    "fno_stock_options_universe"
                ]=stock_opt_universe

                expmap=stock_opt_universe[[
                    "Underlying",
                    "Nearest_Expiry"
                ]].copy().rename(
                    columns={
                        "Underlying":"Stock",
                        "Nearest_Expiry":"Expiry"
                    }
                )

                p7_queue=opt_queue.copy()
                p7_queue["Queue_Side"]=p7_queue["Funnel_Side"]
                p7_queue["Queue_Priority"]=p7_queue[
                    "Funnel_Priority"
                ]
                p7_queue[
                    "Normalized_Options_Score"
                ]=p7_queue["Phase7_Core_Score"]

                p7_queue=p7_queue.merge(
                    expmap,
                    on="Stock",
                    how="left"
                )

                missing_expiry=p7_queue[
                    "Expiry"
                ].isna().sum()

                if missing_expiry:
                    raise RuntimeError(
                        f"{missing_expiry} Phase-7 candidates "
                        "have no mapped options expiry."
                    )

                opt_progress=st.progress(0)

                def _oneclick_opt_progress(i,total,stock):
                    detail.write(
                        f"Heavy Options 40: {stock} — {i}/{total}"
                    )
                    opt_progress.progress(i/total)
                    overall.progress(
                        min(0.67 + 0.31*(i/total),0.98)
                    )

                hr40,_=run_heavy_queue_batch(
                    kite,
                    p7_queue,
                    bullish_n=20,
                    bearish_n=20,
                    strikes_each_side=int(one_heavy_wings),
                    risk_free_rate_pct=float(one_rate),
                    keep_details=False,
                    progress_callback=_oneclick_opt_progress,
                )

                st.session_state[
                    "heavy_options_40_result"
                ]=hr40

                final_options=build_final_options_confirmation_ranking(
                    hr40
                )

                st.session_state[
                    "final_options_confirmation"
                ]=final_options

                final7c=build_phase7c_confirmed_ranking(
                    core,
                    hr40,
                )
                st.session_state[
                    "phase7c_heavy_options_result"
                ]=hr40
                st.session_state[
                    "phase7c_final_ranking"
                ]=final7c

                p7d=build_phase7d_conflict_conviction(
                    final7c
                )
                st.session_state[
                    "phase7d_final_ranking"
                ]=p7d
                st.session_state["full_refresh_time"]=pd.Timestamp.now()
                st.session_state["rf_cache_time"]=pd.Timestamp.now()
                st.session_state["rs_cache_time"]=pd.Timestamp.now()
                st.session_state["futures_cache_time"]=pd.Timestamp.now()
                st.session_state["options_cache_time"]=pd.Timestamp.now()

                # -------------------------------------------------
                # STAGE 6 — AUTOMATIC LIVE TIMING (PHASE 7F.2)
                # -------------------------------------------------
                stage.write(
                    "Stage 6/7 — Live RF + RVOL + VWAP + IB "
                    "for Phase-7D tradeable candidates"
                )

                live_candidates=p7d[
                    p7d["P7D_Final_Action"].isin(
                        ["LONG","LONG WATCH","SHORT","SHORT WATCH"]
                    )
                ][["Stock"]].drop_duplicates().copy()

                if live_candidates.empty:
                    raise RuntimeError(
                        "Phase 7D produced no tradeable candidates "
                        "for live timing."
                    )

                detail.write(
                    f"Collecting live timing for {len(live_candidates)} "
                    "Phase-7D candidates..."
                )

                live_auto=build_phase7f_live_feed(
                    kite,
                    live_candidates,
                    interval=one_live_interval,
                    ib_minutes=int(one_ib_minutes),
                    rvol_lookback_sessions=int(one_rvol_sessions),
                )

                st.session_state["phase7f_live_feed"]=live_auto

                auto_gate=build_phase7e_live_entry_gate(
                    p7d,
                    live_auto,
                )

                st.session_state[
                    "phase7f_auto_entry_gate"
                ]=auto_gate

                overall.progress(0.995)

                # -------------------------------------------------
                # STAGE 7 — PHASE 7G STATE MONITOR
                # -------------------------------------------------
                stage.write(
                    "Stage 7/7 — Candidate state transitions"
                )

                previous_snapshot=st.session_state.get(
                    "phase7g_previous_gate_snapshot"
                )

                monitored,new_log=build_phase7g_state_transitions(
                    auto_gate,
                    previous_snapshot
                )

                st.session_state[
                    "phase7g_current_monitor"
                ]=monitored

                if "phase7g_transition_log" not in st.session_state:
                    st.session_state[
                        "phase7g_transition_log"
                    ]=pd.DataFrame()

                if new_log is not None and not new_log.empty:
                    st.session_state[
                        "phase7g_transition_log"
                    ]=pd.concat(
                        [
                            st.session_state["phase7g_transition_log"],
                            new_log
                        ],
                        ignore_index=True
                    )

                st.session_state[
                    "phase7g_previous_gate_snapshot"
                ]=auto_gate.copy()

                overall.progress(1.0)
                detail.empty()
                stage.empty()

                state_counts=auto_gate[
                    "P7E_Entry_State"
                ].value_counts()

                st.success(
                    "FULL INSTITUTIONAL REFRESH COMPLETE ✅ — "
                    f"RF {len(rf_result)}/{total} | "
                    f"RS {len(stock_rs)} | "
                    f"Futures {len(fdf)} | "
                    f"Heavy Options {len(hr40)}/40 | "
                    f"READY {int(state_counts.get('READY',0))} | "
                    f"WAIT {int(state_counts.get('WAIT',0))} | "
                    f"INVALIDATED {int(state_counts.get('INVALIDATED',0))}"
                )

            except Exception as e:
                detail.empty()
                st.error(
                    f"One-Click Full Institutional Run stopped: {e}"
                )

    if "fast_live_refresh_time" in st.session_state:
        fast_ts=pd.Timestamp(
            st.session_state["fast_live_refresh_time"]
        )
        st.caption(
            "Latest Fast Live Refresh: "
            + fast_ts.strftime("%Y-%m-%d %H:%M:%S")
        )

    if "phase7d_final_ranking" in st.session_state:
        oneclick_result=st.session_state[
            "phase7d_final_ranking"
        ].copy()

        st.markdown("### Latest Cached Institutional Ranking")

        oc_counts=oneclick_result[
            "P7D_Final_Action"
        ].value_counts()

        m1,m2,m3,m4,m5,m6=st.columns(6)
        m1.metric("LONG",int(oc_counts.get("LONG",0)))
        m2.metric(
            "LONG WATCH",
            int(oc_counts.get("LONG WATCH",0))
        )
        m3.metric("NEUTRAL",int(oc_counts.get("NEUTRAL",0)))
        m4.metric(
            "SHORT WATCH",
            int(oc_counts.get("SHORT WATCH",0))
        )
        m5.metric("SHORT",int(oc_counts.get("SHORT",0)))
        m6.metric("AVOID",int(oc_counts.get("AVOID",0)))

        oneclick_cols=[
            "P7D_Rank","Stock","Sector",
            "P7D_Institutional_Score",
            "P7D_Final_Action",
            "P7D_Adjusted_Conviction",
            "P7D_Conviction_Grade",
            "P7_RF_Score",
            "P7_Sector_RS_Score",
            "P7_Stock_RS_Score",
            "P7_Futures_Score",
            "P7_Options_Score",
            "Institutional_Signal",
            "Lightweight_Heavy_Alignment",
            "P7D_Soft_Conflict_Flags",
            "P7D_Hard_Veto_Flags",
        ]

        st.dataframe(
            oneclick_result[[
                c for c in oneclick_cols
                if c in oneclick_result.columns
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Latest Institutional Ranking",
            data=oneclick_result.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="oneclick_final_institutional_ranking.csv",
            mime="text/csv"
        )

    if "phase7f_auto_entry_gate" in st.session_state:
        oneclick_gate=st.session_state[
            "phase7f_auto_entry_gate"
        ].copy()

        st.markdown("### One-Click Live Timing Results")

        gate_counts=oneclick_gate[
            "P7E_Entry_State"
        ].value_counts()

        g1,g2,g3=st.columns(3)
        g1.metric("READY",int(gate_counts.get("READY",0)))
        g2.metric("WAIT",int(gate_counts.get("WAIT",0)))
        g3.metric(
            "INVALIDATED",
            int(gate_counts.get("INVALIDATED",0))
        )

        live_cols=[
            "Stock","Sector","P7D_Final_Action",
            "P7D_Institutional_Score",
            "P7D_Adjusted_Conviction",
            "P7E_Direction","P7E_Timing_Score",
            "P7E_Participation_State",
            "P7E_Entry_State",
            "Live_RF","RVOL_Same_Time",
            "LTP","VWAP","IB_High","IB_Low",
            "Live_Data_Status","P7E_Why"
        ]

        st.dataframe(
            oneclick_gate[[
                c for c in live_cols
                if c in oneclick_gate.columns
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download One-Click Live Timing CSV",
            data=oneclick_gate.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="oneclick_live_timing_gate.csv",
            mime="text/csv"
        )

    if "phase7g_current_monitor" in st.session_state:
        monitor=st.session_state[
            "phase7g_current_monitor"
        ].copy()

        st.markdown("### One-Click Candidate Monitor")

        changed=int(
            monitor["P7G_State_Changed"].sum()
        ) if "P7G_State_Changed" in monitor.columns else 0

        wait_ready=int(
            (
                monitor["P7G_Transition"]
                =="WAIT -> READY"
            ).sum()
        ) if "P7G_Transition" in monitor.columns else 0

        ready_wait=int(
            (
                monitor["P7G_Transition"]
                =="READY -> WAIT"
            ).sum()
        ) if "P7G_Transition" in monitor.columns else 0

        ready_invalid=int(
            (
                monitor["P7G_Transition"]
                =="READY -> INVALIDATED"
            ).sum()
        ) if "P7G_Transition" in monitor.columns else 0

        q1,q2,q3,q4=st.columns(4)
        q1.metric("State Changes",changed)
        q2.metric("WAIT → READY",wait_ready)
        q3.metric("READY → WAIT",ready_wait)
        q4.metric(
            "READY → INVALIDATED",
            ready_invalid
        )

        monitor_cols=[
            "Stock","P7D_Final_Action",
            "P7G_Previous_State",
            "P7E_Entry_State",
            "P7G_Transition",
            "P7E_Timing_Score",
            "Live_RF","RVOL_Same_Time",
            "LTP","VWAP","IB_High","IB_Low",
            "P7E_Why"
        ]

        st.dataframe(
            monitor[[
                c for c in monitor_cols
                if c in monitor.columns
            ]],
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    required_p7=[
        "rf_result",
        "stock_rs_result",
        "futures_result",
        "final_options_confirmation",
    ]
    missing_p7=[k for k in required_p7 if k not in st.session_state]

    if missing_p7:
        st.warning(
            "Run the live modules in this session before building Phase 7: "
            "RF ALL → Stock + Sector RS ALL → Futures ALL → Options full 40."
        )
        st.write("Missing:",", ".join(missing_p7))
    else:
        st.markdown("### Phase 7B — Two-Stage Institutional Funnel")
        st.caption(
            "Stage 1 ranks the full universe using RF + Sector RS + "
            "Stock RS + Futures. Heavy Options is then reserved for "
            "the strongest 20 bullish + 20 bearish candidates."
        )

        if st.button("Build Phase 7 Core Funnel",type="primary"):
            try:
                core,bull_core,bear_core,opt_queue=build_phase7_core_funnel(
                    st.session_state["rf_result"],
                    st.session_state["stock_rs_result"],
                    st.session_state["futures_result"],
                    bullish_n=20,
                    bearish_n=20,
                )
                st.session_state["phase7_core_ranking"]=core
                st.session_state["phase7_core_bullish"]=bull_core
                st.session_state["phase7_core_bearish"]=bear_core
                st.session_state["phase7_options_queue"]=opt_queue
            except Exception as e:
                st.error(f"Phase 7B core funnel failed: {e}")

        if "phase7_core_ranking" in st.session_state:
            core=st.session_state["phase7_core_ranking"].copy()
            bull_core=st.session_state["phase7_core_bullish"].copy()
            bear_core=st.session_state["phase7_core_bearish"].copy()
            opt_queue=st.session_state["phase7_options_queue"].copy()

            complete_core=int(core["Core_Data_Complete"].sum())

            c1,c2,c3=st.columns(3)
            c1.metric("4-Layer Complete Stocks",complete_core)
            c2.metric("Bullish Options Queue",len(bull_core))
            c3.metric("Bearish Options Queue",len(bear_core))

            core_cols=[
                "Core_Rank","Stock","Sector","Phase7_Core_Score",
                "Core_Direction","P7_RF_Score","P7_Sector_RS_Score",
                "P7_Stock_RS_Score","P7_Futures_Score",
                "Institutional_Signal","Core_Data_Layers"
            ]

            st.markdown("#### Full-Universe Core Ranking")
            st.dataframe(
                core[[c for c in core_cols if c in core.columns]],
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Phase 7 Core Ranking CSV",
                data=core.to_csv(index=False).encode("utf-8"),
                file_name="phase7_core_full_universe.csv",
                mime="text/csv"
            )

            st.download_button(
                "Download Phase 7 Options Confirmation Queue",
                data=opt_queue.to_csv(index=False).encode("utf-8"),
                file_name="phase7_options_confirmation_queue.csv",
                mime="text/csv"
            )

            st.markdown("### Phase 7C — Heavy Options Confirmation")

            c7a,c7b=st.columns(2)
            with c7a:
                p7c_wings=st.slider(
                    "Phase 7C strikes each side ATM",
                    min_value=8,
                    max_value=15,
                    value=12,
                    step=1,
                    key="p7c_wings"
                )
            with c7b:
                p7c_rate=st.number_input(
                    "Phase 7C risk-free rate %",
                    min_value=0.0,
                    max_value=20.0,
                    value=6.5,
                    step=0.1,
                    key="p7c_rate"
                )

            if st.button("Run Phase 7C Heavy Confirmation",type="primary"):
                p7c_progress=st.progress(0)
                p7c_status=st.empty()

                def _p7c_progress(i,total,stock):
                    p7c_status.write(
                        f"Phase 7C heavy options: {stock} — {i}/{total}"
                    )
                    p7c_progress.progress(i/total)

                try:
                    # Reuse the validated 20 bullish + 20 bearish heavy engine.
                    p7c_queue=opt_queue.copy()

                    # Translate Phase-7 funnel fields into the heavy-options queue schema.
                    p7c_queue["Queue_Side"]=p7c_queue["Funnel_Side"]
                    p7c_queue["Queue_Priority"]=p7c_queue["Funnel_Priority"]
                    p7c_queue["Normalized_Options_Score"]=p7c_queue[
                        "Phase7_Core_Score"
                    ]

                    # Heavy engine expects Expiry. Pull it from the current
                    # lightweight options universe if available.
                    if "fno_stock_options_universe" in st.session_state:
                        expmap=st.session_state[
                            "fno_stock_options_universe"
                        ][["Underlying","Nearest_Expiry"]].copy()
                        expmap=expmap.rename(
                            columns={
                                "Underlying":"Stock",
                                "Nearest_Expiry":"Expiry"
                            }
                        )
                        p7c_queue=p7c_queue.merge(
                            expmap,
                            on="Stock",
                            how="left"
                        )
                    else:
                        st.error(
                            "Load F&O Options Universe first so Phase 7C "
                            "can map nearest option expiries."
                        )
                        st.stop()

                    hr7c,_=run_heavy_queue_batch(
                        kite,
                        p7c_queue,
                        bullish_n=20,
                        bearish_n=20,
                        strikes_each_side=int(p7c_wings),
                        risk_free_rate_pct=float(p7c_rate),
                        keep_details=False,
                        progress_callback=_p7c_progress,
                    )

                    final7c=build_phase7c_confirmed_ranking(
                        core,
                        hr7c,
                    )

                    st.session_state["phase7c_heavy_options_result"]=hr7c
                    st.session_state["phase7c_final_ranking"]=final7c

                except Exception as e:
                    st.error(f"Phase 7C failed: {e}")

                finally:
                    p7c_status.empty()

            if "phase7c_final_ranking" in st.session_state:
                final7c=st.session_state[
                    "phase7c_final_ranking"
                ].copy()

                dcounts=final7c["Final_Phase7_Decision"].value_counts()

                z1,z2,z3,z4,z5=st.columns(5)
                z1.metric("LONG",int(dcounts.get("LONG",0)))
                z2.metric("LONG WATCH",int(dcounts.get("LONG WATCH",0)))
                z3.metric("NEUTRAL",int(dcounts.get("NEUTRAL",0)))
                z4.metric("SHORT WATCH",int(dcounts.get("SHORT WATCH",0)))
                z5.metric("SHORT",int(dcounts.get("SHORT",0)))

                final_cols=[
                    "Final_Phase7_Rank","Stock","Sector",
                    "Final_Phase7_Score","Final_Phase7_Decision",
                    "Final_Phase7_Conviction",
                    "Phase7_Core_Score",
                    "P7_RF_Score","P7_Sector_RS_Score",
                    "P7_Stock_RS_Score","P7_Futures_Score",
                    "P7_Options_Score",
                    "Institutional_Signal",
                    "Heavy_Options_Bias",
                    "Lightweight_Heavy_Alignment",
                    "Options_Confirmation_State",
                    "Gamma_Regime","Zero_Gamma_Level",
                    "Final_Phase7_Veto_Flags"
                ]

                st.markdown("#### Final Phase 7 Confirmed Ranking")
                st.dataframe(
                    final7c[[
                        c for c in final_cols
                        if c in final7c.columns
                    ]],
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Final Phase 7 Confirmed Ranking",
                    data=final7c.to_csv(index=False).encode("utf-8"),
                    file_name="phase7c_final_confirmed_ranking.csv",
                    mime="text/csv"
                )

                st.markdown("### Phase 7D — Conflict & Conviction Engine")
                st.caption(
                    "The institutional score is preserved unchanged. "
                    "Hard vetoes invalidate a setup; soft conflicts reduce "
                    "conviction instead of automatically forcing AVOID."
                )

                p7d=build_phase7d_conflict_conviction(final7c)
                st.session_state["phase7d_final_ranking"]=p7d

                ac=p7d["P7D_Final_Action"].value_counts()
                q1,q2,q3,q4,q5,q6=st.columns(6)
                q1.metric("LONG",int(ac.get("LONG",0)))
                q2.metric("LONG WATCH",int(ac.get("LONG WATCH",0)))
                q3.metric("NEUTRAL",int(ac.get("NEUTRAL",0)))
                q4.metric("SHORT WATCH",int(ac.get("SHORT WATCH",0)))
                q5.metric("SHORT",int(ac.get("SHORT",0)))
                q6.metric("AVOID",int(ac.get("AVOID",0)))

                p7d_cols=[
                    "P7D_Rank","Stock","Sector",
                    "P7D_Institutional_Score","P7D_Final_Action",
                    "P7D_Adjusted_Conviction","P7D_Conviction_Grade",
                    "P7D_Conflict_Penalty",
                    "P7_RF_Score","P7_Sector_RS_Score",
                    "P7_Stock_RS_Score","P7_Futures_Score",
                    "P7_Options_Score",
                    "Institutional_Signal",
                    "Lightweight_Heavy_Alignment",
                    "P7D_Soft_Conflict_Flags",
                    "P7D_Hard_Veto_Flags",
                    "Gamma_Regime","Zero_Gamma_Level"
                ]

                st.dataframe(
                    p7d[[c for c in p7d_cols if c in p7d.columns]],
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Phase 7D Conflict & Conviction Ranking",
                    data=p7d.to_csv(index=False).encode("utf-8"),
                    file_name="phase7d_conflict_conviction_ranking.csv",
                    mime="text/csv"
                )

                st.markdown("### Phase 7E — Live Entry & Timing Gate")
                st.caption(
                    "Paper-trading gate only. Phase 7D selects the candidate; "
                    "7E requires live RF, VWAP/location, volume participation "
                    "and Initial Balance evidence before READY."
                )

                live_candidates=p7d[
                    p7d["P7D_Final_Action"].isin(
                        ["LONG","LONG WATCH","SHORT","SHORT WATCH"]
                    )
                ][["Stock"]].drop_duplicates().copy()

                st.download_button(
                    "Download Phase 7E Live Input Template",
                    data=live_candidates.assign(
                        Live_RF=np.nan,
                        LTP=np.nan,
                        VWAP=np.nan,
                        Day_Volume=np.nan,
                        Avg_Volume_Same_Time=np.nan,
                        IB_High=np.nan,
                        IB_Low=np.nan,
                        Open=np.nan,
                        Day_High=np.nan,
                        Day_Low=np.nan
                    ).to_csv(index=False).encode("utf-8"),
                    file_name="phase7e_live_input_template.csv",
                    mime="text/csv"
                )

                p7e_file=st.file_uploader(
                    "Upload live Phase 7E input CSV",
                    type=["csv"],
                    key="phase7e_live_upload"
                )

                if p7e_file is not None:
                    try:
                        live7e=pd.read_csv(p7e_file)
                        p7e=build_phase7e_live_entry_gate(p7d,live7e)
                        st.session_state["phase7e_entry_gate"]=p7e

                        ec=p7e["P7E_Entry_State"].value_counts()
                        e1,e2,e3=st.columns(3)
                        e1.metric("READY",int(ec.get("READY",0)))
                        e2.metric("WAIT",int(ec.get("WAIT",0)))
                        e3.metric("INVALIDATED",int(ec.get("INVALIDATED",0)))

                        p7e_cols=[
                            "Stock","Sector","P7D_Final_Action",
                            "P7D_Institutional_Score",
                            "P7D_Adjusted_Conviction",
                            "P7E_Direction","P7E_Timing_Score",
                            "P7E_Participation_State","P7E_Entry_State",
                            "Live_RF","LTP","VWAP",
                            "Day_Volume","Avg_Volume_Same_Time",
                            "IB_High","IB_Low","P7E_Why"
                        ]

                        st.dataframe(
                            p7e[[c for c in p7e_cols if c in p7e.columns]],
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download Phase 7E Entry Gate Results",
                            data=p7e.to_csv(index=False).encode("utf-8"),
                            file_name="phase7e_live_entry_gate.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"Phase 7E input failed: {e}")

                st.markdown("### Phase 7F — Automatic Live Data Feed")
                st.caption(
                    "Automatically builds the Phase 7E timing input from Kite "
                    "for current Phase-7D candidates. Phase 7F.2 requires both "
                    "directional RF confirmation and minimum same-time RVOL >=0.75x "
                    "before READY. Paper/observation use only."
                )

                f1,f2,f3=st.columns(3)
                with f1:
                    live_interval=st.selectbox(
                        "Live candle interval",
                        ["5minute","15minute"],
                        index=0,
                        key="p7f_interval"
                    )
                with f2:
                    ib_minutes=st.selectbox(
                        "Initial Balance duration",
                        [30,45,60],
                        index=2,
                        key="p7f_ib_minutes"
                    )
                with f3:
                    rvol_sessions=st.selectbox(
                        "RVOL lookback sessions",
                        [10,15,20],
                        index=2,
                        key="p7f_rvol_sessions"
                    )

                if st.button("Build Automatic Live Feed",type="primary"):
                    try:
                        auto_candidates=p7d[
                            p7d["P7D_Final_Action"].isin(
                                ["LONG","LONG WATCH","SHORT","SHORT WATCH"]
                            )
                        ][["Stock"]].drop_duplicates().copy()

                        with st.spinner(
                            "Collecting live quotes and intraday candles..."
                        ):
                            live_auto=build_phase7f_live_feed(
                                kite,
                                auto_candidates,
                                interval=live_interval,
                                ib_minutes=int(ib_minutes),
                                rvol_lookback_sessions=int(rvol_sessions),
                            )

                        st.session_state["phase7f_live_feed"]=live_auto

                        auto_gate=build_phase7e_live_entry_gate(
                            p7d,
                            live_auto,
                        )
                        st.session_state["phase7f_auto_entry_gate"]=auto_gate

                    except Exception as e:
                        st.error(f"Phase 7F live feed failed: {e}")

                if "phase7f_live_feed" in st.session_state:
                    live_auto=st.session_state[
                        "phase7f_live_feed"
                    ].copy()

                    ok_feed=int(
                        (live_auto["Live_Data_Status"]=="OK").sum()
                    ) if "Live_Data_Status" in live_auto.columns else 0

                    g1,g2=st.columns(2)
                    g1.metric("Live Candidates",len(live_auto))
                    g2.metric("Live Data OK",ok_feed)

                    st.dataframe(
                        live_auto,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Download Automatic Live Feed CSV",
                        data=live_auto.to_csv(index=False).encode("utf-8"),
                        file_name="phase7f_automatic_live_feed.csv",
                        mime="text/csv"
                    )

                if "phase7f_auto_entry_gate" in st.session_state:
                    auto_gate=st.session_state[
                        "phase7f_auto_entry_gate"
                    ].copy()

                    ec=auto_gate["P7E_Entry_State"].value_counts()

                    a1,a2,a3=st.columns(3)
                    a1.metric("AUTO READY",int(ec.get("READY",0)))
                    a2.metric("AUTO WAIT",int(ec.get("WAIT",0)))
                    a3.metric(
                        "AUTO INVALIDATED",
                        int(ec.get("INVALIDATED",0))
                    )

                    auto_cols=[
                        "Stock","Sector","P7D_Final_Action",
                        "P7D_Institutional_Score",
                        "P7D_Adjusted_Conviction",
                        "P7E_Timing_Score","P7E_Participation_State",
                        "P7E_Entry_State","Live_RF","LTP","VWAP","Day_Volume",
                        "Avg_Volume_Same_Time","IB_High","IB_Low",
                        "Live_Data_Status","P7E_Why"
                    ]

                    st.dataframe(
                        auto_gate[[
                            c for c in auto_cols
                            if c in auto_gate.columns
                        ]],
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Download Automatic Phase 7E Gate Results",
                        data=auto_gate.to_csv(index=False).encode("utf-8"),
                        file_name="phase7f_auto_entry_gate.csv",
                        mime="text/csv"
                    )

                    st.markdown("### Phase 7G — Candidate Monitor & State Transitions")
                    st.caption(
                        "Tracks WAIT / READY / INVALIDATED changes between "
                        "successive live snapshots. Monitoring only — no orders."
                    )

                    previous_snapshot = st.session_state.get(
                        "phase7g_previous_gate_snapshot"
                    )

                    monitored, new_log = build_phase7g_state_transitions(
                        auto_gate,
                        previous_snapshot
                    )

                    st.session_state["phase7g_current_monitor"] = monitored

                    # Append only actual state changes to the running log.
                    if "phase7g_transition_log" not in st.session_state:
                        st.session_state["phase7g_transition_log"] = pd.DataFrame()

                    if new_log is not None and not new_log.empty:
                        old_log = st.session_state["phase7g_transition_log"]
                        st.session_state["phase7g_transition_log"] = pd.concat(
                            [old_log, new_log],
                            ignore_index=True
                        )

                    st.session_state[
                        "phase7g_previous_gate_snapshot"
                    ] = auto_gate.copy()

                    changed_count = int(
                        monitored["P7G_State_Changed"].sum()
                    ) if "P7G_State_Changed" in monitored.columns else 0

                    wait_ready = int(
                        (monitored["P7G_Transition"]=="WAIT -> READY").sum()
                    ) if "P7G_Transition" in monitored.columns else 0

                    ready_wait = int(
                        (monitored["P7G_Transition"]=="READY -> WAIT").sum()
                    ) if "P7G_Transition" in monitored.columns else 0

                    ready_invalid = int(
                        (
                            monitored["P7G_Transition"]
                            =="READY -> INVALIDATED"
                        ).sum()
                    ) if "P7G_Transition" in monitored.columns else 0

                    m1,m2,m3,m4=st.columns(4)
                    m1.metric("State Changes",changed_count)
                    m2.metric("WAIT → READY",wait_ready)
                    m3.metric("READY → WAIT",ready_wait)
                    m4.metric("READY → INVALIDATED",ready_invalid)

                    monitor_cols=[
                        "Stock","P7D_Final_Action",
                        "P7D_Institutional_Score",
                        "P7D_Adjusted_Conviction",
                        "P7G_Previous_State","P7E_Entry_State",
                        "P7G_Transition","P7E_Timing_Score",
                        "P7E_Participation_State","Live_RF",
                        "RVOL_Same_Time","LTP","VWAP",
                        "IB_High","IB_Low","P7E_Why"
                    ]

                    st.dataframe(
                        monitored[[
                            c for c in monitor_cols
                            if c in monitored.columns
                        ]],
                        use_container_width=True,
                        hide_index=True
                    )

                    transition_log = st.session_state[
                        "phase7g_transition_log"
                    ].copy()

                    if not transition_log.empty:
                        st.markdown("#### Phase 7G Transition Log")
                        st.dataframe(
                            transition_log,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download Phase 7G Transition Log",
                            data=transition_log.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="phase7g_transition_log.csv",
                            mime="text/csv"
                        )

                    st.markdown("### Phase 7H — Paper Trading Engine")
                    st.caption(
                        "Simulation only. No Zerodha orders are placed. "
                        "Trades open only from READY candidates."
                    )

                    ph1,ph2=st.columns(2)
                    with ph1:
                        paper_risk=st.number_input(
                            "Paper risk budget per trade (₹)",
                            min_value=100.0,
                            max_value=100000.0,
                            value=1000.0,
                            step=100.0,
                            key="p7h_risk"
                        )
                    with ph2:
                        paper_rr=st.number_input(
                            "Paper target R multiple",
                            min_value=1.0,
                            max_value=5.0,
                            value=2.0,
                            step=0.5,
                            key="p7h_rr"
                        )

                    if "phase7h_paper_trades" not in st.session_state:
                        st.session_state["phase7h_paper_trades"]=pd.DataFrame()

                    # Phase 7H.1 uses a completely isolated portfolio state.
                    # This prevents the original Phase 7H test trades from
                    # contaminating portfolio-risk validation.
                    if "phase7h1_gated_trades" not in st.session_state:
                        st.session_state["phase7h1_gated_trades"]=pd.DataFrame()

                    reset_col1,reset_col2=st.columns([1,3])
                    with reset_col1:
                        if st.button(
                            "Reset Phase 7H.1 Portfolio",
                            key="p7h1_reset_portfolio"
                        ):
                            st.session_state["phase7h1_gated_trades"]=pd.DataFrame()
                            st.session_state.pop(
                                "phase7h1_admission_log",
                                None
                            )
                            st.success(
                                "Phase 7H.1 portfolio reset to 0 positions / ₹0 heat."
                            )

                    # Phase 7H display still uses its original paper portfolio.
                    paper_trades=st.session_state[
                        "phase7h_paper_trades"
                    ].copy()

                    # Independent state used only by Phase 7H.1.
                    gated_paper_trades=st.session_state[
                        "phase7h1_gated_trades"
                    ].copy()

                    paper_trades=update_phase7h_paper_trades(
                        paper_trades,
                        monitored
                    )

                    if st.button("Create Paper Trades From READY"):
                        paper_trades=build_phase7h_paper_trades(
                            monitored,
                            existing_trades=paper_trades,
                            risk_per_trade=float(paper_risk),
                            target_r_multiple=float(paper_rr),
                        )

                    st.session_state[
                        "phase7h_paper_trades"
                    ]=paper_trades

                    if not paper_trades.empty:
                        open_count=int(
                            (paper_trades["Paper_Status"]=="OPEN").sum()
                        )
                        closed_count=int(
                            (paper_trades["Paper_Status"]=="CLOSED").sum()
                        )

                        realized_pnl=pd.to_numeric(
                            paper_trades.get(
                                "Realized_PnL",
                                pd.Series(dtype=float)
                            ),
                            errors="coerce"
                        ).fillna(0).sum()

                        realized_r=pd.to_numeric(
                            paper_trades.get(
                                "Realized_R",
                                pd.Series(dtype=float)
                            ),
                            errors="coerce"
                        ).dropna()

                        avg_r=float(realized_r.mean()) if len(realized_r) else 0.0

                        h1,h2,h3,h4=st.columns(4)
                        h1.metric("Open Paper Trades",open_count)
                        h2.metric("Closed Paper Trades",closed_count)
                        h3.metric("Realized Paper P&L",f"₹{realized_pnl:,.0f}")
                        h4.metric("Average Realized R",f"{avg_r:.2f}R")

                        st.dataframe(
                            paper_trades,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download Phase 7H Paper Trades",
                            data=paper_trades.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="phase7h_paper_trades.csv",
                            mime="text/csv"
                        )

                    st.markdown("### Phase 7H.1 — Portfolio Risk Gate")
                    st.caption(
                        "Paper-trading portfolio controls. Admission is based on "
                        "conviction priority, max positions, portfolio heat and sector concentration."
                    )

                    pr1,pr2,pr3=st.columns(3)
                    with pr1:
                        max_positions=st.number_input(
                            "Max open paper positions",
                            min_value=1,
                            max_value=20,
                            value=5,
                            step=1,
                            key="p7h1_max_positions"
                        )
                    with pr2:
                        max_heat=st.number_input(
                            "Max portfolio heat (₹)",
                            min_value=500.0,
                            max_value=100000.0,
                            value=5000.0,
                            step=500.0,
                            key="p7h1_max_heat"
                        )
                    with pr3:
                        max_sector=st.number_input(
                            "Max positions per sector",
                            min_value=1,
                            max_value=10,
                            value=2,
                            step=1,
                            key="p7h1_max_sector"
                        )

                    if st.button("Apply Portfolio Risk Gate"):
                        if (
                            not gated_paper_trades.empty
                            and "Paper_Status" in gated_paper_trades.columns
                            and (gated_paper_trades["Paper_Status"]=="OPEN").any()
                        ):
                            st.info(
                                "Existing OPEN Phase 7H.1 trades will count toward "
                                "position, heat and sector limits. Use Reset Phase 7H.1 "
                                "Portfolio for a clean validation run."
                            )

                        gated_trades,admission_log=apply_phase7h1_portfolio_risk_gate(
                            monitored,
                            existing_trades=gated_paper_trades,
                            risk_per_trade=float(paper_risk),
                            target_r_multiple=float(paper_rr),
                            max_open_positions=int(max_positions),
                            max_portfolio_heat=float(max_heat),
                            max_sector_positions=int(max_sector),
                        )

                        st.session_state["phase7h1_gated_trades"]=gated_trades
                        st.session_state["phase7h1_admission_log"]=admission_log

                    if "phase7h1_admission_log" in st.session_state:
                        admission_log=st.session_state[
                            "phase7h1_admission_log"
                        ].copy()

                        if not admission_log.empty:
                            admitted=int(
                                (admission_log["Decision"]=="ADMIT").sum()
                            )
                            rejected=int(
                                (admission_log["Decision"]=="REJECT").sum()
                            )

                            open_now=int(
                                (
                                    st.session_state["phase7h1_gated_trades"][
                                        "Paper_Status"
                                    ]=="OPEN"
                                ).sum()
                            ) if not st.session_state[
                                "phase7h1_gated_trades"
                            ].empty else 0

                            heat_now=pd.to_numeric(
                                st.session_state[
                                    "phase7h1_gated_trades"
                                ].loc[
                                    st.session_state[
                                        "phase7h1_gated_trades"
                                    ]["Paper_Status"]=="OPEN",
                                    "Risk_Budget"
                                ],
                                errors="coerce"
                            ).fillna(0).sum() if not st.session_state[
                                "phase7h1_gated_trades"
                            ].empty else 0.0

                            x1,x2,x3,x4=st.columns(4)
                            x1.metric("Admitted",admitted)
                            x2.metric("Rejected",rejected)
                            x3.metric("Open Positions",open_now)
                            x4.metric("Portfolio Heat",f"₹{heat_now:,.0f}")

                            st.dataframe(
                                admission_log,
                                use_container_width=True,
                                hide_index=True
                            )

                            st.download_button(
                                "Download Phase 7H.1 Admission Log",
                                data=admission_log.to_csv(
                                    index=False
                                ).encode("utf-8"),
                                file_name="phase7h1_portfolio_risk_admission_log.csv",
                                mime="text/csv"
                            )

                            gated_view=st.session_state[
                                "phase7h1_gated_trades"
                            ].copy()

                            if not gated_view.empty:
                                st.download_button(
                                    "Download Phase 7H.1 Gated Paper Portfolio",
                                    data=gated_view.to_csv(
                                        index=False
                                    ).encode("utf-8"),
                                    file_name="phase7h1_gated_paper_portfolio.csv",
                                    mime="text/csv"
                                )

        st.markdown("### Phase 7A — Five-Layer Confirmed Ranking")

        if st.button("Build Phase 7 Master Score"):
            try:
                p7=build_master_institutional_score(
                    st.session_state["rf_result"],
                    st.session_state["stock_rs_result"],
                    st.session_state["futures_result"],
                    st.session_state["final_options_confirmation"],
                )
                st.session_state["phase7_master_result"]=p7
            except Exception as e:
                st.error(f"Phase 7 integration failed: {e}")

        if "phase7_master_result" in st.session_state:
            p7=st.session_state["phase7_master_result"].copy()

            complete=int(p7["P7_Data_Complete"].sum())
            longs=int((p7["P7_Decision"]=="LONG").sum())
            shorts=int((p7["P7_Decision"]=="SHORT").sum())
            watch=int(p7["P7_Decision"].isin(["LONG WATCH","SHORT WATCH"]).sum())

            p1,p2,p3,p4=st.columns(4)
            p1.metric("Complete 5-Layer Stocks",complete)
            p2.metric("LONG",longs)
            p3.metric("SHORT",shorts)
            p4.metric("WATCH",watch)

            cols=[
                "P7_Rank","Stock","Sector",
                "Institutional_Composite_Score","P7_Decision","P7_Conviction",
                "P7_RF_Score","P7_Sector_RS_Score","P7_Stock_RS_Score",
                "P7_Futures_Score","P7_Options_Score",
                "Institutional_Signal","Final_Options_Classification",
                "P7_Data_Layers","P7_Data_Complete","P7_Veto_Flags"
            ]

            st.markdown("### Phase 7 Institutional Ranking")
            st.dataframe(
                p7[[c for c in cols if c in p7.columns]],
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Phase 7 Master Ranking CSV",
                data=p7.to_csv(index=False).encode("utf-8"),
                file_name="phase7_master_institutional_ranking.csv",
                mime="text/csv"
            )


with tabs[13]:
    st.subheader("Persistent Institutional Journal — Phase 8A")
    st.caption(
        "Stores Phase-7 rankings, state transitions, paper trades and portfolio-risk "
        "admissions in a lightweight SQLite journal."
    )

    db_path = init_journal_db()

    st.warning(
        "Streamlit Community Cloud local storage is not guaranteed to survive app "
        "redeploys/reboots. Use Download Journal DB regularly. The same SQLite file "
        "will be persistent when this app is later run on your local/server storage."
    )

    j1,j2,j3 = st.columns(3)
    j1.metric("Journal DB", db_path)
    j2.metric("DB Size", f"{database_size_bytes()/1024:.1f} KB")

    counts = snapshot_counts()
    total_rows = int(counts["rows"].sum()) if not counts.empty else 0
    j3.metric("Stored Rows", total_rows)

    st.markdown("### Save Current System State")

    save_cols = st.columns(4)

    with save_cols[0]:
        if st.button("Save Phase 7D Ranking", key="p8_save_p7d"):
            df = st.session_state.get("phase7d_final_ranking")
            n = append_snapshot("PHASE7D_RANKING", df)
            st.success(f"Saved {n} Phase 7D rows.")

    with save_cols[1]:
        if st.button("Save Phase 7G Log", key="p8_save_p7g"):
            df = st.session_state.get("phase7g_transition_log")
            n = append_snapshot("PHASE7G_TRANSITIONS", df)
            st.success(f"Saved {n} transition rows.")

    with save_cols[2]:
        if st.button("Save Gated Trades", key="p8_save_p7h1"):
            df = st.session_state.get("phase7h1_gated_trades")
            n = append_snapshot("PHASE7H1_GATED_TRADES", df)
            st.success(f"Saved {n} gated-trade rows.")

    with save_cols[3]:
        if st.button("Save Admission Log", key="p8_save_admission"):
            df = st.session_state.get("phase7h1_admission_log")
            n = append_snapshot("PHASE7H1_ADMISSION", df)
            st.success(f"Saved {n} admission rows.")

    if st.button("Save Full Phase 8A Snapshot", type="primary"):
        saved = {}
        for typ, key in [
            ("PHASE7D_RANKING", "phase7d_final_ranking"),
            ("PHASE7G_TRANSITIONS", "phase7g_transition_log"),
            ("PHASE7H1_GATED_TRADES", "phase7h1_gated_trades"),
            ("PHASE7H1_ADMISSION", "phase7h1_admission_log"),
        ]:
            saved[typ] = append_snapshot(
                typ,
                st.session_state.get(key)
            )

        st.success(
            "Snapshot saved: "
            + " | ".join(f"{k}={v}" for k, v in saved.items())
        )

    st.markdown("### Journal Summary")
    counts = snapshot_counts()
    st.dataframe(counts, use_container_width=True, hide_index=True)

    st.markdown("### Phase 8D — Automatic Journal Bridge")
    st.caption(
        "This is the bridge from the live Phase-7 loop into the SQLite journal. "
        "It automatically saves live research, state transitions and paper-portfolio "
        "snapshots whenever Automatic Fast Refresh runs."
    )

    p8d_counts = snapshot_counts()

    def _p8d_count(snapshot_type):
        if p8d_counts.empty:
            return 0
        rows = p8d_counts.loc[
            p8d_counts["snapshot_type"] == snapshot_type,
            "rows"
        ]
        return int(rows.iloc[0]) if len(rows) else 0

    d1,d2,d3,d4 = st.columns(4)
    d1.metric(
        "Auto Research Rows",
        _p8d_count("PHASE8D_AUTO_RESEARCH")
    )
    d2.metric(
        "Auto Transition Rows",
        _p8d_count("PHASE8D_AUTO_TRANSITIONS")
    )
    d3.metric(
        "Auto Paper Rows",
        _p8d_count("PHASE8D_AUTO_PAPER")
    )

    last_auto_save = st.session_state.get(
        "phase8d_last_auto_journal_time"
    )

    d4.metric(
        "Last Auto Save",
        (
            pd.Timestamp(last_auto_save).strftime("%H:%M:%S")
            if last_auto_save is not None
            else "Not yet"
        )
    )

    if (
        _p8d_count("PHASE8D_AUTO_RESEARCH")==0
        and _p8d_count("PHASE8D_AUTO_TRANSITIONS")==0
        and _p8d_count("PHASE8D_AUTO_PAPER")==0
    ):
        st.info(
            "No Phase 8D automatic journal records yet. "
            "Go to Phase 7 Master, enable Automatic Fast Refresh, and let at least "
            "one refresh cycle complete."
        )
    else:
        p8d_type = st.selectbox(
            "View Phase 8D dataset",
            [
                "PHASE8D_AUTO_RESEARCH",
                "PHASE8D_AUTO_TRANSITIONS",
                "PHASE8D_AUTO_PAPER",
            ],
            key="p8d_view_type"
        )

        p8d_hist = read_snapshots(
            snapshot_type=p8d_type,
            limit=5000,
        )

        st.dataframe(
            p8d_hist,
            use_container_width=True,
            hide_index=True
        )

        if not p8d_hist.empty:
            st.download_button(
                "Download Selected Phase 8D Data",
                data=p8d_hist.to_csv(index=False).encode("utf-8"),
                file_name=f"{p8d_type.lower()}.csv",
                mime="text/csv"
            )

    st.markdown("### Phase 8B — Automated Research Snapshots")
    st.caption(
        "Scheduled research capture on Streamlit reruns. "
        "The app does not run in the background; if it reruns within the "
        "configured time window, the scheduled slot is saved once."
    )

    slot_text = ", ".join(DEFAULT_CAPTURE_SLOTS)
    st.write("**Default NSE research checkpoints:**", slot_text)

    tol = st.slider(
        "Scheduled capture tolerance (minutes)",
        min_value=2,
        max_value=15,
        value=7,
        step=1,
        key="p8b_tolerance"
    )

    research_now = build_research_snapshot(
        phase7d=st.session_state.get("phase7d_final_ranking"),
        phase7e=st.session_state.get("phase7f_auto_entry_gate"),
        phase7g=st.session_state.get("phase7g_current_monitor"),
        gated_trades=st.session_state.get("phase7h1_gated_trades"),
        admission_log=st.session_state.get("phase7h1_admission_log"),
    )

    r1,r2,r3 = st.columns(3)
    r1.metric("Current Research Rows", len(research_now))

    due = due_capture_slot(
        tolerance_minutes=int(tol)
    )

    if due:
        r2.metric("Due Slot", due["slot"])
        r3.metric("Distance", f'{due["delta_minutes"]:.1f} min')
    else:
        r2.metric("Due Slot", "NONE")
        r3.metric("Distance", "—")

    # Automatic save-on-rerun if a scheduled slot is currently due.
    if due and not research_now.empty:
        key = capture_key_for_slot(due["slot"])
        n_auto, status_auto = append_snapshot_once(
            "PHASE8B_RESEARCH",
            research_now,
            capture_key=key,
        )

        if status_auto == "SAVED":
            st.success(
                f'Automatic Phase 8B capture saved for {due["slot"]}: '
                f'{n_auto} rows.'
            )
        elif status_auto == "ALREADY_CAPTURED":
            st.info(
                f'Phase 8B slot {due["slot"]} was already captured today.'
            )

    if st.button("Capture Research Snapshot Now", type="primary"):
        manual_key = (
            "MANUAL:"
            + pd.Timestamp.now().strftime("%Y-%m-%d:%H:%M:%S")
        )
        n_manual, status_manual = append_snapshot_once(
            "PHASE8B_RESEARCH",
            research_now,
            capture_key=manual_key,
        )
        st.success(
            f"Manual research snapshot: {status_manual} / {n_manual} rows."
        )

    if not research_now.empty:
        st.markdown("#### Current Research Dataset Preview")
        st.dataframe(
            research_now,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Current Research Snapshot CSV",
            data=research_now.to_csv(index=False).encode("utf-8"),
            file_name="phase8b_current_research_snapshot.csv",
            mime="text/csv"
        )

    registry = capture_registry()

    if not registry.empty:
        st.markdown("#### Scheduled Capture Registry")
        st.dataframe(
            registry,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("### Inspect Stored Data")

    available_types = (
        counts["snapshot_type"].tolist()
        if not counts.empty
        else []
    )

    if available_types:
        selected_type = st.selectbox(
            "Snapshot type",
            available_types,
            key="p8_snapshot_type"
        )
        stock_filter = st.text_input(
            "Optional stock filter",
            "",
            key="p8_stock_filter"
        )

        hist = read_snapshots(
            snapshot_type=selected_type,
            stock=stock_filter.strip().upper() or None,
            limit=5000,
        )

        st.dataframe(
            hist,
            use_container_width=True,
            hide_index=True
        )

        if not hist.empty:
            st.download_button(
                "Download Selected Journal Data CSV",
                data=hist.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_type.lower()}_journal.csv",
                mime="text/csv"
            )

    st.markdown("### Backup / Restore")

    st.download_button(
        "Download Journal SQLite DB",
        data=export_db_bytes(),
        file_name="institutional_market_journal.db",
        mime="application/octet-stream"
    )

    restore_file = st.file_uploader(
        "Restore journal from SQLite DB backup",
        type=["db", "sqlite", "sqlite3"],
        key="p8_restore_db"
    )

    if restore_file is not None:
        if st.button("Restore Uploaded Journal DB"):
            try:
                restore_db_bytes(restore_file.getvalue())
                st.success("Journal DB restored successfully.")
            except Exception as e:
                st.error(f"Journal restore failed: {e}")

    with st.expander("Danger Zone"):
        confirm_clear = st.checkbox(
            "I understand this clears the local journal",
            key="p8_confirm_clear"
        )
        if st.button(
            "Clear Local Journal",
            disabled=not confirm_clear,
            key="p8_clear_db"
        ):
            clear_journal_db()
            st.success("Local journal cleared.")


with tabs[14]:
    st.subheader("Performance & Expectancy Analytics — Phase 8C")
    st.caption(
        "Research framework only. Metrics become statistically meaningful only "
        "after enough CLOSED paper trades accumulate."
    )

    # Current gated paper trades
    current_trades = st.session_state.get(
        "phase7h1_gated_trades",
        pd.DataFrame()
    )

    # Stored Phase 8B research rows
    research_hist = read_snapshots(
        snapshot_type="PHASE8B_RESEARCH",
        limit=5000,
    )

    enriched = merge_trade_context(
        current_trades,
        research_hist
    )

    closed_count = 0
    if not enriched.empty and "Paper_Status" in enriched.columns:
        closed_count = int(
            (enriched["Paper_Status"] == "CLOSED").sum()
        )

    a1,a2,a3 = st.columns(3)
    a1.metric("Paper Trades", len(enriched))
    a2.metric("Closed Trades", closed_count)
    a3.metric(
        "Research Rows",
        len(research_hist)
    )

    if enriched.empty:
        st.info(
            "No Phase 7H.1 gated paper trades are available yet."
        )
    else:
        st.markdown("### Enriched Trade Dataset")
        st.dataframe(
            enriched,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Enriched Trade Dataset",
            data=enriched.to_csv(index=False).encode("utf-8"),
            file_name="phase8c_enriched_trade_dataset.csv",
            mime="text/csv"
        )

        tables = build_expectancy_dashboard(enriched)

        overall = tables.get("overall", pd.DataFrame())

        if overall.empty:
            st.warning(
                "There are no CLOSED trades with Realized_R yet. "
                "Phase 8C is installed correctly, but expectancy metrics "
                "will stay blank until exits occur."
            )
        else:
            st.markdown("### Overall Paper Performance")
            st.dataframe(
                overall,
                use_container_width=True,
                hide_index=True
            )

            row = overall.iloc[0]

            p1,p2,p3,p4,p5 = st.columns(5)
            p1.metric("Trades", int(row["Trades"]))
            p2.metric("Win Rate", f'{row["Win_Rate_%"]:.1f}%')
            p3.metric("Expectancy", f'{row["Expectancy_R"]:.2f}R')
            p4.metric(
                "Profit Factor",
                (
                    f'{row["Profit_Factor"]:.2f}'
                    if pd.notna(row["Profit_Factor"])
                    else "NA"
                )
            )
            p5.metric("Max DD", f'{row["Max_Drawdown_R"]:.2f}R')

            group_names = {
                "Paper_Side": "LONG vs SHORT",
                "Sector": "Sector Performance",
                "Score_Band": "Institutional Score Bands",
                "Conviction_Band": "Conviction Bands",
                "RVOL_Band": "RVOL Bands",
                "Timing_Band": "Timing Score Bands",
                "P7D_Final_Action": "Phase 7D Action",
                "P7E_Participation_State": "Participation State",
            }

            for key, title in group_names.items():
                table = tables.get(key, pd.DataFrame())

                if table is not None and not table.empty:
                    st.markdown(f"### {title}")
                    st.dataframe(
                        table,
                        use_container_width=True,
                        hide_index=True
                    )

    st.markdown("### Phase 8C Interpretation Guardrail")
    st.info(
        "Do not optimize RF, RVOL, timing-score or composite-score thresholds "
        "from a handful of trades. First accumulate a materially larger sample "
        "across different market regimes, sectors and both LONG/SHORT directions."
    )

    st.divider()
    st.subheader("Phase 8D — Automatic Journal Bridge Status")

    p8d_research = read_snapshots(
        snapshot_type="PHASE8D_AUTO_RESEARCH",
        limit=10000,
    )
    p8d_transitions = read_snapshots(
        snapshot_type="PHASE8D_AUTO_TRANSITIONS",
        limit=10000,
    )
    p8d_paper = read_snapshots(
        snapshot_type="PHASE8D_AUTO_PAPER",
        limit=10000,
    )

    ad1,ad2,ad3 = st.columns(3)
    ad1.metric("Auto Research Rows", len(p8d_research))
    ad2.metric("Auto Transition Rows", len(p8d_transitions))
    ad3.metric("Auto Paper Rows", len(p8d_paper))

    if p8d_research.empty:
        st.warning(
            "Phase 8D has not captured any automatic research rows yet. "
            "Phase 8E entry-context matching will show NO_PRIOR_SNAPSHOT "
            "until at least one Phase 8D auto-refresh snapshot exists before a new trade entry."
        )
    else:
        st.success(
            "Phase 8D automatic journal data is available for Phase 8E entry-time matching."
        )

        with st.expander("Preview latest Phase 8D research rows"):
            st.dataframe(
                p8d_research.head(100),
                use_container_width=True,
                hide_index=True
            )

    st.divider()
    st.subheader("Precise Entry Context + Equity Curve — Phase 8E")
    st.caption(
        "Matches each paper trade to the nearest research snapshot at or before "
        "its entry time, then builds realized-R and P&L equity curves from CLOSED trades."
    )

    entry_lookback=st.slider(
        "Maximum allowed entry-context age (minutes)",
        min_value=5,
        max_value=120,
        value=60,
        step=5,
        key="p8e_context_age"
    )

    p8e_trades=st.session_state.get(
        "phase7h1_gated_trades",
        pd.DataFrame()
    )

    p8e_research=read_snapshots(
        snapshot_type="PHASE8D_AUTO_RESEARCH",
        limit=10000,
    )

    if p8e_research.empty:
        p8e_research=read_snapshots(
            snapshot_type="PHASE8B_RESEARCH",
            limit=10000,
        )

    p8e_enriched=build_phase8e_entry_context_dataset(
        p8e_trades,
        p8e_research,
        max_lookback_minutes=int(entry_lookback),
    )

    if p8e_enriched.empty:
        st.info(
            "No gated paper trades are available for Phase 8E yet."
        )
    else:
        match_counts=p8e_enriched[
            "Entry_Context_Status"
        ].value_counts()

        e1,e2,e3=st.columns(3)
        e1.metric(
            "Matched Entry Context",
            int(match_counts.get("MATCHED",0))
        )
        e2.metric(
            "Stale Context",
            int(match_counts.get("STALE_PRIOR_SNAPSHOT",0))
        )
        e3.metric(
            "No Prior Snapshot",
            int(match_counts.get("NO_PRIOR_SNAPSHOT",0))
        )

        st.markdown("#### Entry-Time Enriched Trades")
        st.dataframe(
            p8e_enriched,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Phase 8E Entry Context Dataset",
            data=p8e_enriched.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="phase8e_entry_context_dataset.csv",
            mime="text/csv"
        )

        curve,curve_summary=build_phase8e_equity_curve(
            p8e_enriched
        )

        if curve.empty:
            st.warning(
                "No CLOSED trades with Realized_R yet. "
                "Entry-context matching is active; equity curves will populate "
                "automatically after paper trades close."
            )
        else:
            st.markdown("#### Realized Performance Summary")
            st.dataframe(
                curve_summary,
                use_container_width=True,
                hide_index=True
            )

            sr=curve_summary.iloc[0]

            c1,c2,c3,c4=st.columns(4)
            c1.metric(
                "Closed Trades",
                int(sr["Closed_Trades"])
            )
            c2.metric(
                "Total R",
                f'{sr["Total_R"]:.2f}R'
            )
            c3.metric(
                "Max DD",
                f'{sr["Max_Drawdown_R"]:.2f}R'
            )
            c4.metric(
                "Realized P&L",
                f'₹{sr["Total_Realized_PnL"]:,.0f}'
            )

            st.markdown("#### Realized R Equity Curve")
            st.line_chart(
                curve.set_index("Trade_Number")[
                    ["Cumulative_R"]
                ]
            )

            st.markdown("#### Drawdown in R")
            st.line_chart(
                curve.set_index("Trade_Number")[
                    ["Drawdown_R"]
                ]
            )

            st.markdown("#### Realized ₹ P&L Curve")
            st.line_chart(
                curve.set_index("Trade_Number")[
                    ["Cumulative_PnL"]
                ]
            )

            st.download_button(
                "Download Phase 8E Equity Curve CSV",
                data=curve.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="phase8e_equity_curve.csv",
                mime="text/csv"
            )

    st.divider()
    st.subheader("Daily Institutional Session Report — Phase 8F")
    st.caption(
        "Builds a compact daily scorecard from Phase 8D automatic research, "
        "state transitions and paper-trade records."
    )

    p8f_date = st.date_input(
        "Session date",
        value=pd.Timestamp.now().date(),
        key="p8f_session_date"
    )

    p8f_research = read_snapshots(
        snapshot_type="PHASE8D_AUTO_RESEARCH",
        limit=20000,
    )
    p8f_transitions = read_snapshots(
        snapshot_type="PHASE8D_AUTO_TRANSITIONS",
        limit=20000,
    )
    p8f_paper = read_snapshots(
        snapshot_type="PHASE8D_AUTO_PAPER",
        limit=20000,
    )

    p8f_summary,p8f_latest,p8f_tr,p8f_paper_day = (
        build_phase8f_daily_session_report(
            p8f_research,
            p8f_transitions,
            p8f_paper,
            session_date=p8f_date,
        )
    )

    srow = p8f_summary.iloc[0]

    f1,f2,f3,f4,f5 = st.columns(5)
    f1.metric("Stocks", int(srow["Latest_Stocks"]))
    f2.metric("READY", int(srow["READY"]))
    f3.metric("WAIT", int(srow["WAIT"]))
    f4.metric("INVALIDATED", int(srow["INVALIDATED"]))
    f5.metric("State Changes", int(srow["State_Changes"]))

    f6,f7,f8,f9 = st.columns(4)
    f6.metric("Paper Open", int(srow["Paper_Open"]))
    f7.metric("Paper Closed", int(srow["Paper_Closed"]))
    f8.metric("Realized P&L", f'₹{srow["Realized_PnL"]:,.0f}')
    f9.metric("Realized R", f'{srow["Realized_R"]:.2f}R')

    st.markdown("#### Daily Session Summary")
    st.dataframe(
        p8f_summary,
        use_container_width=True,
        hide_index=True
    )

    if not p8f_latest.empty:
        st.markdown("#### Latest Stock State for Selected Session")

        latest_cols = [
            "Stock","Sector","P7D_Final_Action",
            "P7D_Institutional_Score",
            "P7D_Adjusted_Conviction",
            "P7E_Entry_State","P7E_Timing_Score",
            "P7E_Participation_State",
            "Live_RF","RVOL_Same_Time",
            "LTP","VWAP","IB_High","IB_Low",
            "P7G_Transition",
            "Paper_Status","Paper_Side",
            "Realized_PnL","Realized_R",
        ]

        st.dataframe(
            p8f_latest[[
                c for c in latest_cols
                if c in p8f_latest.columns
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Phase 8F Daily Stock States",
            data=p8f_latest.to_csv(index=False).encode("utf-8"),
            file_name=f"phase8f_daily_stock_states_{p8f_date}.csv",
            mime="text/csv"
        )

    st.download_button(
        "Download Phase 8F Session Summary",
        data=p8f_summary.to_csv(index=False).encode("utf-8"),
        file_name=f"phase8f_session_summary_{p8f_date}.csv",
        mime="text/csv"
    )


with tabs[15]:
    st.subheader("Phase 9 — 20-Day Historical Validation")
    st.caption(
        "Primary pre-deployment gate. Historical signals are frozen at or before 11:15, "
        "then post-11:15 prices are revealed separately. This prevents look-ahead bias."
    )

    st.info(
        "Leakage rule: 30-minute signal features may use bars only through the 10:45 start timestamp "
        "(that candle completes at 11:15). Post-signal performance is evaluated from 11:15 onward."
    )

    d1,d2,d3=st.columns(3)
    default_end=(pd.Timestamp.now().date()-dt.timedelta(days=1))
    default_start=default_end-dt.timedelta(days=35)
    with d1:
        p9_start=st.date_input("Validation start",value=default_start,key="p9_start")
    with d2:
        p9_end=st.date_input("Validation end",value=default_end,key="p9_end")
    with d3:
        p9_max_days=st.number_input("Maximum trading sessions",min_value=5,max_value=60,value=20,step=1,key="p9_days")

    source=st.radio(
        "Historical signal source",
        ["Phase 8 Journal", "Upload Research-Lab / Signal CSV"],
        horizontal=True,
        key="p9_source"
    )

    signals_raw=pd.DataFrame()
    source_note=""

    if source=="Phase 8 Journal":
        p9_auto=read_snapshots(snapshot_type="PHASE8D_AUTO_RESEARCH",limit=50000)
        if p9_auto.empty:
            p9_auto=read_snapshots(snapshot_type="PHASE8B_RESEARCH",limit=50000)
        signals_raw=p9_auto
        if signals_raw.empty:
            source_note="No historical Phase 8 research snapshots are stored yet. Use the upload option for older sessions."
        else:
            source_note=f"Loaded {len(signals_raw):,} research snapshot rows from the journal."
    else:
        p9_upload=st.file_uploader(
            "Upload historical signal/research CSV",
            type=["csv"],
            key="p9_upload"
        )
        if p9_upload is not None:
            try:
                signals_raw=pd.read_csv(p9_upload)
                source_note=f"Loaded {len(signals_raw):,} uploaded rows."
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    if source_note:
        st.write(source_note)

    signals,p9_msg=_p9_prepare_signals(signals_raw,p9_start,p9_end,cutoff="11:15")
    if p9_msg!="OK":
        st.warning(p9_msg)
    elif not signals.empty:
        # Keep only the most recent N actual sessions in the selected range.
        valid_days=sorted(pd.Series(signals["_P9_Date"].unique()).dropna().tolist())[-int(p9_max_days):]
        signals=signals[signals["_P9_Date"].isin(valid_days)].copy()

        c1,c2,c3,c4=st.columns(4)
        c1.metric("Sessions",len(valid_days))
        c2.metric("Frozen Signals",len(signals))
        c3.metric("Long",int((signals["P9_Side"]=="LONG").sum()))
        c4.metric("Short",int((signals["P9_Side"]=="SHORT").sum()))

        preview_cols=[c for c in ["_P9_Time","_P9_Date","Stock","P9_Side","P7D_Institutional_Score","P7D_Final_Action","P7E_Timing_Score","P7E_Entry_State","Live_RF","RVOL_Same_Time"] if c in signals.columns]
        with st.expander("Frozen 11:15 signal sample"):
            st.dataframe(signals[preview_cols].head(200),use_container_width=True,hide_index=True)

        if not st.session_state.access_token:
            st.warning("Connect Kite to reveal post-11:15 historical price paths and score the signals.")
        elif st.button("Run 20-Day Historical Validation",type="primary",key="p9_run"):
            kite.set_access_token(st.session_state.access_token)
            universe=get_fno_equity_universe(kite)
            token_map=dict(zip(universe["Stock"],universe["instrument_token"]))
            rows=[]
            errors=[]
            progress=st.progress(0)
            status=st.empty()
            total=len(signals)

            for n,(_,sig) in enumerate(signals.iterrows(),start=1):
                stock=sig["Stock"]
                day=sig["_P9_Date"]
                side=sig["P9_Side"]
                status.write(f"Historical validation: {day} | {stock} | {n}/{total}")
                try:
                    token=token_map.get(stock)
                    if token is None:
                        raise ValueError("Current equity instrument token not found")
                    start_dt=dt.datetime.combine(day,dt.time(9,0))
                    end_dt=dt.datetime.combine(day,dt.time(15,30))
                    raw5=pd.DataFrame(kite.historical_data(int(token),start_dt,end_dt,"5minute",oi=False))
                    metrics=_p9_forward_metrics(raw5,day,side)
                    if metrics is None:
                        raise ValueError("Insufficient 5-minute history for session")
                    base=sig.to_dict()
                    base.update(metrics)
                    rows.append(base)
                except Exception as e:
                    errors.append([str(day),stock,str(e)])
                progress.progress(n/total)
                time.sleep(0.08)

            status.empty()
            result=pd.DataFrame(rows)
            st.session_state["phase9_validation_result"]=result
            st.session_state["phase9_validation_errors"]=pd.DataFrame(errors,columns=["Date","Stock","Error"])

        if "phase9_validation_result" in st.session_state:
            vr=st.session_state["phase9_validation_result"].copy()
            if vr.empty:
                st.warning("No historical rows were validated.")
            else:
                summary=_p9_summary(vr)
                st.markdown("### Validation Gate Summary")
                st.dataframe(summary,use_container_width=True,hide_index=True)

                sr=summary.iloc[0]
                m1,m2,m3,m4=st.columns(4)
                m1.metric("Close Hit Rate",f'{sr["Close_Hit_Rate_%"]:.1f}%')
                m2.metric("Avg Close Edge",f'{sr["Avg_Directional_Close_%"]:.3f}%')
                m3.metric("Profit Factor",f'{sr["Profit_Factor_Returns"]:.2f}' if pd.notna(sr["Profit_Factor_Returns"]) else "—")
                m4.metric("Engineering Gate",sr["Preliminary_Engineering_Gate"])

                st.markdown("### Forward Performance")
                display=[c for c in ["_P9_Date","Stock","P9_Side","P7D_Institutional_Score","P7D_Final_Action","P7E_Timing_Score","P9_Entry_1115","P9_Ret_30m_%","P9_Ret_60m_%","P9_Ret_1430_%","P9_Ret_Close_%","P9_MFE_%","P9_MAE_%","P9_Close_Hit"] if c in vr.columns]
                st.dataframe(vr[display].sort_values(["_P9_Date","P9_Ret_Close_%"],ascending=[False,False]),use_container_width=True,hide_index=True)

                st.markdown("### Long vs Short")
                by_side=(vr.groupby("P9_Side",dropna=False).agg(
                    Signals=("Stock","size"),
                    Hit_Rate=("P9_Close_Hit","mean"),
                    Avg_Close=("P9_Ret_Close_%","mean"),
                    Avg_MFE=("P9_MFE_%","mean"),
                    Avg_MAE=("P9_MAE_%","mean"),
                ).reset_index())
                by_side["Hit_Rate"]=by_side["Hit_Rate"]*100
                st.dataframe(by_side,use_container_width=True,hide_index=True)

                if "P7D_Institutional_Score" in vr.columns:
                    score_num=pd.to_numeric(vr["P7D_Institutional_Score"],errors="coerce")
                    tmp=vr.assign(_score=score_num).dropna(subset=["_score"]).copy()
                    if not tmp.empty:
                        tmp["Score_Band"]=pd.cut(tmp["_score"],bins=[-np.inf,60,70,80,90,np.inf],labels=["<=60","60-70","70-80","80-90","90+"])
                        band=tmp.groupby("Score_Band",observed=True).agg(Signals=("Stock","size"),Hit_Rate=("P9_Close_Hit","mean"),Avg_Close=("P9_Ret_Close_%","mean"),Avg_MFE=("P9_MFE_%","mean"),Avg_MAE=("P9_MAE_%","mean")).reset_index()
                        band["Hit_Rate"]=band["Hit_Rate"]*100
                        st.markdown("### Institutional Score Calibration")
                        st.dataframe(band,use_container_width=True,hide_index=True)

                st.download_button(
                    "Download 20-Day Validation Results",
                    data=vr.to_csv(index=False).encode("utf-8"),
                    file_name="phase9_20day_historical_validation.csv",
                    mime="text/csv",
                    key="p9_download"
                )

        if "phase9_validation_errors" in st.session_state and not st.session_state["phase9_validation_errors"].empty:
            with st.expander("Historical validation errors"):
                st.dataframe(st.session_state["phase9_validation_errors"],use_container_width=True,hide_index=True)

    st.divider()
    st.markdown("### Deployment Rule")
    st.write(
        "Use this historical gate before live deployment. After the replay is stable, use only 1–2 live sessions "
        "to validate Kite/session/API/Streamlit operations. Historical options-chain validation requires stored "
        "11:15 option snapshots; do not reconstruct expired option positioning from today's chain."
    )
