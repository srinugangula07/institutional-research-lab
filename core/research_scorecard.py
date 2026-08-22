from __future__ import annotations

import pandas as pd


def build_research_scorecard(
    signals,
    portfolio_summary_table,
    ic_summary_table,
    auction_summary_table,
    sector_neutral_summary_table,
    false_discovery_table,
    futures_sessions=0,
    cost_bps=10,
):
    active = signals[signals["Signal"].isin(["LONG", "SHORT"])].copy()
    standalone_net = active["Oriented_Return_1515_%"].mean() - float(cost_bps) / 100.0
    portfolio_holdout = portfolio_summary_table[portfolio_summary_table["Sample"].eq("HOLDOUT")]
    ic_holdout = ic_summary_table[
        ic_summary_table["Sample"].eq("HOLDOUT") & ic_summary_table["Horizon"].eq("15:15")
    ]
    auction_holdout = auction_summary_table[auction_summary_table["Sample"].eq("HOLDOUT")]
    sector_holdout = sector_neutral_summary_table[sector_neutral_summary_table["Sample"].eq("HOLDOUT")]
    gates = [
        ("Historical data health", True, f"{signals['session_date'].nunique()} sessions"),
        ("Standalone signal expectancy", standalone_net > 0, f"{standalone_net:.4f}% after costs"),
        ("Cross-sectional holdout", not portfolio_holdout.empty and portfolio_holdout.iloc[0]["Net_Expectancy_%"] > 0, f"{portfolio_holdout.iloc[0]['Net_Expectancy_%']:.4f}%" if not portfolio_holdout.empty else "Unavailable"),
        ("Holdout rank IC", not ic_holdout.empty and ic_holdout.iloc[0]["Mean_Rank_IC"] >= 0.03 and ic_holdout.iloc[0]["Positive_IC_Rate_%"] >= 55, f"IC {ic_holdout.iloc[0]['Mean_Rank_IC']:.4f}" if not ic_holdout.empty else "Unavailable"),
        ("Morning auction feature", not auction_holdout.empty and auction_holdout["Mean_Rank_IC"].max() >= 0.03, f"Best IC {auction_holdout['Mean_Rank_IC'].max():.4f}" if not auction_holdout.empty else "Unavailable"),
        ("Sector-neutral residual IC", not sector_holdout.empty and sector_holdout["Mean_Rank_IC"].max() >= 0.03, f"Best IC {sector_holdout['Mean_Rank_IC'].max():.4f}" if not sector_holdout.empty else "Unavailable"),
        ("False-discovery control", bool(false_discovery_table["Survives_5pct_FDR"].any()), f"{int(false_discovery_table['Survives_5pct_FDR'].sum())} candidates survive"),
        ("Futures confirmation sample", int(futures_sessions) >= 60, f"{int(futures_sessions)}/60 sessions"),
    ]
    result = pd.DataFrame(gates, columns=["Research_Gate", "Passed", "Evidence"])
    result["Status"] = result["Passed"].map({True: "PASS", False: "FAIL / PENDING"})
    return result
