
import io
import numpy as np
import pandas as pd
import streamlit as st

APP_BUILD = "RESEARCH_LAB_BULK_VALIDATION_V1"

st.set_page_config(
    page_title="Institutional Research, Backtesting & Market Replay Lab",
    layout="wide",
)

st.title("Institutional Research, Backtesting & Market Replay Lab")
st.caption(f"Build: {APP_BUILD} | Bulk historical research | No Market-Dashboard module dependency")

# -----------------------------
# Helpers
# -----------------------------
def _num(s):
    return pd.to_numeric(s, errors="coerce")

def _first_existing(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _spearman(x, y):
    a = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    if len(a) < 5:
        return np.nan
    return a["x"].rank(method="average").corr(a["y"].rank(method="average"))

def _pearson(x, y):
    a = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    if len(a) < 5:
        return np.nan
    return a["x"].corr(a["y"])

def _safe_return_target(df):
    candidates = [
        "Raw_EOD_Return_%",
        "Directional_EOD_Return_%",
        "EOD_Return_%",
        "Return_EOD_%",
        "Forward_Return_%",
        "Future_Return_%",
        "Outcome_Return_%",
    ]
    return _first_existing(df, candidates)

def _session_col(df):
    return _first_existing(
        df,
        ["Session","Replay_Session","Trading_Date","Date","date","Snapshot_Date"]
    )

def _stock_col(df):
    return _first_existing(df, ["Stock","Symbol","stock","Ticker"])

def _feature_candidates(df, target):
    blocked = {
        target,
        "EOD_Close","Exit_Price","Realized_PnL","Realized_R",
        "Long_MFE_%","Long_MAE_%","MFE_%","MAE_%",
        "Outcome","Hit","Win","Target_Hit","Stop_Hit",
    }
    feats=[]
    for c in df.columns:
        if c in blocked:
            continue
        s=_num(df[c])
        if s.notna().sum() >= max(20, int(len(df)*0.05)) and s.nunique(dropna=True) >= 5:
            feats.append(c)
    return feats

def information_coefficient(df, features, target, session=None):
    rows=[]
    for f in features:
        ic=_spearman(df[f],df[target])
        pc=_pearson(df[f],df[target])
        daily=[]
        if session and session in df.columns:
            for _,g in df.groupby(session):
                v=_spearman(g[f],g[target])
                if pd.notna(v):
                    daily.append(v)
        rows.append({
            "Feature":f,
            "Spearman_IC":ic,
            "Pearson_Corr":pc,
            "Mean_Daily_IC":np.mean(daily) if daily else np.nan,
            "IC_Std":np.std(daily, ddof=1) if len(daily)>1 else np.nan,
            "IC_IR":(
                np.mean(daily)/np.std(daily,ddof=1)
                if len(daily)>1 and np.std(daily,ddof=1)>0 else np.nan
            ),
            "Positive_Daily_IC_%":(
                np.mean(np.array(daily)>0)*100 if daily else np.nan
            ),
            "Sessions":len(daily),
        })
    out=pd.DataFrame(rows)
    if not out.empty:
        out["Abs_Mean_Daily_IC"]=out["Mean_Daily_IC"].abs()
        out=out.sort_values(
            ["Abs_Mean_Daily_IC","Spearman_IC"],
            ascending=[False,False],
            na_position="last",
        ).reset_index(drop=True)
    return out

def quantile_attribution(df, features, target, q=5):
    rows=[]
    for f in features:
        x=pd.DataFrame({"f":_num(df[f]),"y":_num(df[target])}).dropna()
        if len(x)<50 or x["f"].nunique()<q:
            continue
        try:
            x["bucket"]=pd.qcut(x["f"],q=q,duplicates="drop")
        except Exception:
            continue
        b=x.groupby("bucket",observed=True)["y"].agg(["count","mean","median"]).reset_index()
        if len(b)<2:
            continue
        spread=float(b["mean"].iloc[-1]-b["mean"].iloc[0])
        rows.append({
            "Feature":f,
            "Rows":len(x),
            "Bottom_Q_Avg_Return_%":float(b["mean"].iloc[0]),
            "Top_Q_Avg_Return_%":float(b["mean"].iloc[-1]),
            "TopMinusBottom_%":spread,
            "Monotonic_Corr":b["mean"].reset_index(drop=True).corr(
                pd.Series(range(len(b)),dtype=float)
            ),
        })
    out=pd.DataFrame(rows)
    if not out.empty:
        out["Abs_Spread"]=out["TopMinusBottom_%"].abs()
        out=out.sort_values("Abs_Spread",ascending=False).reset_index(drop=True)
    return out

def linear_feature_model(train, test, features, target):
    Xtr=train[features].apply(pd.to_numeric,errors="coerce")
    Xte=test[features].apply(pd.to_numeric,errors="coerce")
    ytr=_num(train[target])

    med=Xtr.median()
    Xtr=Xtr.fillna(med)
    Xte=Xte.fillna(med)

    mu=Xtr.mean()
    sd=Xtr.std().replace(0,1)
    Ztr=(Xtr-mu)/sd
    Zte=(Xte-mu)/sd

    valid=ytr.notna()
    if valid.sum()<20:
        return np.full(len(test),np.nan), pd.Series(dtype=float)

    A=np.column_stack([np.ones(valid.sum()),Ztr.loc[valid].values])
    y=ytr.loc[valid].values
    ridge=1e-3
    reg=np.eye(A.shape[1])*ridge
    reg[0,0]=0
    beta=np.linalg.pinv(A.T@A+reg)@(A.T@y)

    pred=np.column_stack([np.ones(len(Zte)),Zte.values])@beta
    coef=pd.Series(beta[1:],index=features)
    return pred,coef

def walk_forward(df, features, target, session, top_n=10):
    sessions=sorted(pd.to_datetime(df[session],errors="coerce").dropna().dt.date.unique())
    if len(sessions)<6:
        return pd.DataFrame(), pd.DataFrame()

    rows=[]
    coef_rows=[]
    for i in range(5,len(sessions)):
        train_days=sessions[:i]
        test_day=sessions[i]
        tr=df[pd.to_datetime(df[session],errors="coerce").dt.date.isin(train_days)].copy()
        te=df[pd.to_datetime(df[session],errors="coerce").dt.date.eq(test_day)].copy()
        if tr.empty or te.empty:
            continue

        pred,coef=linear_feature_model(tr,te,features,target)
        te=te.copy()
        te["Model_Score"]=pred
        te["_target"]=_num(te[target])
        te=te.dropna(subset=["Model_Score","_target"])
        if te.empty:
            continue

        te=te.sort_values("Model_Score")
        n=min(int(top_n),max(1,len(te)//2))
        shorts=te.head(n).copy()
        longs=te.tail(n).copy()
        shorts["Side"]="SHORT"
        longs["Side"]="LONG"
        shorts["Directional_Return_%"]=-shorts["_target"]
        longs["Directional_Return_%"]=longs["_target"]
        pick=pd.concat([shorts,longs],ignore_index=True)
        pick["Session"]=str(test_day)
        rows.append(pick)

        cr={"Session":str(test_day)}
        for k,v in coef.items():
            cr[k]=v
        coef_rows.append(cr)

    picks=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    coefs=pd.DataFrame(coef_rows)
    return picks,coefs

def backtest_summary(picks):
    if picks is None or picks.empty:
        return pd.DataFrame()
    r=_num(picks["Directional_Return_%"]).dropna()
    if r.empty:
        return pd.DataFrame()
    wins=r[r>0]
    losses=r[r<0]
    gp=wins.sum()
    gl=abs(losses.sum())
    daily=picks.assign(
        Directional_Return_Num=_num(picks["Directional_Return_%"])
    ).groupby("Session")["Directional_Return_Num"].mean()
    equity=daily.cumsum()
    dd=equity-equity.cummax()
    return pd.DataFrame([{
        "Signals":len(r),
        "Sessions":picks["Session"].nunique(),
        "Hit_Rate_%":round((r>0).mean()*100,2),
        "Avg_Directional_Return_%":round(r.mean(),4),
        "Median_Directional_Return_%":round(r.median(),4),
        "Profit_Factor_Proxy":round(gp/gl,3) if gl>0 else np.nan,
        "Avg_Daily_Portfolio_%":round(daily.mean(),4),
        "Best_Day_%":round(daily.max(),4),
        "Worst_Day_%":round(daily.min(),4),
        "Max_Drawdown_Proxy_%":round(dd.min(),4),
    }])

def ablation_test(df, base_features, target, session, top_n=10):
    base_picks,_=walk_forward(df,base_features,target,session,top_n)
    base=backtest_summary(base_picks)
    if base.empty:
        return pd.DataFrame()
    base_avg=float(base.iloc[0]["Avg_Directional_Return_%"])
    base_hit=float(base.iloc[0]["Hit_Rate_%"])

    rows=[{
        "Removed_Feature":"NONE (BASE)",
        "Features_Used":len(base_features),
        "Avg_Directional_Return_%":base_avg,
        "Hit_Rate_%":base_hit,
        "Delta_Avg_vs_Base_%":0.0,
        "Delta_Hit_vs_Base_pp":0.0,
    }]
    for f in base_features:
        ff=[x for x in base_features if x!=f]
        if not ff:
            continue
        p,_=walk_forward(df,ff,target,session,top_n)
        s=backtest_summary(p)
        if s.empty:
            continue
        avg=float(s.iloc[0]["Avg_Directional_Return_%"])
        hit=float(s.iloc[0]["Hit_Rate_%"])
        rows.append({
            "Removed_Feature":f,
            "Features_Used":len(ff),
            "Avg_Directional_Return_%":avg,
            "Hit_Rate_%":hit,
            "Delta_Avg_vs_Base_%":avg-base_avg,
            "Delta_Hit_vs_Base_pp":hit-base_hit,
        })
    return pd.DataFrame(rows).sort_values(
        "Delta_Avg_vs_Base_%",
        ascending=False
    ).reset_index(drop=True)

# -----------------------------
# Dataset input
# -----------------------------
tabs=st.tabs([
    "Historical Import",
    "Information Coefficient",
    "Feature Attribution & Ablation",
    "Walk-Forward Backtest",
    "Robustness",
    "About / Research Rules",
])

with tabs[0]:
    st.subheader("Historical Import Bridge")
    st.caption(
        "Upload the bulk 20-day validation CSV from the Market Intelligence research run. "
        "No Kite login and no Dashboard analytics modules are required."
    )
    upload=st.file_uploader("Upload research CSV",type=["csv"],key="bulk_upload")
    if upload is not None:
        try:
            df=pd.read_csv(upload)
            st.session_state["research_df"]=df
            st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if "research_df" in st.session_state:
        df=st.session_state["research_df"].copy()
        s=_session_col(df)
        k=_stock_col(df)
        t=_safe_return_target(df)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Rows",f"{len(df):,}")
        c2.metric("Columns",len(df.columns))
        c3.metric("Sessions",df[s].nunique() if s else "—")
        c4.metric("Stocks",df[k].nunique() if k else "—")
        st.write("Detected outcome target:", t or "Not detected")
        st.dataframe(df.head(300),use_container_width=True,hide_index=True)

        st.download_button(
            "Download Loaded Dataset",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="research_lab_loaded_dataset.csv",
            mime="text/csv",
        )

with tabs[1]:
    st.subheader("Information Coefficient")
    if "research_df" not in st.session_state:
        st.info("Upload the bulk validation CSV in Historical Import first.")
    else:
        df=st.session_state["research_df"].copy()
        target=_safe_return_target(df)
        session=_session_col(df)
        if not target:
            st.error("No recognized forward/EOD outcome column found.")
        else:
            features=_feature_candidates(df,target)
            selected=st.multiselect(
                "Features",
                features,
                default=features[:min(20,len(features))],
                key="ic_features"
            )
            if selected:
                ic=information_coefficient(df,selected,target,session)
                st.session_state["ic_result"]=ic
                st.dataframe(ic,use_container_width=True,hide_index=True)
                st.download_button(
                    "Download IC Results",
                    data=ic.to_csv(index=False).encode("utf-8"),
                    file_name="research_lab_information_coefficient.csv",
                    mime="text/csv",
                )

with tabs[2]:
    st.subheader("Feature Attribution & Ablation")
    if "research_df" not in st.session_state:
        st.info("Upload the bulk validation CSV first.")
    else:
        df=st.session_state["research_df"].copy()
        target=_safe_return_target(df)
        session=_session_col(df)
        if not target or not session:
            st.error("Target or session column is missing.")
        else:
            features=_feature_candidates(df,target)
            selected=st.multiselect(
                "Research features",
                features,
                default=features[:min(12,len(features))],
                key="attrib_features"
            )
            if selected:
                st.markdown("### Quantile Attribution")
                qa=quantile_attribution(df,selected,target,q=5)
                st.dataframe(qa,use_container_width=True,hide_index=True)

                st.markdown("### Walk-Forward Feature Ablation")
                top_n=st.number_input(
                    "Top/Bottom stocks per session",
                    min_value=3,max_value=30,value=10,step=1,
                    key="ablation_topn"
                )
                if st.button("Run Feature Ablation",type="primary"):
                    with st.spinner("Running walk-forward ablation..."):
                        ab=ablation_test(df,selected,target,session,int(top_n))
                    st.session_state["ablation_result"]=ab
                if "ablation_result" in st.session_state:
                    ab=st.session_state["ablation_result"]
                    st.dataframe(ab,use_container_width=True,hide_index=True)
                    st.download_button(
                        "Download Feature Ablation",
                        data=ab.to_csv(index=False).encode("utf-8"),
                        file_name="research_lab_feature_ablation.csv",
                        mime="text/csv",
                    )

with tabs[3]:
    st.subheader("Walk-Forward Backtest")
    if "research_df" not in st.session_state:
        st.info("Upload the bulk validation CSV first.")
    else:
        df=st.session_state["research_df"].copy()
        target=_safe_return_target(df)
        session=_session_col(df)
        if not target or not session:
            st.error("Target or session column is missing.")
        else:
            features=_feature_candidates(df,target)
            selected=st.multiselect(
                "Model features",
                features,
                default=features[:min(10,len(features))],
                key="wf_features"
            )
            top_n=st.number_input(
                "Long + Short picks per side/session",
                3,30,10,1,key="wf_topn"
            )
            if st.button("Run Walk-Forward Backtest",type="primary"):
                with st.spinner("Training only on prior sessions and testing on next session..."):
                    picks,coefs=walk_forward(
                        df,selected,target,session,int(top_n)
                    )
                    summary=backtest_summary(picks)
                st.session_state["wf_picks"]=picks
                st.session_state["wf_coefs"]=coefs
                st.session_state["wf_summary"]=summary

            if "wf_summary" in st.session_state:
                sm=st.session_state["wf_summary"]
                if not sm.empty:
                    r=sm.iloc[0]
                    a,b,c,d=st.columns(4)
                    a.metric("Signals",int(r["Signals"]))
                    b.metric("Hit Rate",f'{r["Hit_Rate_%"]:.1f}%')
                    c.metric("Avg Directional",f'{r["Avg_Directional_Return_%"]:.3f}%')
                    d.metric("PF Proxy",f'{r["Profit_Factor_Proxy"]:.2f}' if pd.notna(r["Profit_Factor_Proxy"]) else "—")
                    st.dataframe(sm,use_container_width=True,hide_index=True)

                picks=st.session_state["wf_picks"]
                if not picks.empty:
                    cols=[c for c in [
                        "Session","Stock","Side","Model_Score",
                        "Directional_Return_%",target
                    ] if c in picks.columns]
                    st.dataframe(picks[cols],use_container_width=True,hide_index=True)
                    st.download_button(
                        "Download Walk-Forward Picks",
                        data=picks.to_csv(index=False).encode("utf-8"),
                        file_name="research_lab_walkforward_picks.csv",
                        mime="text/csv",
                    )

with tabs[4]:
    st.subheader("Robustness")
    if "research_df" not in st.session_state:
        st.info("Upload the bulk validation CSV first.")
    else:
        df=st.session_state["research_df"].copy()
        target=_safe_return_target(df)
        session=_session_col(df)
        if not target or not session:
            st.error("Target or session column is missing.")
        else:
            features=_feature_candidates(df,target)
            selected=st.multiselect(
                "Robustness features",
                features,
                default=features[:min(8,len(features))],
                key="rob_features"
            )
            if selected and st.button("Run Robustness Grid",type="primary"):
                rows=[]
                for n in [5,10,15,20]:
                    p,_=walk_forward(df,selected,target,session,n)
                    s=backtest_summary(p)
                    if not s.empty:
                        rr=s.iloc[0].to_dict()
                        rr["Top_Bottom_N"]=n
                        rows.append(rr)
                rob=pd.DataFrame(rows)
                st.session_state["rob_result"]=rob
            if "rob_result" in st.session_state:
                rob=st.session_state["rob_result"]
                st.dataframe(rob,use_container_width=True,hide_index=True)
                st.download_button(
                    "Download Robustness Grid",
                    data=rob.to_csv(index=False).encode("utf-8"),
                    file_name="research_lab_robustness.csv",
                    mime="text/csv",
                )

with tabs[5]:
    st.subheader("Research Rules")
    st.write(
        "This lab is intentionally separated from the live Market Dashboard. "
        "It analyzes frozen historical datasets and does not call current Kite quotes."
    )
    st.info(
        "No future-data leakage: signals/features must be frozen before outcome columns are evaluated. "
        "At an 11:15 decision using 30-minute bars, the 10:45-start bar is the last completed bar."
    )
    st.warning(
        "Exact historical Options/Greeks must come from archived option-chain snapshots. "
        "Missing historical evidence must remain unavailable rather than being synthesized."
    )
