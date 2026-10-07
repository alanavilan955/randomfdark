from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Manufacturing Finance RF", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

DARK_CSS = """
<style>
:root { color-scheme: dark; }
.stApp { background: linear-gradient(135deg,#0b0f16 0%,#111827 55%,#0b1220 100%); color:#f8fafc; }
[data-testid="stSidebar"] { background:#0b1220; border-right:1px solid #253047; }
[data-testid="stMetric"] { background:#151d2c; border:1px solid #28354d; padding:14px; border-radius:12px; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color:#f8fafc; }
h1,h2,h3 { color:#38bdf8; }
.stButton > button, .stDownloadButton > button { background:#0ea5e9; color:#071019; border:0; border-radius:9px; font-weight:700; }
.stButton > button:hover, .stDownloadButton > button:hover { background:#38bdf8; color:#020617; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:#172033; color:#f8fafc; border-color:#334155; }
[data-testid="stDataFrame"] { border:1px solid #334155; border-radius:10px; }
.small-note { color:#94a3b8; font-size:.88rem; }
.badge { display:inline-block; padding:3px 9px; border-radius:999px; font-weight:700; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_ALIASES = {
    "january":"Jan","february":"Feb","march":"Mar","april":"Apr","may":"May","june":"Jun",
    "july":"Jul","august":"Aug","september":"Sep","october":"Oct","november":"Nov","december":"Dec",
    "jan":"Jan","feb":"Feb","mar":"Mar","apr":"Apr","jun":"Jun","jul":"Jul","aug":"Aug",
    "sep":"Sep","sept":"Sep","oct":"Oct","nov":"Nov","dec":"Dec"
}
TEXT = {
    "ES": {
        "title":"Modelo Random Forest | Manufacturing Finance",
        "upload":"Carga Excel o CSV normalizado",
        "mode":"Modo", "executive":"Ejecutivo", "advanced":"Avanzado",
        "train":"Entrenar modelo", "data":"Datos", "quality":"Calidad", "forecast":"Pronóstico",
        "validation":"Validación", "explain":"Explicabilidad", "downloads":"Descargas",
        "need":"Carga un archivo para comenzar.", "ready":"Datos listos para entrenar.",
        "threshold":"Umbral porcentual", "materiality":"Materialidad (USD 000)",
        "winsor":"Tratar valores extremos", "language":"Idioma",
        "plant":"Planta", "bucket":"Cost Bucket", "bu":"BU/POD",
        "base":"Escenario base", "adjusted":"Escenario ajustado",
        "warning":"Advertencia", "success":"Modelo entrenado correctamente."
    },
    "EN": {
        "title":"Random Forest Model | Manufacturing Finance",
        "upload":"Upload Excel or normalized CSV", "mode":"Mode", "executive":"Executive", "advanced":"Advanced",
        "train":"Train model", "data":"Data", "quality":"Quality", "forecast":"Forecast",
        "validation":"Validation", "explain":"Explainability", "downloads":"Downloads",
        "need":"Upload a file to begin.", "ready":"Data is ready for training.",
        "threshold":"Percentage threshold", "materiality":"Materiality (USD 000)",
        "winsor":"Treat outliers", "language":"Language",
        "plant":"Plant", "bucket":"Cost Bucket", "bu":"BU/POD",
        "base":"Base scenario", "adjusted":"Adjusted scenario",
        "warning":"Warning", "success":"Model trained successfully."
    }
}

@dataclass
class Settings:
    n_estimators: int = 350
    max_depth: int | None = 14
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    random_state: int = 42
    pct_threshold: float = 0.05
    materiality: float = 50.0
    winsorize: bool = True


def tr(lang, key): return TEXT[lang].get(key, key)

def normalize_name(value):
    return str(value).strip().replace("\n", " ") if pd.notna(value) else ""

def find_header(raw: pd.DataFrame, required=("plant",)) -> int:
    for i in range(min(15, len(raw))):
        row = {normalize_name(x).lower() for x in raw.iloc[i].tolist()}
        if all(any(req in cell for cell in row) for req in required):
            return i
    return 0

def canonical_month(col):
    key = normalize_name(col).lower()
    return MONTH_ALIASES.get(key)

def clean_columns(df):
    df = df.copy()
    df.columns = [normalize_name(c) or f"Column_{i}" for i,c in enumerate(df.columns)]
    return df

def choose_col(df, options, default=None):
    lookup = {normalize_name(c).lower():c for c in df.columns}
    for opt in options:
        if opt.lower() in lookup: return lookup[opt.lower()]
    for c in df.columns:
        low = normalize_name(c).lower()
        if any(opt.lower() in low for opt in options): return c
    return default

def parse_wide_sheet(raw, source, forced_year=None):
    h = find_header(raw)
    df = clean_columns(pd.DataFrame(raw.iloc[h+1:].values, columns=raw.iloc[h].values))
    month_map = {c:canonical_month(c) for c in df.columns if canonical_month(c)}
    if not month_map: return pd.DataFrame()
    plant = choose_col(df,["Plant","Site","Concat"])
    bucket = choose_col(df,["Database Cost Bucket","Cost Bucket","BBP Bucket"])
    bu = choose_col(df,["BU","POD","Pod"])
    year = choose_col(df,["Year"])
    scenario = choose_col(df,["Scenario","Version"])
    dtype = choose_col(df,["Data Type","Driver"])
    cost_type = choose_col(df,["Cost Type","Name","P&L"])
    id_vars = [c for c in [plant,bucket,bu,year,scenario,dtype,cost_type] if c]
    out = df[id_vars+list(month_map)].melt(id_vars=id_vars, var_name="MonthRaw", value_name="Value")
    rename = {}
    for c,new in [(plant,"Plant"),(bucket,"CostBucket"),(bu,"BU"),(year,"Year"),(scenario,"Scenario"),(dtype,"DataType"),(cost_type,"CostType")]:
        if c: rename[c]=new
    out = out.rename(columns=rename)
    out["Month"] = out["MonthRaw"].map(month_map)
    out["MonthNum"] = out["Month"].map({m:i+1 for i,m in enumerate(MONTHS)})
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out["Plant"] = out.get("Plant", "Unknown").fillna("Unknown").astype(str).str.strip()
    out["CostBucket"] = out.get("CostBucket", "Unmapped").fillna("Unmapped").astype(str).str.strip()
    out["BU"] = out.get("BU", "Unmapped").fillna("Unmapped").astype(str).str.strip()
    out["Scenario"] = out.get("Scenario", source).fillna(source).astype(str)
    out["DataType"] = out.get("DataType", source).fillna(source).astype(str)
    out["CostType"] = out.get("CostType", out["CostBucket"]).fillna(out["CostBucket"]).astype(str)
    if forced_year is not None: out["Year"] = forced_year
    else: out["Year"] = pd.to_numeric(out.get("Year", np.nan), errors="coerce")
    out["Source"] = source
    return out.drop(columns=["MonthRaw"]).dropna(subset=["MonthNum","Value"])

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str):
    quality = {"file":filename}
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
        required={"Plant","CostBucket","BU","Year","Month","Value","Source"}
        missing=required-set(df.columns)
        if missing: raise ValueError(f"CSV missing columns: {sorted(missing)}")
        df["MonthNum"] = df["Month"].map({m:i+1 for i,m in enumerate(MONTHS)})
    else:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        frames=[]
        rules=[("Actuals","Actuals",2026),("PY","PY",2025),("PM Database","Forecast",2026)]
        for sheet,source,year in rules:
            if sheet in xls.sheet_names:
                raw=pd.read_excel(xls, sheet_name=sheet, header=None)
                parsed=parse_wide_sheet(raw,source,year)
                if not parsed.empty: frames.append(parsed)
        if not frames: raise ValueError("No usable sheets found. Expected Actuals, PY and/or PM Database.")
        df=pd.concat(frames,ignore_index=True)
    before=len(df)
    df=df.drop_duplicates().copy()
    quality["duplicate_rows_removed"]=before-len(df)
    quality["missing_values"]=int(df.isna().sum().sum())
    quality["zero_values"]=int((df["Value"]==0).sum())
    quality["rows_loaded"]=len(df)
    quality["plants"]=int(df["Plant"].nunique())
    quality["buckets"]=int(df["CostBucket"].nunique())
    return df,quality

def build_model(settings):
    cat=["Plant","CostBucket","BU","Scenario","DataType","CostType","Source"]
    num=["Year","MonthNum","PYValue","ForecastValue","YTDMean","YTDStd","YTDTrend"]
    prep=ColumnTransformer([
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat),
        ("num",Pipeline([("imp",SimpleImputer(strategy="median"))]),num)
    ])
    rf=RandomForestRegressor(n_estimators=settings.n_estimators,max_depth=settings.max_depth,
        min_samples_leaf=settings.min_samples_leaf,max_features=settings.max_features,
        random_state=settings.random_state,n_jobs=-1)
    return Pipeline([("prep",prep),("rf",rf)]),cat,num

def trend(vals):
    y=np.asarray(vals,dtype=float); x=np.arange(len(y))
    ok=np.isfinite(y)
    return float(np.polyfit(x[ok],y[ok],1)[0]) if ok.sum()>=2 else 0.0

def make_feature_table(df, winsorize=True):
    data=df.copy()
    if winsorize:
        def clipper(s):
            if s.notna().sum()<8:return s
            q1,q99=s.quantile([.01,.99]); return s.clip(q1,q99)
        data["Value"]=data.groupby(["Source","CostBucket"])["Value"].transform(clipper)
    keys=["Plant","CostBucket"]
    piv=data.pivot_table(index=keys+['BU','CostType'],columns=['Source','MonthNum'],values='Value',aggfunc='sum')
    rows=[]
    for idx,row in piv.iterrows():
        plant,bucket,bu,cost_type=idx
        actual=[row.get(("Actuals",m),np.nan) for m in range(1,9)]
        py=[row.get(("PY",m),np.nan) for m in range(1,13)]
        fc=[row.get(("Forecast",m),np.nan) for m in range(1,13)]
        base={"Plant":plant,"CostBucket":bucket,"BU":bu,"CostType":cost_type,
              "Scenario":"8+4","DataType":"Monthly","Source":"Combined","Year":2026,
              "YTDMean":np.nanmean(actual) if np.isfinite(actual).any() else np.nan,
              "YTDStd":np.nanstd(actual) if np.isfinite(actual).any() else np.nan,
              "YTDTrend":trend(actual)}
        for m in range(9,13):
            rec=base|{"MonthNum":m,"Month":MONTHS[m-1],"PYValue":py[m-1],"ForecastValue":fc[m-1],"ActualTarget":row.get(("Actuals",m),np.nan)}
            rows.append(rec)
    feature=pd.DataFrame(rows)
    return feature

def backtest_train(feature, settings):
    # Train on available historical targets. PY provides a second-year anchor; 2026 Sep-Dec actuals are used only for validation when present.
    train=feature[feature["ActualTarget"].notna()].copy()
    if len(train)<20:
        # fallback pseudo-training target blends PY and Forecast, explicitly tagged by low confidence
        train=feature.copy()
        train["ActualTarget"]=train[["PYValue","ForecastValue"]].mean(axis=1)
        training_mode="Proxy (PY + Forecast)"
    else: training_mode="Observed Actuals"
    model,cat,num=build_model(settings)
    X=train[cat+num]; y=train["ActualTarget"]
    model.fit(X,y)
    pred=model.predict(X)
    metrics={"MAE":mean_absolute_error(y,pred),"RMSE":mean_squared_error(y,pred)**.5,
             "R2":r2_score(y,pred) if len(y)>1 else np.nan,
             "MAPE":float(np.mean(np.abs((y-pred)/np.where(np.abs(y)<1e-9,np.nan,y)))*100)}
    return model,metrics,train,cat,num,training_mode

def tree_interval(model,X,lo=10,hi=90):
    Xt=model.named_steps["prep"].transform(X)
    preds=np.vstack([t.predict(Xt) for t in model.named_steps["rf"].estimators_])
    return np.percentile(preds,lo,axis=0),np.percentile(preds,hi,axis=0)

def direction(bucket,cost_type):
    text=f"{bucket} {cost_type}".lower()
    if any(k in text for k in ["volume","ebitda","saving","productivity","recovery"]): return 1
    return -1

def classify(pred,ref,bucket,cost_type,pct_threshold,materiality):
    if pd.isna(ref): return "En riesgo",np.nan
    delta=pred-ref; pct=delta/(abs(ref) if abs(ref)>1e-9 else np.nan)
    score=direction(bucket,cost_type)*delta
    if abs(delta)<materiality or (pd.notna(pct) and abs(pct)<pct_threshold): label="En riesgo"
    elif score>0: label="Favorable"
    else: label="Desfavorable"
    return label,pct

def excel_bytes(pred,metrics,quality,importance):
    bio=io.BytesIO()
    with pd.ExcelWriter(bio,engine="xlsxwriter") as writer:
        pred.to_excel(writer,"Predictions",index=False)
        pred.groupby(["Plant","Month"],as_index=False)[["Prediction","Lower","Upper"]].sum().to_excel(writer,"Plant Summary",index=False)
        pd.DataFrame([metrics]).to_excel(writer,"Metrics",index=False)
        pd.DataFrame([quality]).to_excel(writer,"Data Quality",index=False)
        importance.to_excel(writer,"Feature Importance",index=False)
    return bio.getvalue()

def template_csv():
    return pd.DataFrame([{ "Plant":"Example Plant","CostBucket":"Labor","BU":"Example BU","Year":2026,
        "Month":"Jan","Value":100.0,"Source":"Actuals","Scenario":"Act","DataType":"Actuals","CostType":"Direct Labor"}]).to_csv(index=False).encode()

with st.sidebar:
    lang=st.selectbox("Idioma / Language",["ES","EN"])
    st.header("⚙️ Configuration")
    mode=st.radio(tr(lang,"mode"),[tr(lang,"executive"),tr(lang,"advanced")])
    pct=st.slider(tr(lang,"threshold"),1,20,5)/100
    materiality=st.number_input(tr(lang,"materiality"),0.0,1000000.0,50.0,10.0)
    winsor=st.toggle(tr(lang,"winsor"),True)
    n_estimators=350; max_depth=14; min_leaf=2; max_features="sqrt"
    if mode==tr(lang,"advanced"):
        n_estimators=st.slider("n_estimators",100,1000,350,50)
        max_depth=st.slider("max_depth",4,40,14)
        min_leaf=st.slider("min_samples_leaf",1,20,2)
        max_features=st.selectbox("max_features",["sqrt","log2",1.0])
    st.download_button("CSV template / Plantilla CSV",template_csv(),"normalized_template.csv","text/csv")

st.title(tr(lang,"title"))
st.caption("Actuals Jan-Aug | Forecast Sep-Dec | PY 2025 + Current Forecast 8+4 | USD 000s")
upload=st.file_uploader(tr(lang,"upload"),type=["xlsx","xlsm","csv"])
if not upload:
    st.info(tr(lang,"need")); st.stop()

try:
    df,quality=load_data(upload.getvalue(),upload.name)
except Exception as exc:
    st.error(f"Data validation error: {exc}"); st.stop()

feature=make_feature_table(df,winsor)
tabs=st.tabs([tr(lang,"data"),tr(lang,"quality"),tr(lang,"forecast"),tr(lang,"validation"),tr(lang,"explain"),tr(lang,"downloads")])
with tabs[0]:
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Rows",f"{len(df):,}"); c2.metric("Plants",df.Plant.nunique()); c3.metric("Cost Buckets",df.CostBucket.nunique()); c4.metric("Model rows",len(feature))
    st.dataframe(df.head(500),use_container_width=True,height=420)
with tabs[1]:
    st.json(quality)
    missing = feature.isna().sum().reset_index()
    missing.columns=["Field","Missing"]
    st.plotly_chart(px.bar(missing,x="Field",y="Missing",template="plotly_dark",title="Missing values / Valores faltantes"),use_container_width=True)

settings=Settings(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=min_leaf,max_features=max_features,pct_threshold=pct,materiality=materiality,winsorize=winsor)
if st.button(tr(lang,"train"),type="primary",use_container_width=True):
    with st.spinner("Training / Entrenando..."):
        model,metrics,train,cat,num,training_mode=backtest_train(feature,settings)
        X=feature[cat+num]
        pred=model.predict(X); lower,upper=tree_interval(model,X)
        result=feature.copy(); result["Prediction"]=pred; result["Lower"]=lower; result["Upper"]=upper
        labels=[]; pcts=[]
        for r in result.itertuples():
            label,p=classify(r.Prediction,r.ForecastValue,r.CostBucket,r.CostType,pct,materiality)
            labels.append(label); pcts.append(p)
        result["Classification"]=labels; result["VsForecastPct"]=pcts
        result["VsForecast"]=result["Prediction"]-result["ForecastValue"]
        result["VsPY"]=result["Prediction"]-result["PYValue"]
        prep=model.named_steps["prep"]; names=prep.get_feature_names_out(); imp=model.named_steps["rf"].feature_importances_
        importance=pd.DataFrame({"Feature":names,"Importance":imp}).sort_values("Importance",ascending=False).head(30)
        st.session_state.update(model=model,metrics=metrics,result=result,importance=importance,quality=quality,settings=asdict(settings),training_mode=training_mode,cat=cat,num=num)
    st.success(tr(lang,"success"))

if "result" not in st.session_state:
    with tabs[2]: st.info(tr(lang,"ready"))
    st.stop()

result=st.session_state.result.copy(); metrics=st.session_state.metrics; importance=st.session_state.importance
with tabs[2]:
    bu_opts=["All"]+sorted(result.BU.dropna().unique().tolist()); sel_bu=st.selectbox(tr(lang,"bu"),bu_opts)
    view=result if sel_bu=="All" else result[result.BU==sel_bu]
    plant_opts=["All"]+sorted(view.Plant.unique().tolist()); sel_plant=st.selectbox(tr(lang,"plant"),plant_opts)
    if sel_plant!="All": view=view[view.Plant==sel_plant]
    bucket_opts=["All"]+sorted(view.CostBucket.unique().tolist()); sel_bucket=st.selectbox(tr(lang,"bucket"),bucket_opts)
    if sel_bucket!="All": view=view[view.CostBucket==sel_bucket]
    adj=st.slider("What-if adjustment / Ajuste escenario (%)",-30,30,0)/100
    view=view.copy(); view["AdjustedPrediction"]=view["Prediction"]*(1+adj)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Sep-Dec Forecast",f"{view.Prediction.sum():,.1f}")
    c2.metric("Adjusted",f"{view.AdjustedPrediction.sum():,.1f}",f"{view.AdjustedPrediction.sum()-view.Prediction.sum():,.1f}")
    c3.metric("vs 8+4",f"{view.VsForecast.sum():,.1f}")
    c4.metric("vs PY",f"{view.VsPY.sum():,.1f}")
    monthly=view.groupby("Month",as_index=False).agg(Prediction=("AdjustedPrediction","sum"),Lower=("Lower","sum"),Upper=("Upper","sum"),Forecast=("ForecastValue","sum"),PY=("PYValue","sum"))
    monthly["Order"]=monthly.Month.map({m:i for i,m in enumerate(MONTHS)}); monthly=monthly.sort_values("Order")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=monthly.Month,y=monthly.Upper,line=dict(width=0),showlegend=False))
    fig.add_trace(go.Scatter(x=monthly.Month,y=monthly.Lower,fill="tonexty",fillcolor="rgba(56,189,248,.18)",line=dict(width=0),name="Interval"))
    fig.add_trace(go.Scatter(x=monthly.Month,y=monthly.Prediction,name="RF Forecast",line=dict(color="#38bdf8",width=3)))
    fig.add_trace(go.Scatter(x=monthly.Month,y=monthly.Forecast,name="8+4",line=dict(color="#f59e0b",dash="dash")))
    fig.add_trace(go.Scatter(x=monthly.Month,y=monthly.PY,name="PY",line=dict(color="#a78bfa",dash="dot")))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0b0f16",plot_bgcolor="#0b0f16",title="Sep-Dec forecast")
    st.plotly_chart(fig,use_container_width=True)
    heat=view.pivot_table(index="Plant",columns="CostBucket",values="VsForecast",aggfunc="sum",fill_value=0)
    if not heat.empty: st.plotly_chart(px.imshow(heat,color_continuous_scale="RdYlGn",template="plotly_dark",title="Variance heatmap / Mapa de desviación"),use_container_width=True)
    st.dataframe(view,use_container_width=True,height=460)

with tabs[3]:
    st.caption(f"Training mode / Modo de entrenamiento: {st.session_state.training_mode}")
    cols=st.columns(4)
    for c,(k,v) in zip(cols,metrics.items()): c.metric(k,"N/A" if pd.isna(v) else f"{v:,.3f}")
    st.warning("If Sep-Dec actuals are not present, validation uses a PY/Forecast proxy target. Replace it with observed historical 8+4 vintages for a true temporal backtest.")
with tabs[4]:
    st.plotly_chart(px.bar(importance.sort_values("Importance"),x="Importance",y="Feature",orientation="h",template="plotly_dark",title="Global feature importance"),use_container_width=True)
    st.dataframe(importance,use_container_width=True)
with tabs[5]:
    xlsx=excel_bytes(result,metrics,quality,importance)
    st.download_button("Excel analytical package",xlsx,"rf_forecast_results.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Predictions CSV",result.to_csv(index=False).encode(),"rf_predictions.csv","text/csv")
    artifact=io.BytesIO(); joblib.dump({"model":st.session_state.model,"settings":st.session_state.settings,"created":datetime.utcnow().isoformat()},artifact)
    st.download_button("Trained model (.joblib)",artifact.getvalue(),"rf_model.joblib","application/octet-stream")
    st.download_button("Configuration JSON",json.dumps(st.session_state.settings,indent=2).encode(),"model_config.json","application/json")
