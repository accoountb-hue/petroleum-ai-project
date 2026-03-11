import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Petroleum Data Analysis with AI",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Theme ----------------
st.markdown("""
<style>
:root{
    --bg:#08101f; --bg2:#0d1530; --panel:#111936; --panel2:#182750;
    --text:#eef3ff; --muted:#b8c7ea; --accent:#6aa8ff; --accent2:#3de0d0;
    --border:rgba(122,176,255,.18);
}
.block-container{padding-top:1rem;padding-bottom:1rem;max-width:1400px;}
[data-testid="stAppViewContainer"]{
    background:
      linear-gradient(rgba(8,16,31,.80), rgba(8,16,31,.93)),
      url("https://images.unsplash.com/photo-1513828583688-c52646db42da?q=80&w=1800&auto=format&fit=crop");
    background-size: cover;
    background-attachment: fixed;
}
[data-testid="stSidebar"]{
    background: linear-gradient(180deg, rgba(17,25,54,.98), rgba(24,39,80,.98));
}
@keyframes floatY {0%{transform:translateY(0px)}50%{transform:translateY(-6px)}100%{transform:translateY(0px)}}
@keyframes fadeUp {from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
.hero{
    padding:1.6rem 1.7rem;border-radius:26px;
    background:linear-gradient(135deg, rgba(17,25,54,.88), rgba(24,39,80,.88));
    border:1px solid var(--border);
    box-shadow:0 25px 60px rgba(0,0,0,.28);
    animation:fadeUp .8s ease;
}
.hero h1{margin:0 0 .3rem 0;font-size:2.35rem;line-height:1.1}
.hero p{margin:0;color:#d4dfff;font-size:1.01rem}
.badge{
    display:inline-block;padding:6px 12px;border-radius:999px;font-size:12px;
    background:rgba(106,168,255,.12);border:1px solid rgba(106,168,255,.30);color:#d7e6ff;margin-bottom:.6rem
}
.notice{
    padding:.9rem 1rem;border-radius:16px;
    background:rgba(255,209,102,.08);border:1px solid rgba(255,209,102,.22);color:#ffe8aa;
    animation:fadeUp 1s ease;
}
.metric-box{
    padding:1rem;border-radius:20px;
    background:linear-gradient(180deg, rgba(15,23,48,.90), rgba(22,38,78,.90));
    border:1px solid rgba(142,214,255,.16);
    box-shadow:0 10px 24px rgba(0,0,0,.15);
    animation:fadeUp .8s ease;
}
.metric-label{font-size:.82rem;color:var(--muted)}
.metric-value{font-size:1.8rem;font-weight:800;margin-top:.35rem}
.section-card{
    padding:1rem 1rem .85rem 1rem;border-radius:22px;
    background:linear-gradient(180deg, rgba(17,25,54,.92), rgba(24,39,80,.92));
    border:1px solid var(--border);
    box-shadow:0 18px 36px rgba(0,0,0,.18);
    animation:fadeUp .9s ease;
    margin-top:1rem;
}
.feature-card{
    padding:1rem;border-radius:20px;background:linear-gradient(180deg, rgba(15,23,48,.90), rgba(22,38,78,.90));
    border:1px solid rgba(142,214,255,.12);height:100%;
}
.stTabs [data-baseweb="tab-list"]{gap:8px;}
.stTabs [data-baseweb="tab"]{
    background:#0f1730;border:1px solid rgba(106,168,255,.14);border-radius:14px;padding:.4rem .8rem;color:#dbe7ff
}
.stTabs [aria-selected="true"]{
    background:linear-gradient(135deg, rgba(106,168,255,.18), rgba(61,224,208,.12)) !important;
    border-color:rgba(106,168,255,.35) !important;
}
div.stButton > button, div.stDownloadButton > button{
    border:none;border-radius:14px;padding:.7rem 1rem;font-weight:700;
    background:linear-gradient(135deg, #6aa8ff, #3de0d0);color:#08101f
}
</style>
""", unsafe_allow_html=True)

SUPPORTED = [".csv", ".xlsx", ".xls", ".txt", ".json"]
SYNONYMS = {
    "time": ["time", "date", "day", "days", "datetime", "timestamp"],
    "well": ["well", "well_name", "wellname", "api", "uwi"],
    "production": ["production", "prod", "oil_rate", "qo", "q_oil", "liquid_rate", "rate", "production_rate"],
    "pressure": ["pressure", "pres", "bhp", "reservoir_pressure", "p_res", "p"],
    "water_cut": ["watercut", "water_cut", "wc", "bsw"],
    "gor": ["gor", "gas_oil_ratio", "g_o_r", "gasoilratio"]
}

def demo_dataset():
    np.random.seed(42)
    days = np.arange(1, 61)
    wells = ["Well_A", "Well_B", "Well_C"]
    rows = []
    for i, well in enumerate(wells):
        base_q = 1500 - i * 220
        base_p = 3600 - i * 130
        wc = 0.08 + i * 0.025
        gor = 420 + i * 30
        for d in days:
            q = base_q * np.exp(-0.016 * d) + np.random.normal(0, 22)
            p = base_p - 7.5 * d + np.random.normal(0, 9)
            water = wc + 0.0025 * d + np.random.normal(0, 0.003)
            gas = gor + 2.1 * d + np.random.normal(0, 6)
            rows.append([well, d, max(q, 30), max(p, 1000), max(min(water, 0.95), 0), max(gas, 0)])
    df = pd.DataFrame(rows, columns=["Well", "Day", "Production", "Pressure", "WaterCut", "GOR"])
    df.loc[20, "Production"] = 2400
    return df

def load_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    if suffix == ".txt":
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        try:
            return pd.read_csv(io.StringIO(content))
        except Exception:
            return pd.read_csv(io.StringIO(content), sep="\t")
    if suffix == ".json":
        return pd.read_json(uploaded_file)
    raise ValueError("Unable to read file")

def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum() or ch == "_")

def auto_detect_columns(df: pd.DataFrame):
    normalized = {c: normalize_name(c) for c in df.columns}
    detected = {"time": None, "well": None, "production": None, "pressure": None, "water_cut": None, "gor": None}
    for role, names in SYNONYMS.items():
        for col, norm in normalized.items():
            if any(s in norm for s in names):
                detected[role] = col
                break
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if detected["production"] is None and num_cols:
        detected["production"] = num_cols[0]
    if detected["pressure"] is None and len(num_cols) > 1:
        detected["pressure"] = num_cols[1]
    return detected

def time_to_numeric(series):
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().sum() == len(series):
        base = dt.min()
        x = (dt - base).dt.total_seconds() / 86400.0
        return x.values, dt
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().sum() == len(series):
        return num.values.astype(float), None
    return np.arange(len(series), dtype=float), None

def detect_outliers_iqr(series):
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if len(valid) < 5:
        return pd.Series(False, index=series.index)
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (s < lo) | (s > hi)

def fit_decline_models(time_vals, q_vals):
    t = np.asarray(time_vals, dtype=float)
    q = np.asarray(q_vals, dtype=float)
    mask = np.isfinite(t) & np.isfinite(q) & (q > 0)
    t = t[mask]; q = q[mask]
    if len(q) < 5:
        return None
    q0 = q[0]

    lnq = np.log(q)
    exp_model = LinearRegression().fit(t.reshape(-1,1), lnq)
    D_exp = max(1e-9, -float(exp_model.coef_[0]))
    qhat_exp = q0 * np.exp(-D_exp * (t - t[0]))
    rmse_exp = float(np.sqrt(np.mean((q - qhat_exp)**2)))

    y_h = (q0 / q) - 1.0
    harm_model = LinearRegression().fit(t.reshape(-1,1), y_h)
    D_harm = max(1e-9, float(harm_model.coef_[0]))
    qhat_harm = q0 / (1 + D_harm * (t - t[0]))
    rmse_harm = float(np.sqrt(np.mean((q - qhat_harm)**2)))

    best = None
    for b in np.arange(0.1, 1.6, 0.1):
        y = (q0 / q)**b - 1.0
        model = LinearRegression().fit(t.reshape(-1,1), y)
        D = max(1e-9, float(model.coef_[0]) / b)
        qhat = q0 / np.power(1 + b * D * (t - t[0]), 1.0 / b)
        rmse = float(np.sqrt(np.mean((q - qhat)**2)))
        if best is None or rmse < best["rmse"]:
            best = {"b": float(b), "D": float(D), "qhat": qhat, "rmse": rmse}
    results = {
        "Exponential": {"rmse": rmse_exp, "qhat": qhat_exp, "D": D_exp},
        "Harmonic": {"rmse": rmse_harm, "qhat": qhat_harm, "D": D_harm},
        "Hyperbolic": best
    }
    best_name = min(results.keys(), key=lambda k: results[k]["rmse"])
    return best_name, results

def forecast_from_decline(time_vals, q_vals, periods=12):
    fit = fit_decline_models(time_vals, q_vals)
    if fit is None:
        return None
    best_name, results = fit
    t = np.asarray(time_vals, dtype=float)
    q = np.asarray(q_vals, dtype=float)
    mask = np.isfinite(t) & np.isfinite(q) & (q > 0)
    t = t[mask]; q = q[mask]
    q0 = q[0]
    future_t = np.arange(t[-1] + 1, t[-1] + periods + 1, dtype=float)

    if best_name == "Exponential":
        D = results["Exponential"]["D"]
        pred = q0 * np.exp(-D * (future_t - t[0]))
    elif best_name == "Harmonic":
        D = results["Harmonic"]["D"]
        pred = q0 / (1 + D * (future_t - t[0]))
    else:
        b = results["Hyperbolic"]["b"]; D = results["Hyperbolic"]["D"]
        pred = q0 / np.power(1 + b * D * (future_t - t[0]), 1.0 / b)
    return best_name, future_t, pred, results

def ai_summary(df, mapping):
    insights, recs = [], []
    prod_col = mapping.get("production")
    press_col = mapping.get("pressure")
    wc_col = mapping.get("water_cut")
    gor_col = mapping.get("gor")
    if prod_col:
        prod = pd.to_numeric(df[prod_col], errors="coerce").dropna()
        if len(prod) >= 5:
            slope = float(LinearRegression().fit(np.arange(len(prod)).reshape(-1,1), prod.values).coef_[0])
            if slope < 0:
                insights.append(f"Production is declining by about {abs(slope):.2f} units per record.")
                recs.append("Compare decline with expected reservoir depletion behavior.")
            elif slope > 0:
                insights.append(f"Production is increasing by about {slope:.2f} units per record.")
            n_out = int(detect_outliers_iqr(df[prod_col]).fillna(False).sum())
            if n_out:
                insights.append(f"{n_out} possible production outlier(s) detected.")
                recs.append("Validate unusual production points before engineering interpretation.")
    if press_col:
        p = pd.to_numeric(df[press_col], errors="coerce").dropna()
        if len(p) >= 5:
            slope = float(LinearRegression().fit(np.arange(len(p)).reshape(-1,1), p.values).coef_[0])
            if slope < 0:
                insights.append(f"Pressure is declining by about {abs(slope):.2f} units per record.")
                recs.append("Assess pressure maintenance need or operational optimization.")
    if wc_col:
        wc = pd.to_numeric(df[wc_col], errors="coerce").dropna()
        if len(wc) >= 5:
            slope = float(LinearRegression().fit(np.arange(len(wc)).reshape(-1,1), wc.values).coef_[0])
            if slope > 0:
                insights.append(f"Water cut is increasing by {slope:.4f} per record.")
                recs.append("Rising water cut may indicate water breakthrough or coning effects.")
    if gor_col:
        gor = pd.to_numeric(df[gor_col], errors="coerce").dropna()
        if len(gor) >= 5:
            slope = float(LinearRegression().fit(np.arange(len(gor)).reshape(-1,1), gor.values).coef_[0])
            if slope > 0:
                insights.append(f"GOR is increasing by {slope:.2f} units per record.")
                recs.append("Review gas behavior and depletion response.")
    if not insights:
        insights.append("Not enough mapped data for strong AI-style interpretation yet.")
    if not recs:
        recs.append("Upload richer time-series data or use the demo dataset.")
    return insights, recs

def text_report(df, mapping, selected_well, insights, recs, best_decline_name=None):
    lines = ["Petroleum Data Analysis with AI - Engineering Report", "="*55]
    lines.append(f"Rows analyzed: {len(df)}")
    if selected_well is not None:
        lines.append(f"Selected well: {selected_well}")
    lines.append("")
    lines.append("Mapped columns:")
    for k, v in mapping.items():
        if v:
            lines.append(f"- {k}: {v}")
    if best_decline_name:
        lines.append(f"- Best decline model: {best_decline_name}")
    lines.append("")
    lines.append("AI Insights:")
    for item in insights:
        lines.append(f"* {item}")
    lines.append("")
    lines.append("Recommendations:")
    for item in recs:
        lines.append(f"* {item}")
    return "\n".join(lines)

# ---------------- Header ----------------
st.markdown("""
<div class="hero">
  <div class="badge">Version 4 • Animated Professional Dashboard</div>
  <h1>Petroleum Data Analysis with AI</h1>
  <p>Upload petroleum datasets, analyze multi-well behavior, run decline analysis, detect anomalies, and generate AI-style engineering insights.</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="notice">Now includes demo data, animated professional styling, multi-well comparison, anomaly screening, decline curve analysis, forecasting, and exportable reports.</div>', unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Workspace")
    source = st.radio("Choose data source", ["Upload your file", "Use demo dataset"])
    uploaded = None
    if source == "Upload your file":
        uploaded = st.file_uploader("Upload structured dataset", type=["csv", "xlsx", "xls", "txt", "json"])
        st.caption("Supported: CSV, Excel, TXT, JSON")
    else:
        st.success("Demo dataset selected.")
    st.markdown("---")
    st.subheader("Features")
    st.write("• Auto column detection")
    st.write("• Multi-well support")
    st.write("• Charts: Production / Pressure / Water Cut / GOR")
    st.write("• Anomaly detection")
    st.write("• Decline Curve Analysis")
    st.write("• Forecasting")
    st.write("• AI engineering insights")
    st.write("• Report export")

# ---------------- Load Data ----------------
if source == "Use demo dataset":
    raw_df = demo_dataset()
else:
    if uploaded is None:
        st.info("Upload a structured dataset from the sidebar or switch to demo mode.")
        st.stop()
    try:
        raw_df = load_uploaded_file(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

if raw_df.empty:
    st.warning("The dataset is empty.")
    st.stop()

det = auto_detect_columns(raw_df)

# KPI row
st.markdown('<div class="section-card">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-box"><div class="metric-label">Rows</div><div class="metric-value">{len(raw_df)}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-box"><div class="metric-label">Columns</div><div class="metric-value">{len(raw_df.columns)}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-box"><div class="metric-label">Detected Production</div><div class="metric-value">{det["production"] or "-"}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-box"><div class="metric-label">Detected Well</div><div class="metric-value">{det["well"] or "-"}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Preview", "Column Mapping", "Visual Analytics", "AI Insights", "Reports"])

with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Dataset Preview")
    st.dataframe(raw_df.head(30), use_container_width=True)
    st.write("Columns:", list(raw_df.columns))
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Map Your Columns")
    cols = list(raw_df.columns)
    num_cols = [c for c in raw_df.columns if pd.api.types.is_numeric_dtype(raw_df[c])]
    left, mid, right = st.columns(3)
    with left:
        time_col = st.selectbox("Time / Date", cols, index=cols.index(det["time"]) if det["time"] in cols else 0)
        well_opts = ["None"] + cols
        well_col = st.selectbox("Well column", well_opts, index=well_opts.index(det["well"]) if det["well"] in well_opts else 0)
    with mid:
        prod_opts = ["None"] + num_cols
        prod_col = st.selectbox("Production", prod_opts, index=prod_opts.index(det["production"]) if det["production"] in prod_opts else 0)
        press_col = st.selectbox("Pressure", prod_opts, index=prod_opts.index(det["pressure"]) if det["pressure"] in prod_opts else 0)
    with right:
        wc_col = st.selectbox("Water Cut", prod_opts, index=prod_opts.index(det["water_cut"]) if det["water_cut"] in prod_opts else 0)
        gor_col = st.selectbox("GOR", prod_opts, index=prod_opts.index(det["gor"]) if det["gor"] in prod_opts else 0)
    mapping = {
        "time": time_col,
        "well": None if well_col == "None" else well_col,
        "production": None if prod_col == "None" else prod_col,
        "pressure": None if press_col == "None" else press_col,
        "water_cut": None if wc_col == "None" else wc_col,
        "gor": None if gor_col == "None" else gor_col,
    }
    st.success("Column mapping updated.")
    st.markdown('</div>', unsafe_allow_html=True)

try:
    mapping
except NameError:
    cols = list(raw_df.columns)
    mapping = {
        "time": det["time"] or cols[0],
        "well": det["well"],
        "production": det["production"],
        "pressure": det["pressure"],
        "water_cut": det["water_cut"],
        "gor": det["gor"],
    }

df = raw_df.copy()
selected_well = None
if mapping["well"]:
    wells = ["All"] + sorted(df[mapping["well"]].dropna().astype(str).unique().tolist())
    selected_well = st.selectbox("Filter by well", wells, key="well_filter")
    if selected_well != "All":
        df = df[df[mapping["well"]].astype(str) == selected_well].copy()

best_decline_name = None
insights, recs = [], []

with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Visual Analytics")
    a, b = st.columns(2)
    if mapping["production"]:
        fig = px.line(df, x=mapping["time"], y=mapping["production"],
                      color=mapping["well"] if mapping["well"] and selected_well == "All" else None,
                      title="Production vs Time", template="plotly_dark")
        a.plotly_chart(fig, use_container_width=True)
    if mapping["pressure"]:
        fig = px.line(df, x=mapping["time"], y=mapping["pressure"],
                      color=mapping["well"] if mapping["well"] and selected_well == "All" else None,
                      title="Pressure vs Time", template="plotly_dark")
        b.plotly_chart(fig, use_container_width=True)
    c, d = st.columns(2)
    if mapping["water_cut"]:
        fig = px.line(df, x=mapping["time"], y=mapping["water_cut"],
                      color=mapping["well"] if mapping["well"] and selected_well == "All" else None,
                      title="Water Cut vs Time", template="plotly_dark")
        c.plotly_chart(fig, use_container_width=True)
    if mapping["gor"]:
        fig = px.line(df, x=mapping["time"], y=mapping["gor"],
                      color=mapping["well"] if mapping["well"] and selected_well == "All" else None,
                      title="GOR vs Time", template="plotly_dark")
        d.plotly_chart(fig, use_container_width=True)

    if mapping["well"] and mapping["production"] and selected_well == "All":
        st.markdown("---")
        st.markdown("**Multi-Well Comparison**")
        grp = df.groupby(mapping["well"])[mapping["production"]].agg(["mean", "max", "min", "count"]).reset_index()
        grp.columns = [mapping["well"], "Average_Production", "Max_Production", "Min_Production", "Records"]
        left, right = st.columns([1.2, .8])
        left.dataframe(grp, use_container_width=True)
        fig = px.bar(grp, x=mapping["well"], y="Average_Production", title="Average Production by Well", template="plotly_dark")
        right.plotly_chart(fig, use_container_width=True)

    if mapping["production"]:
        st.markdown("---")
        st.markdown("**Production Anomaly Detection**")
        anom_mask = detect_outliers_iqr(df[mapping["production"]])
        temp = df.copy()
        temp["_anomaly"] = np.where(anom_mask.fillna(False), "Possible Outlier", "Normal")
        fig = px.scatter(temp, x=mapping["time"], y=mapping["production"], color="_anomaly",
                         title="Production Outlier Screening", template="plotly_dark",
                         color_discrete_map={"Normal":"#3de0d0","Possible Outlier":"#ff6b6b"})
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("AI Insights + Decline Analysis")
    if mapping["production"]:
        x_num, dt_values = time_to_numeric(df[mapping["time"]])
        q = pd.to_numeric(df[mapping["production"]], errors="coerce").values
        fit = fit_decline_models(x_num, q)
        if fit is None:
            st.warning("Not enough valid production data for decline analysis.")
        else:
            best_decline_name, results = fit
            st.success(f"Best decline model: {best_decline_name}")
            hist_mask = np.isfinite(x_num) & np.isfinite(q) & (q > 0)
            time_hist = df.loc[hist_mask, mapping["time"]]
            q_hist = q[hist_mask]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_hist, y=q_hist, mode="lines+markers", name="Actual"))
            for model_name, info in results.items():
                fig.add_trace(go.Scatter(x=time_hist, y=info["qhat"], mode="lines", name=f"{model_name} Fit"))
            fig.update_layout(template="plotly_dark", title="Decline Model Comparison")
            st.plotly_chart(fig, use_container_width=True)

            metrics_df = pd.DataFrame({
                "Model": list(results.keys()),
                "RMSE": [results[k]["rmse"] for k in results.keys()]
            }).sort_values("RMSE")
            st.dataframe(metrics_df, use_container_width=True)

            horizon = st.slider("Forecast periods", 6, 60, 18, 6)
            fc = forecast_from_decline(x_num, q, periods=horizon)
            if fc is not None:
                best_name, future_t, pred, _ = fc
                if dt_values is not None and dt_values.notna().sum() >= 2:
                    step = dt_values.dropna().iloc[-1] - dt_values.dropna().iloc[-2]
                    if pd.isna(step) or step == pd.Timedelta(0):
                        step = pd.Timedelta(days=1)
                    future_time = [dt_values.dropna().iloc[-1] + step * (i + 1) for i in range(horizon)]
                else:
                    future_time = future_t
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=time_hist, y=q_hist, mode="lines+markers", name="Historical"))
                fig2.add_trace(go.Scatter(x=future_time, y=pred, mode="lines+markers", name=f"{best_name} Forecast"))
                fig2.update_layout(template="plotly_dark", title="Production Forecast")
                st.plotly_chart(fig2, use_container_width=True)

    insights, recs = ai_summary(df, mapping)
    left, right = st.columns(2)
    with left:
        st.markdown("**AI Insights**")
        for item in insights:
            st.write("• " + item)
    with right:
        st.markdown("**Recommendations**")
        for item in recs:
            st.write("• " + item)
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Export")
    report = text_report(df, mapping, selected_well, insights, recs, best_decline_name)
    st.download_button("Download engineering report (.txt)", data=report.encode("utf-8"), file_name="petroleum_ai_report.txt", mime="text/plain")
    st.download_button("Download current filtered data (.csv)", data=df.to_csv(index=False).encode("utf-8"), file_name="filtered_petroleum_data.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)
