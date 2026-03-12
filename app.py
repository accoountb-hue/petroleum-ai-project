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
)

# ---------- Professional Theme ----------
st.markdown("""
<style>
:root{
    --bg:#07101d;
    --bg2:#0b1324;
    --panel:#0f1b33;
    --panel2:#16233d;
    --card:#182742;
    --text:#f8fbff;
    --muted:#d4def2;
    --soft:#b8c7e6;
    --accent:#4da3ff;
    --accent2:#2dd4bf;
    --gold:#ffd166;
    --danger:#ff6b6b;
    --border:rgba(255,255,255,.10);
    --shadow:0 18px 42px rgba(0,0,0,.20);
}

[data-testid="stAppViewContainer"]{
    background:
      linear-gradient(rgba(7,16,29,.72), rgba(7,16,29,.82)),
      url("https://images.unsplash.com/photo-1513828583688-c52646db42da?q=80&w=1800&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.block-container{
    max-width: 1380px;
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
}

html, body, [class*="css"] {
    color: var(--text);
}

.main-title{
    font-size: 2.9rem;
    font-weight: 800;
    color: white;
    line-height: 1.1;
    margin-bottom: .45rem;
}

.sub-title{
    color: var(--muted);
    font-size: 1.07rem;
    line-height: 1.8;
}

.hero{
    background: linear-gradient(135deg, rgba(16,27,51,.90), rgba(12,19,36,.88));
    border: 1px solid var(--border);
    border-radius: 26px;
    padding: 1.6rem 1.6rem 1.3rem 1.6rem;
    box-shadow: var(--shadow);
}

.badge{
    display:inline-block;
    padding:6px 12px;
    border-radius:999px;
    font-size:12px;
    color:#e5efff;
    background:rgba(77,163,255,.16);
    border:1px solid rgba(77,163,255,.28);
    margin-bottom:.8rem;
}

.section-card{
    margin-top: 1rem;
    background: linear-gradient(180deg, rgba(15,27,51,.92), rgba(17,29,52,.90));
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.15rem 1.15rem 1rem 1.15rem;
    box-shadow: 0 12px 34px rgba(0,0,0,.14);
}

.metric-card{
    background: linear-gradient(180deg, rgba(24,39,66,.96), rgba(17,29,52,.96));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem;
}

.metric-title{
    color: var(--soft);
    font-size: .84rem;
}

.metric-value{
    color: white;
    font-size: 1.9rem;
    font-weight: 800;
    margin-top: .25rem;
}

.info-grid{
    display:grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
}

.info-box{
    background: linear-gradient(180deg, rgba(24,39,66,.96), rgba(17,29,52,.96));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem;
}

.info-box h4{
    color:white;
    margin:0 0 .45rem 0;
}

.info-box p{
    color: var(--muted);
    margin:0;
    line-height:1.75;
}

.rank-box{
    background: linear-gradient(180deg, rgba(31,53,89,.96), rgba(17,29,52,.96));
    border:1px solid var(--border);
    border-radius:18px;
    padding:1rem;
}

.footer-box{
    margin-top:1rem;
    text-align:center;
    color:#dbe6fb;
    padding:1rem;
    border-radius:18px;
    background: rgba(12,19,36,.72);
    border:1px solid rgba(255,255,255,.08);
}

.ai-box{
    background: linear-gradient(180deg, rgba(26,45,76,.98), rgba(17,29,52,.96));
    border:1px solid rgba(77,163,255,.25);
    border-radius:18px;
    padding:1rem;
}

.small-note{
    color: var(--muted);
    line-height: 1.8;
    font-size: 1rem;
}

/* tabs */
.stTabs [data-baseweb="tab-list"]{
    gap:8px;
}
.stTabs [data-baseweb="tab"]{
    background:#16233d;
    border:1px solid rgba(255,255,255,.08);
    border-radius:14px;
    color:white;
    padding:.46rem .88rem;
}
.stTabs [aria-selected="true"]{
    background: linear-gradient(135deg, rgba(77,163,255,.23), rgba(45,212,191,.16)) !important;
    border-color: rgba(77,163,255,.35) !important;
}

/* buttons */
div.stButton > button, div.stDownloadButton > button{
    border:none;
    border-radius:14px;
    padding:.78rem 1rem;
    font-weight:700;
    background: linear-gradient(135deg, #4da3ff, #2dd4bf);
    color:#07111f;
}

/* inputs readable */
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div{
    background:#ffffff !important;
    color:#111827 !important;
}
.stSelectbox svg, .stMultiSelect svg{
    fill:#111827 !important;
}
.stTextInput input, .stNumberInput input, .stDateInput input{
    background:#ffffff !important;
    color:#111827 !important;
}

div[role="radiogroup"] label{
    color:white !important;
}

div[data-testid="stDataFrame"]{
    border-radius: 16px;
    overflow: hidden;
}

@media (max-width: 900px){
    .info-grid{
        grid-template-columns: 1fr;
    }
    .main-title{
        font-size: 2.1rem;
    }
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
    t = t[mask]
    q = q[mask]
    if len(q) < 5:
        return None

    q0 = q[0]

    lnq = np.log(q)
    exp_model = LinearRegression().fit(t.reshape(-1, 1), lnq)
    D_exp = max(1e-9, -float(exp_model.coef_[0]))
    qhat_exp = q0 * np.exp(-D_exp * (t - t[0]))
    rmse_exp = float(np.sqrt(np.mean((q - qhat_exp) ** 2)))

    y_h = (q0 / q) - 1.0
    harm_model = LinearRegression().fit(t.reshape(-1, 1), y_h)
    D_harm = max(1e-9, float(harm_model.coef_[0]))
    qhat_harm = q0 / (1 + D_harm * (t - t[0]))
    rmse_harm = float(np.sqrt(np.mean((q - qhat_harm) ** 2)))

    best = None
    for b in np.arange(0.1, 1.6, 0.1):
        y = (q0 / q) ** b - 1.0
        model = LinearRegression().fit(t.reshape(-1, 1), y)
        D = max(1e-9, float(model.coef_[0]) / b)
        qhat = q0 / np.power(1 + b * D * (t - t[0]), 1.0 / b)
        rmse = float(np.sqrt(np.mean((q - qhat) ** 2)))
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
    t = t[mask]
    q = q[mask]
    q0 = q[0]
    future_t = np.arange(t[-1] + 1, t[-1] + periods + 1, dtype=float)

    if best_name == "Exponential":
        D = results["Exponential"]["D"]
        pred = q0 * np.exp(-D * (future_t - t[0]))
    elif best_name == "Harmonic":
        D = results["Harmonic"]["D"]
        pred = q0 / (1 + D * (future_t - t[0]))
    else:
        b = results["Hyperbolic"]["b"]
        D = results["Hyperbolic"]["D"]
        pred = q0 / np.power(1 + b * D * (future_t - t[0]), 1.0 / b)

    return best_name, future_t, pred, results

def rank_wells(df, well_col, production_col):
    grp = df.groupby(well_col)[production_col].agg(["mean", "max", "min", "count"]).reset_index()
    grp.columns = [well_col, "Average_Production", "Max_Production", "Min_Production", "Records"]
    grp = grp.sort_values("Average_Production", ascending=False).reset_index(drop=True)
    grp["Rank"] = np.arange(1, len(grp) + 1)
    return grp

def missing_values_summary(df):
    missing = df.isna().sum()
    result = pd.DataFrame({
        "Column": missing.index,
        "Missing_Values": missing.values,
        "Missing_Percentage": (missing.values / len(df)) * 100
    }).sort_values("Missing_Values", ascending=False)
    return result

def correlation_data(df, mapping):
    cols = []
    for key in ["production", "pressure", "water_cut", "gor"]:
        col = mapping.get(key)
        if col:
            cols.append(col)
    if len(cols) < 2:
        return None
    corr_df = df[cols].apply(pd.to_numeric, errors="coerce")
    return corr_df.corr()

def ai_summary(df, mapping):
    insights, recs = [], []

    prod_col = mapping.get("production")
    press_col = mapping.get("pressure")
    wc_col = mapping.get("water_cut")
    gor_col = mapping.get("gor")

    prod = pd.to_numeric(df[prod_col], errors="coerce") if prod_col else pd.Series(dtype=float)
    press = pd.to_numeric(df[press_col], errors="coerce") if press_col else pd.Series(dtype=float)
    wc = pd.to_numeric(df[wc_col], errors="coerce") if wc_col else pd.Series(dtype=float)
    gor = pd.to_numeric(df[gor_col], errors="coerce") if gor_col else pd.Series(dtype=float)

    well_class = "Unclassified"

    if len(prod.dropna()) >= 5:
        prod_clean = prod.dropna()
        prod_start, prod_end = prod_clean.iloc[0], prod_clean.iloc[-1]
        prod_change = prod_end - prod_start
        prod_pct = (prod_change / prod_start) * 100 if prod_start != 0 else 0

        if prod_change < 0:
            insights.append(f"Production declined by {abs(prod_pct):.1f}% over the analyzed period.")
            recs.append("Review production decline against expected reservoir and operational performance.")
        else:
            insights.append(f"Production increased by {prod_pct:.1f}% over the analyzed period.")

        slope = float(LinearRegression().fit(np.arange(len(prod_clean)).reshape(-1, 1), prod_clean.values).coef_[0])
        if slope < -5:
            insights.append("The well shows a strong negative production trend.")
        elif slope < 0:
            insights.append("The well shows a mild production decline.")
        else:
            insights.append("The well shows stable to improving production behavior.")

        n_out = int(detect_outliers_iqr(prod).fillna(False).sum())
        if n_out > 0:
            insights.append(f"{n_out} production anomaly point(s) detected.")
            recs.append("Validate unusual production spikes or drops before final engineering interpretation.")

    wc_change = 0
    gor_change = 0

    if len(press.dropna()) >= 5:
        press_clean = press.dropna()
        press_start, press_end = press_clean.iloc[0], press_clean.iloc[-1]
        press_change = press_end - press_start
        if press_change < 0:
            insights.append(f"Reservoir pressure decreased by {abs(press_change):.1f} units.")
            recs.append("Evaluate depletion management and pressure support conditions.")

    if len(wc.dropna()) >= 5:
        wc_clean = wc.dropna()
        wc_start, wc_end = wc_clean.iloc[0], wc_clean.iloc[-1]
        wc_change = wc_end - wc_start
        if wc_change > 0.03:
            insights.append(f"Water cut increased by {wc_change:.3f}, indicating stronger water influence.")
            recs.append("Investigate potential water breakthrough, coning, or sweep changes.")
        elif wc_change > 0:
            insights.append("Water cut is rising gradually.")

    if len(gor.dropna()) >= 5:
        gor_clean = gor.dropna()
        gor_start, gor_end = gor_clean.iloc[0], gor_clean.iloc[-1]
        gor_change = gor_end - gor_start
        if gor_change > 30:
            insights.append(f"GOR increased by {gor_change:.1f}, suggesting stronger gas contribution.")
            recs.append("Review gas behavior, depletion effects, and possible gas breakthrough.")
        elif gor_change > 0:
            insights.append("GOR is trending upward slightly.")

    if len(prod.dropna()) >= 5 and len(press.dropna()) >= 5:
        tmp = pd.DataFrame({"prod": prod, "press": press}).dropna()
        if len(tmp) >= 5:
            corr = tmp["prod"].corr(tmp["press"])
            insights.append(f"Production-pressure correlation = {corr:.2f}.")
            if corr > 0.5:
                recs.append("Production appears strongly linked to pressure depletion behavior.")

    if len(prod.dropna()) >= 5:
        prod_change = prod.dropna().iloc[-1] - prod.dropna().iloc[0]

        if prod_change < 0 and wc_change > 0.03:
            well_class = "Water-Risk Well"
        elif prod_change < 0 and gor_change > 30:
            well_class = "Gas-Risk Well"
        elif prod_change < 0:
            well_class = "Declining Well"
        else:
            well_class = "Stable / Improving Well"

    insights.append(f"AI well classification: {well_class}.")

    if not recs:
        recs.append("Continue monitoring with more historical data for stronger AI interpretation.")

    return insights, recs, well_class

def text_report(df, mapping, selected_well, insights, recs, best_decline_name=None, well_class=None):
    lines = ["Petroleum Data Analysis with AI - Engineering Report", "=" * 55]
    lines.append(f"Rows analyzed: {len(df)}")
    if selected_well is not None:
        lines.append(f"Selected well: {selected_well}")
    if well_class:
        lines.append(f"AI well classification: {well_class}")
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

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="badge">Version 7 • AI Petroleum Data Scientist</div>
  <div class="main-title">Petroleum Data Analysis with AI</div>
  <div class="sub-title">
    An AI-oriented petroleum data science platform for trend interpretation, well comparison,
    anomaly detection, decline diagnostics, correlation review, and engineering reporting.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-card">
  <h3 style="margin-top:0;">Why This Platform Matters</h3>
  <div class="info-grid">
    <div class="info-box">
      <h4>Petroleum Data Science Workflow</h4>
      <p>Transforms structured well data into analytical outputs through visualization, anomaly screening, and AI-style interpretation.</p>
    </div>
    <div class="info-box">
      <h4>Operational Intelligence</h4>
      <p>Supports production review, water and gas risk tracking, and comparative assessment across wells.</p>
    </div>
    <div class="info-box">
      <h4>Portfolio-Ready Engineering Tool</h4>
      <p>Demonstrates applied petroleum analytics using data science concepts in a clean professional dashboard.</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Controls ----------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Start Analysis")
ctrl1, ctrl2 = st.columns([1.05, 1.45])

with ctrl1:
    source = st.radio("Choose data source", ["Upload your file", "Use demo dataset"])

with ctrl2:
    uploaded = None
    if source == "Upload your file":
        uploaded = st.file_uploader(
            "Upload structured dataset",
            type=["csv", "xlsx", "xls", "txt", "json"]
        )
        st.caption("Supported: CSV, Excel, TXT, JSON")
    else:
        st.info("Demo petroleum dataset is active.")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-card">
  <h3 style="margin-top:0;">How to Use This Platform</h3>
  <div class="info-grid">
    <div class="info-box">
      <h4>1. Load Data</h4>
      <p>Upload a real petroleum dataset or start with the built-in demo data.</p>
    </div>
    <div class="info-box">
      <h4>2. Map Columns</h4>
      <p>Select engineering columns such as time, production, pressure, water cut, and GOR.</p>
    </div>
    <div class="info-box">
      <h4>3. Generate Insights</h4>
      <p>Review charts, compare wells, inspect AI classification, evaluate correlations, and export the report.</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Load data ----------
if source == "Use demo dataset":
    raw_df = demo_dataset()
else:
    if uploaded is None:
        st.info("Upload a structured dataset to continue.")
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

# ---------- KPI row ----------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Rows</div><div class="metric-value">{len(raw_df)}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Columns</div><div class="metric-value">{len(raw_df.columns)}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Detected Production</div><div class="metric-value">{det["production"] or "-"}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Detected Well</div><div class="metric-value">{det["well"] or "-"}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Data Preview",
    "Column Mapping",
    "Analytics",
    "AI Data Scientist",
    "Reports"
])

with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Overview")
    st.markdown("""
<div class="info-grid">
  <div class="info-box">
    <h4>Supported Inputs</h4>
    <p>Structured petroleum datasets with time, production, pressure, water cut, GOR, and well identifiers.</p>
  </div>
  <div class="info-box">
    <h4>Data Science Features</h4>
    <p>Includes anomaly screening, correlation review, decline analysis, well ranking, and AI-style classification.</p>
  </div>
  <div class="info-box">
    <h4>Engineering Outputs</h4>
    <p>Provides practical decision-support style summaries suitable for academic and technical presentation.</p>
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Dataset Preview")
    st.dataframe(raw_df.head(30), use_container_width=True)
    st.write("Columns:", list(raw_df.columns))
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Map Engineering Columns")

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
insights, recs, well_class = [], [], "Unclassified"

with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Analytics")

    a, b = st.columns(2)
    if mapping["production"]:
        fig = px.line(
            df,
            x=mapping["time"],
            y=mapping["production"],
            color=mapping["well"] if mapping["well"] and selected_well == "All" else None,
            title="Production vs Time",
            template="plotly_dark"
        )
        a.plotly_chart(fig, use_container_width=True)

    if mapping["pressure"]:
        fig = px.line(
            df,
            x=mapping["time"],
            y=mapping["pressure"],
            color=mapping["well"] if mapping["well"] and selected_well == "All" else None,
            title="Pressure vs Time",
            template="plotly_dark"
        )
        b.plotly_chart(fig, use_container_width=True)

    c, d = st.columns(2)
    if mapping["water_cut"]:
        fig = px.line(
            df,
            x=mapping["time"],
            y=mapping["water_cut"],
            color=mapping["well"] if mapping["well"] and selected_well == "All" else None,
            title="Water Cut vs Time",
            template="plotly_dark"
        )
        c.plotly_chart(fig, use_container_width=True)

    if mapping["gor"]:
        fig = px.line(
            df,
            x=mapping["time"],
            y=mapping["gor"],
            color=mapping["well"] if mapping["well"] and selected_well == "All" else None,
            title="GOR vs Time",
            template="plotly_dark"
        )
        d.plotly_chart(fig, use_container_width=True)

    if mapping["well"] and mapping["production"] and selected_well == "All":
        st.markdown("---")
        st.subheader("Well Performance Ranking")
        ranking_df = rank_wells(df, mapping["well"], mapping["production"])

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f'<div class="rank-box"><b>Top Well</b><br><br>{ranking_df.iloc[0][mapping["well"]]}<br><span style="color:#d6e4ff;">Avg. Production: {ranking_df.iloc[0]["Average_Production"]:.2f}</span></div>', unsafe_allow_html=True)
        with r2:
            if len(ranking_df) > 1:
                st.markdown(f'<div class="rank-box"><b>Second Rank</b><br><br>{ranking_df.iloc[1][mapping["well"]]}<br><span style="color:#d6e4ff;">Avg. Production: {ranking_df.iloc[1]["Average_Production"]:.2f}</span></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="rank-box"><b>Lowest Ranked Well</b><br><br>{ranking_df.iloc[-1][mapping["well"]]}<br><span style="color:#d6e4ff;">Avg. Production: {ranking_df.iloc[-1]["Average_Production"]:.2f}</span></div>', unsafe_allow_html=True)

        st.dataframe(ranking_df, use_container_width=True)

    if mapping["production"]:
        st.markdown("---")
        st.subheader("Production Anomaly Detection")
        anom_mask = detect_outliers_iqr(df[mapping["production"]])
        temp = df.copy()
        temp["_anomaly"] = np.where(anom_mask.fillna(False), "Possible Outlier", "Normal")
        fig = px.scatter(
            temp,
            x=mapping["time"],
            y=mapping["production"],
            color="_anomaly",
            title="Production Outlier Screening",
            template="plotly_dark",
            color_discrete_map={"Normal": "#2dd4bf", "Possible Outlier": "#ff6b6b"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Missing Values Summary")
    miss_df = missing_values_summary(df)
    st.dataframe(miss_df, use_container_width=True)

    corr = correlation_data(df, mapping)
    if corr is not None:
        st.markdown("---")
        st.subheader("Correlation Matrix")
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("AI Petroleum Data Scientist")

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

    insights, recs, well_class = ai_summary(df, mapping)

    class_col, kpi_col = st.columns([1.2, 1])
    with class_col:
        st.markdown(f"""
<div class="ai-box">
    <h3 style="margin-top:0;">AI Well Classification</h3>
    <p style="font-size:1.2rem;font-weight:700;color:white;margin-bottom:0;">{well_class}</p>
</div>
""", unsafe_allow_html=True)

    with kpi_col:
        outlier_count = 0
        if mapping.get("production"):
            outlier_count = int(detect_outliers_iqr(df[mapping["production"]]).fillna(False).sum())
        st.markdown(f"""
<div class="ai-box">
    <h3 style="margin-top:0;">AI Data Summary</h3>
    <p style="margin:0;color:#d8e4ff;">Rows analyzed: <b>{len(df)}</b></p>
    <p style="margin:0;color:#d8e4ff;">Anomalies detected: <b>{outlier_count}</b></p>
    <p style="margin:0;color:#d8e4ff;">Best decline model: <b>{best_decline_name if best_decline_name else "-"}</b></p>
</div>
""", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.subheader("AI Insights")
        for item in insights:
            st.write("• " + item)

    with right:
        st.subheader("Recommended Actions")
        for item in recs:
            st.write("• " + item)

    st.markdown("""
<div class="section-card" style="margin-top:1rem;">
  <h3 style="margin-top:0;">AI Summary Box</h3>
  <p class="small-note">
    This module behaves like an early-stage petroleum data scientist assistant. It reviews production decline,
    pressure depletion, water-cut growth, GOR movement, anomalies, and basic cross-variable relationships
    to produce a structured engineering interpretation.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab6:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Reports")

    report = text_report(df, mapping, selected_well, insights, recs, best_decline_name, well_class)

    st.download_button(
        "Download engineering report (.txt)",
        data=report.encode("utf-8"),
        file_name="petroleum_ai_report.txt",
        mime="text/plain"
    )

    st.download_button(
        "Download current filtered data (.csv)",
        data=df.to_csv(index=False),
        file_name="filtered_petroleum_data.csv",
        mime="text/csv"
    )

    st.markdown("""
<div class="footer-box">
    Developed by Abbas • Petroleum Engineering • AI Petroleum Data Scientist Platform
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
