import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Petroleum Data Analysis with AI",
    page_icon="🛢️",
    layout="wide",
)

# ---------- Theme ----------
st.markdown("""
<style>
:root{
    --text:#f8fbff;
    --muted:#d4def2;
    --soft:#b8c7e6;
    --border:rgba(255,255,255,.10);
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
html, body, [class*="css"] { color: var(--text); }
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
    padding: 1.6rem;
    box-shadow: 0 18px 42px rgba(0,0,0,.20);
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
    padding: 1.15rem;
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
.ai-box{
    background: linear-gradient(180deg, rgba(26,45,76,.98), rgba(17,29,52,.96));
    border:1px solid rgba(77,163,255,.25);
    border-radius:18px;
    padding:1rem;
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
.stTabs [data-baseweb="tab-list"]{ gap:8px; }
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
div.stButton > button, div.stDownloadButton > button{
    border:none;
    border-radius:14px;
    padding:.78rem 1rem;
    font-weight:700;
    background: linear-gradient(135deg, #4da3ff, #2dd4bf);
    color:#07111f;
}
.stSelectbox div[data-baseweb="select"] > div{
    background:#ffffff !important;
    color:#111827 !important;
}
.stSelectbox svg{ fill:#111827 !important; }
div[role="radiogroup"] label{ color:white !important; }
@media (max-width: 900px){
    .info-grid{ grid-template-columns: 1fr; }
    .main-title{ font-size: 2.1rem; }
}
</style>
""", unsafe_allow_html=True)

SUPPORTED = [".csv", ".xlsx", ".xls", ".txt", ".json"]
SYNONYMS = {
    "time": ["time", "date", "day", "days", "datetime", "timestamp"],
    "well": ["well", "well_name", "wellname", "api", "uwi"],
    "production": ["production", "prod", "oil_rate", "qo", "q_oil", "liquid_rate", "rate", "production_rate"],
    "pressure": ["pressure", "pres", "bhp", "reservoir_pressure", "p_res"],
    "water_cut": ["watercut", "water_cut", "wc", "bsw"],
    "gor": ["gor", "gas_oil_ratio", "g_o_r", "gasoilratio"]
}

# ---------- Helpers ----------
def demo_dataset():
    np.random.seed(42)
    days = np.arange(1, 91)
    wells = ["Well_A", "Well_B", "Well_C", "Well_D"]
    rows = []

    for i, well in enumerate(wells):
        base_q = 1600 - i * 220
        base_p = 3700 - i * 110
        wc = 0.07 + i * 0.02
        gor = 420 + i * 25

        for d in days:
            q = base_q * np.exp(-0.013 * d) + np.random.normal(0, 18)
            p = base_p - 5.8 * d + np.random.normal(0, 7)
            water = wc + 0.0018 * d + np.random.normal(0, 0.0025)
            gas = gor + 1.7 * d + np.random.normal(0, 5)

            rows.append([
                well,
                d,
                max(q, 20),
                max(p, 900),
                max(min(water, 0.95), 0),
                max(gas, 0)
            ])

    df = pd.DataFrame(rows, columns=["Well", "Day", "Production", "Pressure", "WaterCut", "GOR"])
    df.loc[25, "Production"] = df.loc[25, "Production"] * 1.7
    df.loc[170, "Production"] = df.loc[170, "Production"] * 0.4
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
    detected = {
        "time": None,
        "well": None,
        "production": None,
        "pressure": None,
        "water_cut": None,
        "gor": None
    }

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

def rank_wells(df, well_col, production_col):
    grp = df.groupby(well_col)[production_col].agg(["mean", "max", "min", "count"]).reset_index()
    grp.columns = [well_col, "Average_Production", "Max_Production", "Min_Production", "Records"]
    grp = grp.sort_values("Average_Production", ascending=False).reset_index(drop=True)
    grp["Rank"] = np.arange(1, len(grp) + 1)
    return grp

def missing_values_summary(df):
    missing = df.isna().sum()
    return pd.DataFrame({
        "Column": missing.index,
        "Missing_Values": missing.values,
        "Missing_Percentage": (missing.values / len(df)) * 100
    }).sort_values("Missing_Values", ascending=False)

def unique_analysis_cols(mapping):
    vals = []
    for key in ["production", "pressure", "water_cut", "gor"]:
        v = mapping.get(key)
        if v and v not in vals:
            vals.append(v)
    return vals

def correlation_matrix_safe(df, mapping):
    cols = unique_analysis_cols(mapping)
    if len(cols) < 2:
        return None
    corr_df = df[cols].apply(pd.to_numeric, errors="coerce")
    return corr_df.corr()

# ---------- Decline / Prediction ----------
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

def predictive_ai(df, mapping):
    results = {}
    time_col = mapping.get("time")
    if not time_col:
        return results

    for key in ["production", "pressure", "water_cut", "gor"]:
        col = mapping.get(key)
        if not col:
            continue

        x, _ = time_to_numeric(df[time_col])
        y = pd.to_numeric(df[col], errors="coerce").values

        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 5:
            continue

        model = LinearRegression()
        model.fit(x_clean.reshape(-1, 1), y_clean)

        horizon = 30
        future_x = np.arange(x_clean[-1] + 1, x_clean[-1] + horizon + 1)
        pred = model.predict(future_x.reshape(-1, 1))

        results[key] = {
            "future_x": future_x,
            "pred": pred,
            "historical_x": x_clean,
            "historical_y": y_clean,
            "slope": float(model.coef_[0]),
        }

    return results

# ---------- AI ----------
def calculate_risk_scores(df, mapping):
    prod_col = mapping.get("production")
    press_col = mapping.get("pressure")
    wc_col = mapping.get("water_cut")
    gor_col = mapping.get("gor")

    prod = pd.to_numeric(df[prod_col], errors="coerce").dropna() if prod_col else pd.Series(dtype=float)
    press = pd.to_numeric(df[press_col], errors="coerce").dropna() if press_col else pd.Series(dtype=float)
    wc = pd.to_numeric(df[wc_col], errors="coerce").dropna() if wc_col else pd.Series(dtype=float)
    gor = pd.to_numeric(df[gor_col], errors="coerce").dropna() if gor_col else pd.Series(dtype=float)

    water_risk = 0.0
    gas_risk = 0.0
    depletion_risk = 0.0
    anomaly_risk = 0.0

    if len(wc) >= 5:
        water_risk = min(max((wc.iloc[-1] - wc.iloc[0]) * 1200, 0), 100)
    if len(gor) >= 5:
        gas_risk = min(max((gor.iloc[-1] - gor.iloc[0]) * 1.2, 0), 100)
    if len(press) >= 5:
        depletion_risk = min(max((press.iloc[0] - press.iloc[-1]) / 8, 0), 100)
    if prod_col:
        anomalies = int(detect_outliers_iqr(df[prod_col]).fillna(False).sum())
        anomaly_risk = min(anomalies * 15, 100)

    return {
        "water_risk": round(water_risk, 1),
        "gas_risk": round(gas_risk, 1),
        "depletion_risk": round(depletion_risk, 1),
        "anomaly_risk": round(anomaly_risk, 1),
    }

def calculate_reservoir_health_score(df, mapping, risks):
    score = 100.0
    score -= risks["water_risk"] * 0.25
    score -= risks["gas_risk"] * 0.20
    score -= risks["depletion_risk"] * 0.25
    score -= risks["anomaly_risk"] * 0.15

    prod_col = mapping.get("production")
    if prod_col:
        prod = pd.to_numeric(df[prod_col], errors="coerce").dropna()
        if len(prod) >= 5 and prod.iloc[0] != 0:
            prod_decline_pct = ((prod.iloc[0] - prod.iloc[-1]) / prod.iloc[0]) * 100
            score -= max(prod_decline_pct, 0) * 0.20

    return max(min(round(score, 1), 100), 0)

def classify_well(df, mapping, risks):
    prod_col = mapping.get("production")
    press_col = mapping.get("pressure")
    wc_col = mapping.get("water_cut")
    gor_col = mapping.get("gor")

    prod = pd.to_numeric(df[prod_col], errors="coerce").dropna() if prod_col else pd.Series(dtype=float)
    press = pd.to_numeric(df[press_col], errors="coerce").dropna() if press_col else pd.Series(dtype=float)
    wc = pd.to_numeric(df[wc_col], errors="coerce").dropna() if wc_col else pd.Series(dtype=float)
    gor = pd.to_numeric(df[gor_col], errors="coerce").dropna() if gor_col else pd.Series(dtype=float)

    prod_change = prod.iloc[-1] - prod.iloc[0] if len(prod) >= 5 else 0
    press_change = press.iloc[-1] - press.iloc[0] if len(press) >= 5 else 0
    wc_change = wc.iloc[-1] - wc.iloc[0] if len(wc) >= 5 else 0
    gor_change = gor.iloc[-1] - gor.iloc[0] if len(gor) >= 5 else 0

    if risks["anomaly_risk"] >= 45:
        return "Data-Anomaly Well"
    if prod_change < 0 and wc_change > 0.03 and risks["water_risk"] >= 40:
        return "Water-Risk Well"
    if prod_change < 0 and gor_change > 30 and risks["gas_risk"] >= 40:
        return "Gas-Risk Well"
    if prod_change < 0 and press_change < 0 and risks["depletion_risk"] >= 35:
        return "Pressure-Depletion Well"
    if prod_change < 0:
        return "Declining Well"
    return "Stable / Improving Well"

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

    prod_clean = prod.dropna()
    press_clean = press.dropna()
    wc_clean = wc.dropna()
    gor_clean = gor.dropna()

    risks = calculate_risk_scores(df, mapping)
    health_score = calculate_reservoir_health_score(df, mapping, risks)
    well_class = classify_well(df, mapping, risks)

    prod_pct = 0
    wc_change = 0
    gor_change = 0

    if len(prod_clean) >= 5 and prod_clean.iloc[0] != 0:
        prod_start, prod_end = prod_clean.iloc[0], prod_clean.iloc[-1]
        prod_change = prod_end - prod_start
        prod_pct = (prod_change / prod_start) * 100

        if prod_change < 0:
            insights.append(f"Production declined by {abs(prod_pct):.1f}% over the analyzed period.")
            recs.append("Review production decline versus expected field and reservoir performance.")
        else:
            insights.append(f"Production increased by {prod_pct:.1f}% over the analyzed period.")

        slope = float(LinearRegression().fit(np.arange(len(prod_clean)).reshape(-1, 1), prod_clean.values).coef_[0])
        if slope < -10:
            insights.append("Production trend severity: severe decline.")
        elif slope < -3:
            insights.append("Production trend severity: moderate decline.")
        elif slope < 0:
            insights.append("Production trend severity: mild decline.")
        else:
            insights.append("Production trend severity: stable to improving.")

        n_out = int(detect_outliers_iqr(df[prod_col]).fillna(False).sum())
        if n_out > 0:
            insights.append(f"{n_out} production anomaly point(s) detected.")
            recs.append("Validate abnormal spikes or drops before engineering decisions.")

    if len(press_clean) >= 5:
        press_drop = press_clean.iloc[0] - press_clean.iloc[-1]
        if press_drop > 0:
            insights.append(f"Reservoir pressure dropped by {press_drop:.1f} units.")
            recs.append("Evaluate depletion support and pressure-maintenance conditions.")

    if len(wc_clean) >= 5:
        wc_change = wc_clean.iloc[-1] - wc_clean.iloc[0]
        if wc_change > 0.03:
            insights.append(f"Water cut increased by {wc_change:.3f}, indicating stronger water influence.")
            recs.append("Investigate possible water breakthrough, coning, or sweep changes.")
        elif wc_change > 0:
            insights.append("Water cut is rising gradually.")

    if len(gor_clean) >= 5:
        gor_change = gor_clean.iloc[-1] - gor_clean.iloc[0]
        if gor_change > 30:
            insights.append(f"GOR increased by {gor_change:.1f}, suggesting stronger gas contribution.")
            recs.append("Review gas behavior, phase changes, and breakthrough risk.")
        elif gor_change > 0:
            insights.append("GOR is trending upward slightly.")

    if len(prod_clean) >= 5 and len(press_clean) >= 5:
        tmp = pd.DataFrame({
            "prod": prod_clean.reset_index(drop=True),
            "press": press_clean.reset_index(drop=True)
        }).dropna()
        if len(tmp) >= 5:
            corr = tmp["prod"].corr(tmp["press"])
            insights.append(f"Production-pressure correlation = {corr:.2f}.")
            if corr > 0.5:
                recs.append("Production appears strongly linked to pressure depletion behavior.")

    if len(prod_clean) >= 5:
        if prod_pct < 0 and wc_change > 0.03:
            insights.append("Smart AI logic: decline pattern is consistent with water breakthrough risk.")
        if prod_pct < 0 and gor_change > 30:
            insights.append("Smart AI logic: decline pattern is consistent with gas breakthrough risk.")
        if prod_pct < 0 and len(press_clean) >= 5 and (press_clean.iloc[-1] < press_clean.iloc[0]):
            insights.append("Smart AI logic: production decline is likely depletion-driven.")
        if prod_pct < 0 and len(press_clean) >= 5 and abs(press_clean.iloc[-1] - press_clean.iloc[0]) < 10:
            insights.append("Smart AI logic: production decline with nearly stable pressure may suggest operational or formation-related issues.")

    insights.append(f"AI well classification: {well_class}.")
    insights.append(f"Reservoir health score: {health_score}/100.")
    insights.append(
        f"Risk scores → Water: {risks['water_risk']}, Gas: {risks['gas_risk']}, Depletion: {risks['depletion_risk']}, Data quality: {risks['anomaly_risk']}."
    )

    if not recs:
        recs.append("Continue monitoring with more historical data for stronger AI interpretation.")

    return insights, recs, well_class, risks, health_score

def generate_ai_report(df, mapping, selected_well, insights, recs, best_decline_name, well_class, risks, health_score, decline_params_text):
    lines = []
    lines.append("AI PETROLEUM DATA SCIENTIST REPORT")
    lines.append("=" * 60)
    lines.append(f"Rows analyzed: {len(df)}")
    if selected_well is not None:
        lines.append(f"Selected well: {selected_well}")
    lines.append(f"AI well classification: {well_class}")
    lines.append(f"Reservoir health score: {health_score}/100")
    if best_decline_name:
        lines.append(f"Best decline model: {best_decline_name}")
    if decline_params_text:
        lines.append(f"Decline parameters: {decline_params_text}")
    lines.append("")

    lines.append("Mapped Columns:")
    for k, v in mapping.items():
        if v:
            lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("Risk Scores:")
    lines.append(f"- Water breakthrough risk: {risks['water_risk']}/100")
    lines.append(f"- Gas breakthrough risk: {risks['gas_risk']}/100")
    lines.append(f"- Pressure depletion risk: {risks['depletion_risk']}/100")
    lines.append(f"- Data quality / anomaly risk: {risks['anomaly_risk']}/100")

    lines.append("")
    lines.append("AI Insights:")
    for item in insights:
        lines.append(f"* {item}")

    lines.append("")
    lines.append("Recommended Actions:")
    for item in recs:
        lines.append(f"* {item}")

    lines.append("")
    lines.append("Summary:")
    lines.append(
        "This report combines smart petroleum reasoning, deeper production/pressure/water/gas analysis, "
        "risk scoring, reservoir health assessment, predictive diagnostics, and decline interpretation "
        "to support early-stage engineering decision making."
    )

    return "\n".join(lines)

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="badge">Version 10 • Fixed + Phase 2 Complete</div>
  <div class="main-title">Petroleum Data Analysis with AI</div>
  <div class="sub-title">
    Full AI petroleum analytics with smart logic, deeper engineering interpretation,
    duplicate-safe mapping, risk scoring, reservoir health scoring, predictive diagnostics, and AI report writing.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-card">
  <h3 style="margin-top:0;">What Was Fixed</h3>
  <div class="info-grid">
    <div class="info-box">
      <h4>Duplicate Mapping Protection</h4>
      <p>The app now stops safely if the same column is assigned to multiple engineering roles.</p>
    </div>
    <div class="info-box">
      <h4>Stable Correlation Logic</h4>
      <p>Duplicate-safe analysis columns prevent the DuplicateError crash from happening again.</p>
    </div>
    <div class="info-box">
      <h4>Phase 2 Complete</h4>
      <p>Deep petroleum AI, risk scoring, health score, prediction, ranking, and AI report writing are included.</p>
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
        uploaded = st.file_uploader("Upload structured dataset", type=["csv", "xlsx", "xls", "txt", "json"])
        st.caption("Supported: CSV, Excel, TXT, JSON")
    else:
        st.info("Demo petroleum dataset is active.")
st.markdown('</div>', unsafe_allow_html=True)

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
    "AI Engine",
    "Reports"
])

with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Overview")
    st.markdown("""
<div class="info-grid">
  <div class="info-box">
    <h4>Duplicate-Protected Workflow</h4>
    <p>Wrong mappings are stopped immediately before any chart or AI logic starts.</p>
  </div>
  <div class="info-box">
    <h4>Deep Petroleum Analysis</h4>
    <p>Production, pressure, water cut, and GOR are interpreted together through smart AI logic.</p>
  </div>
  <div class="info-box">
    <h4>Portfolio-Ready Output</h4>
    <p>The platform now includes risk scores, well classification, prediction, and AI report export.</p>
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

    selected_mapping_values = [v for v in mapping.values() if v is not None]
    if len(selected_mapping_values) != len(set(selected_mapping_values)):
        st.error("Duplicate column mapping detected. Please choose a different column for each role.")
        st.stop()

    st.success("Column mapping updated.")
    st.json(mapping)
    st.markdown('</div>', unsafe_allow_html=True)

# fallback mapping if tab3 not interacted with yet
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

selected_mapping_values = [v for v in mapping.values() if v is not None]
if len(selected_mapping_values) != len(set(selected_mapping_values)):
    st.error("Duplicate column mapping detected. Fix the mapping in Column Mapping tab.")
    st.stop()

df = raw_df.copy()
selected_well = None
if mapping["well"]:
    wells = ["All"] + sorted(df[mapping["well"]].dropna().astype(str).unique().tolist())
    selected_well = st.selectbox("Filter by well", wells, key="well_filter")
    if selected_well != "All":
        df = df[df[mapping["well"]].astype(str) == selected_well].copy()

best_decline_name = None
decline_params_text = ""
insights, recs, well_class = [], [], "Unclassified"
risks = {"water_risk": 0, "gas_risk": 0, "depletion_risk": 0, "anomaly_risk": 0}
health_score = 0

with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Analytics")

    if mapping["production"]:
        fig_prod = go.Figure()
        if mapping["well"] and selected_well == "All":
            for well_name, sub in df.groupby(mapping["well"]):
                fig_prod.add_trace(go.Scatter(x=sub[mapping["time"]], y=sub[mapping["production"]], mode="lines", name=str(well_name)))
        else:
            fig_prod.add_trace(go.Scatter(x=df[mapping["time"]], y=df[mapping["production"]], mode="lines+markers", name="Production"))
        fig_prod.update_layout(template="plotly_dark", title="Production vs Time")
        st.plotly_chart(fig_prod, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if mapping["pressure"]:
            fig_press = go.Figure()
            if mapping["well"] and selected_well == "All":
                for well_name, sub in df.groupby(mapping["well"]):
                    fig_press.add_trace(go.Scatter(x=sub[mapping["time"]], y=sub[mapping["pressure"]], mode="lines", name=str(well_name)))
            else:
                fig_press.add_trace(go.Scatter(x=df[mapping["time"]], y=df[mapping["pressure"]], mode="lines+markers", name="Pressure"))
            fig_press.update_layout(template="plotly_dark", title="Pressure vs Time")
            st.plotly_chart(fig_press, use_container_width=True)

    with c2:
        if mapping["water_cut"]:
            fig_wc = go.Figure()
            if mapping["well"] and selected_well == "All":
                for well_name, sub in df.groupby(mapping["well"]):
                    fig_wc.add_trace(go.Scatter(x=sub[mapping["time"]], y=sub[mapping["water_cut"]], mode="lines", name=str(well_name)))
            else:
                fig_wc.add_trace(go.Scatter(x=df[mapping["time"]], y=df[mapping["water_cut"]], mode="lines+markers", name="Water Cut"))
            fig_wc.update_layout(template="plotly_dark", title="Water Cut vs Time")
            st.plotly_chart(fig_wc, use_container_width=True)

    if mapping["gor"]:
        fig_gor = go.Figure()
        if mapping["well"] and selected_well == "All":
            for well_name, sub in df.groupby(mapping["well"]):
                fig_gor.add_trace(go.Scatter(x=sub[mapping["time"]], y=sub[mapping["gor"]], mode="lines", name=str(well_name)))
        else:
            fig_gor.add_trace(go.Scatter(x=df[mapping["time"]], y=df[mapping["gor"]], mode="lines+markers", name="GOR"))
        fig_gor.update_layout(template="plotly_dark", title="GOR vs Time")
        st.plotly_chart(fig_gor, use_container_width=True)

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

        fig_anom = go.Figure()
        normal = temp[temp["_anomaly"] == "Normal"]
        out = temp[temp["_anomaly"] == "Possible Outlier"]

        fig_anom.add_trace(go.Scatter(x=normal[mapping["time"]], y=normal[mapping["production"]], mode="markers", name="Normal"))
        fig_anom.add_trace(go.Scatter(x=out[mapping["time"]], y=out[mapping["production"]], mode="markers", name="Possible Outlier"))
        fig_anom.update_layout(template="plotly_dark", title="Production Outlier Screening")
        st.plotly_chart(fig_anom, use_container_width=True)

    st.markdown("---")
    st.subheader("Missing Values Summary")
    st.dataframe(missing_values_summary(df), use_container_width=True)

    corr = correlation_matrix_safe(df, mapping)
    if corr is not None:
        st.markdown("---")
        st.subheader("Correlation Matrix")
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorscale="Blues"
        ))
        fig_corr.update_layout(template="plotly_dark", title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("AI Engine")

    insights, recs, well_class, risks, health_score = ai_summary(df, mapping)

    if mapping["production"]:
        x_num, _ = time_to_numeric(df[mapping["time"]])
        q = pd.to_numeric(df[mapping["production"]], errors="coerce").values
        fit = fit_decline_models(x_num, q)

        if fit is None:
            st.warning("Not enough valid production data for decline analysis.")
        else:
            best_decline_name, results = fit
            st.success(f"Best decline model: {best_decline_name}")

            hist_mask = np.isfinite(x_num) & np.isfinite(q) & (q > 0)
            time_hist = np.array(df.loc[hist_mask, mapping["time"]])
            q_hist = q[hist_mask]

            fig_decline = go.Figure()
            fig_decline.add_trace(go.Scatter(x=time_hist, y=q_hist, mode="lines+markers", name="Actual"))
            for model_name, info in results.items():
                fig_decline.add_trace(go.Scatter(x=time_hist, y=info["qhat"], mode="lines", name=f"{model_name} Fit"))
            fig_decline.update_layout(template="plotly_dark", title="Decline Model Comparison")
            st.plotly_chart(fig_decline, use_container_width=True)

            metrics_df = pd.DataFrame({
                "Model": list(results.keys()),
                "RMSE": [results[k]["rmse"] for k in results.keys()]
            }).sort_values("RMSE")
            st.dataframe(metrics_df, use_container_width=True)

            if best_decline_name == "Exponential":
                decline_params_text = f"D = {results['Exponential']['D']:.5f}"
            elif best_decline_name == "Harmonic":
                decline_params_text = f"D = {results['Harmonic']['D']:.5f}"
            else:
                decline_params_text = f"D = {results['Hyperbolic']['D']:.5f}, b = {results['Hyperbolic']['b']:.2f}"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Water Risk</div><div class="metric-value">{risks["water_risk"]}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Gas Risk</div><div class="metric-value">{risks["gas_risk"]}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Depletion Risk</div><div class="metric-value">{risks["depletion_risk"]}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Reservoir Health</div><div class="metric-value">{health_score}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

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
    <h3 style="margin-top:0;">AI Summary</h3>
    <p style="margin:0;color:#d8e4ff;">Rows analyzed: <b>{len(df)}</b></p>
    <p style="margin:0;color:#d8e4ff;">Anomalies detected: <b>{outlier_count}</b></p>
    <p style="margin:0;color:#d8e4ff;">Best decline model: <b>{best_decline_name if best_decline_name else "-"}</b></p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Predictive AI")

    pred_results = predictive_ai(df, mapping)
    labels = {
        "production": "Production",
        "pressure": "Pressure",
        "water_cut": "Water Cut",
        "gor": "GOR",
    }

    if pred_results:
        pred_choice = st.selectbox("Select variable for predictive AI", list(pred_results.keys()))
        pr = pred_results[pred_choice]

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=pr["historical_x"], y=pr["historical_y"], mode="lines+markers", name="Historical"))
        fig_pred.add_trace(go.Scatter(x=pr["future_x"], y=pr["pred"], mode="lines+markers", name="Predicted"))
        fig_pred.update_layout(template="plotly_dark", title=f"{labels[pred_choice]} Predictive AI Outlook")
        st.plotly_chart(fig_pred, use_container_width=True)

        st.write(f"Predicted trend slope: {pr['slope']:.3f}")

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.subheader("AI Insights")
        for item in insights:
            st.write("• " + item)
    with right:
        st.subheader("Recommended Actions")
        for item in recs:
            st.write("• " + item)

    st.markdown('</div>', unsafe_allow_html=True)

with tab6:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Reports")

    report = generate_ai_report(
        df=df,
        mapping=mapping,
        selected_well=selected_well,
        insights=insights,
        recs=recs,
        best_decline_name=best_decline_name,
        well_class=well_class,
        risks=risks,
        health_score=health_score,
        decline_params_text=decline_params_text
    )

    st.download_button(
        "Download AI engineering report (.txt)",
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
    Developed by Abbas • Petroleum Engineering • Fixed Phase 2 AI Platform
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
