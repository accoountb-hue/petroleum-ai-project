import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(
    page_title="Petroleum Data Analysis with AI",
    page_icon="🛢️",
    layout="wide",
)

# ---------------- THEME + READABILITY FIX ----------------
st.markdown("""
<style>
:root{
    --text:#ffffff;
    --muted:#e6ecff;
    --soft:#c8d5f0;
    --border:rgba(255,255,255,.12);
}

/* Dark readable background */
[data-testid="stAppViewContainer"]{
    background:
      linear-gradient(rgba(4,10,20,.90), rgba(4,10,20,.94)),
      url("https://images.unsplash.com/photo-1513828583688-c52646db42da?q=80&w=1800&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.block-container{
    max-width: 1420px;
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
}

/* Force text readability */
html, body,
p, li, span, div, label,
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
.stAlert,
.stCaption{
    color:#ffffff !important;
    opacity:1 !important;
}

.main-title{
    font-size: 3rem;
    font-weight: 800;
    color: white;
    line-height: 1.1;
    margin-bottom: .45rem;
}
.sub-title{
    color: var(--muted);
    font-size: 1.08rem;
    line-height: 1.85;
}
.hero{
    background: linear-gradient(135deg, rgba(10,18,35,.93), rgba(12,19,36,.91));
    border: 1px solid var(--border);
    border-radius: 26px;
    padding: 1.6rem;
    box-shadow: 0 18px 42px rgba(0,0,0,.24);
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

.section-card,
.metric-card,
.info-box,
.ai-box,
.rank-box,
.footer-box,
.ai-insight-box{
    background: rgba(8, 16, 30, 0.90) !important;
    border: 1px solid rgba(255,255,255,.14) !important;
    backdrop-filter: blur(8px);
}

.section-card{
    margin-top: 1rem;
    border-radius: 22px;
    padding: 1.15rem;
    box-shadow: 0 12px 34px rgba(0,0,0,.18);
}
.metric-card{
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

.ai-box, .rank-box, .footer-box{
    border-radius:18px;
    padding:1rem;
}
.footer-box{
    margin-top:1rem;
    text-align:center;
}

.ai-insight-box{
    border-radius:18px;
    padding:20px;
    margin-top:10px;
}

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
.stSelectbox svg{
    fill:#111827 !important;
}
div[role="radiogroup"] label{
    color:#ffffff !important;
}

[data-testid="stDataFrame"]{
    background: rgba(255,255,255,0.96) !important;
    border-radius: 12px;
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
    "field": ["field", "field_name", "asset"],
    "production": ["production", "prod", "oil_rate", "qo", "q_oil", "liquid_rate", "rate", "production_rate"],
    "pressure": ["pressure", "pres", "bhp", "reservoir_pressure", "p_res"],
    "water_cut": ["watercut", "water_cut", "wc", "bsw"],
    "gor": ["gor", "gas_oil_ratio", "g_o_r", "gasoilratio"]
}

# ---------------- DATA HELPERS ----------------
def demo_dataset():
    np.random.seed(42)
    days = np.arange(1, 181)
    wells = ["Well_A", "Well_B", "Well_C", "Well_D", "Well_E", "Well_F"]
    fields = {
        "Well_A": "North_Field",
        "Well_B": "North_Field",
        "Well_C": "North_Field",
        "Well_D": "South_Field",
        "Well_E": "South_Field",
        "Well_F": "South_Field",
    }
    rows = []

    for i, well in enumerate(wells):
        base_q = 1700 - i * 170
        base_p = 3850 - i * 90
        wc = 0.05 + i * 0.018
        gor = 390 + i * 22

        for d in days:
            q = base_q * np.exp(-0.0095 * d) + np.random.normal(0, 16)
            p = base_p - 4.2 * d + np.random.normal(0, 6)
            water = wc + 0.00155 * d + np.random.normal(0, 0.002)
            gas = gor + 1.5 * d + np.random.normal(0, 5)

            rows.append([
                fields[well],
                well,
                d,
                max(q, 15),
                max(p, 800),
                max(min(water, 0.95), 0),
                max(gas, 0)
            ])

    df = pd.DataFrame(rows, columns=["Field", "Well", "Day", "Production", "Pressure", "WaterCut", "GOR"])

    # anomalies
    df.loc[40, "Production"] *= 1.8
    df.loc[220, "Production"] *= 0.45
    df.loc[510, "Pressure"] *= 0.87
    df.loc[777, "GOR"] *= 1.22
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
        "field": None,
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

def rank_fields(df, field_col, production_col):
    grp = df.groupby(field_col)[production_col].agg(["mean", "max", "min", "count"]).reset_index()
    grp.columns = [field_col, "Average_Production", "Max_Production", "Min_Production", "Records"]
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

# ---------------- DECLINE / FORECAST ----------------
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

def calculate_eur(df, mapping, decline_results, best_decline_name):
    prod_col = mapping.get("production")
    time_col = mapping.get("time")
    if not prod_col or not time_col or decline_results is None:
        return None

    x_num, _ = time_to_numeric(df[time_col])
    q = pd.to_numeric(df[prod_col], errors="coerce").values
    mask = np.isfinite(x_num) & np.isfinite(q) & (q > 0)
    t = x_num[mask]
    q = q[mask]

    if len(q) < 5:
        return None

    q0 = q[0]
    current_cum = np.sum(q)

    future_t = np.arange(t[-1] + 1, t[-1] + 365 + 1, dtype=float)

    if best_decline_name == "Exponential":
        D = decline_results["Exponential"]["D"]
        pred = q0 * np.exp(-D * (future_t - t[0]))
    elif best_decline_name == "Harmonic":
        D = decline_results["Harmonic"]["D"]
        pred = q0 / (1 + D * (future_t - t[0]))
    else:
        b = decline_results["Hyperbolic"]["b"]
        D = decline_results["Hyperbolic"]["D"]
        pred = q0 / np.power(1 + b * D * (future_t - t[0]), 1.0 / b)

    eur = current_cum + np.sum(np.maximum(pred, 0))
    return round(float(eur), 2)

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

        if len(x_clean) < 10:
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

# ---------------- RESERVOIR AI ----------------
def estimate_drive_mechanism(df, mapping):
    prod_col = mapping.get("production")
    press_col = mapping.get("pressure")
    wc_col = mapping.get("water_cut")
    gor_col = mapping.get("gor")

    if not prod_col or not press_col:
        return "Insufficient data"

    prod = pd.to_numeric(df[prod_col], errors="coerce").dropna()
    press = pd.to_numeric(df[press_col], errors="coerce").dropna()
    wc = pd.to_numeric(df[wc_col], errors="coerce").dropna() if wc_col else pd.Series(dtype=float)
    gor = pd.to_numeric(df[gor_col], errors="coerce").dropna() if gor_col else pd.Series(dtype=float)

    if len(prod) < 5 or len(press) < 5:
        return "Insufficient data"

    press_drop_pct = ((press.iloc[0] - press.iloc[-1]) / press.iloc[0]) * 100 if press.iloc[0] != 0 else 0
    wc_rise = (wc.iloc[-1] - wc.iloc[0]) if len(wc) >= 5 else 0
    gor_rise = (gor.iloc[-1] - gor.iloc[0]) if len(gor) >= 5 else 0

    if wc_rise > 0.05 and press_drop_pct < 12:
        return "Water Drive"
    if gor_rise > 45 and press_drop_pct < 18:
        return "Gas Cap Drive"
    if press_drop_pct > 20 and gor_rise > 25:
        return "Solution Gas Drive"
    return "Mixed Drive"

def reservoir_diagnostics(df, mapping):
    findings = []

    prod_col = mapping.get("production")
    press_col = mapping.get("pressure")
    wc_col = mapping.get("water_cut")
    gor_col = mapping.get("gor")

    prod = pd.to_numeric(df[prod_col], errors="coerce").dropna() if prod_col else pd.Series(dtype=float)
    press = pd.to_numeric(df[press_col], errors="coerce").dropna() if press_col else pd.Series(dtype=float)
    wc = pd.to_numeric(df[wc_col], errors="coerce").dropna() if wc_col else pd.Series(dtype=float)
    gor = pd.to_numeric(df[gor_col], errors="coerce").dropna() if gor_col else pd.Series(dtype=float)

    if len(prod) >= 5 and len(wc) >= 5:
        if prod.iloc[-1] < prod.iloc[0] and (wc.iloc[-1] - wc.iloc[0]) > 0.03:
            findings.append("Possible water breakthrough detected.")
            if wc.iloc[-1] > 0.45:
                findings.append("Water trend may be consistent with edge-water support or water coning.")

    if len(prod) >= 5 and len(gor) >= 5:
        if prod.iloc[-1] < prod.iloc[0] and (gor.iloc[-1] - gor.iloc[0]) > 35:
            findings.append("Possible gas breakthrough detected.")
            if gor.iloc[-1] > gor.mean() * 1.15:
                findings.append("Gas behavior may indicate gas-cap expansion or gas coning.")

    if len(prod) >= 5 and len(press) >= 5:
        if prod.iloc[-1] < prod.iloc[0] and press.iloc[-1] < press.iloc[0]:
            findings.append("Production decline is consistent with reservoir pressure depletion.")

    if not findings:
        findings.append("No dominant reservoir diagnostic pattern detected from current data.")

    return findings

# ---------------- PRODUCTION ENGINEERING AI ----------------
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

def calculate_production_efficiency_score(df, mapping):
    prod_col = mapping.get("production")
    wc_col = mapping.get("water_cut")
    press_col = mapping.get("pressure")
    gor_col = mapping.get("gor")

    if not prod_col:
        return 0.0

    prod = pd.to_numeric(df[prod_col], errors="coerce").dropna()
    if len(prod) < 5:
        return 0.0

    current_prod = prod.iloc[-1]
    peak_prod = prod.max()
    production_factor = (current_prod / peak_prod) * 100 if peak_prod > 0 else 0

    wc_penalty = 0
    if wc_col:
        wc = pd.to_numeric(df[wc_col], errors="coerce").dropna()
        if len(wc) >= 5:
            wc_penalty = min(wc.iloc[-1] * 35, 35)

    press_penalty = 0
    if press_col:
        press = pd.to_numeric(df[press_col], errors="coerce").dropna()
        if len(press) >= 5 and press.iloc[0] > 0:
            drop_pct = ((press.iloc[0] - press.iloc[-1]) / press.iloc[0]) * 100
            press_penalty = min(max(drop_pct * 0.6, 0), 25)

    gor_penalty = 0
    if gor_col:
        gor = pd.to_numeric(df[gor_col], errors="coerce").dropna()
        if len(gor) >= 5:
            gor_increase = max(gor.iloc[-1] - gor.iloc[0], 0)
            gor_penalty = min(gor_increase * 0.05, 15)

    score = production_factor - wc_penalty - press_penalty - gor_penalty
    return round(max(min(score, 100), 0), 1)

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

def detect_underperforming_well(df, mapping):
    well_col = mapping.get("well")
    prod_col = mapping.get("production")
    if not well_col or not prod_col:
        return None, None

    ranking = rank_wells(df, well_col, prod_col)
    if len(ranking) == 0:
        return None, None

    worst_well = ranking.iloc[-1][well_col]
    worst_avg = ranking.iloc[-1]["Average_Production"]
    return worst_well, worst_avg

def estimate_production_loss(df, mapping):
    prod_col = mapping.get("production")
    if not prod_col:
        return None

    prod = pd.to_numeric(df[prod_col], errors="coerce").dropna()
    if len(prod) < 10:
        return None

    potential = float(prod.quantile(0.90))
    actual = float(prod.iloc[-1])
    loss = max(potential - actual, 0)
    return {
        "potential": round(potential, 2),
        "actual": round(actual, 2),
        "loss": round(loss, 2)
    }

def well_problem_detection(df, mapping, risks):
    problems = []

    prod_col = mapping.get("production")
    press_col = mapping.get("pressure")
    wc_col = mapping.get("water_cut")
    gor_col = mapping.get("gor")

    prod = pd.to_numeric(df[prod_col], errors="coerce").dropna() if prod_col else pd.Series(dtype=float)
    press = pd.to_numeric(df[press_col], errors="coerce").dropna() if press_col else pd.Series(dtype=float)

    if len(prod) >= 5 and len(press) >= 5:
        if prod.iloc[-1] < prod.iloc[0] and abs(press.iloc[-1] - press.iloc[0]) < 10:
            problems.append("Possible tubing, choke, scale, or near-wellbore restriction.")

    if risks["water_risk"] > 40:
        problems.append("Possible water coning or water breakthrough.")
    if risks["gas_risk"] > 40:
        problems.append("Possible gas coning or gas breakthrough.")
    if risks["anomaly_risk"] > 30:
        problems.append("Potential sensor issue or unstable operating condition.")
    if len(prod) >= 5 and len(press) >= 5:
        if prod.iloc[-1] < prod.iloc[0] and press.iloc[-1] < press.iloc[0]:
            problems.append("Reservoir support may be weakening under current production conditions.")

    if not problems:
        problems.append("No dominant well problem detected from current data.")
    return problems

def artificial_lift_suggestions(df, mapping, risks):
    suggestions = []

    prod_col = mapping.get("production")
    press_col = mapping.get("pressure")
    wc_col = mapping.get("water_cut")

    prod = pd.to_numeric(df[prod_col], errors="coerce").dropna() if prod_col else pd.Series(dtype=float)
    press = pd.to_numeric(df[press_col], errors="coerce").dropna() if press_col else pd.Series(dtype=float)
    wc = pd.to_numeric(df[wc_col], errors="coerce").dropna() if wc_col else pd.Series(dtype=float)

    if len(prod) >= 5 and len(press) >= 5:
        if prod.iloc[-1] < prod.iloc[0] and press.iloc[-1] < press.iloc[0]:
            suggestions.append("Consider artificial lift optimization such as gas lift or ESP review to sustain production.")

    if len(wc) >= 5 and wc.iloc[-1] > 0.35:
        suggestions.append("High water cut may reduce lift efficiency; review lift design and produced-water handling.")

    if not suggestions:
        suggestions.append("Current data does not strongly require artificial lift intervention, but continue monitoring.")
    return suggestions

def workover_recommendations(df, mapping, risks):
    recs = []

    if risks["water_risk"] > 40:
        recs.append("Evaluate water shutoff or coning control workover.")
    if risks["gas_risk"] > 40:
        recs.append("Review gas control strategy and completion behavior.")
    if risks["depletion_risk"] > 40:
        recs.append("Consider reservoir management actions before aggressive workover.")
    if risks["anomaly_risk"] > 30:
        recs.append("Inspect instrumentation and production system stability before intervention.")

    prod_col = mapping.get("production")
    press_col = mapping.get("pressure")
    if prod_col and press_col:
        prod = pd.to_numeric(df[prod_col], errors="coerce").dropna()
        press = pd.to_numeric(df[press_col], errors="coerce").dropna()
        if len(prod) >= 5 and len(press) >= 5:
            if prod.iloc[-1] < prod.iloc[0] and abs(press.iloc[-1] - press.iloc[0]) < 10:
                recs.append("Consider stimulation, scale removal, or tubing/choke inspection.")

    if not recs:
        recs.append("No strong workover recommendation at this stage; continue surveillance.")
    return recs

def production_optimization_recommendations(df, mapping, risks):
    recs = []

    if risks["water_risk"] > 40:
        recs.append("Investigate water breakthrough and evaluate water shutoff or coning control options.")
    if risks["gas_risk"] > 40:
        recs.append("Review gas handling, flowing conditions, and gas breakthrough behavior.")
    if risks["depletion_risk"] > 40:
        recs.append("Assess pressure support strategy and consider reservoir-management actions.")
    if risks["anomaly_risk"] > 30:
        recs.append("Validate sensor quality and investigate operational disturbances behind abnormal production points.")

    prod_col = mapping.get("production")
    press_col = mapping.get("pressure")
    if prod_col and press_col:
        prod = pd.to_numeric(df[prod_col], errors="coerce").dropna()
        press = pd.to_numeric(df[press_col], errors="coerce").dropna()
        if len(prod) >= 5 and len(press) >= 5:
            if prod.iloc[-1] < prod.iloc[0] and abs(press.iloc[-1] - press.iloc[0]) < 10:
                recs.append("Production decline with near-stable pressure may indicate choke or near-wellbore flow restriction.")

    if not recs:
        recs.append("Maintain current operating strategy and continue monitoring.")
    return recs

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
    efficiency_score = calculate_production_efficiency_score(df, mapping)
    well_class = classify_well(df, mapping, risks)
    drive_mech = estimate_drive_mechanism(df, mapping)

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

    insights.extend(reservoir_diagnostics(df, mapping))
    insights.append(f"Estimated drive mechanism: {drive_mech}.")
    insights.append(f"AI well classification: {well_class}.")
    insights.append(f"Reservoir health score: {health_score}/100.")
    insights.append(f"Production efficiency score: {efficiency_score}/100.")
    insights.append(
        f"Risk scores → Water: {risks['water_risk']}, Gas: {risks['gas_risk']}, Depletion: {risks['depletion_risk']}, Data quality: {risks['anomaly_risk']}."
    )

    prod_opt = production_optimization_recommendations(df, mapping, risks)
    workover = workover_recommendations(df, mapping, risks)
    lift_recs = artificial_lift_suggestions(df, mapping, risks)
    well_probs = well_problem_detection(df, mapping, risks)

    for item in prod_opt + workover:
        if item not in recs:
            recs.append(item)

    return {
        "insights": insights,
        "recommendations": recs,
        "well_class": well_class,
        "risks": risks,
        "health_score": health_score,
        "efficiency_score": efficiency_score,
        "lift_recs": lift_recs,
        "well_problems": well_probs,
        "drive_mechanism": drive_mech
    }

# ---------------- ML AI ----------------
def prepare_ml_dataset(df, mapping, target_key="production"):
    time_col = mapping.get("time")
    target_col = mapping.get(target_key)
    if not time_col or not target_col:
        return None, None, None

    feature_cols = []
    for key in ["pressure", "water_cut", "gor"]:
        col = mapping.get(key)
        if col and col != target_col:
            feature_cols.append(col)

    work = df.copy()
    x_time, _ = time_to_numeric(work[time_col])
    work["_time_numeric"] = x_time

    use_cols = ["_time_numeric"] + feature_cols + [target_col]
    work = work[use_cols].apply(pd.to_numeric, errors="coerce").dropna()

    if len(work) < 30:
        return None, None, None

    X = work.drop(columns=[target_col])
    y = work[target_col]
    return X, y, list(X.columns)

def train_ml_models(df, mapping, target_key="production"):
    X, y, feature_names = prepare_ml_dataset(df, mapping, target_key)
    if X is None:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42),
    }

    results = []
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))
        results.append({"Model": name, "RMSE": rmse, "R2": r2})
        trained[name] = model

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained[best_model_name]

    last_row = X.iloc[-1].copy()
    future_rows = []
    for i in range(1, 31):
        new_row = last_row.copy()
        new_row["_time_numeric"] = last_row["_time_numeric"] + i
        future_rows.append(new_row)
    future_X = pd.DataFrame(future_rows)
    future_pred = best_model.predict(future_X)

    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "future_pred": future_pred,
        "feature_names": feature_names,
        "historical_y": y.values,
        "historical_idx": np.arange(len(y)),
        "future_idx": np.arange(len(y), len(y) + len(future_pred))
    }

# ---------------- REPORT / PDF ----------------
def generate_ai_report_text(df, mapping, selected_well, ai_pack, best_decline_name, decline_params_text, eur_value, ml_info):
    lines = []
    lines.append("AI PETROLEUM ENGINEERING PLATFORM REPORT")
    lines.append("=" * 72)
    lines.append(f"Rows analyzed: {len(df)}")
    if selected_well is not None:
        lines.append(f"Selected well: {selected_well}")
    lines.append(f"AI well classification: {ai_pack['well_class']}")
    lines.append(f"Estimated drive mechanism: {ai_pack['drive_mechanism']}")
    lines.append(f"Reservoir health score: {ai_pack['health_score']}/100")
    lines.append(f"Production efficiency score: {ai_pack['efficiency_score']}/100")
    if best_decline_name:
        lines.append(f"Best decline model: {best_decline_name}")
    if decline_params_text:
        lines.append(f"Decline parameters: {decline_params_text}")
    if eur_value is not None:
        lines.append(f"Estimated EUR: {eur_value}")
    if ml_info is not None:
        lines.append(f"Best ML model: {ml_info['best_model_name']}")
    lines.append("")

    lines.append("Mapped Columns:")
    for k, v in mapping.items():
        if v:
            lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("Risk Scores:")
    lines.append(f"- Water breakthrough risk: {ai_pack['risks']['water_risk']}/100")
    lines.append(f"- Gas breakthrough risk: {ai_pack['risks']['gas_risk']}/100")
    lines.append(f"- Pressure depletion risk: {ai_pack['risks']['depletion_risk']}/100")
    lines.append(f"- Data quality / anomaly risk: {ai_pack['risks']['anomaly_risk']}/100")

    lines.append("")
    lines.append("AI Insights:")
    for item in ai_pack["insights"]:
        lines.append(f"* {item}")

    lines.append("")
    lines.append("Optimization / Workover Recommendations:")
    for item in ai_pack["recommendations"]:
        lines.append(f"* {item}")

    lines.append("")
    lines.append("Artificial Lift Suggestions:")
    for item in ai_pack["lift_recs"]:
        lines.append(f"* {item}")

    return "\n".join(lines)

def create_pdf_report(ai_pack, best_decline_name, decline_params_text, eur_value, ml_info):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp.name, pagesize=letter)
    width, height = letter
    y = height - 50

    def draw_line(text, font="Helvetica", size=10, step=15):
        nonlocal y
        if y < 70:
            c.showPage()
            y = height - 50
        c.setFont(font, size)
        c.drawString(50, y, text[:110])
        y -= step

    draw_line("Petroleum AI Engineering Report", font="Helvetica-Bold", size=16, step=24)
    draw_line(f"AI Well Classification: {ai_pack['well_class']}", size=11)
    draw_line(f"Estimated Drive Mechanism: {ai_pack['drive_mechanism']}", size=11)
    draw_line(f"Reservoir Health Score: {ai_pack['health_score']}/100", size=11)
    draw_line(f"Production Efficiency Score: {ai_pack['efficiency_score']}/100", size=11)
    draw_line(f"Best Decline Model: {best_decline_name if best_decline_name else '-'}", size=11)
    draw_line(f"Decline Parameters: {decline_params_text if decline_params_text else '-'}", size=11)
    draw_line(f"EUR: {eur_value if eur_value is not None else '-'}", size=11)
    draw_line(f"Best ML Model: {ml_info['best_model_name'] if ml_info is not None else '-'}", size=11, step=22)

    draw_line("Risk Scores", font="Helvetica-Bold", size=12, step=18)
    for k, v in ai_pack["risks"].items():
        draw_line(f"- {k}: {v}")

    draw_line("AI Insights", font="Helvetica-Bold", size=12, step=18)
    for item in ai_pack["insights"][:8]:
        draw_line(f"- {item}")

    draw_line("Recommendations", font="Helvetica-Bold", size=12, step=18)
    for item in ai_pack["recommendations"][:8]:
        draw_line(f"- {item}")

    draw_line("Artificial Lift Suggestions", font="Helvetica-Bold", size=12, step=18)
    for item in ai_pack["lift_recs"][:5]:
        draw_line(f"- {item}")

    c.save()
    return temp.name

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
  <div class="badge">Version 13 • Integrated Petroleum AI Platform</div>
  <div class="main-title">Petroleum Data Analysis with AI</div>
  <div class="sub-title">
    Integrated one-file platform covering Advanced Reservoir AI, Smart Well Diagnostics,
    Field Optimization AI, Machine Learning AI, Professional Dashboard, and a Business-Level Demo layer.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-card">
  <h3 style="margin-top:0;">Integrated Modules</h3>
  <div class="info-grid">
    <div class="info-box">
      <h4>Reservoir + Production AI</h4>
      <p>Water breakthrough, gas breakthrough, drive mechanism estimation, workover logic, and optimization suggestions.</p>
    </div>
    <div class="info-box">
      <h4>Field + Well + ML</h4>
      <p>Field forecasting, best/worst well analysis, ML model comparison, and predictive analytics.</p>
    </div>
    <div class="info-box">
      <h4>Professional Output</h4>
      <p>Executive summary, professional charts, TXT and PDF reports, plus business-layer demo widgets.</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------- CONTROLS ----------------
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

# ---------------- LOAD DATA ----------------
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

# ---------------- KPI ROW ----------------
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

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "Data Preview",
    "Column Mapping",
    "Engineering Analytics",
    "AI Engine",
    "Machine Learning",
    "Reports & Platform Demo"
])

with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Integrated Roadmap Status")
    st.markdown("""
<div class="info-grid">
  <div class="info-box">
    <h4>Phase 5 + 6</h4>
    <p>Advanced Reservoir AI and Smart Well Diagnostics are integrated in the AI logic.</p>
  </div>
  <div class="info-box">
    <h4>Phase 7 + 8</h4>
    <p>Field Optimization AI and Machine Learning AI are included with forecasting and model comparison.</p>
  </div>
  <div class="info-box">
    <h4>Phase 9 + 10</h4>
    <p>Professional dashboard is real; business layer is included as a demo UI because real auth/database/API need external services.</p>
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
        field_opts = ["None"] + cols
        well_col = st.selectbox("Well column", well_opts, index=well_opts.index(det["well"]) if det["well"] in well_opts else 0)
        field_col = st.selectbox("Field column", field_opts, index=field_opts.index(det["field"]) if det["field"] in field_opts else 0)

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
        "field": None if field_col == "None" else field_col,
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

try:
    mapping
except NameError:
    cols = list(raw_df.columns)
    mapping = {
        "time": det["time"] or cols[0],
        "well": det["well"],
        "field": det["field"],
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
eur_value = None

with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Engineering Analytics")

    if mapping["production"]:
        fig_prod = go.Figure()
        if mapping["well"] and selected_well == "All":
            for well_name, sub in df.groupby(mapping["well"]):
                fig_prod.add_trace(go.Scatter(
                    x=sub[mapping["time"]],
                    y=sub[mapping["production"]],
                    mode="lines",
                    name=str(well_name),
                    hovertemplate="Time: %{x}<br>Production: %{y:.2f}<extra></extra>"
                ))
        else:
            fig_prod.add_trace(go.Scatter(
                x=df[mapping["time"]],
                y=df[mapping["production"]],
                mode="lines+markers",
                name="Production",
                hovertemplate="Time: %{x}<br>Production: %{y:.2f}<extra></extra>"
            ))
        fig_prod.update_layout(template="plotly_dark", title="Production vs Time", title_font_size=22, font=dict(size=14), hovermode="x unified")
        st.plotly_chart(fig_prod, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if mapping["pressure"]:
            fig_press = go.Figure()
            if mapping["well"] and selected_well == "All":
                for well_name, sub in df.groupby(mapping["well"]):
                    fig_press.add_trace(go.Scatter(
                        x=sub[mapping["time"]],
                        y=sub[mapping["pressure"]],
                        mode="lines",
                        name=str(well_name)
                    ))
            else:
                fig_press.add_trace(go.Scatter(
                    x=df[mapping["time"]],
                    y=df[mapping["pressure"]],
                    mode="lines+markers",
                    name="Pressure"
                ))
            fig_press.update_layout(template="plotly_dark", title="Pressure vs Time", title_font_size=22, font=dict(size=14), hovermode="x unified")
            st.plotly_chart(fig_press, use_container_width=True)

    with c2:
        if mapping["water_cut"]:
            fig_wc = go.Figure()
            if mapping["well"] and selected_well == "All":
                for well_name, sub in df.groupby(mapping["well"]):
                    fig_wc.add_trace(go.Scatter(
                        x=sub[mapping["time"]],
                        y=sub[mapping["water_cut"]],
                        mode="lines",
                        name=str(well_name)
                    ))
            else:
                fig_wc.add_trace(go.Scatter(
                    x=df[mapping["time"]],
                    y=df[mapping["water_cut"]],
                    mode="lines+markers",
                    name="Water Cut"
                ))
            fig_wc.update_layout(template="plotly_dark", title="Water Cut vs Time", title_font_size=22, font=dict(size=14), hovermode="x unified")
            st.plotly_chart(fig_wc, use_container_width=True)

    if mapping["gor"]:
        fig_gor = go.Figure()
        if mapping["well"] and selected_well == "All":
            for well_name, sub in df.groupby(mapping["well"]):
                fig_gor.add_trace(go.Scatter(
                    x=sub[mapping["time"]],
                    y=sub[mapping["gor"]],
                    mode="lines",
                    name=str(well_name)
                ))
        else:
            fig_gor.add_trace(go.Scatter(
                x=df[mapping["time"]],
                y=df[mapping["gor"]],
                mode="lines+markers",
                name="GOR"
            ))
        fig_gor.update_layout(template="plotly_dark", title="GOR vs Time", title_font_size=22, font=dict(size=14), hovermode="x unified")
        st.plotly_chart(fig_gor, use_container_width=True)

    if mapping["well"] and mapping["production"] and selected_well == "All":
        st.markdown("---")
        st.subheader("Well Performance Ranking")
        ranking_df = rank_wells(df, mapping["well"], mapping["production"])
        st.dataframe(ranking_df, use_container_width=True)

    if mapping["field"] and mapping["production"]:
        st.markdown("---")
        st.subheader("Field Level Analysis")
        field_rank_df = rank_fields(df, mapping["field"], mapping["production"])
        st.dataframe(field_rank_df, use_container_width=True)

        fig_field = go.Figure()
        fig_field.add_trace(go.Bar(x=field_rank_df[mapping["field"]], y=field_rank_df["Average_Production"]))
        fig_field.update_layout(template="plotly_dark", title="Field Production Comparison", title_font_size=22, font=dict(size=14))
        st.plotly_chart(fig_field, use_container_width=True)

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
        fig_anom.update_layout(template="plotly_dark", title="Production Outlier Screening", title_font_size=22, font=dict(size=14))
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
        fig_corr.update_layout(template="plotly_dark", title="Correlation Matrix", title_font_size=22, font=dict(size=14))
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("AI Engine")

    ai_pack = ai_summary(df, mapping)

    st.markdown("## Executive Summary")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Reservoir Health", f"{ai_pack['health_score']}/100")
    with s2:
        st.metric("Production Efficiency", f"{ai_pack['efficiency_score']}/100")
    with s3:
        st.metric("Well Classification", ai_pack["well_class"])
    with s4:
        highest_risk = max(ai_pack["risks"], key=ai_pack["risks"].get)
        st.metric("Highest Risk", highest_risk.replace("_", " ").title())

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
            fig_decline.update_layout(template="plotly_dark", title="Decline Model Comparison", title_font_size=22, font=dict(size=14), hovermode="x unified")
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

            eur_value = calculate_eur(df, mapping, results, best_decline_name)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Water Risk</div><div class="metric-value">{ai_pack["risks"]["water_risk"]}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Gas Risk</div><div class="metric-value">{ai_pack["risks"]["gas_risk"]}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Depletion Risk</div><div class="metric-value">{ai_pack["risks"]["depletion_risk"]}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Data Quality Risk</div><div class="metric-value">{ai_pack["risks"]["anomaly_risk"]}</div></div>', unsafe_allow_html=True)

    if eur_value is not None:
        st.markdown("---")
        st.markdown(f"""
<div class="ai-box">
    <h3 style="margin-top:0;">Estimated Ultimate Recovery (EUR)</h3>
    <p><b>{eur_value}</b></p>
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
        fig_pred.update_layout(template="plotly_dark", title=f"{labels[pred_choice]} Predictive AI Outlook", title_font_size=22, font=dict(size=14), hovermode="x unified")
        st.plotly_chart(fig_pred, use_container_width=True)
        st.write(f"Predicted trend slope: {pr['slope']:.3f}")

    st.markdown("---")
    worst_well, worst_avg = detect_underperforming_well(df, mapping)
    loss_pack = estimate_production_loss(df, mapping)
    p1, p2 = st.columns(2)

    with p1:
        if worst_well is not None:
            st.markdown(f"""
<div class="ai-box">
    <h3 style="margin-top:0;">Underperforming Well</h3>
    <p>Detected well: <b>{worst_well}</b></p>
    <p>Average production: <b>{worst_avg:.2f}</b></p>
</div>
""", unsafe_allow_html=True)

    with p2:
        if loss_pack is not None:
            st.markdown(f"""
<div class="ai-box">
    <h3 style="margin-top:0;">Production Loss Estimation</h3>
    <p>Potential production: <b>{loss_pack['potential']}</b></p>
    <p>Actual production: <b>{loss_pack['actual']}</b></p>
    <p>Estimated loss: <b>{loss_pack['loss']}</b></p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="ai-insight-box">', unsafe_allow_html=True)
        st.markdown("### AI Insights")
        for item in ai_pack["insights"]:
            st.markdown(f"<p style='color:#ffffff !important; font-size:18px; font-weight:600; line-height:1.9; margin-bottom:10px;'>• {item}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="ai-insight-box">', unsafe_allow_html=True)
        st.markdown("### Optimization Recommendations")
        for item in ai_pack["recommendations"]:
            st.markdown(f"<p style='color:#ffffff !important; font-size:18px; font-weight:600; line-height:1.9; margin-bottom:10px;'>• {item}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown('<div class="ai-insight-box">', unsafe_allow_html=True)
        st.markdown("### Well Problem Detection")
        for item in ai_pack["well_problems"]:
            st.markdown(f"<p style='color:#ffffff !important; font-size:18px; font-weight:600; line-height:1.9; margin-bottom:10px;'>• {item}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="ai-insight-box">', unsafe_allow_html=True)
        st.markdown("### Artificial Lift Suggestions")
        for item in ai_pack["lift_recs"]:
            st.markdown(f"<p style='color:#ffffff !important; font-size:18px; font-weight:600; line-height:1.9; margin-bottom:10px;'>• {item}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab6:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Machine Learning AI")

    target_choice = st.selectbox(
        "Choose target for ML models",
        ["production", "pressure", "water_cut"]
    )

    ml_info = train_ml_models(df, mapping, target_choice)

    if ml_info is None:
        st.warning("Not enough clean data to train ML models.")
    else:
        st.success(f"Best ML model: {ml_info['best_model_name']}")
        st.dataframe(ml_info["results_df"], use_container_width=True)

        fig_ml = go.Figure()
        fig_ml.add_trace(go.Scatter(
            x=ml_info["historical_idx"],
            y=ml_info["historical_y"],
            mode="lines+markers",
            name=f"Historical {target_choice}"
        ))
        fig_ml.add_trace(go.Scatter(
            x=ml_info["future_idx"],
            y=ml_info["future_pred"],
            mode="lines+markers",
            name=f"Predicted {target_choice}"
        ))
        fig_ml.update_layout(
            template="plotly_dark",
            title=f"Machine Learning Forecast — {target_choice}",
            title_font_size=22,
            font=dict(size=14),
            hovermode="x unified"
        )
        st.plotly_chart(fig_ml, use_container_width=True)

        st.markdown('<div class="ai-insight-box">', unsafe_allow_html=True)
        st.markdown("### ML Interpretation")
        st.markdown(f"<p style='color:#ffffff !important; font-size:18px; font-weight:600; line-height:1.9;'>• Best-performing model for this dataset: {ml_info['best_model_name']}.</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#ffffff !important; font-size:18px; font-weight:600; line-height:1.9;'>• Features used: {', '.join(ml_info['feature_names'])}.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab7:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Reports & Platform Demo")

    ai_pack = ai_summary(df, mapping)
    ml_info_for_report = train_ml_models(df, mapping, "production")

    report_text = generate_ai_report_text(
        df=df,
        mapping=mapping,
        selected_well=selected_well,
        ai_pack=ai_pack,
        best_decline_name=best_decline_name,
        decline_params_text=decline_params_text,
        eur_value=eur_value,
        ml_info=ml_info_for_report
    )

    st.download_button(
        "Download AI engineering report (.txt)",
        data=report_text.encode("utf-8"),
        file_name="petroleum_ai_report.txt",
        mime="text/plain"
    )

    st.download_button(
        "Download current filtered data (.csv)",
        data=df.to_csv(index=False),
        file_name="filtered_petroleum_data.csv",
        mime="text/csv"
    )

    st.markdown("### Download Professional PDF Report")
    if st.button("Generate PDF Report"):
        pdf_path = create_pdf_report(
            ai_pack=ai_pack,
            best_decline_name=best_decline_name,
            decline_params_text=decline_params_text,
            eur_value=eur_value,
            ml_info=ml_info_for_report
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF",
                f.read(),
                file_name="petroleum_ai_report.pdf",
                mime="application/pdf"
            )

    st.markdown("---")
    st.subheader("Business-Level Demo Layer")
    st.info("This section is a demo UI only. Real accounts, cloud DB, and API need external services.")

    b1, b2 = st.columns(2)

    with b1:
        st.markdown("""
<div class="ai-box">
    <h3 style="margin-top:0;">User Accounts (Demo)</h3>
    <p>Plan: Admin / Engineer / Viewer</p>
    <p>Current mode: Demo only</p>
</div>
""", unsafe_allow_html=True)

    with b2:
        api_payload = {
            "platform": "Petroleum AI Platform",
            "selected_well": selected_well,
            "well_class": ai_pack["well_class"],
            "health_score": ai_pack["health_score"],
            "risks": ai_pack["risks"],
            "best_decline_model": best_decline_name,
        }
        st.markdown("### API Payload Preview")
        st.code(json.dumps(api_payload, indent=2), language="json")

    st.markdown("""
<div class="footer-box">
    Developed by Abbas • Petroleum Engineering • Integrated Petroleum AI Platform
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
