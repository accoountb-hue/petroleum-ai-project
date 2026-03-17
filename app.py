import io
import json
import os
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(
    page_title="PetroScope",
    page_icon="🛢️",
    layout="wide",
)

# =========================================================
# STORAGE / AUTH
# =========================================================
DATA_DIR = "storage"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
os.makedirs(DATA_DIR, exist_ok=True)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users: dict) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def safe_name(name: str) -> str:
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in ("_", "-", " "):
            keep.append(ch)
    cleaned = "".join(keep).strip().replace(" ", "_")
    return cleaned[:80] if cleaned else "project"


def user_folder(username: str) -> str:
    path = os.path.join(DATA_DIR, safe_name(username))
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "projects"), exist_ok=True)
    os.makedirs(os.path.join(path, "uploads"), exist_ok=True)
    return path


def user_projects_dir(username: str) -> str:
    path = os.path.join(user_folder(username), "projects")
    os.makedirs(path, exist_ok=True)
    return path


def user_uploads_dir(username: str) -> str:
    path = os.path.join(user_folder(username), "uploads")
    os.makedirs(path, exist_ok=True)
    return path


def save_uploaded_file_for_user(username: str, uploaded_file) -> str:
    filename = safe_name(uploaded_file.name)
    save_path = os.path.join(user_uploads_dir(username), filename)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def list_user_files(username: str) -> list:
    folder = user_uploads_dir(username)
    files = []
    for f in os.listdir(folder):
        full = os.path.join(folder, f)
        if os.path.isfile(full):
            files.append({
                "name": f,
                "path": full,
                "modified": datetime.fromtimestamp(os.path.getmtime(full)).strftime("%Y-%m-%d %H:%M"),
                "size_kb": round(os.path.getsize(full) / 1024, 1)
            })
    files.sort(key=lambda x: x["modified"], reverse=True)
    return files


def save_project(username: str, project_name: str, payload: dict) -> str:
    project_name = safe_name(project_name)
    payload["project_name"] = project_name
    payload["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(user_projects_dir(username), f"{project_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def list_projects(username: str) -> list:
    folder = user_projects_dir(username)
    projects = []
    for f in os.listdir(folder):
        if f.endswith(".json"):
            full = os.path.join(folder, f)
            try:
                with open(full, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                projects.append({
                    "name": data.get("project_name", f.replace(".json", "")),
                    "saved_at": data.get("saved_at", ""),
                    "path": full
                })
            except Exception:
                projects.append({
                    "name": f.replace(".json", ""),
                    "saved_at": "",
                    "path": full
                })
    projects.sort(key=lambda x: x["saved_at"], reverse=True)
    return projects


def load_project(username: str, project_name: str):
    path = os.path.join(user_projects_dir(username), f"{safe_name(project_name)}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_project(username: str, project_name: str) -> None:
    path = os.path.join(user_projects_dir(username), f"{safe_name(project_name)}.json")
    if os.path.exists(path):
        os.remove(path)


# =========================================================
# SESSION
# =========================================================
if "user" not in st.session_state:
    st.session_state.user = None

if "loaded_project_name" not in st.session_state:
    st.session_state.loaded_project_name = None

if "loaded_project_data" not in st.session_state:
    st.session_state.loaded_project_data = None

if "source_mode" not in st.session_state:
    st.session_state.source_mode = "Use demo dataset"


# =========================================================
# UI
# =========================================================
st.markdown("""
<style>
:root{
    --bg:#0b1220;
    --panel:#121b2d;
    --panel-soft:#172236;
    --border:#25324a;
    --text:#f4f7fb;
    --muted:#b7c4d8;
    --accent:#4f8cff;
}

[data-testid="stAppViewContainer"]{
    background: linear-gradient(180deg, #0b1220 0%, #0e1628 100%);
}

.block-container{
    max-width: 1380px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

html, body,
h1, h2, h3, h4, h5, h6,
label,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"]{
    color: var(--text) !important;
    opacity: 1 !important;
}

.app-shell{
    border:1px solid var(--border);
    background: linear-gradient(180deg, rgba(18,27,45,.96), rgba(18,27,45,.94));
    border-radius:20px;
    padding:20px 22px;
}

.hero{
    border:1px solid var(--border);
    background: linear-gradient(180deg, rgba(18,27,45,.98), rgba(23,34,54,.96));
    border-radius:18px;
    padding:20px 22px;
}

.hero-top{
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    gap:16px;
    flex-wrap:wrap;
}

.badge{
    display:inline-block;
    font-size:12px;
    padding:6px 10px;
    border-radius:999px;
    background:#18243a;
    border:1px solid var(--border);
    color:#d9e4f5;
}

.title{
    font-size:2.2rem;
    font-weight:800;
    line-height:1.1;
    margin:10px 0 8px 0;
    color:white;
}

.subtitle{
    color:var(--muted) !important;
    font-size:1rem;
    line-height:1.7;
    margin:0;
    max-width:820px;
}

.section{
    border:1px solid var(--border);
    background: var(--panel);
    border-radius:16px;
    padding:16px 18px;
    margin-top:14px;
}

.metric-card{
    border:1px solid var(--border);
    background: var(--panel-soft);
    border-radius:14px;
    padding:14px 16px;
    min-height:92px;
}

.metric-label{
    color:var(--muted) !important;
    font-size:.84rem;
    margin-bottom:6px;
}

.metric-value{
    font-size:1.7rem;
    font-weight:800;
    color:white;
}

.note-box{
    border-left:4px solid var(--accent);
    background:#10192b;
    border-radius:10px;
    padding:12px 14px;
    margin-top:10px;
    color:var(--text);
}

.clean-box{
    border:1px solid var(--border);
    background:#111a2b;
    border-radius:14px;
    padding:14px 16px;
}

.insight-box{
    border:1px solid var(--border);
    background:#10192a;
    border-radius:14px;
    padding:16px;
    height:100%;
}

.small{
    color:var(--muted) !important;
    font-size:.92rem;
    line-height:1.7;
}

.footer-box{
    border:1px solid var(--border);
    background:#10192a;
    border-radius:14px;
    padding:14px 16px;
    text-align:center;
    color:var(--muted) !important;
}

.status-chip{
    display:inline-block;
    padding:5px 10px;
    border-radius:999px;
    font-size:12px;
    border:1px solid var(--border);
    background:#162036;
    color:#dbe6f7 !important;
    margin-right:8px;
    margin-bottom:8px;
}

.mapping-box{
    border:1px solid var(--border);
    background:#10192a;
    border-radius:12px;
    padding:12px 14px;
}

.mapping-item{
    padding:8px 0;
    border-bottom:1px solid rgba(255,255,255,.06);
    color:#eef4ff;
}

.mapping-item:last-child{
    border-bottom:none;
}

.auth-card{
    max-width:520px;
    margin:30px auto;
    border:1px solid var(--border);
    background:#121b2d;
    border-radius:18px;
    padding:22px;
}

.stTabs [data-baseweb="tab-list"]{
    gap:8px;
}
.stTabs [data-baseweb="tab"]{
    background:#162036;
    border:1px solid var(--border);
    border-radius:10px;
    color:var(--text);
    padding:.45rem .85rem;
}
.stTabs [aria-selected="true"]{
    background:#213252 !important;
    border-color:#36507e !important;
}

div.stButton > button, div.stDownloadButton > button{
    border:none;
    border-radius:10px;
    padding:.7rem 1rem;
    font-weight:700;
    background:#4f8cff;
    color:white;
}

.stSelectbox div[data-baseweb="select"] > div,
.stTextInput input,
textarea{
    background:#ffffff !important;
    color:#111827 !important;
}
.stSelectbox svg{
    fill:#111827 !important;
}

div[role="radiogroup"] label{
    color:var(--text) !important;
}

[data-testid="stDataFrame"]{
    background:#ffffff !important;
    border-radius:10px;
}

pre, code, textarea{
    color:#111827 !important;
}

@media (max-width: 900px){
    .title{
        font-size:1.8rem;
    }
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# AUTH SCREEN
# =========================================================
users = load_users()

if st.session_state.user is None:
    st.markdown("""
    <div class="auth-card">
        <div class="badge">PetroScope Workspace</div>
        <div class="title" style="font-size:1.8rem;">Account Access</div>
        <p class="subtitle">Create an account or log in to keep your projects and uploaded files.</p>
    </div>
    """, unsafe_allow_html=True)

    auth_mode = st.radio("Access", ["Login", "Register"], horizontal=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if auth_mode == "Register":
        confirm_password = st.text_input("Confirm password", type="password")
        if st.button("Create account"):
            uname = safe_name(username)
            if not uname:
                st.error("Enter a valid username.")
            elif len(password) < 4:
                st.error("Password should be at least 4 characters.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif uname in users:
                st.error("User already exists.")
            else:
                users[uname] = {
                    "password": hash_password(password),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                save_users(users)
                st.success("Account created. You can log in now.")

    else:
        if st.button("Login"):
            uname = safe_name(username)
            if uname in users and users[uname]["password"] == hash_password(password):
                st.session_state.user = uname
                st.rerun()
            else:
                st.error("Invalid username or password.")

    st.stop()

# =========================================================
# DATA HELPERS
# =========================================================
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
    df.loc[40, "Production"] *= 1.8
    df.loc[220, "Production"] *= 0.45
    df.loc[510, "Pressure"] *= 0.87
    df.loc[777, "GOR"] *= 1.22
    return df


def load_data_from_path(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suffix == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        try:
            return pd.read_csv(io.StringIO(content))
        except Exception:
            return pd.read_csv(io.StringIO(content), sep="\t")
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {suffix}")


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


# =========================================================
# AI UPGRADE HELPERS
# =========================================================
def quality_score_from_series(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    total = len(s)
    valid = s.notna().sum()
    completeness = 100 * valid / total if total else 0

    if valid < 5:
        return {
            "completeness": round(completeness, 1),
            "volatility": 0.0,
            "outlier_pct": 0.0,
            "score": round(completeness * 0.4, 1)
        }

    clean = s.dropna()
    outlier_mask = detect_outliers_iqr(s).fillna(False)
    outlier_pct = 100 * outlier_mask.sum() / total if total else 0
    mean_abs = float(np.mean(np.abs(clean))) if len(clean) else 1.0
    diff_std = float(clean.diff().dropna().std()) if len(clean) > 5 else 0.0
    volatility = 100 * diff_std / (mean_abs + 1e-9)
    raw_score = 100 - (100 - completeness) * 0.7 - outlier_pct * 1.2 - min(volatility * 0.35, 30)
    return {
        "completeness": round(completeness, 1),
        "volatility": round(volatility, 1),
        "outlier_pct": round(outlier_pct, 1),
        "score": round(max(min(raw_score, 100), 0), 1)
    }


def evaluate_data_quality(df: pd.DataFrame, mapping: dict) -> dict:
    result = {"overall_score": 0.0, "details": {}}
    scores = []
    for key in ["production", "pressure", "water_cut", "gor"]:
        col = mapping.get(key)
        if col:
            q = quality_score_from_series(df[col])
            result["details"][key] = q
            scores.append(q["score"])
    result["overall_score"] = round(float(np.mean(scores)), 1) if scores else 0.0
    return result


def detect_regime_change(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if len(s) < 20:
        return {"changed": False, "strength": 0.0, "message": "Not enough data"}

    first = s.iloc[: len(s) // 2]
    second = s.iloc[len(s) // 2 :]

    x1 = np.arange(len(first)).reshape(-1, 1)
    x2 = np.arange(len(second)).reshape(-1, 1)

    slope1 = float(LinearRegression().fit(x1, first.values).coef_[0]) if len(first) > 2 else 0.0
    slope2 = float(LinearRegression().fit(x2, second.values).coef_[0]) if len(second) > 2 else 0.0

    mean_abs = float(np.mean(np.abs(s))) + 1e-9
    strength = 100 * abs(slope2 - slope1) / mean_abs
    changed = strength > 0.7

    if changed:
        if abs(slope2) > abs(slope1):
            msg = "Trend regime change detected; behavior became steeper."
        else:
            msg = "Trend regime change detected; behavior became softer."
    else:
        msg = "No major regime change detected."

    return {
        "changed": changed,
        "strength": round(float(strength), 2),
        "slope_first": round(slope1, 4),
        "slope_second": round(slope2, 4),
        "message": msg
    }


def score_water_breakthrough(df: pd.DataFrame, mapping: dict) -> float:
    wc_col = mapping.get("water_cut")
    prod_col = mapping.get("production")
    if not wc_col or not prod_col:
        return 0.0

    wc = pd.to_numeric(df[wc_col], errors="coerce").dropna()
    prod = pd.to_numeric(df[prod_col], errors="coerce").dropna()
    if len(wc) < 8 or len(prod) < 8:
        return 0.0

    wc_change = max(float(wc.iloc[-1] - wc.iloc[0]), 0)
    wc_accel = max(float(wc.diff().dropna().tail(5).mean()), 0)
    prod_drop = max(float((prod.iloc[0] - prod.iloc[-1]) / (abs(prod.iloc[0]) + 1e-9)), 0)

    score = wc_change * 900 + wc_accel * 8000 + prod_drop * 35
    return round(max(min(score, 100), 0), 1)


def score_gas_breakthrough(df: pd.DataFrame, mapping: dict) -> float:
    gor_col = mapping.get("gor")
    prod_col = mapping.get("production")
    if not gor_col or not prod_col:
        return 0.0

    gor = pd.to_numeric(df[gor_col], errors="coerce").dropna()
    prod = pd.to_numeric(df[prod_col], errors="coerce").dropna()
    if len(gor) < 8 or len(prod) < 8:
        return 0.0

    gor_change = max(float(gor.iloc[-1] - gor.iloc[0]), 0)
    gor_accel = max(float(gor.diff().dropna().tail(5).mean()), 0)
    prod_drop = max(float((prod.iloc[0] - prod.iloc[-1]) / (abs(prod.iloc[0]) + 1e-9)), 0)

    score = gor_change * 0.18 + gor_accel * 0.9 + prod_drop * 30
    return round(max(min(score, 100), 0), 1)


def compute_confidence_score(data_quality_score: float, best_cv_rmse: float, target_mean_abs: float) -> float:
    if target_mean_abs <= 0:
        return max(min(data_quality_score * 0.5, 100), 0)
    error_ratio = 100 * best_cv_rmse / (target_mean_abs + 1e-9)
    score = 0.65 * data_quality_score + 35 - 0.55 * error_ratio
    return round(max(min(score, 100), 0), 1)


def build_univariate_supervised(series: pd.Series) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").reset_index(drop=True)
    df = pd.DataFrame({"y": s})
    df["t"] = np.arange(len(df))
    for lag in [1, 2, 3, 5, 7, 10]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["diff_1"] = df["y"].shift(1) - df["y"].shift(2)
    df["diff_3"] = df["y"].shift(1) - df["y"].shift(4)
    df["roll3"] = df["y"].shift(1).rolling(3).mean()
    df["roll5"] = df["y"].shift(1).rolling(5).mean()
    df["roll7"] = df["y"].shift(1).rolling(7).mean()
    df["roll3_std"] = df["y"].shift(1).rolling(3).std()
    df["roll7_std"] = df["y"].shift(1).rolling(7).std()
    df["ema5"] = df["y"].shift(1).ewm(span=5, adjust=False).mean()
    return df.dropna().reset_index(drop=True)


def time_series_cv_score(X: pd.DataFrame, y: pd.Series, models: dict) -> tuple:
    n = len(X)
    n_splits = min(5, max(2, n // 25))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for model_name, model in models.items():
        rmses = []
        r2s = []

        for train_idx, test_idx in splitter.split(X):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            mdl = model
            mdl.fit(X_train, y_train)
            pred = mdl.predict(X_test)

            rmses.append(np.sqrt(mean_squared_error(y_test, pred)))
            r2s.append(r2_score(y_test, pred))

        rows.append({
            "Model": model_name,
            "CV_RMSE": float(np.mean(rmses)),
            "CV_R2": float(np.mean(r2s))
        })

    res = pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)
    return res.iloc[0]["Model"], res


def recursive_feature_row(history_values: list, step_t: int) -> dict:
    arr = np.array(history_values, dtype=float)
    d = {"t": step_t}
    for lag in [1, 2, 3, 5, 7, 10]:
        d[f"lag_{lag}"] = arr[-lag] if len(arr) >= lag else arr[-1]
    d["diff_1"] = d["lag_1"] - d["lag_2"]
    d["diff_3"] = d["lag_1"] - d["lag_5"] if len(arr) >= 5 else d["lag_1"] - d["lag_2"]
    d["roll3"] = float(np.mean(arr[-3:])) if len(arr) >= 3 else float(np.mean(arr))
    d["roll5"] = float(np.mean(arr[-5:])) if len(arr) >= 5 else float(np.mean(arr))
    d["roll7"] = float(np.mean(arr[-7:])) if len(arr) >= 7 else float(np.mean(arr))
    d["roll3_std"] = float(np.std(arr[-3:])) if len(arr) >= 3 else 0.0
    d["roll7_std"] = float(np.std(arr[-7:])) if len(arr) >= 7 else 0.0
    if len(arr) >= 5:
        ema = pd.Series(arr).ewm(span=5, adjust=False).mean().iloc[-1]
    else:
        ema = float(np.mean(arr))
    d["ema5"] = float(ema)
    return d


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


def hybrid_forecast_engine(df: pd.DataFrame, mapping: dict, target_key: str = "production", horizon: int = 30) -> dict:
    target_col = mapping.get(target_key)
    time_col = mapping.get("time")
    if not target_col:
        return {"ok": False, "reason": "Target column not mapped."}

    s = pd.to_numeric(df[target_col], errors="coerce").dropna().reset_index(drop=True)
    if len(s) < 25:
        return {"ok": False, "reason": "Not enough clean data for advanced forecast."}

    supervised = build_univariate_supervised(s)
    if len(supervised) < 20:
        return {"ok": False, "reason": "Not enough supervised rows."}

    X = supervised.drop(columns=["y"])
    y = supervised["y"]

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=220, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=260, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    }

    best_model_name, cv_df = time_series_cv_score(X, y, models)
    best_model = models[best_model_name]
    best_model.fit(X, y)

    history = list(s.values)
    future_pred = []

    for step in range(1, horizon + 1):
        row = recursive_feature_row(history, len(history) + step - 1)
        pred = float(best_model.predict(pd.DataFrame([row]))[0])

        if target_key in ["production", "pressure"]:
            pred = max(pred, 0.0)
        if target_key == "water_cut":
            pred = max(min(pred, 0.99), 0.0)
        if target_key == "gor":
            pred = max(pred, 0.0)

        history.append(pred)
        future_pred.append(pred)

    future_pred = np.array(future_pred, dtype=float)

    decline_name = None
    decline_pred = None
    blend_weight_ml = 1.0

    if target_key == "production" and time_col:
        x_num, _ = time_to_numeric(df[time_col])
        q = pd.to_numeric(df[target_col], errors="coerce").values
        fit = fit_decline_models(x_num, q)
        if fit is not None:
            decline_name, results = fit
            mask = np.isfinite(x_num) & np.isfinite(q) & (q > 0)
            t = x_num[mask]
            q_clean = q[mask]
            q0 = q_clean[0]
            future_t = np.arange(t[-1] + 1, t[-1] + horizon + 1, dtype=float)

            if decline_name == "Exponential":
                D = results["Exponential"]["D"]
                decline_pred = q0 * np.exp(-D * (future_t - t[0]))
                decline_rmse = results["Exponential"]["rmse"]
            elif decline_name == "Harmonic":
                D = results["Harmonic"]["D"]
                decline_pred = q0 / (1 + D * (future_t - t[0]))
                decline_rmse = results["Harmonic"]["rmse"]
            else:
                b = results["Hyperbolic"]["b"]
                D = results["Hyperbolic"]["D"]
                decline_pred = q0 / np.power(1 + b * D * (future_t - t[0]), 1.0 / b)
                decline_rmse = results["Hyperbolic"]["rmse"]

            ml_rmse = float(cv_df.iloc[0]["CV_RMSE"])
            inv_ml = 1 / (ml_rmse + 1e-9)
            inv_decl = 1 / (decline_rmse + 1e-9)
            blend_weight_ml = float(inv_ml / (inv_ml + inv_decl))
            future_pred = blend_weight_ml * future_pred + (1 - blend_weight_ml) * decline_pred

    train_pred = best_model.predict(X)
    residual_sigma = float(np.std(y - train_pred))
    lower = future_pred - 1.96 * residual_sigma
    upper = future_pred + 1.96 * residual_sigma

    target_mean_abs = float(np.mean(np.abs(y))) if len(y) else 1.0
    dq = quality_score_from_series(s)
    confidence = compute_confidence_score(dq["score"], float(cv_df.iloc[0]["CV_RMSE"]), target_mean_abs)

    return {
        "ok": True,
        "target_key": target_key,
        "historical_y": s.values,
        "historical_idx": np.arange(len(s)),
        "future_idx": np.arange(len(s), len(s) + horizon),
        "future_pred": future_pred,
        "lower": lower,
        "upper": upper,
        "best_model_name": best_model_name,
        "cv_results": cv_df,
        "confidence": confidence,
        "residual_sigma": residual_sigma,
        "decline_name": decline_name,
        "blend_weight_ml": round(blend_weight_ml, 3),
    }


def advanced_ai_analysis(df: pd.DataFrame, mapping: dict) -> dict:
    quality = evaluate_data_quality(df, mapping)
    prod_col = mapping.get("production")
    press_col = mapping.get("pressure")
    wc_col = mapping.get("water_cut")
    gor_col = mapping.get("gor")

    insights = []
    evidence = []
    recommendations = []
    limitations = []

    regime_prod = detect_regime_change(df[prod_col]) if prod_col else {"changed": False, "message": "No production"}
    regime_press = detect_regime_change(df[press_col]) if press_col else {"changed": False, "message": "No pressure"}

    water_break_score = score_water_breakthrough(df, mapping)
    gas_break_score = score_gas_breakthrough(df, mapping)

    prod = pd.to_numeric(df[prod_col], errors="coerce").dropna() if prod_col else pd.Series(dtype=float)
    press = pd.to_numeric(df[press_col], errors="coerce").dropna() if press_col else pd.Series(dtype=float)
    wc = pd.to_numeric(df[wc_col], errors="coerce").dropna() if wc_col else pd.Series(dtype=float)
    gor = pd.to_numeric(df[gor_col], errors="coerce").dropna() if gor_col else pd.Series(dtype=float)

    if len(prod) >= 8 and prod.iloc[0] != 0:
        prod_drop_pct = 100 * (prod.iloc[0] - prod.iloc[-1]) / abs(prod.iloc[0])
        evidence.append(f"Production moved from {prod.iloc[0]:.2f} to {prod.iloc[-1]:.2f} ({prod_drop_pct:.1f}% change).")
        if prod_drop_pct > 10:
            insights.append("Production has materially declined over the observed period.")
            recommendations.append("Review whether decline is reservoir-driven, operational, or both.")

    if len(press) >= 8:
        press_drop = float(press.iloc[0] - press.iloc[-1])
        evidence.append(f"Pressure changed by {press_drop:.2f} units over the analyzed window.")
        if press_drop > 0:
            insights.append("Pressure depletion signal is present.")
            recommendations.append("Check depletion support and pressure-maintenance strategy.")

    if len(wc) >= 8:
        wc_change = float(wc.iloc[-1] - wc.iloc[0])
        evidence.append(f"Water cut changed by {wc_change:.4f}.")
        if wc_change > 0.03:
            insights.append("Water cut growth is strong enough to deserve engineering attention.")
            recommendations.append("Investigate water breakthrough, coning, sweep efficiency, or completion behavior.")

    if len(gor) >= 8:
        gor_change = float(gor.iloc[-1] - gor.iloc[0])
        evidence.append(f"GOR changed by {gor_change:.2f}.")
        if gor_change > 30:
            insights.append("GOR increase suggests stronger gas influence in the production behavior.")
            recommendations.append("Check gas-cap behavior, gas breakthrough, and flowing conditions.")

    if regime_prod.get("changed"):
        insights.append("Production trend regime change detected.")
        evidence.append(regime_prod["message"])
        recommendations.append("Separate historical performance into more stable operating periods before making decisions.")

    if regime_press.get("changed"):
        insights.append("Pressure trend regime change detected.")
        evidence.append(regime_press["message"])

    if water_break_score >= 55:
        insights.append("Water breakthrough risk is high.")
        evidence.append(f"Water breakthrough score = {water_break_score}/100.")
    elif water_break_score >= 30:
        insights.append("Water breakthrough risk is moderate.")
        evidence.append(f"Water breakthrough score = {water_break_score}/100.")

    if gas_break_score >= 55:
        insights.append("Gas breakthrough risk is high.")
        evidence.append(f"Gas breakthrough score = {gas_break_score}/100.")
    elif gas_break_score >= 30:
        insights.append("Gas breakthrough risk is moderate.")
        evidence.append(f"Gas breakthrough score = {gas_break_score}/100.")

    if quality["overall_score"] < 60:
        limitations.append("Data quality is weak; model outputs should be treated carefully.")
        recommendations.append("Clean the dataset and validate suspicious points before relying on the forecast.")
    elif quality["overall_score"] < 80:
        limitations.append("Data quality is moderate; forecast confidence is acceptable but not ideal.")
    else:
        evidence.append(f"Overall data quality score = {quality['overall_score']}/100.")

    if not insights:
        insights.append("No dominant abnormal engineering pattern was detected with high confidence.")

    return {
        "quality": quality,
        "regime_prod": regime_prod,
        "regime_press": regime_press,
        "water_break_score": water_break_score,
        "gas_break_score": gas_break_score,
        "insights": insights,
        "evidence": evidence,
        "recommendations": list(dict.fromkeys(recommendations)),
        "limitations": list(dict.fromkeys(limitations)),
    }


# =========================================================
# REPORT / PDF
# =========================================================
def generate_ai_report_text(
    df,
    mapping,
    selected_well,
    ai_pack,
    advanced_pack,
    forecast_pack,
    best_decline_name,
    decline_params_text,
    eur_value
):
    lines = []
    lines.append("PETROSCOPE ADVANCED AI REPORT")
    lines.append("=" * 72)
    lines.append(f"Rows analyzed: {len(df)}")
    if selected_well is not None:
        lines.append(f"Selected well: {selected_well}")

    lines.append(f"Well status: {ai_pack['well_class']}")
    lines.append(f"Estimated drive mechanism: {ai_pack['drive_mechanism']}")
    lines.append(f"Reservoir health score: {ai_pack['health_score']}/100")
    lines.append(f"Production efficiency score: {ai_pack['efficiency_score']}/100")
    lines.append(f"Water breakthrough score: {advanced_pack['water_break_score']}/100")
    lines.append(f"Gas breakthrough score: {advanced_pack['gas_break_score']}/100")
    lines.append(f"Data quality score: {advanced_pack['quality']['overall_score']}/100")

    if best_decline_name:
        lines.append(f"Best decline model: {best_decline_name}")
    if decline_params_text:
        lines.append(f"Decline parameters: {decline_params_text}")
    if eur_value is not None:
        lines.append(f"Estimated EUR: {eur_value}")

    if forecast_pack and forecast_pack.get("ok"):
        lines.append(f"Best forecast model: {forecast_pack['best_model_name']}")
        lines.append(f"Forecast confidence: {forecast_pack['confidence']}/100")
        if forecast_pack.get("decline_name"):
            lines.append(f"Hybrid blend: ML + {forecast_pack['decline_name']} (ML weight = {forecast_pack['blend_weight_ml']})")

    lines.append("")
    lines.append("Mapped Columns:")
    for k, v in mapping.items():
        if v:
            lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("Main Findings:")
    for item in advanced_pack["insights"]:
        lines.append(f"* {item}")

    lines.append("")
    lines.append("Evidence:")
    for item in advanced_pack["evidence"]:
        lines.append(f"* {item}")

    lines.append("")
    lines.append("Suggested Actions:")
    for item in list(dict.fromkeys(ai_pack["recommendations"] + advanced_pack["recommendations"])):
        lines.append(f"* {item}")

    lines.append("")
    lines.append("Limitations:")
    if advanced_pack["limitations"]:
        for item in advanced_pack["limitations"]:
            lines.append(f"* {item}")
    else:
        lines.append("* No major limitations detected from current data quality checks.")

    return "\n".join(lines)


def create_pdf_report(ai_pack, advanced_pack, forecast_pack, best_decline_name, decline_params_text, eur_value):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp.name, pagesize=letter)
    _, height = letter
    y = height - 50

    def draw_line(text, font="Helvetica", size=10, step=15):
        nonlocal y
        if y < 70:
            c.showPage()
            y = height - 50
        c.setFont(font, size)
        c.drawString(50, y, text[:110])
        y -= step

    draw_line("PetroScope Advanced AI Report", font="Helvetica-Bold", size=16, step=24)
    draw_line(f"Well status: {ai_pack['well_class']}", size=11)
    draw_line(f"Estimated drive mechanism: {ai_pack['drive_mechanism']}", size=11)
    draw_line(f"Reservoir health score: {ai_pack['health_score']}/100", size=11)
    draw_line(f"Production efficiency score: {ai_pack['efficiency_score']}/100", size=11)
    draw_line(f"Water breakthrough score: {advanced_pack['water_break_score']}/100", size=11)
    draw_line(f"Gas breakthrough score: {advanced_pack['gas_break_score']}/100", size=11)
    draw_line(f"Data quality score: {advanced_pack['quality']['overall_score']}/100", size=11)
    draw_line(f"Best decline model: {best_decline_name if best_decline_name else '-'}", size=11)
    draw_line(f"Decline parameters: {decline_params_text if decline_params_text else '-'}", size=11)
    draw_line(f"EUR: {eur_value if eur_value is not None else '-'}", size=11)
    draw_line(f"Forecast model: {forecast_pack['best_model_name'] if forecast_pack and forecast_pack.get('ok') else '-'}", size=11)
    draw_line(f"Forecast confidence: {forecast_pack['confidence'] if forecast_pack and forecast_pack.get('ok') else '-'}", size=11, step=22)

    draw_line("Main Findings", font="Helvetica-Bold", size=12, step=18)
    for item in advanced_pack["insights"][:8]:
        draw_line(f"- {item}")

    draw_line("Evidence", font="Helvetica-Bold", size=12, step=18)
    for item in advanced_pack["evidence"][:8]:
        draw_line(f"- {item}")

    draw_line("Recommendations", font="Helvetica-Bold", size=12, step=18)
    merged_recs = list(dict.fromkeys(ai_pack["recommendations"] + advanced_pack["recommendations"]))
    for item in merged_recs[:8]:
        draw_line(f"- {item}")

    c.save()
    return temp.name


# =========================================================
# HEADER + ACCOUNT INFO
# =========================================================
st.sidebar.write(f"Logged in as: **{st.session_state.user}**")
if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.session_state.loaded_project_name = None
    st.session_state.loaded_project_data = None
    st.rerun()

st.markdown("""
<div class="app-shell">
    <div class="hero">
        <div class="hero-top">
            <div>
                <span class="badge">PetroScope Advanced Workspace</span>
                <div class="title">PetroScope</div>
                <p class="subtitle">
                    A stronger petroleum analytics workspace with hybrid forecasting, data-quality scoring,
                    breakthrough detection, confidence scoring, and evidence-based AI reporting.
                </p>
            </div>
            <div class="clean-box" style="min-width:260px;">
                <div class="small"><b>Current use</b></div>
                <div class="small">Well review • Forecasting • Diagnostics • Saved projects • AI report</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section">
    <span class="status-chip">Advanced AI</span>
    <span class="status-chip">Hybrid Forecast</span>
    <span class="status-chip">Data Quality</span>
    <span class="status-chip">Evidence Report</span>
</div>
""", unsafe_allow_html=True)

# =========================================================
# PROJECTS SIDEBAR
# =========================================================
st.sidebar.markdown("### My Projects")
projects_list = list_projects(st.session_state.user)

if projects_list:
    selected_project_name = st.sidebar.selectbox(
        "Saved projects",
        [p["name"] for p in projects_list]
    )
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("Load Project", key="load_project_btn"):
            loaded = load_project(st.session_state.user, selected_project_name)
            if loaded:
                st.session_state.loaded_project_name = selected_project_name
                st.session_state.loaded_project_data = loaded
                st.success(f"Loaded project: {selected_project_name}")
    with c2:
        if st.button("Delete Project", key="delete_project_btn"):
            delete_project(st.session_state.user, selected_project_name)
            st.session_state.loaded_project_name = None
            st.session_state.loaded_project_data = None
            st.rerun()
else:
    st.sidebar.caption("No saved projects yet.")

st.sidebar.markdown("### My Files")
saved_files = list_user_files(st.session_state.user)
if saved_files:
    for f in saved_files[:8]:
        st.sidebar.caption(f"• {f['name']} ({f['modified']})")
else:
    st.sidebar.caption("No uploaded files yet.")

# =========================================================
# CONTROLS
# =========================================================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("Workspace")
ctrl1, ctrl2 = st.columns([1.0, 1.4])

with ctrl1:
    source = st.radio("Data source", ["Upload file", "Use demo dataset"], index=0 if st.session_state.source_mode == "Upload file" else 1)
    st.session_state.source_mode = source

with ctrl2:
    uploaded = None
    selected_saved_file = None

    if source == "Upload file":
        upload_choice = st.radio("Upload mode", ["Upload new file", "Use saved file"], horizontal=True)

        if upload_choice == "Upload new file":
            uploaded = st.file_uploader("Upload structured dataset", type=["csv", "xlsx", "xls", "txt", "json"])
            st.caption("Supported formats: CSV, Excel, TXT, JSON")

            if uploaded is not None:
                save_path = save_uploaded_file_for_user(st.session_state.user, uploaded)
                st.success(f"File saved: {Path(save_path).name}")

        else:
            files = list_user_files(st.session_state.user)
            if files:
                selected_saved_file = st.selectbox("Saved files", [f["name"] for f in files])
            else:
                st.info("No saved files found. Upload one first.")
    else:
        st.info("Demo dataset is loaded.")
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================
if source == "Use demo dataset":
    raw_df = demo_dataset()
    current_file_name = "demo_dataset"
else:
    if uploaded is not None:
        try:
            raw_df = load_uploaded_file(uploaded)
            current_file_name = uploaded.name
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")
            st.stop()
    elif selected_saved_file is not None:
        try:
            file_path = os.path.join(user_uploads_dir(st.session_state.user), selected_saved_file)
            raw_df = load_data_from_path(file_path)
            current_file_name = selected_saved_file
        except Exception as e:
            st.error(f"Could not read saved file: {e}")
            st.stop()
    else:
        st.info("Choose a file to continue.")
        st.stop()

if raw_df.empty:
    st.warning("The dataset is empty.")
    st.stop()

det = auto_detect_columns(raw_df)

# =========================================================
# KPI ROW
# =========================================================
st.markdown('<div class="section">', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Rows</div><div class="metric-value">{len(raw_df)}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Columns</div><div class="metric-value">{len(raw_df.columns)}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Detected production</div><div class="metric-value">{det["production"] or "-"}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Detected well</div><div class="metric-value">{det["well"] or "-"}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Home",
    "Dataset",
    "Setup",
    "Review",
    "Advanced AI",
    "Forecast",
    "Export",
    "Projects"
])

with tab1:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Workspace Notes")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
<div class="clean-box">
    <h4 style="margin-top:0;">Purpose</h4>
    <p class="small">Used to review production trends, pressure behavior, water cut, GOR, forecast risk, and generate stronger evidence-based engineering notes.</p>
</div>
""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
<div class="clean-box">
    <h4 style="margin-top:0;">Session</h4>
    <p class="small">Last opened: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    <p class="small">Source: {source}</p>
    <p class="small">File: {current_file_name}</p>
</div>
""", unsafe_allow_html=True)

    with col_c:
        loaded_name = st.session_state.loaded_project_name if st.session_state.loaded_project_name else "None"
        st.markdown(f"""
<div class="clean-box">
    <h4 style="margin-top:0;">Project</h4>
    <p class="small">Loaded project: {loaded_name}</p>
    <p class="small">User: {st.session_state.user}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="note-box">
    <b>Recommended workflow:</b> load dataset → confirm columns → review trends →
    inspect advanced AI findings → run hybrid forecast → save project → export report.
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Dataset Preview")
    st.dataframe(raw_df.head(30), use_container_width=True)
    st.write("Columns:", list(raw_df.columns))
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Column Setup")

    cols = list(raw_df.columns)
    num_cols = [c for c in raw_df.columns if pd.api.types.is_numeric_dtype(raw_df[c])]
    left, mid, right = st.columns(3)

    with left:
        time_col = st.selectbox("Time / date", cols, index=cols.index(det["time"]) if det["time"] in cols else 0)
        well_opts = ["None"] + cols
        field_opts = ["None"] + cols
        well_col = st.selectbox("Well column", well_opts, index=well_opts.index(det["well"]) if det["well"] in well_opts else 0)
        field_col = st.selectbox("Field column", field_opts, index=field_opts.index(det["field"]) if det["field"] in field_opts else 0)

    with mid:
        prod_opts = ["None"] + num_cols
        prod_col = st.selectbox("Production", prod_opts, index=prod_opts.index(det["production"]) if det["production"] in prod_opts else 0)
        press_col = st.selectbox("Pressure", prod_opts, index=prod_opts.index(det["pressure"]) if det["pressure"] in prod_opts else 0)

    with right:
        wc_col = st.selectbox("Water cut", prod_opts, index=prod_opts.index(det["water_cut"]) if det["water_cut"] in prod_opts else 0)
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
        st.error("Duplicate column mapping detected. Choose a different column for each role.")
        st.stop()

    st.success("Column setup updated.")
    st.markdown('<div class="mapping-box">', unsafe_allow_html=True)
    for k, v in mapping.items():
        shown = v if v is not None else "-"
        st.markdown(f'<div class="mapping-item"><b>{k}</b>: {shown}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

selected_mapping_values = [v for v in mapping.values() if v is not None]
if len(selected_mapping_values) != len(set(selected_mapping_values)):
    st.error("Duplicate column mapping detected. Fix the setup in Setup tab.")
    st.stop()

df = raw_df.copy()
selected_well = None
if mapping["well"]:
    wells = ["All"] + sorted(df[mapping["well"]].dropna().astype(str).unique().tolist())
    selected_well = st.selectbox("Well filter", wells, key="well_filter")
    if selected_well != "All":
        df = df[df[mapping["well"]].astype(str) == selected_well].copy()

best_decline_name = None
decline_params_text = ""
eur_value = None

with tab4:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Performance Review")

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
        fig_prod.update_layout(template="plotly_dark", title="Production vs Time", title_font_size=20, font=dict(size=13), hovermode="x unified")
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
            fig_press.update_layout(template="plotly_dark", title="Pressure vs Time", title_font_size=20, font=dict(size=13), hovermode="x unified")
            st.plotly_chart(fig_press, use_container_width=True)

    with c2:
        if mapping["water_cut"]:
            fig_wc = go.Figure()
            if mapping["well"] and selected_well == "All":
                for well_name, sub in df.groupby(mapping["well"]):
                    fig_wc.add_trace(go.Scatter(x=sub[mapping["time"]], y=sub[mapping["water_cut"]], mode="lines", name=str(well_name)))
            else:
                fig_wc.add_trace(go.Scatter(x=df[mapping["time"]], y=df[mapping["water_cut"]], mode="lines+markers", name="Water Cut"))
            fig_wc.update_layout(template="plotly_dark", title="Water Cut vs Time", title_font_size=20, font=dict(size=13), hovermode="x unified")
            st.plotly_chart(fig_wc, use_container_width=True)

    if mapping["gor"]:
        fig_gor = go.Figure()
        if mapping["well"] and selected_well == "All":
            for well_name, sub in df.groupby(mapping["well"]):
                fig_gor.add_trace(go.Scatter(x=sub[mapping["time"]], y=sub[mapping["gor"]], mode="lines", name=str(well_name)))
        else:
            fig_gor.add_trace(go.Scatter(x=df[mapping["time"]], y=df[mapping["gor"]], mode="lines+markers", name="GOR"))
        fig_gor.update_layout(template="plotly_dark", title="GOR vs Time", title_font_size=20, font=dict(size=13), hovermode="x unified")
        st.plotly_chart(fig_gor, use_container_width=True)

    if mapping["well"] and mapping["production"] and selected_well == "All":
        st.markdown("---")
        st.subheader("Well Ranking")
        ranking_df = rank_wells(df, mapping["well"], mapping["production"])
        st.dataframe(ranking_df, use_container_width=True)

    if mapping["field"] and mapping["production"]:
        st.markdown("---")
        st.subheader("Field Summary")
        field_rank_df = rank_fields(df, mapping["field"], mapping["production"])
        st.dataframe(field_rank_df, use_container_width=True)

    if mapping["production"]:
        st.markdown("---")
        st.subheader("Anomaly Check")
        anom_mask = detect_outliers_iqr(df[mapping["production"]])
        temp = df.copy()
        temp["_anomaly"] = np.where(anom_mask.fillna(False), "Possible Outlier", "Normal")

        fig_anom = go.Figure()
        normal = temp[temp["_anomaly"] == "Normal"]
        out = temp[temp["_anomaly"] == "Possible Outlier"]

        fig_anom.add_trace(go.Scatter(x=normal[mapping["time"]], y=normal[mapping["production"]], mode="markers", name="Normal"))
        fig_anom.add_trace(go.Scatter(x=out[mapping["time"]], y=out[mapping["production"]], mode="markers", name="Possible Outlier"))
        fig_anom.update_layout(template="plotly_dark", title="Production Outlier Check", title_font_size=20, font=dict(size=13))
        st.plotly_chart(fig_anom, use_container_width=True)

    st.markdown("---")
    st.subheader("Missing Values")
    st.dataframe(missing_values_summary(df), use_container_width=True)

    corr = correlation_matrix_safe(df, mapping)
    if corr is not None:
        st.markdown("---")
        st.subheader("Correlation")
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorscale="Blues"
        ))
        fig_corr.update_layout(template="plotly_dark", title="Correlation Matrix", title_font_size=20, font=dict(size=13))
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Advanced AI")

    ai_pack = {
        "risks": calculate_risk_scores(df, mapping),
        "health_score": calculate_reservoir_health_score(df, mapping, calculate_risk_scores(df, mapping)),
        "efficiency_score": calculate_production_efficiency_score(df, mapping),
        "well_class": classify_well(df, mapping, calculate_risk_scores(df, mapping)),
        "drive_mechanism": estimate_drive_mechanism(df, mapping),
        "well_problems": well_problem_detection(df, mapping, calculate_risk_scores(df, mapping)),
        "lift_recs": artificial_lift_suggestions(df, mapping, calculate_risk_scores(df, mapping)),
        "recommendations": production_optimization_recommendations(df, mapping, calculate_risk_scores(df, mapping)) + workover_recommendations(df, mapping, calculate_risk_scores(df, mapping)),
    }

    advanced_pack = advanced_ai_analysis(df, mapping)

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Data Quality", f"{advanced_pack['quality']['overall_score']}/100")
    with q2:
        st.metric("Water Breakthrough", f"{advanced_pack['water_break_score']}/100")
    with q3:
        st.metric("Gas Breakthrough", f"{advanced_pack['gas_break_score']}/100")
    with q4:
        regime_flag = "Yes" if advanced_pack["regime_prod"]["changed"] else "No"
        st.metric("Regime Change", regime_flag)

    st.markdown("---")
    st.markdown("### Data Quality Details")
    quality_rows = []
    for k, v in advanced_pack["quality"]["details"].items():
        quality_rows.append({
            "Variable": k,
            "Score": v["score"],
            "Completeness %": v["completeness"],
            "Outlier %": v["outlier_pct"],
            "Volatility %": v["volatility"],
        })
    if quality_rows:
        st.dataframe(pd.DataFrame(quality_rows), use_container_width=True)

    st.markdown("---")
    l1, l2 = st.columns(2)

    with l1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### Main Findings")
        for item in advanced_pack["insights"]:
            st.markdown(f"<p style='color:#ffffff; font-size:17px; font-weight:600; line-height:1.8; margin-bottom:8px;'>• {item}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with l2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### Evidence")
        for item in advanced_pack["evidence"]:
            st.markdown(f"<p style='color:#ffffff; font-size:16px; font-weight:500; line-height:1.8; margin-bottom:8px;'>• {item}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    l3, l4 = st.columns(2)

    with l3:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### Suggested Actions")
        merged_recs = list(dict.fromkeys(ai_pack["recommendations"] + advanced_pack["recommendations"]))
        for item in merged_recs:
            st.markdown(f"<p style='color:#ffffff; font-size:16px; font-weight:500; line-height:1.8; margin-bottom:8px;'>• {item}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with l4:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### Limits / Cautions")
        if advanced_pack["limitations"]:
            for item in advanced_pack["limitations"]:
                st.markdown(f"<p style='color:#ffffff; font-size:16px; font-weight:500; line-height:1.8; margin-bottom:8px;'>• {item}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#ffffff; font-size:16px; font-weight:500; line-height:1.8;'>• No major limitations detected from current data quality checks.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab6:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Forecast")

    target_choice = st.selectbox(
        "Target variable",
        ["production", "pressure", "water_cut", "gor"]
    )
    horizon = st.slider("Forecast horizon", 10, 90, 30, 5)

    forecast_pack = hybrid_forecast_engine(df, mapping, target_choice, horizon)

    if not forecast_pack.get("ok"):
        st.warning(forecast_pack.get("reason", "Forecast unavailable."))
    else:
        st.success(f"Best model: {forecast_pack['best_model_name']}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Forecast Confidence", f"{forecast_pack['confidence']}/100")
        with c2:
            if forecast_pack.get("decline_name"):
                st.metric("Hybrid Decline", forecast_pack["decline_name"])
            else:
                st.metric("Hybrid Decline", "-")
        with c3:
            st.metric("ML Weight", forecast_pack["blend_weight_ml"])

        st.dataframe(forecast_pack["cv_results"], use_container_width=True)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=forecast_pack["historical_idx"],
            y=forecast_pack["historical_y"],
            mode="lines+markers",
            name="Historical"
        ))
        fig_fc.add_trace(go.Scatter(
            x=forecast_pack["future_idx"],
            y=forecast_pack["future_pred"],
            mode="lines+markers",
            name="Forecast"
        ))
        fig_fc.add_trace(go.Scatter(
            x=np.concatenate([forecast_pack["future_idx"], forecast_pack["future_idx"][::-1]]),
            y=np.concatenate([forecast_pack["upper"], forecast_pack["lower"][::-1]]),
            fill="toself",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Interval",
            hoverinfo="skip"
        ))
        fig_fc.update_layout(
            template="plotly_dark",
            title=f"Hybrid Forecast — {target_choice}",
            title_font_size=20,
            font=dict(size=13),
            hovermode="x unified"
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown(f"""
<div class="note-box">
    <b>Forecast note:</b> confidence = <b>{forecast_pack['confidence']}/100</b>.
    For production, the engine blends ML with decline behavior when decline fitting is available.
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab7:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Export")

    ai_pack = {
        "risks": calculate_risk_scores(df, mapping),
        "health_score": calculate_reservoir_health_score(df, mapping, calculate_risk_scores(df, mapping)),
        "efficiency_score": calculate_production_efficiency_score(df, mapping),
        "well_class": classify_well(df, mapping, calculate_risk_scores(df, mapping)),
        "drive_mechanism": estimate_drive_mechanism(df, mapping),
        "well_problems": well_problem_detection(df, mapping, calculate_risk_scores(df, mapping)),
        "lift_recs": artificial_lift_suggestions(df, mapping, calculate_risk_scores(df, mapping)),
        "recommendations": production_optimization_recommendations(df, mapping, calculate_risk_scores(df, mapping)) + workover_recommendations(df, mapping, calculate_risk_scores(df, mapping)),
    }
    advanced_pack = advanced_ai_analysis(df, mapping)

    if mapping["production"]:
        x_num, _ = time_to_numeric(df[mapping["time"]])
        q = pd.to_numeric(df[mapping["production"]], errors="coerce").values
        fit = fit_decline_models(x_num, q)
        if fit is not None:
            best_decline_name, results = fit
            if best_decline_name == "Exponential":
                decline_params_text = f"D = {results['Exponential']['D']:.5f}"
            elif best_decline_name == "Harmonic":
                decline_params_text = f"D = {results['Harmonic']['D']:.5f}"
            else:
                decline_params_text = f"D = {results['Hyperbolic']['D']:.5f}, b = {results['Hyperbolic']['b']:.2f}"
            eur_value = calculate_eur(df, mapping, results, best_decline_name)

    forecast_pack_export = hybrid_forecast_engine(df, mapping, "production", 30) if mapping.get("production") else None

    report_text = generate_ai_report_text(
        df=df,
        mapping=mapping,
        selected_well=selected_well,
        ai_pack=ai_pack,
        advanced_pack=advanced_pack,
        forecast_pack=forecast_pack_export,
        best_decline_name=best_decline_name,
        decline_params_text=decline_params_text,
        eur_value=eur_value
    )

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download AI report (.txt)",
            data=report_text.encode("utf-8"),
            file_name="petroscope_advanced_ai_report.txt",
            mime="text/plain"
        )
    with d2:
        st.download_button(
            "Download filtered data (.csv)",
            data=df.to_csv(index=False),
            file_name="filtered_petroscope_data.csv",
            mime="text/csv"
        )

    st.markdown("### PDF Report")
    if st.button("Generate PDF"):
        pdf_path = create_pdf_report(
            ai_pack=ai_pack,
            advanced_pack=advanced_pack,
            forecast_pack=forecast_pack_export,
            best_decline_name=best_decline_name,
            decline_params_text=decline_params_text,
            eur_value=eur_value
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF",
                f.read(),
                file_name="petroscope_advanced_ai_report.pdf",
                mime="application/pdf"
            )

    st.markdown("---")
    st.subheader("API Preview")
    api_payload = {
        "platform": "PetroScope",
        "selected_well": selected_well,
        "well_status": ai_pack["well_class"],
        "health_score": ai_pack["health_score"],
        "data_quality_score": advanced_pack["quality"]["overall_score"],
        "water_breakthrough_score": advanced_pack["water_break_score"],
        "gas_breakthrough_score": advanced_pack["gas_break_score"],
        "best_decline_model": best_decline_name,
        "forecast_model": forecast_pack_export["best_model_name"] if forecast_pack_export and forecast_pack_export.get("ok") else None,
        "forecast_confidence": forecast_pack_export["confidence"] if forecast_pack_export and forecast_pack_export.get("ok") else None,
    }
    st.text_area("API preview", json.dumps(api_payload, indent=2), height=260)

    st.markdown("""
<div class="footer-box">
    PetroScope • Advanced engineering workspace for production data review
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab8:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Projects")

    c1, c2 = st.columns([1.2, 0.8])

    with c1:
        st.markdown("### Save Current Project")
        project_name = st.text_input("Project name", value=st.session_state.loaded_project_name or "")
        project_note = st.text_area("Project note", height=100)

        if st.button("Save Project"):
            if not project_name.strip():
                st.error("Enter a project name.")
            else:
                adv_pack = advanced_ai_analysis(df, mapping)
                fc_pack = hybrid_forecast_engine(df, mapping, "production", 30) if mapping.get("production") else None
                preview_records = raw_df.head(10).to_dict(orient="records")
                payload = {
                    "note": project_note,
                    "source_mode": source,
                    "file_name": current_file_name,
                    "mapping": mapping,
                    "selected_well": selected_well,
                    "data_preview": preview_records,
                    "row_count": int(len(raw_df)),
                    "columns": list(raw_df.columns),
                    "data_quality_score": adv_pack["quality"]["overall_score"],
                    "water_breakthrough_score": adv_pack["water_break_score"],
                    "gas_breakthrough_score": adv_pack["gas_break_score"],
                    "forecast_confidence": fc_pack["confidence"] if fc_pack and fc_pack.get("ok") else None,
                }
                save_project(st.session_state.user, project_name, payload)
                st.session_state.loaded_project_name = safe_name(project_name)
                st.success("Project saved.")

    with c2:
        st.markdown("### Project Summary")
        adv_pack = advanced_ai_analysis(df, mapping)
        st.markdown(f"""
<div class="clean-box">
    <p class="small">Current file: <b>{current_file_name}</b></p>
    <p class="small">Rows: <b>{len(raw_df)}</b></p>
    <p class="small">Data quality: <b>{adv_pack['quality']['overall_score']}/100</b></p>
    <p class="small">Loaded project: <b>{st.session_state.loaded_project_name or "-"}</b></p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Saved Projects")

    current_projects = list_projects(st.session_state.user)
    if current_projects:
        proj_df = pd.DataFrame(current_projects)[["name", "saved_at"]]
        proj_df.columns = ["Project", "Saved At"]
        st.dataframe(proj_df, use_container_width=True)
    else:
        st.info("No saved projects yet.")

    if st.session_state.loaded_project_data:
        st.markdown("---")
        st.markdown("### Loaded Project Details")
        loaded = st.session_state.loaded_project_data
        st.markdown(f"""
<div class="clean-box">
    <p class="small"><b>Name:</b> {loaded.get("project_name","-")}</p>
    <p class="small"><b>Saved at:</b> {loaded.get("saved_at","-")}</p>
    <p class="small"><b>File:</b> {loaded.get("file_name","-")}</p>
    <p class="small"><b>Rows:</b> {loaded.get("row_count","-")}</p>
    <p class="small"><b>Data quality:</b> {loaded.get("data_quality_score","-")}</p>
    <p class="small"><b>Water breakthrough:</b> {loaded.get("water_breakthrough_score","-")}</p>
    <p class="small"><b>Gas breakthrough:</b> {loaded.get("gas_breakthrough_score","-")}</p>
    <p class="small"><b>Forecast confidence:</b> {loaded.get("forecast_confidence","-")}</p>
    <p class="small"><b>Note:</b> {loaded.get("note","-")}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
