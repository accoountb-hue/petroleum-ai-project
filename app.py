import io
import os
import json
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
    return cleaned[:80] if cleaned else "item"


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


def list_user_files(username: str) -> List[dict]:
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


def list_projects(username: str) -> List[dict]:
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


def load_project(username: str, project_name: str) -> Optional[dict]:
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
# UI STYLE
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


def demo_dataset() -> pd.DataFrame:
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


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
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


def auto_detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
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


def time_to_numeric(series: pd.Series):
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().sum() == len(series):
        base = dt.min()
        x = (dt - base).dt.total_seconds() / 86400.0
        return x.values, dt

    num = pd.to_numeric(series, errors="coerce")
    if num.notna().sum() == len(series):
        return num.values.astype(float), None

    return np.arange(len(series), dtype=float), None


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if len(valid) < 5:
        return pd.Series(False, index=series.index)

    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (s < lo) | (s > hi)


def rank_wells(df: pd.DataFrame, well_col: str, production_col: str) -> pd.DataFrame:
    grp = df.groupby(well_col)[production_col].agg(["mean", "max", "min", "count"]).reset_index()
    grp.columns = [well_col, "Average_Production", "Max_Production", "Min_Production", "Records"]
    grp = grp.sort_values("Average_Production", ascending=False).reset_index(drop=True)
    grp["Rank"] = np.arange(1, len(grp) + 1)
    return grp


def rank_fields(df: pd.DataFrame, field_col: str, production_col: str) -> pd.DataFrame:
    grp = df.groupby(field_col)[production_col].agg(["mean", "max", "min", "count"]).reset_index()
    grp.columns = [field_col, "Average_Production", "Max_Production", "Min_Production", "Records"]
    grp = grp.sort_values("Average_Production", ascending=False).reset_index(drop=True)
    grp["Rank"] = np.arange(1, len(grp) + 1)
    return grp


def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    return pd.DataFrame({
        "Column": missing.index,
        "Missing_Values": missing.values,
        "Missing_Percentage": (missing.values / len(df)) * 100
    }).sort_values("Missing_Values", ascending=False)


def unique_analysis_cols(mapping: dict) -> List[str]:
    vals = []
    for key in ["production", "pressure", "water_cut", "gor"]:
        v = mapping.get(key)
        if v and v not in vals:
            vals.append(v)
    return vals


def correlation_matrix_safe(df: pd.DataFrame, mapping: dict) -> Optional[pd.DataFrame]:
    cols = unique_analysis_cols(mapping)
    if len(cols) < 2:
        return None
    corr_df = df[cols].apply(pd.to_numeric, errors="coerce")
    return corr_df.corr()

# =========================================================
# DECLINE / FORECAST
# =========================================================
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


def calculate_eur(df: pd.DataFrame, mapping: dict, decline_results: dict, best_decline_name: str) -> Optional[float]:
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


def simple_forecast(df: pd.DataFrame, mapping: dict, target_key: str, horizon: int = 30) -> Optional[dict]:
    time_col = mapping.get("time")
    target_col = mapping.get(target_key)
    if not time_col or not target_col:
        return None

    x_num, _ = time_to_numeric(df[time_col])
    y = pd.to_numeric(df[target_col], errors="coerce").values
    mask = np.isfinite(x_num) & np.isfinite(y)
    x = x_num[mask]
    y = y[mask]

    if len(x) < 10:
        return None

    X = x.reshape(-1, 1)

    # مقارنة بسيطة بين RF و GB
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    best_name = None
    best_model = None
    best_rmse = None
    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))
        rows.append({"Model": name, "RMSE": rmse, "R2": r2})
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model

    future_x = np.arange(x[-1] + 1, x[-1] + horizon + 1).reshape(-1, 1)
    future_pred = best_model.predict(future_x)

    if target_key in ["production", "pressure", "gor"]:
        future_pred = np.maximum(future_pred, 0)
    if target_key == "water_cut":
        future_pred = np.clip(future_pred, 0, 0.99)

    return {
        "results_df": pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True),
        "best_model_name": best_name,
        "historical_x": x,
        "historical_y": y,
        "future_x": future_x.flatten(),
        "future_pred": future_pred
    }

# =========================================================
# ENGINEERING AI
# =========================================================
def calculate_risk_scores(df: pd.DataFrame, mapping: dict) -> dict:
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


def estimate_drive_mechanism(df: pd.DataFrame, mapping: dict) -> str:
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


def calculate_reservoir_health_score(df: pd.DataFrame, mapping: dict, risks: dict) -> float:
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


def calculate_production_efficiency_score(df: pd.DataFrame, mapping: dict) -> float:
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


def classify_well(df: pd.DataFrame, mapping: dict, risks: dict) -> str:
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

# =========================================================
# ADVANCED AI
# =========================================================
def advanced_ai_analysis(df: pd.DataFrame, mapping: dict) -> dict:
    results = {
        "ok": False,
        "decline": "Unknown",
        "slope": 0.0,
        "anomalies": 0,
        "water": "Unknown",
        "gas": "Unknown",
        "pressure": "Unknown",
        "score": 0,
        "report": "No report generated."
    }

    prod_col = mapping.get("production")
    press_col = mapping.get("pressure")
    wc_col = mapping.get("water_cut")
    gor_col = mapping.get("gor")

    required = [prod_col, press_col, wc_col, gor_col]
    if any(col is None for col in required):
        results["report"] = "Advanced AI needs Production, Pressure, Water Cut, and GOR columns."
        return results

    work = df.copy()

    for col in [prod_col, press_col, wc_col, gor_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work[[prod_col, press_col, wc_col, gor_col]].dropna().reset_index(drop=True)

    if len(work) < 10:
        results["report"] = "Not enough clean rows for Advanced AI analysis."
        return results

    production = work[prod_col].values.astype(float)
    pressure = work[press_col].values.astype(float)
    wc = work[wc_col].values.astype(float)
    gor = work[gor_col].values.astype(float)

    x = np.arange(len(production)).reshape(-1, 1)

    try:
        model = LinearRegression()
        model.fit(x, production)
        slope = float(model.coef_[0])
    except Exception:
        slope = 0.0

    if slope < -5:
        decline_type = "Strong decline"
    elif slope < -1:
        decline_type = "Moderate decline"
    else:
        decline_type = "Stable / Mild decline"

    try:
        iso = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso.fit_predict(production.reshape(-1, 1))
        anomaly_count = int((anomalies == -1).sum())
    except Exception:
        anomaly_count = 0

    try:
        wc_trend = float(np.polyfit(np.arange(len(wc)), wc, 1)[0])
    except Exception:
        wc_trend = 0.0

    if wc_trend > 0.01:
        water_risk = "High water breakthrough risk"
    elif wc_trend > 0.003:
        water_risk = "Moderate water risk"
    else:
        water_risk = "Normal"

    try:
        gor_trend = float(np.polyfit(np.arange(len(gor)), gor, 1)[0])
    except Exception:
        gor_trend = 0.0

    if gor_trend > 5:
        gas_risk = "Gas breakthrough likely"
    elif gor_trend > 1:
        gas_risk = "Moderate gas increase"
    else:
        gas_risk = "Stable"

    pressure_drop = float(pressure[0] - pressure[-1])

    if pressure_drop > 500:
        pressure_status = "Severe pressure depletion"
    elif pressure_drop > 150:
        pressure_status = "Moderate pressure depletion"
    else:
        pressure_status = "Normal pressure behavior"

    score = 100

    if slope < -5:
        score -= 25
    elif slope < -1:
        score -= 12

    if anomaly_count > 5:
        score -= 15
    elif anomaly_count > 0:
        score -= 8

    if wc_trend > 0.01:
        score -= 20
    elif wc_trend > 0.003:
        score -= 10

    if gor_trend > 5:
        score -= 15
    elif gor_trend > 1:
        score -= 8

    if pressure_drop > 500:
        score -= 20
    elif pressure_drop > 150:
        score -= 10

    score = max(min(int(score), 100), 0)

    report = f"""
AI RESERVOIR REPORT

- Decline Trend: {decline_type}
- Production Slope: {slope:.3f}
- Anomalies Detected: {anomaly_count}
- Water Behavior: {water_risk}
- Gas Behavior: {gas_risk}
- Pressure Status: {pressure_status}

FINAL AI SCORE: {score}/100
"""

    results.update({
        "ok": True,
        "decline": decline_type,
        "slope": slope,
        "anomalies": anomaly_count,
        "water": water_risk,
        "gas": gas_risk,
        "pressure": pressure_status,
        "score": score,
        "report": report
    })
    return results


def build_advanced_ai_metrics_df(ai_result: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"Metric": "Decline Trend", "Value": ai_result["decline"]},
        {"Metric": "Production Slope", "Value": round(ai_result["slope"], 3)},
        {"Metric": "Anomalies", "Value": ai_result["anomalies"]},
        {"Metric": "Water Behavior", "Value": ai_result["water"]},
        {"Metric": "Gas Behavior", "Value": ai_result["gas"]},
        {"Metric": "Pressure Status", "Value": ai_result["pressure"]},
        {"Metric": "Final AI Score", "Value": ai_result["score"]},
    ])

# =========================================================
# REPORT / PDF
# =========================================================
def generate_report_text(
    df: pd.DataFrame,
    mapping: dict,
    selected_well: Optional[str],
    ai_basic: dict,
    ai_advanced: dict,
    forecast_pack: Optional[dict],
    best_decline_name: Optional[str],
    decline_params_text: str,
    eur_value: Optional[float]
) -> str:
    lines = []
    lines.append("PETROSCOPE ENGINEERING REPORT")
    lines.append("=" * 64)
    lines.append(f"Rows analyzed: {len(df)}")
    if selected_well is not None:
        lines.append(f"Selected well: {selected_well}")

    lines.append(f"Well status: {ai_basic['well_class']}")
    lines.append(f"Estimated drive mechanism: {ai_basic['drive_mechanism']}")
    lines.append(f"Reservoir health score: {ai_basic['health_score']}/100")
    lines.append(f"Production efficiency score: {ai_basic['efficiency_score']}/100")
    lines.append(f"Advanced AI score: {ai_advanced['score']}/100")

    if best_decline_name:
        lines.append(f"Best decline model: {best_decline_name}")
    if decline_params_text:
        lines.append(f"Decline parameters: {decline_params_text}")
    if eur_value is not None:
        lines.append(f"Estimated EUR: {eur_value}")
    if forecast_pack:
        lines.append(f"Forecast model: {forecast_pack['best_model_name']}")

    lines.append("")
    lines.append("Mapped Columns:")
    for k, v in mapping.items():
        if v:
            lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("Basic Risks:")
    lines.append(f"- Water risk: {ai_basic['risks']['water_risk']}/100")
    lines.append(f"- Gas risk: {ai_basic['risks']['gas_risk']}/100")
    lines.append(f"- Depletion risk: {ai_basic['risks']['depletion_risk']}/100")
    lines.append(f"- Anomaly risk: {ai_basic['risks']['anomaly_risk']}/100")

    lines.append("")
    lines.append("Advanced AI:")
    lines.append(ai_advanced["report"].strip())

    return "\n".join(lines)


def create_pdf_report(
    ai_basic: dict,
    ai_advanced: dict,
    best_decline_name: Optional[str],
    decline_params_text: str,
    eur_value: Optional[float],
    forecast_pack: Optional[dict]
) -> str:
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

    draw_line("PetroScope Engineering Report", font="Helvetica-Bold", size=16, step=24)
    draw_line(f"Well status: {ai_basic['well_class']}", size=11)
    draw_line(f"Drive mechanism: {ai_basic['drive_mechanism']}", size=11)
    draw_line(f"Reservoir health score: {ai_basic['health_score']}/100", size=11)
    draw_line(f"Production efficiency score: {ai_basic['efficiency_score']}/100", size=11)
    draw_line(f"Advanced AI score: {ai_advanced['score']}/100", size=11)
    draw_line(f"Best decline model: {best_decline_name if best_decline_name else '-'}", size=11)
    draw_line(f"Decline parameters: {decline_params_text if decline_params_text else '-'}", size=11)
    draw_line(f"EUR: {eur_value if eur_value is not None else '-'}", size=11)
    draw_line(f"Forecast model: {forecast_pack['best_model_name'] if forecast_pack else '-'}", size=11, step=22)

    draw_line("Advanced AI Summary", font="Helvetica-Bold", size=12, step=18)
    for line in ai_advanced["report"].strip().split("\n"):
        if line.strip():
            draw_line(line.strip())

    c.save()
    return temp.name

# =========================================================
# HEADER + SIDEBAR
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
                <span class="badge">PetroScope Workspace</span>
                <div class="title">PetroScope</div>
                <p class="subtitle">
                    A practical workspace for reviewing well performance, checking reservoir behavior,
                    running advanced AI screening, forecasting, and exporting engineering reports.
                </p>
            </div>
            <div class="clean-box" style="min-width:260px;">
                <div class="small"><b>Current use</b></div>
                <div class="small">Well review • Forecast • Advanced AI • Reports</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section">
    <span class="status-chip">Workspace</span>
    <span class="status-chip">Review</span>
    <span class="status-chip">Advanced AI</span>
    <span class="status-chip">Forecast</span>
    <span class="status-chip">Reports</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### My Projects")
projects_list = list_projects(st.session_state.user)
if projects_list:
    selected_project_name = st.sidebar.selectbox("Saved projects", [p["name"] for p in projects_list])
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
# KPI
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
    <p class="small">Used to review production trends, pressure behavior, water cut, GOR, and well performance from structured field data.</p>
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
    inspect advanced AI → forecast → save project → export report.
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

# Basic AI bundle
risks = calculate_risk_scores(df, mapping)
ai_basic = {
    "risks": risks,
    "health_score": calculate_reservoir_health_score(df, mapping, risks),
    "efficiency_score": calculate_production_efficiency_score(df, mapping),
    "well_class": classify_well(df, mapping, risks),
    "drive_mechanism": estimate_drive_mechanism(df, mapping),
}

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
                    name=str(well_name)
                ))
        else:
            fig_prod.add_trace(go.Scatter(
                x=df[mapping["time"]],
                y=df[mapping["production"]],
                mode="lines+markers",
                name="Production"
            ))
        fig_prod.update_layout(template="plotly_dark", title="Production vs Time")
        st.plotly_chart(fig_prod, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if mapping["pressure"]:
            fig_press = go.Figure()
            fig_press.add_trace(go.Scatter(
                x=df[mapping["time"]],
                y=df[mapping["pressure"]],
                mode="lines+markers",
                name="Pressure"
            ))
            fig_press.update_layout(template="plotly_dark", title="Pressure vs Time")
            st.plotly_chart(fig_press, use_container_width=True)

    with c2:
        if mapping["water_cut"]:
            fig_wc = go.Figure()
            fig_wc.add_trace(go.Scatter(
                x=df[mapping["time"]],
                y=df[mapping["water_cut"]],
                mode="lines+markers",
                name="Water Cut"
            ))
            fig_wc.update_layout(template="plotly_dark", title="Water Cut vs Time")
            st.plotly_chart(fig_wc, use_container_width=True)

    if mapping["gor"]:
        fig_gor = go.Figure()
        fig_gor.add_trace(go.Scatter(
            x=df[mapping["time"]],
            y=df[mapping["gor"]],
            mode="lines+markers",
            name="GOR"
        ))
        fig_gor.update_layout(template="plotly_dark", title="GOR vs Time")
        st.plotly_chart(fig_gor, use_container_width=True)

    if mapping["well"] and mapping["production"] and selected_well == "All":
        st.markdown("---")
        st.subheader("Well Ranking")
        st.dataframe(rank_wells(df, mapping["well"], mapping["production"]), use_container_width=True)

    if mapping["field"] and mapping["production"]:
        st.markdown("---")
        st.subheader("Field Summary")
        st.dataframe(rank_fields(df, mapping["field"], mapping["production"]), use_container_width=True)

    if mapping["production"]:
        st.markdown("---")
        st.subheader("Anomaly Check")
        anom_mask = detect_outliers_iqr(df[mapping["production"]])
        temp = df.copy()
        temp["_anomaly"] = np.where(anom_mask.fillna(False), "Possible Outlier", "Normal")

        fig_anom = go.Figure()
        normal = temp[temp["_anomaly"] == "Normal"]
        out = temp[temp["_anomaly"] == "Possible Outlier"]

        fig_anom.add_trace(go.Scatter(
            x=normal[mapping["time"]],
            y=normal[mapping["production"]],
            mode="markers",
            name="Normal"
        ))
        fig_anom.add_trace(go.Scatter(
            x=out[mapping["time"]],
            y=out[mapping["production"]],
            mode="markers",
            name="Possible Outlier"
        ))
        fig_anom.update_layout(template="plotly_dark", title="Production Outlier Check")
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
        fig_corr.update_layout(template="plotly_dark", title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Advanced AI")

    ai_res = advanced_ai_analysis(df, mapping)

    if not ai_res["ok"]:
        st.warning(ai_res["report"])
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("AI Score", f"{ai_res['score']}/100")
        with m2:
            st.metric("Anomalies", ai_res["anomalies"])
        with m3:
            st.metric("Decline", ai_res["decline"])
        with m4:
            st.metric("Pressure", ai_res["pressure"])

        st.markdown("---")
        st.subheader("AI Summary Table")
        st.dataframe(build_advanced_ai_metrics_df(ai_res), use_container_width=True)

        st.markdown("---")
        st.subheader("AI Reservoir Report")
        st.text_area("AI Report", ai_res["report"], height=260)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
<div class="insight-box">
    <h4 style="margin-top:0;">Water Analysis</h4>
</div>
""", unsafe_allow_html=True)
            st.write(ai_res["water"])

        with c2:
            st.markdown("""
<div class="insight-box">
    <h4 style="margin-top:0;">Gas Analysis</h4>
</div>
""", unsafe_allow_html=True)
            st.write(ai_res["gas"])

    st.markdown('</div>', unsafe_allow_html=True)

with tab6:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Forecast")

    target_choice = st.selectbox("Target variable", ["production", "pressure", "water_cut", "gor"])
    horizon = st.slider("Forecast horizon", 10, 90, 30, 5)

    forecast_pack = simple_forecast(df, mapping, target_choice, horizon)

    if forecast_pack is None:
        st.warning("Not enough clean data for forecast.")
    else:
        st.success(f"Best model: {forecast_pack['best_model_name']}")
        st.dataframe(forecast_pack["results_df"], use_container_width=True)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=forecast_pack["historical_x"],
            y=forecast_pack["historical_y"],
            mode="lines+markers",
            name="Historical"
        ))
        fig_fc.add_trace(go.Scatter(
            x=forecast_pack["future_x"],
            y=forecast_pack["future_pred"],
            mode="lines+markers",
            name="Forecast"
        ))
        fig_fc.update_layout(template="plotly_dark", title=f"Forecast — {target_choice}")
        st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab7:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Export")

    forecast_export = simple_forecast(df, mapping, "production", 30)

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

    advanced_ai_res = advanced_ai_analysis(df, mapping)

    report_text = generate_report_text(
        df=df,
        mapping=mapping,
        selected_well=selected_well,
        ai_basic=ai_basic,
        ai_advanced=advanced_ai_res,
        forecast_pack=forecast_export,
        best_decline_name=best_decline_name,
        decline_params_text=decline_params_text,
        eur_value=eur_value
    )

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download report (.txt)",
            data=report_text.encode("utf-8"),
            file_name="petroscope_report.txt",
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
            ai_basic=ai_basic,
            ai_advanced=advanced_ai_res,
            best_decline_name=best_decline_name,
            decline_params_text=decline_params_text,
            eur_value=eur_value,
            forecast_pack=forecast_export
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF",
                f.read(),
                file_name="petroscope_report.pdf",
                mime="application/pdf"
            )

    st.markdown("### Advanced AI Export Summary")
    if advanced_ai_res["ok"]:
        st.text_area("Advanced AI Report", advanced_ai_res["report"], height=220)
    else:
        st.warning(advanced_ai_res["report"])

    api_payload = {
        "platform": "PetroScope",
        "selected_well": selected_well,
        "well_status": ai_basic["well_class"],
        "health_score": ai_basic["health_score"],
        "advanced_ai_score": advanced_ai_res["score"],
        "best_decline_model": best_decline_name,
        "forecast_model": forecast_export["best_model_name"] if forecast_export else None,
    }
    st.text_area("API preview", json.dumps(api_payload, indent=2), height=220)

    st.markdown("""
<div class="footer-box">
    PetroScope • Engineering workspace for production data review
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
                preview_records = raw_df.head(10).to_dict(orient="records")
                adv = advanced_ai_analysis(df, mapping)
                payload = {
                    "note": project_note,
                    "source_mode": source,
                    "file_name": current_file_name,
                    "mapping": mapping,
                    "selected_well": selected_well,
                    "data_preview": preview_records,
                    "row_count": int(len(raw_df)),
                    "columns": list(raw_df.columns),
                    "advanced_ai_score": adv["score"],
                    "advanced_ai_decline": adv["decline"],
                }
                save_project(st.session_state.user, project_name, payload)
                st.session_state.loaded_project_name = safe_name(project_name)
                st.success("Project saved.")

    with c2:
        adv = advanced_ai_analysis(df, mapping)
        st.markdown("### Project Summary")
        st.markdown(f"""
<div class="clean-box">
    <p class="small">Current file: <b>{current_file_name}</b></p>
    <p class="small">Rows: <b>{len(raw_df)}</b></p>
    <p class="small">Advanced AI score: <b>{adv['score']}/100</b></p>
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
        loaded = st.session_state.loaded_project_data
        st.markdown("---")
        st.markdown("### Loaded Project Details")
        st.markdown(f"""
<div class="clean-box">
    <p class="small"><b>Name:</b> {loaded.get("project_name","-")}</p>
    <p class="small"><b>Saved at:</b> {loaded.get("saved_at","-")}</p>
    <p class="small"><b>File:</b> {loaded.get("file_name","-")}</p>
    <p class="small"><b>Rows:</b> {loaded.get("row_count","-")}</p>
    <p class="small"><b>Advanced AI score:</b> {loaded.get("advanced_ai_score","-")}</p>
    <p class="small"><b>Decline:</b> {loaded.get("advanced_ai_decline","-")}</p>
    <p class="small"><b>Note:</b> {loaded.get("note","-")}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
