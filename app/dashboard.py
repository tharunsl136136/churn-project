"""
dashboard.py
------------
8-page Streamlit dashboard for the Telco Customer Churn Prediction project.

Pages
-----
1. 🏠 Overview          – project summary, dataset stats, churn rate
2. 📊 EDA               – key EDA figures from reports/figures/
3. ⚙️  Feature Engineering – engineered features table + distributions
4. 🤖 Model Comparison  – CV scores, ROC/PR curves, metrics table
5. 🎯 Threshold & Cost  – threshold sweep, cost curve, optimal threshold
6. 🔍 Explainability    – SHAP beeswarm, bar, dependence plots
7. 🔮 Single Prediction – live prediction with SHAP waterfall
8. 📂 Batch Prediction  – CSV upload, risk scoring, download

Run with:
    py -3.11 -m streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from PIL import Image

from data_utils import load_raw, class_balance
from features import ChurnFeatureEngineer
from evaluate import compute_metrics, find_best_threshold

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "reports" / "results"
MDL_DIR = ROOT / "models"


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifact():
    path = MDL_DIR / "best_pipeline.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner="Loading data…")
def load_data():
    return load_raw()


@st.cache_data(show_spinner="Loading splits…")
def load_splits():
    path = MDL_DIR / "split_indices.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def show_fig(fname: str, caption: str = "", width: int = None):
    path = FIG_DIR / fname
    if path.exists():
        img = Image.open(path)
        st.image(img, caption=caption, use_container_width=(width is None))
    else:
        st.warning(f"Figure not found: {fname}")


def load_csv(fname: str) -> pd.DataFrame | None:
    path = RES_DIR / fname
    return pd.read_csv(path) if path.exists() else None


def _transform_X(pipeline, X):
    X_fe   = pipeline.named_steps["fe"].transform(X)
    X_prep = pipeline.named_steps["prep"].transform(X_fe)
    try:
        names = list(pipeline.named_steps["prep"].get_feature_names_out())
        names = [n.split("__")[-1] if "__" in n else n for n in names]
    except Exception:
        names = [f"f{i}" for i in range(X_prep.shape[1])]
    return X_prep, names


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("📡 Churn Risk Studio")
st.sidebar.caption("TELCO CUSTOMER ANALYTICS")
st.sidebar.markdown("**IBM Telco dataset**  \n7,043 customer records")
st.sidebar.markdown("---")

artifact = load_artifact()
if artifact:
    st.sidebar.markdown("### Best Model")
    st.sidebar.markdown(f"**{artifact.get('model_name', 'LightGBM_Tuned')}**")
    st.sidebar.metric("Val PR-AUC",  f"{artifact.get('val_prauc',  0):.4f}")
    st.sidebar.metric("Val ROC-AUC", f"{artifact.get('val_rocauc', 0):.4f}")
    st.sidebar.metric("Opt. Threshold", f"{artifact.get('opt_threshold', 0.5):.3f}")
else:
    st.sidebar.warning("Model not loaded. Run notebooks first.")

st.sidebar.markdown("---")

# ── Top navigation ────────────────────────────────────────────────────────────
pages = [
    "🏠 Overview",
    "📊 EDA",
    "⚙️ Feature Engineering",
    "🤖 Model Comparison",
    "🎯 Threshold & Cost",
    "🔍 Explainability",
    "🔮 Single Prediction",
    "📂 Batch Prediction",
]

st.markdown("""
<style>
/* Global canvas and type */
[data-testid="stAppViewContainer"] {
    background: #f4f7fb;
}
[data-testid="stHeader"] {
    background: rgba(244, 247, 251, 0.92);
}
div.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #172b4d;
    letter-spacing: -0.025em;
}
h1 {
    font-weight: 750;
    margin-bottom: 0.4rem;
}
[data-testid="stMarkdownContainer"] p {
    color: #42526b;
}

/* Branded dashboard masthead */
.app-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 1.15rem 1.5rem;
    margin: 0.1rem 0 1rem 0;
    border-radius: 1rem;
    color: #fff;
    background: linear-gradient(115deg, #142b4d 0%, #2457a7 68%, #3478c7 100%);
    box-shadow: 0 8px 24px rgba(25, 55, 99, 0.15);
}
.app-banner-title {
    margin: 0;
    color: #fff !important;
    font-size: 1.45rem;
    font-weight: 750;
    letter-spacing: -0.02em;
}
.app-banner-subtitle {
    margin: 0.2rem 0 0 0;
    color: #dce9fb !important;
    font-size: 0.92rem;
}
.app-banner-tag {
    padding: 0.4rem 0.7rem;
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 999px;
    color: #fff;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    white-space: nowrap;
}

/* Spacious, high-contrast top navigation */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem 0.6rem;
    padding: 0.7rem 0.8rem;
    margin: 0 0 1.5rem 0;
    background: #fff;
    border: 1px solid #e1e8f0;
    border-radius: 0.9rem;
    box-shadow: 0 3px 12px rgba(30, 55, 90, 0.05);
}
div[data-testid="stRadio"] > div[role="radiogroup"] label {
    min-height: 2.75rem;
    padding: 0.62rem 1rem;
    border: 1px solid #d9e1eb;
    border-radius: 0.7rem;
    background: white;
    color: #263445 !important;
    opacity: 1 !important;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
}
div[data-testid="stRadio"] > div[role="radiogroup"] label:hover {
    background: #eef4fc;
    border-color: #9bb8df;
    transform: translateY(-1px);
}
div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {
    color: white !important;
    background: #214f93;
    border-color: #214f93;
    box-shadow: 0 3px 9px rgba(33, 79, 147, 0.2);
}
div[data-testid="stRadio"] > div[role="radiogroup"] label p {
    margin: 0;
    color: inherit !important;
    opacity: 1 !important;
}

/* Metric cards, tabs, tables, and action buttons */
[data-testid="stMetric"] {
    padding: 1rem 1.05rem;
    background: #fff;
    border: 1px solid #e2e9f1;
    border-radius: 0.9rem;
    box-shadow: 0 4px 14px rgba(30, 55, 90, 0.045);
}
[data-testid="stMetricLabel"] p {
    color: #64748b !important;
    font-size: 0.85rem;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    color: #172b4d;
    font-weight: 750;
}
button[data-baseweb="tab"] {
    color: #52647d;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #214f93;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #e2e9f1;
    border-radius: 0.8rem;
    overflow: hidden;
}
div.stButton > button, div[data-testid="stDownloadButton"] button {
    min-height: 2.7rem;
    border-radius: 0.65rem;
    font-weight: 650;
}
@media (max-width: 640px) {
    .app-banner { align-items: flex-start; flex-direction: column; }
    .app-banner-title { font-size: 1.2rem; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-banner">
  <div>
    <p class="app-banner-title">Churn Risk Studio</p>
    <p class="app-banner-subtitle">Explore customer trends, understand risk, and score customers.</p>
  </div>
  <div class="app-banner-tag">TELCO ANALYTICS</div>
</div>
""", unsafe_allow_html=True)

page = st.radio(
    "Navigate",
    pages,
    horizontal=True,
    label_visibility="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📡 Telco Customer Churn Prediction")
    st.markdown("""
    **Goal:** Identify customers likely to cancel their subscription *before* they leave,
    enabling targeted retention campaigns.

    **Dataset:** IBM Watson Telco Customer Churn (~7,043 rows, 20 features)  
    **Best model:** LightGBM (Optuna-tuned, 50 trials, 5-fold stratified CV)  
    **Primary metric:** PR-AUC (preferred over ROC-AUC for imbalanced classes)
    """)

    df = load_data()
    bal = class_balance(df["Churn"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churned",         f"{bal['n_churn']:,}")
    c3.metric("Retained",        f"{bal['n_no_churn']:,}")
    c4.metric("Churn Rate",      f"{bal['churn_rate']}%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        show_fig("01_class_balance.png", "Class Balance")
    with col2:
        show_fig("02_churn_rate_by_segment.png", "Churn Rate by Segment")

    st.markdown("### Pipeline Summary")
    st.markdown("""
    | Step | Notebook | Key Output |
    |------|----------|------------|
    | EDA | 01_eda | 7 figures, data issues CSV |
    | Feature Engineering | 02_preprocessing | 12 new features, preprocessors saved |
    | Modelling & Tuning | 03_modeling | 8 models, Optuna 50 trials each |
    | Evaluation | 04_evaluation | ROC/PR/lift/calibration, threshold, cost |
    | Explainability | 05_explainability | SHAP beeswarm, dependence, waterfall |
    | Error Analysis | 06_error_analysis | FN/FP profiles, fairness, learning curve |
    """)

    st.markdown("### Cost Assumptions")
    st.info("**FN cost = \\$500** (lost CLV when a churner is missed)  \n"
            "**FP cost = \\$50** (wasted retention offer)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Class & Segments", "Numeric Distributions", "Correlations", "Data Issues"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            show_fig("01_class_balance.png", "Class Balance")
            show_fig("03_churn_by_num_services.png", "Churn by # Services")
        with c2:
            show_fig("02_churn_rate_by_segment.png", "Churn Rate by Segment")
            show_fig("10_tenure_group_churn.png", "Churn by Tenure Group")

    with tab2:
        show_fig("04_numeric_distributions.png", "Numeric Feature Distributions")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            show_fig("05_correlation_heatmap.png", "Pearson Correlation Heatmap")
        with c2:
            show_fig("07_cramers_v.png", "Cramér's V (Categorical Associations)")
        show_fig("06_multicollinearity_check.png", "Multicollinearity Check")

    with tab4:
        issues = load_csv("data_issues.csv")
        if issues is not None:
            st.dataframe(issues, use_container_width=True)
        st.markdown("""
        **Key findings:**
        - 11 rows have `TotalCharges` as blank string (tenure = 0) → filled with 0.0
        - `TotalCharges` and `tenure × MonthlyCharges` have r = 0.997 (near-perfect multicollinearity)
        - 26.54% churn rate → class imbalance addressed via `class_weight='balanced'`
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Feature Engineering":
    st.title("⚙️ Feature Engineering")

    st.markdown("### 12 Engineered Features")
    fe_table = pd.DataFrame([
        ("tenure_group",                "Ordinal tenure band (0-12, 13-24, 25-48, 49-60, 60+)",
         "Non-linear loyalty effect; new customers churn at ~48%"),
        ("num_services",                "Count of add-on services subscribed (0–6)",
         "Strong inverse relationship with churn rate"),
        ("has_protection_bundle",       "Count of security/backup/device/support services (0–4)",
         "Ecosystem embeddedness proxy"),
        ("has_streaming",               "1 if subscribed to any streaming service",
         "Entertainment value; distinct churn signal"),
        ("avg_monthly_spend",           "TotalCharges / (tenure + 1)",
         "Spend rate independent of tenure; price sensitivity proxy"),
        ("charge_gap",                  "MonthlyCharges − avg_monthly_spend",
         "Positive = recent price increase; known churn trigger"),
        ("is_month_to_month",           "1 if Contract = Month-to-month",
         "Month-to-month customers churn at ~43% vs ~3% for 2-year"),
        ("is_autopay",                  "1 if automatic payment method",
         "Reduces billing friction → lower churn"),
        ("contract_internet",           "Contract × InternetService interaction string",
         "Month-to-month + Fiber optic = highest churn segment"),
        ("monthly_charges_per_service", "MonthlyCharges / (num_services + 1)",
         "Price-value perception signal"),
        ("is_new_customer",             "1 if tenure ≤ 6 months",
         "First 6 months = highest churn risk window"),
        ("is_family",                   "1 if Partner=Yes OR Dependents=Yes",
         "Higher switching costs → lower churn"),
    ], columns=["Feature", "Description", "Justification"])
    st.dataframe(fe_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        show_fig("08_engineered_features_churn.png", "Engineered Features vs Churn")
        show_fig("10_tenure_group_churn.png", "Tenure Group Churn Rate")
    with c2:
        show_fig("09_engineered_continuous_kde.png", "Continuous Feature KDEs")
        show_fig("11_contract_internet_interaction.png", "Contract × Internet Interaction")

    st.markdown("---")
    prep = load_csv("preprocessing_summary.csv")
    if prep is not None:
        st.markdown("### Preprocessing Summary")
        st.dataframe(prep, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")

    tab1, tab2, tab3 = st.tabs(["Metrics Table", "ROC / PR Curves", "Tuning"])

    with tab1:
        metrics = load_csv("metrics_final.csv")
        if metrics is None:
            metrics = load_csv("metrics_validation.csv")
        if metrics is not None:
            st.markdown("### Validation Set Metrics (all models)")
            # Highlight best PR-AUC row
            numeric_cols = metrics.select_dtypes(include="number").columns.tolist()
            st.dataframe(
                metrics.style.highlight_max(subset=["PR_AUC", "ROC_AUC", "F2"],
                                            color="#d4edda"),
                use_container_width=True,
            )
        else:
            st.info("Run notebook 04 to generate metrics CSVs.")

        cv = load_csv("cv_scores_all_models.csv")
        if cv is not None:
            st.markdown("### 5-Fold CV PR-AUC (training set)")
            st.dataframe(cv, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            show_fig("15_roc_curves.png", "ROC Curves – All Models")
        with c2:
            show_fig("16_pr_curves.png", "PR Curves – All Models")
        show_fig("17_lift_gain_LightGBM_Tuned.png", "Lift & Gain – LightGBM Tuned")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            show_fig("14_optuna_lightgbm.png", "Optuna – LightGBM (50 trials)")
        with c2:
            show_fig("14_optuna_xgboost.png", "Optuna – XGBoost (50 trials)")

        hp = load_csv("best_hyperparameters.csv")
        if hp is not None:
            st.markdown("### Best Hyperparameters")
            st.dataframe(hp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – Threshold & Cost
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Threshold & Cost":
    st.title("🎯 Threshold Optimisation & Cost Analysis")

    if artifact:
        t = artifact.get("opt_threshold", 0.5)
        st.info(f"**Optimal F2 threshold (validation set): {t:.3f}**  \n"
                f"This threshold maximises F2 (β=2), weighting recall 2× over precision.")

    c1, c2 = st.columns(2)
    with c1:
        show_fig("19_threshold_LightGBM_Tuned.png", "Threshold Sweep – LightGBM Tuned")
    with c2:
        show_fig("20_cost_curve.png", "Business Cost Curve")

    c1, c2 = st.columns(2)
    with c1:
        show_fig("15b_cm_LightGBM_Tuned.png", "Confusion Matrix – Default t=0.5")
    with c2:
        show_fig("19b_cm_LightGBM_Tuned_optimal.png", "Confusion Matrix – Optimal Threshold")

    st.markdown("---")
    tc = load_csv("threshold_comparison.csv")
    if tc is not None:
        st.markdown("### Threshold Comparison")
        st.dataframe(tc, use_container_width=True)

    ca = load_csv("cost_analysis.csv")
    if ca is not None:
        st.markdown("### Cost Analysis (sample)")
        st.dataframe(ca.head(20), use_container_width=True)

    st.markdown("""
    ### Cost Model
    | Error Type | Business Impact | Cost |
    |------------|----------------|------|
    | False Negative (missed churner) | Customer leaves; CLV lost | **\\$500** |
    | False Positive (wasted offer)   | Retention offer to loyal customer | **\\$50** |

    The optimal threshold minimises `FN × $500 + FP × $50`.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – Explainability
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Explainability":
    st.title("🔍 SHAP Explainability")

    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** — Lundberg & Lee (2017)  
    TreeExplainer computes exact SHAP values in O(TLD²) time.  
    Values represent each feature's contribution to the prediction relative to the base rate.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Global Summary", "Feature Importance", "Dependence Plots", "Importance Comparison"]
    )

    with tab1:
        show_fig("21_shap_summary_LightGBM_Tuned.png",
                 "SHAP Beeswarm – Each dot = one customer. Red = high feature value.")

    with tab2:
        show_fig("22_shap_bar_LightGBM_Tuned.png",
                 "Mean |SHAP| – Global feature importance ranking")
        imp = load_csv("shap_importance.csv")
        if imp is not None:
            st.dataframe(imp.head(15), use_container_width=True)

    with tab3:
        dep_features = [
            ("tenure",                          "tenure"),
            ("MonthlyCharges",                  "MonthlyCharges"),
            ("charge_gap",                      "charge_gap"),
            ("is_month_to_month",               "is_month_to_month"),
            ("PaymentMethod_Electronic_check",  "PaymentMethod_Electronic_check"),
        ]
        for fname_part, label in dep_features:
            show_fig(f"23_shap_dep_{fname_part}_LightGBM_Tuned.png",
                     f"Dependence: {label}")

    with tab4:
        show_fig("25_importance_comparison_LightGBM_Tuned.png",
                 "SHAP vs Native vs Permutation Importance (normalised)")
        ic = load_csv("importance_comparison.csv")
        if ic is not None:
            st.dataframe(ic, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 – Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Single Prediction":
    st.title("🔮 Single Customer Churn Prediction")

    if artifact is None:
        st.error("Model artifact not found. Run the notebooks first.")
        st.stop()

    pipeline  = artifact["pipeline"]
    threshold = artifact.get("opt_threshold", 0.5)

    st.markdown(f"Using optimal threshold **{threshold:.3f}** (F2-maximising).")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender         = st.selectbox("Gender",         ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner        = st.selectbox("Partner",        ["Yes", "No"])
        dependents     = st.selectbox("Dependents",     ["Yes", "No"])

    with col2:
        st.subheader("Services")
        phone_service    = st.selectbox("Phone Service",    ["Yes", "No"])
        multiple_lines   = st.selectbox("Multiple Lines",   ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security  = st.selectbox("Online Security",  ["Yes", "No"])
        online_backup    = st.selectbox("Online Backup",    ["Yes", "No"])
        device_prot      = st.selectbox("Device Protection",["Yes", "No"])
        tech_support     = st.selectbox("Tech Support",     ["Yes", "No"])
        streaming_tv     = st.selectbox("Streaming TV",     ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

    with col3:
        st.subheader("Account")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method  = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                          float(tenure * monthly_charges), step=1.0)

    if st.button("🚀 Predict Churn", type="primary"):
        input_df = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior_citizen,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_prot, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }])

        prob = pipeline.predict_proba(input_df)[0, 1]
        pred = int(prob >= threshold)

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Probability", f"{prob:.1%}")
        r2.metric("Prediction", "⚠️ Churn" if pred else "✅ No Churn")
        r3.metric("Threshold Used", f"{threshold:.3f}")

        if pred:
            st.error(f"⚠️ HIGH CHURN RISK — Probability: {prob:.1%}")
        else:
            st.success(f"✅ LOW CHURN RISK — Probability: {prob:.1%}")
        st.progress(float(prob))

        # SHAP waterfall
        st.markdown("### Why this prediction?")
        try:
            X_prep, feature_names = _transform_X(pipeline, input_df)
            model     = pipeline.named_steps["model"]
            explainer = shap.TreeExplainer(model)
            sv_out    = explainer.shap_values(X_prep)

            if isinstance(sv_out, list):
                sv   = sv_out[1][0]
                base = (explainer.expected_value[1]
                        if isinstance(explainer.expected_value, (list, np.ndarray))
                        else explainer.expected_value)
            else:
                sv   = sv_out[0]
                base = float(explainer.expected_value)

            exp = shap.Explanation(
                values        = sv,
                base_values   = float(base),
                data          = X_prep[0],
                feature_names = feature_names,
            )
            fig, _ = plt.subplots(figsize=(10, 5))
            shap.waterfall_plot(exp, max_display=15, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 – Batch Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Churn Prediction")

    if artifact is None:
        st.error("Model artifact not found. Run the notebooks first.")
        st.stop()

    pipeline  = artifact["pipeline"]
    threshold = artifact.get("opt_threshold", 0.5)

    st.markdown(f"""
    Upload a CSV with the same columns as the training data (without `Churn`).  
    Predictions use the optimal threshold **{threshold:.3f}**.
    """)

    st.markdown("**Required columns:** gender, SeniorCitizen, Partner, Dependents, tenure, "
                "PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, "
                "DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, "
                "PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        ids = batch_df.pop("customerID") if "customerID" in batch_df.columns \
              else pd.RangeIndex(len(batch_df))
        if "Churn" in batch_df.columns:
            batch_df = batch_df.drop(columns=["Churn"])

        batch_df["TotalCharges"] = pd.to_numeric(
            batch_df["TotalCharges"], errors="coerce"
        ).fillna(0)

        try:
            probs = pipeline.predict_proba(batch_df)[:, 1]
            preds = (probs >= threshold).astype(int)

            results = batch_df.copy()
            results.insert(0, "CustomerID", ids)
            results["ChurnProbability"] = probs.round(4)
            results["ChurnPrediction"]  = preds
            results["Risk"] = pd.cut(
                probs, bins=[0, 0.3, 0.6, 1.0],
                labels=["Low", "Medium", "High"],
            )

            st.success(f"Processed **{len(results):,}** customers.")

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total",       len(results))
            m2.metric("High Risk",   int((results["Risk"] == "High").sum()))
            m3.metric("Medium Risk", int((results["Risk"] == "Medium").sum()))
            m4.metric("Low Risk",    int((results["Risk"] == "Low").sum()))

            # Risk distribution chart
            fig, ax = plt.subplots(figsize=(5, 3))
            counts = results["Risk"].value_counts().reindex(["High", "Medium", "Low"])
            counts.plot(kind="bar", ax=ax, color=["#dc3545", "#fd7e14", "#28a745"],
                        edgecolor="white")
            ax.set_title("Risk Distribution")
            ax.set_ylabel("Count")
            ax.set_xlabel("")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Probability histogram
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.hist(probs, bins=30, color="#4878CF", edgecolor="white", alpha=0.8)
            ax2.axvline(threshold, color="red", linestyle="--",
                        label=f"Threshold = {threshold:.3f}")
            ax2.set_xlabel("Churn Probability")
            ax2.set_ylabel("Count")
            ax2.set_title("Predicted Probability Distribution")
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            # Preview table
            st.markdown("### Preview (first 50 rows)")
            st.dataframe(
                results[["CustomerID", "ChurnProbability", "ChurnPrediction", "Risk"]].head(50),
                use_container_width=True,
            )

            # Download
            csv_bytes = results.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Full Results CSV",
                csv_bytes,
                "churn_predictions.csv",
                "text/csv",
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)
