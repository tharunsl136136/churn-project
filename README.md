# Telco Customer Churn Prediction
**Advanced Machine Learning Assignment**

> **Dataset:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
> **Python:** 3.11 · **Best model:** LightGBM (Optuna-tuned) · **Primary metric:** PR-AUC

---

## Problem Statement

Customer churn — when a subscriber cancels their service — costs telecom companies
billions annually. Acquiring a new customer costs 5–25× more than retaining an existing
one (Harvard Business Review, 2014). This project builds a production-ready churn
prediction system to identify at-risk customers *before* they leave, enabling targeted
retention campaigns with a measurable business cost model.

**Target:** `Churn` (binary: 1 = churned, 0 = retained)  
**Churn rate:** 26.54% — class imbalance addressed via `class_weight='balanced'`
and `scale_pos_weight` (pos_weight ≈ 2.77)

---

## Setup

```bash
# 1. Clone / download the project
cd churn_project

# 2. Install all dependencies (Python 3.11 required)
py -3.11 -m pip install -r requirements.txt

# 3. Verify dataset is present
#    data/WA_Fn-UseC_-Telco-Customer-Churn.csv  ← already included
```

---

## How to Run

### Option A — Full pipeline in one command
```bash
py -3.11 run_all.py
```
Reproduces: data cleaning → 60/20/20 split → 7 default models → Optuna tuning (50 trials)
→ threshold optimisation → all plots → all CSVs → `models/best_pipeline.joblib`

### Option B — Step-by-step notebooks
```bash
py -3.11 -m jupyter notebook
```
Open notebooks in order:

| # | Notebook | Content |
|---|----------|---------|
| 01 | `01_eda.ipynb` | EDA, class balance, correlations, Cramér's V |
| 02 | `02_preprocessing_feature_engineering.ipynb` | 12 engineered features, preprocessors |
| 03 | `03_modeling_and_tuning.ipynb` | 8 models, Optuna 50 trials each |
| 04 | `04_evaluation_and_threshold.ipynb` | ROC/PR/lift/calibration, threshold, cost |
| 05 | `05_explainability_shap.ipynb` | SHAP beeswarm, dependence, waterfall |
| 06 | `06_error_analysis.ipynb` | FN/FP profiles, fairness, learning curve |

### Option C — Execute notebooks non-interactively
```bash
py -3.11 -m jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda_executed.ipynb
```

### Launch the dashboard
```bash
py -3.11 -m streamlit run app/dashboard.py
```
Opens at `http://localhost:8501` — 8 pages covering EDA, model comparison,
threshold analysis, SHAP explainability, single prediction, and batch scoring.

### Run tests
```bash
py -3.11 -m pytest tests/ -v
```
77 tests, all passing. Covers data loading, feature engineering, model pipelines,
evaluation metrics, and end-to-end integration.

---

## Project Structure

```
churn_project/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← raw dataset (7,043 rows)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_modeling_and_tuning.ipynb
│   ├── 04_evaluation_and_threshold.ipynb
│   ├── 05_explainability_shap.ipynb
│   └── 06_error_analysis.ipynb
├── src/
│   ├── data_utils.py      ← load_raw, make_splits, cramers_v, class_balance
│   ├── features.py        ← ChurnFeatureEngineer (12 features), build_preprocessor
│   ├── models.py          ← pipeline builders, tune_xgboost/tune_lightgbm (Optuna)
│   ├── evaluate.py        ← compute_metrics, threshold_sweep, cost_analysis, bootstrap_ci
│   └── explain.py         ← get_shap_values, plot_shap_*, compare_importances
├── models/
│   ├── best_pipeline.joblib      ← {pipeline, model_name, best_params, val_prauc, ...}
│   ├── all_pipelines.joblib
│   ├── pipeline_*.joblib         ← 9 individual model pipelines
│   ├── preprocessor_linear.joblib
│   ├── preprocessor_tree.joblib
│   └── split_indices.joblib
├── reports/
│   ├── figures/           ← 31 PNG plots at 300 dpi (drop directly into report)
│   └── results/           ← 21 CSV files (metrics, hyperparameters, SHAP, etc.)
├── app/
│   └── dashboard.py       ← 8-page Streamlit app
├── tests/
│   ├── conftest.py
│   ├── test_data_utils.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_evaluate.py
│   └── test_pipeline_integration.py
├── requirements.txt       ← pinned versions (Python 3.11)
├── run_all.py             ← one-command full pipeline reproduction
├── REPORT_NOTES.md        ← rubric → figure/table mapping + key numbers
└── README.md
```

---

## Models Compared

| Model | Val PR-AUC | Val ROC-AUC | Val F2 |
|-------|-----------|------------|--------|
| Dummy (most frequent) | 0.2654 | 0.5000 | 0.0000 |
| Decision Tree | 0.5860 | 0.8175 | 0.6774 |
| Stacking (XGB+LGBM→LR) | 0.5768 | 0.8048 | 0.6351 |
| LightGBM (default) | 0.5671 | 0.7996 | 0.5430 |
| XGBoost (default) | 0.5505 | 0.7861 | 0.5445 |
| Random Forest | 0.6101 | 0.8188 | 0.6192 |
| CatBoost | 0.6245 | 0.8249 | 0.6867 |
| Logistic Regression | 0.6508 | 0.8370 | 0.7122 |
| XGBoost Tuned | 0.6424 | 0.8375 | 0.7171 |
| **LightGBM Tuned ✓** | **0.6328** | **0.8342** | **0.7198** |
| LightGBM Tuned (opt. threshold t=0.32) | 0.6328 | 0.8342 | **0.7549** |

*All metrics on validation set (1,409 samples). See `reports/results/metrics_final.csv`.*

---

## Key Results — LightGBM Tuned (Validation Set)

| Metric | Default t=0.50 | Optimal t=0.32 |
|--------|---------------|----------------|
| ROC-AUC | 0.8342 | 0.8342 |
| PR-AUC | 0.6328 | 0.6328 |
| Precision | 0.5192 | 0.4497 |
| Recall | 0.7968 | **0.9091** |
| F1 | 0.6287 | 0.6018 |
| **F2 (β=2)** | 0.7198 | **0.7549** |
| MCC | 0.4764 | 0.4491 |
| Brier Score | 0.1664 | 0.1664 |

**Bootstrap 95% CI (1,000 resamples):**
- PR-AUC: 0.6328 [0.5866, 0.6839]
- ROC-AUC: 0.8342 [0.8127, 0.8561]

**McNemar test** (LightGBM Tuned vs Logistic Regression): χ²=38.84, p<0.001 → significantly different

---

## Engineered Features (12 total)

| Feature | Description | Justification |
|---------|-------------|---------------|
| `tenure_group` | Ordinal tenure band | Non-linear loyalty; new customers churn at ~48% |
| `num_services` | Add-on service count (0–6) | Strong inverse churn relationship |
| `has_protection_bundle` | Security/backup/device/support count | Ecosystem embeddedness |
| `has_streaming` | 1 if any streaming service | Entertainment value signal |
| `avg_monthly_spend` | TotalCharges / (tenure+1) | Spend rate independent of tenure |
| `charge_gap` | MonthlyCharges − avg_monthly_spend | Recent price increase signal |
| `is_month_to_month` | 1 if month-to-month contract | ~43% churn vs ~3% for 2-year |
| `is_autopay` | 1 if automatic payment | Reduces billing friction |
| `contract_internet` | Contract × InternetService interaction | Highest-risk segment flag |
| `monthly_charges_per_service` | MonthlyCharges / (num_services+1) | Price-value perception |
| `is_new_customer` | 1 if tenure ≤ 6 months | Critical early churn window |
| `is_family` | 1 if Partner=Yes OR Dependents=Yes | Higher switching costs |

---

## Top SHAP Features (LightGBM Tuned)

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------|
| 1 | `is_month_to_month` | 0.664 |
| 2 | `tenure` | 0.296 |
| 3 | `charge_gap` | 0.195 |
| 4 | `MonthlyCharges` | 0.178 |
| 5 | `PaymentMethod_Electronic check` | 0.177 |

---

## Cost Analysis

Cost model: FN = $500 (missed churner), FP = $50 (wasted offer)

| Threshold | FN | FP | Total Cost |
|-----------|----|----|-----------|
| 0.05 | 4 | 845 | $44,250 |
| **0.12** | **8** | **675** | **$36,250** ← minimum |
| 0.32 | 34 | 416 | $37,800 |
| 0.50 | 76 | 276 | $51,800 |

*Minimum cost threshold (t=0.12) differs from F2-optimal (t=0.32) — the business
should choose based on whether cost minimisation or recall maximisation is the priority.*

---

## Data Leakage Prevention

- All feature engineering is inside `ChurnFeatureEngineer` (stateless sklearn transformer)
- All preprocessing (scaling, encoding) is inside `build_preprocessor` (fitted only on train)
- Full pipelines: `ChurnFeatureEngineer → ColumnTransformer → Classifier`
- No statistics from validation/test data ever touch the training pipeline
- `RANDOM_STATE = 42` set in all stochastic operations

---

## Screenshots

> Add screenshots here after launching the dashboard.
> Suggested captures:
> - Page 1 (Overview): dataset metrics cards
> - Page 4 (Model Comparison): metrics table with highlighted best row
> - Page 5 (Threshold & Cost): threshold sweep + cost curve side by side
> - Page 6 (Explainability): SHAP beeswarm plot
> - Page 7 (Single Prediction): prediction result + SHAP waterfall

```
reports/figures/21_shap_summary_LightGBM_Tuned.png     ← SHAP beeswarm
reports/figures/15_roc_curves.png                       ← ROC all models
reports/figures/16_pr_curves.png                        ← PR all models
reports/figures/19_threshold_LightGBM_Tuned.png         ← threshold sweep
reports/figures/20_cost_curve.png                       ← business cost curve
reports/figures/25_importance_comparison_LightGBM_Tuned.png  ← importance comparison
```

---

## Literature References

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
2. Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*, 785–794.
4. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*, 30.
5. Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. *ICML*, 233–240.
6. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot. *PLOS ONE*, 10(3).
7. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
8. McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions. *Psychometrika*, 12(2), 153–157.
9. Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895–1923.
10. Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357.
