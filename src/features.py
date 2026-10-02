"""
features.py
===========
All feature engineering implemented as sklearn-compatible TransformerMixin
classes so they slot cleanly into a Pipeline with NO data leakage.

Design principles
-----------------
* Every engineered feature has a documented WHY so it can be cited in the report.
* ChurnFeatureEngineer is stateless (no fit-time statistics) → safe to apply
  before or after the train/test split.
* A separate ColumnTransformerBuilder builds the full preprocessing pipeline
  (scaling + encoding) that MUST be fit only on training data.
* Two variants are provided:
    - build_linear_pipeline()  : OHE + StandardScaler  (for LR, SVM)
    - build_tree_pipeline()    : OHE without scaling    (for RF, XGB, LGBM, CatBoost)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

# ── Column lists (post feature-engineering) ────────────────────────────────────
# These are used by ColumnTransformerBuilder and by models.py
NUMERIC_COLS: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "num_services",
    "has_protection_bundle",
    "avg_monthly_spend",
    "charge_gap",
    "monthly_charges_per_service",
    # binary flags (numeric 0/1)
    "has_streaming",
    "is_month_to_month",
    "is_autopay",
    "is_new_customer",
    "is_family",
    "SeniorCitizen",
]

CATEGORICAL_COLS: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
    "contract_internet",          # interaction feature
]

# Contract is handled separately because it has a natural ordinal order
CONTRACT_COL: list[str] = ["Contract"]

# Add-on service columns in the raw data
_SERVICE_COLS: list[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

_PROTECTION_COLS: list[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
]

_STREAMING_COLS: list[str] = ["StreamingTV", "StreamingMovies"]

_AUTO_PAYMENT_METHODS: set[str] = {
    "Bank transfer (automatic)",
    "Credit card (automatic)",
}


# ══════════════════════════════════════════════════════════════════════════════
# Feature Engineering Transformer
# ══════════════════════════════════════════════════════════════════════════════

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds 11 domain-driven features to the raw feature DataFrame.

    This transformer is STATELESS (fit() is a no-op) so it is safe to
    apply before the train/test split or inside a Pipeline.

    Engineered features and their justification
    -------------------------------------------
    1.  tenure_group
        WHY: Churn risk is non-linear in tenure. New customers (0-12 months)
        churn at ~48%; long-tenured customers at <10%. Binning captures this
        non-linearity explicitly for linear models and improves tree splits.
        Bins: 0-12, 13-24, 25-48, 49-60, 60+

    2.  num_services
        WHY: EDA shows a strong inverse relationship between number of
        subscribed add-on services and churn rate. Customers with 0 services
        churn at ~50%; those with 5-6 services at <15%. A single count
        feature captures this more compactly than 6 binary columns.

    3.  has_protection_bundle
        WHY: Security/backup/device-protection/tech-support services form a
        "protection bundle". Customers with more of these are more embedded
        in the ecosystem. Count (0-4) captures partial adoption.

    4.  has_streaming
        WHY: Streaming services (TV + Movies) represent entertainment value.
        Customers who use streaming may churn for content reasons distinct
        from service quality. Binary flag separates this signal.

    5.  avg_monthly_spend  =  TotalCharges / (tenure + 1)
        WHY: TotalCharges is nearly perfectly correlated with tenure ×
        MonthlyCharges (r=0.997). avg_monthly_spend captures the *rate* of
        spending independently of how long the customer has been with the
        company, which is a better proxy for price sensitivity.

    6.  charge_gap  =  MonthlyCharges - avg_monthly_spend
        WHY: A positive charge_gap means the customer's current bill is
        higher than their historical average — a signal of a recent price
        increase or plan upgrade. Price increases are a known churn trigger.

    7.  is_month_to_month
        WHY: Month-to-month contract customers churn at ~43% vs ~3% for
        two-year contracts. A binary flag makes this the most important
        single feature explicit and easy for linear models to use.

    8.  is_autopay
        WHY: Automatic payment methods (bank transfer, credit card) reduce
        billing friction. Customers on autopay are less likely to notice
        their bill and decide to cancel. Binary flag captures this.

    9.  contract_internet  (interaction)
        WHY: The combination of contract type and internet service type
        captures a joint effect: month-to-month + fiber optic customers
        churn at the highest rate of any segment. Interaction features
        help linear models capture this without relying on non-linearity.

    10. monthly_charges_per_service  =  MonthlyCharges / (num_services + 1)
        WHY: Normalises monthly charges by the number of services received.
        A high value means the customer pays a lot relative to what they get,
        which is a price-value perception signal.

    11. is_new_customer  (tenure <= 6)
        WHY: The first 6 months are the highest-churn period. A binary flag
        explicitly marks this critical window for the model.

    12. is_family  (Partner == 'Yes' OR Dependents == 'Yes')
        WHY: Family customers have higher switching costs (multiple lines,
        shared plans). They tend to churn less. A single flag is more
        interpretable than two separate binary columns.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ChurnFeatureEngineer":
        # Stateless — nothing to learn from training data
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # ── 1. tenure_group ───────────────────────────────────────────────────
        bins   = [0, 12, 24, 48, 60, np.inf]
        labels = ["0-12", "13-24", "25-48", "49-60", "60+"]
        X["tenure_group"] = pd.cut(
            X["tenure"], bins=bins, labels=labels, include_lowest=True
        ).astype(str)

        # ── 2. num_services ───────────────────────────────────────────────────
        X["num_services"] = (
            X[_SERVICE_COLS].apply(lambda col: (col == "Yes").astype(int))
            .sum(axis=1)
        )

        # ── 3. has_protection_bundle ──────────────────────────────────────────
        X["has_protection_bundle"] = (
            X[_PROTECTION_COLS].apply(lambda col: (col == "Yes").astype(int))
            .sum(axis=1)
        )

        # ── 4. has_streaming ──────────────────────────────────────────────────
        X["has_streaming"] = (
            X[_STREAMING_COLS].apply(lambda col: (col == "Yes").astype(int))
            .max(axis=1)
        )

        # ── 5. avg_monthly_spend ──────────────────────────────────────────────
        X["avg_monthly_spend"] = (
            X["TotalCharges"] / (X["tenure"] + 1)
        ).round(4)

        # ── 6. charge_gap ─────────────────────────────────────────────────────
        X["charge_gap"] = (
            X["MonthlyCharges"] - X["avg_monthly_spend"]
        ).round(4)

        # ── 7. is_month_to_month ──────────────────────────────────────────────
        X["is_month_to_month"] = (
            X["Contract"] == "Month-to-month"
        ).astype(int)

        # ── 8. is_autopay ─────────────────────────────────────────────────────
        X["is_autopay"] = (
            X["PaymentMethod"].isin(_AUTO_PAYMENT_METHODS)
        ).astype(int)

        # ── 9. contract_internet interaction ──────────────────────────────────
        X["contract_internet"] = (
            X["Contract"].astype(str) + "_" + X["InternetService"].astype(str)
        )

        # ── 10. monthly_charges_per_service ───────────────────────────────────
        X["monthly_charges_per_service"] = (
            X["MonthlyCharges"] / (X["num_services"] + 1)
        ).round(4)

        # ── 11. is_new_customer ───────────────────────────────────────────────
        X["is_new_customer"] = (X["tenure"] <= 6).astype(int)

        # ── 12. is_family ─────────────────────────────────────────────────────
        X["is_family"] = (
            (X["Partner"] == "Yes") | (X["Dependents"] == "Yes")
        ).astype(int)

        return X


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing Pipeline Builders
# ══════════════════════════════════════════════════════════════════════════════

def _numeric_pipe(scale: bool = True) -> Pipeline:
    """Impute median → optionally StandardScale."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _categorical_pipe() -> Pipeline:
    """Impute most_frequent → OneHotEncode."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False,
                                  drop="if_binary")),
    ])


def _contract_pipe() -> Pipeline:
    """
    Ordinal encode Contract: Two year=0, One year=1, Month-to-month=2.
    WHY: Contract has a natural ordinal churn-risk ordering. Ordinal
    encoding preserves this order and avoids creating 3 OHE columns
    for a feature that is already captured by is_month_to_month.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=[["Two year", "One year", "Month-to-month"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles all column types.

    Parameters
    ----------
    scale : bool
        True  → include StandardScaler (use for Logistic Regression, SVM)
        False → skip scaling           (use for tree-based models)

    Returns
    -------
    sklearn ColumnTransformer (unfitted)
    """
    all_cat = CATEGORICAL_COLS + CONTRACT_COL
    return ColumnTransformer(
        transformers=[
            ("num",      _numeric_pipe(scale=scale),  NUMERIC_COLS),
            ("cat",      _categorical_pipe(),          CATEGORICAL_COLS),
            ("contract", _contract_pipe(),             CONTRACT_COL),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_linear_pipeline(model) -> Pipeline:
    """
    Full pipeline for linear models (LR, SVM):
      ChurnFeatureEngineer → ColumnTransformer (with scaling) → model
    """
    return Pipeline([
        ("fe",    ChurnFeatureEngineer()),
        ("prep",  build_preprocessor(scale=True)),
        ("model", model),
    ])


def build_tree_pipeline(model) -> Pipeline:
    """
    Full pipeline for tree-based models (RF, XGB, LGBM, CatBoost):
      ChurnFeatureEngineer → ColumnTransformer (no scaling) → model
    WHY no scaling: decision trees split on thresholds, not distances,
    so feature scale does not affect their performance.
    """
    return Pipeline([
        ("fe",    ChurnFeatureEngineer()),
        ("prep",  build_preprocessor(scale=False)),
        ("model", model),
    ])
