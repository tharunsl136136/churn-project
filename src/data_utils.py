"""
data_utils.py
=============
Handles all raw-data loading, cleaning, and splitting for the
Telco Customer Churn dataset.

Design principles
-----------------
* Every cleaning decision is documented with a WHY comment so it can be
  cited directly in the report.
* No fitting happens here – this module is purely deterministic transforms
  so it is safe to call before the train/test split.
* All paths are resolved relative to this file so the project is portable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parents[1]   # churn_project/
DATA_DIR = _ROOT / "data"
RAW_CSV  = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TARGET_COL   = "Churn"

# Add-on service columns used in feature engineering and EDA
SERVICE_COLS: list[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


# ══════════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_raw(path: str | Path = RAW_CSV) -> pd.DataFrame:
    """
    Load the raw CSV and apply the minimum safe cleaning steps that must
    happen before ANY analysis (including EDA).

    Cleaning steps applied here
    ---------------------------
    1. TotalCharges blank strings → NaN → 0.0
       WHY: 11 rows have tenure=0 (brand-new customers who have never been
       billed). Coercing to NaN then filling with 0 is the correct domain
       interpretation – they have zero total charges so far.

    2. Churn Yes/No → 1/0
       WHY: sklearn and all downstream code expects a numeric target.

    3. customerID dropped
       WHY: unique identifier with zero predictive signal; keeping it would
       risk accidental leakage if a model memorised IDs.

    4. "No internet service" / "No phone service" → "No"
       WHY: these are redundant encodings of the same information already
       captured by InternetService / PhoneService columns. Collapsing them
       reduces cardinality and prevents the one-hot encoder from creating
       spurious extra columns.

    Parameters
    ----------
    path : path to the raw CSV (defaults to data/WA_Fn-UseC_-Telco-Customer-Churn.csv)

    Returns
    -------
    pd.DataFrame with cleaned columns, numeric target, no customerID.
    """
    df = pd.read_csv(path)

    # ── Step 1: Fix TotalCharges ───────────────────────────────────────────────
    # pd.to_numeric with errors='coerce' turns blank strings into NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # All NaN rows have tenure == 0 (verified in EDA notebook cell 2.1)
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # ── Step 2: Encode target ──────────────────────────────────────────────────
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    # ── Step 3: Drop ID column ─────────────────────────────────────────────────
    df = df.drop(columns=["customerID"])

    # ── Step 4: Normalise redundant categorical values ─────────────────────────
    # Columns that can contain "No internet service"
    internet_dependent_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in internet_dependent_cols:
        df[col] = df[col].replace("No internet service", "No")

    # Column that can contain "No phone service"
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Splitting
# ══════════════════════════════════════════════════════════════════════════════

def split_xy(
    df: pd.DataFrame,
    target: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X from target series y."""
    return df.drop(columns=[target]), df[target]


def make_splits(
    df: pd.DataFrame,
    val_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,    pd.Series,    pd.Series]:
    """
    Stratified 60 / 20 / 20 train / validation / test split.

    WHY stratified: the dataset has ~26.5% churn. Without stratification,
    random chance could produce folds with very different churn rates,
    making CV scores noisy and potentially misleading.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X, y = split_xy(df)

    # First cut: 60% train, 40% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=val_size + test_size,   # 0.40
        stratify=y,
        random_state=random_state,
    )

    # Second cut: split the 40% temp equally into val and test (50/50 → 20/20)
    relative_test = test_size / (val_size + test_size)   # 0.5
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
# EDA helpers
# ══════════════════════════════════════════════════════════════════════════════

def class_balance(y: pd.Series) -> dict:
    """
    Return a dict with class counts and churn rate.
    Used in EDA and in run_all.py progress output.
    """
    counts = y.value_counts()
    return {
        "n_no_churn": int(counts.get(0, 0)),
        "n_churn":    int(counts.get(1, 0)),
        "churn_rate": round(float(y.mean()) * 100, 2),
    }


def get_feature_types(
    df: pd.DataFrame,
) -> Tuple[list[str], list[str]]:
    """
    Return (numerical_cols, categorical_cols) for a feature DataFrame
    (i.e. after dropping the target).
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return num_cols, cat_cols


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Cramér's V association measure between two categorical series.
    Range [0, 1] where 0 = no association, 1 = perfect association.
    Used in EDA to quantify categorical feature–target relationships.

    Formula: V = sqrt(chi2 / (n * (min(r,k) - 1)))
    Reference: Cramér (1946), Mathematical Methods of Statistics.
    """
    from scipy.stats import chi2_contingency

    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    # Bias-corrected version (Bergsma & Wicher, 2013)
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2 / denom))


def tenure_bucket(tenure: pd.Series) -> pd.Series:
    """
    Bin tenure (months) into labelled bands for EDA visualisations.
    Bins: 0-12, 13-24, 25-48, 49-60, 60+
    These same bins are used in feature engineering (features.py).
    """
    bins   = [0, 12, 24, 48, 60, np.inf]
    labels = ["0-12", "13-24", "25-48", "49-60", "60+"]
    return pd.cut(tenure, bins=bins, labels=labels, include_lowest=True)
