"""
explain.py
==========
SHAP-based model explainability helpers for the churn prediction project.

Why SHAP? (cite in report)
--------------------------
Lundberg & Lee (2017) proved that SHAP (SHapley Additive exPlanations) values
are the UNIQUE additive feature attribution method satisfying three axioms:
  1. Local accuracy  : attributions sum to the model output
  2. Missingness     : absent features have zero attribution
  3. Consistency     : if a feature's contribution increases, its SHAP value
                       cannot decrease

This makes SHAP theoretically superior to:
  - Permutation importance (global only, ignores interactions)
  - LIME (local only, unstable, approximate)
  - Gain-based importance (biased toward high-cardinality features)

TreeExplainer is used for tree-based models because it computes exact SHAP
values in O(TLD^2) time (T=trees, L=leaves, D=depth) without sampling,
making it both fast and exact (Lundberg et al., 2020).

Functions
---------
get_shap_values()          – compute SHAP values for a tree pipeline
plot_shap_summary()        – beeswarm plot (global feature impact distribution)
plot_shap_bar()            – mean |SHAP| bar chart (global importance ranking)
plot_shap_dependence()     – dependence plot for one feature (+ interaction)
plot_shap_waterfall()      – single-prediction waterfall explanation
plot_shap_force()          – single-prediction force plot (HTML)
shap_importance_df()       – DataFrame of mean |SHAP| per feature
compare_importances()      – SHAP vs model-native vs permutation importance
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT   = Path(__file__).resolve().parents[1]
FIG_DIR = _ROOT / "reports" / "figures"
RES_DIR = _ROOT / "reports" / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# Transform helpers
# ══════════════════════════════════════════════════════════════════════════════

def _transform_X(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Pass X through all pipeline steps EXCEPT the final model.
    Returns (X_transformed_array, feature_names_list).

    WHY: SHAP TreeExplainer operates on the raw model input (post-preprocessing),
    not on the original DataFrame. We need both the array and the feature names
    to label the SHAP plots correctly.
    """
    # Steps: fe → prep → model  (indices 0, 1, 2)
    X_fe   = pipeline.named_steps["fe"].transform(X)
    X_prep = pipeline.named_steps["prep"].transform(X_fe)

    try:
        feature_names = list(pipeline.named_steps["prep"].get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(X_prep.shape[1])]

    # Clean up sklearn's verbose prefixes (e.g. "num__tenure" → "tenure")
    feature_names = [
        n.split("__")[-1] if "__" in n else n
        for n in feature_names
    ]

    return X_prep, feature_names


# ══════════════════════════════════════════════════════════════════════════════
# SHAP value computation
# ══════════════════════════════════════════════════════════════════════════════

def get_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
    sample_size: int | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Compute SHAP values for the positive class (churn = 1).

    Parameters
    ----------
    pipeline    : fitted sklearn Pipeline (fe → prep → model)
    X           : raw feature DataFrame (before any transformation)
    sample_size : if set, subsample X for speed (use None for full dataset)

    Returns
    -------
    shap_values  : np.ndarray of shape (n_samples, n_features)
    X_transformed: np.ndarray of shape (n_samples, n_features)
    feature_names: list[str]
    """
    if sample_size is not None and sample_size < len(X):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X = X.iloc[idx].reset_index(drop=True)

    X_prep, feature_names = _transform_X(pipeline, X)
    model = pipeline.named_steps["model"]

    # TreeExplainer: exact, fast, works for RF / XGB / LGBM / CatBoost
    explainer   = shap.TreeExplainer(model)
    shap_output = explainer.shap_values(X_prep)

    # For binary classifiers, shap_values is a list [class0, class1]
    if isinstance(shap_output, list):
        shap_vals = shap_output[1]   # class 1 = churn
    else:
        shap_vals = shap_output

    return shap_vals, X_prep, feature_names


def shap_importance_df(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Return a DataFrame of mean |SHAP| per feature, sorted descending.
    This is the global feature importance ranking.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Global plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_summary(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list[str],
    model_name: str,
    max_display: int = 20,
) -> Path:
    """
    Beeswarm summary plot.
    Each dot = one sample. X-axis = SHAP value (impact on churn probability).
    Color = feature value (red=high, blue=low).
    WHY: Shows both the direction and magnitude of each feature's effect,
    and the distribution of effects across all customers.
    """
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_df,
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Summary (Beeswarm) – {model_name}", fontweight="bold", pad=12)
    plt.tight_layout()
    path = FIG_DIR / f"21_shap_summary_{model_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_shap_bar(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list[str],
    model_name: str,
    max_display: int = 20,
) -> Path:
    """
    Bar chart of mean |SHAP| values — global feature importance ranking.
    WHY: Easier to read than the beeswarm for a quick importance ranking.
    Directly comparable to model-native and permutation importance.
    """
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values, X_df,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Feature Importance (mean |SHAP|) – {model_name}",
              fontweight="bold", pad=12)
    plt.tight_layout()
    path = FIG_DIR / f"22_shap_bar_{model_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list[str],
    feature: str,
    model_name: str,
    interaction_feature: str = "auto",
) -> Path:
    """
    Dependence plot for one feature.
    X-axis = feature value, Y-axis = SHAP value.
    Color = interaction feature (auto-selected by SHAP if 'auto').
    WHY: Reveals the non-linear relationship between a feature and churn
    probability, and shows interaction effects via the color dimension.
    """
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    if feature not in X_df.columns:
        raise ValueError(f"Feature '{feature}' not found. Available: {X_df.columns.tolist()}")

    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        feature, shap_values, X_df,
        interaction_index=interaction_feature,
        show=False,
    )
    plt.title(f"SHAP Dependence: {feature} – {model_name}", fontweight="bold")
    plt.tight_layout()
    safe_feat = feature.replace(" ", "_").replace("/", "_")
    path = FIG_DIR / f"23_shap_dep_{safe_feat}_{model_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Local (single-prediction) plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_waterfall(
    pipeline: Pipeline,
    X_single: pd.DataFrame,
    model_name: str,
    label: str = "sample",
) -> Path:
    """
    Waterfall plot for a single prediction.
    Shows how each feature pushes the prediction from the base value
    (average model output) to the final prediction.
    WHY: The most intuitive local explanation — directly answers
    'why did the model predict churn for THIS customer?'
    """
    X_prep, feature_names = _transform_X(pipeline, X_single)
    model     = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)

    shap_output = explainer.shap_values(X_prep)
    if isinstance(shap_output, list):
        sv   = shap_output[1][0]
        base = explainer.expected_value[1] if isinstance(
            explainer.expected_value, (list, np.ndarray)
        ) else explainer.expected_value
    else:
        sv   = shap_output[0]
        base = explainer.expected_value

    exp = shap.Explanation(
        values        = sv,
        base_values   = float(base),
        data          = X_prep[0],
        feature_names = feature_names,
    )

    plt.figure(figsize=(10, 5))
    shap.waterfall_plot(exp, max_display=15, show=False)
    plt.title(f"SHAP Waterfall – {model_name} ({label})",
              fontweight="bold", pad=12)
    plt.tight_layout()
    safe_label = label.replace(" ", "_")
    path = FIG_DIR / f"24_shap_waterfall_{model_name.replace(' ', '_')}_{safe_label}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Importance comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_importances(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    shap_values: np.ndarray,
    feature_names: list[str],
    model_name: str,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Compare three importance methods for the top_n features:
      1. SHAP mean |SHAP|          – theoretically grounded, interaction-aware
      2. Model-native importance   – gain-based (XGB/LGBM) or impurity (RF)
      3. Permutation importance    – model-agnostic, measures actual predictive
                                     contribution on held-out data

    WHY compare: Each method has different biases. Agreement across methods
    increases confidence in the importance ranking. Disagreements reveal
    potential issues (e.g., gain-based importance is biased toward high-
    cardinality features; permutation importance is sensitive to correlated
    features).

    Returns a merged DataFrame for the report.
    """
    model = pipeline.named_steps["model"]
    X_prep, _ = _transform_X(pipeline, X_val)

    # ── 1. SHAP importance ────────────────────────────────────────────────────
    shap_imp = shap_importance_df(shap_values, feature_names).set_index("feature")
    shap_imp.columns = ["shap_importance"]

    # ── 2. Model-native importance ────────────────────────────────────────────
    native_imp = None
    if hasattr(model, "feature_importances_"):
        imp_arr = model.feature_importances_
        native_imp = pd.DataFrame({
            "feature":          feature_names[:len(imp_arr)],
            "native_importance": imp_arr,
        }).set_index("feature")

    # ── 3. Permutation importance (on validation set) ─────────────────────────
    perm = permutation_importance(
        model, X_prep, y_val.values,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="average_precision",
        n_jobs=-1,
    )
    perm_imp = pd.DataFrame({
        "feature":              feature_names[:len(perm.importances_mean)],
        "permutation_importance": perm.importances_mean,
    }).set_index("feature")

    # ── Merge & normalise ─────────────────────────────────────────────────────
    merged = shap_imp.join(perm_imp, how="outer")
    if native_imp is not None:
        merged = merged.join(native_imp, how="outer")

    # Normalise each column to [0, 1] for fair visual comparison
    for col in merged.columns:
        col_max = merged[col].max()
        if col_max > 0:
            merged[col] = (merged[col] / col_max).round(4)

    merged = merged.sort_values("shap_importance", ascending=False).head(top_n)
    return merged


def plot_importance_comparison(
    importance_df: pd.DataFrame,
    model_name: str,
) -> Path:
    """Bar chart comparing normalised SHAP, native, and permutation importance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_df.plot(kind="barh", ax=ax, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Normalised Importance (0–1)")
    ax.set_title(f"Feature Importance Comparison – {model_name}", fontweight="bold")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    path = FIG_DIR / f"25_importance_comparison_{model_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path
