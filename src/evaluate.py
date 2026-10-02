"""
evaluate.py
===========
All evaluation utilities for the churn prediction project.

Covers
------
* compute_metrics()        – full metric suite for one model
* plot_roc_curves()        – overlay ROC for all models
* plot_pr_curves()         – overlay PR for all models
* plot_confusion_matrix()  – annotated heatmap
* plot_lift_gain()         – lift / cumulative gain chart
* plot_calibration()       – reliability diagram + ECE
* calibrate_model()        – Platt / isotonic calibration wrapper
* threshold_sweep()        – F1, F2, precision, recall vs threshold
* find_best_threshold()    – maximise F2 (recall-weighted)
* cost_analysis()          – expected business cost vs threshold
* bootstrap_ci()           – 95% CI for any metric via bootstrap
* mcnemar_test()           – paired statistical test between two models
* save_metrics_table()     – write comparison CSV

Metric justification (cite in report Section 5)
-----------------------------------------------
* Accuracy   : MISLEADING at 26.5% churn. A model predicting "No Churn"
               always achieves 73.5% accuracy with zero recall on churners.
* ROC-AUC    : Threshold-independent ranking quality. Useful for comparing
               models but can be optimistic under class imbalance because
               the large TN pool inflates the FPR denominator.
* PR-AUC     : Area under Precision-Recall curve. Focuses entirely on the
               minority (churn) class. Preferred primary metric here.
               (Davis & Goadrich, 2006; Saito & Rehmsmeier, 2015)
* F1 (churn) : Harmonic mean of precision and recall for the churn class.
               Balanced view of false positives and false negatives.
* F2 (churn) : β=2 weights recall twice as heavily as precision.
               WHY: In a retention campaign, missing a churner (FN) costs
               ~$500 in lost CLV; a wasted offer (FP) costs ~$50. The
               asymmetric cost justifies prioritising recall.
* MCC        : Matthews Correlation Coefficient. Robust to class imbalance;
               considers all four quadrants of the confusion matrix.
               Range [-1, 1]; 0 = random, 1 = perfect.
* Brier score: Mean squared error of probability estimates. Measures
               calibration quality alongside discrimination.
* Log loss   : Cross-entropy loss. Penalises confident wrong predictions.
* Balanced Acc: Mean of sensitivity and specificity. Useful summary under
               imbalance when both classes matter equally.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, fbeta_score,
    log_loss, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT   = Path(__file__).resolve().parents[1]
FIG_DIR = _ROOT / "reports" / "figures"
RES_DIR = _ROOT / "reports" / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ── Plot helpers ───────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["savefig.dpi"] = 300

def _savefig(name: str) -> Path:
    path = FIG_DIR / name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Core metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Return a flat dict of all evaluation metrics for one model.
    All metrics are computed on the churn class (label = 1).
    """
    return {
        "Accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "Balanced_Acc":    round(balanced_accuracy_score(y_true, y_pred), 4),
        "Precision":       round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":          round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":              round(f1_score(y_true, y_pred, zero_division=0), 4),
        "F2":              round(fbeta_score(y_true, y_pred, beta=2, zero_division=0), 4),
        "ROC_AUC":         round(roc_auc_score(y_true, y_prob), 4),
        "PR_AUC":          round(average_precision_score(y_true, y_prob), 4),
        "MCC":             round(matthews_corrcoef(y_true, y_pred), 4),
        "Brier_Score":     round(brier_score_loss(y_true, y_prob), 4),
        "Log_Loss":        round(log_loss(y_true, y_prob), 4),
    }


def save_metrics_table(
    metrics_dict: dict[str, dict],
    fname: str = "metrics_comparison.csv",
) -> Path:
    """
    metrics_dict: {model_name: metrics_dict_from_compute_metrics}
    Saves a sorted comparison table to reports/results/.
    """
    df = pd.DataFrame(metrics_dict).T.sort_values("PR_AUC", ascending=False)
    path = RES_DIR / fname
    df.to_csv(path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Confusion matrix
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    fname: str | None = None,
) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred: No Churn", "Pred: Churn"],
        yticklabels=["True: No Churn", "True: Churn"],
        ax=ax, linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix – {model_name}\n"
                 f"TN={tn}  FP={fp}  FN={fn}  TP={tp}", fontsize=10)
    plt.tight_layout()
    fname = fname or f"cm_{model_name.replace(' ', '_')}.png"
    return _savefig(fname)


# ══════════════════════════════════════════════════════════════════════════════
# ROC & PR curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(
    models_probs: dict[str, np.ndarray],
    y_true: np.ndarray,
    fname: str = "15_roc_curves.png",
) -> Path:
    """Overlay ROC curves for all models on one axes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in models_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    return _savefig(fname)


def plot_pr_curves(
    models_probs: dict[str, np.ndarray],
    y_true: np.ndarray,
    fname: str = "16_pr_curves.png",
) -> Path:
    """Overlay Precision-Recall curves for all models."""
    baseline_precision = y_true.mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in models_probs.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, linewidth=1.5, label=f"{name} (AP={ap:.3f})")
    ax.axhline(baseline_precision, color="k", linestyle="--",
               linewidth=1, label=f"Random (AP={baseline_precision:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves – All Models", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    return _savefig(fname)


# ══════════════════════════════════════════════════════════════════════════════
# Lift / Gain chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_lift_gain(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    fname: str | None = None,
) -> Path:
    """
    Cumulative gain and lift chart.
    WHY: Shows how much better the model is at identifying churners compared
    to random targeting. A lift of 3 at the top 20% means the model finds
    3× more churners than random selection in that segment.
    """
    n = len(y_true)
    churn_rate = y_true.mean()

    # Sort by predicted probability descending
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]

    # Cumulative gain: fraction of all churners captured at each depth
    cum_churners = np.cumsum(y_sorted)
    total_churners = y_true.sum()
    gain = cum_churners / total_churners

    # Population fraction
    pop_frac = np.arange(1, n + 1) / n

    # Lift = gain / pop_frac
    lift = gain / pop_frac

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gain chart
    axes[0].plot(pop_frac * 100, gain * 100,
                 color="#D65F5F", linewidth=2, label=model_name)
    axes[0].plot([0, 100], [0, 100], "k--", linewidth=1, label="Random")
    axes[0].set_xlabel("% Population Contacted")
    axes[0].set_ylabel("% Churners Captured")
    axes[0].set_title("Cumulative Gain Chart", fontweight="bold")
    axes[0].legend()

    # Lift chart
    axes[1].plot(pop_frac * 100, lift,
                 color="#4878CF", linewidth=2, label=model_name)
    axes[1].axhline(1.0, color="k", linestyle="--", linewidth=1, label="Random")
    axes[1].set_xlabel("% Population Contacted")
    axes[1].set_ylabel("Lift")
    axes[1].set_title("Lift Chart", fontweight="bold")
    axes[1].legend()

    plt.suptitle(f"Lift & Gain – {model_name}", fontweight="bold")
    plt.tight_layout()
    fname = fname or f"17_lift_gain_{model_name.replace(' ', '_')}.png"
    return _savefig(fname)


# ══════════════════════════════════════════════════════════════════════════════
# Calibration
# ══════════════════════════════════════════════════════════════════════════════

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).
    Measures how well predicted probabilities match observed frequencies.
    ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|
    A well-calibrated model has ECE close to 0.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return round(float(ece), 4)


def calibrate_model(pipeline, X_train, y_train, method: str = "isotonic"):
    """
    Wrap a fitted pipeline in CalibratedClassifierCV.
    method: 'sigmoid' (Platt scaling) or 'isotonic'
    WHY: Raw boosting models often output overconfident probabilities.
    Calibration maps raw scores to better-calibrated probabilities,
    improving Brier score and log loss without changing ranking (ROC-AUC).
    """
    # cv=None means 'prefit' in sklearn 1.4+:
    # the estimator is already fitted; only the calibration layer is trained.
    cal = CalibratedClassifierCV(
        estimator=pipeline,
        method=method,
        cv=None,      # sklearn 1.4+: None = use prefit estimator
    )
    cal.fit(X_train, y_train)
    return cal


def plot_calibration_curve(
    models_probs: dict[str, np.ndarray],
    y_true: np.ndarray,
    fname: str = "18_calibration_curves.png",
) -> Path:
    """Reliability diagram for multiple models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    for name, y_prob in models_probs.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ece = expected_calibration_error(y_true, y_prob)
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.5,
                label=f"{name} (ECE={ece:.3f})")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration (Reliability) Curves", fontweight="bold")
    ax.legend(fontsize=7)
    plt.tight_layout()
    return _savefig(fname)


# ══════════════════════════════════════════════════════════════════════════════
# Threshold analysis
# ══════════════════════════════════════════════════════════════════════════════

def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """
    Compute precision, recall, F1, F2 at every threshold from 0.05 to 0.95.
    Returns a DataFrame indexed by threshold.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": round(t, 3),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
            "f2":        round(fbeta_score(y_true, y_pred, beta=2, zero_division=0), 4),
        })
    return pd.DataFrame(rows).set_index("threshold")


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f2",
) -> tuple[float, float]:
    """
    Return (best_threshold, best_score) that maximises the chosen metric.
    Default metric is F2 because recall is more valuable than precision
    in a churn retention context (missing a churner costs ~$500).
    """
    df = threshold_sweep(y_true, y_prob)
    best_t = float(df[metric].idxmax())
    best_s = float(df[metric].max())
    return round(best_t, 3), round(best_s, 4)


def plot_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    fname: str | None = None,
) -> Path:
    df = threshold_sweep(y_true, y_prob)
    best_t_f2 = float(df["f2"].idxmax())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df.index, df["precision"], label="Precision", linewidth=1.5)
    ax.plot(df.index, df["recall"],    label="Recall",    linewidth=1.5)
    ax.plot(df.index, df["f1"],        label="F1",        linewidth=1.5)
    ax.plot(df.index, df["f2"],        label="F2 (β=2)",  linewidth=2,
            color="#D65F5F")
    ax.axvline(best_t_f2, color="#D65F5F", linestyle="--", linewidth=1.2,
               label=f"Best F2 threshold = {best_t_f2:.2f}")
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=1,
               label="Default threshold = 0.50")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Sweep – {model_name}", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = fname or f"19_threshold_{model_name.replace(' ', '_')}.png"
    return _savefig(fname)


# ══════════════════════════════════════════════════════════════════════════════
# Business cost analysis
# ══════════════════════════════════════════════════════════════════════════════

def cost_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 500.0,   # cost of missing a churner (False Negative)
    cost_fp: float = 50.0,    # cost of a wasted retention offer (False Positive)
    fname: str = "20_cost_curve.png",
) -> pd.DataFrame:
    """
    Compute total expected business cost at each threshold.

    Cost assumptions (editable parameters, cite in report):
      FN cost = $500  : estimated customer lifetime value lost when a churner
                        is not identified and leaves without intervention.
      FP cost = $50   : cost of a retention offer (discount / call centre time)
                        sent to a customer who would not have churned anyway.

    Returns a DataFrame with columns: threshold, FN, FP, total_cost.
    Saves a cost curve plot.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "threshold":  round(t, 3),
            "TP": int(tp), "TN": int(tn),
            "FP": int(fp), "FN": int(fn),
            "total_cost": fn * cost_fn + fp * cost_fp,
        })
    df = pd.DataFrame(rows)

    best_t   = df.loc[df["total_cost"].idxmin(), "threshold"]
    best_cost = df["total_cost"].min()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["threshold"], df["total_cost"],
            color="#4878CF", linewidth=2, label="Total cost")
    ax.axvline(best_t, color="#D65F5F", linestyle="--", linewidth=1.5,
               label=f"Min cost @ t={best_t}  (${best_cost:,.0f})")
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=1,
               label="Default t=0.50")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel(f"Total Cost ($)  [FN=${cost_fn}, FP=${cost_fp}]")
    ax.set_title("Business Cost Curve", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _savefig(fname)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Statistical comparison
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """
    Bootstrap 95% confidence interval for a metric.

    Parameters
    ----------
    metric_fn : callable(y_true, y_prob) → float
    Returns (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_prob[idx])
            scores.append(s)
        except Exception:
            continue
    alpha = (1 - ci) / 2
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1 - alpha) * 100))
    point = metric_fn(y_true, y_prob)
    return round(point, 4), round(lower, 4), round(upper, 4)


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict:
    """
    McNemar's test for paired comparison of two classifiers.
    Tests H0: the two models make the same errors.
    A significant p-value (< 0.05) means the models differ significantly.

    Reference: McNemar (1947); Dietterich (1998) for ML context.
    """
    # Contingency table: rows = model A correct/wrong, cols = model B
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n_01 = int(( correct_a & ~correct_b).sum())  # A right, B wrong
    n_10 = int((~correct_a &  correct_b).sum())  # A wrong, B right

    # McNemar statistic with continuity correction
    if (n_01 + n_10) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_01": 0, "n_10": 0}

    statistic = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {
        "statistic": round(float(statistic), 4),
        "p_value":   round(float(p_value), 4),
        "n_01":      n_01,
        "n_10":      n_10,
        "significant_at_0.05": bool(p_value < 0.05),
    }
