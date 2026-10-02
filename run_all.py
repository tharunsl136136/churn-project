"""
run_all.py
----------
Reproduces the full training pipeline end-to-end in one command.

Steps
-----
1. Load & clean data          (data_utils.load_raw)
2. 60/20/20 stratified split  (data_utils.make_splits)
3. Train all 7 default models (models.get_baseline_pipelines + get_advanced_pipelines)
4. Tune LightGBM with Optuna  (models.tune_lightgbm, 50 trials)
5. Evaluate best model        (evaluate.compute_metrics, threshold, cost)
6. Save all plots & CSVs      (evaluate.*)
7. Save pipeline artifact     (models/best_pipeline.joblib)

Usage
-----
    py -3.11 run_all.py
"""

import sys
import warnings
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from data_utils import load_raw, make_splits, class_balance
from models import (
    get_baseline_pipelines, get_advanced_pipelines,
    tune_lightgbm, build_tuned_pipeline, cv_score,
)
from evaluate import (
    compute_metrics, plot_roc_curves, plot_pr_curves,
    plot_confusion_matrix, plot_threshold_sweep,
    find_best_threshold, cost_analysis, save_metrics_table,
    bootstrap_ci,
)
from sklearn.metrics import roc_auc_score, average_precision_score

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

SEP = "=" * 60


def main():
    print(SEP)
    print("  Telco Customer Churn – Full Reproduction Pipeline")
    print(SEP)

    # ── 1. Load data ───────────────────────────────────────────────
    print("\n[1/7] Loading & cleaning data...")
    df = load_raw()
    bal = class_balance(df["Churn"])
    print(f"      Rows: {len(df)}  |  Churn rate: {bal['churn_rate']}%")

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df)
    pos_weight = round((y_train == 0).sum() / (y_train == 1).sum(), 4)
    print(f"      Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")
    print(f"      pos_weight = {pos_weight}")

    # ── 2. Train baseline models ───────────────────────────────────
    print("\n[2/7] Training baseline models...")
    baselines = get_baseline_pipelines()
    all_metrics, all_probs = {}, {}

    for name, pipe in baselines.items():
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_val)[:, 1]
        y_pred = pipe.predict(X_val)
        all_probs[name]   = y_prob
        all_metrics[name] = compute_metrics(y_val.values, y_pred, y_prob)
        print(f"      {name:25s}  PR-AUC={all_metrics[name]['PR_AUC']}")

    # ── 3. Train advanced models ───────────────────────────────────
    print("\n[3/7] Training advanced models...")
    advanced = get_advanced_pipelines(pos_weight=pos_weight)

    for name, pipe in advanced.items():
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_val)[:, 1]
        y_pred = pipe.predict(X_val)
        all_probs[name]   = y_prob
        all_metrics[name] = compute_metrics(y_val.values, y_pred, y_prob)
        print(f"      {name:25s}  PR-AUC={all_metrics[name]['PR_AUC']}")

    # ── 4. Tune LightGBM ──────────────────────────────────────────
    print("\n[4/7] Tuning LightGBM (Optuna, 50 trials)...")
    study = tune_lightgbm(X_train, y_train, n_trials=50)
    best_params = study.best_params
    print(f"      Best CV PR-AUC: {study.best_value:.4f}")
    print(f"      Best params: {best_params}")

    tuned_pipe = build_tuned_pipeline("LightGBM", best_params, pos_weight)
    tuned_pipe.fit(X_train, y_train)

    y_prob_val  = tuned_pipe.predict_proba(X_val)[:, 1]
    y_prob_test = tuned_pipe.predict_proba(X_test)[:, 1]
    y_pred_val  = tuned_pipe.predict(X_val)

    all_probs["LightGBM_Tuned"]   = y_prob_val
    all_metrics["LightGBM_Tuned"] = compute_metrics(y_val.values, y_pred_val, y_prob_val)
    print(f"      Val PR-AUC: {all_metrics['LightGBM_Tuned']['PR_AUC']}")

    # ── 5. Threshold optimisation & test evaluation ────────────────
    print("\n[5/7] Evaluating on test set...")
    best_t, best_f2 = find_best_threshold(y_val.values, y_prob_val, metric="f2")
    print(f"      Optimal F2 threshold (val): {best_t}  F2={best_f2}")

    y_pred_test_opt = (y_prob_test >= best_t).astype(int)
    test_metrics    = compute_metrics(y_test.values, y_pred_test_opt, y_prob_test)
    print(f"      Test ROC-AUC : {test_metrics['ROC_AUC']}")
    print(f"      Test PR-AUC  : {test_metrics['PR_AUC']}")
    print(f"      Test F1      : {test_metrics['F1']}")
    print(f"      Test F2      : {test_metrics['F2']}")

    # Bootstrap CI for PR-AUC
    _, lo, hi = bootstrap_ci(
        y_test.values, y_prob_test,
        metric_fn=average_precision_score,
        n_bootstrap=500,
    )
    print(f"      PR-AUC 95% CI: [{lo:.4f}, {hi:.4f}]")

    # ── 6. Save plots & CSVs ───────────────────────────────────────
    print("\n[6/7] Saving plots and metrics...")
    plot_roc_curves(all_probs, y_val.values)
    plot_pr_curves(all_probs,  y_val.values)
    plot_confusion_matrix(y_test.values, y_pred_test_opt, "LightGBM_Tuned_Optimal")
    plot_threshold_sweep(y_val.values, y_prob_val, "LightGBM_Tuned")
    cost_analysis(y_val.values, y_prob_val)

    all_metrics["LightGBM_Tuned_Test"] = test_metrics
    save_metrics_table(all_metrics, fname="metrics_run_all.csv")
    print("      Plots → reports/figures/")
    print("      Metrics → reports/results/metrics_run_all.csv")

    # ── 7. Save pipeline artifact ──────────────────────────────────
    print("\n[7/7] Saving pipeline artifact...")
    artifact = {
        "pipeline":      tuned_pipe,
        "model_name":    "LightGBM_Tuned",
        "best_params":   best_params,
        "opt_threshold": best_t,
        "val_prauc":     all_metrics["LightGBM_Tuned"]["PR_AUC"],
        "val_rocauc":    all_metrics["LightGBM_Tuned"]["ROC_AUC"],
        "test_metrics":  test_metrics,
        "pos_weight":    pos_weight,
    }
    out = MODELS_DIR / "best_pipeline.joblib"
    joblib.dump(artifact, out)
    print(f"      Saved: {out}")

    print(f"\n{SEP}")
    print("  ✅  Pipeline complete!")
    print(f"  ROC-AUC : {test_metrics['ROC_AUC']}")
    print(f"  PR-AUC  : {test_metrics['PR_AUC']}")
    print(f"  F1      : {test_metrics['F1']}")
    print(f"  F2      : {test_metrics['F2']}")
    print(f"  Threshold: {best_t}")
    print(SEP)
    print("\nNext steps:")
    print("  • Run notebooks 01–06 for full analysis")
    print("  • Launch dashboard: py -3.11 -m streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
