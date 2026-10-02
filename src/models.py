"""
models.py
=========
Defines every model pipeline and the Optuna-based hyperparameter tuning
function used in notebook 03.

Model catalogue
---------------
Baselines
  - DummyClassifier (most_frequent)   : sanity floor
  - LogisticRegression                : linear baseline
  - DecisionTreeClassifier            : single-tree baseline

Advanced
  - RandomForestClassifier            : bagging ensemble
  - XGBClassifier                     : gradient boosting (XGBoost)
  - LGBMClassifier                    : gradient boosting (LightGBM)
  - CatBoostClassifier                : gradient boosting (CatBoost)
  - StackingClassifier                : XGB + LGBM + LR meta-learner

Why gradient boosting for this problem (cite in report)
-------------------------------------------------------
1. Mixed data types: the dataset has both numeric and categorical features.
   Boosting trees handle both natively after OHE; they do not require
   feature scaling and are robust to outliers.
2. Non-linear interactions: churn is driven by combinations of features
   (e.g., month-to-month AND fiber optic). Boosting trees capture these
   interactions automatically through deep splits.
3. Imbalance: scale_pos_weight (XGBoost) and class_weight (LGBM) directly
   incorporate class imbalance into the loss function.
4. Empirical evidence: Chen & Guestrin (2016) and Ke et al. (2017) show
   gradient boosting consistently wins on tabular classification benchmarks.
5. Interpretability: SHAP TreeExplainer provides exact, fast explanations
   for any tree-based model (Lundberg & Lee, 2017).

Why alternatives are weaker
---------------------------
- Logistic Regression: assumes linear decision boundary; misses interactions
  without manual feature crosses.
- SVM: does not scale well to 7k rows with mixed types; no native probability
  calibration; slow to tune.
- Neural networks: overkill for 7k rows; prone to overfitting without large
  amounts of data; less interpretable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import optuna
from pathlib import Path
from typing import Any

from sklearn.dummy       import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree        import DecisionTreeClassifier
from sklearn.ensemble    import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from features import build_linear_pipeline, build_tree_pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parents[1]
MODELS_DIR = _ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_CV_FOLDS   = 5
SCORING      = "average_precision"   # PR-AUC — primary metric (see evaluate.py)


# ══════════════════════════════════════════════════════════════════════════════
# Model catalogue
# ══════════════════════════════════════════════════════════════════════════════

def get_baseline_pipelines() -> dict[str, Any]:
    """
    Return unfitted baseline pipelines.
    These set the performance floor that advanced models must beat.
    """
    return {
        "Dummy_MostFrequent": build_tree_pipeline(
            DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
        ),
        "LogisticRegression": build_linear_pipeline(
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )
        ),
        "DecisionTree": build_tree_pipeline(
            DecisionTreeClassifier(
                max_depth=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
    }


def get_advanced_pipelines(pos_weight: float = 2.77) -> dict[str, Any]:
    """
    Return unfitted advanced model pipelines.

    Parameters
    ----------
    pos_weight : float
        Ratio of negative to positive class samples (~5174/1869 ≈ 2.77).
        Used by XGBoost's scale_pos_weight parameter.
    """
    return {
        "RandomForest": build_tree_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "XGBoost": build_tree_pipeline(
            XGBClassifier(
                n_estimators=500,
                scale_pos_weight=pos_weight,   # handles imbalance in loss
                use_label_encoder=False,
                eval_metric="aucpr",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )
        ),
        "LightGBM": build_tree_pipeline(
            LGBMClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
        ),
        "CatBoost": build_tree_pipeline(
            CatBoostClassifier(
                iterations=500,
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=0,
            )
        ),
    }


def get_stacking_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pos_weight: float = 2.77,
) -> Any:
    """
    Build a StackingClassifier with XGBoost + LightGBM as base learners
    and Logistic Regression as the meta-learner.

    WHY stacking: base learners capture different aspects of the data
    (XGBoost is better at sharp boundaries; LGBM handles large feature
    spaces efficiently). The meta-learner learns how to combine their
    probability outputs optimally.

    The stacking pipeline wraps the FULL feature-engineering + preprocessing
    inside each base estimator so there is no leakage.
    """
    base_estimators = [
        ("xgb", build_tree_pipeline(
            XGBClassifier(
                n_estimators=300,
                scale_pos_weight=pos_weight,
                use_label_encoder=False,
                eval_metric="aucpr",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )
        )),
        ("lgbm", build_tree_pipeline(
            LGBMClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
        )),
    ]

    meta_learner = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    # passthrough=True passes original features to meta-learner alongside
    # base-learner predictions, giving it more information to work with.
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )
    return stacking


# ══════════════════════════════════════════════════════════════════════════════
# Optuna tuning
# ══════════════════════════════════════════════════════════════════════════════

def _make_cv() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    pos_weight: float = 2.77,
) -> optuna.Study:
    """
    Tune XGBoost with Optuna, optimising PR-AUC via 5-fold stratified CV.

    Search space documented here so it can be cited in the report:
      n_estimators      : [100, 800]   — number of boosting rounds
      max_depth         : [3, 8]       — tree depth (controls complexity)
      learning_rate     : [0.01, 0.3]  — shrinkage (log scale)
      subsample         : [0.6, 1.0]   — row sampling per tree
      colsample_bytree  : [0.5, 1.0]   — feature sampling per tree
      min_child_weight  : [1, 10]      — minimum sum of instance weight in leaf
      gamma             : [0, 5]       — minimum loss reduction for split
      reg_alpha         : [0, 2]       — L1 regularisation
      reg_lambda        : [1, 5]       — L2 regularisation
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1, 5),
            "scale_pos_weight": pos_weight,
            "use_label_encoder": False,
            "eval_metric":      "aucpr",
            "random_state":     RANDOM_STATE,
            "n_jobs":           -1,
            "verbosity":        0,
        }
        pipeline = build_tree_pipeline(XGBClassifier(**params))
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=_make_cv(), scoring=SCORING, n_jobs=-1,
        )
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_churn",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
) -> optuna.Study:
    """
    Tune LightGBM with Optuna, optimising PR-AUC via 5-fold stratified CV.

    Search space:
      n_estimators    : [100, 800]
      num_leaves      : [20, 150]   — controls model complexity
      learning_rate   : [0.01, 0.3] — log scale
      max_depth       : [-1, 10]    — -1 = unlimited
      min_child_samples: [10, 100]  — minimum data in leaf
      subsample       : [0.6, 1.0]
      colsample_bytree: [0.5, 1.0]
      reg_alpha       : [0, 2]
      reg_lambda      : [0, 5]
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", -1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0, 5),
            "class_weight":      "balanced",
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        pipeline = build_tree_pipeline(LGBMClassifier(**params))
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=_make_cv(), scoring=SCORING, n_jobs=-1,
        )
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name="lightgbm_churn",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def build_tuned_pipeline(model_name: str, best_params: dict,
                         pos_weight: float = 2.77) -> Any:
    """
    Reconstruct a full pipeline from Optuna best_params.
    Called after tuning to build the final model for evaluation.
    """
    if model_name == "XGBoost":
        model = XGBClassifier(
            **best_params,
            scale_pos_weight=pos_weight,
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        return build_tree_pipeline(model)

    elif model_name == "LightGBM":
        model = LGBMClassifier(
            **best_params,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        return build_tree_pipeline(model)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def cv_score(pipeline, X_train: pd.DataFrame,
             y_train: pd.Series) -> tuple[float, float]:
    """Return (mean, std) PR-AUC from 5-fold stratified CV."""
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=_make_cv(), scoring=SCORING, n_jobs=-1,
    )
    return round(float(scores.mean()), 4), round(float(scores.std()), 4)
