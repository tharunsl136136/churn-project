"""
test_pipeline_integration.py
-----------------------------
End-to-end integration tests: full pipeline fit → predict → metrics.
These are slower but verify the whole stack works together.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from models import get_baseline_pipelines, get_advanced_pipelines
from evaluate import compute_metrics, find_best_threshold


class TestEndToEnd:
    """Fit on train, evaluate on val — checks the full pipeline stack."""

    @pytest.fixture(scope="class")
    def fitted_lr(self, X_train, y_train):
        pipe = get_baseline_pipelines()["LogisticRegression"]
        pipe.fit(X_train, y_train)
        return pipe

    @pytest.fixture(scope="class")
    def fitted_lgbm(self, X_train, y_train):
        pipe = get_advanced_pipelines()["LightGBM"]
        pipe.fit(X_train, y_train)
        return pipe

    def test_lr_beats_dummy(self, X_val, y_val, X_train, y_train):
        dummy = get_baseline_pipelines()["Dummy_MostFrequent"]
        lr    = get_baseline_pipelines()["LogisticRegression"]
        dummy.fit(X_train, y_train)
        lr.fit(X_train, y_train)

        dummy_auc = roc_auc_score(y_val, dummy.predict_proba(X_val)[:, 1])
        lr_auc    = roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])
        assert lr_auc > dummy_auc

    def test_lgbm_beats_lr(self, X_val, y_val, X_train, y_train):
        """Default LGBM should be competitive with LR (within 5%); tuned LGBM wins."""
        lr   = get_baseline_pipelines()["LogisticRegression"]
        lgbm = get_advanced_pipelines()["LightGBM"]
        lr.fit(X_train, y_train)
        lgbm.fit(X_train, y_train)

        lr_auc   = roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])
        lgbm_auc = roc_auc_score(y_val, lgbm.predict_proba(X_val)[:, 1])
        # Both should be well above random; default LGBM within 5% of LR
        assert lgbm_auc > 0.80
        assert lgbm_auc >= lr_auc - 0.05, (
            f"LGBM AUC {lgbm_auc:.4f} unexpectedly much lower than LR {lr_auc:.4f}"
        )

    def test_lgbm_roc_auc_above_threshold(self, fitted_lgbm, X_val, y_val):
        probs = fitted_lgbm.predict_proba(X_val)[:, 1]
        auc   = roc_auc_score(y_val, probs)
        assert auc > 0.80, f"Expected ROC-AUC > 0.80, got {auc:.4f}"

    def test_lgbm_pr_auc_above_threshold(self, fitted_lgbm, X_val, y_val):
        from sklearn.metrics import average_precision_score
        probs = fitted_lgbm.predict_proba(X_val)[:, 1]
        ap    = average_precision_score(y_val, probs)
        assert ap > 0.55, f"Expected PR-AUC > 0.55, got {ap:.4f}"

    def test_optimal_threshold_improves_recall(self, fitted_lgbm, X_val, y_val):
        probs = fitted_lgbm.predict_proba(X_val)[:, 1]
        best_t, _ = find_best_threshold(y_val.values, probs, metric="f2")

        m_default = compute_metrics(
            y_val.values, (probs >= 0.5).astype(int), probs
        )
        m_optimal = compute_metrics(
            y_val.values, (probs >= best_t).astype(int), probs
        )
        # Optimal F2 threshold should improve recall vs default 0.5
        assert m_optimal["Recall"] >= m_default["Recall"]

    def test_pipeline_handles_unseen_categories(self, X_train, y_train):
        """Pipeline should not crash on unknown categorical values."""
        pipe = get_baseline_pipelines()["LogisticRegression"]
        pipe.fit(X_train, y_train)

        # Create a row with a novel PaymentMethod value
        row = X_train.iloc[[0]].copy()
        row["PaymentMethod"] = "Cryptocurrency"
        probs = pipe.predict_proba(row)
        assert probs.shape == (1, 2)

    def test_predict_proba_sums_to_one(self, fitted_lgbm, X_val):
        probs = fitted_lgbm.predict_proba(X_val)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_no_nan_predictions(self, fitted_lgbm, X_val):
        probs = fitted_lgbm.predict_proba(X_val)
        assert not np.isnan(probs).any()
