"""
test_evaluate.py
----------------
Tests for src/evaluate.py
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from evaluate import (
    compute_metrics,
    threshold_sweep,
    find_best_threshold,
    cost_analysis,
    bootstrap_ci,
    mcnemar_test,
    expected_calibration_error,
)


class TestComputeMetrics:
    def test_returns_eleven_keys(self, dummy_probs):
        y_true, y_pred, y_prob = dummy_probs
        m = compute_metrics(y_true, y_pred, y_prob)
        assert len(m) == 11

    def test_expected_keys(self, dummy_probs):
        y_true, y_pred, y_prob = dummy_probs
        m = compute_metrics(y_true, y_pred, y_prob)
        for key in ["Accuracy", "Balanced_Acc", "Precision", "Recall",
                    "F1", "F2", "ROC_AUC", "PR_AUC", "MCC",
                    "Brier_Score", "Log_Loss"]:
            assert key in m

    def test_values_in_range(self, dummy_probs):
        y_true, y_pred, y_prob = dummy_probs
        m = compute_metrics(y_true, y_pred, y_prob)
        for key in ["Accuracy", "Balanced_Acc", "Precision", "Recall",
                    "F1", "F2", "ROC_AUC", "PR_AUC"]:
            assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} out of [0,1]"

    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        m = compute_metrics(y_true, y_pred, y_prob)
        assert m["Accuracy"]  == 1.0
        assert m["Precision"] == 1.0
        assert m["Recall"]    == 1.0
        assert m["F1"]        == 1.0
        assert m["ROC_AUC"]   == 1.0

    def test_all_wrong_classifier(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.9, 0.1, 0.1])
        m = compute_metrics(y_true, y_pred, y_prob)
        assert m["Accuracy"]  == 0.0
        assert m["Recall"]    == 0.0


class TestThresholdFunctions:
    def test_sweep_shape(self, dummy_probs):
        y_true, _, y_prob = dummy_probs
        df = threshold_sweep(y_true, y_prob)
        assert len(df) == 91
        assert set(df.columns) == {"precision", "recall", "f1", "f2"}

    def test_sweep_values_in_range(self, dummy_probs):
        y_true, _, y_prob = dummy_probs
        df = threshold_sweep(y_true, y_prob)
        assert (df >= 0).all().all()
        assert (df <= 1).all().all()

    def test_find_best_threshold_range(self, dummy_probs):
        y_true, _, y_prob = dummy_probs
        t, score = find_best_threshold(y_true, y_prob, metric="f2")
        assert 0.05 <= t <= 0.95
        assert 0.0  <= score <= 1.0

    def test_find_best_threshold_f1(self, dummy_probs):
        y_true, _, y_prob = dummy_probs
        t, score = find_best_threshold(y_true, y_prob, metric="f1")
        assert 0.05 <= t <= 0.95

    def test_high_recall_at_low_threshold(self, dummy_probs):
        y_true, _, y_prob = dummy_probs
        df = threshold_sweep(y_true, y_prob)
        # At threshold 0.1, recall should be very high
        assert df.loc[0.1, "recall"] >= 0.8


class TestCostAnalysis:
    def test_returns_dataframe(self, dummy_probs):
        y_true, _, y_prob = dummy_probs
        df = cost_analysis(y_true, y_prob)
        assert len(df) == 91
        assert "total_cost" in df.columns

    def test_cost_non_negative(self, dummy_probs):
        y_true, _, y_prob = dummy_probs
        df = cost_analysis(y_true, y_prob)
        assert (df["total_cost"] >= 0).all()

    def test_cost_decreases_then_increases(self, dummy_probs):
        """Cost curve should have a minimum somewhere below 0.95."""
        y_true, _, y_prob = dummy_probs
        df = cost_analysis(y_true, y_prob)
        min_idx = df["total_cost"].idxmin()
        # Min cost threshold should not be at the very top extreme
        assert df.loc[min_idx, "threshold"] < 0.95


class TestBootstrapCI:
    def test_returns_three_values(self, dummy_probs):
        y_true, _, y_prob = dummy_probs
        result = bootstrap_ci(y_true, y_prob, metric_fn=roc_auc_score,
                              n_bootstrap=100)
        assert len(result) == 3

    def test_ci_ordering(self, dummy_probs):
        y_true, _, y_prob = dummy_probs
        point, lo, hi = bootstrap_ci(y_true, y_prob, metric_fn=roc_auc_score,
                                     n_bootstrap=100)
        assert lo <= point <= hi

    def test_ci_width_reasonable(self, dummy_probs):
        y_true, _, y_prob = dummy_probs
        _, lo, hi = bootstrap_ci(y_true, y_prob, metric_fn=roc_auc_score,
                                 n_bootstrap=200)
        assert (hi - lo) < 0.3   # CI should not be absurdly wide


class TestMcNemarTest:
    def test_identical_predictions_not_significant(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        result = mcnemar_test(y_true, y_pred, y_pred)
        assert result["p_value"] == 1.0
        # significant_at_0.05 may not be present when n_01+n_10==0
        assert result.get("significant_at_0.05", False) is False

    def test_returns_expected_keys(self, dummy_probs):
        y_true, y_pred, _ = dummy_probs
        rng = np.random.default_rng(0)
        y_pred2 = rng.integers(0, 2, size=len(y_true))
        result = mcnemar_test(y_true, y_pred, y_pred2)
        for key in ["statistic", "p_value", "n_01", "n_10", "significant_at_0.05"]:
            assert key in result

    def test_p_value_in_range(self, dummy_probs):
        y_true, y_pred, _ = dummy_probs
        rng = np.random.default_rng(1)
        y_pred2 = rng.integers(0, 2, size=len(y_true))
        result = mcnemar_test(y_true, y_pred, y_pred2)
        assert 0.0 <= result["p_value"] <= 1.0


class TestECE:
    def test_perfect_calibration(self):
        # Perfectly calibrated: prob 0.3 → 30% of those are positive
        rng = np.random.default_rng(42)
        y_prob = rng.uniform(0, 1, 1000)
        y_true = (rng.uniform(0, 1, 1000) < y_prob).astype(int)
        ece = expected_calibration_error(y_true, y_prob)
        assert ece < 0.1   # should be small for well-calibrated probs

    def test_ece_non_negative(self, dummy_probs):
        y_true, _, y_prob = dummy_probs
        ece = expected_calibration_error(y_true, y_prob)
        assert ece >= 0.0
