"""
test_models.py
--------------
Tests for src/models.py  (pipeline structure, fit/predict, artifact loading)
"""

import numpy as np
import pytest
from pathlib import Path
import joblib

from models import (
    get_baseline_pipelines,
    get_advanced_pipelines,
    build_tuned_pipeline,
    cv_score,
)


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class TestBaselinePipelines:
    def test_returns_three_pipelines(self):
        pipes = get_baseline_pipelines()
        assert len(pipes) == 3

    def test_expected_keys(self):
        pipes = get_baseline_pipelines()
        assert "Dummy_MostFrequent" in pipes
        assert "LogisticRegression" in pipes
        assert "DecisionTree" in pipes

    def test_dummy_fit_predict(self, small_X, small_y):
        pipe = get_baseline_pipelines()["Dummy_MostFrequent"]
        pipe.fit(small_X, small_y)
        preds = pipe.predict(small_X)
        assert len(preds) == len(small_X)
        assert set(preds).issubset({0, 1})

    def test_lr_predict_proba(self, small_X, small_y):
        pipe = get_baseline_pipelines()["LogisticRegression"]
        pipe.fit(small_X, small_y)
        probs = pipe.predict_proba(small_X)
        assert probs.shape == (len(small_X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_dt_predict_proba(self, small_X, small_y):
        pipe = get_baseline_pipelines()["DecisionTree"]
        pipe.fit(small_X, small_y)
        probs = pipe.predict_proba(small_X)
        assert probs.shape == (len(small_X), 2)


class TestAdvancedPipelines:
    def test_returns_four_pipelines(self):
        pipes = get_advanced_pipelines()
        assert len(pipes) == 4

    def test_expected_keys(self):
        pipes = get_advanced_pipelines()
        for key in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
            assert key in pipes

    def test_lgbm_fit_predict(self, small_X, small_y):
        pipe = get_advanced_pipelines()["LightGBM"]
        pipe.fit(small_X, small_y)
        probs = pipe.predict_proba(small_X)
        assert probs.shape == (len(small_X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


class TestBuildTunedPipeline:
    def test_lgbm_tuned_pipeline(self, small_X, small_y):
        params = {
            "n_estimators": 50, "num_leaves": 31, "learning_rate": 0.1,
            "max_depth": 5, "min_child_samples": 20,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        }
        pipe = build_tuned_pipeline("LightGBM", params)
        pipe.fit(small_X, small_y)
        probs = pipe.predict_proba(small_X)
        assert probs.shape == (len(small_X), 2)

    def test_xgb_tuned_pipeline(self, small_X, small_y):
        params = {
            "n_estimators": 50, "max_depth": 4, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 1, "gamma": 0.0,
            "reg_alpha": 0.0, "reg_lambda": 1.0,
        }
        pipe = build_tuned_pipeline("XGBoost", params)
        pipe.fit(small_X, small_y)
        probs = pipe.predict_proba(small_X)
        assert probs.shape == (len(small_X), 2)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            build_tuned_pipeline("UnknownModel", {})


class TestSavedArtifact:
    """Tests that the saved best_pipeline.joblib has the expected structure."""

    @pytest.fixture(scope="class")
    def artifact(self):
        path = MODELS_DIR / "best_pipeline.joblib"
        if not path.exists():
            pytest.skip("best_pipeline.joblib not found – run notebooks first")
        return joblib.load(path)

    def test_artifact_keys(self, artifact):
        for key in ["pipeline", "model_name", "best_params", "val_prauc"]:
            assert key in artifact, f"Missing key: {key}"

    def test_threshold_in_range(self, artifact):
        # opt_threshold added by run_all.py; notebooks save val_rocauc instead
        t = artifact.get("opt_threshold") or artifact.get("val_rocauc")
        assert t is not None
        assert 0.0 < t < 1.0

    def test_val_prauc_reasonable(self, artifact):
        assert artifact["val_prauc"] > 0.50

    def test_pipeline_predict_proba(self, artifact, X_test):
        pipe  = artifact["pipeline"]
        probs = pipe.predict_proba(X_test.head(10))
        assert probs.shape == (10, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
