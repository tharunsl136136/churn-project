"""
test_features.py
----------------
Tests for src/features.py
"""

import numpy as np
import pandas as pd
import pytest

from features import (
    ChurnFeatureEngineer,
    build_preprocessor,
    build_linear_pipeline,
    build_tree_pipeline,
    NUMERIC_COLS, CATEGORICAL_COLS, CONTRACT_COL,
)
from sklearn.dummy import DummyClassifier


class TestChurnFeatureEngineer:
    def test_fit_returns_self(self, X_train):
        fe = ChurnFeatureEngineer()
        result = fe.fit(X_train)
        assert result is fe

    def test_output_columns_added(self, X_train):
        fe = ChurnFeatureEngineer()
        out = fe.transform(X_train.head(10))
        expected_new = [
            "tenure_group", "num_services", "has_protection_bundle",
            "has_streaming", "avg_monthly_spend", "charge_gap",
            "is_month_to_month", "is_autopay", "contract_internet",
            "monthly_charges_per_service", "is_new_customer", "is_family",
        ]
        for col in expected_new:
            assert col in out.columns, f"Missing column: {col}"

    def test_no_original_columns_dropped(self, X_train):
        fe = ChurnFeatureEngineer()
        out = fe.transform(X_train.head(10))
        for col in X_train.columns:
            assert col in out.columns

    def test_num_services_range(self, X_train):
        fe = ChurnFeatureEngineer()
        out = fe.transform(X_train)
        assert out["num_services"].between(0, 6).all()

    def test_is_month_to_month_binary(self, X_train):
        fe = ChurnFeatureEngineer()
        out = fe.transform(X_train)
        assert set(out["is_month_to_month"].unique()).issubset({0, 1})

    def test_is_autopay_binary(self, X_train):
        fe = ChurnFeatureEngineer()
        out = fe.transform(X_train)
        assert set(out["is_autopay"].unique()).issubset({0, 1})

    def test_is_new_customer_binary(self, X_train):
        fe = ChurnFeatureEngineer()
        out = fe.transform(X_train)
        assert set(out["is_new_customer"].unique()).issubset({0, 1})

    def test_is_family_binary(self, X_train):
        fe = ChurnFeatureEngineer()
        out = fe.transform(X_train)
        assert set(out["is_family"].unique()).issubset({0, 1})

    def test_avg_monthly_spend_positive(self, X_train):
        fe = ChurnFeatureEngineer()
        out = fe.transform(X_train)
        assert (out["avg_monthly_spend"] >= 0).all()

    def test_stateless_fit_does_not_change_transform(self, X_train):
        fe = ChurnFeatureEngineer()
        out1 = fe.transform(X_train.head(20))
        fe.fit(X_train)
        out2 = fe.transform(X_train.head(20))
        pd.testing.assert_frame_equal(out1, out2)

    def test_contract_internet_interaction(self, X_train):
        fe = ChurnFeatureEngineer()
        out = fe.transform(X_train.head(5))
        for _, row in out.iterrows():
            expected = f"{row['Contract']}_{row['InternetService']}"
            assert row["contract_internet"] == expected


class TestBuildPreprocessor:
    def test_fit_transform_shape(self, X_train):
        fe = ChurnFeatureEngineer()
        X_fe = fe.transform(X_train)
        prep = build_preprocessor(scale=True)
        X_out = prep.fit_transform(X_fe)
        assert X_out.shape[0] == len(X_train)
        assert X_out.shape[1] > 30   # OHE expands columns

    def test_no_nans_after_preprocessing(self, X_train):
        fe = ChurnFeatureEngineer()
        X_fe = fe.transform(X_train)
        prep = build_preprocessor(scale=False)
        X_out = prep.fit_transform(X_fe)
        assert not np.isnan(X_out).any()

    def test_tree_pipeline_predict_proba(self, small_X, small_y):
        pipe = build_tree_pipeline(DummyClassifier(strategy="most_frequent"))
        pipe.fit(small_X, small_y)
        probs = pipe.predict_proba(small_X)
        assert probs.shape == (len(small_X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_linear_pipeline_predict_proba(self, small_X, small_y):
        pipe = build_linear_pipeline(DummyClassifier(strategy="most_frequent"))
        pipe.fit(small_X, small_y)
        probs = pipe.predict_proba(small_X)
        assert probs.shape == (len(small_X), 2)
