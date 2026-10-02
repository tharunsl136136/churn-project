"""
test_data_utils.py
------------------
Tests for src/data_utils.py
"""

import numpy as np
import pandas as pd
import pytest

from data_utils import (
    load_raw, make_splits, split_xy,
    class_balance, cramers_v, tenure_bucket,
    RANDOM_STATE, TARGET_COL,
)


class TestLoadRaw:
    def test_shape(self, raw_df):
        assert raw_df.shape[0] == 7043
        assert raw_df.shape[1] == 20   # 21 cols - customerID

    def test_no_customer_id(self, raw_df):
        assert "customerID" not in raw_df.columns

    def test_target_binary(self, raw_df):
        assert set(raw_df[TARGET_COL].unique()).issubset({0, 1})

    def test_total_charges_numeric(self, raw_df):
        assert pd.api.types.is_numeric_dtype(raw_df["TotalCharges"])

    def test_no_blank_total_charges(self, raw_df):
        assert raw_df["TotalCharges"].isna().sum() == 0

    def test_no_internet_service_string(self, raw_df):
        for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]:
            assert "No internet service" not in raw_df[col].values

    def test_no_phone_service_string(self, raw_df):
        assert "No phone service" not in raw_df["MultipleLines"].values

    def test_churn_rate(self, raw_df):
        rate = raw_df[TARGET_COL].mean()
        assert 0.25 < rate < 0.28


class TestMakeSplits:
    def test_sizes(self, splits):
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == 7043
        assert abs(len(X_train) / total - 0.60) < 0.02
        assert abs(len(X_val)   / total - 0.20) < 0.02
        assert abs(len(X_test)  / total - 0.20) < 0.02

    def test_stratification(self, splits):
        _, _, _, y_train, y_val, y_test = splits
        rates = [s.mean() for s in (y_train, y_val, y_test)]
        # All splits should have churn rate within 1% of each other
        assert max(rates) - min(rates) < 0.01

    def test_no_overlap(self, splits):
        X_train, X_val, X_test = splits[0], splits[1], splits[2]
        idx_train = set(X_train.index)
        idx_val   = set(X_val.index)
        idx_test  = set(X_test.index)
        assert len(idx_train & idx_val)  == 0
        assert len(idx_train & idx_test) == 0
        assert len(idx_val   & idx_test) == 0

    def test_x_y_alignment(self, splits):
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        assert len(X_train) == len(y_train)
        assert len(X_val)   == len(y_val)
        assert len(X_test)  == len(y_test)


class TestHelpers:
    def test_class_balance_keys(self, raw_df):
        bal = class_balance(raw_df[TARGET_COL])
        assert "n_churn" in bal and "n_no_churn" in bal and "churn_rate" in bal

    def test_class_balance_values(self, raw_df):
        bal = class_balance(raw_df[TARGET_COL])
        assert bal["n_churn"] + bal["n_no_churn"] == 7043
        assert 25.0 < bal["churn_rate"] < 28.0

    def test_cramers_v_range(self, raw_df):
        v = cramers_v(raw_df["Contract"], raw_df[TARGET_COL].astype(str))
        assert 0.0 <= v <= 1.0

    def test_cramers_v_identical(self, raw_df):
        # Perfect association with itself
        v = cramers_v(raw_df["Contract"], raw_df["Contract"])
        assert v > 0.95

    def test_tenure_bucket_categories(self, raw_df):
        buckets = tenure_bucket(raw_df["tenure"])
        expected = {"0-12", "13-24", "25-48", "49-60", "60+"}
        assert set(buckets.cat.categories) == expected

    def test_tenure_bucket_no_nulls(self, raw_df):
        buckets = tenure_bucket(raw_df["tenure"])
        assert buckets.isna().sum() == 0
