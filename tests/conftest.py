"""
conftest.py
-----------
Shared pytest fixtures for the churn project test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_utils import load_raw, make_splits, split_xy


@pytest.fixture(scope="session")
def raw_df():
    return load_raw()


@pytest.fixture(scope="session")
def splits(raw_df):
    return make_splits(raw_df)


@pytest.fixture(scope="session")
def X_train(splits):
    return splits[0]


@pytest.fixture(scope="session")
def X_val(splits):
    return splits[1]


@pytest.fixture(scope="session")
def X_test(splits):
    return splits[2]


@pytest.fixture(scope="session")
def y_train(splits):
    return splits[3]


@pytest.fixture(scope="session")
def y_val(splits):
    return splits[4]


@pytest.fixture(scope="session")
def y_test(splits):
    return splits[5]


@pytest.fixture(scope="session")
def small_X(X_train):
    """50-row subsample for fast model tests."""
    return X_train.iloc[:50].reset_index(drop=True)


@pytest.fixture(scope="session")
def small_y(y_train):
    return y_train.iloc[:50].reset_index(drop=True)


@pytest.fixture(scope="session")
def dummy_probs():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob
