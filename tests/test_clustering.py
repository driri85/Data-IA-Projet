import os
import joblib

import pytest
import numpy as np

from clustering_utils import load_diabetes_df, build_preprocessor, preprocess_df, search_best_pair


def test_load_diabetes_df_has_expected_columns():
    df = load_diabetes_df()
    # diabetes dataset has 10 feature columns + target
    assert df.shape[0] > 0
    assert "target" in df.columns
    assert df.shape[1] >= 11


def test_preprocessor_transforms_shape():
    df = load_diabetes_df()
    numerical_cols = [c for c in df.columns if c != "target"]
    X_scaled, pre = preprocess_df(df, numerical_cols)
    assert X_scaled.shape[0] == df.shape[0]
    # number of features should equal number of columns passed
    assert X_scaled.shape[1] == len(numerical_cols)


def test_search_best_pair_returns_expected_keys():
    df = load_diabetes_df()
    numerical_cols = [c for c in df.columns if c != "target"]
    res = search_best_pair(df, numerical_cols)
    assert isinstance(res, dict)
    assert "best_score" in res
    assert "pair" in res
    assert "best_algo" in res


def test_best_score_reasonable_threshold():
    # Ensure the search returns a non-trivial silhouette score
    df = load_diabetes_df()
    numerical_cols = [c for c in df.columns if c != "target"]
    res = search_best_pair(df, numerical_cols)
    # Expect silhouette between -1 and 1, and usually > 0.3 for some pair
    assert -1.0 <= res["best_score"] <= 1.0
    assert res["best_score"] >= 0.3


if __name__ == "__main__":
    pytest.main([os.path.dirname(__file__)])
