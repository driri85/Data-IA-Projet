import os
import unittest

from clustering_utils import load_diabetes_df, preprocess_df, search_best_pair


class TestClusteringUtils(unittest.TestCase):
    def test_load_diabetes_df_has_expected_columns(self):
        df = load_diabetes_df()
        self.assertGreater(df.shape[0], 0)
        self.assertIn("target", df.columns)
        self.assertGreaterEqual(df.shape[1], 11)

    def test_preprocessor_transforms_shape(self):
        df = load_diabetes_df()
        numerical_cols = [c for c in df.columns if c != "target"]
        X_scaled, pre = preprocess_df(df, numerical_cols)
        self.assertEqual(X_scaled.shape[0], df.shape[0])
        self.assertEqual(X_scaled.shape[1], len(numerical_cols))

    def test_search_best_pair_returns_expected_keys(self):
        df = load_diabetes_df()
        numerical_cols = [c for c in df.columns if c != "target"]
        res = search_best_pair(df, numerical_cols)
        self.assertIsInstance(res, dict)
        self.assertIn("best_score", res)
        self.assertIn("pair", res)
        self.assertIn("best_algo", res)

    def test_best_score_reasonable_threshold(self):
        df = load_diabetes_df()
        numerical_cols = [c for c in df.columns if c != "target"]
        res = search_best_pair(df, numerical_cols)
        self.assertGreaterEqual(res["best_score"], -1.0)
        self.assertLessEqual(res["best_score"], 1.0)
        self.assertGreaterEqual(res["best_score"], 0.3)


if __name__ == "__main__":
    unittest.main()
