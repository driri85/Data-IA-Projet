import itertools
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


def load_diabetes_df() -> pd.DataFrame:
    from sklearn.datasets import load_diabetes

    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target
    return df


def build_preprocessor(columns: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([("num", numeric_transformer, columns)])


def preprocess_df(df: pd.DataFrame, columns: List[str]) -> Tuple[np.ndarray, ColumnTransformer]:
    pre = build_preprocessor(columns)
    X_scaled = pre.fit_transform(df[columns])
    return X_scaled, pre


def evaluate_pair(
    df: pd.DataFrame,
    pair: Tuple[str, str],
    n_clusters_range: Tuple[int, int] = (2, 6),
) -> Dict[str, Any]:
    """Evaluate clustering on a pair of features and return best configuration (max silhouette).

    Returns dict with keys: best_score, best_algo, best_k, labels, pair
    """
    X_subset, _ = preprocess_df(df, list(pair))

    best = {"best_score": -1.0}
    for k in range(n_clusters_range[0], n_clusters_range[1]):
        # KMeans
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_subset)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_subset, labels)
            if score > best["best_score"]:
                best = {
                    "best_score": float(score),
                    "best_algo": "kmeans",
                    "best_k": int(k),
                    "labels": labels,
                    "pair": pair,
                }

        # Agglomerative
        ac = AgglomerativeClustering(n_clusters=k)
        labels = ac.fit_predict(X_subset)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_subset, labels)
            if score > best["best_score"]:
                best = {
                    "best_score": float(score),
                    "best_algo": "agglomerative",
                    "best_k": int(k),
                    "labels": labels,
                    "pair": pair,
                }

    return best


def search_best_pair(df: pd.DataFrame, numerical_cols: List[str]) -> Dict[str, Any]:
    best_overall = {"best_score": -1.0}
    for pair in itertools.combinations(numerical_cols, 2):
        res = evaluate_pair(df, pair)
        if res.get("best_score", -1) > best_overall["best_score"]:
            best_overall = res
    return best_overall
