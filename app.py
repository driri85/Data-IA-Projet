import streamlit as st
import pandas as pd
from sklearn.datasets import load_diabetes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.write("""
# My first app
Hello *world!*
""")

# Chargement du dataset diabetes
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

st.write(df)

# Statistiques descriptives pour le dataset diabetes
st.write(df.describe())

# Heatmap de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', fmt='.2f')
plt.title('Matrice de Corrélation - Dataset Diabetes')
st.pyplot(plt)

X = df.drop(columns=['target'])
y = df['target']

# Pour le ColumnTransformer, toutes les colonnes de ce dataset sont numériques
numerical_cols = X.columns.tolist()
categorical_cols = []

from sklearn.model_selection import train_test_split

# Division 60/20/20 pour le nouveau dataset
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

st.write(f"Nouveaux splits (Diabetes) - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Pipeline simplifié car les données sont déjà pré-traitées dans load_diabetes
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols)
    ])

from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# D"efinition de trois modeles lineaires regularises
ridge = Ridge(alpha=10.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Cr"eation du Voting Regressor
voting_model = VotingRegressor(
    estimators=[
        ('ridge', ridge),
        ('lasso', lasso),
        ('elastic', elastic)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', voting_model)
])

# Entraînement de l'ensemble (Voting Regressor)
model_pipeline.fit(X_train, y_train)
score_val = model_pipeline.score(X_val, y_val)

st.write(f"Score R2 final avec l'Ensemble (Voting) : {score_val:.4f}")

y_pred = model_pipeline.predict(X_val)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_val, y=y_pred, alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)
plt.xlabel("Valeurs Réelles (Progression Diabète)")
plt.ylabel("Prédictions")
plt.title(f"Performance du modèle - Diabetes (R²: {score_val:.4f})")
st.pyplot(plt)
joblib.dump(model_pipeline, "model.pkl")