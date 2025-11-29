#!/usr/bin/env python3
# Script opcional para treinar o modelo localmente e exportar artifacts (modelo .joblib)
import joblib
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

MODEL_PATH = "rf_credit_model.joblib"

print("Carregando dataset credit-g (OpenML)...")
dataset = openml.datasets.get_dataset("credit-g")
X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
df = pd.concat([X, y.rename("target")], axis=1)
df["target_label"] = df["target"].map({"good": 0, "bad": 1})
df.drop(columns=["target"], inplace=True)
FEATURES = [c for c in df.columns if c != "target_label"]

X = df[FEATURES]
y = df["target_label"]

num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

preproc = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
    ],
    remainder="drop"
)

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
pipe = Pipeline([("preproc", preproc), ("clf", clf)])

print("Treinando...")
pipe.fit(X, y)
print("Salvando modelo em", MODEL_PATH)
joblib.dump(pipe, MODEL_PATH)
print("Terminou.")
