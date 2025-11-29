import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import openml

# =====================================================
# CONFIGURAÇÃO
# =====================================================
st.set_page_config(page_title="Avaliação de Risco de Crédito", layout="wide")
MODEL_PATH = "rf_credit_model.joblib"

st.title("Avaliação de Risco de Crédito — Sistema Inteligente")
st.write("""
Este aplicativo utiliza o dataset German Credit (OpenML) para treinar um modelo de 
Random Forest que prevê se um cliente representa alto ou baixo risco de crédito.
""")


# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

@st.cache_data(show_spinner=True)
def load_data():
    dataset = openml.datasets.get_dataset("credit-g")
    X, y, _, _ = dataset.get_data(dataset_format="dataframe",
                                  target=dataset.default_target_attribute)
    df = pd.concat([X, y.rename("target")], axis=1)
    df["target_label"] = df["target"].map({"good": 0, "bad": 1})
    df.drop(columns=["target"], inplace=True)
    return df


def build_preprocessor(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if "target_label" in num_cols:
        num_cols.remove("target_label")

    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="drop"
    )

    return preproc


def train_model(df, features):
    X = df[features]
    y = df["target_label"]

    preproc = build_preprocessor(df)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    pipe = Pipeline([
        ("preproc", preproc),
        ("clf", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_PATH)

    return pipe, X_train, X_test, y_train, y_test


# =====================================================
# CARREGAR DADOS
# =====================================================
df = load_data()
FEATURES = [c for c in df.columns if c != "target_label"]

st.subheader("Amostra dos dados")
st.dataframe(df.head())


# =====================================================
# CARREGAR OU TREINAR MODELO
# =====================================================
if os.path.exists(MODEL_PATH):
    pipe = joblib.load(MODEL_PATH)
    st.success("Modelo carregado com sucesso.")
else:
    st.warning("Modelo não encontrado. Treinando novo modelo...")
    pipe, X_train, X_test, y_train, y_test = train_model(df, FEATURES)
    st.success("Modelo treinado e salvo.")


# =====================================================
# AVALIAÇÃO DO MODELO
# =====================================================
st.header("Desempenho do Modelo")

X = df[FEATURES]
y = df["target_label"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

y_pred = pipe.predict(X_test)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Métricas")
    st.write(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"Precisão: {precision_score(y_test, y_pred):.4f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.4f}")
    st.write(f"F1-score: {f1_score(y_test, y_pred):.4f}")

with col2:
    st.subheader("Matriz de Confusão")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    st.pyplot(fig)

st.subheader("Relatório de Classificação")
st.text(classification_report(
    y_test, y_pred,
    target_names=["good (baixo risco)", "bad (alto risco)"]
))


# =====================================================
# PREDIÇÃO MANUAL VIA CSV
# =====================================================
st.header("Predição de Novo Cliente")

uploaded = st.file_uploader("Envie um CSV contendo uma única linha com os atributos.", type=["csv"])

if uploaded is not None:
    try:
        input_df = pd.read_csv(uploaded)
        st.write("Dados recebidos:")
        st.dataframe(input_df)

        X_input = input_df[FEATURES]
        pred = pipe.predict(X_input)[0]
        prob = pipe.predict_proba(X_input)[0][1]

        label = "ALTO RISCO" if pred == 1 else "BAIXO RISCO"

        st.subheader(f"Resultado: **{label}**")
        st.write(f"Probabilidade estimada de mau pagador: **{prob:.2%}**")

    except Exception as e:
        st.error("Erro ao processar CSV: " + str(e))
else:
    st.info("Envie um arquivo CSV para fazer a previsão.")
