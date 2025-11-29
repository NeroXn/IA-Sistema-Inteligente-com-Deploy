#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Streamlit app para Avaliação de Risco de Crédito
# Uso: streamlit run app.py

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
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import openml

MODEL_PATH = "rf_credit_model.joblib"
PREPROC_PATH = "preprocessor.joblib"

st.set_page_config(page_title="Avaliação de Risco de Crédito", layout="wide")

st.title("Avaliação de Risco de Crédito — Demo")
st.markdown(
    """
Este app carrega um dataset público (German Credit via OpenML), treina um modelo de Random Forest
(se ainda não existir), e permite fazer predições de risco (bom / mau pagador).
Mostra também métricas e importância das features.
"""
)

@st.cache_data(show_spinner=False)
def load_data():
    # Carrega dataset 'credit-g' do OpenML (German Credit)
    # Retorna DataFrame com X e y
    dataset = openml.datasets.get_dataset("credit-g")
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    df = pd.concat([X, y.rename("target")], axis=1)
    # ajustar nomes: target é 'good'/'bad' -> vamos mapear para 0 = good (baixo risco), 1 = bad (alto risco)
    df["target_label"] = df["target"].map({"good": 0, "bad": 1})
    df.drop(columns=["target"], inplace=True)
    return df

df = load_data()

st.sidebar.header("Dados")
if st.sidebar.checkbox("Mostrar amostra do dataset"):
    st.dataframe(df.sample(10))

st.sidebar.markdown("### Preparar / Treinar modelo")
train_button = st.sidebar.button("Treinar / Re-treinar modelo (demora)")

# Seleção de features simples: usa todas as colunas originais exceto target_label
FEATURES = [c for c in df.columns if c != "target_label"]

def build_preprocessor(df):
    # Detecta colunas categóricas e numéricas
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Remove target se presente
    if "target_label" in num_cols:
        num_cols.remove("target_label")
    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor, num_cols, cat_cols

def train_and_save_model(df, features):
    X = df[features]
    y = df["target_label"]
    preprocessor, num_cols, cat_cols = build_preprocessor(X.join(y))
    # Pipeline: preproc + clf
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = Pipeline([("preproc", preprocessor), ("clf", clf)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pipe.fit(X_train, y_train)
    # salvar
    joblib.dump(pipe, MODEL_PATH)
    return pipe, (X_train, X_test, y_train, y_test), (num_cols, cat_cols)

# Treinamento condicional
model_exists = os.path.exists(MODEL_PATH)
if (not model_exists) or train_button:
    with st.spinner("Treinando modelo... isso pode levar alguns segundos"):
        pipe, split_data, cols = train_and_save_model(df, FEATURES)
    st.success("Modelo treinado e salvo em disk.")
else:
    pipe = joblib.load(MODEL_PATH)
    st.sidebar.success("Modelo carregado do arquivo.")

# Se carregado, avalie rapidamente no holdout
X = df[FEATURES]
y = df["target_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
y_pred = pipe.predict(X_test)

st.subheader("Métricas no conjunto de teste")
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
st.write(f"Acurácia: {acc:.4f}")
st.write(f"Precisão: {prec:.4f}")
st.write(f"Recall: {rec:.4f}")
st.write(f"F1-score: {f1:.4f}")

st.subheader("Matriz de Confusão")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.imshow(cm, interpolation='nearest')
ax.set_title("Matriz de Confusão")
ax.set_xlabel("Predito")
ax.set_ylabel("Verdadeiro")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i,j], ha="center", va="center", color="white")
st.pyplot(fig)

st.subheader("Relatório de Classificação (por classe)")
st.text(classification_report(y_test, y_pred, target_names=["good (baixo risco)","bad (alto risco)"]))

# Importância por permutation importance (mais robusta para pipelines com OHE)
with st.spinner("Calculando importância das features (permutation)..."):
    try:
        r = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importances = r.importances_mean
        # nomes das features após preproc
        preproc = pipe.named_steps["preproc"]
        # Para exibir nomes originais, precisamos transformar X_test uma vez para obter as colunas
        transformed = preproc.fit_transform(X_test)  # safe: preproc usa refit internamente
        # Construir nomes das colunas transformadas
        # Obter num col names
        num_cols = X_test.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
        # encoder categories
        ohe = preproc.named_transformers_["cat"]
        ohe_feature_names = []
        if hasattr(ohe, 'get_feature_names_out'):
            ohe_feature_names = list(ohe.get_feature_names_out(cat_cols))
        else:
            # fallback
            for i, c in enumerate(cat_cols):
                vals = getattr(ohe, 'categories_', [])[i]
                ohe_feature_names += [f"{c}_{v}" for v in vals]
        feature_names = num_cols + ohe_feature_names
        # criar dataframe
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        fig2, ax2 = plt.subplots(figsize=(8,5))
        ax2.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
        ax2.set_xlabel("Permutation importance")
        ax2.set_title("Top 20 features por importância")
        st.pyplot(fig2)
    except Exception as e:
        st.error("Erro ao calcular importância: " + str(e))

st.sidebar.header("Predição manual")
st.sidebar.markdown("Preencha os dados do cliente para obter uma predição de risco.")

# Montar formulário com as features principais. Para simplicidade, vamos pedir variáveis chave extraídas do dataset credit-g.
# Vamos criar inputs para algumas colunas presentes no dataset original.
# Exibir os nomes das colunas para referência:
if st.sidebar.checkbox("Mostrar nomes originais das colunas"):
    st.sidebar.write(FEATURES)

# Mapeamento rápido de alguns campos (ajuste conforme o dataset real)
# Para robustez vamos permitir que o usuário envie um CSV também.
st.sidebar.markdown("### Opcional: subir CSV com uma linha (colunas iguais às do dataset)")
uploaded = st.sidebar.file_uploader("Upload CSV (opcional)", type=["csv"])
if uploaded is not None:
    input_df = pd.read_csv(uploaded)
    st.sidebar.write("Primeiras linhas do CSV enviado:")
    st.sidebar.write(input_df.head())
    if "target_label" in input_df.columns:
        input_df = input_df.drop(columns=["target_label"])
else:
    # Criar um formulário manual com alguns campos genéricos
    st.sidebar.markdown("Ou preencher manualmente:")
    # selecionar algumas colunas padrão do credit-g:
    try:
        # inferir category columns to build widgets
        sample = df.iloc[0]
        # Provide generic inputs for typical columns (idade, duration, amount, purpose, credit_history, saving_accounts, employment, personal_status, other_debtors, property_magnitude)
        duration = st.sidebar.number_input("duration (months)", min_value=1, max_value=300, value=12)
        amount = st.sidebar.number_input("credit amount", min_value=100, max_value=1000000, value=1000)
        age = st.sidebar.number_input("age", min_value=18, max_value=100, value=35)
        # For categorical fields provide text inputs (user must match labels of dataset)
        purpose = st.sidebar.text_input("purpose (ex: radio/TV, education, business, car...)", value="radio/TV")
        housing = st.sidebar.selectbox("housing (own/rent/for free)", ["own", "rent", "for free"])
        job = st.sidebar.selectbox("job (0..3)", ["0","1","2","3"])
        # Assemble a minimal row - other features set to typical values or blank
        input_dict = {}
        # try to fill with dataset column names if present
        for col in FEATURES:
            if col.lower() in ["duration", "credit_amount", "amount", "duration_in_month", "credit amount"]:
                input_dict[col] = duration if "duration" in col.lower() else amount
            elif "age" in col.lower():
                input_dict[col] = age
            elif "purpose" in col.lower():
                input_dict[col] = purpose
            elif "housing" in col.lower():
                input_dict[col] = housing
            elif "job" in col.lower():
                input_dict[col] = job
            else:
                # default: try to fill with first value from df
                input_dict[col] = df[col].mode().iloc[0] if df[col].dtype == 'O' else float(df[col].median())
        input_df = pd.DataFrame([input_dict])

if st.sidebar.button("Predizer risco"):
    try:
        X_input = input_df[FEATURES]
        pred = pipe.predict(X_input)[0]
        pred_proba = pipe.predict_proba(X_input)[0][1]  # prob of class 1 (bad)
        label = "ALTO RISCO (mau pagador)" if pred == 1 else "BAIXO RISCO (bom pagador)"
        st.sidebar.markdown(f"## Predição: **{label}**")
        st.sidebar.markdown(f"Probabilidade estimada de mau pagador: **{pred_proba:.2%}**")
        # explicação simples: mostrar importância local via SHAP (se disponível)
        with st.spinner("Calculando explicação local (SHAP), se disponível..."):
            try:
                import shap
                explainer = shap.Explainer(pipe.named_steps["clf"], pipe.named_steps["preproc"].transform(X_train))
                # transformar input
                X_trans = pipe.named_steps["preproc"].transform(X_input)
                shap_values = explainer(X_trans)
                st.sidebar.markdown("Resumo SHAP (valores de contribuição):")
                shap_html = shap.plots.waterfall(shap_values[0], show=False)
                st.sidebar.write(" (visualização SHAP não exibida inline neste demo)")
            except Exception as e:
                st.sidebar.info("SHAP não disponível ou falha ao computar: " + str(e))
    except Exception as e:
        st.error("Erro ao predizer: " + str(e))

st.write("---")
st.markdown("### Observações finais")
st.markdown(
    """
- O dataset usado é o German Credit (OpenML 'credit-g').  
- Classe 0 = good (baixo risco); classe 1 = bad (alto risco).  
- Limitações: precisamos garantir que as entradas do formulário mapeiem corretamente os nomes/categorias do dataset original; em produção, harmonização e verificação de dados é obrigatória.
"""
)
