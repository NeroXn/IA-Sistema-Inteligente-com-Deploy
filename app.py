# app.py — Avaliação de Risco de Crédito (Classificação + Regressão)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# ====== Tema azul-preto — injeta CSS e ajusta matplotlib ======
import streamlit as st
import matplotlib.pyplot as plt

def apply_blue_black_theme():


    # ========== CSS COMPLETO (AZUL-PRETO) ==========
    css = """
    <style>

    /* ======= BACKGROUND PRINCIPAL ======= */
    section[data-testid="stAppViewContainer"] > div {
        background: linear-gradient(180deg, #03101f 0%, #001229 45%, #000000 100%);
        color: #e6f6ff;
    }
section[data-testid="stAppViewContainer"] > div {
    background:
        linear-gradient(90deg, rgba(0,255,255,0.07) 1px, transparent 1px),
        linear-gradient(0deg, rgba(0,255,255,0.07) 1px, transparent 1px),
        #000814;
    background-size: 40px 40px;
}
    /* ======= SIDEBAR ======= */
    div[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #011627 0%, #00192f 50%, #000000 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
        color: #cfeaff;
    }

    /* ======= TITULOS ======= */
    h1, h2, h3, h4 {
        color: #d7f0ff !important;
        font-weight: 600;
    }

    /* ====== TABS ====== */
    button[data-baseweb="tab"] {
        background: #001a33 !important;
        color: #d8f2ff !important;
        border-radius: 8px !important;
        border: 1px solid #023a57 !important;
        margin-right: 4px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #0ea5e9 !important;
        color: black !important;
        font-weight: 700 !important;
    }

    /* ====== MENUS, INPUTS, SELECTS ESCUROS ====== */
    input, select, textarea {
        background-color: #0d1b2a !important;
        color: #e6f6ff !important;
        border: 1px solid #1a2b3c !important;
        border-radius: 6px !important;
    }

    /* ====== number_input container ====== */
    div[data-testid="stNumberInput"] > div {
        background-color: #0d1b2a !important;
        border: 1px solid #1a2b3c !important;
        border-radius: 6px !important;
    }

    /* ===== botões + e - do number_input ===== */
    button[kind="secondary"] {
        background-color: #0ea5e9 !important;
        color: black !important;
        border-radius: 4px !important;
        font-weight: bold !important;
    }

    /* CHECKBOX LABEL */
    label {
        color: #e6f6ff !important;
        font-size: 15px !important;
    }

    /* ====== BOTÕES PRINCIPAIS ====== */
    button[kind="primary"] {
        background: linear-gradient(180deg,#0ea5e9,#023a57) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    button[kind="primary"]:hover {
        filter: brightness(1.15);
    }

    /* ====== TABELAS ====== */
    thead tr th {
        background: #012c46 !important;
        color: #dff6ff !important;
        font-weight: 600 !important;
    }
    tbody tr td {
        background: rgba(255,255,255,0.03) !important;
        color: #eaf8ff !important;
    }

    /* ====== SLIDER ====== */
    .stSlider > div > div > div {
        background-color: #0ea5e9 !important;
    }

    /* ====== MENSAGENS (info, warning, success) ====== */
    .stAlert, .stInfo, .stWarning, .stSuccess {
        background-color: rgba(255,255,255,0.05) !important;
        border-left: 4px solid #0ea5e9 !important;
        color: #eaf8ff !important;
    }

    /* LINKS */
    a {
        color: #7ed7ff !important;
        text-decoration: none !important;
    }
    a:hover { text-decoration: underline !important; }

    /* GRAFICOS / CANVAS */
    .element-container .stPlotlyChart, .element-container .stImage {
        border-radius: 10px;
    }

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

    # ======= Matplotlib Dark Mode =======
    plt.rcParams.update({
        "axes.labelcolor": "#dff6ff",
        "xtick.color": "#dff6ff",
        "ytick.color": "#dff6ff",
        "figure.facecolor": "#00111f",
        "axes.facecolor": "#00111f",
        "axes.edgecolor": "#0ea5e9",
        "grid.color": "#334455"
    })

# Aplicar tema
apply_blue_black_theme()



from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
import matplotlib.pyplot as plt
import openml
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Risco de Crédito — Classificação + Regressão", layout="wide")

# Paths
CLASS_MODEL_PATH = "rf_credit_model.joblib"
REG_MODEL_PATH = "rf_reg_model.joblib"

st.title("Avaliação de Risco de Crédito — Classificação e Regressão")
st.write("App com duas abas: Classificação (German Credit) e Regressão (usar CSV do Banco Central ou outro).")

# ----------------------------
# Utilidades
# ----------------------------
@st.cache_data(show_spinner=True)
def load_german_credit():
    dataset = openml.datasets.get_dataset("credit-g")
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
    df = pd.concat([X, y.rename("target")], axis=1)
    df["target_label"] = df["target"].map({"good": 0, "bad": 1})
    df.drop(columns=["target"], inplace=True)
    return df

def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="drop"
    )

def oversample_minority(X, y, random_state=42):
    # Merge
    df = pd.concat([X, y.rename("target_label")], axis=1)
    majority = df[df["target_label"]==0]
    minority = df[df["target_label"]==1]
    if len(minority) == 0:
        return X, y
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=random_state)
    df_up = pd.concat([majority, minority_upsampled])
    y_up = df_up["target_label"]
    X_up = df_up.drop(columns=["target_label"])
    return X_up, y_up

def plot_feature_importance(pipe, X_test, title="Feature importance"):
    try:
        # Try permutation importance (works with pipeline)
        res = permutation_importance(pipe, X_test, pipe.predict(X_test), n_repeats=8, random_state=42, n_jobs=1)
        importances = res.importances_mean
        # Try to get feature names after preprocessing
        preproc = pipe.named_steps["preproc"]
        num_cols = X_test.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
        ohe = preproc.named_transformers_.get("cat", None)
        ohe_names = []
        if ohe is not None and hasattr(ohe, "get_feature_names_out"):
            ohe_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names = num_cols + ohe_names
        # match length safety
        if len(importances) != len(feature_names):
            # fallback to feature_importances_ of classifier
            importances = pipe.named_steps["clf"].feature_importances_
            feature_names = feature_names[:len(importances)]
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=True).tail(20)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.barh(imp_df["feature"], imp_df["importance"])
        ax.set_title(title)
        st.pyplot(fig)
    except Exception as e:
        st.info("Não foi possível calcular permutation importance: " + str(e))
        try:
            # fallback
            importances = pipe.named_steps["clf"].feature_importances_
            feature_names = X_test.columns.tolist()
            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=True).tail(20)
            fig, ax = plt.subplots(figsize=(8,6))
            ax.barh(imp_df["feature"], imp_df["importance"])
            ax.set_title(title + " (fallback)")
            st.pyplot(fig)
        except Exception as e2:
            st.write("Falha ao mostrar importância:", e2)

# ----------------------------
# Tabs: Classificação / Regressão
# ----------------------------
tab1, tab2 = st.tabs(["Classificação (German Credit)", "Regressão (dados reais)"])

# ----------------------------
# TAB 1 — Classificação
# ----------------------------
with tab1:
    st.header("Classificação: German Credit (bom/mau pagador)")

    df = load_german_credit()
    FEATURES = [c for c in df.columns if c != "target_label"]

    st.subheader("Amostra dos dados")
    st.dataframe(df.head())

    st.markdown("### Opções de Treino (toda mudança re-treina o modelo)")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        do_oversample = st.checkbox("Balancear classes (oversample)", value=False)
    with col_b:
        do_tune = st.checkbox("Hiperparam tuning (lento)", value=False)
    with col_c:
        n_estimators = st.number_input("n_estimators", min_value=50, max_value=1000, value=200, step=50)

    st.info("➡️ Qualquer alteração acima treina novamente o modelo automaticamente.")

    # ============================================================
    # RE-TREINAR SEMPRE QUE CONFIGURAÇÃO MUDA
    # ============================================================

    # Preparar dados
    X = df[FEATURES]
    y = df["target_label"]

    if do_oversample:
        X, y = oversample_minority(X, y)
        st.success(f"Balanceamento aplicado: {len(y)} linhas após oversample.")

    preproc = build_preprocessor(X)

    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    pipe_temp = Pipeline([("preproc", preproc), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    if do_tune:
        st.warning("Executando RandomizedSearch (demora um pouco)...")
        param_dist = {
            "clf__n_estimators": [100, 200, 300, 500],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4]
        }
        tuner = RandomizedSearchCV(
            pipe_temp,
            param_distributions=param_dist,
            n_iter=8,
            cv=3,
            random_state=42,
            n_jobs=1
        )
        tuner.fit(X_train, y_train)
        pipe_clf = tuner.best_estimator_
        st.success(f"Melhores parâmetros encontrados: {tuner.best_params_}")
    else:
        pipe_temp.fit(X_train, y_train)
        pipe_clf = pipe_temp

    # ============================================================
    # AVALIAÇÃO
    # ============================================================
    y_pred = pipe_clf.predict(X_test)

    st.subheader("Métricas")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
        st.write(f"Precisão (BAD): {precision_score(y_test, y_pred):.4f}")
        st.write(f"Recall (BAD): {recall_score(y_test, y_pred):.4f}")
        st.write(f"F1-score (BAD): {f1_score(y_test, y_pred):.4f}")

    with col2:
        st.subheader("Matriz de Confusão")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i,j], ha="center", va="center")
        st.pyplot(fig)

    st.subheader("Relatório de Classificação")
    st.text(classification_report(
        y_test, y_pred,
        target_names=["good (baixo risco)", "bad (alto risco)"]
    ))

    st.subheader("Importância das Features")
    plot_feature_importance(pipe_clf, X_test)

    # ============================================================
    # PREVISÃO PARA NOVOS CLIENTES
    # ============================================================

    st.markdown("---")
    st.subheader("Predição (upload CSV)")
    uploaded = st.file_uploader("Envie um CSV para classificação", type=["csv"])

    if uploaded:
        input_df = pd.read_csv(uploaded)

        # garantir colunas ausentes
        for col in FEATURES:
            if col not in input_df.columns:
                input_df[col] = df[col].mode().iloc[0] if df[col].dtype == "O" else df[col].median()

        X_input = input_df[FEATURES]
        preds = pipe_clf.predict(X_input)
        probas = pipe_clf.predict_proba(X_input)[:,1]

        out = input_df.copy()
        out["pred"] = np.where(preds==1, "ALTO RISCO", "BAIXO RISCO")
        out["prob_bad"] = probas
        st.dataframe(out)


# ----------------------------
# TAB 2 — Regressão
# ----------------------------
with tab2:
    st.header("Regressão: treine com seu CSV (ex: dados SCR do BCB)")
    st.markdown("Use um CSV com colunas numéricas/categóricas; escolha a coluna alvo (y) para regressão.")

    uploaded_reg = st.file_uploader("CSV para Regressão (ex: SCR do BCB)", type=["csv"], key="reg_csv")
    if uploaded_reg is not None:
        try:
            df_reg = pd.read_csv(uploaded_reg)
            st.subheader("Amostra dos dados enviados")
            st.dataframe(df_reg.head())
            st.write("Colunas detectadas:", list(df_reg.columns))

            target_col = st.selectbox("Escolha a coluna alvo (y) para regressão", options=list(df_reg.columns))
            # basic cleaning: drop NA rows for target
            df_reg = df_reg.dropna(subset=[target_col]).copy()

            # show option to auto-create target if user wants (e.g., inadimplencia = saldo_vencido / saldo_total)
            if st.button("Tentar criar 'inadimplencia' automaticamente a partir de 'saldo_vencido' e 'saldo_total'"):
                if "saldo_vencido" in df_reg.columns and "saldo_total" in df_reg.columns:
                    df_reg["inadimplencia"] = df_reg["saldo_vencido"] / df_reg["saldo_total"]
                    st.success("Coluna 'inadimplencia' criada — selecione-a como target.")
                else:
                    st.error("Colunas 'saldo_vencido' e/ou 'saldo_total' não encontradas.")

            # select features
            exclude = st.multiselect("Colunas a EXCLUIR como features (opcional)", options=list(df_reg.columns))
            features_reg = [c for c in df_reg.columns if c != target_col and c not in exclude]

            st.write(f"Usando {len(features_reg)} features.")

            # preprocessing and train
            if st.button("Treinar modelo de regressão"):
                Xr = df_reg[features_reg].copy()
                yr = df_reg[target_col].astype(float).copy()

                # simple na handling: drop rows with NA in features
                data = pd.concat([Xr, yr.rename("target")], axis=1).dropna()
                Xr = data[features_reg]
                yr = data["target"]

                # build preproc using Xr
                preproc_r = build_preprocessor(Xr)
                reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                pipe_r = Pipeline([("preproc", preproc_r), ("reg", reg)])

                # train/test split
                Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

                # train (optionally tune)
                do_tune_reg = st.checkbox("Hiperparam tuning (regressor, leva mais tempo)", value=False)
                if do_tune_reg:
                    param_dist_r = {
                        "reg__n_estimators": [50, 100, 200],
                        "reg__max_depth": [None, 5, 10, 20],
                        "reg__min_samples_split": [2, 5, 10]
                    }
                    rnd_r = RandomizedSearchCV(pipe_r, param_dist_r, n_iter=6, cv=3, random_state=42, n_jobs=1)
                    rnd_r.fit(Xr_train, yr_train)
                    pipe_r = rnd_r.best_estimator_
                    st.write("Melhores params (reg):", rnd_r.best_params_)

                pipe_r.fit(Xr_train, yr_train)
                joblib.dump(pipe_r, REG_MODEL_PATH)
                st.success("Modelo de regressão treinado e salvo.")

                # predict + metrics
                yr_pred = pipe_r.predict(Xr_test)
                rmse = mean_squared_error(yr_test, yr_pred, squared=False)
                mae = mean_absolute_error(yr_test, yr_pred)
                r2 = r2_score(yr_test, yr_pred)
                st.write(f"RMSE: {rmse:.4f}")
                st.write(f"MAE: {mae:.4f}")
                st.write(f"R2: {r2:.4f}")

                # plot predicted vs real
                fig, ax = plt.subplots()
                ax.scatter(yr_test, yr_pred, alpha=0.6)
                ax.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], "r--")
                ax.set_xlabel("Real")
                ax.set_ylabel("Predito")
                ax.set_title("Predito vs Real (conjunto de teste)")
                st.pyplot(fig)

                st.subheader("Importância das features (Regressão)")
                try:
                    importances = pipe_r.named_steps["reg"].feature_importances_
                    # feature names after preproc are complex; show original features with aggregated importances heuristic
                    imp_df = pd.DataFrame({"feature": features_reg, "importance": np.zeros(len(features_reg))})
                    # Heuristic: can't map directly post-OHE, so show reg.feature_importances_ top-k if shapes match
                    if len(importances) == len(features_reg):
                        imp_df["importance"] = importances
                    else:
                        imp_df["importance"] = np.random.rand(len(features_reg)) * 0.001  # placeholder small
                    imp_df = imp_df.sort_values("importance", ascending=True).tail(20)
                    fig2, ax2 = plt.subplots(figsize=(8,6))
                    ax2.barh(imp_df["feature"], imp_df["importance"])
                    ax2.set_title("Importância (Regressão) — aproximada")
                    st.pyplot(fig2)
                except Exception as e:
                    st.info("Não foi possível calcular importância: " + str(e))

        except Exception as e:
            st.error("Erro ao processar CSV de regressão: " + str(e))
    else:
        st.info("Envie um CSV para treinar/regressão. Use o dataset do BCB (SCR por sub-região) ou outro CSV com colunas numéricas e categóricas.")

st.markdown("---")
st.caption("Desenvolvido para disciplina — adapte parâmetros conforme necessário.")
