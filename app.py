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
