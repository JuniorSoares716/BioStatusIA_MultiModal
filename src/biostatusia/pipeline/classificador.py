import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC

from biostatusia.pipeline.avaliacao_modelos import balancear, calcular_score_clinico, selecionar_melhor_modelo

MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models"


def _achatar_biomarcadores(d: dict, prefixo: str = "") -> dict:
    """Achata um dict aninhado de biomarcadores em pares chave->valor numérico
    (notação 'grupo.campo'), ignorando valores não numéricos. Genérico o
    suficiente para qualquer modalidade (sinal temporal, DICOM, volume 3D)
    sem precisar de um vetorizador específico por família."""
    saida: dict = {}
    for k, v in d.items():
        chave = f"{prefixo}.{k}" if prefixo else str(k)
        if isinstance(v, dict):
            saida.update(_achatar_biomarcadores(v, chave))
        elif isinstance(v, bool):
            continue
        elif isinstance(v, (int, float)):
            saida[chave] = float(v)
    return saida


def vetorizar_biomarcadores_generico(registros: list[dict]) -> tuple[np.ndarray, list[str]]:
    """
    Converte uma lista de registros {"biomarcadores": {...}, ...} (formato
    comum a `biomarcadores.json` e a `extrair_lote_temporal/_dicom/_volumetrico`)
    num array X, usando a união de todas as chaves numéricas encontradas nos
    registros (preenchendo com NaN onde uma chave não existir numa amostra —
    a imputação de NaN já é tratada rio abaixo, no treino).
    """
    achatados = [_achatar_biomarcadores(r.get("biomarcadores", {}) or {}) for r in registros]
    todas_chaves = sorted(set().union(*[set(a.keys()) for a in achatados])) if achatados else []
    X = np.array([[a.get(c, np.nan) for c in todas_chaves] for a in achatados], dtype=float)
    return X, todas_chaves


def _vetor(biomarcadores: dict) -> list[float]:
    m = biomarcadores.get("morfologia", {})
    t = biomarcadores.get("textura_glcm", {})
    d = biomarcadores.get("distribuicao_intensidade", {})
    return [
        m.get("circularidade", 0),
        m.get("solidez", 0),
        t.get("contraste", 0),
        t.get("homogeneidade", 0),
        t.get("energia", 0),
        t.get("entropia", 0),
        d.get("snr", 0),
        d.get("assimetria", 0),
        d.get("curtose", 0),
    ]


def treinar(registros: list[dict]) -> dict:
    """
    registros: [{"biomarcadores": dict, "label": 0|1}, ...]
    Treina SVM e RandomForest, retorna métricas completas para os gráficos.
    """
    X = np.array([_vetor(r["biomarcadores"]) for r in registros])
    y = np.array([r["label"] for r in registros])
    return treinar_vetores(X, y)


def treinar_vetores(X: np.ndarray, y: np.ndarray, scaling: str = "standard",
                    familia: str = "", feature_names: list[str] | None = None,
                    balancear_treino: str = "smote", imputacao: str = "auto",
                    selecao_features: bool = True) -> dict:
    """Versão genérica com escalamento dinâmico baseado na estratégia decidida.

    Protocolo de validação (nested, sem vazamento entre seleção e avaliação final):
    1) Split externo 80/20 (estratificado). O fold de 20% é reservado e só é
       tocado no passo (3) — nunca participa da escolha do vencedor.
    1.5) Seleção de features via SHAP (opcional, `selecao_features=True` por
       padrão): um RandomForest "sonda" é treinado SÓ no fold de treino, o
       ranking de importância SHAP decide quais features ficam, e o mesmo
       recorte de colunas é aplicado ao treino e ao teste — nunca ao contrário.
    2) Seleção do vencedor: CV repetida (5-fold x 3) executada inteiramente
       dentro do fold de 80% de treino; imputação, escalonamento e
       balanceamento são reajustados a cada fold, usando só a partição de
       treino daquele fold. O modelo vencedor é o de maior score clínico
       médio nessa CV interna.
    3) Avaliação final: cada um dos 6 candidatos é treinado uma vez no
       treino completo (80%) e avaliado uma vez no teste (20%) — essas
       métricas finais são as reportadas/exibidas, mas NÃO influenciam
       qual modelo foi escolhido (isso já foi decidido no passo 2).
    """
    from sklearn.base import clone
    from sklearn.model_selection import RepeatedStratifiedKFold
    from biostatusia.pipeline.dados_tabulares import imputar_treino_teste, selecionar_features_shap
    from biostatusia.pipeline.avaliacao_modelos import calcular_metricas_classificacao

    classes = sorted(set(y.tolist()))

    def _scaler():
        if scaling == "robust":
            return RobustScaler()
        if scaling == "standard":
            return StandardScaler()
        return None

    # T1 — split externo ANTES de qualquer ajuste. O fold de teste (20%) só
    # será usado no passo 3 (avaliação final), nunca na seleção do vencedor.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_raw, X_test_raw, info_imputacao = imputar_treino_teste(
        X_train_raw, X_test_raw, metodo=imputacao
    )

    # Seleção de features via SHAP — decidida SÓ com o treino; o mesmo recorte
    # de colunas é aplicado ao teste (nunca o contrário, para não vazar dados).
    info_selecao_features = {"aplicado": False}
    if selecao_features:
        idx_features, info_selecao_features = selecionar_features_shap(
            X_train_raw, y_train, feature_names=feature_names
        )
        if info_selecao_features.get("aplicado"):
            X_train_raw = X_train_raw[:, idx_features]
            X_test_raw = X_test_raw[:, idx_features]
            if feature_names and len(feature_names) == info_selecao_features["n_original"]:
                feature_names = [feature_names[i] for i in idx_features]

    # SVM (RBF, probability=True) tem custo super-quadrático no número de
    # amostras — medido em ~9.5s para um único treino com N=15.000 (e o
    # protocolo de CV interna treina ~16 vezes: 5-fold×3 repetições + o
    # treino final). Em datasets grandes (dezenas de milhares de linhas,
    # comuns em CSVs baixados do Kaggle), isso passa de 10 minutos parado
    # sem nenhum log — parece travado, mas só está muito lento. Acima do
    # limite, pula o SVM (os outros 5 candidatos escalam bem melhor) e
    # reporta isso de forma explícita, em vez de deixar a pessoa esperando
    # sem explicação.
    LIMITE_AMOSTRAS_SVM = 10_000
    modelos_pulados: dict[str, str] = {}
    modelos = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=2000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
    }
    if len(X_train_raw) > LIMITE_AMOSTRAS_SVM:
        del modelos["SVM"]
        modelos_pulados["SVM"] = (
            f"Pulado: {len(X_train_raw)} amostras de treino > {LIMITE_AMOSTRAS_SVM} — "
            f"SVM (kernel RBF) tem custo super-quadrático no número de amostras e "
            f"levaria muitos minutos para treinar nesse tamanho. Os outros 5 "
            f"classificadores escalam bem e continuam rodando normalmente."
        )

    # ── Passo 2 — Seleção via CV interna (100% dentro do treino) ────────────
    n_min = int(np.min(np.bincount(y_train))) if len(y_train) else 2
    n_splits = max(2, min(5, n_min))
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=42)

    campos_cv = ("sensibilidade", "especificidade", "f1", "auc", "acuracia", "mcc", "kappa", "ece")
    metricas_cv: dict = {}
    for nome, modelo_base in modelos.items():
        cv_scores: dict = {k: [] for k in campos_cv}
        for fold_train, fold_val in rskf.split(X_train_raw, y_train):
            Xf_tr_raw, Xf_val_raw = X_train_raw[fold_train], X_train_raw[fold_val]
            Xf_tr_raw, Xf_val_raw, _ = imputar_treino_teste(Xf_tr_raw, Xf_val_raw, metodo=imputacao)
            sc_fold = _scaler()
            if sc_fold is not None:
                sc_fold.fit(Xf_tr_raw)
                Xf_tr, Xf_val = sc_fold.transform(Xf_tr_raw), sc_fold.transform(Xf_val_raw)
            else:
                Xf_tr, Xf_val = Xf_tr_raw, Xf_val_raw
            yf_tr = y_train[fold_train]
            Xf_tr, yf_tr, _ = balancear(Xf_tr, yf_tr, metodo=balancear_treino)

            m = clone(modelo_base)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(Xf_tr, yf_tr)
            y_v = y_train[fold_val]
            y_p = m.predict(Xf_val)
            y_pr = m.predict_proba(Xf_val)
            mf = calcular_metricas_classificacao(y_v, y_p, y_pr, classes=classes)
            for k in campos_cv:
                cv_scores[k].append(mf[k])

        medias = {k: round(float(np.mean(v)), 4) for k, v in cv_scores.items()}
        metricas_cv[nome] = {
            k: {"media": medias[k], "desvio": round(float(np.std(v)), 4)}
            for k, v in cv_scores.items()
        }
        metricas_cv[nome]["score_clinico"] = round(float(calcular_score_clinico({
            "auc": medias["auc"], "mcc": medias["mcc"],
            "ece": medias["ece"], "sensibilidade": medias["sensibilidade"],
        })), 4)

    metricas_para_selecao = {
        nome: {
            "auc": metricas_cv[nome]["auc"]["media"],
            "acuracia": metricas_cv[nome]["acuracia"]["media"],
            "sensibilidade": metricas_cv[nome]["sensibilidade"]["media"],
            "mcc": metricas_cv[nome]["mcc"]["media"],
            "ece": metricas_cv[nome]["ece"]["media"],
            "score_clinico": metricas_cv[nome]["score_clinico"],
        }
        for nome in modelos
    }
    # Vencedor decidido AQUI, só com a CV interna — o teste held-out (20%)
    # ainda não foi tocado e será usado só para reportar a métrica final.
    selecao = selecionar_melhor_modelo(metricas_para_selecao)
    melhor = selecao["melhor_modelo"]

    # ── Passo 1 (continuação) / Passo 3 — treino final + avaliação held-out ──
    scaler = _scaler()
    if scaler is not None:
        scaler.fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        X_train, X_test = X_train_raw, X_test_raw

    # Balanceamento do treino final (SMOTE/ADASYN) — evita que o classificador
    # aprenda a "sempre prever a classe majoritária" em bases desbalanceadas
    # (ex.: datasets clínicos com poucos casos positivos ou negativos raros).
    # Falha de forma segura (mantém os dados originais) se o imblearn não
    # estiver disponível ou a classe minoritária for pequena demais.
    X_train, y_train, info_balanceamento = balancear(X_train, y_train, metodo=balancear_treino)

    resultado: dict = {
        "metricas": {}, "metricas_cv": metricas_cv, "roc_data": {}, "confusion_matrix": {},
        "balanceamento": info_balanceamento, "imputacao": info_imputacao,
        "selecao_features_shap": info_selecao_features,
        "n_classes": len(classes), "classes": classes,
        "modelos_pulados": modelos_pulados,
        "protocolo_validacao": {
            "selecao": f"CV interna {n_splits}-fold x3 repetições, dentro do treino (80%)",
            "avaliacao_final": "held-out 20%, nunca usado na seleção do vencedor",
            "selecao_features": info_selecao_features.get("metodo") if info_selecao_features.get("aplicado") else "nenhuma (dimensionalidade já baixa ou indisponível)",
        },
    }

    MODEL_DIR.mkdir(exist_ok=True)

    for nome, modelo in modelos.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            modelo.fit(X_train, y_train)
            t_treino = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = modelo.predict(X_test)
        latencia_ms = round((time.perf_counter() - t0) * 1000 / max(1, len(X_test)), 4)
        y_prob = modelo.predict_proba(X_test)

        metricas_finais = calcular_metricas_classificacao(y_test, y_pred, y_prob, classes=classes)
        metricas_finais["latencia_inferencia_ms"] = latencia_ms
        metricas_finais["tempo_treino_s"] = round(t_treino, 3)
        # Score clínico do held-out — exibido para referência/comparação, mas
        # NÃO é usado para decidir o vencedor (essa decisão já foi tomada via
        # metricas_cv, acima, antes de este fold de teste ser sequer olhado).
        metricas_finais["score_clinico"] = round(float(calcular_score_clinico(metricas_finais)), 4)
        resultado["metricas"][nome] = metricas_finais

        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            resultado["roc_data"][nome] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        resultado["confusion_matrix"][nome] = cm.tolist()

        with open(MODEL_DIR / f"modelo_{nome.lower()}.pkl", "wb") as f:
            pickle.dump({"modelo": modelo, "scaler": scaler}, f)

    # Vencedor já decidido pela CV interna (passo 2) — aqui só anexamos os
    # metadados de seleção ao resultado final.
    resultado.update(selecao)

    # Persistir o vencedor do pódio para inferência individual (Aba 4 / Laudo Individual).
    try:
        from biostatusia.pipeline.inferencia import salvar_modelo_vencedor
        salvar_modelo_vencedor(
            nome=melhor, modelo=modelos[melhor], scaler=scaler,
            familia=familia or "IMG", feature_names=feature_names,
            metricas=resultado["metricas"][melhor],
        )
    except Exception:
        pass

    return resultado


def classificar(biomarcadores: dict) -> tuple[str, float]:
    """Classifica usando o melhor modelo salvo em disco. Retorna (categoria, probabilidade)."""
    nomes_possiveis = [
        "logisticregression", "knn", "svm", "randomforest", "gradientboosting", "mlp"
    ]
    for nome in nomes_possiveis:
        caminho = MODEL_DIR / f"modelo_{nome}.pkl"
        if caminho.exists():
            with open(caminho, "rb") as f:
                salvo = pickle.load(f)
            X = np.array([_vetor(biomarcadores)])
            X_scaled = salvo["scaler"].transform(X) if salvo.get("scaler") is not None else X
            prob = float(salvo["modelo"].predict_proba(X_scaled)[0][1])
            return ("MALIGNO" if prob > 0.5 else "BENIGNO"), round(prob, 4)
    return "INDEFINIDO", 0.0
