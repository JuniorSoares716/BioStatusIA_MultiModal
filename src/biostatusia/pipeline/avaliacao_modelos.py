"""
Avaliação enriquecida de modelos — Fase 3 do plano de expansão.
Protocolo: CV estratificada REPETIDA (5x3) + teste A/B (McNemar) + métricas clínicas.
Sem vazamento: escala e balanceamento ajustados por partição (T1/T2);
seleção com piso clínico de sensibilidade (T3); intervalos de confiança 95% (T4).
"""
import time
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


_MODELOS = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
}


# ── Balanceamento de classes (SMOTE / ADASYN) ─────────────────────────────────

def balancear(X: np.ndarray, y: np.ndarray, metodo: str = "smote") -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Reamostra o conjunto de treino para equilibrar as classes.
    Só aplica se houver desbalanceamento e amostras suficientes (SMOTE precisa
    de ao menos k_neighbors+1 exemplos na classe minoritária). Falha de forma
    segura devolvendo os dados originais.
    """
    info = {"aplicado": False, "metodo": metodo}
    classes, contagens = np.unique(y, return_counts=True)
    if len(classes) < 2:
        info["motivo"] = "classe única"
        return X, y, info

    minoria = int(contagens.min())
    info["distribuicao_original"] = {int(c): int(n) for c, n in zip(classes, contagens)}
    if contagens.max() == contagens.min():
        info["motivo"] = "já balanceado"
        return X, y, info

    k = min(5, minoria - 1)
    if k < 1:
        info["motivo"] = f"classe minoritária com {minoria} amostras — insuficiente para reamostragem"
        return X, y, info

    try:
        if metodo == "adasyn":
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(random_state=42, n_neighbors=k)
        else:
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=42, k_neighbors=k)
        X_bal, y_bal = sampler.fit_resample(X, y)
        cls_b, cont_b = np.unique(y_bal, return_counts=True)
        info["aplicado"] = True
        info["distribuicao_balanceada"] = {int(c): int(n) for c, n in zip(cls_b, cont_b)}
        return X_bal, y_bal, info
    except Exception as e:
        info["erro"] = str(e)
        return X, y, info


# ── Seleção de features (RFE / PCA) ───────────────────────────────────────────

def selecionar_features(X: np.ndarray, y: np.ndarray, metodo: str = "rfe",
                        n_features: int | None = None):
    """
    Reduz a dimensionalidade por importância (RFE) ou variância (PCA).
    Retorna (X_reduzido, transformador_ou_None, info). Segura para datasets pequenos.
    """
    info = {"metodo": metodo, "aplicado": False, "n_original": X.shape[1]}
    n_alvo = n_features or max(2, min(X.shape[1], X.shape[0] // 3, 20))
    if X.shape[1] <= n_alvo:
        info["motivo"] = "dimensionalidade já baixa"
        return X, None, info

    try:
        if metodo == "pca":
            from sklearn.decomposition import PCA
            transf = PCA(n_components=n_alvo, random_state=42)
            X_red = transf.fit_transform(X)
            info["variancia_explicada"] = round(float(transf.explained_variance_ratio_.sum()), 4)
        else:
            from sklearn.feature_selection import RFE
            transf = RFE(RandomForestClassifier(n_estimators=50, random_state=42),
                         n_features_to_select=n_alvo)
            X_red = transf.fit_transform(X, y)
            info["mascara_selecionadas"] = transf.support_.tolist()
        info["aplicado"] = True
        info["n_selecionadas"] = n_alvo
        return X_red, transf, info
    except Exception as e:
        info["erro"] = str(e)
        return X, None, info


def avaliar_modelos(X: np.ndarray, y: np.ndarray, familia: str = "",
                    balancear_treino: str = "smote",
                    feature_names: list[str] | None = None,
                    persistir_vencedor: bool = True) -> dict:
    """
    Avalia todos os modelos via 5-fold CV + conjunto de teste 20%.
    Retorna métricas completas: sensibilidade, especificidade, precisão, recall,
    F1, AUC, MCC (Matthews), Kappa (Cohen), ECE, latência e tempo de treino.
    Aplica balanceamento (SMOTE/ADASYN) apenas no conjunto de treino, e calcula
    importância de features por SHAP para o modelo vencedor.

    Funciona para classificação BINÁRIA (2 classes) e MULTI-CLASSE (3+) — as
    métricas usam o significado clínico exato (sensibilidade/especificidade
    da classe positiva) quando binário, e médias macro/one-vs-rest quando
    multi-classe (ver `calcular_metricas_classificacao`).
    """
    if len(X) < 10 or len(set(y.tolist())) < 2:
        return {"aviso": f"Treino não executado: {len(X)} amostras, {len(set(y.tolist()))} classes."}

    classes = sorted(set(y.tolist()))

    # T1 — split ANTES de qualquer escalonamento/balanceamento (sem vazamento).
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaler ajustado SÓ no treino; o teste é apenas transformado.
    scaler = StandardScaler().fit(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Treino final (compartilhado entre modelos): escala + balanceamento
    # aplicados apenas ao treino — o teste permanece intocado.
    X_train_scaled = scaler.transform(X_train_raw)
    X_train_bal, y_train_bal, info_balanceamento = balancear(
        X_train_scaled, y_train, metodo=balancear_treino
    )

    # T2/T4 — CV estratificada REPETIDA; escala e balanceamento reajustados
    # DENTRO de cada fold, sobre a partição de treino do fold.
    n_min = int(np.min(np.bincount(y_train)))
    n_splits = max(2, min(5, n_min))
    n_repeats = 3
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )
    resultado: dict = {
        "familia": familia,
        "n_amostras": len(X),
        "n_classes": len(classes),
        "classes": classes,
        "balanceamento": info_balanceamento,
        "cv_protocolo": {"n_splits": n_splits, "n_repeats": n_repeats},
        "metricas": {},
        "metricas_cv": {},
        "roc_data": {},
        "confusion_matrix": {},
        "comparacao_ab": {},
        "shap": {},
        "modelos_pulados": {},
    }

    predicoes_teste: dict = {}
    modelos_treinados: dict = {}
    campos_cv = ("sensibilidade", "especificidade", "f1", "auc", "acuracia", "mcc", "kappa", "ece")

    # SVM (RBF, probability=True) tem custo super-quadrático no número de
    # amostras — em datasets grandes, os ~16 treinos do protocolo de CV
    # interna (5-fold×3 + final) podem passar de 10 minutos sem nenhum log,
    # parecendo travado. Usa uma cópia LOCAL de _MODELOS (nunca modifica o
    # dict global, que outros módulos como fusao_multimodal.py também usam).
    LIMITE_AMOSTRAS_SVM = 10_000
    modelos_locais = dict(_MODELOS)
    if len(X_train_raw) > LIMITE_AMOSTRAS_SVM:
        del modelos_locais["SVM"]
        resultado["modelos_pulados"]["SVM"] = (
            f"Pulado: {len(X_train_raw)} amostras de treino > {LIMITE_AMOSTRAS_SVM} — "
            f"SVM (kernel RBF) tem custo super-quadrático no número de amostras e "
            f"levaria muitos minutos para treinar nesse tamanho. Os outros 5 "
            f"classificadores escalam bem e continuam rodando normalmente."
        )

    for nome, modelo_base in modelos_locais.items():
        # ── T2/T4 — CV repetida; escala e balanceamento POR fold ───────────
        cv_scores: dict = {k: [] for k in campos_cv}

        for fold_train, fold_val in rskf.split(X_train_raw, y_train):
            sc_fold = StandardScaler().fit(X_train_raw[fold_train])
            Xf_tr = sc_fold.transform(X_train_raw[fold_train])
            Xf_val = sc_fold.transform(X_train_raw[fold_val])
            yf_tr = y_train[fold_train]
            Xf_tr, yf_tr, _ = balancear(Xf_tr, yf_tr, metodo=balancear_treino)

            m = _clonar_modelo(nome)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(Xf_tr, yf_tr)
            y_v = y_train[fold_val]
            y_p = m.predict(Xf_val)
            y_pr = m.predict_proba(Xf_val)
            mf = calcular_metricas_classificacao(y_v, y_p, y_pr, classes=classes)
            for k in campos_cv:
                cv_scores[k].append(mf[k])

        medias_cv = {k: round(float(np.mean(v)), 4) for k, v in cv_scores.items()}
        resultado["metricas_cv"][nome] = {
            k: {
                "media": medias_cv[k],
                "desvio": round(float(np.std(v)), 4),
                "ic95": _ic95(v),
            }
            for k, v in cv_scores.items()
        }
        # Score clínico calculado SÓ com a CV interna — é isso que decide o
        # vencedor (ver selecionar_melhor_modelo abaixo), nunca as métricas
        # do fold de teste held-out.
        resultado["metricas_cv"][nome]["score_clinico"] = round(float(calcular_score_clinico({
            "auc": medias_cv["auc"], "mcc": medias_cv["mcc"],
            "ece": medias_cv["ece"], "sensibilidade": medias_cv["sensibilidade"],
        })), 4)

        # ── Treino final + avaliação no teste (treino escalado+balanceado) ─
        modelo_final = _clonar_modelo(nome)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo_final.fit(X_train_bal, y_train_bal)
        t_treino = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = modelo_final.predict(X_test)
        y_prob = modelo_final.predict_proba(X_test)
        latencia_ms = round((time.perf_counter() - t0) * 1000, 2)

        predicoes_teste[nome] = y_pred
        modelos_treinados[nome] = modelo_final

        metricas_finais = calcular_metricas_classificacao(y_test, y_pred, y_prob, classes=classes)
        metricas_finais["latencia_inferencia_ms"] = latencia_ms
        metricas_finais["tempo_treino_s"] = round(t_treino, 3)
        # Score clínico multiobjetivo para o modelo
        metricas_finais["score_clinico"] = round(float(calcular_score_clinico(metricas_finais)), 4)
        resultado["metricas"][nome] = metricas_finais

        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            resultado["roc_data"][nome] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        resultado["confusion_matrix"][nome] = cm.tolist()

    # ── Seleção BioStatusIA: Multiobjetivo Clinicamente Orientada + veto de piso de sensibilidade ──
    # Decidida SOMENTE com a CV interna (metricas_cv) — o fold de teste (held-out)
    # usado em resultado["metricas"] nunca participa da escolha do vencedor.
    metricas_para_selecao = {
        nome: {
            "auc": mcv["auc"]["media"], "acuracia": mcv["acuracia"]["media"],
            "sensibilidade": mcv["sensibilidade"]["media"], "mcc": mcv["mcc"]["media"],
            "ece": mcv["ece"]["media"], "score_clinico": mcv["score_clinico"],
        }
        for nome, mcv in resultado["metricas_cv"].items()
    }
    selecao = selecionar_melhor_modelo(metricas_para_selecao)
    resultado.update(selecao)
    melhor = selecao["melhor_modelo"]
    resultado["protocolo_validacao"] = {
        "selecao": f"CV interna {n_splits}-fold x{n_repeats} repetições, dentro do treino (80%)",
        "avaliacao_final": "held-out 20%, nunca usado na seleção do vencedor",
    }

    # ── Interpretabilidade SHAP para o modelo vencedor ────────────────────
    resultado["shap"] = _shap_importancia(
        modelos_treinados[melhor], X_train_bal, X_test, feature_names, melhor
    )

    # ── Persistir o vencedor do pódio para inferência individual ──────────
    if persistir_vencedor:
        from biostatusia.pipeline.inferencia import salvar_modelo_vencedor
        resultado["modelo_persistido"] = salvar_modelo_vencedor(
            nome=melhor, modelo=modelos_treinados[melhor], scaler=scaler,
            familia=familia, feature_names=feature_names,
            metricas=resultado["metricas"][melhor],
        )

    # ── Teste A/B: McNemar entre melhor e baseline (primeiro modelo) ──────
    baseline = list(predicoes_teste.keys())[0]
    if baseline != melhor and baseline in predicoes_teste:
        resultado["comparacao_ab"] = _mcnemar_test(
            y_test,
            predicoes_teste[baseline],
            predicoes_teste[melhor],
            baseline,
            melhor,
        )

    return resultado


def calcular_score_clinico(metricas: dict, s_min: float = 0.80,
                           w_auc: float = 0.40, w_mcc: float = 0.40,
                           w_ece: float = 0.20, penalty_lambda: float = 1.0) -> float:
    """
    Função de Seleção AutoML Clinicamente Orientada (BioStatusIA).
    Equilibra capacidade discriminatória (AUROC), robustez contra desbalanceamento (MCC)
    e confiabilidade probabilística (ECE), aplicando penalidade para modelos que falham
    no piso de sensibilidade clínica exigido para triagem / screening (S_min).
    """
    auc = metricas.get("auc", 0.0)
    mcc = metricas.get("mcc", 0.0)
    ece = metricas.get("ece", 0.0)
    sens = metricas.get("sensibilidade", 0.0)

    # Normaliza MCC de [-1, 1] para [0, 1]
    mcc_norm = max(0.0, (mcc + 1.0) / 2.0)
    # ECE é penalizado (menor é melhor)
    ece_penalty = min(1.0, max(0.0, ece))

    # Score base multiobjetivo
    score_base = (w_auc * auc) + (w_mcc * mcc_norm) - (w_ece * ece_penalty)

    # Penalidade por déficit de sensibilidade mínima
    deficit_sens = max(0.0, s_min - sens)
    penalidade = penalty_lambda * (deficit_sens ** 1.5)

    return float(score_base - penalidade)


def selecionar_melhor_modelo(metricas: dict, s_min: float = 0.80) -> dict:
    """
    Seleção final do modelo vencedor (BioStatusIA) — com veto de segurança clínica.

    Corrige a falha identificada em revisão: antes, o "vencedor clínico" era
    simplesmente o argmax de calcular_score_clinico() em TODOS os candidatos.
    Como a penalidade por sensibilidade insuficiente é suave (não um corte
    rígido), um dataset onde NENHUM dos 6 classificadores atinge o piso de
    sensibilidade (S_min) ainda produzia um "vencedor" — o menos pior do lote —
    sem qualquer sinalização de que o critério de segurança clínica não foi
    cumprido por ninguém (ex.: SVM com Sensibilidade=0.0 e MCC negativo).

    Agora: o vencedor clínico é escolhido apenas entre os candidatos que
    atingem sensibilidade >= s_min. Se nenhum atingir, ainda é necessário
    devolver algum modelo (o pipeline não pode simplesmente parar), mas o
    resultado sinaliza explicitamente `piso_sensibilidade_atingido=False` e
    um aviso, para que a interface e o laudo não apresentem esse caso como
    uma seleção clínica normal.
    """
    nomes = list(metricas.keys())

    # Seleção Convencional (critério clássico: maior AUC bruta / acurácia) — inalterada.
    melhor_convencional = max(nomes, key=lambda k: (metricas[k]["auc"], metricas[k]["acuracia"]))

    elegiveis = [n for n in nomes if metricas[n].get("sensibilidade", 0.0) >= s_min]
    piso_atingido = len(elegiveis) > 0
    pool = elegiveis if piso_atingido else nomes

    melhor_clinico = max(pool, key=lambda k: metricas[k]["score_clinico"])
    melhor = melhor_clinico

    aviso = None
    if not piso_atingido:
        aviso = (
            f"Nenhum dos {len(nomes)} classificadores atingiu o piso mínimo de "
            f"sensibilidade clínica (S_min={s_min:.2f}) neste dataset. O modelo "
            f"'{melhor}' foi retornado apenas como o menos inadequado do lote "
            f"(Sensibilidade={metricas[melhor]['sensibilidade']:.3f}) e NÃO deve "
            f"ser interpretado como uma seleção clinicamente validada — recomenda-se "
            f"revisão manual, re-balanceamento ou coleta adicional de dados antes de "
            f"qualquer uso além de suporte exploratório."
        )

    return {
        "melhor_modelo": melhor,
        "melhor_modelo_clinico": melhor_clinico,
        "melhor_modelo_convencional": melhor_convencional,
        "selecao_divergente": melhor_clinico != melhor_convencional,
        "piso_sensibilidade_atingido": piso_atingido,
        "aviso_piso_sensibilidade": aviso,
        "criterio_selecao": (
            f"Multiobjetivo Clínico [Score={metricas[melhor]['score_clinico']:.4f} | "
            f"AUROC={metricas[melhor]['auc']:.2f}, MCC={metricas[melhor]['mcc']:.2f}, "
            f"ECE={metricas[melhor]['ece']:.3f}, Sens={metricas[melhor]['sensibilidade']:.2f}]"
            + ("" if piso_atingido else " — ⚠ piso de sensibilidade NÃO atingido por nenhum candidato")
        ),
    }


def _ic95(valores: list[float]) -> list[float]:
    """Intervalo de confiança 95% (t-Student) da média das métricas por fold — T4."""
    v = np.asarray(valores, dtype=float)
    n = len(v)
    if n < 2:
        return [round(float(v.mean()), 4), round(float(v.mean()), 4)] if n else [0.0, 0.0]
    from scipy.stats import t as t_dist
    media = float(v.mean())
    erro = float(v.std(ddof=1) / np.sqrt(n))
    margem = float(t_dist.ppf(0.975, df=n - 1)) * erro
    return [round(media - margem, 4), round(media + margem, 4)]


def _calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) — versão binária (probabilidade da
    classe positiva). Mantida para compatibilidade exata com o cálculo
    original; usada como o "caso binário" dentro de
    `calcular_metricas_classificacao`."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return ece


def _calibration_error_multiclasse(y_true: np.ndarray, y_proba: np.ndarray,
                                    classes: list, n_bins: int = 10) -> float:
    """
    ECE para multi-classe (calibração de confiança): usa a confiança da
    predição (probabilidade da classe mais provável) e se essa predição
    acertou ou não — a generalização padrão do ECE binário para N classes.
    Para N=2, dá um número conceitualmente equivalente ao binário, mas não
    idêntico bit-a-bit (por isso o caminho binário usa `_calibration_error`
    diretamente, para preservar o comportamento exato já testado).
    """
    if y_proba is None or y_proba.size == 0:
        return 0.0
    idx_pred = np.argmax(y_proba, axis=1)
    confianca = y_proba[np.arange(len(y_proba)), idx_pred]
    classes_arr = np.asarray(classes)
    pred_rotulo = classes_arr[idx_pred]
    acertos = (pred_rotulo == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (confianca >= bins[i]) & (confianca < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = acertos[mask].mean()
        conf = confianca[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def calcular_metricas_classificacao(y_true: np.ndarray, y_pred: np.ndarray,
                                     y_proba: np.ndarray | None = None,
                                     classes: list | None = None) -> dict:
    """
    Conjunto padrão de métricas do BioStatusIA — generalizado para binário
    (2 classes) e multi-classe (3+), num único lugar (antes essa lógica
    estava duplicada em `avaliacao_modelos.py` e `classificador.py`, cada
    uma assumindo rigidamente 2 classes).

    Para BINÁRIO, "sensibilidade"/"especificidade" mantêm o significado
    clínico exato (recall da classe positiva/negativa) — comportamento
    idêntico ao que já existia antes desta generalização. Para
    MULTI-CLASSE, viram médias macro (recall médio entre as classes /
    especificidade one-vs-rest média) — não há uma única "classe positiva"
    quando há mais de 2 categorias.

    `y_proba`: matriz (n_amostras, n_classes) de probabilidades — a ordem
    das colunas deve corresponder à ordem de `classes`.
    `classes`: rótulos possíveis, em ordem; se None, é inferida dos dados.
    """
    if classes is None:
        classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    n_classes = len(classes)
    binario = n_classes == 2

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if binario:
        tn, fp, fn, tp = cm.ravel()
        sensib = tp / (tp + fn + 1e-8)
        especif = tn / (tn + fp + 1e-8)
        pos_label = classes[1]
        precisao = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        if y_proba is not None:
            idx_pos = classes.index(pos_label)
            y_prob_pos = y_proba[:, idx_pos]
            auc = roc_auc_score(y_true, y_prob_pos) if len(set(y_true.tolist())) > 1 else 0.0
            ece = _calibration_error(y_true, y_prob_pos)
        else:
            auc, ece = 0.0, 0.0
    else:
        sensibs, especifs = [], []
        for i in range(n_classes):
            tp = cm[i, i]
            fn_i = cm[i, :].sum() - tp
            fp_i = cm[:, i].sum() - tp
            tn_i = cm.sum() - tp - fn_i - fp_i
            sensibs.append(tp / (tp + fn_i + 1e-8))
            especifs.append(tn_i / (tn_i + fp_i + 1e-8))
        sensib = float(np.mean(sensibs))
        especif = float(np.mean(especifs))
        precisao = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if y_proba is not None and y_proba.shape[1] == n_classes and len(set(y_true.tolist())) > 1:
            try:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                     average="macro", labels=classes)
            except ValueError:
                auc = 0.0
            ece = _calibration_error_multiclasse(y_true, y_proba, classes)
        else:
            auc, ece = 0.0, 0.0

    m = {
        "acuracia": round(float(accuracy_score(y_true, y_pred)), 4),
        "sensibilidade": round(float(sensib), 4),
        "especificidade": round(float(especif), 4),
        "precisao": round(float(precisao), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "auc": round(float(auc), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "ece": round(float(ece), 4),
        "n_classes": n_classes,
        "classes": [int(c) if isinstance(c, (int, np.integer)) else c for c in classes],
    }
    return m


def _mcnemar_test(y_true, pred_a, pred_b, nome_a: str, nome_b: str) -> dict:
    """Testa se os dois modelos diferem significativamente (McNemar)."""
    b = int(np.sum((pred_a == y_true) & (pred_b != y_true)))  # A acerta, B erra
    c = int(np.sum((pred_a != y_true) & (pred_b == y_true)))  # B acerta, A erra
    if b + c == 0:
        return {"p_valor": 1.0, "diferenca_significativa": False}
    # McNemar com correção de continuidade
    chi2 = (abs(b - c) - 1) ** 2 / (b + c + 1e-10)
    from scipy.stats import chi2 as chi2_dist
    p = float(1 - chi2_dist.cdf(chi2, df=1))
    return {
        "modelo_a": nome_a,
        "modelo_b": nome_b,
        "b_a_acerta_b_erra": b,
        "c_b_acerta_a_erra": c,
        "chi2": round(chi2, 4),
        "p_valor": round(p, 6),
        "diferenca_significativa": p < 0.05,
    }


def _clonar_modelo(nome: str):
    """Instância fresca (evita contaminação entre folds)."""
    from sklearn.base import clone
    return clone(_MODELOS[nome])


def _shap_importancia(modelo, X_train: np.ndarray, X_test: np.ndarray,
                      feature_names: list[str] | None, nome_modelo: str) -> dict:
    """
    Importância global de features via SHAP para o modelo vencedor.
    Usa TreeExplainer para modelos de árvore e KernelExplainer (amostrado) como
    fallback. Falha de forma segura devolvendo {'disponivel': False}.

    BINÁRIO (2 classes): usa a magnitude do SHAP da classe positiva — igual
    ao comportamento original.
    MULTI-CLASSE (3+): não existe uma única "classe positiva" — a
    importância de cada feature é a média da magnitude (|SHAP|) entre
    TODAS as classes, dando um ranking de "o que mais pesa na decisão, não
    importa qual classe", em vez de explicar arbitrariamente só uma delas.
    """
    resultado = {"disponivel": False, "modelo": nome_modelo}
    n_feat = X_test.shape[1]
    nomes = feature_names if (feature_names and len(feature_names) == n_feat) \
        else [f"f{i}" for i in range(n_feat)]
    n_classes = len(getattr(modelo, "classes_", [0, 1]))
    binario = n_classes == 2
    try:
        import shap

        if nome_modelo in ("RandomForest", "GradientBoosting"):
            explainer = shap.TreeExplainer(modelo)
            valores = explainer.shap_values(X_test)
        else:
            fundo = shap.sample(X_train, min(50, len(X_train)), random_state=42)
            if binario:
                explainer = shap.KernelExplainer(lambda d: modelo.predict_proba(d)[:, 1], fundo)
            else:
                explainer = shap.KernelExplainer(lambda d: modelo.predict_proba(d), fundo)
            valores = explainer.shap_values(X_test[:min(30, len(X_test))], nsamples=100)

        # Normaliza para uma matriz (n_amostras, n_features) de magnitude —
        # diferentes versões do shap devolvem list[array] (uma por classe),
        # um array 3D (n_amostras, n_features, n_classes), ou já 2D.
        if isinstance(valores, list):
            if binario:
                # Comportamento original, preservado bit-a-bit: só a classe positiva.
                valores_abs = np.abs(np.asarray(valores[1]))
            else:
                # Multi-classe: média da magnitude entre TODAS as classes.
                valores_abs = np.mean([np.abs(np.asarray(v)) for v in valores], axis=0)
        else:
            valores = np.asarray(valores)
            if valores.ndim == 3:
                if binario:
                    # shap>=0.45 devolve (n_amostras, n_features, n_classes) para
                    # binário; sem selecionar a classe, o ravel() desalinha os
                    # nomes das features — comportamento original preservado.
                    valores_abs = np.abs(valores[..., -1])
                else:
                    valores_abs = np.abs(valores).mean(axis=-1)
            else:
                valores_abs = np.abs(valores)

        importancia = valores_abs.mean(axis=0).ravel()
        if importancia.shape[0] != len(nomes):
            resultado["motivo"] = (f"SHAP devolveu {importancia.shape[0]} valores para "
                                   f"{len(nomes)} features — ranking descartado.")
            return resultado
        ranking = sorted(zip(nomes, importancia.tolist()), key=lambda kv: kv[1], reverse=True)
        resultado.update({
            "disponivel": True,
            "importancia_media_abs": {n: round(float(v), 5) for n, v in ranking},
            "top_features": [n for n, _ in ranking[:10]],
        })
        if not binario:
            resultado["nota"] = (
                f"Importância é a média entre as {n_classes} classes "
                f"(não existe uma única 'classe positiva' em multi-classe)."
            )
    except Exception as e:
        resultado["motivo"] = str(e)
    return resultado
