"""
Fusão tardia (decision-level) entre N modalidades quaisquer (BioStatusIA).

Generalização do módulo original (que aceitava só 2 modalidades: imagem +
tabular) para qualquer combinação de 2 a 5 modalidades (imagem, tabular,
sinal temporal, DICOM, volume 3D) — a pedido, para que a fusão não fique
restrita ao caso mais comum. Também generalizado para classificação
MULTI-CLASSE (não só binária) — as probabilidades de cada modalidade agora
são matrizes (n_amostras, n_classes), e a fusão (média ponderada e
stacking) opera sobre essas matrizes inteiras.

Fusão tardia (não fusão precoce nem intermediária) pelos mesmos motivos de
antes: datasets pequenos (fusão precoce aumentaria a dimensionalidade sem
aumentar amostras), cada modalidade já tem seu próprio pipeline AutoML com
CV interna e calibração (ECE), e fusão tardia tolera modalidade ausente por
amostra.

Duas variantes, ambas com evidência comparada no mesmo fold de teste
held-out (nunca usado para decidir pesos nem para treinar o meta-modelo):

  - Média ponderada pela confiabilidade (AUC − ECE) de cada modalidade,
    medida via CV interna no treino — generalizada para N pesos que somam 1.
  - Stacking: um meta-modelo (Logistic Regression) treinado nas
    probabilidades OUT-OF-FOLD de cada uma das N modalidades.

PRÉ-REQUISITO — correspondência entre amostras: `X_i[k]` de cada modalidade
e `y[k]` precisam ser da MESMA amostra/paciente/exame. Ver
`alinhamento_multimodal.py`, que resolve esse alinhamento para N modalidades
antes de chamar este módulo.
"""
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from biostatusia.pipeline.avaliacao_modelos import (
    _MODELOS, _calibration_error, _calibration_error_multiclasse,
    calcular_metricas_classificacao, calcular_score_clinico, selecionar_melhor_modelo,
)


def _fit_predict_proba_oof(X: np.ndarray, y: np.ndarray, modelo_nome: str,
                            cv: StratifiedKFold, n_classes: int) -> np.ndarray:
    """
    Gera probabilidades OUT-OF-FOLD (estilo `cross_val_predict`), como uma
    matriz (n_amostras, n_classes): em cada fold, treina um modelo do zero
    SÓ com a partição de treino do fold, e prediz a partição de validação —
    nunca um modelo vê a amostra que está prevendo. Usado tanto para medir
    a confiabilidade de cada modalidade quanto para treinar o meta-modelo
    do stacking, sem vazamento.
    """
    proba_oof = np.zeros((len(y), n_classes), dtype=float)
    for fold_train, fold_val in cv.split(X, y):
        modelo = clone(_MODELOS[modelo_nome])
        modelo.fit(X[fold_train], y[fold_train])
        proba_oof[fold_val] = modelo.predict_proba(X[fold_val])
    return proba_oof


def _pesos_confiabilidade(aucs: list[float], eces: list[float]) -> list[float]:
    """
    Peso maior para a modalidade com melhor discriminação (AUC mais alto) E
    melhor calibração (ECE mais baixo). Generalizado para N modalidades —
    usa max(1e-6, AUC-ECE) por modalidade, normalizado para somar 1, nunca
    zerando/invertendo o peso de uma modalidade só porque é mais fraca.
    """
    scores = [max(1e-6, auc - ece) for auc, ece in zip(aucs, eces)]
    total = sum(scores)
    return [s / total for s in scores]


def treinar_fusao_tardia(modalidades: list[dict], y: np.ndarray,
                          n_splits: int = 5, random_state: int = 42,
                          imputar: bool = True) -> dict:
    """
    Fusão tardia (média ponderada + stacking) entre 2 a 5 modalidades já
    alinhadas por amostra (ver `alinhamento_multimodal.py`). Funciona para
    classificação BINÁRIA e MULTI-CLASSE.

    `modalidades`: lista de dicts, um por modalidade participante:
        {"nome": "imagem", "X": np.ndarray, "modelo": "RandomForest"}
    onde "modelo" é o nome de um dos 6 classificadores do BioStatusIA (ex.:
    já escolhido pelo AutoML de cada modalidade isoladamente).

    `imputar`: se True (padrão), valores ausentes (NaN) em qualquer
    modalidade são imputados com a mediana — ajustada SOMENTE no fold de
    treino, nunca vendo o held-out.
    """
    if len(modalidades) < 2:
        raise ValueError(
            f"Fusão exige pelo menos 2 modalidades; recebido {len(modalidades)}."
        )

    n = len(y)
    for m in modalidades:
        if len(m["X"]) != n:
            raise ValueError(
                f"Fusão exige amostras alinhadas 1:1 entre modalidades: "
                f"modalidade '{m['nome']}' tem {len(m['X'])} amostras, y tem {n}. "
                f"Alinhe todas as modalidades pela mesma amostra/paciente antes de chamar esta função."
            )
    nomes = [m["nome"] for m in modalidades]
    if len(set(nomes)) != len(nomes):
        raise ValueError(f"Nomes de modalidade duplicados: {nomes}")

    if n < 10 or len(set(y.tolist())) < 2:
        return {"aviso": f"Fusão não executada: {n} amostras, {len(set(y.tolist()))} classes."}

    classes = sorted(set(y.tolist()))
    n_classes = len(classes)
    binario = n_classes == 2

    idx = np.arange(n)
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y, test_size=0.2, random_state=random_state, stratify=y
    )

    n_min = int(np.min(np.bincount(y_train)))
    n_splits_efetivo = max(2, min(n_splits, n_min))
    cv = StratifiedKFold(n_splits=n_splits_efetivo, shuffle=True, random_state=random_state)

    if imputar:
        from biostatusia.pipeline.dados_tabulares import imputar_treino_teste

    p_oof: dict[str, np.ndarray] = {}
    p_test: dict[str, np.ndarray] = {}
    aucs_oof: dict[str, float] = {}
    eces_oof: dict[str, float] = {}

    def _auc_generico(y_ref, proba):
        if len(set(y_ref.tolist())) < 2:
            return 0.0
        if binario:
            return float(roc_auc_score(y_ref, proba[:, 1]))
        try:
            return float(roc_auc_score(y_ref, proba, multi_class="ovr", average="macro", labels=classes))
        except ValueError:
            return 0.0

    def _ece_generico(y_ref, proba):
        if binario:
            return float(_calibration_error(y_ref, proba[:, 1]))
        return float(_calibration_error_multiclasse(y_ref, proba, classes))

    for m in modalidades:
        nome, X, modelo_nome = m["nome"], m["X"], m["modelo"]
        X_train, X_test = X[idx_train], X[idx_test]

        if imputar:
            X_train, X_test, _ = imputar_treino_teste(X_train, X_test, metodo="auto")

        p_oof[nome] = _fit_predict_proba_oof(X_train, y_train, modelo_nome, cv, n_classes)
        aucs_oof[nome] = _auc_generico(y_train, p_oof[nome])
        eces_oof[nome] = _ece_generico(y_train, p_oof[nome])

        modelo_final = clone(_MODELOS[modelo_nome]).fit(X_train, y_train)
        p_test[nome] = modelo_final.predict_proba(X_test)

    pesos_lista = _pesos_confiabilidade([aucs_oof[n] for n in nomes], [eces_oof[n] for n in nomes])
    pesos = dict(zip(nomes, pesos_lista))

    # Média ponderada elementwise das matrizes de probabilidade (n_amostras, n_classes)
    p_fusao_media = sum(pesos[n] * p_test[n] for n in nomes)

    # Stacking: meta-modelo treinado SÓ nas probabilidades OOF do treino —
    # o held-out só entra depois, na hora de avaliar. Entrada: probabilidades
    # de todas as modalidades concatenadas horizontalmente.
    X_meta_treino = np.column_stack([p_oof[n] for n in nomes])
    X_meta_teste = np.column_stack([p_test[n] for n in nomes])
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_meta_treino, y_train)
    p_fusao_stack = meta.predict_proba(X_meta_teste)
    # Garante que as colunas do stacking estão na mesma ordem de `classes`
    # (LogisticRegression ordena por `meta.classes_`, que deveria bater com
    # `classes`, mas reordena explicitamente por segurança).
    if list(meta.classes_) != classes:
        ordem = [list(meta.classes_).index(c) for c in classes]
        p_fusao_stack = p_fusao_stack[:, ordem]

    def _pred(proba):
        return np.asarray(classes)[np.argmax(proba, axis=1)]

    metricas = {
        nome: calcular_metricas_classificacao(y_test, _pred(p_test[nome]), p_test[nome], classes=classes)
        for nome in nomes
    }
    metricas["fusao_media_ponderada"] = calcular_metricas_classificacao(
        y_test, _pred(p_fusao_media), p_fusao_media, classes=classes
    )
    metricas["fusao_stacking"] = calcular_metricas_classificacao(
        y_test, _pred(p_fusao_stack), p_fusao_stack, classes=classes
    )
    for nome_m in metricas:
        metricas[nome_m]["score_clinico"] = round(float(calcular_score_clinico(metricas[nome_m])), 4)

    # Reaproveita o MESMO critério de seleção (com veto de piso de
    # sensibilidade) usado no AutoML de cada modalidade isolada — agora
    # decidindo entre as N modalidades sozinhas e as 2 formas de fusão.
    selecao = selecionar_melhor_modelo(metricas)

    if binario:
        # (n_modalidades,) um coeficiente por modalidade — igual a antes.
        stacking_coefs = {
            **{n: round(float(c), 4) for n, c in zip(nomes, meta.coef_[0])},
            "intercepto": round(float(meta.intercept_[0]), 4),
        }
    else:
        # Multi-classe: meta.coef_ tem forma (n_classes, n_modalidades*n_classes).
        # Reporta só um resumo (norma dos coeficientes por modalidade, por
        # classe) em vez de expor a matriz inteira — mais legível.
        stacking_coefs = {"formato": "multi-classe", "n_classes": n_classes, "por_modalidade": {}}
        for i, nome_mod in enumerate(nomes):
            cols = slice(i * n_classes, (i + 1) * n_classes)
            stacking_coefs["por_modalidade"][nome_mod] = round(
                float(np.linalg.norm(meta.coef_[:, cols])), 4
            )

    return {
        "modalidades": nomes,
        "n_classes": n_classes,
        "classes": classes,
        "metricas": metricas,
        "pesos_confiabilidade": {n: round(float(pesos[n]), 4) for n in nomes},
        "metricas_oof_treino": {
            n: {"auc": round(float(aucs_oof[n]), 4), "ece": round(float(eces_oof[n]), 4)} for n in nomes
        },
        "stacking_coeficientes": stacking_coefs,
        "protocolo_validacao": {
            "pesos_e_stacking": "estimados via CV interna (out-of-fold) dentro do treino (80%)",
            "avaliacao_final": "held-out 20%, nunca usado para calcular pesos nem treinar o stacking",
        },
        **selecao,
    }
