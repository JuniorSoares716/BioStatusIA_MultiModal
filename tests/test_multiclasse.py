"""Testes para o suporte a classificação MULTI-CLASSE (3+ classes), mantendo
o comportamento binário exatamente igual ao que já existia."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.avaliacao_modelos import (
    calcular_metricas_classificacao, _calibration_error, avaliar_modelos,
)
from biostatusia.pipeline.classificador import treinar_vetores
from biostatusia.pipeline.fusao_multimodal import treinar_fusao_tardia


# ── calcular_metricas_classificacao ─────────────────────────────────────────

def test_binario_bate_exatamente_com_calculo_manual_antigo():
    rng = np.random.RandomState(0)
    n = 200
    y_true = rng.randint(0, 2, n)
    y_proba = rng.dirichlet([1, 1], n)
    y_pred = np.argmax(y_proba, axis=1)

    m = calcular_metricas_classificacao(y_true, y_pred, y_proba, classes=[0, 1])

    from sklearn.metrics import confusion_matrix, roc_auc_score
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensib_esperado = tp / (tp + fn + 1e-8)
    especif_esperado = tn / (tn + fp + 1e-8)
    auc_esperado = roc_auc_score(y_true, y_proba[:, 1])
    ece_esperado = _calibration_error(y_true, y_proba[:, 1])

    assert abs(m["sensibilidade"] - sensib_esperado) < 1e-4
    assert abs(m["especificidade"] - especif_esperado) < 1e-4
    assert abs(m["auc"] - auc_esperado) < 1e-4
    assert abs(m["ece"] - ece_esperado) < 1e-4
    assert m["n_classes"] == 2


def test_multiclasse_nao_quebra_e_produz_metricas_validas():
    rng = np.random.RandomState(1)
    n = 200
    y_true = rng.randint(0, 4, n)
    y_proba = rng.dirichlet([1, 1, 1, 1], n)
    y_pred = np.argmax(y_proba, axis=1)

    m = calcular_metricas_classificacao(y_true, y_pred, y_proba, classes=[0, 1, 2, 3])
    assert m["n_classes"] == 4
    assert 0.0 <= m["auc"] <= 1.0
    assert 0.0 <= m["ece"] <= 1.0
    assert 0.0 <= m["sensibilidade"] <= 1.0
    assert 0.0 <= m["especificidade"] <= 1.0


# ── treinar_vetores (pipeline tabular) ──────────────────────────────────────

def test_treinar_vetores_multiclasse_aprende_sinal_real():
    rng = np.random.RandomState(7)
    n = 300
    X = rng.randn(n, 15)
    scores = np.column_stack([
        X[:, 0] + X[:, 1], X[:, 2] + X[:, 3], X[:, 4] + X[:, 5], X[:, 6] + X[:, 7],
    ])
    y = np.argmax(scores, axis=1)

    res = treinar_vetores(X, y, familia="tabular")
    assert res["n_classes"] == 4
    melhor = res["melhor_modelo"]
    m = res["metricas"][melhor]
    assert m["acuracia"] > 0.4  # bem acima do acaso (25%)
    cm = np.array(res["confusion_matrix"][melhor])
    assert cm.shape == (4, 4)
    assert res["roc_data"] == {}  # ROC não se aplica a multi-classe


def test_treinar_vetores_binario_continua_funcionando(monkeypatch, tmp_path):
    monkeypatch.setattr("biostatusia.pipeline.classificador.MODEL_DIR", tmp_path)
    rng = np.random.RandomState(3)
    n = 150
    X = rng.randn(n, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    res = treinar_vetores(X, y, familia="tabular")
    assert res["n_classes"] == 2
    melhor = res["melhor_modelo"]
    assert res["roc_data"].get(melhor) is not None  # ROC continua presente no binário


# ── avaliar_modelos (pipeline de imagem/sinal) ──────────────────────────────

def test_avaliar_modelos_multiclasse(monkeypatch, tmp_path):
    monkeypatch.setattr("biostatusia.pipeline.inferencia.MODEL_DIR", tmp_path, raising=False)
    rng = np.random.RandomState(5)
    n = 200
    X = rng.randn(n, 12)
    scores = np.column_stack([X[:, 0], X[:, 3], X[:, 6]])
    y = np.argmax(scores, axis=1)

    res = avaliar_modelos(X, y, familia="teste", persistir_vencedor=False)
    assert res.get("n_classes") == 3
    assert res.get("melhor_modelo") is not None
    m = res["metricas"][res["melhor_modelo"]]
    assert m["acuracia"] > 0.33  # acima do acaso (33%)


# ── treinar_fusao_tardia (fusão multimodal) ─────────────────────────────────

def test_fusao_multiclasse_supera_modalidades_isoladas():
    rng = np.random.RandomState(11)
    n = 400
    z = rng.randn(n, 4)
    y = np.argmax(z, axis=1)
    X_a = np.column_stack([z[:, 0], z[:, 1], rng.randn(n) * 0.5, rng.randn(n) * 0.5])
    X_b = np.column_stack([rng.randn(n) * 0.5, rng.randn(n) * 0.5, z[:, 2], z[:, 3]])

    res = treinar_fusao_tardia([
        {"nome": "imagem", "X": X_a, "modelo": "RandomForest"},
        {"nome": "tabular", "X": X_b, "modelo": "LogisticRegression"},
    ], y)

    assert res["n_classes"] == 4
    acc_isolado = max(res["metricas"]["imagem"]["acuracia"], res["metricas"]["tabular"]["acuracia"])
    acc_fusao = max(res["metricas"]["fusao_media_ponderada"]["acuracia"], res["metricas"]["fusao_stacking"]["acuracia"])
    assert acc_fusao > acc_isolado


def test_fusao_binaria_continua_funcionando():
    rng = np.random.RandomState(42)
    n = 300
    z1 = rng.randn(n)
    z2 = rng.randn(n)
    y = ((z1 + z2) > 0).astype(int)
    X_a = np.column_stack([z1, rng.randn(n) * 0.3])
    X_b = np.column_stack([z2, rng.randn(n) * 0.3])

    res = treinar_fusao_tardia([
        {"nome": "imagem", "X": X_a, "modelo": "RandomForest"},
        {"nome": "tabular", "X": X_b, "modelo": "LogisticRegression"},
    ], y)
    assert res["n_classes"] == 2
    assert "intercepto" in res["stacking_coeficientes"]  # formato binário preservado


# ── SHAP (interpretabilidade + seleção de features) ─────────────────────────

def test_shap_importancia_binario_identico_ao_calculo_manual():
    from biostatusia.pipeline.avaliacao_modelos import _shap_importancia
    from sklearn.ensemble import RandomForestClassifier
    import shap

    rng = np.random.RandomState(3)
    n, n_feat = 150, 15
    X = rng.randn(n, n_feat)
    y = (X[:, 1] + X[:, 7] > 0).astype(int)
    modelo = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
    nomes = [f"f{i}" for i in range(n_feat)]

    res = _shap_importancia(modelo, X, X[:50], nomes, "RandomForest")

    explainer = shap.TreeExplainer(modelo)
    valores = explainer.shap_values(X[:50])
    if isinstance(valores, list):
        valores = valores[1]
    valores = np.asarray(valores)
    if valores.ndim == 3:
        valores = valores[..., -1]
    importancia_esperada = np.abs(valores).mean(axis=0).ravel()
    ranking_esperado = sorted(zip(nomes, importancia_esperada.tolist()), key=lambda kv: kv[1], reverse=True)
    esperado = {n: round(float(v), 5) for n, v in ranking_esperado}

    assert res["importancia_media_abs"] == esperado


def test_shap_importancia_multiclasse_captura_features_de_todas_as_classes():
    from biostatusia.pipeline.avaliacao_modelos import _shap_importancia
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(9)
    n, n_feat = 300, 30
    X = rng.randn(n, n_feat)
    scores = np.column_stack([X[:, 3] + X[:, 4], X[:, 15] + X[:, 16], X[:, 25] + X[:, 26], rng.randn(n)])
    y = np.argmax(scores, axis=1)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

    res = _shap_importancia(modelo, X, X[:60], [f"f{i}" for i in range(n_feat)], "RandomForest")
    assert res["disponivel"]
    assert "nota" in res  # sinaliza que é média multi-classe

    informativas = {"f3", "f4", "f15", "f16", "f25", "f26"}
    top8 = set(res["top_features"][:8])
    assert len(top8 & informativas) >= 4  # captura a maioria, espalhadas entre as 4 classes


def test_selecionar_features_shap_multiclasse_nao_ignora_classes_minoritarias():
    from biostatusia.pipeline.dados_tabulares import selecionar_features_shap

    rng = np.random.RandomState(9)
    n, n_feat = 300, 30
    X = rng.randn(n, n_feat)
    scores = np.column_stack([X[:, 3] + X[:, 4], X[:, 15] + X[:, 16], X[:, 25] + X[:, 26], rng.randn(n)])
    y = np.argmax(scores, axis=1)

    idx, info = selecionar_features_shap(X, y, feature_names=[f"f{i}" for i in range(n_feat)])
    assert info["aplicado"]
    informativas = {3, 4, 15, 16, 25, 26}
    achadas = informativas & set(idx.tolist())
    assert len(achadas) >= 4  # a seleção não deve ignorar as classes "não vistas" por uma única classe arbitrária
