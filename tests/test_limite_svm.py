"""Testes para o limite de tamanho do SVM — datasets grandes pulam o SVM
(custo super-quadrático no número de amostras, medido em ~10s por treino
com N=15.000, e o protocolo de CV interna repete isso ~16 vezes) em vez de
travar por muitos minutos sem nenhum log."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.classificador import treinar_vetores
from biostatusia.pipeline.avaliacao_modelos import avaliar_modelos, _MODELOS


def test_treinar_vetores_pula_svm_em_dataset_grande():
    rng = np.random.RandomState(3)
    n = 13000  # 80% do treino (~10.400) > LIMITE_AMOSTRAS_SVM (10_000)
    X = rng.randn(n, 8)
    y = (X[:, 0] + X[:, 3] > 0).astype(int)

    res = treinar_vetores(X, y, familia="tabular", selecao_features=False)
    assert "SVM" in res["modelos_pulados"]
    assert "SVM" not in res["metricas"]
    # os outros 5 continuam treinando normalmente
    assert set(res["metricas"].keys()) == {"LogisticRegression", "KNN", "RandomForest", "GradientBoosting", "MLP"}
    assert res["melhor_modelo"] in res["metricas"]


def test_treinar_vetores_mantem_svm_em_dataset_pequeno():
    rng = np.random.RandomState(3)
    n = 300
    X = rng.randn(n, 8)
    y = (X[:, 0] + X[:, 3] > 0).astype(int)

    res = treinar_vetores(X, y, familia="tabular", selecao_features=False)
    assert res["modelos_pulados"] == {}
    assert "SVM" in res["metricas"]


def test_avaliar_modelos_pula_svm_em_dataset_grande(tmp_path, monkeypatch):
    monkeypatch.setattr("biostatusia.pipeline.inferencia.MODEL_DIR", tmp_path, raising=False)
    rng = np.random.RandomState(4)
    n = 13000
    X = rng.randn(n, 8)
    y = (X[:, 0] + X[:, 3] > 0).astype(int)

    res = avaliar_modelos(X, y, familia="teste", persistir_vencedor=False)
    assert "SVM" in res["modelos_pulados"]
    assert "SVM" not in res["metricas"]


def test_avaliar_modelos_nao_modifica_o_dict_global_de_modelos():
    """_MODELOS é compartilhado com fusao_multimodal.py — pular o SVM para um
    dataset grande não pode remover a entrada do dict global."""
    rng = np.random.RandomState(4)
    n = 13000
    X = rng.randn(n, 8)
    y = (X[:, 0] + X[:, 3] > 0).astype(int)
    avaliar_modelos(X, y, familia="teste", persistir_vencedor=False)
    assert "SVM" in _MODELOS  # dict global intacto
