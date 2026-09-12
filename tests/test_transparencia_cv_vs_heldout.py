"""Testes para a transparência entre CV interna (decide o vencedor e o veto
de sensibilidade) e held-out (métrica final reportada) — antes, o ranking
exibido misturava os dois sem indicar qual era qual, ordenando por um
critério mas destacando o vencedor por outro."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.classificador import treinar_vetores
from biostatusia.pipeline.avaliacao_modelos import avaliar_modelos


def test_treinar_vetores_anexa_sensibilidade_e_score_cv_a_cada_modelo():
    rng = np.random.RandomState(1)
    X = rng.randn(200, 8)
    y = (X[:, 0] + X[:, 3] > 0).astype(int)
    res = treinar_vetores(X, y, familia="tabular", selecao_features=False)

    for nome, m in res["metricas"].items():
        assert "sensibilidade_cv" in m
        assert "score_clinico_cv" in m
        # os valores CV devem bater com os de metricas_cv daquele modelo
        assert m["sensibilidade_cv"] == res["metricas_cv"][nome]["sensibilidade"]["media"]
        assert m["score_clinico_cv"] == res["metricas_cv"][nome]["score_clinico"]


def test_vencedor_e_sempre_o_de_maior_score_clinico_cv_nao_held_out():
    """O vencedor é decidido pelo score CV — não precisa ser o de maior
    score_clinico no held-out (isso é o comportamento correto e esperado:
    o held-out é só reportado, nunca usado para a seleção)."""
    rng = np.random.RandomState(1)
    X = rng.randn(200, 8)
    y = (X[:, 0] + X[:, 3] > 0).astype(int)
    res = treinar_vetores(X, y, familia="tabular", selecao_features=False)
    melhor = res["melhor_modelo"]

    maior_score_cv = max(res["metricas"].values(), key=lambda m: m["score_clinico_cv"])
    assert res["metricas"][melhor]["score_clinico_cv"] == maior_score_cv["score_clinico_cv"]


def test_avaliar_modelos_tambem_anexa_metricas_cv(tmp_path, monkeypatch):
    monkeypatch.setattr("biostatusia.pipeline.inferencia.MODEL_DIR", tmp_path, raising=False)
    rng = np.random.RandomState(2)
    X = rng.randn(150, 6)
    y = (X[:, 0] > 0).astype(int)
    res = avaliar_modelos(X, y, familia="teste", persistir_vencedor=False)

    for nome, m in res["metricas"].items():
        assert "sensibilidade_cv" in m
        assert "score_clinico_cv" in m
