"""Testes para pipeline/inferencia.py::prever_exemplar — inferência de
amostra única (Seção A: Laudo de Amostra), incluindo o suporte a
multi-classe que faltava (antes fixava a coluna 1 de probabilidade)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.inferencia import salvar_modelo_vencedor, prever_exemplar
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def test_prever_exemplar_binario_preserva_categoria_e_probabilidade(monkeypatch, tmp_path):
    monkeypatch.setattr("biostatusia.pipeline.inferencia.MODEL_DIR", tmp_path)
    rng = np.random.RandomState(0)
    X = rng.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    scaler = StandardScaler().fit(X)
    modelo = RandomForestClassifier(random_state=0).fit(scaler.transform(X), y)
    salvar_modelo_vencedor("RandomForest", modelo, scaler, "TESTE_BIN",
                           feature_names=[f"f{i}" for i in range(5)], metricas={"auc": 0.9})

    vetor = np.array([2.0, 0, 0, 0, 0])  # deve favorecer claramente a classe 1
    res = prever_exemplar(vetor, "TESTE_BIN")
    assert res["disponivel"]
    assert res["usou_fallback"] is False
    assert res["n_classes"] == 2
    assert res["categoria"] in ("MALIGNO/POSITIVO", "BENIGNO/NEGATIVO")
    assert "probabilidade_positiva" in res  # campo binário preservado
    json.dumps(res)  # não deve levantar erro de serialização


def test_prever_exemplar_multiclasse_nao_fixa_arbitrariamente_a_classe_1(monkeypatch, tmp_path):
    monkeypatch.setattr("biostatusia.pipeline.inferencia.MODEL_DIR", tmp_path)
    rng = np.random.RandomState(1)
    X = rng.randn(200, 6)
    y = np.argmax(np.column_stack([X[:, 0], X[:, 2], X[:, 4]]), axis=1)  # 3 classes
    scaler = StandardScaler().fit(X)
    modelo = RandomForestClassifier(random_state=0).fit(scaler.transform(X), y)
    salvar_modelo_vencedor("RandomForest", modelo, scaler, "TESTE_MULTI",
                           feature_names=[f"f{i}" for i in range(6)],
                           metricas={"n_classes": 3, "classes": [0, 1, 2]})

    # vetor desenhado para favorecer claramente a classe 2 (índice da maior feature é f4)
    vetor = np.array([-2.0, 0, -2.0, 0, 3.0, 0])
    res = prever_exemplar(vetor, "TESTE_MULTI")
    assert res["disponivel"]
    assert res["n_classes"] == 3
    assert res["classe_prevista"] == 2
    assert len(res["distribuicao_probabilidade"]) == 3
    assert "probabilidade_positiva" not in res  # não faz sentido em multi-classe
    json.dumps(res)  # todos os campos devem ser serializáveis (sem np.int64/np.float64)


def test_prever_exemplar_classes_nao_json_nativas_sao_convertidas(monkeypatch, tmp_path):
    """Regressão: sklearn expõe .classes_ como np.int64 — precisa virar int nativo."""
    monkeypatch.setattr("biostatusia.pipeline.inferencia.MODEL_DIR", tmp_path)
    rng = np.random.RandomState(2)
    X = rng.randn(60, 4)
    y = (X[:, 0] > 0).astype(np.int64)
    scaler = StandardScaler().fit(X)
    modelo = RandomForestClassifier(random_state=0).fit(scaler.transform(X), y)
    salvar_modelo_vencedor("RandomForest", modelo, scaler, "TESTE_TIPOS",
                           feature_names=[f"f{i}" for i in range(4)], metricas={})

    res = prever_exemplar(np.array([1.0, 0, 0, 0]), "TESTE_TIPOS")
    assert isinstance(res["classe_prevista"], int)  # não np.int64
    json.dumps(res)  # deve serializar sem TypeError
