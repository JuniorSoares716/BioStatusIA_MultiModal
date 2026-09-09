"""Tier 1 — inferência individual (achatar, persistência e previsão defensiva)."""
import numpy as np

from biostatusia.pipeline.inferencia import (
    achatar_biomarcadores, salvar_modelo_vencedor, carregar_modelo_vencedor,
    prever_exemplar,
)


def test_achatar_biomarcadores_ignora_bool_e_achata_dict():
    bio = {"grupo": {"a": 1.0, "b": 2.0, "flag": True}, "c": 3.0}
    v = achatar_biomarcadores(bio)
    assert v.dtype == np.float32
    assert set(v.tolist()) == {1.0, 2.0, 3.0}   # booleano ignorado


def test_roundtrip_e_previsao_com_ajuste_dimensional(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import biostatusia.pipeline.inferencia as inf

    monkeypatch.setattr(inf, "MODEL_DIR", tmp_path)

    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(0, 1, (10, 4)), rng.normal(3, 1, (10, 4))])
    y = np.array([0] * 10 + [1] * 10)
    scaler = StandardScaler().fit(X)
    modelo = LogisticRegression().fit(scaler.transform(X), y)

    salvar_modelo_vencedor("LogReg", modelo, scaler, "TESTE",
                           ["f0", "f1", "f2", "f3"], {"auc": 0.9})
    art = carregar_modelo_vencedor("TESTE")
    assert art is not None and art["nome"] == "LogReg"

    # vetor com dimensão MENOR que o esperado → ajuste defensivo (preenche com zero)
    r = prever_exemplar(np.array([0.1, 0.2]), "TESTE")
    assert r["disponivel"] is True
    assert r["categoria"] in ("MALIGNO/POSITIVO", "BENIGNO/NEGATIVO")
    assert 0.0 <= r["probabilidade_positiva"] <= 1.0


def test_prever_sem_modelo_retorna_aviso(tmp_path, monkeypatch):
    import biostatusia.pipeline.inferencia as inf
    monkeypatch.setattr(inf, "MODEL_DIR", tmp_path)
    r = prever_exemplar(np.array([0.1, 0.2, 0.3]), "INEXISTENTE", permitir_fallback=False)
    assert r["disponivel"] is False
