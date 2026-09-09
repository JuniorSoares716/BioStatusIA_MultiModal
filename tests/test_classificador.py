"""Tier 1 — treinador padrão (treinar_vetores) e guardas do AutoML enriquecido."""
import numpy as np

from biostatusia.pipeline.classificador import treinar_vetores
from biostatusia.pipeline.avaliacao_modelos import avaliar_modelos, balancear

METRICAS_ESPERADAS = (
    "acuracia", "sensibilidade", "especificidade", "precisao", "recall",
    "f1", "auc", "mcc", "kappa", "ece", "latencia_inferencia_ms", "tempo_treino_s",
)


def test_treinar_vetores_metricas_completas(dados_binarios):
    X, y = dados_binarios
    res = treinar_vetores(X, y, scaling="standard", familia="TESTE")
    m = res["metricas"][res["melhor_modelo"]]
    faltando = [k for k in METRICAS_ESPERADAS if k not in m]
    assert not faltando, f"métricas ausentes: {faltando}"


def test_campeao_tem_maior_auc(dados_binarios):
    X, y = dados_binarios
    res = treinar_vetores(X, y)
    aucs = {n: mm["auc"] for n, mm in res["metricas"].items()}
    assert aucs[res["melhor_modelo"]] == max(aucs.values())


def test_todos_os_6_modelos(dados_binarios):
    X, y = dados_binarios
    res = treinar_vetores(X, y)
    assert len(res["metricas"]) == 6


def test_avaliar_modelos_guarda_amostras_insuficientes():
    X = np.random.RandomState(0).rand(6, 4)
    y = np.array([0, 1, 0, 1, 0, 1])
    res = avaliar_modelos(X, y, persistir_vencedor=False)
    assert "aviso" in res  # <10 amostras → não treina


def test_balancear_seguro_e_equilibra():
    rng = np.random.RandomState(0)
    X = rng.rand(40, 4)
    y = np.array([0] * 30 + [1] * 10)
    Xb, yb, info = balancear(X, y, "smote")
    assert Xb.shape[1] == X.shape[1]                     # não altera nº de features
    if info.get("aplicado"):                             # se imblearn disponível
        _, contagens = np.unique(yb, return_counts=True)
        assert contagens.min() == contagens.max()        # classes equilibradas
