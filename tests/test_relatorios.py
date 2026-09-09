"""Tier 1 — laudo populacional determinístico e formatação de métricas."""
from biostatusia.pipeline.relatorios import (
    construir_podio, _fmt_metrica, construir_laudo_populacional,
    correlacoes_biomarcadores,
)


def test_fmt_metrica_ausente_e_numero():
    assert _fmt_metrica(None) == "—"
    assert _fmt_metrica(0.12345) == "0.123"
    assert _fmt_metrica("X") == "X"


def test_podio_ordenado_por_auc(metricas_completas):
    podio = construir_podio(metricas_completas, "RandomForest")
    assert podio[0]["modelo"] == "RandomForest"
    assert podio[0]["campeao"] is True
    aucs = [p["auc"] for p in podio]
    assert aucs == sorted(aucs, reverse=True)


def test_laudo_populacional_deterministico(metricas_completas):
    pd = {
        "familia": "F3", "modo": "dataset_rotulado", "n_imagens": 50,
        "metricas": metricas_completas, "melhor_modelo": "RandomForest",
    }
    a = construir_laudo_populacional(dict(pd))
    b = construir_laudo_populacional(dict(pd))
    assert a == b                              # determinismo
    assert "Aviso ético" in a                  # aviso ético obrigatório
    assert "RandomForest" in a


def test_correlacoes_respeitam_limiar():
    # b = 2*a (correlação perfeita); c é ruído
    vetores = [[1, 2, 5], [2, 4, 1], [3, 6, 9], [4, 8, 2], [5, 10, 7]]
    nomes = ["a", "b", "c"]
    pares = correlacoes_biomarcadores(vetores, nomes, limiar=0.7)
    assert any({p["a"], p["b"]} == {"a", "b"} for p in pares)
    assert all(abs(p["correlacao"]) >= 0.7 for p in pares)
