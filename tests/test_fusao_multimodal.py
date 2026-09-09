"""Testes para biostatusia.pipeline.fusao_multimodal — fusão tardia N-a-N (média ponderada + stacking)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.fusao_multimodal import treinar_fusao_tardia


def _dataset_complementar_2(n=300, seed=42):
    """Duas modalidades com sinal PARCIAL e complementar sobre o mesmo y —
    nenhuma delas sozinha é tão boa quanto as duas juntas."""
    rng = np.random.RandomState(seed)
    z1 = rng.randn(n)
    z2 = rng.randn(n)
    y = ((z1 + z2) > 0).astype(int)
    X_a = np.column_stack([z1, rng.randn(n) * 0.3, rng.randn(n) * 0.3])
    X_b = np.column_stack([z2, rng.randn(n) * 0.3, rng.randn(n) * 0.3])
    return [
        {"nome": "imagem", "X": X_a, "modelo": "RandomForest"},
        {"nome": "tabular", "X": X_b, "modelo": "LogisticRegression"},
    ], y


def _dataset_complementar_4(n=400, seed=1):
    """Quatro modalidades, cada uma vendo só 1/4 do sinal real."""
    rng = np.random.RandomState(seed)
    zs = [rng.randn(n) for _ in range(4)]
    y = (sum(zs) > 0).astype(int)
    nomes = ["imagem", "tabular", "sinal_temporal", "dicom"]
    modelos = ["RandomForest", "LogisticRegression", "GradientBoosting", "KNN"]
    modalidades = []
    for i in range(4):
        Xi = np.column_stack([zs[i], rng.randn(n) * 0.3, rng.randn(n) * 0.3])
        modalidades.append({"nome": nomes[i], "X": Xi, "modelo": modelos[i]})
    return modalidades, y


def test_fusao_2_modalidades_supera_melhor_isolada():
    modalidades, y = _dataset_complementar_2()
    res = treinar_fusao_tardia(modalidades, y)

    auc_isolado_max = max(res["metricas"]["imagem"]["auc"], res["metricas"]["tabular"]["auc"])
    auc_fusao_max = max(
        res["metricas"]["fusao_media_ponderada"]["auc"],
        res["metricas"]["fusao_stacking"]["auc"],
    )
    assert auc_fusao_max > auc_isolado_max


def test_fusao_4_modalidades_supera_qualquer_uma_isolada():
    modalidades, y = _dataset_complementar_4()
    res = treinar_fusao_tardia(modalidades, y)

    assert set(res["modalidades"]) == {"imagem", "tabular", "sinal_temporal", "dicom"}
    auc_isolado_max = max(res["metricas"][n]["auc"] for n in res["modalidades"])
    auc_fusao_max = max(
        res["metricas"]["fusao_media_ponderada"]["auc"],
        res["metricas"]["fusao_stacking"]["auc"],
    )
    assert auc_fusao_max > auc_isolado_max
    # pesos de 4 modalidades devem somar ~1 (tolerância maior aqui: cada peso
    # já vem arredondado a 4 casas decimais para exibição, então a soma de 4
    # valores arredondados independentemente pode desviar até ~2e-4 de 1.0 —
    # isso é esperado, não indica erro no cálculo dos pesos em si)
    assert abs(sum(res["pesos_confiabilidade"].values()) - 1.0) < 5e-4
    # coeficientes de stacking: um por modalidade + intercepto
    assert len(res["stacking_coeficientes"]) == 4 + 1


def test_fusao_nao_piora_quando_uma_modalidade_e_ruido_puro():
    rng = np.random.RandomState(1)
    n = 300
    z1 = rng.randn(n)
    y = (z1 > 0).astype(int)
    X_boa = np.column_stack([z1, rng.randn(n) * 0.2, rng.randn(n) * 0.2])
    X_ruido = rng.randn(n, 3)  # sem nenhuma relação com y

    res = treinar_fusao_tardia([
        {"nome": "boa", "X": X_boa, "modelo": "RandomForest"},
        {"nome": "ruido", "X": X_ruido, "modelo": "LogisticRegression"},
    ], y)

    auc_boa = res["metricas"]["boa"]["auc"]
    auc_fusao_media = res["metricas"]["fusao_media_ponderada"]["auc"]
    auc_fusao_stack = res["metricas"]["fusao_stacking"]["auc"]
    assert auc_fusao_media >= auc_boa - 0.05
    assert auc_fusao_stack >= auc_boa - 0.05
    assert res["pesos_confiabilidade"]["ruido"] < res["pesos_confiabilidade"]["boa"]
    assert res["melhor_modelo"] != "ruido"


def test_amostras_desalinhadas_levanta_erro_claro():
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError, match="alinhadas 1:1"):
        treinar_fusao_tardia([
            {"nome": "a", "X": rng.randn(100, 3), "modelo": "RandomForest"},
            {"nome": "b", "X": rng.randn(90, 3), "modelo": "LogisticRegression"},
        ], rng.randint(0, 2, 100))


def test_menos_de_2_modalidades_levanta_erro():
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError, match="pelo menos 2"):
        treinar_fusao_tardia([
            {"nome": "a", "X": rng.randn(50, 3), "modelo": "RandomForest"},
        ], rng.randint(0, 2, 50))


def test_nomes_duplicados_levanta_erro():
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError, match="duplicados"):
        treinar_fusao_tardia([
            {"nome": "a", "X": rng.randn(50, 3), "modelo": "RandomForest"},
            {"nome": "a", "X": rng.randn(50, 3), "modelo": "LogisticRegression"},
        ], rng.randint(0, 2, 50))


def test_amostras_insuficientes_retorna_aviso_sem_quebrar():
    rng = np.random.RandomState(0)
    res = treinar_fusao_tardia([
        {"nome": "a", "X": rng.randn(5, 3), "modelo": "RandomForest"},
        {"nome": "b", "X": rng.randn(5, 3), "modelo": "LogisticRegression"},
    ], np.array([0, 1, 0, 1, 0]))
    assert "aviso" in res


def test_pesos_de_confiabilidade_somam_um():
    modalidades, y = _dataset_complementar_2(seed=7)
    res = treinar_fusao_tardia(modalidades, y)
    soma = sum(res["pesos_confiabilidade"].values())
    # tolerância maior por causa do arredondamento a 4 casas decimais na exibição
    assert abs(soma - 1.0) < 5e-4


def test_protocolo_validacao_presente_e_sinaliza_sem_vazamento():
    modalidades, y = _dataset_complementar_2(seed=3)
    res = treinar_fusao_tardia(modalidades, y)
    assert "protocolo_validacao" in res
    assert "held-out" in res["protocolo_validacao"]["avaliacao_final"]


def test_resultado_expoe_veto_de_sensibilidade_reaproveitado():
    modalidades, y = _dataset_complementar_2(seed=9)
    res = treinar_fusao_tardia(modalidades, y)
    for chave in ("melhor_modelo", "piso_sensibilidade_atingido", "criterio_selecao"):
        assert chave in res
