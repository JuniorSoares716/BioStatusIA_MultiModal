"""Testes para biostatusia.pipeline.alinhamento_multimodal — alinhamento N-a-N."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.alinhamento_multimodal import (
    alinhar_modalidades, detectar_coluna_id, indice_por_arquivo, indice_tabular,
)


def test_indice_por_arquivo_normaliza_nome():
    registros = [{"caminho": "/x/Paciente_007.PNG"}, {"caminho": "/y/paciente_008.png"}]
    idx = indice_por_arquivo(registros)
    assert idx == {"paciente_007": 0, "paciente_008": 1}


def test_detectar_coluna_id_reconhece_variacoes():
    assert detectar_coluna_id(["image_id", "idade"]) == 0
    assert detectar_coluna_id(["idade", "patient_id"]) == 1
    assert detectar_coluna_id(["idade", "diagnostico"]) is None


def test_indice_tabular_sem_coluna_id_retorna_vazio():
    idx, coluna = indice_tabular(["idade", "diagnostico"], [["30", "sim"]])
    assert idx == {} and coluna is None


def test_indice_tabular_com_coluna_id():
    header = ["patient_id", "idade"]
    data = [["p001", "30"], ["p002", "40"]]
    idx, coluna = indice_tabular(header, data)
    assert coluna == "patient_id"
    assert idx == {"p001": 0, "p002": 1}


def test_alinhamento_2_modalidades_perfeito():
    registros_img = [{"caminho": f"/x/p{i:03d}.png"} for i in range(15)]
    header = ["image_id", "idade"]
    data = [[f"p{i:03d}", str(30 + i)] for i in range(15)]
    idx_img = indice_por_arquivo(registros_img)
    idx_tab, _ = indice_tabular(header, data)

    r = alinhar_modalidades({"imagem": idx_img, "tabular": idx_tab})
    assert r["alinhado"] is True
    assert r["n_pareados"] == 15
    assert set(r["modalidades"]) == {"imagem", "tabular"}


def test_alinhamento_3_modalidades_por_nome_de_arquivo():
    registros_img = [{"caminho": f"/x/paciente_{i:03d}.png"} for i in range(15)]
    registros_sinal = [{"caminho": f"/y/paciente_{i:03d}.edf"} for i in range(15)]
    header = ["patient_id", "idade"]
    data = [[f"paciente_{i:03d}", str(30 + i)] for i in range(15)]

    idx_img = indice_por_arquivo(registros_img)
    idx_sinal = indice_por_arquivo(registros_sinal)
    idx_tab, _ = indice_tabular(header, data)

    r = alinhar_modalidades({"imagem": idx_img, "sinal_temporal": idx_sinal, "tabular": idx_tab})
    assert r["alinhado"] is True
    assert r["n_pareados"] == 15
    assert set(r["modalidades"]) == {"imagem", "sinal_temporal", "tabular"}


def test_modalidade_sem_indice_e_excluida_mas_as_outras_ainda_alinham():
    registros_img = [{"caminho": f"/x/paciente_{i:03d}.png"} for i in range(15)]
    registros_sinal = [{"caminho": f"/y/paciente_{i:03d}.edf"} for i in range(15)]
    idx_img = indice_por_arquivo(registros_img)
    idx_sinal = indice_por_arquivo(registros_sinal)

    r = alinhar_modalidades({"imagem": idx_img, "sinal_temporal": idx_sinal, "tabular": {}})
    assert r["alinhado"] is True
    assert set(r["modalidades"]) == {"imagem", "sinal_temporal"}
    assert r["modalidades_excluidas"] == ["tabular"]


def test_menos_de_2_modalidades_utilizaveis_falha():
    registros_img = [{"caminho": f"/x/p{i}.png"} for i in range(15)]
    idx_img = indice_por_arquivo(registros_img)
    r = alinhar_modalidades({"imagem": idx_img, "tabular": {}, "sinal_temporal": {}})
    assert r["alinhado"] is False
    assert "identificador" in r["motivo"]


def test_poucas_amostras_em_comum_falha():
    registros_img = [{"caminho": f"/x/p{i}.png"} for i in range(15)]
    registros_sinal = [{"caminho": f"/y/outro_nome_{i}.edf"} for i in range(15)]  # nomes não batem
    idx_img = indice_por_arquivo(registros_img)
    idx_sinal = indice_por_arquivo(registros_sinal)
    r = alinhar_modalidades({"imagem": idx_img, "sinal_temporal": idx_sinal}, n_minimo=10)
    assert r["alinhado"] is False
    assert "amostra(s) em comum" in r["motivo"]


def test_normalizacao_ignora_extensao_e_maiusculas():
    registros_img = [{"caminho": "/x/IMG_007.PNG"}]
    header = ["filename"]
    data = [["img_007.png"]]
    idx_img = indice_por_arquivo(registros_img)
    idx_tab, _ = indice_tabular(header, data)
    r = alinhar_modalidades({"imagem": idx_img, "tabular": idx_tab}, n_minimo=1)
    assert r["alinhado"] is True
    assert r["n_pareados"] == 1
