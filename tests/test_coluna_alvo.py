"""Testes para a seleção manual de coluna-alvo (candidatos_coluna_alvo +
label_idx_forcado em detectar_schema)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.dados_tabulares import candidatos_coluna_alvo, detectar_schema


def _dataset_healthcare():
    header = ["Patient_ID", "Age", "Gender", "Medical Condition", "Admission Type", "Test Results"]
    condicoes = ["Diabetes", "Hypertension", "Asthma", "Obesity", "Arthritis", "Cancer"]
    admissao = ["Emergency", "Elective", "Urgent"]
    resultados = ["Normal", "Abnormal", "Inconclusive"]
    generos = ["Male", "Female"]
    data = []
    for i in range(60):
        data.append([
            f"P{i}", str(20 + i % 60), generos[i % 2], condicoes[i % 6],
            admissao[i % 3], resultados[i % 3],
        ])
    return header, data


def test_candidatos_encontra_todas_as_colunas_categoricas():
    header, data = _dataset_healthcare()
    cands = candidatos_coluna_alvo(header, data)
    nomes = {c["nome"] for c in cands}
    assert nomes == {"Gender", "Medical Condition", "Admission Type", "Test Results"}


def test_candidatos_reporta_numero_de_classes_correto():
    header, data = _dataset_healthcare()
    cands = candidatos_coluna_alvo(header, data)
    por_nome = {c["nome"]: c for c in cands}
    assert por_nome["Medical Condition"]["n_classes"] == 6
    assert por_nome["Gender"]["n_classes"] == 2
    assert por_nome["Test Results"]["n_classes"] == 3


def test_deteccao_automatica_pega_ultima_coluna_por_padrao():
    header, data = _dataset_healthcare()
    schema = detectar_schema(header, data)
    assert schema["label_name"] == "Test Results"  # última coluna — comportamento antigo


def test_label_idx_forcado_sobrepoe_a_heuristica():
    header, data = _dataset_healthcare()
    idx_medical_condition = header.index("Medical Condition")
    schema = detectar_schema(header, data, label_idx_forcado=idx_medical_condition)
    assert schema["label_name"] == "Medical Condition"
    assert schema["label_idx"] == idx_medical_condition


def test_label_idx_forcado_invalido_cai_no_automatico():
    header, data = _dataset_healthcare()
    schema = detectar_schema(header, data, label_idx_forcado=999)  # fora do range
    assert schema["label_name"] == "Test Results"  # ignora o índice inválido, usa a heurística


def test_dataset_sem_colunas_categoricas_suficientes_nao_gera_candidatos_demais():
    # Uma coluna de ID única por linha (60 valores distintos) não deve virar candidata
    header = ["id", "valor"]
    data = [[str(i), str(i * 2)] for i in range(60)]
    cands = candidatos_coluna_alvo(header, data)
    assert cands == []
