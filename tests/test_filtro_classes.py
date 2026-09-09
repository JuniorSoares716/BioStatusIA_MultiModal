"""Testes para a escolha explícita de classes dentro de uma coluna-alvo com
mais de 2 valores (contagem real por classe + filtro para exatamente 2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.dados_tabulares import candidatos_coluna_alvo, filtrar_por_classes


def _dataset_desbalanceado():
    header = ["id", "idade", "Medical Condition"]
    pesos = {"Diabetes": 40, "Hypertension": 25, "Asthma": 15, "Cancer": 10, "Obesity": 7, "Arthritis": 3}
    data = []
    i = 0
    for cond, peso in pesos.items():
        for _ in range(peso):
            data.append([f"p{i}", str(30 + i % 50), cond])
            i += 1
    return header, data


def test_candidatos_reporta_contagem_real_por_classe():
    header, data = _dataset_desbalanceado()
    cands = candidatos_coluna_alvo(header, data)
    mc = next(c for c in cands if c["nome"] == "Medical Condition")
    contagem = {c["valor"]: c["n"] for c in mc["classes"]}
    assert contagem == {
        "Diabetes": 40, "Hypertension": 25, "Asthma": 15,
        "Cancer": 10, "Obesity": 7, "Arthritis": 3,
    }


def test_classes_ordenadas_por_frequencia_decrescente():
    header, data = _dataset_desbalanceado()
    cands = candidatos_coluna_alvo(header, data)
    mc = next(c for c in cands if c["nome"] == "Medical Condition")
    valores_em_ordem = [c["valor"] for c in mc["classes"]]
    assert valores_em_ordem == ["Diabetes", "Hypertension", "Asthma", "Cancer", "Obesity", "Arthritis"]


def test_filtrar_por_classes_mantem_apenas_as_escolhidas():
    header, data = _dataset_desbalanceado()
    filtrado = filtrar_por_classes(header, data, 2, ["Diabetes", "Cancer"])
    assert len(filtrado) == 50  # 40 + 10
    assert set(row[2] for row in filtrado) == {"Diabetes", "Cancer"}


def test_filtrar_por_classes_com_lista_vazia_nao_mantem_nada():
    header, data = _dataset_desbalanceado()
    filtrado = filtrar_por_classes(header, data, 2, [])
    assert filtrado == []


def test_filtrar_por_classe_inexistente_nao_mantem_nada():
    header, data = _dataset_desbalanceado()
    filtrado = filtrar_por_classes(header, data, 2, ["NaoExiste"])
    assert filtrado == []
