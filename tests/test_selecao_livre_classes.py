"""Testes para: seleção livre de classes (sem trava em 2), com aviso claro
(sem crash nem resultado silenciosamente errado) quando mais de 2 classes
acabam sendo usadas — e treino normal quando exatamente 2 são escolhidas."""
import csv
import re
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _montar_app_com_crew_fake():
    fake_crew_module = types.ModuleType("biostatusia.crew")

    class FakeCrewOutput:
        def __str__(self):
            return "Laudo simulado."

    class FakeCrewGenerico:
        def kickoff(self, inputs=None):
            return FakeCrewOutput()

    class WGen:
        def crew(self):
            return FakeCrewGenerico()

    for nome in ("BioStatusIACrew", "BioStatusIACrewTabular", "BioStatusIACrewSinal",
                 "BioStatusIACrewImagem3D", "BioStatusIACrewInterativo"):
        setattr(fake_crew_module, nome, WGen)
    sys.modules["biostatusia.crew"] = fake_crew_module

    from biostatusia.app import app
    return app


def _montar_csv_multiclasse(tmp_path):
    header = ["id", "idade", "Medical Condition"]
    pesos = {"Diabetes": 40, "Hypertension": 25, "Asthma": 15, "Cancer": 10, "Obesity": 7, "Arthritis": 3}
    rows = []
    i = 0
    for cond, peso in pesos.items():
        for _ in range(peso):
            rows.append([f"p{i}", str(30 + i % 50), cond])
            i += 1
    with open(tmp_path / "dados.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return tmp_path


def test_selecionar_todas_as_classes_treina_multiclasse_de_verdade(tmp_path):
    app = _montar_app_com_crew_fake()
    from biostatusia.database import buscar_resultado
    _montar_csv_multiclasse(tmp_path)
    client = app.test_client()

    r1 = client.post("/analisar", data={"caminho_manual": str(tmp_path)})
    j1 = r1.get_json()
    todas = [c["valor"] for c in next(
        x for x in j1["colunas_candidatas"] if x["nome"] == "Medical Condition"
    )["classes"]]

    r2 = client.post("/analisar", data={
        "caminho_manual": j1["dataset_path"], "confirmado": "1",
        "coluna_alvo": "Medical Condition", "classes_selecionadas": todas,
    })
    assert r2.status_code == 302  # não quebra

    m = re.search(r"/resultados/(\d+)", r2.headers.get("Location", ""))
    p = buscar_resultado(int(m.group(1)))["pipeline"]
    assert p["tabular_stats"]["n_amostras"] == 100  # todas as amostras, sem filtro
    assert p.get("n_classes") == 6  # treinou de verdade com as 6 classes
    assert p.get("melhor_modelo") not in (None, "N/A")


def test_selecionar_exatamente_2_classes_ainda_treina_normalmente(tmp_path):
    app = _montar_app_com_crew_fake()
    from biostatusia.database import buscar_resultado
    _montar_csv_multiclasse(tmp_path)
    client = app.test_client()

    r1 = client.post("/analisar", data={"caminho_manual": str(tmp_path)})
    j1 = r1.get_json()

    r2 = client.post("/analisar", data={
        "caminho_manual": j1["dataset_path"], "confirmado": "1",
        "coluna_alvo": "Medical Condition", "classes_selecionadas": ["Diabetes", "Cancer"],
    })
    m = re.search(r"/resultados/(\d+)", r2.headers.get("Location", ""))
    p = buscar_resultado(int(m.group(1)))["pipeline"]
    assert p["tabular_stats"]["n_amostras"] == 50
    assert p.get("melhor_modelo") not in (None, "N/A")
