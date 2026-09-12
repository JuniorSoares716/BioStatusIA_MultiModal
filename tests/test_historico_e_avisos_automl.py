"""Testes para: (1) separação de 'modo' e 'categoria' no histórico (antes
misturava rótulo de modo com diagnóstico real na mesma coluna/campo), e
(2) exibição do motivo real de 'AutoML não executado' (antes calculado e
guardado, mas nunca mostrado na tela)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_listar_resultados_completo_extrai_modo_do_pipeline_json(tmp_path, monkeypatch):
    monkeypatch.setattr("biostatusia.database.DB_PATH", tmp_path / "teste.db")
    from biostatusia.database import salvar, salvar_resultado, listar_resultados_completo

    a1 = salvar("/tmp/x.csv", "TABULAR", "laudo")
    salvar_resultado("/tmp/x.csv", 100, {"modo": "tabular"}, "RandomForest", a1)

    a2 = salvar("/tmp/img.png", "BENIGNO", "laudo")
    salvar_resultado("/tmp/imgs", 50, {"modo": "dataset_rotulado"}, "SVM", a2)

    resultados = listar_resultados_completo()
    por_id = {r["id"]: r for r in resultados}
    assert por_id[a1 if False else resultados[-1]["id"]]  # sanity: nao usado, so garante execucao
    modos = {r["categoria"]: r["modo"] for r in resultados}
    assert modos["TABULAR"] == "tabular"
    assert modos["BENIGNO"] == "dataset_rotulado"


def test_historico_nao_mostra_modo_como_badge_de_diagnostico(monkeypatch, tmp_path):
    import types
    fake_crew_module = types.ModuleType("biostatusia.crew")

    class FakeCrewOutput:
        def __str__(self):
            return "x"

    class FakeCrewGenerico:
        def kickoff(self, inputs=None):
            return FakeCrewOutput()

    class WGen:
        def crew(self):
            return FakeCrewGenerico()

    for nome in ("BioStatusIACrew", "BioStatusIACrewTabular", "BioStatusIACrewSinal",
                 "BioStatusIACrewImagem3D", "BioStatusIACrewInterativo"):
        setattr(fake_crew_module, nome, WGen)
    monkeypatch.setitem(sys.modules, "biostatusia.crew", fake_crew_module)

    from biostatusia.app import app
    from biostatusia.database import salvar, salvar_resultado

    salvar_resultado("/tmp/x.csv", 195, {"modo": "tabular"}, "N/A", salvar("/tmp/x.csv", "TABULAR", "laudo"))
    salvar_resultado("/tmp/imgs", 780, {"modo": "dataset_rotulado"}, "SVM", salvar("/tmp/imgs", "BENIGNO", "laudo"))

    client = app.test_client()
    resp = client.get("/historico")
    html = resp.get_data(as_text=True)
    assert "Tabular" in html  # coluna Modo mostra o modo real
    assert "Dataset Rotulado" in html
    assert "BENIGNO" in html  # diagnostico real continua aparecendo como badge
    # "TABULAR" nao deveria mais aparecer como badge de diagnostico (era o bug)
    assert 'text-on-surface-variant">TABULAR<' not in html


def test_automl_nao_executado_mostra_aviso_especifico(monkeypatch):
    import types
    fake_crew_module = types.ModuleType("biostatusia.crew")

    class FakeCrewOutput:
        def __str__(self):
            return "x"

    class FakeCrewGenerico:
        def kickoff(self, inputs=None):
            return FakeCrewOutput()

    class WGen:
        def crew(self):
            return FakeCrewGenerico()

    for nome in ("BioStatusIACrew", "BioStatusIACrewTabular", "BioStatusIACrewSinal",
                 "BioStatusIACrewImagem3D", "BioStatusIACrewInterativo"):
        setattr(fake_crew_module, nome, WGen)
    monkeypatch.setitem(sys.modules, "biostatusia.crew", fake_crew_module)

    from biostatusia.app import app
    from biostatusia.database import salvar, salvar_resultado

    client = app.test_client()

    # Caso 1: erro real de treino
    p1 = {"metricas": {}, "melhor_modelo": "N/A", "modo": "tabular",
          "erro_classificador": "The least populated class in y has only 1 member."}
    r1 = salvar_resultado("/tmp/x.csv", 5, p1, "N/A", salvar("/tmp/x.csv", "TABULAR", "laudo"))
    h1 = client.get(f"/resultados/{r1}").get_data(as_text=True)
    assert "least populated class" in h1

    # Caso 2: aviso especifico de dados insuficientes
    p2 = {"metricas": {}, "melhor_modelo": "N/A", "modo": "tabular",
          "aviso_classificador": "Treino não executado: 8 amostras, 2 classes."}
    r2 = salvar_resultado("/tmp/x.csv", 8, p2, "N/A", salvar("/tmp/x.csv", "TABULAR", "laudo"))
    h2 = client.get(f"/resultados/{r2}").get_data(as_text=True)
    assert "8 amostras, 2 classes" in h2

    # Caso 3: falha especifica da parte tabular dentro de multimodal
    p3 = {"metricas": {}, "melhor_modelo": "N/A", "modo": "multimodal",
          "erro_classificador_tabular": "could not convert string to float: N/A"}
    r3 = salvar_resultado("/tmp/x.csv", 100, p3, "N/A", salvar("/tmp/x.csv", "MULTIMODAL_EXPANDIDO", "laudo"))
    h3 = client.get(f"/resultados/{r3}").get_data(as_text=True)
    assert "could not convert string to float" in h3
