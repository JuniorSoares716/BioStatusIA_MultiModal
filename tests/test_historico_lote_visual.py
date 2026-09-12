"""Testes para a unificação visual entre Histórico e Resumo do Lote —
coluna Status e botão "Imprimir relatório" em ambas as páginas."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _montar_app_com_crew_fake():
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
    sys.modules["biostatusia.crew"] = fake_crew_module

    from biostatusia.app import app
    return app


def test_historico_extrai_erro_do_pipeline_json(tmp_path, monkeypatch):
    monkeypatch.setattr("biostatusia.database.DB_PATH", tmp_path / "teste.db")
    from biostatusia.database import salvar, salvar_resultado, listar_resultados_completo

    a1 = salvar("/tmp/ok.csv", "TABULAR", "laudo")
    salvar_resultado("/tmp/ok.csv", 100, {"modo": "tabular"}, "RandomForest", a1)

    a2 = salvar("/tmp/falha.csv", "TABULAR", "laudo")
    salvar_resultado("/tmp/falha.csv", 5, {"modo": "tabular", "erro_classificador": "erro real aqui"}, "N/A", a2)

    resultados = listar_resultados_completo()
    por_erro = {r["dataset_path"]: r["erro"] for r in resultados}
    assert por_erro["/tmp/ok.csv"] == ""
    assert por_erro["/tmp/falha.csv"] == "erro real aqui"


def test_historico_mostra_status_e_botao_de_relatorio():
    app = _montar_app_com_crew_fake()
    from biostatusia.database import salvar, salvar_resultado

    a1 = salvar("/tmp/ok_status_teste.csv", "TABULAR", "laudo")
    r1 = salvar_resultado("/tmp/ok_status_teste.csv", 100, {"modo": "tabular"}, "RandomForest", a1)

    a2 = salvar("/tmp/falha_status_teste.csv", "TABULAR", "laudo")
    r2 = salvar_resultado("/tmp/falha_status_teste.csv", 5, {"modo": "tabular", "erro_classificador": "erro real"}, "N/A", a2)

    client = app.test_client()
    html = client.get("/historico").get_data(as_text=True)

    assert "Concluído" in html
    assert "Erro no treino" in html
    assert "Imprimir relatório" in html
    assert f"/relatorio_pdf/{r1}" in html
    assert f"/relatorio_pdf/{r2}" in html
    assert f"/resultados/{r1}" in html
    assert f"/resultados/{r2}" in html


def test_lote_tambem_mostra_botao_de_relatorio():
    app = _montar_app_com_crew_fake()
    from biostatusia.database import salvar, salvar_resultado

    ids = []
    for i in range(2):
        a = salvar(f"/tmp/base_{i}.csv", "TABULAR", "laudo")
        rid = salvar_resultado(f"/tmp/base_{i}.csv", 20, {"modo": "tabular"}, "RandomForest", a)
        ids.append(rid)

    client = app.test_client()
    html = client.get("/lote?ids=" + ",".join(str(i) for i in ids)).get_data(as_text=True)
    assert html.count("Imprimir relatório") == 2
    for rid in ids:
        assert f"/relatorio_pdf/{rid}" in html


def test_historico_modos_distintos_no_lugar_de_codigos_internos():
    app = _montar_app_com_crew_fake()
    from biostatusia.database import salvar, salvar_resultado

    salvar_resultado("/tmp/a.csv", 10, {"modo": "tabular"}, "N/A", salvar("/tmp/a.csv", "TABULAR", "laudo"))
    salvar_resultado("/tmp/b.png", 10, {"modo": "dataset_rotulado"}, "N/A", salvar("/tmp/b.png", "BENIGNO", "laudo"))

    client = app.test_client()
    html = client.get("/historico").get_data(as_text=True)
    assert "Modos Distintos" in html
    assert "Tabular" in html
    assert "Dataset Rotulado" in html
