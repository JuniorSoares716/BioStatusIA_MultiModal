"""Tier 2 — rotas Flask (GET read-only + validação de entrada)."""


def test_home(client):
    r = client.get("/")
    assert r.status_code == 200


def test_historico(client):
    r = client.get("/historico")
    assert r.status_code == 200


def test_resultado_inexistente_404(client):
    assert client.get("/resultados/999999999").status_code == 404


def test_api_historico_json(client):
    r = client.get("/api/historico")
    assert r.status_code == 200
    assert isinstance(r.get_json(), list)


def test_laudo_amostra_sem_entrada_400(client):
    r = client.post("/laudo_amostra", data={})
    assert r.status_code == 400


def test_laudo_populacional_inexistente_404(client):
    assert client.get("/laudo_populacional/999999999").status_code == 404


def test_max_content_length_4gb():
    from biostatusia.app import app
    assert app.config["MAX_CONTENT_LENGTH"] == 4096 * 1024 * 1024


def test_laudo_amostra_com_crew_mockada(client, monkeypatch, tmp_path):
    """/laudo_amostra com upload de CSV e o agente (crew) substituído por um dublê —
    valida o fluxo da rota (extração + persistência + resposta) sem LLM/crewai."""
    import sys
    import types

    fake = types.ModuleType("biostatusia.crew")

    class _CrewStub:
        def kickoff(self, inputs=None):
            return ("## Laudo de Amostra\n"
                    "**Achado Principal:** achado de teste.\n"
                    "**Aviso Ético:** não substitui avaliação médica.")

    class BioStatusIACrewInterativo:
        def crew(self):
            return _CrewStub()

    fake.BioStatusIACrewInterativo = BioStatusIACrewInterativo
    monkeypatch.setitem(sys.modules, "biostatusia.crew", fake)
    # (o banco já está isolado em arquivo temporário pelo fixture `client`)

    csv = tmp_path / "amostra.csv"
    csv.write_text("a,b,label\n1,2,M\n3,4,B\n", encoding="utf-8")
    with open(csv, "rb") as f:
        r = client.post(
            "/laudo_amostra",
            data={"arquivo": (f, "amostra.csv")},
            content_type="multipart/form-data",
        )
    assert r.status_code == 200
    body = r.get_json()
    assert "laudo_html" in body
    assert "Achado Principal" in body["laudo_html"]
