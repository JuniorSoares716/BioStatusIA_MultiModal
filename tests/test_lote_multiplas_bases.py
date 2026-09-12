"""Testes para o processamento em lote (múltiplas bases independentes
enviadas de uma vez): detecção de bases misturadas num único upload,
processamento individual de cada uma, e a página de resumo do lote."""
import csv
import random
import re
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.app import detectar_multiplas_bases


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


def _csv_dataset(pasta: Path, seed: int, n: int = 20):
    pasta.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with open(pasta / "dados.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idade", "diagnostico"])
        for j in range(n):
            w.writerow([30 + j, rng.choice(["sim", "nao"])])


# ── detectar_multiplas_bases() ──────────────────────────────────────────────

def test_detecta_multiplas_bases_tabulares_independentes():
    tmp = Path(tempfile.mkdtemp())
    for i in range(5):
        _csv_dataset(tmp / f"dataset_{i}", seed=i)

    resultado = detectar_multiplas_bases(tmp)
    assert resultado is not None
    assert len(resultado) == 5
    assert {b["modo"] for b in resultado} == {"tabular"}


def test_nao_confunde_benign_malignant_com_lote():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "benign").mkdir()
    (tmp / "malignant").mkdir()
    for i in range(10):
        (tmp / "benign" / f"img_{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)
        (tmp / "malignant" / f"img_{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)

    assert detectar_multiplas_bases(tmp) is None


def test_nao_dispara_com_menos_de_3_subpastas():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "imagens").mkdir()
    (tmp / "tabular").mkdir()
    (tmp / "imagens" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)
    _csv_dataset(tmp / "tabular", seed=1)

    assert detectar_multiplas_bases(tmp) is None


# ── Portão de confirmação em /analisar ──────────────────────────────────────

def test_analisar_detecta_lote_e_oferece_escolha():
    app = _montar_app_com_crew_fake()
    client = app.test_client()

    tmp = Path(tempfile.mkdtemp())
    for i in range(4):
        _csv_dataset(tmp / f"base_{i}", seed=i)

    r = client.post("/analisar", data={"caminho_manual": str(tmp)})
    assert r.status_code == 200
    j = r.get_json()
    assert j["multiplas_bases"] is True
    assert len(j["bases"]) == 4


def test_analisar_ignorar_lote_processa_como_base_unica():
    app = _montar_app_com_crew_fake()
    client = app.test_client()

    tmp = Path(tempfile.mkdtemp())
    for i in range(4):
        _csv_dataset(tmp / f"base_{i}", seed=i)

    r1 = client.post("/analisar", data={"caminho_manual": str(tmp)})
    dataset_path = r1.get_json()["dataset_path"]

    r2 = client.post("/analisar", data={"caminho_manual": dataset_path, "ignorar_lote": "1"})
    assert r2.status_code == 302  # segue o fluxo normal de uma base só


def test_cada_base_do_lote_processada_de_forma_independente():
    app = _montar_app_com_crew_fake()
    from biostatusia.database import buscar_resultado
    client = app.test_client()

    tmp = Path(tempfile.mkdtemp())
    for i in range(3):
        _csv_dataset(tmp / f"base_{i}", seed=i)

    r1 = client.post("/analisar", data={"caminho_manual": str(tmp)})
    bases = r1.get_json()["bases"]

    ids = []
    for b in bases:
        r = client.post("/analisar", data={"caminho_manual": b["caminho"]})
        m = re.search(r"/resultados/(\d+)", r.headers.get("Location", ""))
        assert m, f"base {b['nome']} não foi processada com sucesso"
        ids.append(int(m.group(1)))

    assert len(set(ids)) == 3  # todos diferentes
    for rid, b in zip(ids, bases):
        dados = buscar_resultado(rid)
        assert dados["dataset_path"] == b["caminho"]  # cada uma leu a base certa, não uma mistura


# ── Rota /lote (resumo) ──────────────────────────────────────────────────────

def test_rota_lote_resume_resultados_ja_processados():
    app = _montar_app_com_crew_fake()
    from biostatusia.database import salvar, salvar_resultado

    ids = []
    for i in range(3):
        a = salvar(f"/tmp/base_{i}.csv", "TABULAR", "laudo")
        rid = salvar_resultado(f"/tmp/base_{i}.csv", 20, {"modo": "tabular"}, "RandomForest", a)
        ids.append(rid)

    client = app.test_client()
    resp = client.get("/lote?ids=" + ",".join(str(i) for i in ids))
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert html.count("Ver resultado") == 3


def test_rota_lote_com_ids_inexistentes_nao_quebra():
    app = _montar_app_com_crew_fake()
    client = app.test_client()
    resp = client.get("/lote?ids=999997,999998,999999")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "não puderam ser carregadas" in html or "Nenhum resultado encontrado" in html
