"""Testes para: suporte a .xlsx no carregador tabular, e blindagem contra 500."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.dados_tabulares import carregar_csv
from biostatusia.pipeline.io_utils import eh_tabular


def test_xlsx_e_reconhecido_como_tabular():
    assert eh_tabular(Path("dados.xlsx")) is True
    assert eh_tabular(Path("dados.csv")) is True
    assert eh_tabular(Path("dados.pdf")) is False


def test_carregar_xlsx_produz_mesmo_formato_que_csv(tmp_path):
    from openpyxl import Workbook

    caminho = tmp_path / "A4C.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["ECHO", "idade", "Infarction Label"])
    for i in range(15):
        ws.append([f"A4C_{i:03d}", 40 + i, i % 2])
    wb.save(caminho)

    resultado = carregar_csv(str(caminho))
    assert resultado is not None
    header, data = resultado
    assert header == ["ECHO", "idade", "Infarction Label"]
    assert len(data) == 15
    assert data[0][0] == "A4C_000"


def test_carregar_xlsx_inexistente_retorna_none():
    assert carregar_csv("/caminho/que/nao/existe.xlsx") is None


def test_arquivo_csv_com_conteudo_excel_binario_e_lido_corretamente(tmp_path):
    """Extensão .csv, mas o conteúdo é um Excel binário de verdade (arquivo
    exportado/renomeado com a extensão errada) — deve detectar pelo
    conteúdo (assinatura ZIP) e ler como planilha mesmo assim."""
    from openpyxl import Workbook

    caminho_real_xlsx = tmp_path / "origem.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["id", "idade", "diagnostico"])
    for i in range(12):
        ws.append([f"p{i}", 30 + i, "sim" if i % 2 else "nao"])
    wb.save(caminho_real_xlsx)

    caminho_enganado = tmp_path / "dados.csv"  # extensão .csv, conteúdo Excel
    caminho_enganado.write_bytes(caminho_real_xlsx.read_bytes())

    resultado = carregar_csv(str(caminho_enganado))
    assert resultado is not None
    header, data = resultado
    assert header == ["id", "idade", "diagnostico"]
    assert len(data) == 12


def test_arquivo_xlsx_com_conteudo_texto_puro_e_lido_corretamente(tmp_path):
    """Extensão .xlsx, mas o conteúdo é texto CSV puro — deve detectar pelo
    conteúdo (não é um ZIP) e ler como texto delimitado mesmo assim."""
    caminho_enganado = tmp_path / "dados.xlsx"  # extensão .xlsx, conteúdo texto
    caminho_enganado.write_text("id,idade,diagnostico\n" + "\n".join(
        f"p{i},{30+i},{'sim' if i % 2 else 'nao'}" for i in range(12)
    ))

    resultado = carregar_csv(str(caminho_enganado))
    assert resultado is not None
    header, data = resultado
    assert header == ["id", "idade", "diagnostico"]
    assert len(data) == 12


def test_erro_500_em_analisar_devolve_json_e_nao_pagina_generica(monkeypatch):
    import types
    fake_crew_module = types.ModuleType("biostatusia.crew")

    class FakeOutput:
        def __str__(self):
            return "Laudo simulado."

    class FakeCrewQuebrada:
        def kickoff(self, inputs=None):
            return FakeOutput()

    class WrapperQuebrado:
        def crew(self):
            return FakeCrewQuebrada()

    class WrapperGenerico:
        def crew(self):
            return FakeCrewQuebrada()

    fake_crew_module.BioStatusIACrew = WrapperQuebrado
    fake_crew_module.BioStatusIACrewTabular = WrapperGenerico
    fake_crew_module.BioStatusIACrewSinal = WrapperGenerico
    fake_crew_module.BioStatusIACrewImagem3D = WrapperGenerico
    fake_crew_module.BioStatusIACrewInterativo = WrapperGenerico
    monkeypatch.setitem(sys.modules, "biostatusia.crew", fake_crew_module)

    from biostatusia import app as app_module
    import importlib
    importlib.reload(app_module)

    # Força _consolidar_imagem a quebrar (simula um dataset fora do padrão,
    # ex.: máscaras de segmentação sem os artefatos esperados da crew)
    def _consolidar_quebrado(pasta_run, modo):
        raise ValueError("Falha simulada de consolidação")

    monkeypatch.setattr(app_module, "_consolidar_imagem", _consolidar_quebrado)

    client = app_module.app.test_client()
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    (tmp / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)

    resp = client.post("/analisar", data={"caminho_manual": str(tmp)})
    # Não deve ser um crash sem tratamento (500 puro) nem travar o processo —
    # deve ser tratado e devolver algo navegável (redirect com erro reportado
    # no pipeline, já que agora _consolidar_imagem é protegida por try/except).
    assert resp.status_code in (302, 500)
    if resp.status_code == 500:
        dados = resp.get_json()
        assert dados is not None and dados.get("erro") is True
        assert "mensagem" in dados
