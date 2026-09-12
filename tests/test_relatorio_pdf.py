"""Testes para o relatório PDF final (combina Estatísticas, Pré-processamento,
AutoML, Fusão Multimodal e Laudo num único arquivo)."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.relatorio_pdf import gerar_relatorio_pdf


def _pipeline_minimo():
    metricas = {
        "RandomForest": {
            "acuracia": 0.9, "sensibilidade": 0.9, "especificidade": 0.9, "f1": 0.9,
            "auc": 0.9, "ece": 0.05, "score_clinico": 0.8, "score_clinico_cv": 0.75,
            "sensibilidade_cv": 0.85, "tempo_treino_s": 0.2,
        },
    }
    return {
        "modo": "tabular", "n_amostras": 100,
        "metricas": metricas, "melhor_modelo": "RandomForest",
        "confusion_matrix": {"RandomForest": [[8, 2], [1, 9]]},
    }


def test_gerar_relatorio_pdf_com_dados_minimos_nao_quebra():
    resultado = {
        "id": 1, "data_hora": "2026-01-01", "dataset_path": "/tmp/dados.csv",
        "n_imagens": 100, "pipeline": _pipeline_minimo(), "melhor_modelo": "RandomForest",
        "laudo": None,
    }
    pdf_bytes = gerar_relatorio_pdf(resultado)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500


def test_relatorio_pdf_nao_tem_glifos_ausentes():
    """Regressão: símbolos Unicode (⚠, ★, •) não existem na fonte padrão do
    reportlab e viravam '(cid:127)' no texto extraído — bug real encontrado
    e corrigido durante a implementação desta funcionalidade."""
    import pdfplumber
    from io import BytesIO

    p = _pipeline_minimo()
    p["piso_sensibilidade_atingido"] = False
    p["aviso_piso_sensibilidade"] = "Nenhum modelo atingiu o piso mínimo."
    resultado = {
        "id": 1, "data_hora": "2026-01-01", "dataset_path": "/tmp/dados.csv",
        "n_imagens": 100, "pipeline": p, "melhor_modelo": "RandomForest",
        "laudo": "# Título\n\n- Item de lista um\n- Item de lista dois\n\n**Negrito** no texto.",
    }
    pdf_bytes = gerar_relatorio_pdf(resultado)
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        texto_total = "\n".join((page.extract_text() or "") for page in pdf.pages)
    assert "(cid:" not in texto_total
    assert "Item de lista um" in texto_total
    assert "AVISO" in texto_total


def test_relatorio_pdf_com_fusao_multimodal():
    p = _pipeline_minimo()
    p["modo"] = "multimodal"
    p["alinhamento_multimodal"] = {"alinhado": True, "n_pareados": 50}
    p["fusao_multimodal"] = {
        "modalidades": ["imagem", "tabular"], "melhor_modelo": "fusao_stacking",
        "metricas": {
            "imagem": {"acuracia": 0.7, "sensibilidade": 0.65, "auc": 0.75, "f1": 0.6},
            "fusao_stacking": {"acuracia": 0.88, "sensibilidade": 0.85, "auc": 0.93, "f1": 0.86},
        },
    }
    resultado = {
        "id": 1, "data_hora": "2026-01-01", "dataset_path": "/tmp/dados.csv",
        "n_imagens": 100, "pipeline": p, "melhor_modelo": "RandomForest", "laudo": None,
    }
    pdf_bytes = gerar_relatorio_pdf(resultado)

    import pdfplumber
    from io import BytesIO
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        texto = "\n".join((page.extract_text() or "") for page in pdf.pages)
    assert "Fusão Multimodal" in texto
    assert "fusao_stacking" in texto.lower() or "stacking" in texto.lower()


def test_relatorio_pdf_sem_alinhamento_multimodal_nao_mostra_secao_fusao():
    p = _pipeline_minimo()
    resultado = {
        "id": 1, "data_hora": "2026-01-01", "dataset_path": "/tmp/dados.csv",
        "n_imagens": 100, "pipeline": p, "melhor_modelo": "RandomForest", "laudo": None,
    }
    pdf_bytes = gerar_relatorio_pdf(resultado)
    import pdfplumber
    from io import BytesIO
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        texto = "\n".join((page.extract_text() or "") for page in pdf.pages)
    assert "Fusão Multimodal" not in texto


def test_rota_relatorio_pdf_funciona_ponta_a_ponta(monkeypatch):
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

    analise_id = salvar("/tmp/img.png", "BENIGNO", "Laudo teste")
    resultado_id = salvar_resultado("/tmp/dataset", 20, _pipeline_minimo(), "RandomForest", analise_id)

    client = app.test_client()
    resp = client.get(f"/relatorio_pdf/{resultado_id}")
    assert resp.status_code == 200
    assert resp.content_type == "application/pdf"
    assert resp.data.startswith(b"%PDF")
    assert "attachment" in resp.headers.get("Content-Disposition", "")


def test_rota_relatorio_pdf_404_para_resultado_inexistente():
    from biostatusia.app import app
    client = app.test_client()
    resp = client.get("/relatorio_pdf/999999")
    assert resp.status_code == 404
