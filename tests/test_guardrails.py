"""Guardrails — invariantes da constituição do projeto."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
TEMPLATES = RAIZ / "src" / "biostatusia" / "templates"
CONFIG = RAIZ / "src" / "biostatusia" / "config"


def test_aviso_etico_em_todas_as_telas():
    for tela in ("tela1_upload.html", "tela2_resultados.html", "tela3_historico.html"):
        txt = (TEMPLATES / tela).read_text(encoding="utf-8").lower()
        assert "não substitui" in txt, f"aviso ético ausente em {tela}"


def test_laudo_amostra_tem_5_secoes():
    txt = (CONFIG / "tasks.yaml").read_text(encoding="utf-8")
    for secao in ("Achado Principal", "Severidade", "Comparação com Referência",
                  "Recomendação Imediata", "Aviso Ético"):
        assert secao in txt, f"seção obrigatória ausente: {secao}"


def test_nav_sem_pacientes_relatorios():
    # Links "Pacientes"/"Relatórios" foram removidos da navegação
    for tela in ("tela1_upload.html", "tela2_resultados.html", "tela3_historico.html"):
        txt = (TEMPLATES / tela).read_text(encoding="utf-8")
        assert ">Pacientes<" not in txt
        assert ">Relatórios<" not in txt


def test_max_content_length_nao_reduzido():
    from biostatusia.app import app
    assert app.config["MAX_CONTENT_LENGTH"] == 4096 * 1024 * 1024
