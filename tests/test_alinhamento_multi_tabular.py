"""Testes para: reconhecimento de 'echo' como coluna de ID, múltiplos
arquivos tabulares, e alinhamento sem exigir rótulo prévio na imagem
(casos descobertos testando com o dataset real HMC-QU)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.alinhamento_multimodal import detectar_coluna_id, indice_por_arquivo
from biostatusia.pipeline.io_utils import encontrar_todos_tabulares


def test_echo_e_reconhecido_como_coluna_de_id():
    assert detectar_coluna_id(["ECHO", "Infarction Label"]) == 0
    assert detectar_coluna_id(["idade", "echo"]) == 1


def test_encontrar_todos_tabulares_encontra_mais_de_um_arquivo(tmp_path):
    (tmp_path / "A2C.xlsx").write_bytes(b"PK\x03\x04" + b"0" * 20)
    (tmp_path / "A4C.xlsx").write_bytes(b"PK\x03\x04" + b"0" * 20)
    (tmp_path / "notas.txt").write_text("nao e tabular relevante aqui, mas .txt conta como tabular tambem")

    arquivos = encontrar_todos_tabulares(tmp_path)
    nomes = {a.name for a in arquivos}
    assert nomes == {"A2C.xlsx", "A4C.xlsx", "notas.txt"}


def test_indice_por_arquivo_nao_exige_rotulo_previo():
    # Registros SEM rótulo de pasta (label=None) — comum quando o rótulo só
    # existe na tabela, não na estrutura de pastas de imagem.
    registros = [{"caminho": f"/x/p{i}.png", "label": None, "categoria": "INDEFINIDO"} for i in range(10)]
    idx = indice_por_arquivo(registros)
    assert len(idx) == 10  # todos entram no índice, apesar de não terem rótulo ainda
