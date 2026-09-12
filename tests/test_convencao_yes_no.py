"""Teste para a convenção de rótulo por pasta "yes"/"no" — comum em datasets
de detecção de tumor (ex.: o dataset clássico "Brain MRI Images for Brain
Tumor Detection" do Kaggle usa exatamente essa convenção). Antes desta
correção, imagens nessas pastas ficavam sem rótulo (INDEFINIDO), mesmo
quando o dataset tinha uma distinção binária clara."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.io_utils import label_pasta, listar_imagens


def test_label_pasta_reconhece_yes_no():
    assert label_pasta("yes") == 1
    assert label_pasta("no") == 0
    assert label_pasta("Yes") == 1  # case-insensitive
    assert label_pasta("NO") == 0


def test_listar_imagens_rotula_pastas_yes_no_corretamente(tmp_path):
    (tmp_path / "yes").mkdir()
    (tmp_path / "no").mkdir()
    for i in range(3):
        (tmp_path / "yes" / f"img_{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)
    for i in range(2):
        (tmp_path / "no" / f"img_{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)

    registros = listar_imagens(tmp_path)
    from collections import Counter
    c = Counter(r["categoria"] for r in registros)
    assert c["MALIGNO"] == 3
    assert c["BENIGNO"] == 2
    assert c.get("INDEFINIDO", 0) == 0


def test_pastas_benignas_malignas_vazias_nao_atrapalham_yes_no(tmp_path):
    """Regressão do caso real reportado: pastas benign/malignant vazias
    (decoy) coexistindo com as pastas yes/no que têm as imagens de verdade."""
    (tmp_path / "benign").mkdir()
    (tmp_path / "malignant").mkdir()
    (tmp_path / "yes").mkdir()
    (tmp_path / "no").mkdir()
    (tmp_path / "yes" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)
    (tmp_path / "no" / "b.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)

    registros = listar_imagens(tmp_path)
    assert len(registros) == 2  # as pastas vazias nao contribuem arquivos, mas nao quebram nada
    categorias = {r["categoria"] for r in registros}
    assert categorias == {"MALIGNO", "BENIGNO"}
