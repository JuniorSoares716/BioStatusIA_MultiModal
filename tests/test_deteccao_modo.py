"""Tier 1 — detecção de modo (detectar_estrutura) e predicados io_utils."""
from pathlib import Path

from biostatusia.app import detectar_estrutura
from biostatusia.pipeline import io_utils


def _touch(p: Path) -> Path:
    p.write_bytes(b"")
    return p


def test_imagem_unica(tmp_path):
    assert detectar_estrutura(_touch(tmp_path / "a.png")) == "imagem_unica"


def test_tabular(tmp_path):
    assert detectar_estrutura(_touch(tmp_path / "a.csv")) == "tabular"


def test_sinal_temporal(tmp_path):
    assert detectar_estrutura(_touch(tmp_path / "a.mat")) == "sinal_temporal"
    assert detectar_estrutura(_touch(tmp_path / "b.edf")) == "sinal_temporal"


def test_dicom_2d(tmp_path):
    assert detectar_estrutura(_touch(tmp_path / "a.dcm")) == "imagem_dicom_2d"


def test_volume_nii(tmp_path):
    assert detectar_estrutura(_touch(tmp_path / "a.nii")) == "volume_3d"
    assert detectar_estrutura(_touch(tmp_path / "b.nii.gz")) == "volume_3d"


def test_volume_serie_dicom(tmp_path):
    for i in range(12):
        _touch(tmp_path / f"s{i}.dcm")
    assert detectar_estrutura(tmp_path) == "volume_3d"


def test_dataset_rotulado(dataset_rotulado):
    assert detectar_estrutura(dataset_rotulado) == "dataset_rotulado"


def test_imagens_soltas(tmp_path):
    for i in range(3):
        _touch(tmp_path / f"x{i}.jpg")
    assert detectar_estrutura(tmp_path) == "imagens_soltas"


def test_multimodal(tmp_path):
    _touch(tmp_path / "img.png")
    _touch(tmp_path / "tab.csv")
    assert detectar_estrutura(tmp_path) == "multimodal"


def test_invalido_audio_video(tmp_path):
    # F2 (áudio) e F5 (vídeo) foram removidos do escopo → invalido
    assert detectar_estrutura(_touch(tmp_path / "a.wav")) == "invalido"
    assert detectar_estrutura(_touch(tmp_path / "a.mp4")) == "invalido"


def test_invalido_pasta_vazia(tmp_path):
    assert detectar_estrutura(tmp_path) == "invalido"


# ── io_utils ─────────────────────────────────────────────────────────────────

def test_label_pasta():
    assert io_utils.label_pasta("benign") == 0
    assert io_utils.label_pasta("MALIGNANT") == 1
    assert io_utils.label_pasta("qualquer") is None


def test_predicados_extensao():
    assert io_utils.eh_imagem(Path("x.png"))
    assert not io_utils.eh_imagem(Path("x_mask.png"))  # máscaras ignoradas
    assert io_utils.eh_tabular(Path("x.csv"))
    assert io_utils.eh_dicom(Path("x.dcm"))
    assert io_utils.eh_volumetrico(Path("x.nii.gz"))   # duplo sufixo
    assert io_utils.eh_sinal_temporal(Path("x.edf"))
