"""Tier 1 — contrato SinalNormalizado por família (io_sinais.carregar_sinal)."""
from biostatusia.pipeline.io_sinais import carregar_sinal, SinalNormalizado


def test_f1_temporal(tmp_path, gerar_ecg):
    p = tmp_path / "e.mat"
    gerar_ecg(p, False)
    s = carregar_sinal(str(p))
    assert isinstance(s, SinalNormalizado)
    assert s.dados.ndim == 2                 # (canais, amostras)
    assert s.taxa_amostragem > 0
    assert len(s.dados_viz) <= 2000 or all(len(c) <= 2000 for c in s.dados_viz)


def test_f3_dicom(tmp_path, gerar_dicom):
    p = tmp_path / "d.dcm"
    gerar_dicom(p, True)
    s = carregar_sinal(str(p))
    assert s.familia == "F3"
    assert s.dados.ndim == 2                 # (H, W)


def test_f4_volume(tmp_path, gerar_nifti):
    p = tmp_path / "v.nii"
    gerar_nifti(p, True)
    s = carregar_sinal(str(p))
    assert s.familia == "F4"
    assert s.dados.ndim == 3                 # (D, H, W)


def test_extensao_fora_de_escopo_levanta(tmp_path):
    import pytest
    p = tmp_path / "x.xyz"
    p.write_bytes(b"")
    with pytest.raises(ValueError):
        carregar_sinal(str(p))
