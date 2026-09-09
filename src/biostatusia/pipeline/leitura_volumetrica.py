"""
Leitor F4 — Volumes 3D.
Suporta: .nii, .nii.gz (nibabel), .mha (SimpleITK).
"""
from pathlib import Path

import numpy as np


def ler_volume_3d(path: Path) -> "SinalNormalizado":
    from biostatusia.pipeline.io_sinais import SinalNormalizado

    ext = path.suffix.lower()
    suffixes = [s.lower() for s in path.suffixes]

    if ext in {".nii"} or suffixes == [".nii", ".gz"]:
        return _ler_nifti(path)
    if ext == ".mha":
        return _ler_mha(path)

    raise ValueError(f"Extensão volumétrica não suportada: {ext}")


def _ler_nifti(path: Path) -> "SinalNormalizado":
    from biostatusia.pipeline.io_sinais import SinalNormalizado
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("Instale nibabel: uv add nibabel")

    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)

    # Garantir 3D (pegar primeiro volume se 4D)
    if data.ndim == 4:
        data = data[..., 0]
    # nibabel: (X, Y, Z) — converter para (Z, Y, X) = axial-first
    data = np.transpose(data, (2, 1, 0))

    mn, mx = data.min(), data.max()
    if mx > mn:
        data = (data - mn) / (mx - mn)

    voxel_size = list(img.header.get_zooms()[:3])

    return SinalNormalizado(
        familia="F4",
        tipo="RM/TC NIfTI",
        dados=data.astype(np.float32),
        taxa_amostragem=0.0,
        canais=["volume"],
        metadados={
            "shape": list(data.shape),
            "voxel_size_mm": [round(v, 3) for v in voxel_size],
            "n_slices": data.shape[0],
            "orientacao": str(nib.aff2axcodes(img.affine)),
        },
        caminho_original=str(path),
    )


def _ler_mha(path: Path) -> "SinalNormalizado":
    from biostatusia.pipeline.io_sinais import SinalNormalizado
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError("Instale SimpleITK: uv add SimpleITK")

    img = sitk.ReadImage(str(path))
    data = sitk.GetArrayFromImage(img).astype(np.float32)  # já (Z, Y, X)

    mn, mx = data.min(), data.max()
    if mx > mn:
        data = (data - mn) / (mx - mn)

    spacing = list(img.GetSpacing())

    return SinalNormalizado(
        familia="F4",
        tipo="TC/MHA",
        dados=data,
        taxa_amostragem=0.0,
        canais=["volume"],
        metadados={
            "shape": list(data.shape),
            "voxel_size_mm": [round(v, 3) for v in spacing],
            "n_slices": data.shape[0],
        },
        caminho_original=str(path),
    )


def slice_para_png_base64(volume: np.ndarray, idx: int) -> str:
    """Retorna um slice axial como string base64 PNG para a UI."""
    import base64
    import io

    from PIL import Image

    idx = max(0, min(idx, volume.shape[0] - 1))
    slc = (volume[idx] * 255).astype(np.uint8)
    img = Image.fromarray(slc, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
