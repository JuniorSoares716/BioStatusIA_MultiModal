"""
Leitor F3 — Imagens DICOM 2D.
Detecta modalidade via SOPClassUID ou tag Modality.
Aplica WindowCenter/WindowWidth do header.
"""
from pathlib import Path

import numpy as np


# SOPClassUID → modalidade legível
_SOP_MAP = {
    "1.2.840.10008.5.1.4.1.1.1":   "Radiografia Computadorizada",
    "1.2.840.10008.5.1.4.1.1.1.1": "Radiografia Digital",
    "1.2.840.10008.5.1.4.1.1.1.2": "Mamografia Digital",
    "1.2.840.10008.5.1.4.1.1.2":   "Tomografia Computadorizada",
    "1.2.840.10008.5.1.4.1.1.4":   "Ressonância Magnética",
    "1.2.840.10008.5.1.4.1.1.6.1": "Ultrassonografia",
    "1.2.840.10008.5.1.4.1.1.20":  "Medicina Nuclear",
    "1.2.840.10008.5.1.4.1.1.128": "PET",
    "1.2.840.10008.5.1.4.1.1.481.2": "Radioterapia",
}


def ler_dicom(path: Path) -> "SinalNormalizado":
    from biostatusia.pipeline.io_sinais import SinalNormalizado

    try:
        import pydicom
    except ImportError:
        raise ImportError("Instale pydicom: uv add pydicom")

    ds = pydicom.dcmread(str(path))

    # Pixel data
    if not hasattr(ds, "PixelData"):
        raise ValueError(f"DICOM '{path.name}' não contém pixel data.")

    arr = ds.pixel_array.astype(np.float32)

    # Aplicar Rescale Slope/Intercept (Hounsfield para CT)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # Janelamento (WindowCenter / WindowWidth)
    arr_display = _aplicar_janelamento(arr, ds)

    # Normalizar para [0, 1]
    mn, mx = arr_display.min(), arr_display.max()
    if mx > mn:
        arr_norm = (arr_display - mn) / (mx - mn)
    else:
        arr_norm = arr_display - mn

    # Metadados clínicos do header DICOM
    sop = str(getattr(ds, "SOPClassUID", ""))
    modalidade = _SOP_MAP.get(sop, str(getattr(ds, "Modality", "DICOM")))
    pixel_spacing = list(getattr(ds, "PixelSpacing", [1.0, 1.0]))

    return SinalNormalizado(
        familia="F3",
        tipo=modalidade,
        dados=arr_norm.astype(np.float32),
        taxa_amostragem=0.0,
        canais=["grayscale"],
        metadados={
            "modalidade": modalidade,
            "sop_class_uid": sop,
            "pixel_spacing_mm": pixel_spacing,
            "shape": list(arr_norm.shape),
            "hounsfield_min": round(float(arr.min()), 1),
            "hounsfield_max": round(float(arr.max()), 1),
            "patient_id": str(getattr(ds, "PatientID", "")),
            "study_date": str(getattr(ds, "StudyDate", "")),
            "institution": str(getattr(ds, "InstitutionName", "")),
        },
        caminho_original=str(path),
    )


def _aplicar_janelamento(arr: np.ndarray, ds) -> np.ndarray:
    """Aplica window/level do header DICOM para exibição."""
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is None or ww is None:
        return arr
    # Pode ser uma lista (múltiplas janelas) — usa a primeira
    if hasattr(wc, "__iter__"):
        wc = float(list(wc)[0])
        ww = float(list(ww)[0])
    else:
        wc = float(wc)
        ww = float(ww)
    low  = wc - ww / 2
    high = wc + ww / 2
    return np.clip(arr, low, high)


def ler_serie_dicom(pasta: Path) -> "SinalNormalizado":
    """
    Lê uma pasta de DICOMs como volume 3D (serie axial).
    Ordena por InstanceNumber, empilha em (D, H, W).
    """
    from biostatusia.pipeline.io_sinais import SinalNormalizado

    try:
        import pydicom
    except ImportError:
        raise ImportError("Instale pydicom: uv add pydicom")

    arquivos = sorted(pasta.rglob("*.dcm"))
    if not arquivos:
        raise ValueError(f"Nenhum .dcm encontrado em '{pasta}'")

    slices = []
    for f in arquivos:
        try:
            ds = pydicom.dcmread(str(f))
            if hasattr(ds, "PixelData"):
                slices.append(ds)
        except Exception:
            continue

    # Ordenar por InstanceNumber
    slices.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))

    stack = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        stack.append(arr * slope + intercept)

    volume = np.stack(stack, axis=0)  # (D, H, W)
    mn, mx = volume.min(), volume.max()
    if mx > mn:
        volume = (volume - mn) / (mx - mn)

    first = slices[0]
    sop = str(getattr(first, "SOPClassUID", ""))
    modalidade = _SOP_MAP.get(sop, str(getattr(first, "Modality", "DICOM")))

    return SinalNormalizado(
        familia="F4",
        tipo=f"{modalidade} (série DICOM)",
        dados=volume.astype(np.float32),
        taxa_amostragem=0.0,
        canais=["axial"],
        metadados={
            "n_slices": len(slices),
            "shape": list(volume.shape),
            "modalidade": modalidade,
            "pasta": str(pasta),
        },
        caminho_original=str(pasta),
    )
