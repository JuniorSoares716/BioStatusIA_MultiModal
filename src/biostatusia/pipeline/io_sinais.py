"""
Dispatcher central para todas as famílias de sinal biomédico.
Retorna um SinalNormalizado ou levanta ValueError com mensagem clara.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SinalNormalizado:
    familia: str              # "F1" | "F3" | "F4"
    tipo: str                 # "ECG" | "EEG" | "RX" | ...
    dados: np.ndarray         # shape varia por família (veja abaixo)
    taxa_amostragem: float    # Hz  (0.0 para imagens/volumes)
    canais: list[str]         # nomes dos canais
    metadados: dict = field(default_factory=dict)
    caminho_original: str = ""
    dados_viz: list = field(default_factory=list)  # versão downsampled para UI (≤2000 pts/canal)

    # Shapes esperados por família:
    #   F1 (temporal): (n_canais, n_amostras)
    #   F3 (imagem 2D):(H, W)                — float32 [0,1]
    #   F4 (volume 3D):(D, H, W)             — float32 [0,1]


def _detectar_tipo_temporal(ext: str) -> str:
    """Infere o tipo de sinal a partir da extensão."""
    mapa = {
        ".edf": "EEG/ECG/EDF", ".bdf": "EEG/BDF",
        ".dat": "ECG/PhysioNet", ".hea": "ECG/PhysioNet",
        ".atr": "ECG/Annotations", ".mat": "Sinal/MATLAB",
        ".cnt": "EEG/Neuroscan", ".eeg": "EEG",
        ".rec": "Polissonografia", ".c3d": "Movimento/C3D",
        ".set": "EEG/EEGLAB", ".xml": "Espirometria/XML",
    }
    return mapa.get(ext.lower(), "Sinal Temporal")


def carregar_sinal(caminho: str | Path) -> "SinalNormalizado":
    """
    Ponto de entrada único. Detecta família, chama o leitor certo,
    retorna SinalNormalizado.
    """
    path = Path(caminho)
    ext = path.suffix.lower()
    suffixes = [s.lower() for s in path.suffixes]

    # F1 — Séries Temporais Fisiológicas
    if ext in {".edf", ".bdf", ".dat", ".hea", ".atr", ".mat",
               ".cnt", ".eeg", ".rec", ".c3d", ".set", ".xml"}:
        from biostatusia.pipeline.leitura_temporal import ler_sinal_temporal
        return ler_sinal_temporal(path)

    # F3 — DICOM 2D
    if ext == ".dcm":
        from biostatusia.pipeline.leitura_dicom import ler_dicom
        return ler_dicom(path)

    # F4 — Volume 3D
    if ext in {".nii", ".mha"} or suffixes == [".nii", ".gz"]:
        from biostatusia.pipeline.leitura_volumetrica import ler_volume_3d
        return ler_volume_3d(path)

    raise ValueError(f"Extensão não suportada (escopo F1/F3/F4): {ext} ({path.name})")


def _downsample_para_viz(dados: np.ndarray, max_pts: int = 2000) -> list:
    """Reduz o número de amostras para visualização no frontend."""
    if dados.ndim == 1:
        n = len(dados)
        if n <= max_pts:
            return dados.tolist()
        step = n // max_pts
        return dados[::step][:max_pts].tolist()

    if dados.ndim == 2:
        n = dados.shape[1]
        if n <= max_pts:
            return dados.tolist()
        step = max(1, n // max_pts)
        return dados[:, ::step][:, :max_pts].tolist()

    return []
