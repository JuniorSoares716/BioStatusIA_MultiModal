"""
Leitor F1 — Séries Temporais Fisiológicas.
Suporta: .edf, .bdf, .dat/.hea (PhysioNet), .mat, .xml (espirometria),
         .cnt, .eeg, .rec, .set (formatos EEG).
"""
from pathlib import Path

import numpy as np


def ler_sinal_temporal(path: Path) -> "SinalNormalizado":
    from biostatusia.pipeline.io_sinais import SinalNormalizado, _downsample_para_viz, _detectar_tipo_temporal

    ext = path.suffix.lower()
    tipo = _detectar_tipo_temporal(ext)

    # ── PhysioNet WFDB (.dat + .hea) ─────────────────────────────────────────
    if ext in {".dat", ".hea", ".atr"}:
        return _ler_wfdb(path, tipo)

    # ── EDF / BDF / REC / SET / CNT / EEG — via MNE ─────────────────────────
    if ext in {".edf", ".bdf", ".rec", ".set", ".cnt", ".eeg"}:
        return _ler_mne(path, tipo)

    # ── MATLAB .mat ───────────────────────────────────────────────────────────
    if ext == ".mat":
        return _ler_mat(path, tipo)

    # ── XML (Espirometria) ────────────────────────────────────────────────────
    if ext == ".xml":
        return _ler_xml_espirometria(path)

    # ── C3D (Movimento) ───────────────────────────────────────────────────────
    if ext == ".c3d":
        return _ler_c3d(path)

    raise ValueError(f"Extensão temporal não tratada: {ext}")


# ── Leitores individuais ──────────────────────────────────────────────────────

def _ler_wfdb(path: Path, tipo: str) -> "SinalNormalizado":
    from biostatusia.pipeline.io_sinais import SinalNormalizado, _downsample_para_viz
    try:
        import wfdb
    except ImportError:
        raise ImportError("Instale wfdb: uv add wfdb")

    # WFDB usa o nome base sem extensão
    record_name = str(path.with_suffix(""))
    try:
        record = wfdb.rdrecord(record_name)
    except Exception as e:
        raise ValueError(f"Erro ao ler registro WFDB '{record_name}': {e}")

    dados = record.p_signal.T.astype(np.float32)  # (n_canais, n_amostras)
    canais = record.sig_name if record.sig_name else [f"canal_{i}" for i in range(dados.shape[0])]
    fs = float(record.fs)

    return SinalNormalizado(
        familia="F1",
        tipo=tipo,
        dados=dados,
        taxa_amostragem=fs,
        canais=list(canais),
        metadados={
            "n_sinais": record.n_sig,
            "duracao_s": round(dados.shape[1] / fs, 2),
            "unidades": record.units if record.units else [],
        },
        caminho_original=str(path),
        dados_viz=_downsample_para_viz(dados),
    )


def _ler_mne(path: Path, tipo: str) -> "SinalNormalizado":
    from biostatusia.pipeline.io_sinais import SinalNormalizado, _downsample_para_viz
    try:
        import mne
        mne.set_log_level("WARNING")
    except ImportError:
        raise ImportError("Instale mne: uv add mne")

    try:
        raw = mne.io.read_raw(str(path), preload=True, verbose=False)
    except Exception as e:
        raise ValueError(f"MNE não conseguiu ler '{path.name}': {e}")

    dados = raw.get_data().astype(np.float32)  # (n_canais, n_amostras)
    fs = float(raw.info["sfreq"])
    canais = list(raw.ch_names)
    duracao = dados.shape[1] / fs

    # Inferir tipo de sinal pelos nomes dos canais
    nomes_lower = " ".join(canais).lower()
    if any(k in nomes_lower for k in ("eeg", "fp1", "fp2", "cz", "oz")):
        tipo_inf = "EEG"
    elif any(k in nomes_lower for k in ("ecg", "ekg", "i", "ii", "iii", "avr")):
        tipo_inf = "ECG"
    elif any(k in nomes_lower for k in ("emg", "muscle")):
        tipo_inf = "EMG"
    elif "eog" in nomes_lower:
        tipo_inf = "EOG"
    else:
        tipo_inf = tipo

    return SinalNormalizado(
        familia="F1",
        tipo=tipo_inf,
        dados=dados,
        taxa_amostragem=fs,
        canais=canais,
        metadados={
            "n_canais": len(canais),
            "duracao_s": round(duracao, 2),
            "n_amostras": dados.shape[1],
        },
        caminho_original=str(path),
        dados_viz=_downsample_para_viz(dados),
    )


def _ler_mat(path: Path, tipo: str) -> "SinalNormalizado":
    from biostatusia.pipeline.io_sinais import SinalNormalizado, _downsample_para_viz
    from scipy.io import loadmat

    mat = loadmat(str(path))
    # Heurística: maior array numérico 2D ou 1D
    candidatos = {
        k: v for k, v in mat.items()
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim >= 1
    }
    if not candidatos:
        raise ValueError(f"Nenhum array numérico encontrado em '{path.name}'")

    chave = max(candidatos, key=lambda k: candidatos[k].size)
    arr = candidatos[chave].astype(np.float32)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]  # (1, n)
    elif arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
        arr = arr.T               # garante (n_canais, n_amostras)

    # Tentar encontrar fs no .mat
    fs = 1000.0
    for k in ("fs", "Fs", "samplerate", "SampleRate", "sampling_rate"):
        if k in mat and isinstance(mat[k], np.ndarray):
            try:
                fs = float(mat[k].flat[0])
                break
            except Exception:
                pass

    canais = [f"canal_{i}" for i in range(arr.shape[0])]
    return SinalNormalizado(
        familia="F1",
        tipo=tipo,
        dados=arr,
        taxa_amostragem=fs,
        canais=canais,
        metadados={"chave_mat": chave, "n_amostras": arr.shape[1]},
        caminho_original=str(path),
        dados_viz=_downsample_para_viz(arr),
    )


def _ler_xml_espirometria(path: Path) -> "SinalNormalizado":
    """Lê arquivos XML de espirometria (formato genérico — extrai arrays numéricos)."""
    from biostatusia.pipeline.io_sinais import SinalNormalizado, _downsample_para_viz
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(path))
    root = tree.getroot()

    # Procura por elementos com texto numérico separado por vírgula/espaço
    valores: list[float] = []
    for elem in root.iter():
        texto = (elem.text or "").strip()
        if texto and "," in texto:
            try:
                valores = [float(x) for x in texto.split(",") if x.strip()]
                break
            except ValueError:
                continue

    if not valores:
        raise ValueError(f"Não foi possível extrair série numérica do XML '{path.name}'")

    arr = np.array(valores, dtype=np.float32)[np.newaxis, :]
    return SinalNormalizado(
        familia="F1",
        tipo="Espirometria",
        dados=arr,
        taxa_amostragem=10.0,  # taxa típica de espirômetros
        canais=["fluxo"],
        metadados={"n_amostras": arr.shape[1]},
        caminho_original=str(path),
        dados_viz=_downsample_para_viz(arr),
    )


def _ler_c3d(path: Path) -> "SinalNormalizado":
    """Lê .c3d (biomecânica/movimento) com fallback para numpy se ezc3d ausente."""
    from biostatusia.pipeline.io_sinais import SinalNormalizado, _downsample_para_viz
    try:
        import ezc3d
        c = ezc3d.c3d(str(path))
        dados = c["data"]["points"][0:3, :, :].astype(np.float32)  # (3, n_markers, n_frames)
        dados = dados.reshape(3 * dados.shape[1], dados.shape[2])
        fs = float(c["header"]["points"]["frame_rate"])
        n_markers = dados.shape[0] // 3
        canais = [f"M{i}_{ax}" for i in range(n_markers) for ax in ("X", "Y", "Z")]
        return SinalNormalizado(
            familia="F1", tipo="Movimento/C3D",
            dados=dados, taxa_amostragem=fs, canais=canais,
            metadados={"n_marcadores": n_markers},
            caminho_original=str(path),
            dados_viz=_downsample_para_viz(dados),
        )
    except ImportError:
        raise ImportError("Instale ezc3d para ler arquivos .c3d: uv add ezc3d")
    except Exception as e:
        raise ValueError(f"Erro ao ler C3D '{path.name}': {e}")
