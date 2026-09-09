"""
Extrator F1 — Biomarcadores de Séries Temporais Fisiológicas.
Domínios: tempo, frequência e morfológico (por tipo de sinal).
"""
import numpy as np
from scipy import signal as sig
from scipy.stats import skew, kurtosis


# ── Extração principal ────────────────────────────────────────────────────────

def extrair_biomarcadores_temporal(sinal) -> dict | None:
    """
    Recebe SinalNormalizado (F1) e retorna dict de biomarcadores.
    Aplica análise por tipo: ECG → HRV, EEG → bandas, outros → genérico.
    """
    try:
        dados = sinal.dados          # (n_canais, n_amostras)
        fs = sinal.taxa_amostragem
        tipo = sinal.tipo.upper()

        bio: dict = {
            "tempo": _features_tempo(dados),
            "frequencia": _features_frequencia(dados, fs),
        }

        if "ECG" in tipo or "EKG" in tipo:
            bio["morfologico_ecg"] = _features_ecg(dados, fs)
        elif "EEG" in tipo:
            bio["bandas_eeg"] = _features_eeg(dados, fs)
        elif "EMG" in tipo:
            bio["morfologico_emg"] = _features_emg(dados, fs)
        elif "ESPIRO" in tipo:
            bio["espirometria"] = _features_espirometria(dados, fs)

        return bio
    except Exception as e:
        return {"erro": str(e)}


# ── Domínio Tempo ─────────────────────────────────────────────────────────────

def _features_tempo(dados: np.ndarray) -> dict:
    """Estatísticas temporais sobre todos os canais (média entre canais)."""
    flat = dados.flatten()
    rms = float(np.sqrt(np.mean(flat ** 2)))
    return {
        "rms": round(rms, 6),
        "media": round(float(np.mean(flat)), 6),
        "desvio_padrao": round(float(np.std(flat)), 6),
        "assimetria": round(float(skew(flat)), 4),
        "curtose": round(float(kurtosis(flat)), 4),
        "amplitude_pico_pico": round(float(flat.max() - flat.min()), 6),
        "snr_db": round(float(20 * np.log10(rms / (np.std(flat) + 1e-12) + 1e-12)), 4),
    }


# ── Domínio Frequência ────────────────────────────────────────────────────────

def _features_frequencia(dados: np.ndarray, fs: float) -> dict:
    """PSD via Welch, potência por banda, frequência dominante."""
    canal_ref = dados[0]  # analisa o primeiro canal como referência
    freqs, psd = sig.welch(canal_ref, fs=fs, nperseg=min(256, len(canal_ref) // 4))
    potencia_total = float(np.trapz(psd, freqs)) + 1e-12

    def _banda(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return round(float(np.trapz(psd[mask], freqs[mask])) / potencia_total, 6) if mask.any() else 0.0

    freq_dom = float(freqs[np.argmax(psd)])

    return {
        "frequencia_dominante_hz": round(freq_dom, 3),
        "potencia_total": round(potencia_total, 6),
        "banda_0_5hz":   _banda(0, 5),
        "banda_5_15hz":  _banda(5, 15),
        "banda_15_50hz": _banda(15, 50),
        "banda_50_150hz":_banda(50, 150),
        "centroide_espectral_hz": round(
            float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12)), 3
        ),
    }


# ── ECG — HRV ────────────────────────────────────────────────────────────────

def _features_ecg(dados: np.ndarray, fs: float) -> dict:
    """Detecta picos R e extrai métricas HRV básicas."""
    # Usa o canal com maior variância (mais provável de ser o sinal ECG dominante)
    variancias = np.var(dados, axis=1)
    canal = dados[int(np.argmax(variancias))]

    # Filtro passa-banda 0.5–40 Hz
    if fs > 80:
        sos = sig.butter(4, [0.5 / (fs / 2), 40 / (fs / 2)], btype="band", output="sos")
        canal = sig.sosfiltfilt(sos, canal)

    # Detectar picos R: distância mínima de 0.4s
    dist_min = max(1, int(0.4 * fs))
    picos, _ = sig.find_peaks(canal, distance=dist_min, height=np.std(canal) * 0.5)

    if len(picos) < 3:
        return {"erro": "Poucos picos R detectados (< 3)"}

    rr = np.diff(picos) / fs * 1000  # em ms
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    sdnn = float(np.std(rr))
    pnn50 = float(np.mean(np.abs(np.diff(rr)) > 50) * 100)
    hr_bpm = float(60000 / np.mean(rr))

    return {
        "n_picos_r": len(picos),
        "fc_media_bpm": round(hr_bpm, 1),
        "rr_media_ms": round(float(np.mean(rr)), 2),
        "rr_desvio_ms": round(float(np.std(rr)), 2),
        "hrv_rmssd_ms": round(rmssd, 2),
        "hrv_sdnn_ms": round(sdnn, 2),
        "hrv_pnn50_pct": round(pnn50, 2),
    }


# ── EEG — Bandas de Frequência ────────────────────────────────────────────────

def _features_eeg(dados: np.ndarray, fs: float) -> dict:
    """Potência relativa em cada banda EEG (média entre canais)."""
    bandas = {
        "delta_0_4hz":  (0.5, 4),
        "theta_4_8hz":  (4, 8),
        "alfa_8_13hz":  (8, 13),
        "beta_13_30hz": (13, 30),
        "gama_30_100hz":(30, min(100, fs / 2 - 1)),
    }

    potencias: dict = {}
    total = 0.0
    for nome, (f1, f2) in bandas.items():
        pot = 0.0
        for canal in dados:
            freqs, psd = sig.welch(canal, fs=fs, nperseg=min(256, len(canal) // 4))
            mask = (freqs >= f1) & (freqs < f2)
            if mask.any():
                pot += float(np.trapz(psd[mask], freqs[mask]))
        pot /= dados.shape[0]
        potencias[nome] = pot
        total += pot

    potencias_rel = {k: round(v / (total + 1e-12), 6) for k, v in potencias.items()}
    # Índice alfa/beta (proxy de relaxamento vs atividade)
    alfa = potencias_rel.get("alfa_8_13hz", 0)
    beta = potencias_rel.get("beta_13_30hz", 0)
    potencias_rel["ratio_alfa_beta"] = round(alfa / (beta + 1e-12), 4)
    return potencias_rel


# ── EMG ───────────────────────────────────────────────────────────────────────

def _features_emg(dados: np.ndarray, fs: float) -> dict:
    """RMS envelope e frequência mediana de EMG."""
    canal = dados[0]

    # Filtro passa-alta 20 Hz para remover artefatos de movimento
    if fs > 40:
        sos = sig.butter(4, 20 / (fs / 2), btype="high", output="sos")
        canal = sig.sosfiltfilt(sos, canal)

    rms = float(np.sqrt(np.mean(canal ** 2)))
    freqs, psd = sig.welch(canal, fs=fs, nperseg=min(256, len(canal) // 4))
    cumsum = np.cumsum(psd)
    freq_mediana_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    freq_mediana = float(freqs[min(freq_mediana_idx, len(freqs) - 1)])

    return {
        "rms_emg": round(rms, 6),
        "frequencia_mediana_hz": round(freq_mediana, 2),
        "amplitude_max": round(float(np.max(np.abs(canal))), 6),
    }


# ── Espirometria ──────────────────────────────────────────────────────────────

def _features_espirometria(dados: np.ndarray, fs: float) -> dict:
    """FVC, FEV1 aproximados a partir de curva fluxo-volume."""
    canal = dados[0]
    volume_acum = np.cumsum(canal) / fs  # integral do fluxo = volume (L)

    fvc = float(np.max(volume_acum) - np.min(volume_acum))
    # FEV1: volume no primeiro segundo
    n1s = int(fs)
    fev1 = float(volume_acum[min(n1s, len(volume_acum) - 1)] - volume_acum[0])
    ratio = fev1 / (fvc + 1e-8)
    pef = float(np.max(np.abs(canal)))

    return {
        "fvc_litros": round(fvc, 3),
        "fev1_litros": round(fev1, 3),
        "fev1_fvc_ratio": round(ratio, 4),
        "pef_l_s": round(pef, 3),
    }


# ── Função de lote ────────────────────────────────────────────────────────────

def extrair_lote_temporal(arquivos: list[dict]) -> list[dict]:
    """
    arquivos: [{"caminho": ..., "label": 0|1|None, "categoria": ...}, ...]
    Retorna lista de registros com biomarcadores.
    """
    from biostatusia.pipeline.io_sinais import carregar_sinal

    resultados = []
    for arq in arquivos:
        try:
            sinal = carregar_sinal(arq["caminho"])
            bio = extrair_biomarcadores_temporal(sinal)
            if bio and "erro" not in bio:
                resultados.append({
                    "caminho": arq["caminho"],
                    "label": arq.get("label"),
                    "categoria": arq.get("categoria", "INDEFINIDO"),
                    "biomarcadores": bio,
                    "tipo": sinal.tipo,
                    "taxa_amostragem": sinal.taxa_amostragem,
                })
        except Exception as e:
            resultados.append({
                "caminho": arq["caminho"],
                "erro": str(e),
                "categoria": arq.get("categoria", "INDEFINIDO"),
            })
    return resultados
