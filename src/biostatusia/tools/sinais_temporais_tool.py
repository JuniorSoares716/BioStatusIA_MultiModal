"""Tool CrewAI — F1: Extração de Biomarcadores de Sinais Temporais."""
import json
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SinalTemporalInput(BaseModel):
    caminho_dataset: str = Field(..., description="Caminho do arquivo ou pasta com sinais temporais.")
    pasta_run: str = Field(..., description="Pasta de workspace para persistir JSONs.")
    tipo_sinal: str = Field(default="auto", description="Tipo de sinal: ECG, EEG, EMG, EOG, PPG, etc. (auto = inferir).")


class FerramentaExtrairSinalTemporal(BaseTool):
    name: str = "ferramenta_extrair_sinal_temporal"
    description: str = (
        "Carrega sinais fisiológicos temporais (ECG, EEG, EMG, EOG, PPG, PA, "
        "Espirometria, Polissonografia, Movimento) nos formatos .edf, .bdf, "
        ".dat/.hea (PhysioNet), .mat, .xml e extrai biomarcadores multi-domínio "
        "(tempo, frequência, morfológico). Persiste resultado em JSON."
    )
    args_schema: type[BaseModel] = SinalTemporalInput

    def _run(self, caminho_dataset: str, pasta_run: str, tipo_sinal: str = "auto") -> str:
        from biostatusia.pipeline.io_utils import listar_sinais, eh_sinal_temporal
        from biostatusia.pipeline.extracao_temporal import extrair_lote_temporal
        from biostatusia.pipeline.avaliacao_modelos import avaliar_modelos
        import numpy as np

        base = Path(caminho_dataset.strip("'\""))
        pasta = Path(pasta_run.strip("'\""))
        pasta.mkdir(parents=True, exist_ok=True)

        arquivos = listar_sinais(base, eh_sinal_temporal)
        if not arquivos:
            # Tentar como arquivo único
            if base.is_file() and eh_sinal_temporal(base):
                arquivos = [{"caminho": str(base), "label": None, "categoria": "INDEFINIDO"}]
        if not arquivos:
            return "Erro: nenhum sinal temporal encontrado no caminho fornecido."

        resultados = extrair_lote_temporal(arquivos)
        ok = [r for r in resultados if "biomarcadores" in r]
        erros = [r for r in resultados if "erro" in r]

        # Treino se rotulado
        metricas_clf: dict = {}
        rotulados = [r for r in ok if r.get("label") is not None]
        if len(rotulados) >= 10:
            y = np.array([r["label"] for r in rotulados])
            X_rot = _vetorizar(rotulados)
            familia = rotulados[0].get("tipo", "F1") if rotulados else "F1"
            metricas_clf = avaliar_modelos(X_rot, y, familia=familia)

        payload = {
            "familia": "F1",
            "tipo": ok[0].get("tipo", "Sinal Temporal") if ok else "Sinal Temporal",
            "n_arquivos": len(arquivos),
            "n_processados": len(ok),
            "n_erros": len(erros),
            "biomarcadores": ok,
        }
        if metricas_clf:
            payload["metricas"] = metricas_clf.get("metricas", {})
            payload["metricas_cv"] = metricas_clf.get("metricas_cv", {})
            payload["roc_data"] = metricas_clf.get("roc_data", {})
            payload["confusion_matrix"] = metricas_clf.get("confusion_matrix", {})
            payload["melhor_modelo"] = metricas_clf.get("melhor_modelo", "N/A")
            payload["comparacao_ab"] = metricas_clf.get("comparacao_ab", {})

        with open(pasta / "biomarcadores_temporal.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        tipo_inf = ok[0].get("tipo", "sinal temporal") if ok else "sinal temporal"
        resumo = (
            f"Sinais temporais processados: {len(ok)}/{len(arquivos)} arquivos.\n"
            f"Tipo detectado: {tipo_inf}\n"
            f"Erros: {len(erros)}\n"
        )
        if ok:
            bio0 = ok[0]["biomarcadores"]
            tempo = bio0.get("tempo", {})
            freq = bio0.get("frequencia", {})
            resumo += (
                f"Exemplo (1º arquivo):\n"
                f"  - RMS: {tempo.get('rms', 'N/A')}\n"
                f"  - Freq. dominante: {freq.get('frequencia_dominante_hz', 'N/A')} Hz\n"
                f"  - Desvio: {tempo.get('desvio_padrao', 'N/A')}\n"
            )
            if "morfologico_ecg" in bio0:
                ecg = bio0["morfologico_ecg"]
                resumo += (
                    f"  - FC média: {ecg.get('fc_media_bpm', 'N/A')} bpm\n"
                    f"  - HRV RMSSD: {ecg.get('hrv_rmssd_ms', 'N/A')} ms\n"
                )
            elif "bandas_eeg" in bio0:
                eeg = bio0["bandas_eeg"]
                resumo += f"  - Alfa/Beta ratio: {eeg.get('ratio_alfa_beta', 'N/A')}\n"
        if metricas_clf and "melhor_modelo" in metricas_clf:
            m = metricas_clf["metricas"].get(metricas_clf["melhor_modelo"], {})
            resumo += (
                f"\nClassificador (melhor: {metricas_clf['melhor_modelo']}):\n"
                f"  - AUC: {m.get('auc', 'N/A')}\n"
                f"  - Sensibilidade: {m.get('sensibilidade', 'N/A')}\n"
                f"  - Especificidade: {m.get('especificidade', 'N/A')}\n"
            )
        resumo += "\nResultado completo persistido em biomarcadores_temporal.json."
        return resumo


def _vetorizar(registros: list[dict]) -> "np.ndarray":
    """Achata os biomarcadores em um vetor numérico simples."""
    import numpy as np
    vetores = []
    for r in registros:
        bio = r.get("biomarcadores", {})
        v = []
        for grupo in bio.values():
            if isinstance(grupo, dict):
                for val in grupo.values():
                    if isinstance(val, (int, float)):
                        v.append(float(val))
                    elif isinstance(val, list) and all(isinstance(x, (int, float)) for x in val):
                        v.extend(float(x) for x in val[:5])  # limita para evitar dimensionalidade explodindo
        if v:
            vetores.append(v)
    if not vetores:
        return np.empty((0, 1))
    max_len = max(len(v) for v in vetores)
    padded = [v + [0.0] * (max_len - len(v)) for v in vetores]
    return np.array(padded, dtype=np.float32)
