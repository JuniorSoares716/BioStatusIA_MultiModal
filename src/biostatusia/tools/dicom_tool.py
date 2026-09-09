"""Tool CrewAI — F3: Extração de Biomarcadores de Imagens DICOM 2D."""
import json
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class DicomInput(BaseModel):
    caminho_dataset: str = Field(..., description="Caminho do arquivo .dcm ou pasta com DICOMs 2D.")
    pasta_run: str = Field(..., description="Pasta de workspace para persistir JSONs.")


class FerramentaExtrairDICOM(BaseTool):
    name: str = "ferramenta_extrair_dicom"
    description: str = (
        "Carrega imagens DICOM 2D (Raio-X, Mamografia, Ultrassom) e extrai "
        "biomarcadores radiômicos: morfologia, textura GLCM, distribuição de "
        "intensidade e features específicas por modalidade (densidade mamária, "
        "gradiente de borda, janelamento Hounsfield). Persiste resultado em JSON."
    )
    args_schema: type[BaseModel] = DicomInput

    def _run(self, caminho_dataset: str, pasta_run: str) -> str:
        from biostatusia.pipeline.io_utils import listar_sinais, eh_dicom
        from biostatusia.pipeline.extracao_dicom import extrair_lote_dicom

        base = Path(caminho_dataset.strip("'\""))
        pasta = Path(pasta_run.strip("'\""))
        pasta.mkdir(parents=True, exist_ok=True)

        arquivos = listar_sinais(base, eh_dicom)
        if not arquivos and base.is_file() and eh_dicom(base):
            arquivos = [{"caminho": str(base), "label": None, "categoria": "INDEFINIDO"}]
        if not arquivos:
            return "Erro: nenhum arquivo DICOM 2D encontrado no caminho fornecido."

        resultados = extrair_lote_dicom(arquivos)
        ok = [r for r in resultados if "biomarcadores" in r]
        erros = [r for r in resultados if "erro" in r]

        metricas_clf: dict = {}
        rotulados = [r for r in ok if r.get("label") is not None]
        if len(rotulados) >= 10:
            import numpy as np
            from biostatusia.tools.sinais_temporais_tool import _vetorizar
            from biostatusia.pipeline.avaliacao_modelos import avaliar_modelos
            X = _vetorizar(rotulados)
            y = np.array([r["label"] for r in rotulados])
            metricas_clf = avaliar_modelos(X, y, familia="F3")

        payload = {
            "familia": "F3",
            "tipo": ok[0].get("tipo", "DICOM 2D") if ok else "DICOM 2D",
            "n_arquivos": len(arquivos),
            "n_processados": len(ok),
            "n_erros": len(erros),
            "biomarcadores": ok,
        }
        if metricas_clf:
            payload.update({
                "metricas": metricas_clf.get("metricas", {}),
                "roc_data": metricas_clf.get("roc_data", {}),
                "confusion_matrix": metricas_clf.get("confusion_matrix", {}),
                "melhor_modelo": metricas_clf.get("melhor_modelo", "N/A"),
                "comparacao_ab": metricas_clf.get("comparacao_ab", {}),
            })

        with open(pasta / "biomarcadores_dicom.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        resumo = (
            f"DICOMs 2D processados: {len(ok)}/{len(arquivos)}.\n"
            f"Modalidade: {payload['tipo']}\nErros: {len(erros)}\n"
        )
        if ok:
            bio0 = ok[0]["biomarcadores"]
            morf = bio0.get("morfologia", {})
            glcm = bio0.get("textura_glcm", {})
            rad = bio0.get("radiomico_dicom", {})
            resumo += (
                f"Exemplo (1º DICOM):\n"
                f"  - Circularidade: {morf.get('circularidade', 'N/A')}\n"
                f"  - Solidez: {morf.get('solidez', 'N/A')}\n"
                f"  - Entropia GLCM: {glcm.get('entropia', 'N/A')}\n"
                f"  - Homogeneidade: {glcm.get('homogeneidade', 'N/A')}\n"
                f"  - Densidade alta (%): {rad.get('densidade_alta_pct', 'N/A')}\n"
                f"  - Gradiente médio: {rad.get('gradiente_medio', 'N/A')}\n"
            )
        if metricas_clf.get("melhor_modelo"):
            m = metricas_clf["metricas"].get(metricas_clf["melhor_modelo"], {})
            resumo += (
                f"\nMelhor modelo: {metricas_clf['melhor_modelo']}\n"
                f"  - AUC: {m.get('auc', 'N/A')} | Sensib.: {m.get('sensibilidade', 'N/A')}\n"
            )
        resumo += "\nResultado persistido em biomarcadores_dicom.json."
        return resumo
