"""Tool CrewAI — F4: Extração de Biomarcadores de Volumes 3D."""
import json
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class VolumetricoInput(BaseModel):
    caminho_dataset: str = Field(..., description="Caminho do arquivo .nii, .nii.gz, .mha ou pasta DICOM.")
    pasta_run: str = Field(..., description="Pasta de workspace para persistir JSONs.")


class FerramentaExtrairVolume3D(BaseTool):
    name: str = "ferramenta_extrair_volume_3d"
    description: str = (
        "Carrega volumes 3D (TC, RM, PET/SPECT) nos formatos .nii, .nii.gz, .mha "
        "e extrai biomarcadores volumétricos: estatísticas globais de intensidade, "
        "textura GLCM por plano ortogonal (axial/coronal/sagital) e morfologia 3D "
        "(volume de lesão candidata, sphericity, bounding box). Persiste em JSON."
    )
    args_schema: type[BaseModel] = VolumetricoInput

    def _run(self, caminho_dataset: str, pasta_run: str) -> str:
        from biostatusia.pipeline.io_utils import listar_sinais, eh_volumetrico

        base = Path(caminho_dataset.strip("'\""))
        pasta = Path(pasta_run.strip("'\""))
        pasta.mkdir(parents=True, exist_ok=True)

        arquivos = listar_sinais(base, eh_volumetrico)
        if not arquivos and base.is_file() and eh_volumetrico(base):
            arquivos = [{"caminho": str(base), "label": None, "categoria": "INDEFINIDO"}]

        # Fallback: pasta de DICOMs → série volumétrica
        if not arquivos and base.is_dir():
            dcm_files = list(base.rglob("*.dcm"))
            if len(dcm_files) >= 3:
                arquivos = [{"caminho": str(base), "label": None, "categoria": "INDEFINIDO", "_is_serie": True}]

        if not arquivos:
            return "Erro: nenhum volume 3D encontrado (suporte: .nii, .nii.gz, .mha, pasta DICOM)."

        resultados = []
        for arq in arquivos:
            try:
                if arq.get("_is_serie"):
                    from biostatusia.pipeline.leitura_dicom import ler_serie_dicom
                    sinal = ler_serie_dicom(Path(arq["caminho"]))
                else:
                    from biostatusia.pipeline.io_sinais import carregar_sinal
                    sinal = carregar_sinal(arq["caminho"])
                from biostatusia.pipeline.extracao_volumetrica import extrair_biomarcadores_volumetrico
                bio = extrair_biomarcadores_volumetrico(sinal)
                if bio and "erro" not in bio:
                    resultados.append({
                        "caminho": arq["caminho"],
                        "label": arq.get("label"),
                        "categoria": arq.get("categoria", "INDEFINIDO"),
                        "biomarcadores": bio,
                        "tipo": sinal.tipo,
                    })
            except Exception as e:
                resultados.append({"caminho": arq["caminho"], "erro": str(e)})

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
            metricas_clf = avaliar_modelos(X, y, familia="F4")

        payload = {
            "familia": "F4",
            "tipo": ok[0].get("tipo", "Volume 3D") if ok else "Volume 3D",
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
            })

        with open(pasta / "biomarcadores_volumetrico.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        resumo = (
            f"Volumes 3D processados: {len(ok)}/{len(arquivos)}.\n"
            f"Tipo: {payload['tipo']}\nErros: {len(erros)}\n"
        )
        if ok:
            bio0 = ok[0]["biomarcadores"]
            vol = bio0.get("volumetrico_global", {})
            morf = bio0.get("morfologia_3d", {})
            resumo += (
                f"Exemplo (1º volume):\n"
                f"  - Intensidade média: {vol.get('media_intensidade', 'N/A')}\n"
                f"  - Fração voxels altos: {vol.get('fracao_voxels_altos', 'N/A')}\n"
                f"  - Volume lesão candidata (mm³): {morf.get('volume_lesao_mm3', 'N/A')}\n"
                f"  - Sphericity aprox.: {morf.get('sphericity_aprox', 'N/A')}\n"
                f"  - N° slices: {morf.get('n_slices_total', 'N/A')}\n"
            )
        resumo += "\nResultado persistido em biomarcadores_volumetrico.json."
        return resumo
