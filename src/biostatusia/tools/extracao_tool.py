import json
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from biostatusia.pipeline.extracao import extrair_todos
from biostatusia.pipeline.io_utils import listar_imagens


class ExtracaoLoteInput(BaseModel):
    caminho_dataset: str = Field(..., description="Caminho da pasta ou imagem.")
    pasta_run: str = Field(..., description="Pasta de workspace com analise_base.json.")


class FerramentaExtrairBiomarcadores(BaseTool):
    name: str = "ferramenta_extrair_biomarcadores"
    description: str = (
        "Aplica a estratégia adaptativa (de analise_base.json) e extrai 9 biomarcadores "
        "radiômicos (morfologia + textura GLCM + distribuição) de TODAS as imagens "
        "do dataset. Persiste o resultado em biomarcadores.json e retorna um resumo."
    )
    args_schema: type[BaseModel] = ExtracaoLoteInput

    def _run(self, caminho_dataset: str, pasta_run: str) -> str:
        base = Path(caminho_dataset.strip("'\""))
        pasta = Path(pasta_run.strip("'\""))

        analise_path = pasta / "analise_base.json"
        if not analise_path.exists():
            return f"Erro: analise_base.json não encontrado em {pasta}. Rode a análise de base antes."

        with open(analise_path, "r", encoding="utf-8") as f:
            dados_base = json.load(f)
        estrategia = dados_base.get("estrategia") or {}

        imagens = listar_imagens(base)
        if not imagens:
            return "Erro: nenhuma imagem encontrada."

        registros: list[dict] = []
        for item in imagens:
            bio = extrair_todos(item["caminho"], estrategia=estrategia)
            if bio is not None:
                registros.append({
                    "caminho": item["caminho"],
                    "label": item["label"],
                    "categoria": item["categoria"],
                    "biomarcadores": bio,
                })

        if not registros:
            return "Erro: falha ao extrair biomarcadores de qualquer imagem."

        with open(pasta / "biomarcadores.json", "w", encoding="utf-8") as f:
            json.dump(registros, f, indent=2)

        n_b = sum(1 for r in registros if r["categoria"] == "BENIGNO")
        n_m = sum(1 for r in registros if r["categoria"] == "MALIGNO")
        n_i = sum(1 for r in registros if r["categoria"] == "INDEFINIDO")

        primeiro = registros[0]["biomarcadores"]
        return (
            f"Extração concluída em {len(registros)} imagens "
            f"(BENIGNO={n_b}, MALIGNO={n_m}, INDEFINIDO={n_i}).\n"
            f"Estratégia aplicada: denoising={estrategia.get('denoising')}, "
            f"normalização={estrategia.get('normalizacao')}, "
            f"equalização={estrategia.get('equalizacao')}.\n\n"
            f"Exemplo (primeira imagem):\n"
            f"- Morfologia: solidez={primeiro['morfologia']['solidez']}, "
            f"circularidade={primeiro['morfologia']['circularidade']}\n"
            f"- Textura: entropia={primeiro['textura_glcm']['entropia']}, "
            f"homogeneidade={primeiro['textura_glcm']['homogeneidade']}\n"
            f"- Intensidade: SNR={primeiro['distribuicao_intensidade']['snr']}\n\n"
            f"Dataset completo persistido em biomarcadores.json."
        )
