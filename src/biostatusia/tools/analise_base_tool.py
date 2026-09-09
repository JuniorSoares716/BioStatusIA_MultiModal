import json
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from biostatusia.pipeline.analise_base import analisar_base, decidir_estrategia
from biostatusia.pipeline.io_utils import listar_imagens


class AnaliseBaseInput(BaseModel):
    caminho_dataset: str = Field(..., description="Caminho da pasta ou imagem a analisar.")
    pasta_run: str = Field(..., description="Pasta de workspace onde persistir o resultado JSON.")


class FerramentaAnaliseBase(BaseTool):
    name: str = "ferramenta_analise_base"
    description: str = (
        "Analisa estatisticamente uma base de imagens (intensidade, outliers IQR, "
        "normalidade Shapiro-Wilk, ruído, contraste, tamanhos) e decide a estratégia "
        "de pré-processamento (denoising, normalização, equalização, redimensionamento). "
        "Persiste o resultado completo em JSON e retorna um resumo textual."
    )
    args_schema: type[BaseModel] = AnaliseBaseInput

    def _run(self, caminho_dataset: str, pasta_run: str) -> str:
        base = Path(caminho_dataset.strip("'\""))
        imagens = listar_imagens(base)
        if not imagens:
            return "Erro: nenhuma imagem encontrada no caminho fornecido."

        caminhos = [item["caminho"] for item in imagens]
        analise = analisar_base(caminhos)
        estrategia = decidir_estrategia(analise)

        pasta = Path(pasta_run.strip("'\""))
        pasta.mkdir(parents=True, exist_ok=True)
        with open(pasta / "analise_base.json", "w", encoding="utf-8") as f:
            json.dump({"analise": analise, "estrategia": estrategia}, f, indent=2)

        normalidade = analise.get("teste_normalidade", {})
        return (
            f"Análise concluída em {analise.get('n_imagens_analisadas', 0)} imagens.\n"
            f"- Intensidade média: {analise.get('intensidade_media')} "
            f"(desvio {analise.get('intensidade_desvio')})\n"
            f"- Outliers (IQR): {analise.get('outliers_intensidade')}\n"
            f"- Normalidade (Shapiro p): {normalidade.get('shapiro_p')} "
            f"(normal={normalidade.get('eh_normal')})\n"
            f"- Contraste médio: {analise.get('contraste_medio')}\n"
            f"- Ruído estimado: {analise.get('ruido_medio')}\n"
            f"- Tamanhos consistentes: {analise.get('tamanhos_consistentes')}\n\n"
            f"Estratégia escolhida:\n"
            f"- Denoising: {estrategia['denoising']}\n"
            f"- Normalização: {estrategia['normalizacao']}\n"
            f"- Equalização: {estrategia['equalizacao']}\n"
            f"- Tamanho-alvo: {estrategia['tamanho_alvo']}\n\n"
            f"Resultado completo persistido em analise_base.json."
        )
