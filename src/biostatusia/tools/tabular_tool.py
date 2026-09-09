from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from biostatusia.pipeline.dados_tabulares import (
    analisar_tabular,
    carregar_csv,
    detectar_schema,
    extrair_features,
)


class AnaliseTabularInput(BaseModel):
    caminho_csv: str = Field(..., description="Caminho do arquivo CSV/TXT/TSV para análise.")


class FerramentaAnaliseTabular(BaseTool):
    name: str = "ferramenta_analise_tabular"
    description: str = (
        "Carrega um arquivo CSV/TXT/TSV, identifica a coluna-rótulo e as features "
        "numéricas, e retorna um resumo estatístico das features por classe — "
        "pronto para interpretação clínica."
    )
    args_schema: type[BaseModel] = AnaliseTabularInput

    def _run(self, caminho_csv: str) -> str:
        caminho_limpo = caminho_csv.strip("'\"")
        resultado = carregar_csv(caminho_limpo)
        if not resultado:
            return "Erro: falha ao ler o arquivo CSV/TXT."

        header, data = resultado
        schema = detectar_schema(header, data)
        X, y, label_map = extrair_features(data, schema)
        if X.size == 0:
            return "Erro: nenhuma feature numérica válida encontrada."

        stats = analisar_tabular(X, y, schema)
        n_features_mostrar = min(10, stats["n_features"])

        linhas: list[str] = []
        linhas.append(f"# Resumo Tabular: {Path(caminho_limpo).name}")
        linhas.append("")
        linhas.append(f"- **Amostras**: {stats['n_amostras']}")
        linhas.append(f"- **Features numéricas**: {stats['n_features']}")
        if schema.get("label_name"):
            linhas.append(f"- **Coluna-rótulo**: `{schema['label_name']}`")
        if label_map:
            mapa_str = ", ".join(f"{k}→{v}" for k, v in label_map.items())
            linhas.append(f"- **Mapa de rótulos**: {mapa_str}")
        if stats.get("distribuicao_labels"):
            dist = ", ".join(f"classe {k}: {v}" for k, v in stats["distribuicao_labels"].items())
            linhas.append(f"- **Distribuição de classes**: {dist}")
        linhas.append("")
        linhas.append(f"## Top {n_features_mostrar} Features (estatísticas)")
        linhas.append("")
        linhas.append("| Feature | Média | Mediana | Desvio | Min | Max |")
        linhas.append("|---|---|---|---|---|---|")

        for fname, fstats in list(stats["features"].items())[:n_features_mostrar]:
            linhas.append(
                f"| {fname} | {fstats['media']} | {fstats['mediana']} | "
                f"{fstats['desvio']} | {fstats['min']} | {fstats['max']} |"
            )

        if stats["n_features"] > n_features_mostrar:
            linhas.append("")
            linhas.append(f"_(+{stats['n_features'] - n_features_mostrar} features omitidas no resumo)_")

        return "\n".join(linhas)
