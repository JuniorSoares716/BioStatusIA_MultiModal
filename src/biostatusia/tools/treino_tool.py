import json
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from biostatusia.pipeline.classificador import treinar


class TreinoInput(BaseModel):
    pasta_run: str = Field(..., description="Pasta de workspace com biomarcadores.json.")


class FerramentaTreinarClassificador(BaseTool):
    name: str = "ferramenta_treinar_classificador"
    description: str = (
        "Treina SVM (RBF) e RandomForest (100 árvores) com os biomarcadores extraídos. "
        "Requer dataset rotulado (BENIGNO + MALIGNO, mínimo 10 amostras com 2 classes). "
        "Persiste métricas, ROC e matriz de confusão em metricas.json e retorna resumo."
    )
    args_schema: type[BaseModel] = TreinoInput

    def _run(self, pasta_run: str) -> str:
        pasta = Path(pasta_run.strip("'\""))
        bio_path = pasta / "biomarcadores.json"
        if not bio_path.exists():
            return f"Erro: biomarcadores.json não encontrado em {pasta}. Extraia biomarcadores antes."

        with open(bio_path, "r", encoding="utf-8") as f:
            registros = json.load(f)

        registros_treino = [r for r in registros if r.get("label") is not None]
        labels = {r["label"] for r in registros_treino}

        if len(registros_treino) < 10 or len(labels) < 2:
            msg = (
                f"Treino não executado: {len(registros_treino)} amostras rotuladas, "
                f"{len(labels)} classes (mínimo: 10 amostras, 2 classes)."
            )
            with open(pasta / "metricas.json", "w", encoding="utf-8") as f:
                json.dump({"aviso": msg}, f, indent=2)
            return msg

        try:
            resultado = treinar([
                {"biomarcadores": r["biomarcadores"], "label": r["label"]}
                for r in registros_treino
            ])
        except Exception as e:
            return f"Erro ao treinar: {e}"

        with open(pasta / "metricas.json", "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2)

        melhor = resultado["melhor_modelo"]
        metricas = resultado["metricas"]
        linhas = [
            f"Treino concluído sobre {len(registros_treino)} amostras (test_size=0.2).",
            "",
            f"Melhor modelo: **{melhor}** (selecionado por maior AUC-ROC).",
            "",
            "| Modelo | Acurácia | Precisão | Recall | F1 | AUC |",
            "|---|---|---|---|---|---|",
        ]
        for nome, m in metricas.items():
            linhas.append(
                f"| {nome} | {m['acuracia']} | {m['precisao']} | "
                f"{m['recall']} | {m['f1']} | {m['auc']} |"
            )
        linhas.append("")
        linhas.append("Resultado completo persistido em metricas.json.")
        return "\n".join(linhas)
