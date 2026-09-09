import os
import sys
from pathlib import Path
import kagglehub
import shutil

RAIZ = Path(__file__).resolve().parent.parent  # raiz do repositório (pasta acima de tests/)
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.biostatusia.pipeline.io_utils import criar_pasta_run
from src.biostatusia.tools.analise_base_tool import FerramentaAnaliseBase
from src.biostatusia.tools.extracao_tool import FerramentaExtrairBiomarcadores
from src.biostatusia.tools.treino_tool import FerramentaTreinarClassificador

def run():
    print("Baixando dataset completo do Kaggle (aryashah2k/breast-ultrasound-images-dataset)...")
    try:
        caminho_dataset = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
        print(f"Dataset baixado em: {caminho_dataset}")
    except Exception as e:
        print(f"Erro ao baixar dataset: {e}")
        return

    # O dataset extrai geralmente com a subpasta Dataset_BUSI_with_GT
    base_exames = Path(caminho_dataset) / "Dataset_BUSI_with_GT"
    if not base_exames.exists():
        base_exames = Path(caminho_dataset)

    # Vamos copiar apenas as imagens sem máscara para uma pasta temporária
    # pois o pipeline pode processar as máscaras como se fossem exames
    # se não fizermos isso (embora o código ignore arquivos com "mask" no nome)
    # O código original no IO Utils: "mask" not in arquivo.name.lower()
    # Então não precisamos filtrar aqui.
    
    pasta_run = criar_pasta_run()
    
    saida = RAIZ / "reports" / "resultados_large_dataset.md"
    saida.parent.mkdir(exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        f.write("# Resultados - Base de Imagens Completa (Kaggle)\n\n")
        f.write(f"Dataset localizado em: `{base_exames}`\n\n")

        print("Executando: Analisando Base (Engenheiro PDI)")
        f.write("### Analisando Base (Engenheiro PDI)\n")
        res1 = FerramentaAnaliseBase()._run(str(base_exames), str(pasta_run))
        f.write("```\n" + str(res1) + "\n```\n")

        print("Executando: Extraindo Biomarcadores (Analista Técnico)")
        f.write("### Extraindo Biomarcadores (Analista Técnico)\n")
        res2 = FerramentaExtrairBiomarcadores()._run(str(base_exames), str(pasta_run))
        f.write("```\n" + str(res2) + "\n```\n")

        print("Executando: Treinando Classificador (Cientista de Dados)")
        f.write("### Treinando Classificador (AutoML)\n")
        res3 = FerramentaTreinarClassificador()._run(str(pasta_run))
        f.write("```\n" + str(res3) + "\n```\n")

    print(f"Processamento concluído. Resultados salvos em {saida}")

if __name__ == "__main__":
    run()
