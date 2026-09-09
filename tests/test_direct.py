import os
import sys
from pathlib import Path
import json

RAIZ = Path(__file__).resolve().parent.parent  # raiz do repositório (pasta acima de tests/)
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.biostatusia.pipeline.io_utils import criar_pasta_run
from src.biostatusia.tools.analise_base_tool import FerramentaAnaliseBase
from src.biostatusia.tools.extracao_tool import FerramentaExtrairBiomarcadores
from src.biostatusia.tools.treino_tool import FerramentaTreinarClassificador
from src.biostatusia.tools.tabular_tool import FerramentaAnaliseTabular
from src.biostatusia.tools.sinais_temporais_tool import FerramentaExtrairSinalTemporal

def run():
    saida = RAIZ / "reports" / "resultados_teste_direto.md"
    saida.parent.mkdir(exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        f.write("# Resultados dos Testes da Plataforma BioStatusIA (Execução Direta)\n\n")

        # 1. Imagem
        f.write("## 1. Teste de Imagens (Câncer de Mama Dummy)\n")
        img_path = RAIZ / "tests" / "test_platform_data" / "image_test"
        if img_path.exists():
            pasta_run = criar_pasta_run()
            f.write("### Analisando Base (Engenheiro PDI)\n")
            res1 = FerramentaAnaliseBase()._run(str(img_path), str(pasta_run))
            f.write("```\n" + str(res1) + "\n```\n")
            f.write("### Extraindo Biomarcadores (Analista Técnico)\n")
            res2 = FerramentaExtrairBiomarcadores()._run(str(img_path), str(pasta_run))
            f.write("```\n" + str(res2) + "\n```\n")
            f.write("### Treinando Classificador (Cientista de Dados)\n")
            res3 = FerramentaTreinarClassificador()._run(str(pasta_run))
            f.write("```\n" + str(res3) + "\n```\n")
            f.write("\n---\n")

        # 2. Tabular
        f.write("## 2. Teste Tabular (WBCD-50 CSV)\n")
        csv_path = RAIZ / "dataset_teste_csv" / "wbcd_50.csv"
        if csv_path.exists():
            f.write("### Análise Tabular\n")
            res_tab = FerramentaAnaliseTabular()._run(str(csv_path))
            f.write("```\n" + str(res_tab) + "\n```\n")
            f.write("\n---\n")

        # 3. Sinal
        f.write("## 3. Teste de Sinal (ECG Dummy)\n")
        sig_path = RAIZ / "tests" / "test_platform_data" / "signal_test"
        if sig_path.exists():
            pasta_run2 = criar_pasta_run()
            f.write("### Extração de Sinal Temporal (Analista Sinais Fisiológicos)\n")
            res_sig = FerramentaExtrairSinalTemporal()._run(str(sig_path), str(pasta_run2), "ECG")
            f.write("```\n" + str(res_sig) + "\n```\n")
            f.write("\n---\n")


if __name__ == '__main__':
    run()
