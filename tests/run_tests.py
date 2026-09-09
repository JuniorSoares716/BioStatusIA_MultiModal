import os
import sys
import json
import numpy as np
from PIL import Image
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent  # raiz do repositório (pasta acima de tests/)
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.biostatusia.crew import BioStatusIACrew, BioStatusIACrewTabular, BioStatusIACrewSinal
from src.biostatusia.pipeline.io_utils import criar_pasta_run

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

def generate_datasets():
    test_dir = RAIZ / "tests" / "test_platform_data"
    test_dir.mkdir(exist_ok=True)
    
    # 1. Image
    img_dir = test_dir / "image_test"
    img_dir.mkdir(exist_ok=True)
    img_path = img_dir / "test_img.png"
    if not img_path.exists():
        img_data = np.zeros((256, 256, 3), dtype=np.uint8)
        img_data[100:150, 100:150] = [255, 255, 255]
        Image.fromarray(img_data).save(img_path)
        
    # 2. Signal (Heart Signal as dummy .dat for wfdb or just a .csv that might be recognized if they support it. Wait, eh_sinal_temporal looks for .dat, .edf, etc. For wfdb to work, .dat requires a .hea. Let's create an .edf instead? .edf might require pyedflib or mne. MNE is installed! MNE can read .edf or .mat. Let's create a .mat using scipy.io or just create an .edf)
    # Since wfdb and mne are there, maybe a dummy audio file (.wav) is easier to create without external packages, but the user explicitly requested "como sinais de coração" which implies ECG. I'll create a dummy ECG in .mat format for scipy.io to read.
    # Actually, a simple CSV with .csv extension for Tabular is already handled.
    # What if I create a dummy .mat with 'val' variable (standard for physionet matlab exports)?
    import scipy.io
    sig_dir = test_dir / "signal_test"
    sig_dir.mkdir(exist_ok=True)
    sig_path = sig_dir / "test_ecg.mat"
    if not sig_path.exists():
        # Dummy ECG-like signal
        t = np.linspace(0, 10, 1000)
        ecg = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(1000)
        scipy.io.savemat(sig_path, {'val': ecg.reshape(1, -1)})

    return img_path, sig_path, RAIZ / "dataset_teste_csv" / "wbcd_50.csv"

def run():
    img_path, sig_path, tab_path = generate_datasets()
    results_md = "# Resultados dos Testes da Plataforma BioStatusIA\n\n"
    
    print("[TESTE 1] Imagem (Câncer de Mama Dummy)")
    pasta_run = criar_pasta_run()
    try:
        res = BioStatusIACrew().crew().kickoff(inputs={"caminho_dataset": str(img_path), "pasta_run": str(pasta_run)})
        results_md += "## Teste 1: Imagem (Câncer de Mama Dummy)\n\n"
        results_md += str(res) + "\n\n"
    except Exception as e:
        results_md += f"## Teste 1: Imagem\n**ERRO:** {e}\n\n"

    print("[TESTE 2] Dados Tabulares (WBCD-50)")
    pasta_run = criar_pasta_run()
    try:
        res = BioStatusIACrewTabular().crew().kickoff(inputs={"caminho_csv": str(tab_path)})
        results_md += "## Teste 2: Dados Tabulares (WBCD-50 CSV)\n\n"
        results_md += str(res) + "\n\n"
    except Exception as e:
        results_md += f"## Teste 2: Tabular\n**ERRO:** {e}\n\n"

    print("[TESTE 3] Sinal (Sinal de Coração Dummy)")
    pasta_run = criar_pasta_run()
    try:
        res = BioStatusIACrewSinal().crew().kickoff(inputs={
            "caminho_dataset": str(sig_path), 
            "pasta_run": str(pasta_run),
            "tipo_sinal": "ECG"
        })
        results_md += "## Teste 3: Sinal (Sinal de Coração Dummy)\n\n"
        results_md += str(res) + "\n\n"
    except Exception as e:
        results_md += f"## Teste 3: Sinal\n**ERRO:** {e}\n\n"

    saida = RAIZ / "reports" / "resultados_teste.md"
    saida.parent.mkdir(exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        f.write(results_md)
    print(f"Testes concluídos. Resultados salvos em {saida}")

if __name__ == '__main__':
    run()
