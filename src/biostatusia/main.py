#!/usr/bin/env python
import os
import sys
import webbrowser
from datetime import datetime

# UTF-8 obrigatório no Windows — CrewAI EventBus usa emojis
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Garante que o pacote seja encontrado independente do diretório de execução
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

import markdown
from biostatusia.crew import BioStatusIACrew
from biostatusia.database import salvar, listar


def run() -> None:
    print("\n[BIOSTATSIA] Iniciando Sistema de Laudos Radiologicos com IA...")

    try:
        import kagglehub
        caminho_dataset = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
        base_exames = os.path.join(caminho_dataset, "Dataset_BUSI_with_GT")

        lista_final_exames: list[str] = []
        for cat in ["benign", "malignant"]:
            pasta = os.path.join(base_exames, cat)
            if not os.path.isdir(pasta):
                continue
            arquivos = [
                os.path.join(pasta, f)
                for f in os.listdir(pasta)
                if f.endswith(".png") and "mask" not in f.lower()
            ][:1]
            lista_final_exames.extend(arquivos)

        # Processa apenas a primeira imagem por padrão
        lista_final_exames = lista_final_exames[:1]

    except Exception as e:
        print(f"[AVISO] Nao foi possivel baixar o dataset do Kaggle: {e}")
        print("[INFO] Informe o caminho de uma imagem manualmente:")
        caminho_manual = input("Caminho da imagem: ").strip()
        lista_final_exames = [caminho_manual] if caminho_manual else []

    if not lista_final_exames:
        print("[ERRO] Nenhuma imagem disponivel para analise.")
        return

    laudos_html = ""
    data_atual = datetime.now().strftime("%d/%m/%Y")
    id_processo = f"USG-{datetime.now().strftime('%Y%m%d')}-BOS"

    for i, caminho in enumerate(lista_final_exames, 1):
        nome_img = os.path.basename(caminho)
        categoria = "MALIGNO" if "malignant" in caminho else "BENIGNO"
        badge = "bg-red-100 text-red-700" if categoria == "MALIGNO" else "bg-green-100 text-green-700"

        print(f"[{i}/{len(lista_final_exames)}] Analisando {categoria}: {nome_img}...")

        try:
            resultado = BioStatusIACrew().crew().kickoff(inputs={"caminho_imagem": caminho})
            laudo_texto = str(resultado)
            laudo_html = markdown.markdown(laudo_texto)
            row_id = salvar(caminho, categoria, laudo_texto)
            print(f"[DB] Analise salva — id={row_id}")
        except Exception as exc:
            laudo_html = f"<p><strong>Erro na analise:</strong> {exc}</p>"
            laudo_texto = str(exc)

        laudos_html += f"""
        <div class="grid grid-cols-1 md:grid-cols-3 gap-8 items-start border-b border-gray-100 pb-12 mb-12">
            <div class="col-span-1 pr-6">
                <div class="flex items-center gap-2 mb-4">
                    <span class="px-2 py-1 rounded text-[10px] font-bold uppercase {badge}">{categoria}</span>
                    <h3 class="text-xs font-bold text-gray-400">{nome_img}</h3>
                </div>
                <img src="file:///{caminho}" class="w-full rounded-lg shadow-sm border border-gray-200">
            </div>
            <div class="col-span-2">
                <div class="bg-teal-50/50 p-6 rounded-xl border border-teal-100">
                    <h4 class="text-teal-800 font-bold text-sm mb-3">Laudo dos Agentes IA</h4>
                    <div class="text-sm text-gray-700 leading-relaxed">{laudo_html}</div>
                </div>
            </div>
        </div>
        """

    # Geração dos relatórios HTML
    raiz = os.path.abspath(os.path.join(base_path, "..", ".."))

    _gerar_html(
        os.path.join(raiz, "template_moderno.html"),
        os.path.join(raiz, "dashboard_inteligencia_medica.html"),
        {"{{CONTEUDO_DINAMICO}}": laudos_html, "{{ID_PROCESSO}}": id_processo, "{{DATA_ATUAL}}": data_atual},
    )
    _gerar_html(
        os.path.join(raiz, "template_relatorio.html"),
        os.path.join(raiz, "relatorio_tecnico_biostatsia.html"),
        {"{{LAUDO_FINAL_AQUI}}": laudos_html},
    )

    out = os.path.join(raiz, "dashboard_inteligencia_medica.html")
    if os.path.exists(out):
        webbrowser.open(f"file:///{out}")
        print(f"[OK] Dashboard aberto: {out}")

    historico = listar()
    print(f"\n[DB] Historico de analises ({len(historico)} registro(s)):")
    for row in historico[:5]:
        print(f"  id={row[0]} | {row[1][:19]} | {row[3]:10} | {os.path.basename(row[2])}")


def _gerar_html(template: str, saida: str, substituicoes: dict) -> None:
    if not os.path.exists(template):
        return
    try:
        with open(template, "r", encoding="utf-8") as f:
            html = f.read()
        for chave, valor in substituicoes.items():
            html = html.replace(chave, valor)
        with open(saida, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[OK] Arquivo gerado: {saida}")
    except Exception as e:
        print(f"[ERRO] Falha ao gerar {saida}: {e}")


if __name__ == "__main__":
    run()
