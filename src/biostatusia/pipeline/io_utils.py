from datetime import datetime
from pathlib import Path

# ── Famílias existentes ────────────────────────────────────────────────────────
EXT_IMAGENS   = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
EXT_TABULARES = (".csv", ".txt", ".tsv", ".xlsx")

# ── Famílias de sinais biomédicos (v3 — F1/F3/F4) ─────────────────────────────
EXT_SINAIS_TEMPORAIS = frozenset({
    ".edf", ".bdf", ".dat", ".hea", ".atr",
    ".mat", ".cnt", ".eeg", ".rec", ".c3d", ".set", ".xml",
})
EXT_DICOM           = frozenset({".dcm"})
# .nii.gz precisa de checagem dupla via Path.suffixes — veja eh_volumetrico()
EXT_VOLUMETRICO_SIMPLES = frozenset({".nii", ".mha"})

PASTAS_BENIGNAS = {"benign", "benigno", "normal", "negative", "no", "0"}
PASTAS_MALIGNAS = {"malignant", "malign", "maligno", "abnormal", "positive", "yes", "1"}

RUNS_DIR = Path(__file__).parent.parent / "static" / "runs"


# ── Predicados por família ────────────────────────────────────────────────────

def eh_imagem(arquivo: Path) -> bool:
    return arquivo.suffix.lower() in EXT_IMAGENS and "mask" not in arquivo.name.lower()


def eh_tabular(arquivo: Path) -> bool:
    return arquivo.suffix.lower() in EXT_TABULARES


def eh_sinal_temporal(arquivo: Path) -> bool:
    return arquivo.suffix.lower() in EXT_SINAIS_TEMPORAIS


def eh_dicom(arquivo: Path) -> bool:
    return arquivo.suffix.lower() in EXT_DICOM


def eh_volumetrico(arquivo: Path) -> bool:
    ext = arquivo.suffix.lower()
    suffixes = [s.lower() for s in arquivo.suffixes]
    return ext in EXT_VOLUMETRICO_SIMPLES or suffixes == [".nii", ".gz"]


def eh_sinal_novo(arquivo: Path) -> bool:
    """True para qualquer extensão das famílias de sinal em escopo (F1/F3/F4)."""
    return (
        eh_sinal_temporal(arquivo)
        or eh_dicom(arquivo)
        or eh_volumetrico(arquivo)
    )


# ── Utilitários de rótulo e listagem ─────────────────────────────────────────

def label_pasta(nome_pasta: str) -> int | None:
    nome = nome_pasta.lower()
    if nome in PASTAS_BENIGNAS:
        return 0
    if nome in PASTAS_MALIGNAS:
        return 1
    return None


# Nomes de pasta que indicam uma divisão treino/teste/validação — não são,
# em si, nomes de classe, mesmo quando contêm imagens diretamente.
PASTAS_SPLIT_INDICADOR = {"train", "training", "treino", "test", "testing", "teste",
                          "val", "valid", "validation", "validacao"}

LIMITE_CLASSES_POR_PASTA = 20  # evita falso positivo (ex.: uma pasta por paciente)


def _nome_classe_para_imagem(arq: Path, raiz: Path) -> str | None:
    """Nome de pasta candidato a representar a CLASSE desta imagem — o pai
    imediato, ou o avô quando o pai é um indicador de split (Training/Test)."""
    pai = arq.parent
    if pai == raiz:
        return None
    if pai.name.lower() in PASTAS_SPLIT_INDICADOR:
        avo = pai.parent
        return avo.name if avo != raiz.parent and avo.name else None
    return pai.name


def listar_imagens(caminho: Path) -> list[dict]:
    """
    Lista imagens (famílias existentes). Detecta rótulo pelas pastas pai —
    tanto a convenção binária conhecida (benign/malignant, yes/no, etc.)
    quanto uma estrutura multi-classe genuína (2+ pastas de classe
    distintas, cada uma com suas próprias imagens — ex.: glioma/
    meningioma/notumor/pituitary), inclusive quando organizada dentro de
    pastas de split como Training/Testing.
    """
    if caminho.is_file() and eh_imagem(caminho):
        return [{"caminho": str(caminho), "label": None, "categoria": "INDEFINIDO"}]
    if not caminho.is_dir():
        return []

    arquivos = [arq for arq in sorted(caminho.rglob("*")) if arq.is_file() and eh_imagem(arq)]

    # 1ª tentativa: convenção binária conhecida (mantém a categoria clínica
    # BENIGNO/MALIGNO — mais informativa do que um nome de pasta genérico).
    tem_binario = any(label_pasta(arq.parent.name) is not None for arq in arquivos)
    if tem_binario:
        registros = []
        for arq in arquivos:
            label = label_pasta(arq.parent.name)
            categoria = "MALIGNO" if label == 1 else "BENIGNO" if label == 0 else "INDEFINIDO"
            registros.append({"caminho": str(arq), "label": label, "categoria": categoria})
        return registros

    # 2ª tentativa: estrutura multi-classe genuína — nomes de pasta distintos
    # (que não sejam indicadores de split) usados de forma consistente.
    nomes_por_arquivo = {str(arq): _nome_classe_para_imagem(arq, caminho) for arq in arquivos}
    nomes_distintos = sorted({n for n in nomes_por_arquivo.values() if n})
    if 2 <= len(nomes_distintos) <= LIMITE_CLASSES_POR_PASTA:
        indice_por_nome = {nome: i for i, nome in enumerate(nomes_distintos)}
        registros = []
        for arq in arquivos:
            nome_classe = nomes_por_arquivo[str(arq)]
            if nome_classe is None:
                registros.append({"caminho": str(arq), "label": None, "categoria": "INDEFINIDO"})
            else:
                registros.append({
                    "caminho": str(arq), "label": indice_por_nome[nome_classe],
                    "categoria": nome_classe.upper(),
                })
        return registros

    # Nenhuma convenção de rótulo reconhecida — todas indefinidas (como antes).
    return [{"caminho": str(arq), "label": None, "categoria": "INDEFINIDO"} for arq in arquivos]


def listar_sinais(caminho: Path, predicado) -> list[dict]:
    """Lista arquivos de sinal usando um predicado (ex: eh_sinal_temporal)."""
    if caminho.is_file() and predicado(caminho):
        return [{"caminho": str(caminho), "label": None, "categoria": "INDEFINIDO"}]
    if not caminho.is_dir():
        return []
    registros: list[dict] = []
    for arq in sorted(caminho.rglob("*")):
        if not (arq.is_file() and predicado(arq)):
            continue
        label = label_pasta(arq.parent.name)
        categoria = "MALIGNO" if label == 1 else "BENIGNO" if label == 0 else "INDEFINIDO"
        registros.append({"caminho": str(arq), "label": label, "categoria": categoria})
    return registros


def encontrar_csv(caminho: Path) -> Path | None:
    if caminho.is_file() and eh_tabular(caminho):
        return caminho
    if not caminho.is_dir():
        return None
    for arq in sorted(caminho.rglob("*")):
        if arq.is_file() and eh_tabular(arq):
            return arq
    return None


def encontrar_todos_tabulares(caminho: Path) -> list[Path]:
    """Como `encontrar_csv`, mas devolve TODOS os arquivos tabulares
    encontrados, não só o primeiro. Alguns datasets trazem mais de uma
    tabela relevante (ex.: HMC-QU tem A2C.xlsx e A4C.xlsx, cada um cobrindo
    um subconjunto diferente de pacientes) — usar só o primeiro
    alfabeticamente pode escolher o arquivo errado para o alinhamento."""
    if caminho.is_file():
        return [caminho] if eh_tabular(caminho) else []
    if not caminho.is_dir():
        return []
    return sorted(arq for arq in caminho.rglob("*") if arq.is_file() and eh_tabular(arq))


def criar_pasta_run() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pasta = RUNS_DIR / f"run_{timestamp}"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


ROTULO_MODALIDADE = {
    "imagem": "Imagens (ex.: raio-X, ultrassom, fotos clínicas)",
    "tabular": "Dados tabulares (ex.: prontuário, exames laboratoriais)",
    "sinal_temporal": "Sinais biomédicos (ex.: ECG, EEG, PPG)",
    "dicom": "Imagens DICOM",
    "volume_3d": "Volumes 3D (ex.: CT, MRI)",
}


def _contar_linhas_tabular(arq: Path) -> int:
    """Conta linhas de dado (sem contar cabeçalho) de um arquivo tabular,
    detectando pelo CONTEÚDO se é um .xlsx (assinatura ZIP/"PK") ou texto
    delimitado — evita contar bytes binários como se fossem linhas de texto
    quando a extensão diz uma coisa e o conteúdo é outra."""
    try:
        with open(arq, "rb") as f:
            assinatura = f.read(4)
        if assinatura[:2] == b"PK":
            try:
                from openpyxl import load_workbook
                wb = load_workbook(filename=str(arq), read_only=True, data_only=True)
                n = max(0, wb.worksheets[0].max_row - 1)
                wb.close()
                return n
            except Exception:
                return 0
        with open(arq, "r", encoding="utf-8-sig", errors="ignore") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def inventariar_modalidades(caminho: Path) -> dict:
    """
    Conta quantos arquivos de cada família (imagem, tabular, sinal, DICOM,
    volume 3D) existem em `caminho`, sem extrair nenhuma feature nem rodar
    nenhum modelo — só listagem de arquivos, para ser rápido o suficiente
    para rodar antes de confirmar uma análise multimodal com o usuário.
    """
    contagens = {"imagem": 0, "tabular": 0, "sinal_temporal": 0, "dicom": 0, "volume_3d": 0}
    if caminho.is_file():
        arquivos = [caminho]
    elif caminho.is_dir():
        arquivos = [a for a in caminho.rglob("*") if a.is_file()]
    else:
        arquivos = []

    n_linhas_tabular = 0
    for arq in arquivos:
        if eh_imagem(arq):
            contagens["imagem"] += 1
        elif eh_tabular(arq):
            contagens["tabular"] += 1
            n_linhas_tabular += _contar_linhas_tabular(arq)
        elif eh_sinal_temporal(arq):
            contagens["sinal_temporal"] += 1
        elif eh_dicom(arq):
            contagens["dicom"] += 1
        elif eh_volumetrico(arq):
            contagens["volume_3d"] += 1

    resultado = []
    for familia, n in contagens.items():
        if n > 0:
            item = {"familia": familia, "rotulo": ROTULO_MODALIDADE[familia], "n_arquivos": n}
            if familia == "tabular" and n_linhas_tabular:
                item["n_linhas"] = n_linhas_tabular
            resultado.append(item)
    return {"modalidades": resultado, "n_familias": len(resultado)}
