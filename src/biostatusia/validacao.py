"""
Validações de entrada do usuário (BioStatusIA).

Antes desta correção, o sistema não validava NENHUM campo do formulário de
upload: um `kaggle_id` com uma URL colada por engano (ex.: o link de um
notebook do Kaggle, ou a página do dataset em vez do ID `dono/dataset`) só
falhava lá na frente, dentro do `kagglehub`, com uma mensagem de erro crua e
técnica. Um `caminho_manual` inexistente também só era descoberto depois de
já ter tentado detectar a estrutura da pasta. Este módulo centraliza essas
checagens, sempre devolvendo uma mensagem clara e um exemplo do formato
correto — para a interface poder mostrar isso num modal, em vez de um erro
genérico.

Cada `validar_*` devolve um dict:
    {"valido": bool, "valor": str|None, "erro": str|None,
     "exemplo_correto": str|None, "corrigido_automaticamente": bool}
- "valor": o valor (possivelmente normalizado/corrigido) a ser usado se válido.
- "corrigido_automaticamente": True quando conseguimos extrair o valor certo
  de algo que a pessoa colou errado (ex.: URL completa do Kaggle).
"""
import re
from pathlib import Path

EXTENSOES_ACEITAS = frozenset({
    ".zip", ".tar.gz",
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
    ".csv", ".txt", ".tsv", ".xlsx",
    ".edf", ".bdf", ".dat", ".hea", ".mat", ".cnt", ".eeg", ".rec", ".c3d", ".set", ".xml",
    ".dcm", ".nii", ".gz", ".mha",
})

TIPOS_SINAL_VALIDOS = frozenset({
    "auto", "ECG", "EEG", "EMG", "EOG", "PPG", "PA", "Espirometria",
})

MAX_UPLOAD_BYTES = 4096 * 1024 * 1024  # 4 GB — mesmo limite de app.config["MAX_CONTENT_LENGTH"]

_RE_KAGGLE_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
_RE_KAGGLE_DATASET_URL = re.compile(r"kaggle\.com/datasets/([\w.-]+)/([\w.-]+)", re.IGNORECASE)
_RE_KAGGLE_CODE_URL = re.compile(r"kaggle\.com/code/", re.IGNORECASE)
_RE_KAGGLE_COMPETITIONS_URL = re.compile(r"kaggle\.com/competitions/([\w.-]+)", re.IGNORECASE)


def _resultado(valido: bool, valor: str | None = None, erro: str | None = None,
               exemplo_correto: str | None = None, corrigido: bool = False) -> dict:
    return {
        "valido": valido, "valor": valor, "erro": erro,
        "exemplo_correto": exemplo_correto, "corrigido_automaticamente": corrigido,
    }


def validar_kaggle_id(bruto: str) -> dict:
    """
    Valida/normaliza o campo de dataset do Kaggle. Aceita o formato esperado
    (`dono/dataset`) diretamente, e tenta extrair automaticamente esse ID de
    erros comuns: colar a URL completa da página do dataset, ou colar o link
    de um notebook (que não é, em si, um dataset baixável).
    """
    valor = (bruto or "").strip()
    exemplo = "paultimothymooney/chest-xray-pneumonia"
    if not valor:
        return _resultado(True, valor="")  # campo vazio é permitido (outro método pode ter sido usado)

    if "kaggle.com" in valor.lower():
        m = _RE_KAGGLE_DATASET_URL.search(valor)
        if m:
            return _resultado(True, valor=f"{m.group(1)}/{m.group(2)}", corrigido=True)
        if _RE_KAGGLE_CODE_URL.search(valor):
            return _resultado(
                False, erro=(
                    "Esse é o link de um notebook/código do Kaggle, não de um dataset. "
                    "Abra o notebook, veja a aba \"Input\" (dados usados), clique no "
                    "dataset e copie o ID de lá."
                ),
                exemplo_correto=exemplo,
            )
        if _RE_KAGGLE_COMPETITIONS_URL.search(valor):
            return _resultado(
                False, erro=(
                    "Esse é o link de uma competição do Kaggle. Datasets de competição "
                    "usam outro mecanismo de download (exige aceitar as regras da "
                    "competição no site) e não são suportados diretamente por aqui."
                ),
                exemplo_correto=exemplo,
            )
        return _resultado(
            False, erro="Não conseguimos identificar um dataset válido nesse link do Kaggle.",
            exemplo_correto=exemplo,
        )

    if _RE_KAGGLE_ID.match(valor):
        return _resultado(True, valor=valor)

    return _resultado(
        False, erro=(
            "Formato não reconhecido. O ID do dataset deve ter exatamente uma "
            "barra, no formato \"dono/nome-do-dataset\" (sem 'https://', sem "
            "'kaggle.com')."
        ),
        exemplo_correto=exemplo,
    )


def validar_caminho_manual(bruto: str) -> dict:
    """Valida o caminho local informado — precisa existir no servidor."""
    valor = (bruto or "").strip()
    if not valor:
        return _resultado(True, valor="")

    exemplo = r"C:\Users\clinica\exames\hoje" if "\\" in valor or ":" in valor else "/home/clinica/exames/hoje"
    caminho = Path(valor)
    if not caminho.exists():
        return _resultado(
            False, erro=f"O caminho \"{valor}\" não foi encontrado no servidor.",
            exemplo_correto=exemplo,
        )
    if not (caminho.is_file() or caminho.is_dir()):
        return _resultado(
            False, erro=f"\"{valor}\" existe, mas não é um arquivo nem uma pasta válida.",
            exemplo_correto=exemplo,
        )
    return _resultado(True, valor=valor)


def validar_tipo_sinal(bruto: str) -> dict:
    """Valida o tipo de sinal selecionado contra a lista de opções suportadas."""
    valor = (bruto or "auto").strip()
    if valor not in TIPOS_SINAL_VALIDOS:
        return _resultado(
            False,
            erro=f"Tipo de sinal \"{valor}\" não reconhecido.",
            exemplo_correto="auto (deixa o sistema detectar automaticamente)",
        )
    return _resultado(True, valor=valor)


def validar_arquivo_upload(nome_arquivo: str, tamanho_bytes: int | None = None) -> dict:
    """Valida a extensão (e, se informado, o tamanho) do arquivo enviado."""
    if not nome_arquivo:
        return _resultado(True, valor="")

    sufixos = Path(nome_arquivo.lower()).suffixes
    ext_simples = f".{sufixos[-1].lstrip('.')}" if sufixos else ""
    ext_composta = "".join(sufixos[-2:]) if len(sufixos) >= 2 else ext_simples

    if ext_simples not in EXTENSOES_ACEITAS and ext_composta not in EXTENSOES_ACEITAS:
        return _resultado(
            False,
            erro=(
                f"Extensão \"{ext_simples or '(sem extensão)'}\" não é suportada."
            ),
            exemplo_correto=(
                "Use .zip, .csv/.txt/.tsv/.xlsx (tabular), .png/.jpg/.tif (imagem), "
                ".edf/.mat/.hea (sinal), .dcm (DICOM) ou .nii/.mha (volume 3D)."
            ),
        )

    if tamanho_bytes is not None and tamanho_bytes > MAX_UPLOAD_BYTES:
        limite_gb = MAX_UPLOAD_BYTES / (1024 ** 3)
        tamanho_gb = tamanho_bytes / (1024 ** 3)
        return _resultado(
            False,
            erro=f"Arquivo de {tamanho_gb:.1f} GB excede o limite de {limite_gb:.0f} GB.",
            exemplo_correto=f"Envie um arquivo de até {limite_gb:.0f} GB, ou aponte para um caminho local em vez de fazer upload.",
        )

    return _resultado(True, valor=nome_arquivo)


def validar_entrada_analise(arquivo_nome: str | None, arquivo_tamanho: int | None,
                             caminho_manual: str, kaggle_id: str, tipo_sinal: str) -> dict:
    """
    Validação consolidada dos campos do formulário de análise. Roda todas as
    checagens e devolve a PRIMEIRA falha encontrada (ordem: arquivo, caminho,
    kaggle_id, tipo_sinal), já que o `/analisar` tenta essas fontes nessa
    mesma ordem de prioridade. Se tudo for válido, devolve os valores
    normalizados (ex.: kaggle_id com URL convertida para "dono/dataset").
    """
    r_arquivo = validar_arquivo_upload(arquivo_nome or "", arquivo_tamanho)
    if not r_arquivo["valido"]:
        return {"campo": "arquivo", **r_arquivo}

    r_caminho = validar_caminho_manual(caminho_manual)
    if not r_caminho["valido"]:
        return {"campo": "caminho_manual", **r_caminho}

    r_kaggle = validar_kaggle_id(kaggle_id)
    if not r_kaggle["valido"]:
        return {"campo": "kaggle_id", **r_kaggle}

    r_tipo = validar_tipo_sinal(tipo_sinal)
    if not r_tipo["valido"]:
        return {"campo": "tipo_sinal", **r_tipo}

    nenhuma_fonte = not arquivo_nome and not r_caminho["valor"] and not r_kaggle["valor"]

    return {
        "campo": None, "valido": True, "erro": None, "exemplo_correto": None,
        "corrigido_automaticamente": r_kaggle["corrigido_automaticamente"],
        "kaggle_id_normalizado": r_kaggle["valor"],
        "nenhuma_fonte_informada": nenhuma_fonte,
    }
