"""
Alinhamento amostra-a-amostra entre N modalidades quaisquer (BioStatusIA).

A fusão multimodal (`pipeline/fusao_multimodal.py`) exige que a amostra `i`
de cada modalidade seja da MESMA amostra/paciente/exame. Este módulo resolve
esse pré-requisito de forma genérica — não apenas para imagem + tabular, mas
para qualquer combinação de 2 a 5 modalidades (imagem, tabular, sinal
temporal, DICOM, volume 3D) presentes numa mesma base.

Estratégia: cada modalidade baseada em arquivo (imagem, sinal, DICOM,
volume) usa o próprio nome do arquivo (normalizado) como identificador —
então dois arquivos com o mesmo nome-base em modalidades diferentes (ex.:
"paciente_007.edf" e "paciente_007.dcm") já se alinham sozinhos, sem
precisar de nenhuma coluna extra. A modalidade tabular usa uma coluna de
identificador reconhecida no CSV (`detectar_coluna_id`) para o mesmo fim.

O alinhamento final é a INTERSECÇÃO das chaves presentes em todas as
modalidades participantes — só entram na fusão as amostras que aparecem em
TODAS elas. Modalidades sem identificador utilizável (ex.: tabular sem
coluna de ID) ficam de fora do alinhamento, mas isso é reportado
explicitamente — nunca fundido às cegas.
"""
from pathlib import Path

ID_KEYWORDS = frozenset({
    "id", "filename", "file", "arquivo", "file_name", "filepath",
    "image", "imagem", "image_id", "img_id", "image_name", "imagename",
    "patient_id", "paciente_id", "id_paciente", "subject_id",
    "nome_arquivo", "nome", "caminho",
    "echo", "recording", "recording_id", "exam_id", "exame_id", "study_id",
    "video", "video_id",
})


def _normalizar_nome(valor: str) -> str:
    """Normaliza um nome de arquivo/ID para comparação: remove diretório,
    extensão, espaços e diferenças de maiúsculas/minúsculas."""
    if valor is None:
        return ""
    nome = Path(str(valor).strip().strip("'\"")).stem
    return nome.lower().strip()


def detectar_coluna_id(header: list[str]) -> int | None:
    """Detecta qual coluna do CSV serve como identificador/nome de arquivo
    para casar com as demais modalidades. Só considera nomes de coluna
    reconhecíveis — não tenta adivinhar por conteúdo, para evitar falsos
    positivos."""
    for i, nome in enumerate(header):
        chave = nome.lower().strip().replace(" ", "_").replace("-", "_")
        if chave in ID_KEYWORDS:
            return i
    return None


def indice_por_arquivo(registros: list[dict]) -> dict[str, int]:
    """Mapa nome-de-arquivo-normalizado -> índice, para qualquer modalidade
    baseada em arquivo (imagem, sinal temporal, DICOM, volume 3D). Todos os
    `listar_*`/`extrair_lote_*` do BioStatusIA produzem registros com campo
    "caminho", então essa função serve para as 4 famílias baseadas em arquivo."""
    mapa: dict[str, int] = {}
    for i, registro in enumerate(registros):
        chave = _normalizar_nome(registro.get("caminho", ""))
        if chave:
            mapa[chave] = i
    return mapa


def indice_tabular(header: list[str], data: list[list[str]]) -> tuple[dict[str, int], str | None]:
    """Mapa ID-normalizado -> índice da linha do CSV, usando a coluna de
    identificador detectada. Retorna (mapa_vazio, None) se nenhuma coluna
    reconhecível existir — sinaliza que a modalidade tabular não pode
    participar do alinhamento por falta de um identificador comum."""
    idx_id = detectar_coluna_id(header)
    if idx_id is None:
        return {}, None
    mapa: dict[str, int] = {}
    for i, row in enumerate(data):
        if idx_id < len(row):
            chave = _normalizar_nome(row[idx_id])
            if chave:
                mapa[chave] = i
    return mapa, header[idx_id]


def alinhar_modalidades(indices_por_modalidade: dict[str, dict[str, int]],
                         n_minimo: int = 10) -> dict:
    """
    Alinha N modalidades pela intersecção de identificadores em comum.

    `indices_por_modalidade`: {"imagem": {chave: idx}, "tabular": {chave: idx},
    "sinal_temporal": {chave: idx}, ...} — inclua todas as modalidades
    presentes, mesmo as sem índice (dict vazio) — elas aparecem em
    "modalidades_excluidas" no resultado.

    Retorna:
      - "alinhado": bool
      - "modalidades": lista de nomes das modalidades alinhadas (>= 2)
      - "modalidades_excluidas": nomes que não puderam entrar (sem índice)
      - "indices": {nome_modalidade: [índices na ordem alinhada]}
      - "n_pareados": quantas amostras em comum
      - "motivo": presente quando "alinhado" é False
    """
    utilizaveis = {nome: idx for nome, idx in indices_por_modalidade.items() if idx}
    excluidas = [nome for nome, idx in indices_por_modalidade.items() if not idx]

    if len(utilizaveis) < 2:
        return {
            "alinhado": False,
            "modalidades_excluidas": excluidas,
            "motivo": (
                f"Apenas {len(utilizaveis)} modalidade(s) têm um identificador "
                f"utilizável para alinhamento (mínimo: 2). Modalidades sem "
                f"identificador reconhecível: {', '.join(excluidas) or 'nenhuma'}."
            ),
        }

    nomes = sorted(utilizaveis.keys())
    chaves_comuns = set(utilizaveis[nomes[0]].keys())
    for nome in nomes[1:]:
        chaves_comuns &= set(utilizaveis[nome].keys())

    if len(chaves_comuns) < n_minimo:
        return {
            "alinhado": False,
            "modalidades_excluidas": excluidas,
            "motivo": (
                f"Apenas {len(chaves_comuns)} amostra(s) em comum entre "
                f"{', '.join(nomes)} — mínimo necessário para a fusão é {n_minimo}. "
                f"Confira se os nomes de arquivo (ou a coluna de ID) realmente "
                f"correspondem à mesma pessoa em todas as modalidades."
            ),
        }

    chaves_ordenadas = sorted(chaves_comuns)
    indices = {nome: [utilizaveis[nome][c] for c in chaves_ordenadas] for nome in nomes}

    return {
        "alinhado": True,
        "modalidades": nomes,
        "modalidades_excluidas": excluidas,
        "indices": indices,
        "n_pareados": len(chaves_ordenadas),
    }
