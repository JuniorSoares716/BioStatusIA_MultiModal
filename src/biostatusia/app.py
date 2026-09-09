import json
import zipfile
from pathlib import Path

import markdown as md
from flask import Flask, jsonify, redirect, render_template, request, url_for

from biostatusia.pipeline.io_utils import (
    criar_pasta_run,
    eh_dicom,
    eh_imagem,
    eh_sinal_temporal,
    eh_tabular,
    eh_volumetrico,
    encontrar_csv,
    encontrar_todos_tabulares,
    inventariar_modalidades,
    label_pasta,
    listar_imagens,
    listar_sinais,
)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 4096 * 1024 * 1024  # 4 GB

UPLOAD_DIR = Path(__file__).parent / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.errorhandler(500)
def _erro_interno(e):
    """
    Qualquer exceção não tratada em /analisar caía na página genérica
    "Internal Server Error" do Flask (texto puro, sem nenhum detalhe) — o
    frontend não conseguia mostrar nada útil no modal de erro. Agora, para
    requisições que esperam JSON (o fluxo de upload), devolve uma mensagem
    genérica, mas acionável, em vez de travar sem explicação. O erro original
    completo ainda vai para o log do servidor (`app.logger`), para
    diagnóstico.
    """
    app.logger.exception("Erro interno não tratado")
    if request.path == "/analisar" or request.accept_mimetypes.best == "application/json":
        return jsonify({
            "erro": True, "campo": None,
            "mensagem": (
                "Ocorreu um erro interno inesperado ao processar essa base. "
                "Isso costuma acontecer com datasets em formato incomum "
                "(ex.: imagens que não são fotos clínicas normais, planilhas "
                ".xlsx em vez de .csv, ou estrutura de pastas fora do padrão "
                "esperado). Verifique o log do servidor para o detalhe técnico."
            ),
            "exemplo_correto": None,
        }), 500
    return e


# ── Detecção de modo ──────────────────────────────────────────────────────────

def detectar_estrutura(caminho: Path) -> str:
    """
    Retorna o modo de operação detectado (escopo v3 — F1/F3/F4 + tabular).
    Modos de imagem comum: imagem_unica | imagens_soltas | dataset_rotulado | multimodal
    Modos tabular/sinal:    tabular | sinal_temporal | imagem_dicom_2d | volume_3d
    Multimodal expandido:   multimodal_expandido
    Modo inválido:          invalido
    """
    if caminho.is_file():
        if eh_imagem(caminho):
            return "imagem_unica"
        if eh_tabular(caminho):
            return "tabular"
        if eh_sinal_temporal(caminho):
            return "sinal_temporal"
        if eh_dicom(caminho):
            return "imagem_dicom_2d"
        if eh_volumetrico(caminho):
            return "volume_3d"
        return "invalido"

    if not caminho.is_dir():
        return "invalido"

    # Inventariar conteúdo da pasta
    tem_tab = tem_img = tem_temporal = tem_dicom = tem_vol = False
    for arq in caminho.rglob("*"):
        if not arq.is_file():
            continue
        if eh_tabular(arq):        tem_tab = True
        elif eh_imagem(arq):       tem_img = True
        elif eh_sinal_temporal(arq): tem_temporal = True
        elif eh_dicom(arq):        tem_dicom = True
        elif eh_volumetrico(arq):  tem_vol = True

    # ── Multimodal expandido ─────────────────────────────────────────────────
    n_tipos = sum([tem_tab, tem_img, tem_temporal, tem_dicom, tem_vol])
    if n_tipos > 1 and not (tem_tab and tem_img and n_tipos == 2):
        return "multimodal_expandido"

    # ── Modos com imagem comum ───────────────────────────────────────────────
    if tem_tab and tem_img:
        return "multimodal"
    if tem_tab:
        return "tabular"

    # Verificar rótulos (subpastas benign/malignant) — aplica a qualquer família
    tem_b = tem_m = False
    for sub in caminho.rglob("*"):
        if sub.is_dir():
            label = label_pasta(sub.name)
            if label == 0:
                tem_b = True
            elif label == 1:
                tem_m = True
    if tem_b and tem_m:
        return "dataset_rotulado"

    # ── Modos de arquivo único por família ───────────────────────────────────
    if tem_img:        return "imagens_soltas"
    if tem_temporal:   return "sinal_temporal"
    if tem_vol:        return "volume_3d"
    if tem_dicom:
        # Pasta de DICOMs: muitos → série 3D, poucos → imagem 2D
        n_dcm = sum(1 for f in caminho.rglob("*.dcm") if f.is_file())
        return "volume_3d" if n_dcm >= 10 else "imagem_dicom_2d"

    return "invalido"


# ── Leitura de artefatos JSON dos agentes ─────────────────────────────────────

def _ler_json(pasta: Path, nome: str) -> dict | list | None:
    arq = pasta / nome
    if not arq.exists():
        return None
    with open(arq, "r", encoding="utf-8") as f:
        return json.load(f)


def _carregar_tabular_filtrado(caminho, coluna_alvo: str | None = None,
                                classes_selecionadas: list[str] | None = None):
    """Carrega um CSV/xlsx (via `carregar_csv`) e, se o usuário escolheu
    explicitamente comparar apenas certas classes dentro da coluna-alvo
    (`classes_selecionadas`), filtra as linhas para manter só essas —
    necessário porque o BioStatusIA hoje só treina classificação binária:
    se a coluna tiver mais de 2 valores possíveis, o usuário escolhe
    exatamente quais 2 comparar."""
    from biostatusia.pipeline.dados_tabulares import carregar_csv, filtrar_por_classes

    resultado = carregar_csv(str(caminho))
    if not resultado:
        return None
    header, data = resultado
    if coluna_alvo and coluna_alvo in header and classes_selecionadas:
        idx = header.index(coluna_alvo)
        data = filtrar_por_classes(header, data, idx, classes_selecionadas)
    return header, data



def _indices_alinhamento_previa(base_path: Path) -> dict:
    """
    Constrói os índices de alinhamento (nome de arquivo / ID normalizado)
    para TODAS as modalidades presentes na pasta, usando só listagem de
    arquivos (e leitura das tabelas) — sem extrair nenhuma feature. Rápido o
    bastante para rodar antes do usuário confirmar a análise.

    Importante: NÃO exige que a modalidade já tenha um rótulo (0/1) próprio
    para entrar no alinhamento — o alinhamento é sobre "qual arquivo/linha é
    a mesma amostra", não sobre "quem já sabe o rótulo". Datasets onde o
    rótulo só existe na tabela (não na estrutura de pastas de imagem, ex.:
    HMC-QU) dependem exatamente disso para funcionar.
    """
    from biostatusia.pipeline.alinhamento_multimodal import indice_por_arquivo, indice_tabular
    from biostatusia.pipeline.dados_tabulares import carregar_csv

    indices: dict = {}

    imagens = listar_imagens(base_path)
    if imagens:
        indices["imagem"] = indice_por_arquivo(imagens)

    sinais = listar_sinais(base_path, eh_sinal_temporal)
    if sinais:
        indices["sinal_temporal"] = indice_por_arquivo(sinais)

    dicoms = listar_sinais(base_path, eh_dicom)
    if dicoms:
        indices["dicom"] = indice_por_arquivo(dicoms)

    volumes = listar_sinais(base_path, eh_volumetrico)
    if volumes:
        indices["volume_3d"] = indice_por_arquivo(volumes)

    melhor_idx_tab, _ = _melhor_indice_tabular(base_path, indices, carregar_csv, indice_tabular)
    if melhor_idx_tab:
        indices["tabular"] = melhor_idx_tab

    return indices


def _melhor_indice_tabular(base_path: Path, indices_outras_modalidades: dict,
                            carregar_csv_fn, indice_tabular_fn) -> tuple[dict, Path | None]:
    """
    Alguns datasets trazem mais de um arquivo tabular relevante (ex.:
    HMC-QU tem A2C.xlsx e A4C.xlsx, cada um cobrindo um subconjunto
    diferente de gravações). Usar só o primeiro encontrado alfabeticamente
    pode escolher o arquivo errado — este helper tenta TODOS os arquivos
    tabulares da pasta e devolve o índice do que realmente tiver mais
    amostras em comum com as outras modalidades já identificadas.
    """
    candidatos = encontrar_todos_tabulares(base_path)
    if not candidatos:
        return {}, None

    chaves_outras = set()
    for idx in indices_outras_modalidades.values():
        chaves_outras |= set(idx.keys())

    melhor_idx: dict = {}
    melhor_arquivo: Path | None = None
    melhor_overlap = -1

    for caminho in candidatos:
        resultado = carregar_csv_fn(str(caminho))
        if not resultado:
            continue
        header, data = resultado
        idx_tab, _ = indice_tabular_fn(header, data)
        if not idx_tab:
            continue
        overlap = len(set(idx_tab.keys()) & chaves_outras) if chaves_outras else len(idx_tab)
        if overlap > melhor_overlap:
            melhor_overlap = overlap
            melhor_idx = idx_tab
            melhor_arquivo = caminho

    return melhor_idx, melhor_arquivo


def _processar_fusao_multimodal(base_path: Path, pasta_run: Path,
                                 melhor_imagem: str | None, melhor_tabular: str | None,
                                 coluna_alvo_escolhida: str | None = None,
                                 classes_selecionadas: list[str] | None = None) -> dict:
    """
    Roda depois que o pipeline de imagem (crew) já foi executado de forma
    independente. Tenta alinhar TODAS as modalidades presentes na pasta
    (imagem, tabular, sinal, DICOM, volume) e, se conseguir, calcula a fusão
    tardia (N-a-N) entre elas.

    Extração por família: imagem reaproveita `biomarcadores.json` (já escrito
    pelo agente de imagem); sinal/DICOM/volume são extraídos aqui diretamente
    via `extrair_lote_*` (não passam pelas crews específicas nesse fluxo
    multimodal genérico); tabular é extraído aqui, escolhendo — entre TODOS
    os arquivos tabulares da pasta — o que realmente tiver mais amostras em
    comum com as outras modalidades (ver `_melhor_indice_tabular`), em vez
    de assumir que existe só um arquivo tabular relevante.

    Não exige que uma modalidade já tenha rótulo (0/1) próprio para entrar
    no alinhamento — o rótulo usado para treinar pode vir de qualquer uma
    das modalidades alinhadas (ex.: HMC-QU: as imagens não têm rótulo de
    pasta, só a tabela tem).
    """
    import numpy as np
    from biostatusia.pipeline.dados_tabulares import detectar_schema, extrair_features
    from biostatusia.pipeline.alinhamento_multimodal import (
        indice_por_arquivo, indice_tabular, alinhar_modalidades,
    )
    from biostatusia.pipeline.fusao_multimodal import treinar_fusao_tardia
    from biostatusia.pipeline.classificador import _vetor as _vetor_imagem, vetorizar_biomarcadores_generico
    from biostatusia.pipeline.avaliacao_modelos import _MODELOS as _MODELOS_VALIDOS

    registros_por_modalidade: dict = {}
    indices: dict = {}

    registros_img = _ler_json(pasta_run, "biomarcadores.json") or []
    if registros_img:
        registros_por_modalidade["imagem"] = registros_img
        indices["imagem"] = indice_por_arquivo(registros_img)

    sinais = listar_sinais(base_path, eh_sinal_temporal)
    if sinais:
        from biostatusia.pipeline.extracao_temporal import extrair_lote_temporal
        extraidos = [r for r in extrair_lote_temporal(sinais) if r.get("biomarcadores")]
        if extraidos:
            registros_por_modalidade["sinal_temporal"] = extraidos
            indices["sinal_temporal"] = indice_por_arquivo(extraidos)

    dicoms = listar_sinais(base_path, eh_dicom)
    if dicoms:
        from biostatusia.pipeline.extracao_dicom import extrair_lote_dicom
        extraidos = [r for r in extrair_lote_dicom(dicoms) if r.get("biomarcadores")]
        if extraidos:
            registros_por_modalidade["dicom"] = extraidos
            indices["dicom"] = indice_por_arquivo(extraidos)

    volumes = listar_sinais(base_path, eh_volumetrico)
    if volumes:
        from biostatusia.pipeline.extracao_volumetrica import extrair_lote_volumetrico
        extraidos = [r for r in extrair_lote_volumetrico(volumes) if r.get("biomarcadores")]
        if extraidos:
            registros_por_modalidade["volume_3d"] = extraidos
            indices["volume_3d"] = indice_por_arquivo(extraidos)

    # Tabular: tenta TODOS os arquivos tabulares da pasta, usa o que tiver
    # mais amostras em comum com as outras modalidades (não assume que só
    # existe um arquivo relevante — ex.: HMC-QU tem A2C.xlsx e A4C.xlsx).
    # Se o usuário escolheu classes específicas para comparar (binário a
    # partir de uma coluna com mais de 2 valores), o filtro é aplicado AQUI
    # — antes do alinhamento — para que amostras de classes não escolhidas
    # nunca entrem na fusão nem contem no alinhamento.
    def _carregar_filtrado(caminho):
        return _carregar_tabular_filtrado(caminho, coluna_alvo_escolhida, classes_selecionadas)

    coluna_id_tab = None
    idx_tab, arquivo_tab = _melhor_indice_tabular(base_path, indices, _carregar_filtrado, indice_tabular)
    X_t, y_t = None, None
    if idx_tab and arquivo_tab:
        resultado_csv = _carregar_filtrado(arquivo_tab)
        if resultado_csv:
            header, data = resultado_csv
            _, coluna_id_tab = indice_tabular(header, data)
            label_idx_forcado = header.index(coluna_alvo_escolhida) if coluna_alvo_escolhida in header else None
            schema_t = detectar_schema(header, data, label_idx_forcado=label_idx_forcado)
            X_t, y_t, _ = extrair_features(data, schema_t)
            if X_t.size > 0:
                registros_por_modalidade["tabular"] = {"header": header, "data": data}
                indices["tabular"] = idx_tab

    alinhamento = alinhar_modalidades(indices, n_minimo=10)
    if coluna_id_tab:
        alinhamento["coluna_id_tabular"] = coluna_id_tab
    if arquivo_tab:
        alinhamento["arquivo_tabular_usado"] = arquivo_tab.name
    if not alinhamento["alinhado"]:
        return {"alinhamento_multimodal": alinhamento}

    idx_map = alinhamento["indices"]
    modalidades_specs = []
    candidatos_y: dict = {}

    for nome in alinhamento["modalidades"]:
        idxs = idx_map[nome]
        if nome == "imagem":
            X = np.array([_vetor_imagem(registros_por_modalidade["imagem"][i]["biomarcadores"]) for i in idxs])
            y_nome = np.array([registros_por_modalidade["imagem"][i].get("label") for i in idxs], dtype=object)
            modelo_nome = melhor_imagem if melhor_imagem in _MODELOS_VALIDOS else "RandomForest"
        elif nome == "tabular":
            X = X_t[idxs]
            y_nome = y_t[idxs] if y_t is not None else None
            modelo_nome = melhor_tabular if melhor_tabular in _MODELOS_VALIDOS else "LogisticRegression"
        else:
            regs = [registros_por_modalidade[nome][i] for i in idxs]
            X, _ = vetorizar_biomarcadores_generico(regs)
            y_nome = np.array([registros_por_modalidade[nome][i].get("label") for i in idxs], dtype=object)
            modelo_nome = "RandomForest"

        modalidades_specs.append({"nome": nome, "X": X, "modelo": modelo_nome})
        candidatos_y[nome] = y_nome

    # Fonte do rótulo compartilhado: se o usuário escolheu explicitamente uma
    # coluna-alvo (coluna_alvo_escolhida), o rótulo da modalidade TABULAR tem
    # prioridade máxima — é literalmente a coluna que a pessoa pediu para
    # prever. Sem essa prioridade, a escolha do usuário seria silenciosamente
    # ignorada sempre que outra modalidade (ex.: imagem com pasta
    # benign/malignant) também tivesse um rótulo próprio válido, e viesse
    # antes na ordem alfabética das modalidades.
    y_compartilhado = None
    if coluna_alvo_escolhida and candidatos_y.get("tabular") is not None:
        y_candidato = candidatos_y["tabular"]
        if not any(v is None for v in y_candidato.tolist()):
            y_compartilhado = np.asarray(y_candidato, dtype=int)

    if y_compartilhado is None:
        for nome in alinhamento["modalidades"]:
            y_candidato = candidatos_y.get(nome)
            if y_candidato is not None and not any(v is None for v in y_candidato.tolist()):
                valores_unicos = set(y_candidato.tolist())
                if len(valores_unicos) >= 2:
                    y_compartilhado = np.asarray(y_candidato, dtype=int)
                    break

    if y_compartilhado is None or len(set(y_compartilhado.tolist())) < 2:
        alinhamento["motivo_fusao_nao_calculada"] = (
            "Nenhuma das modalidades alinhadas tem rótulo válido para todas as amostras."
        )
        return {"alinhamento_multimodal": alinhamento}

    try:
        resultado_fusao = treinar_fusao_tardia(modalidades_specs, y_compartilhado)
        return {"alinhamento_multimodal": alinhamento, "fusao_multimodal": resultado_fusao}
    except Exception as e:
        return {"alinhamento_multimodal": alinhamento, "erro_fusao_multimodal": str(e)}


def _computar_engenharia_features(biomarcadores: list, estrategia: dict, limite: int = 24) -> dict | None:
    """Extrai features de engenharia de uma amostra balanceada de imagens e as ordena
    por importância. Retorna estrutura com método, ranking e médias — ou None."""
    import numpy as np
    from biostatusia.pipeline.preprocessamento import (
        engenharia_features, preprocessar_adaptativo, ranquear_features,
    )

    por_cat: dict = {}
    for r in biomarcadores:
        por_cat.setdefault(r.get("categoria", "INDEFINIDO"), []).append(r)

    cota = max(1, limite // max(1, len(por_cat)))
    amostra = [r for lista in por_cat.values() for r in lista[:cota]]
    if not amostra:
        return None

    mapa = {"BENIGNO": 0, "MALIGNO": 1}
    linhas, rotulos, nomes = [], [], None
    for r in amostra:
        caminho = r.get("caminho", "")
        if not caminho or not Path(caminho).exists():
            continue
        img = preprocessar_adaptativo(caminho, estrategia)
        if img is None:
            continue
        feats = engenharia_features(img, estrategia)
        if not feats:
            continue
        if nomes is None:
            nomes = list(feats.keys())
        linhas.append([feats.get(k, 0.0) for k in nomes])
        rotulos.append(mapa.get(r.get("categoria", "")))

    if not linhas or nomes is None:
        return None

    X = np.array(linhas, dtype=float)
    y = None
    if all(v is not None for v in rotulos) and len(set(rotulos)) >= 2:
        y = np.array(rotulos)

    ranking = ranquear_features(X, nomes, y)
    medias = {nomes[i]: round(float(X[:, i].mean()), 4) for i in range(len(nomes))}
    return {
        "metodo": ranking["metodo"],
        "ranking": ranking["ranking"],
        "medias": medias,
        "n_amostras": int(X.shape[0]),
    }


def _consolidar_imagem(pasta_run: Path, modo: str) -> dict:
    """Consolida JSONs da crew de imagem (modos originais)."""
    import numpy as np

    analise_base = _ler_json(pasta_run, "analise_base.json") or {}
    biomarcadores = _ler_json(pasta_run, "biomarcadores.json") or []
    metricas = _ler_json(pasta_run, "metricas.json") or {}

    pipeline_data: dict = {
        "modo": modo,
        "familia": "F3",
        "n_imagens": len(biomarcadores),
        "analise_base": analise_base.get("analise", {}),
        "estrategia_preproc": analise_base.get("estrategia", {}),
    }

    campos = [
        ("morfologia", "circularidade"), ("morfologia", "solidez"),
        ("textura_glcm", "contraste"), ("textura_glcm", "homogeneidade"),
        ("textura_glcm", "energia"), ("textura_glcm", "entropia"),
        ("distribuicao_intensidade", "snr"), ("distribuicao_intensidade", "assimetria"),
        ("distribuicao_intensidade", "curtose"),
    ]
    por_cat: dict = {}
    for r in biomarcadores:
        por_cat.setdefault(r["categoria"], []).append(r["biomarcadores"])

    estatisticas: dict = {}
    for cat, lista in por_cat.items():
        estatisticas[cat] = {"n": len(lista), "campos": {}}
        for grupo, campo in campos:
            valores = [b[grupo][campo] for b in lista if grupo in b and campo in b[grupo]]
            if not valores:
                continue
            arr = np.array(valores)
            estatisticas[cat]["campos"][campo] = {
                "media": round(float(arr.mean()), 4),
                "mediana": round(float(np.median(arr)), 4),
                "desvio": round(float(arr.std()), 4),
                "min": round(float(arr.min()), 4),
                "max": round(float(arr.max()), 4),
                "valores": arr.tolist(),
            }
    pipeline_data["estatisticas"] = estatisticas

    pipeline_data["biomarcadores"] = [
        {
            "caminho": r.get("caminho", ""),
            "categoria": r.get("categoria", ""),
            "biomarcadores": r.get("biomarcadores", {}),
        }
        for r in biomarcadores[:50]
    ]

    try:
        fe = _computar_engenharia_features(
            biomarcadores, pipeline_data.get("estrategia_preproc", {})
        )
        if fe:
            pipeline_data["features_engenharia"] = fe
    except Exception:
        pass

    if metricas and "metricas" in metricas:
        pipeline_data.update(metricas)
    elif metricas and "aviso" in metricas:
        pipeline_data["aviso_classificador"] = metricas["aviso"]

    return pipeline_data


def _achatar_bio_nomeado(bio: dict, prefixo: str = "") -> dict:
    """Achata um dicionário aninhado de biomarcadores em pares {nome: valor} escalares."""
    plano: dict = {}
    for grupo, conteudo in bio.items():
        if isinstance(conteudo, dict):
            for campo, val in conteudo.items():
                if isinstance(val, bool):
                    continue
                if isinstance(val, (int, float)):
                    plano[f"{campo}"] = float(val)
        elif isinstance(conteudo, bool):
            continue
        elif isinstance(conteudo, (int, float)):
            plano[grupo] = float(conteudo)
    return plano


def _estatisticas_sinal(bio_list: list, limite_campos: int = 24) -> dict:
    """Agrega os biomarcadores escalares por categoria no mesmo formato que a
    consolidação de imagem (`estatisticas`), permitindo reusar tabela e boxplot."""
    import numpy as np

    por_cat: dict = {}
    for r in bio_list:
        cat = r.get("categoria", "INDEFINIDO") or "INDEFINIDO"
        plano = _achatar_bio_nomeado(r.get("biomarcadores", {}))
        if plano:
            por_cat.setdefault(cat, []).append(plano)

    estatisticas: dict = {}
    for cat, linhas in por_cat.items():
        campos_todos: list = []
        for p in linhas:
            for k in p:
                if k not in campos_todos:
                    campos_todos.append(k)
        campos_todos = campos_todos[:limite_campos]
        entrada = {"n": len(linhas), "campos": {}}
        for campo in campos_todos:
            valores = [p[campo] for p in linhas if campo in p and np.isfinite(p[campo])]
            if not valores:
                continue
            arr = np.array(valores, dtype=float)
            entrada["campos"][campo] = {
                "media": round(float(arr.mean()), 4),
                "mediana": round(float(np.median(arr)), 4),
                "desvio": round(float(arr.std()), 4),
                "min": round(float(arr.min()), 4),
                "max": round(float(arr.max()), 4),
                "valores": arr.tolist(),
            }
        if entrada["campos"]:
            estatisticas[cat] = entrada
    return estatisticas


def _features_engenharia_sinal(bio_list: list) -> dict | None:
    """Ranqueia os biomarcadores escalares por importância (ANOVA se rotulado,
    variância caso contrário) no formato consumido pela Aba 2."""
    import numpy as np
    from biostatusia.pipeline.preprocessamento import ranquear_features

    mapa = {"BENIGNO": 0, "MALIGNO": 1}
    planos, rotulos = [], []
    nomes: list | None = None
    for r in bio_list:
        plano = _achatar_bio_nomeado(r.get("biomarcadores", {}))
        if not plano:
            continue
        if nomes is None:
            nomes = list(plano.keys())
        planos.append([plano.get(k, 0.0) for k in nomes])
        rotulos.append(mapa.get(r.get("categoria", "")))

    if not planos or nomes is None:
        return None

    X = np.array(planos, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = None
    if all(v is not None for v in rotulos) and len(set(rotulos)) >= 2:
        y = np.array(rotulos)

    ranking = ranquear_features(X, nomes, y)
    medias = {nomes[i]: round(float(X[:, i].mean()), 4) for i in range(len(nomes))}
    return {
        "metodo": ranking["metodo"],
        "ranking": ranking["ranking"],
        "medias": medias,
        "n_amostras": int(X.shape[0]),
    }


def _estrategia_preproc_sinal(familia: str, tipo: str) -> dict:
    """Estratégia de pré-processamento adaptativa e honesta para sinais (não imagem)."""
    if familia == "F1":
        return {
            "eh_sinal": True,
            "filtragem": "Passa-banda 0.5–40 Hz + Notch 50/60 Hz",
            "reamostragem": "Padronização da taxa de amostragem",
            "normalizacao": "Z-score por canal",
            "janela": "Segmentação em janelas com sobreposição",
            "justificativas": [
                "Filtragem passa-banda remove deriva de linha de base e ruído de alta frequência.",
                "Filtro notch elimina interferência da rede elétrica (50/60 Hz).",
                "Normalização z-score por canal harmoniza amplitudes entre derivações.",
            ],
        }
    if familia == "F3":
        return {
            "eh_sinal": True,
            "filtragem": "Janelamento HU (WindowCenter/Width do header DICOM)",
            "reamostragem": "Reescala para pixel spacing uniforme",
            "normalizacao": "Min-Max sobre a janela HU",
            "janela": "ROI centralizada",
            "justificativas": [
                "Janelamento HU realça o tecido de interesse conforme metadados DICOM.",
                "Normalização min-max sobre a janela preserva contraste diagnóstico.",
            ],
        }
    if familia == "F4":
        return {
            "eh_sinal": True,
            "filtragem": "Suavização 3D + clipping de percentis (P5–P95)",
            "reamostragem": "Reamostragem isotrópica de voxels",
            "normalizacao": "Z-score volumétrico",
            "janela": "Recorte do bounding box da lesão",
            "justificativas": [
                "Reamostragem isotrópica torna as métricas morfológicas comparáveis.",
                "Clipping de percentis reduz o efeito de voxels extremos (artefatos).",
            ],
        }
    return {"eh_sinal": True, "justificativas": []}


def _consolidar_sinal(pasta_run: Path, modo: str, familia: str) -> dict:
    """Consolida JSONs das crews de sinal (F1/F3/F4) e enriquece com estatísticas
    descritivas, engenharia de features e estratégia de pré-processamento — de modo
    que as Abas 1 e 2 populem para qualquer família de sinal, não só imagem/tabular."""
    _mapa_json = {
        "F1": "biomarcadores_temporal.json",
        "F3": "biomarcadores_dicom.json",
        "F4": "biomarcadores_volumetrico.json",
    }
    nome_json = _mapa_json.get(familia, "biomarcadores_temporal.json")
    payload = _ler_json(pasta_run, nome_json) or {}

    bio_list = payload.get("biomarcadores", [])
    tipo_sinal = payload.get("tipo", "")

    pipeline_data: dict = {
        "modo": modo,
        "familia": familia,
        "tipo_sinal": tipo_sinal,
        "n_imagens": payload.get("n_processados", 0),
        "n_erros": payload.get("n_erros", 0),
        "biomarcadores_sinal": bio_list,
    }

    # ── Aba 1: estatísticas descritivas dos biomarcadores (tabela + boxplot) ──
    try:
        estatisticas = _estatisticas_sinal(bio_list)
        if estatisticas:
            pipeline_data["estatisticas"] = estatisticas
    except Exception:
        pass

    # ── Aba 2: engenharia de features (ranking de importância) ────────────────
    try:
        fe = _features_engenharia_sinal(bio_list)
        if fe:
            pipeline_data["features_engenharia"] = fe
    except Exception:
        pass

    # ── Aba 2: metadados da base + estratégia de pré-processamento de sinal ────
    pipeline_data["estrategia_preproc"] = _estrategia_preproc_sinal(familia, tipo_sinal)
    primeiro = bio_list[0] if bio_list else {}
    pipeline_data["analise_base_sinal"] = {
        "n_sinais": payload.get("n_processados", 0),
        "n_erros": payload.get("n_erros", 0),
        "tipo": tipo_sinal,
        "taxa_amostragem": primeiro.get("taxa_amostragem", ""),
        "n_canais": len(primeiro.get("canais", []) or []),
    }

    # Repassar métricas se treinou classificador
    for chave in ("metricas", "metricas_cv", "roc_data", "confusion_matrix",
                  "melhor_modelo", "comparacao_ab"):
        if chave in payload:
            pipeline_data[chave] = payload[chave]

    # Dados de visualização do primeiro sinal (downsampled)
    if bio_list and "dados_viz" in bio_list[0]:
        pipeline_data["dados_viz"] = bio_list[0]["dados_viz"]
    if bio_list and "canais" in bio_list[0]:
        pipeline_data["canais"] = bio_list[0].get("canais", [])

    return pipeline_data


# ── Rotas principais ──────────────────────────────────────────────────────────

@app.route("/")
def tela1():
    return render_template("tela1_upload.html")


@app.route("/analisar", methods=["POST"])
def analisar():
    from biostatusia.crew import (
        BioStatusIACrew,
        BioStatusIACrewTabular,
        BioStatusIACrewSinal,
        BioStatusIACrewImagem3D,
    )
    from biostatusia.database import salvar, salvar_resultado
    from biostatusia.pipeline.classificador import treinar_vetores
    from biostatusia.pipeline.llm_timing import medir_llm_kickoff, MedidorEndToEnd
    from biostatusia.pipeline.dados_tabulares import (
        analisar_tabular, carregar_csv, detectar_schema, extrair_features,
        decidir_estrategia_tabular,
    )

    # 1. Resolver caminho do dataset
    medidor = MedidorEndToEnd()
    arquivo = request.files.get("arquivo")
    caminho_manual = request.form.get("caminho_manual", "").strip()
    kaggle_id = request.form.get("kaggle_id", "").strip()
    tipo_sinal_bruto = request.form.get("tipo_sinal", "auto")

    from biostatusia.validacao import validar_entrada_analise
    tamanho_arquivo = None
    if arquivo and arquivo.filename:
        arquivo.stream.seek(0, 2)
        tamanho_arquivo = arquivo.stream.tell()
        arquivo.stream.seek(0)
    validacao = validar_entrada_analise(
        arquivo_nome=arquivo.filename if arquivo else None,
        arquivo_tamanho=tamanho_arquivo,
        caminho_manual=caminho_manual, kaggle_id=kaggle_id, tipo_sinal=tipo_sinal_bruto,
    )
    if not validacao["valido"]:
        return jsonify({
            "erro": True, "campo": validacao["campo"], "mensagem": validacao["erro"],
            "exemplo_correto": validacao["exemplo_correto"],
        }), 400
    kaggle_id = validacao["kaggle_id_normalizado"]

    dataset_path = ""
    if arquivo and arquivo.filename:
        nome = arquivo.filename
        dest = UPLOAD_DIR / nome
        arquivo.save(str(dest))
        if nome.lower().endswith(".zip"):
            pasta_extracao = UPLOAD_DIR / Path(nome).stem
            pasta_extracao.mkdir(exist_ok=True)
            with zipfile.ZipFile(dest, "r") as z:
                z.extractall(pasta_extracao)
            dataset_path = str(pasta_extracao)
        else:
            dataset_path = str(dest)
    elif caminho_manual:
        dataset_path = caminho_manual
    else:
        try:
            import kagglehub
            dataset_to_download = kaggle_id if kaggle_id else "aryashah2k/breast-ultrasound-images-dataset"
            dataset_path = kagglehub.dataset_download(dataset_to_download)
        except Exception as e:
            return jsonify({
                "erro": True, "campo": "kaggle_id",
                "mensagem": (
                    f"O ID \"{kaggle_id}\" parece bem formatado, mas o KaggleHub não "
                    f"conseguiu baixá-lo ({e}). Confira se o dataset existe e é público."
                ),
                "exemplo_correto": "paultimothymooney/chest-xray-pneumonia",
            }), 400

    # 2. Detectar modo
    base_path = Path(dataset_path)
    if not base_path.exists():
        return jsonify({
            "erro": True, "campo": "caminho_manual",
            "mensagem": f"Caminho não encontrado: {dataset_path}",
            "exemplo_correto": r"C:\Users\clinica\exames\hoje",
        }), 400

    modo = detectar_estrutura(base_path)
    if modo == "invalido":
        return jsonify({
            "erro": True, "campo": None,
            "mensagem": (
                "Entrada não reconhecida — nenhum tipo de arquivo suportado foi "
                "encontrado nessa pasta/arquivo."
            ),
            "exemplo_correto": (
                "Envie imagens (.png/.jpg/.tif), CSV/TXT, ZIP, sinais (.edf/.mat/.dat/.hea), "
                "DICOM (.dcm), volume 3D (.nii/.nii.gz/.mha), ou uma pasta com qualquer "
                "combinação desses arquivos."
            ),
        }), 400

    # ── Confirmação prévia: modalidades múltiplas e/ou escolha da coluna-alvo ──
    # Antes de rodar o pipeline pesado (agentes de IA + AutoML), avisa o
    # usuário quando: (a) a base tem mais de uma modalidade (precisa saber se
    # dá para alinhar por paciente), e/ou (b) existe mais de uma coluna
    # plausível como classe-alvo na tabela — a heurística automática só olha
    # a última coluna, o que pode escolher errado (ex.: "Test Results" em vez
    # de "Medical Condition"). Só prossegue com o pipeline pesado se a pessoa
    # confirmar ("confirmado", enviado pelo modal do frontend), opcionalmente
    # já indicando a coluna-alvo escolhida ("coluna_alvo").
    confirmado = request.form.get("confirmado", "").strip() == "1"
    coluna_alvo_escolhida = request.form.get("coluna_alvo", "").strip() or None
    classes_selecionadas = [c.strip() for c in request.form.getlist("classes_selecionadas") if c.strip()] or None

    if not confirmado:
        from biostatusia.pipeline.dados_tabulares import candidatos_coluna_alvo

        precisa_multimodal = modo in ("multimodal", "multimodal_expandido")
        colunas_candidatas: list[dict] = []
        coluna_sugerida = None
        arquivo_tabular_previa = encontrar_csv(base_path)
        if arquivo_tabular_previa:
            resultado_previa = carregar_csv(str(arquivo_tabular_previa))
            if resultado_previa:
                header_previa, data_previa = resultado_previa
                colunas_candidatas = candidatos_coluna_alvo(header_previa, data_previa)
                schema_auto = detectar_schema(header_previa, data_previa)
                coluna_sugerida = schema_auto.get("label_name")

        precisa_escolher_coluna = len(colunas_candidatas) >= 2 or any(
            c["n_classes"] > 2 for c in colunas_candidatas
        )

        if precisa_multimodal or precisa_escolher_coluna:
            resposta: dict = {"precisa_confirmacao": True, "modo": modo, "dataset_path": dataset_path}
            if precisa_multimodal:
                inventario = inventariar_modalidades(base_path)
                from biostatusia.pipeline.alinhamento_multimodal import alinhar_modalidades
                indices_previa = _indices_alinhamento_previa(base_path)
                resposta["modalidades"] = inventario["modalidades"]
                resposta["alinhamento"] = alinhar_modalidades(indices_previa, n_minimo=10)
            if precisa_escolher_coluna:
                resposta["colunas_candidatas"] = colunas_candidatas
                resposta["coluna_sugerida"] = coluna_sugerida
            return jsonify(resposta)

    pasta_run = criar_pasta_run()

    # ── MODO TABULAR PURO ─────────────────────────────────────────────────────
    if modo == "tabular":
        csv_path = encontrar_csv(base_path)
        if not csv_path:
            return "Nenhum arquivo CSV/TXT encontrado.", 400
        try:
            crew_out, llm_timing = medir_llm_kickoff(lambda: BioStatusIACrewTabular().crew().kickoff(
                inputs={"caminho_csv": str(csv_path)}
            ))
            laudo = str(crew_out)
        except Exception as e:
            laudo = f"Erro no agente bioestatístico: {e}"
            llm_timing = None
        medidor.marcar("llm_laudo_pronto")

        resultado_csv = _carregar_tabular_filtrado(csv_path, coluna_alvo_escolhida, classes_selecionadas)
        if not resultado_csv:
            return "Falha ao ler o arquivo tabular.", 400
        header, data = resultado_csv
        label_idx_forcado = header.index(coluna_alvo_escolhida) if coluna_alvo_escolhida in header else None
        schema = detectar_schema(header, data, label_idx_forcado=label_idx_forcado)
        X_raw, y, label_map = extrair_features(data, schema)
        stats_tab = analisar_tabular(X_raw, y, schema)
        estrategia = decidir_estrategia_tabular(stats_tab)
        # Nota: X_raw ainda pode conter NaN aqui — de propósito. A imputação é
        # feita dentro de treinar_vetores(), ajustada apenas no fold de treino,
        # para não vazar estatística do conjunto de teste (ver imputar_treino_teste).

        pipeline_data: dict = {
            "modo": "tabular", "familia": "tabular",
            "n_imagens": stats_tab["n_amostras"],
            "arquivo_tabular": str(csv_path),
            "schema_tabular": schema, "label_map": label_map,
            "tabular_stats": stats_tab, "estrategia_preproc": estrategia,
            "llm_timing": llm_timing,
        }
        melhor = "N/A"
        n_classes_y = len(set(y.tolist())) if y is not None else 0
        if y is not None and n_classes_y >= 2 and len(X_raw) >= 10:
            try:
                res_clf = treinar_vetores(
                    X_raw, y, scaling=estrategia["escalamento"],
                    familia="tabular", feature_names=schema.get("feature_names"),
                    imputacao=estrategia.get("imputacao", "auto"),
                )
                pipeline_data.update(res_clf)
                melhor = res_clf["melhor_modelo"]
            except Exception as e:
                pipeline_data["erro_classificador"] = str(e)
        else:
            pipeline_data["aviso_classificador"] = (
                f"Treino não executado: {len(X_raw)} amostras, {n_classes_y} classes."
            )
        medidor.marcar("automl_pronto")
        pipeline_data["tempo_pipeline"] = medidor.resumo()

        analise_id = salvar(str(csv_path), "TABULAR", laudo)
        resultado_id = salvar_resultado(
            dataset_path=dataset_path, n_imagens=stats_tab["n_amostras"],
            pipeline_data=pipeline_data, melhor_modelo=melhor, analise_id=analise_id,
        )
        return redirect(url_for("tela2", resultado_id=resultado_id))

    # ── MODOS DE SINAL TEMPORAL (F1) ──────────────────────────────────────────
    if modo == "sinal_temporal":
        tipo_sinal = tipo_sinal_bruto.strip() or "auto"
        try:
            crew_out, llm_timing = medir_llm_kickoff(lambda: BioStatusIACrewSinal().crew().kickoff(inputs={
                "caminho_dataset": dataset_path,
                "pasta_run": str(pasta_run),
                "tipo_sinal": tipo_sinal,
            }))
            laudo = str(crew_out)
        except Exception as e:
            laudo = f"Erro no pipeline de sinais: {e}"
            llm_timing = None
        medidor.marcar("llm_laudo_pronto")

        pipeline_data = _consolidar_sinal(pasta_run, modo, "F1")
        pipeline_data["llm_timing"] = llm_timing
        medidor.marcar("automl_pronto")
        pipeline_data["tempo_pipeline"] = medidor.resumo()
        melhor = pipeline_data.get("melhor_modelo", "N/A")
        analise_id = salvar(dataset_path, "SINAL_TEMPORAL", laudo)
        resultado_id = salvar_resultado(
            dataset_path=dataset_path, n_imagens=pipeline_data["n_imagens"],
            pipeline_data=pipeline_data, melhor_modelo=melhor, analise_id=analise_id,
            familia_sinal="F1", sinal_tipo=pipeline_data.get("tipo_sinal", ""),
        )
        return redirect(url_for("tela2", resultado_id=resultado_id))

    # ── MODO DICOM 2D (F3) ────────────────────────────────────────────────────
    if modo == "imagem_dicom_2d":
        try:
            crew_out, llm_timing = medir_llm_kickoff(lambda: BioStatusIACrewImagem3D().crew().kickoff(inputs={
                "caminho_dataset": dataset_path,
                "pasta_run": str(pasta_run),
                "tipo_sinal": "DICOM 2D",
            }))
            laudo = str(crew_out)
        except Exception as e:
            laudo = f"Erro no pipeline DICOM: {e}"
            llm_timing = None
        medidor.marcar("llm_laudo_pronto")

        pipeline_data = _consolidar_sinal(pasta_run, modo, "F3")
        pipeline_data["llm_timing"] = llm_timing
        medidor.marcar("automl_pronto")
        pipeline_data["tempo_pipeline"] = medidor.resumo()
        melhor = pipeline_data.get("melhor_modelo", "N/A")
        analise_id = salvar(dataset_path, "DICOM_2D", laudo)
        resultado_id = salvar_resultado(
            dataset_path=dataset_path, n_imagens=pipeline_data["n_imagens"],
            pipeline_data=pipeline_data, melhor_modelo=melhor, analise_id=analise_id,
            familia_sinal="F3", sinal_tipo=pipeline_data.get("tipo_sinal", ""),
        )
        return redirect(url_for("tela2", resultado_id=resultado_id))

    # ── MODO VOLUME 3D (F4) ────────────────────────────────────────────────────
    if modo == "volume_3d":
        try:
            crew_out, llm_timing = medir_llm_kickoff(lambda: BioStatusIACrewImagem3D().crew().kickoff(inputs={
                "caminho_dataset": dataset_path,
                "pasta_run": str(pasta_run),
                "tipo_sinal": "Volume 3D",
            }))
            laudo = str(crew_out)
        except Exception as e:
            laudo = f"Erro no pipeline volumétrico: {e}"
            llm_timing = None
        medidor.marcar("llm_laudo_pronto")

        pipeline_data = _consolidar_sinal(pasta_run, modo, "F4")
        pipeline_data["llm_timing"] = llm_timing
        medidor.marcar("automl_pronto")
        pipeline_data["tempo_pipeline"] = medidor.resumo()
        melhor = pipeline_data.get("melhor_modelo", "N/A")
        analise_id = salvar(dataset_path, "VOLUME_3D", laudo)
        resultado_id = salvar_resultado(
            dataset_path=dataset_path, n_imagens=pipeline_data["n_imagens"],
            pipeline_data=pipeline_data, melhor_modelo=melhor, analise_id=analise_id,
            familia_sinal="F4", sinal_tipo=pipeline_data.get("tipo_sinal", ""),
        )
        return redirect(url_for("tela2", resultado_id=resultado_id))

    # ── MODOS COM IMAGEM ORIGINAL (imagem_unica, imagens_soltas, dataset_rotulado, multimodal[_expandido]) ──
    imagens = listar_imagens(base_path)

    if not imagens and modo != "multimodal_expandido":
        return "Nenhuma imagem válida encontrada.", 400

    if imagens:
        try:
            crew_out, llm_timing = medir_llm_kickoff(lambda: BioStatusIACrew().crew().kickoff(inputs={
                "caminho_dataset": dataset_path,
                "pasta_run": str(pasta_run),
            }))
            laudo = str(crew_out)
        except Exception as e:
            laudo = f"Erro no pipeline CrewAI: {e}"
            llm_timing = None
        medidor.marcar("llm_laudo_pronto")
        try:
            pipeline_data = _consolidar_imagem(pasta_run, modo)
        except Exception as e:
            pipeline_data = {
                "modo": modo, "familia": "imagem", "n_imagens": len(imagens),
                "erro_consolidacao": (
                    f"Falha ao consolidar os resultados da análise de imagem: {e}. "
                    f"Isso costuma acontecer com imagens fora do padrão esperado "
                    f"(ex.: máscaras de segmentação em vez de fotos clínicas, ou "
                    f"pastas sem a convenção benign/malignant para o rótulo)."
                ),
            }
        pipeline_data["llm_timing"] = llm_timing
        melhor = pipeline_data.get("melhor_modelo", "N/A")
    else:
        # multimodal_expandido sem nenhuma imagem (ex.: sinal + DICOM) — não
        # quebra mais: usa a crew de Sinal para o laudo narrativo (o mais
        # genérico disponível) e monta o pipeline_data manualmente. A
        # classificação/fusão de cada modalidade acontece via
        # `_processar_fusao_multimodal`, abaixo, independente de qual crew
        # gerou o texto do laudo.
        try:
            crew_out, llm_timing = medir_llm_kickoff(lambda: BioStatusIACrewSinal().crew().kickoff(inputs={
                "caminho_dataset": dataset_path,
                "pasta_run": str(pasta_run),
                "tipo_sinal": "auto",
            }))
            laudo = str(crew_out)
        except Exception as e:
            laudo = f"Erro no pipeline CrewAI: {e}"
            llm_timing = None
        medidor.marcar("llm_laudo_pronto")
        pipeline_data = {"modo": modo, "familia": "multimodal_expandido", "n_imagens": 0, "llm_timing": llm_timing}
        melhor = "N/A"

    if modo in ("multimodal", "multimodal_expandido"):
        csv_path = encontrar_csv(base_path)
        X_t, y_t, modelo_tabular_nome = None, None, None
        if csv_path:
            resultado_csv = _carregar_tabular_filtrado(csv_path, coluna_alvo_escolhida, classes_selecionadas)
            if resultado_csv:
                header_t, data_t = resultado_csv
                label_idx_forcado_t = header_t.index(coluna_alvo_escolhida) if coluna_alvo_escolhida in header_t else None
                schema_t = detectar_schema(header_t, data_t, label_idx_forcado=label_idx_forcado_t)
                X_t, y_t, label_map_t = extrair_features(data_t, schema_t)
                if X_t.size > 0:
                    pipeline_data["tabular_stats"] = analisar_tabular(X_t, y_t, schema_t)
                    pipeline_data["schema_tabular"] = schema_t
                    pipeline_data["label_map"] = label_map_t
                    pipeline_data["arquivo_tabular"] = str(csv_path)
                    n_classes_yt = len(set(y_t.tolist())) if y_t is not None else 0
                    if y_t is not None and n_classes_yt >= 2 and len(X_t) >= 10:
                        try:
                            res_tab = treinar_vetores(X_t, y_t)
                            pipeline_data["metricas_tabular"] = res_tab["metricas"]
                            pipeline_data["roc_data_tabular"] = res_tab["roc_data"]
                            pipeline_data["cm_tabular"] = res_tab["confusion_matrix"]
                            pipeline_data["melhor_modelo_tabular"] = res_tab["melhor_modelo"]
                            pipeline_data["piso_sensibilidade_atingido_tabular"] = res_tab.get("piso_sensibilidade_atingido")
                            pipeline_data["aviso_piso_sensibilidade_tabular"] = res_tab.get("aviso_piso_sensibilidade")
                            pipeline_data["criterio_selecao_tabular"] = res_tab.get("criterio_selecao")
                            modelo_tabular_nome = res_tab["melhor_modelo"]
                        except Exception as e:
                            pipeline_data["erro_classificador_tabular"] = str(e)

        # ── Fusão multimodal de verdade (tardia, N-a-N) ─────────────────────
        # Tenta alinhar TODAS as modalidades presentes (imagem, tabular,
        # sinal, DICOM, volume) — não só imagem+tabular — e só calcula a
        # fusão entre as que realmente conseguirem ser casadas pela mesma
        # amostra/paciente (ver alinhamento_multimodal.py).
        try:
            resultado_fusao = _processar_fusao_multimodal(
                base_path, pasta_run, melhor, modelo_tabular_nome,
                coluna_alvo_escolhida, classes_selecionadas,
            )
            pipeline_data.update(resultado_fusao)
        except Exception as e:
            pipeline_data["erro_fusao_multimodal"] = str(e)

    medidor.marcar("automl_pronto")
    pipeline_data["tempo_pipeline"] = medidor.resumo()

    if imagens:
        primeira = imagens[0]
        analise_id = salvar(primeira["caminho"], primeira["categoria"], laudo)
    else:
        analise_id = salvar(dataset_path, "MULTIMODAL_EXPANDIDO", laudo)
    resultado_id = salvar_resultado(
        dataset_path=dataset_path, n_imagens=pipeline_data.get("n_imagens", 0),
        pipeline_data=pipeline_data, melhor_modelo=melhor, analise_id=analise_id,
    )
    return redirect(url_for("tela2", resultado_id=resultado_id))


@app.route("/resultados/<int:resultado_id>")
def tela2(resultado_id: int):
    from biostatusia.database import buscar_resultado, listar

    dados = buscar_resultado(resultado_id)
    if not dados:
        return "Resultado não encontrado.", 404

    pipeline = dados["pipeline"]
    laudo_html = md.markdown(dados["laudo"] or "")
    historico = listar()

    return render_template(
        "tela2_resultados.html",
        dados=dados,
        pipeline=pipeline,
        pipeline_json=json.dumps(pipeline),
        laudo_html=laudo_html,
        historico=historico,
    )


# ── Rota: Página de Histórico de Análises ────────────────────────────────────

@app.route("/historico")
def historico():
    """Página dedicada com a lista de análises anteriores (design Stitch)."""
    from biostatusia.database import listar_resultados_completo

    resultados = listar_resultados_completo(limite=200)
    total = len(resultados)
    familias = sorted({r["familia_sinal"] for r in resultados if r["familia_sinal"]})
    return render_template(
        "tela3_historico.html",
        resultados=resultados,
        total=total,
        familias=familias,
    )


# ── Rota: Histórico para Aba 4 ───────────────────────────────────────────────

@app.route("/api/historico")
def api_historico():
    """Retorna JSON com os últimos 30 resultados do banco para o seletor da Aba 4."""
    from biostatusia.database import listar_resultados_completo
    return jsonify(listar_resultados_completo())


@app.route("/api/exemplos/<int:resultado_id>")
def api_exemplos(resultado_id: int):
    """Exemplos individuais do dataset carregado nesta análise (para a Aba 4)."""
    from biostatusia.database import buscar_resultado

    dados = buscar_resultado(resultado_id)
    if not dados:
        return jsonify({"erro": f"Resultado {resultado_id} não encontrado"}), 404

    pipeline = dados["pipeline"]
    bio_list = pipeline.get("biomarcadores_sinal") or pipeline.get("biomarcadores") or []
    familia = pipeline.get("familia", "")
    tipo = pipeline.get("tipo_sinal", "")

    exemplos = []
    for i, r in enumerate(bio_list[:50]):
        caminho = r.get("caminho", "")
        nome = Path(caminho).name if caminho else f"exemplo_{i}"
        exemplos.append({
            "idx": i,
            "nome": nome,
            "categoria": r.get("categoria", ""),
            "biomarcadores": r.get("biomarcadores", {}),
            "familia": familia,
            "tipo": tipo,
        })

    return jsonify(exemplos)


# ── Via 1: Laudo Populacional (nível da base) ─────────────────────────────────

@app.route("/laudo_populacional/<int:resultado_id>")
def laudo_populacional(resultado_id: int):
    """
    Relatório analítico determinístico do dataset completo: distribuição
    estatística, correlações de biomarcadores e pódio final do AutoML.
    Não depende do LLM — montado a partir do pipeline_data persistido.
    """
    from biostatusia.database import buscar_resultado
    from biostatusia.pipeline.relatorios import (
        construir_laudo_populacional, construir_podio, correlacoes_biomarcadores,
    )

    dados = buscar_resultado(resultado_id)
    if not dados:
        return jsonify({"erro": f"Resultado {resultado_id} não encontrado"}), 404

    pipeline = dict(dados["pipeline"])

    # Enriquecer com correlações calculadas sobre os biomarcadores de sinal, se houver.
    bio_list = pipeline.get("biomarcadores_sinal", [])
    if bio_list:
        from biostatusia.pipeline.inferencia import achatar_biomarcadores
        vetores, nomes = [], []
        for r in bio_list:
            v = achatar_biomarcadores(r.get("biomarcadores", {}))
            if v.size:
                vetores.append(v.tolist())
        if vetores:
            largura = min(len(x) for x in vetores)
            vetores = [x[:largura] for x in vetores]
            nomes = [f"f{i}" for i in range(largura)]
            pipeline["correlacoes_biomarcadores"] = correlacoes_biomarcadores(vetores, nomes)

    laudo_md = construir_laudo_populacional(pipeline)
    podio = construir_podio(pipeline.get("metricas", {}), pipeline.get("melhor_modelo", ""))
    return jsonify({
        "laudo_html": md.markdown(laudo_md, extensions=["tables"]),
        "laudo_md": laudo_md,
        "podio": podio,
    })


# ── Rota: Laudo de Amostra Avulsa ─────────────────────────────────────────────

@app.route("/laudo_amostra", methods=["POST"])
def laudo_amostra():
    """
    Recebe um arquivo avulso (multipart) OU um resultado_id do banco e
    gera laudo do radiologista_ia_interativo sem executar o pipeline completo.

    Casos de uso da Aba 4 — Seção A:
      - Upload de nova amostra (arquivo de imagem, sinal, DICOM, etc.)
      - Seleção de análise anterior pelo resultado_id
    """
    from biostatusia.database import buscar_resultado, salvar_laudo_interativo
    import markdown as md_

    resultado_id_str = request.form.get("resultado_id", "").strip()
    arquivo = request.files.get("arquivo")

    # ── Caso 1: análise já existente no banco ──────────────────────────────
    if resultado_id_str and resultado_id_str.isdigit():
        resultado_id = int(resultado_id_str)
        dados = buscar_resultado(resultado_id)
        if not dados:
            return jsonify({"erro": f"Resultado {resultado_id} não encontrado"}), 404

        pipeline = dados["pipeline"]
        familia = pipeline.get("familia", "")
        tipo_sinal = pipeline.get("tipo_sinal", "") or dados.get("categoria", "")

        # Resumo do pipeline como contexto para o agente
        biomarcadores_ctx = {
            "fonte": "banco",
            "resultado_id": resultado_id,
            "familia": familia,
            "tipo_sinal": tipo_sinal,
            "modo": pipeline.get("modo", ""),
            "n_amostras": pipeline.get("n_imagens", 0),
            "melhor_modelo": dados.get("melhor_modelo", "N/A"),
        }
        # Anexa métricas se existirem
        if pipeline.get("metricas"):
            melhor = dados.get("melhor_modelo", "")
            m = pipeline["metricas"].get(melhor, {})
            biomarcadores_ctx["metricas_modelo"] = {
                "auc": m.get("auc"),
                "sensibilidade": m.get("sensibilidade"),
                "especificidade": m.get("especificidade"),
                "f1": m.get("f1"),
            }
        # Anexa biomarcadores do exemplo selecionado (ou o primeiro, por padrão)
        bio_list = pipeline.get("biomarcadores_sinal") or pipeline.get("biomarcadores") or []
        exemplo_idx_str = request.form.get("exemplo_idx", "").strip()
        exemplo_nome = ""
        if bio_list:
            idx = 0
            if exemplo_idx_str.isdigit() and 0 <= int(exemplo_idx_str) < len(bio_list):
                idx = int(exemplo_idx_str)
            exemplo = bio_list[idx]
            biomarcadores_ctx["exemplo_biomarcadores"] = exemplo.get("biomarcadores", {})
            biomarcadores_ctx["exemplo_idx"] = idx
            caminho_ex = exemplo.get("caminho", "")
            exemplo_nome = Path(caminho_ex).name if caminho_ex else f"exemplo_{idx}"
            biomarcadores_ctx["exemplo_nome"] = exemplo_nome
            biomarcadores_ctx["categoria_exemplo"] = exemplo.get("categoria", "")

        if exemplo_nome:
            descricao_selecao = (
                f"Exemplo '{exemplo_nome}' (#{biomarcadores_ctx.get('exemplo_idx', 0)}) "
                f"do resultado #{resultado_id} — {dados.get('dataset_path', '')}"
            )
        else:
            descricao_selecao = f"Análise do resultado #{resultado_id} — {dados.get('dataset_path', '')}"

    # ── Caso 2: upload de arquivo avulso ──────────────────────────────────
    elif arquivo and arquivo.filename:
        from biostatusia.validacao import validar_arquivo_upload
        arquivo.stream.seek(0, 2)
        tamanho = arquivo.stream.tell()
        arquivo.stream.seek(0)
        r_validacao = validar_arquivo_upload(arquivo.filename, tamanho)
        if not r_validacao["valido"]:
            return jsonify({
                "erro": True, "campo": "arquivo",
                "mensagem": r_validacao["erro"], "exemplo_correto": r_validacao["exemplo_correto"],
            }), 400

        nome = arquivo.filename
        dest = UPLOAD_DIR / nome
        arquivo.save(str(dest))

        # Detecta estrutura do arquivo
        modo = detectar_estrutura(dest)
        familia = ""
        tipo_sinal = ""

        biomarcadores_ctx = {
            "fonte": "upload_avulso",
            "arquivo": nome,
            "modo": modo,
            "familia": familia,
        }

        pasta_run = criar_pasta_run()

        try:
            if eh_imagem(dest):
                from biostatusia.pipeline.extracao import extrair_todos
                bio = extrair_todos(str(dest))
                biomarcadores_ctx["biomarcadores"] = bio or {}
                # Imagens comuns são treinadas via treinar_vetores() sem um
                # `familia` explícito, que persiste sob "IMG" (ver
                # classificador.py) — não "F3" (essa é a família do DICOM).
                familia = "IMG"

            elif eh_sinal_temporal(dest):
                from biostatusia.pipeline.io_sinais import carregar_sinal
                from biostatusia.pipeline.extracao_temporal import extrair_features_temporal
                sinal = carregar_sinal(str(dest))
                feats = extrair_features_temporal(sinal)
                biomarcadores_ctx["biomarcadores"] = feats
                tipo_sinal = sinal.tipo
                # O treino de sinal persiste sob o TIPO detectado (ex.: "ECG",
                # "EEG"), não sob "F1" fixo — ver sinais_temporais_tool.py.
                familia = tipo_sinal or "F1"

            elif eh_dicom(dest):
                from biostatusia.pipeline.extracao_dicom import extrair_features_dicom
                feats = extrair_features_dicom(dest)
                biomarcadores_ctx["biomarcadores"] = feats
                familia = "F3"
                tipo_sinal = "DICOM"

            elif eh_volumetrico(dest):
                from biostatusia.pipeline.io_sinais import carregar_sinal
                from biostatusia.pipeline.extracao_volumetrica import extrair_biomarcadores_volumetrico
                sinal_vol = carregar_sinal(str(dest))
                feats = extrair_biomarcadores_volumetrico(sinal_vol)
                biomarcadores_ctx["biomarcadores"] = feats or {}
                familia = "F4"
                tipo_sinal = "Volume 3D"

            elif eh_tabular(dest):
                from biostatusia.pipeline.dados_tabulares import (
                    carregar_csv, detectar_schema, extrair_features, analisar_tabular
                )
                res = carregar_csv(str(dest))
                if res:
                    header, data = res
                    schema = detectar_schema(header, data)
                    X, y, _ = extrair_features(data, schema)
                    stats = analisar_tabular(X, y, schema)
                    biomarcadores_ctx["tabular_stats"] = {
                        "n_amostras": stats.get("n_amostras", 0),
                        "n_features": stats.get("n_features", 0),
                    }

        except Exception as e:
            biomarcadores_ctx["erro_extracao"] = str(e)

        biomarcadores_ctx["familia"] = familia
        biomarcadores_ctx["tipo_sinal"] = tipo_sinal

        # ── Inferência individual: isola o vencedor do pódio e classifica ──
        bio_extraido = biomarcadores_ctx.get("biomarcadores")
        if isinstance(bio_extraido, dict) and bio_extraido:
            try:
                from biostatusia.pipeline.inferencia import (
                    achatar_biomarcadores, prever_exemplar,
                )
                vetor = achatar_biomarcadores(bio_extraido)
                if vetor.size:
                    biomarcadores_ctx["inferencia_modelo"] = prever_exemplar(
                        vetor, familia or "IMG"
                    )
            except Exception as e:
                biomarcadores_ctx["inferencia_erro"] = str(e)

        descricao_selecao = f"Amostra avulsa: {nome} (modo: {modo})"

    else:
        return jsonify({"erro": "Forneça resultado_id ou um arquivo para análise."}), 400

    # ── Chama o agente interativo ──────────────────────────────────────────
    from biostatusia.crew import BioStatusIACrewInterativo
    from biostatusia.pipeline.llm_timing import medir_llm_kickoff
    try:
        crew_out, llm_timing = medir_llm_kickoff(lambda: BioStatusIACrewInterativo().crew().kickoff(inputs={
            "dados_selecao": json.dumps({"descricao": descricao_selecao}, ensure_ascii=False),
            "tipo_sinal": biomarcadores_ctx.get("tipo_sinal", ""),
            "biomarcadores_trecho": json.dumps(biomarcadores_ctx, ensure_ascii=False),
        }))
        laudo_foco = str(crew_out)
    except Exception as e:
        laudo_foco = f"Erro ao gerar laudo: {e}"
        llm_timing = None

    laudo_id = salvar_laudo_interativo(
        resultado_id=int(resultado_id_str) if resultado_id_str and resultado_id_str.isdigit() else 0,
        laudo_foco=laudo_foco,
    )

    return jsonify({
        "laudo_html": md_.markdown(laudo_foco),
        "laudo_id": laudo_id,
        "llm_timing": llm_timing,
        "inferencia_modelo": biomarcadores_ctx.get("inferencia_modelo"),
        "inferencia_erro": biomarcadores_ctx.get("inferencia_erro"),
    })



if __name__ == "__main__":
    app.run(debug=True, port=5000)
