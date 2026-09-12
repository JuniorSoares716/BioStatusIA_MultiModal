"""
Gera um relatório PDF único, combinando os dados de todas as abas da tela
de resultados (Estatísticas & Biomarcadores, Pré-processamento, AutoML,
Fusão Multimodal e Laudo) — para download/impressão fora da interface web.

Defensivo por design: cada seção só aparece se os dados correspondentes
existirem no `pipeline_data` daquela análise (que varia bastante conforme o
modo — imagem, tabular, sinal, multimodal etc.), igual ao comportamento já
adotado na tela de resultados em si.
"""
import re
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    HRFlowable, ListFlowable, ListItem, KeepTogether,
)

_TEAL = colors.HexColor("#0d9488")
_TEAL_DARK = colors.HexColor("#0f766e")
_TEAL_LIGHT = colors.HexColor("#f0fdfa")
_AMBER = colors.HexColor("#b45309")
_AMBER_LIGHT = colors.HexColor("#fffbeb")
_SLATE = colors.HexColor("#334155")
_SLATE_LIGHT = colors.HexColor("#f8fafc")


def _estilos():
    base = getSampleStyleSheet()
    estilos = {
        "titulo": ParagraphStyle("titulo", parent=base["Title"], textColor=_TEAL_DARK,
                                  fontSize=20, spaceAfter=4),
        "subtitulo": ParagraphStyle("subtitulo", parent=base["Normal"], textColor=_SLATE,
                                     fontSize=10, spaceAfter=14),
        "secao": ParagraphStyle("secao", parent=base["Heading1"], textColor=_TEAL_DARK,
                                 fontSize=14, spaceBefore=18, spaceAfter=8,
                                 borderColor=_TEAL, borderWidth=0, borderPadding=0),
        "subsecao": ParagraphStyle("subsecao", parent=base["Heading2"], textColor=_SLATE,
                                    fontSize=11.5, spaceBefore=10, spaceAfter=6),
        "corpo": ParagraphStyle("corpo", parent=base["Normal"], fontSize=9.5, leading=13.5,
                                 textColor=_SLATE, spaceAfter=6),
        "corpo_bold": ParagraphStyle("corpo_bold", parent=base["Normal"], fontSize=9.5,
                                      leading=13.5, textColor=_SLATE, fontName="Helvetica-Bold"),
        "aviso": ParagraphStyle("aviso", parent=base["Normal"], fontSize=9, leading=13,
                                 textColor=_AMBER, spaceAfter=6),
        "rotulo": ParagraphStyle("rotulo", parent=base["Normal"], fontSize=7.5,
                                  textColor=colors.HexColor("#94a3b8"), spaceAfter=1),
        "valor": ParagraphStyle("valor", parent=base["Normal"], fontSize=11,
                                 textColor=_SLATE, fontName="Helvetica-Bold"),
        "rodape": ParagraphStyle("rodape", parent=base["Normal"], fontSize=7.5,
                                  textColor=colors.HexColor("#94a3b8"), alignment=TA_CENTER),
        "laudo_h": ParagraphStyle("laudo_h", parent=base["Heading3"], textColor=_TEAL_DARK,
                                   fontSize=11, spaceBefore=8, spaceAfter=4),
        "laudo_p": ParagraphStyle("laudo_p", parent=base["Normal"], fontSize=9.5, leading=14,
                                   textColor=_SLATE, spaceAfter=6),
    }
    return estilos


def _fmt_pct(v, casas=1):
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.{casas}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(v, casas=3):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{casas}f}"
    except (TypeError, ValueError):
        return str(v)


def _tabela_estilo_base(cor_cabecalho=_TEAL):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), cor_cabecalho),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _SLATE_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ])


def _markdown_para_flowables(texto: str, estilos: dict) -> list:
    """Conversão simples e defensiva de markdown -> flowables do reportlab
    (não usa um parser de markdown completo; cobre o que os laudos do
    BioStatusIA realmente usam: #/##/### para títulos, **negrito**, listas
    com "-"). Texto fora desse padrão vira parágrafo normal."""
    if not texto:
        return [Paragraph("Laudo não disponível.", estilos["corpo"])]

    flowables = []
    buffer_lista: list[str] = []

    def _fecha_lista():
        if buffer_lista:
            itens = [ListItem(Paragraph(_negrito(li), estilos["laudo_p"]), leftIndent=12) for li in buffer_lista]
            flowables.append(ListFlowable(itens, bulletType="bullet", start="-", leftIndent=10))
            buffer_lista.clear()

    def _negrito(linha: str) -> str:
        # **texto** -> <b>texto</b>; escapa < e > soltos que não fazem parte disso
        linha = linha.replace("&", "&amp;")
        linha = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", linha)
        linha = re.sub(r"(?<!<)/(?!b>)", "/", linha)  # no-op seguro, mantém barras normais
        return linha

    for linha_bruta in texto.splitlines():
        linha = linha_bruta.strip()
        if not linha or linha == "```markdown" or linha == "```":
            _fecha_lista()
            continue
        if linha.startswith("#"):
            _fecha_lista()
            titulo = linha.lstrip("#").strip()
            flowables.append(Paragraph(_negrito(titulo), estilos["laudo_h"]))
            continue
        if linha.startswith(("- ", "* ")):
            buffer_lista.append(linha[2:].strip())
            continue
        _fecha_lista()
        flowables.append(Paragraph(_negrito(linha), estilos["laudo_p"]))

    _fecha_lista()
    return flowables or [Paragraph("Laudo não disponível.", estilos["corpo"])]


def gerar_relatorio_pdf(resultado: dict) -> bytes:
    """
    Monta o relatório PDF final a partir do dict retornado por
    `database.buscar_resultado()` — combina todas as abas da tela de
    resultados num único documento.
    """
    p = resultado.get("pipeline") or {}
    estilos = _estilos()
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        title="Relatório BioStatusIA",
    )
    story: list = []

    # ── Cabeçalho ────────────────────────────────────────────────────────
    story.append(Paragraph("BioStatusIA — Relatório Final de Análise", estilos["titulo"]))
    dataset_nome = str(resultado.get("dataset_path", "—")).replace("\\", "/").split("/")[-1]
    story.append(Paragraph(
        f"Gerado em {resultado.get('data_hora', '—')} &nbsp;-&nbsp; Base: {dataset_nome}",
        estilos["subtitulo"],
    ))

    info_cabecalho = [
        ["Modo de operação", str(p.get("modo", p.get("familia", "—"))).replace("_", " ").title()],
        ["Amostras", str(resultado.get("n_imagens", p.get("n_amostras", "—")))],
        ["Melhor modelo", str(resultado.get("melhor_modelo", "N/A"))],
    ]
    if p.get("classes"):
        info_cabecalho.append(["Classes", ", ".join(str(c) for c in p["classes"])])
    t = Table(info_cabecalho, colWidths=[5 * cm, 11 * cm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), _SLATE),
        ("TEXTCOLOR", (1, 0), (1, -1), _SLATE),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>AVISO:</b> Esta é uma ferramenta de suporte à decisão baseada em IA e não substitui o "
        "diagnóstico clínico realizado por um médico habilitado.",
        estilos["aviso"],
    ))
    story.append(HRFlowable(width="100%", color=_TEAL, thickness=1.2, spaceAfter=10))

    # ── Seção: Estatísticas & Biomarcadores ─────────────────────────────
    stats = p.get("tabular_stats") or p.get("estatisticas")
    if isinstance(stats, dict) and stats.get("features"):
        story.append(Paragraph("Estatísticas & Biomarcadores", estilos["secao"]))
        linhas = [["Variável", "Média", "Mediana", "Desvio", "Min", "Max"]]
        for nome_feat, s in list(stats["features"].items())[:25]:
            linhas.append([
                nome_feat,
                _fmt_num(s.get("media")), _fmt_num(s.get("mediana")),
                _fmt_num(s.get("desvio")), _fmt_num(s.get("min")), _fmt_num(s.get("max")),
            ])
        tbl = Table(linhas, colWidths=[5.2 * cm] + [2.16 * cm] * 5, repeatRows=1)
        tbl.setStyle(_tabela_estilo_base())
        story.append(tbl)
        if len(stats["features"]) > 25:
            story.append(Paragraph(
                f"(mostrando 25 de {len(stats['features'])} variáveis — lista completa disponível na interface)",
                estilos["rotulo"],
            ))
        if p.get("schema_tabular", {}).get("label_name"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"Coluna-alvo: <b>{p['schema_tabular']['label_name']}</b>", estilos["corpo"]))

    # ── Seção: Pré-processamento ─────────────────────────────────────────
    preproc = p.get("preprocessamento") or p.get("estrategia_preproc")
    if isinstance(preproc, dict) and preproc:
        story.append(Paragraph("Pré-processamento", estilos["secao"]))
        linhas_pp = []
        for chave, valor in preproc.items():
            if isinstance(valor, (str, int, float)) and not isinstance(valor, bool):
                linhas_pp.append([chave.replace("_", " ").title(), str(valor)])
        if linhas_pp:
            tbl = Table(linhas_pp, colWidths=[6 * cm, 10 * cm])
            tbl.setStyle(_tabela_estilo_base())
            story.append(tbl)
        justificativas = preproc.get("justificativas") if isinstance(preproc, dict) else None
        if justificativas:
            story.append(Spacer(1, 6))
            story.append(Paragraph("Justificativas dos agentes de engenharia:", estilos["corpo_bold"]))
            itens = [ListItem(Paragraph(str(j), estilos["corpo"])) for j in justificativas[:10]]
            story.append(ListFlowable(itens, bulletType="bullet", start="-"))

    # ── Seção: AutoML (imagem / sinal / DICOM / volume / tabular puro) ──
    _secao_automl(story, estilos, p.get("metricas"), p.get("melhor_modelo"),
                  p.get("confusion_matrix"), p.get("classes"),
                  p.get("piso_sensibilidade_atingido"), p.get("aviso_piso_sensibilidade"),
                  titulo="AutoML — Bateria de Classificadores")

    # ── Seção: AutoML — Dados Tabulares (multimodal) ─────────────────────
    if p.get("metricas_tabular") and p.get("melhor_modelo_tabular"):
        _secao_automl(story, estilos, p["metricas_tabular"], p["melhor_modelo_tabular"],
                      p.get("cm_tabular"), None,
                      p.get("piso_sensibilidade_atingido_tabular"),
                      p.get("aviso_piso_sensibilidade_tabular"),
                      titulo="AutoML — Dados Tabulares")

    # ── Seção: Fusão Multimodal ───────────────────────────────────────────
    fusao = p.get("fusao_multimodal")
    alinhamento = p.get("alinhamento_multimodal")
    if alinhamento is not None:
        story.append(Paragraph("Fusão Multimodal", estilos["secao"]))
        if fusao and fusao.get("metricas"):
            story.append(Paragraph(
                f"{alinhamento.get('n_pareados', '—')} amostra(s) casadas entre "
                f"{' + '.join(fusao.get('modalidades', []))}.",
                estilos["corpo"],
            ))
            linhas_f = [["Candidato", "Acurácia", "Sensib.", "AUC", "F1"]]
            rotulos_f = {
                "imagem": "Imagem (isolada)", "tabular": "Tabular (isolada)",
                "sinal_temporal": "Sinal (isolado)", "dicom": "DICOM (isolado)",
                "volume_3d": "Volume 3D (isolado)",
                "fusao_media_ponderada": "Fusão — média ponderada",
                "fusao_stacking": "Fusão — stacking",
            }
            vencedor_fusao = fusao.get("melhor_modelo")
            for nome, m in fusao["metricas"].items():
                marcador = " (VENCEDOR)" if nome == vencedor_fusao else ""
                linhas_f.append([
                    rotulos_f.get(nome, nome) + marcador,
                    _fmt_pct(m.get("acuracia")), _fmt_pct(m.get("sensibilidade")),
                    _fmt_pct(m.get("auc")), _fmt_pct(m.get("f1")),
                ])
            tbl = Table(linhas_f, colWidths=[6 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm], repeatRows=1)
            tbl.setStyle(_tabela_estilo_base())
            story.append(KeepTogether([tbl, Paragraph("(VENCEDOR) = vencedor entre modalidades isoladas e fusão", estilos["rotulo"])]))
        elif not alinhamento.get("alinhado"):
            story.append(Paragraph(
                f"Fusão não calculada: {alinhamento.get('motivo', 'modalidades não alinhadas.')}",
                estilos["aviso"],
            ))

    # ── Seção: Laudo do Radiologista IA ──────────────────────────────────
    laudo_texto = resultado.get("laudo")
    if laudo_texto:
        story.append(PageBreak())
        story.append(Paragraph("Laudo do Radiologista IA", estilos["secao"]))
        story.extend(_markdown_para_flowables(laudo_texto, estilos))

    # ── Rodapé ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f0"), thickness=0.8))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Relatório gerado automaticamente pelo BioStatusIA — sistema de apoio à decisão clínica "
        "baseado em IA. Não substitui avaliação médica profissional.",
        estilos["rodape"],
    ))

    doc.build(story)
    return buf.getvalue()


def _secao_automl(story, estilos, metricas, melhor, confusion_matrix, classes,
                   piso_ok, aviso_piso, titulo):
    """Seção reutilizável de AutoML — usada tanto para o resultado primário
    (imagem/sinal/DICOM/volume/tabular puro) quanto para o resultado
    tabular dentro de uma análise multimodal."""
    if not metricas or not melhor or melhor not in metricas:
        return

    story.append(Paragraph(titulo, estilos["secao"]))

    piso_ok = True if piso_ok is None else piso_ok
    mv = metricas[melhor]
    if not piso_ok:
        story.append(Paragraph(
            f"<b>AVISO:</b> Nenhum modelo atingiu o piso clínico de segurança. "
            f"{aviso_piso or f'O modelo {melhor} foi retornado apenas como o menos inadequado do lote.'}",
            estilos["aviso"],
        ))
    else:
        story.append(Paragraph(
            f"Vencedor: <b>{melhor}</b> — Acurácia {_fmt_pct(mv.get('acuracia'))}, "
            f"Sensibilidade {_fmt_pct(mv.get('sensibilidade'))}, "
            f"AUC {_fmt_pct(mv.get('auc'))}",
            estilos["corpo"],
        ))

    # Ranking completo, ordenado pelo critério real de seleção (CV interna
    # quando disponível — mesmo raciocínio da tela de resultados).
    def _chave_ordenacao(item):
        _, m = item
        return m.get("score_clinico_cv", m.get("score_clinico", 0)) or 0

    ranking = sorted(metricas.items(), key=_chave_ordenacao, reverse=True)
    linhas = [["Modelo", "Acurácia", "Sensib.", "Sensib.(CV)", "Especif.", "AUC", "F1", "ECE"]]
    for nome, m in ranking:
        marcador = " (VENCEDOR)" if nome == melhor else ""
        linhas.append([
            nome + marcador,
            _fmt_pct(m.get("acuracia")), _fmt_pct(m.get("sensibilidade")),
            _fmt_pct(m.get("sensibilidade_cv")) if m.get("sensibilidade_cv") is not None else "—",
            _fmt_pct(m.get("especificidade")), _fmt_pct(m.get("auc")),
            _fmt_pct(m.get("f1")), _fmt_pct(m.get("ece")),
        ])
    tbl = Table(linhas, colWidths=[3.3 * cm] + [1.85 * cm] * 7, repeatRows=1)
    estilo_tbl = _tabela_estilo_base()
    idx_vencedor = next((i for i, (nome, _) in enumerate(ranking) if nome == melhor), None)
    if idx_vencedor is not None:
        estilo_tbl.add("BACKGROUND", (0, idx_vencedor + 1), (-1, idx_vencedor + 1), _TEAL_LIGHT)
    tbl.setStyle(estilo_tbl)
    story.append(tbl)
    story.append(Paragraph(
        "(VENCEDOR) = vencedor · ranking ordenado pelo score clínico da validação cruzada interna "
        "(critério real de seleção) · demais colunas são do conjunto de teste separado (held-out).",
        estilos["rotulo"],
    ))

    # Matriz de confusão do vencedor
    matriz_conf = (confusion_matrix or {}).get(melhor)
    if matriz_conf:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Matriz de confusão", estilos["subsecao"]))
        rotulos_classes = classes if classes and len(classes) == len(matriz_conf) else list(range(len(matriz_conf)))
        cab = ["real \\ previsto"] + [str(c) for c in rotulos_classes]
        linhas_cm = [cab]
        for i, linha in enumerate(matriz_conf):
            linhas_cm.append([str(rotulos_classes[i])] + [str(v) for v in linha])
        tbl_cm = Table(linhas_cm, colWidths=[2.8 * cm] + [1.6 * cm] * len(matriz_conf), repeatRows=1)
        estilo_cm = _tabela_estilo_base()
        for i in range(len(matriz_conf)):
            estilo_cm.add("BACKGROUND", (i + 1, i + 1), (i + 1, i + 1), _TEAL_LIGHT)
        tbl_cm.setStyle(estilo_cm)
        story.append(KeepTogether(tbl_cm))
