"""
Engenharia de laudos — duas vias distintas:

  1. LAUDO POPULACIONAL (nível da base): distribuição estatística do dataset,
     correlações entre biomarcadores e o pódio final do AutoML. Determinístico,
     montado a partir do `pipeline_data` já persistido — não depende do LLM.

  2. LAUDO INDIVIDUAL (inferência): ver `inferencia.py` + o agente
     `radiologista_ia_interativo`. Este módulo apenas prepara o contexto
     estruturado que alimenta esse agente.
"""
from __future__ import annotations

import numpy as np

_AVISO_ETICO = (
    "> **Aviso ético:** Este relatório é gerado por IA para **suporte à decisão "
    "clínica** e **não substitui** a avaliação de um médico habilitado."
)

_ORDEM_METRICAS = [
    ("auc", "AUC"), ("sensibilidade", "Sensib."), ("especificidade", "Especif."),
    ("f1", "F1"), ("mcc", "MCC"), ("kappa", "Kappa"),
]


def construir_podio(metricas: dict, melhor: str = "") -> list[dict]:
    """Ordena os modelos por AUC (desc) e devolve o ranking com métricas-chave."""
    ranking = sorted(
        metricas.items(),
        key=lambda kv: kv[1].get("auc", 0.0),
        reverse=True,
    )
    podio = []
    for pos, (nome, m) in enumerate(ranking, start=1):
        podio.append({
            "posicao": pos,
            "modelo": nome,
            "campeao": (nome == melhor) or (pos == 1 and not melhor),
            "auc": m.get("auc"),
            "sensibilidade": m.get("sensibilidade"),
            "especificidade": m.get("especificidade"),
            "f1": m.get("f1"),
            "mcc": m.get("mcc"),
            "kappa": m.get("kappa"),
        })
    return podio


def correlacoes_biomarcadores(vetores: list[list[float]], nomes: list[str],
                              limiar: float = 0.7) -> list[dict]:
    """Pares de biomarcadores com |correlação de Pearson| ≥ limiar."""
    if not vetores or len(vetores) < 3:
        return []
    M = np.asarray(vetores, dtype=np.float64)
    if M.ndim != 2 or M.shape[1] < 2:
        return []
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(M, rowvar=False)
    pares = []
    n = corr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            c = corr[i, j]
            if np.isfinite(c) and abs(c) >= limiar:
                ni = nomes[i] if i < len(nomes) else f"f{i}"
                nj = nomes[j] if j < len(nomes) else f"f{j}"
                pares.append({"a": ni, "b": nj, "correlacao": round(float(c), 3)})
    return sorted(pares, key=lambda p: abs(p["correlacao"]), reverse=True)[:15]


def _fmt_metrica(v) -> str:
    """Formata um valor de métrica: número → 3 casas; ausente → travessão."""
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.3f}"
    return str(v)


def _tabela_podio(podio: list[dict]) -> str:
    linhas = ["| # | Modelo | AUC | Sensib. | Especif. | F1 | MCC | Kappa |",
              "|---|--------|-----|---------|----------|----|----|-------|"]
    for p in podio:
        marca = " 🏆" if p["campeao"] else ""
        linhas.append(
            f"| {p['posicao']} | {p['modelo']}{marca} | {_fmt_metrica(p['auc'])} | "
            f"{_fmt_metrica(p['sensibilidade'])} | {_fmt_metrica(p['especificidade'])} | "
            f"{_fmt_metrica(p['f1'])} | {_fmt_metrica(p['mcc'])} | {_fmt_metrica(p['kappa'])} |"
        )
    return "\n".join(linhas)


def construir_laudo_populacional(pipeline_data: dict) -> str:
    """
    Monta o laudo populacional em Markdown a partir do pipeline_data persistido.
    Cobre: cabeçalho, distribuição de classes, pódio AutoML, correlações,
    balanceamento e importância SHAP — quando disponíveis.
    """
    familia = pipeline_data.get("familia", "")
    tipo = pipeline_data.get("tipo_sinal") or pipeline_data.get("modo", "")
    n = pipeline_data.get("n_imagens", 0)

    partes = [
        "# Laudo Populacional da Base",
        f"**Família:** {familia or 'N/A'}  |  **Tipo:** {tipo or 'N/A'}  |  "
        f"**Amostras:** {n}",
        "",
    ]

    # ── Distribuição de classes ──────────────────────────────────────────
    dist = None
    if isinstance(pipeline_data.get("tabular_stats"), dict):
        dist = pipeline_data["tabular_stats"].get("distribuicao_classes")
    if dist:
        partes += ["## Distribuição de Classes", ""]
        partes += [f"- Classe `{k}`: {v} amostras" for k, v in dist.items()]
        partes.append("")

    # ── Pódio AutoML ─────────────────────────────────────────────────────
    metricas = pipeline_data.get("metricas", {})
    if metricas:
        melhor = pipeline_data.get("melhor_modelo", "")
        podio = construir_podio(metricas, melhor)
        partes += ["## Pódio AutoML (5-fold CV + teste 20%)", "",
                   _tabela_podio(podio), ""]
        if melhor:
            partes.append(f"**Campeão adotado:** {melhor} (maior AUC, "
                          f"preferência por sensibilidade ≥ 0.8).")
            partes.append("")

    # ── Balanceamento ────────────────────────────────────────────────────
    balan = pipeline_data.get("balanceamento", {})
    if balan.get("aplicado"):
        partes += ["## Balanceamento de Classes",
                   f"- Método: **{balan.get('metodo', 'smote').upper()}**",
                   f"- Antes: {balan.get('distribuicao_original')}",
                   f"- Depois: {balan.get('distribuicao_balanceada')}", ""]

    # ── SHAP ─────────────────────────────────────────────────────────────
    shap_info = pipeline_data.get("shap", {})
    if shap_info.get("disponivel"):
        top = shap_info.get("top_features", [])[:5]
        partes += ["## Features Mais Influentes (SHAP)",
                   f"- Top: {', '.join(top)}", ""]

    # ── Correlações ──────────────────────────────────────────────────────
    correls = pipeline_data.get("correlacoes_biomarcadores", [])
    if correls:
        partes += ["## Correlações Fortes entre Biomarcadores (|r| ≥ 0.7)", ""]
        partes += [f"- {c['a']} ↔ {c['b']}: r = {c['correlacao']}" for c in correls[:10]]
        partes.append("")

    partes += ["---", _AVISO_ETICO]
    return "\n".join(partes)
