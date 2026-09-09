import numpy as np
from PIL import Image
from scipy import stats
from skimage.restoration import estimate_sigma


def analisar_base(caminhos: list[str]) -> dict:
    """Estatísticas globais sobre a base completa, antes do pré-processamento."""
    intensidades_medias: list[float] = []
    desvios: list[float] = []
    sigmas_ruido: list[float] = []
    tamanhos: list[tuple] = []

    for caminho in caminhos:
        try:
            img = np.array(Image.open(caminho).convert("L"), dtype=np.float32)
        except Exception:
            continue
        intensidades_medias.append(float(img.mean()))
        desvios.append(float(img.std()))
        try:
            sigma = float(estimate_sigma(img / 255.0, channel_axis=None))
        except Exception:
            sigma = 0.0
        sigmas_ruido.append(sigma)
        tamanhos.append(img.shape)

    if not intensidades_medias:
        return {"n_imagens_analisadas": 0}

    arr = np.array(intensidades_medias)

    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    outliers = int(((arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)).sum())

    if len(arr) >= 3:
        stat_val, p_value = stats.shapiro(arr)
        eh_normal: bool | None = bool(p_value > 0.05)
        shapiro_p = round(float(p_value), 4)
    else:
        eh_normal = None
        shapiro_p = None

    tamanhos_unicos = set(tamanhos)

    return {
        "n_imagens_analisadas": len(intensidades_medias),
        "intensidade_media": round(float(arr.mean()), 2),
        "intensidade_desvio": round(float(arr.std()), 2),
        "outliers_intensidade": outliers,
        "teste_normalidade": {
            "shapiro_p": shapiro_p,
            "eh_normal": eh_normal,
        },
        "contraste_medio": round(float(np.mean(desvios)), 2),
        "ruido_medio": round(float(np.mean(sigmas_ruido)), 4),
        "tamanhos_consistentes": len(tamanhos_unicos) == 1,
        "n_tamanhos_unicos": len(tamanhos_unicos),
    }


def decidir_estrategia(analise: dict) -> dict:
    """Decide a estratégia de pré-processamento a partir das características detectadas."""
    estrategia: dict = {
        "denoising": "gaussian",
        "normalizacao": "minmax",
        "equalizacao": "none",
        "redimensionar": True,
        "tamanho_alvo": [256, 256],
        "justificativas": [],
    }

    n = analise.get("n_imagens_analisadas", 0)
    if n == 0:
        estrategia["justificativas"].append("Nenhuma imagem analisada — estratégia padrão.")
        return estrategia

    ruido = analise.get("ruido_medio", 0.0)
    if ruido > 0.05:
        estrategia["denoising"] = "nlmeans"
        estrategia["justificativas"].append(
            f"Ruído médio alto ({ruido:.3f}) → Non-Local Means denoising."
        )
    else:
        estrategia["justificativas"].append(
            f"Ruído aceitável ({ruido:.3f}) → Gaussian blur 5×5 padrão."
        )

    outliers = analise.get("outliers_intensidade", 0)
    if outliers > max(1, n * 0.1):
        estrategia["normalizacao"] = "percentil"
        estrategia["justificativas"].append(
            f"{outliers} imagens outliers ({outliers/n:.0%}) → normalização por percentil 1–99%."
        )
    else:
        estrategia["justificativas"].append(
            f"Poucos outliers ({outliers}/{n}) → normalização min-max padrão."
        )

    contraste = analise.get("contraste_medio", 100)
    if contraste < 30:
        estrategia["equalizacao"] = "clahe"
        estrategia["justificativas"].append(
            f"Contraste médio baixo ({contraste:.1f}) → equalização CLAHE."
        )
    else:
        estrategia["justificativas"].append(
            f"Contraste adequado ({contraste:.1f}) → sem equalização."
        )

    if analise.get("tamanhos_consistentes"):
        estrategia["justificativas"].append(
            "Tamanhos consistentes — redimensionamento padronizado para 256×256."
        )
    else:
        n_tam = analise.get("n_tamanhos_unicos", 0)
        estrategia["justificativas"].append(
            f"{n_tam} tamanhos diferentes encontrados → redimensionamento obrigatório para 256×256."
        )

    norm = analise.get("teste_normalidade", {})
    if norm.get("eh_normal") is False:
        estrategia["justificativas"].append(
            f"Distribuição não-normal (Shapiro p={norm['shapiro_p']}) — reportado apenas."
        )
    elif norm.get("eh_normal") is True:
        estrategia["justificativas"].append(
            f"Distribuição normal (Shapiro p={norm['shapiro_p']}) ✓"
        )

    return estrategia
