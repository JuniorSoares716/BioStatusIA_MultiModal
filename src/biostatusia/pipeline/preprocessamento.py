import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.restoration import denoise_nl_means


def preprocessar(caminho: str, tamanho: tuple[int, int] = (256, 256)) -> np.ndarray | None:
    """Pré-processamento padrão (compatibilidade com main.py)."""
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    redimensionada = cv2.resize(img, tamanho)
    suavizada = cv2.GaussianBlur(redimensionada, (5, 5), 0)
    normalizada = suavizada.astype(np.float32) / 255.0
    return normalizada


def preprocessar_adaptativo(caminho: str, estrategia: dict) -> np.ndarray | None:
    """Aplica a estratégia decidida por decidir_estrategia()."""
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    if estrategia.get("redimensionar", True):
        tamanho = tuple(estrategia.get("tamanho_alvo", [256, 256]))
        img = cv2.resize(img, tamanho)

    if estrategia.get("equalizacao") == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    if estrategia.get("denoising") == "nlmeans":
        img_float = img.astype(np.float32) / 255.0
        img_float = denoise_nl_means(
            img_float, h=0.1, fast_mode=True, patch_size=5, patch_distance=6
        )
        img = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
    else:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    if estrategia.get("normalizacao") == "percentil":
        p1, p99 = np.percentile(img, [1, 99])
        clipped = np.clip(img, p1, p99).astype(np.float32)
        normalizada = (clipped - p1) / ((p99 - p1) + 1e-8)
    else:
        normalizada = img.astype(np.float32) / 255.0

    return normalizada


def engenharia_features(imagem: np.ndarray, estrategia: dict) -> dict:
    """Features derivadas da imagem já pré-processada (preprocessar_adaptativo).

    Recebe array float [0,1] ou uint8 e devolve {nome: valor_float}. A estratégia
    entra como contexto para futuras variações — não altera o cálculo atual.
    """
    if imagem is None or getattr(imagem, "size", 0) == 0:
        return {}

    img = imagem
    if np.issubdtype(img.dtype, np.floating):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    features: dict = {}

    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    features["energia_borda_sobel"] = round(float(magnitude.mean()), 4)

    hist, _ = np.histogram(img, bins=8, range=(0, 255))
    total = hist.sum()
    hist_norm = hist / total if total else hist
    for i, valor in enumerate(hist_norm):
        features[f"hist_bin_{i}"] = round(float(valor), 4)

    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    features["lbp_media"] = round(float(lbp.mean()), 4)
    features["lbp_desvio"] = round(float(lbp.std()), 4)

    h, w = img.shape[:2]
    mh, mw = h // 2, w // 2
    quadrantes = [
        img[:mh, :mw], img[:mh, mw:],
        img[mh:, :mw], img[mh:, mw:],
    ]
    for i, quad in enumerate(quadrantes):
        if quad.size:
            features[f"quad_{i}_media"] = round(float(quad.mean()), 4)
            features[f"quad_{i}_desvio"] = round(float(quad.std()), 4)

    return features


def ranquear_features(matriz: np.ndarray, nomes: list[str], y: np.ndarray | None = None) -> dict:
    """Ordena as features de engenharia por importância.

    Escolhe o método conforme os dados disponíveis:
      - ANOVA F-test (f_classif) quando há rótulos com ≥2 classes → mede a
        separação entre benigno/maligno; devolve F-score e p-valor.
      - Variância quando não há rótulos (imagens soltas).
      - Magnitude absoluta quando há uma única amostra.
    Retorna {"metodo": str, "ranking": [{"nome", "score", "p_valor"}]} ordenado desc.
    """
    if matriz is None or matriz.size == 0 or not nomes:
        return {"metodo": "—", "ranking": []}

    X = np.nan_to_num(np.asarray(matriz, dtype=float))
    n_amostras = X.shape[0]

    p_vals: list[float | None] = [None] * len(nomes)

    if n_amostras == 1:
        metodo = "Magnitude absoluta (amostra única)"
        scores = np.abs(X[0])
    elif y is not None and len(set(np.asarray(y).tolist())) >= 2:
        from sklearn.feature_selection import f_classif
        with np.errstate(all="ignore"):
            f_scores, p = f_classif(X, np.asarray(y))
        scores = np.nan_to_num(f_scores)
        p_vals = [round(float(v), 5) if np.isfinite(v) else None for v in p]
        metodo = "ANOVA F-test (f_classif) — separação entre classes"
    else:
        metodo = "Variância (sem rótulos)"
        scores = X.var(axis=0)

    ranking = sorted(
        (
            {"nome": nomes[i], "score": round(float(scores[i]), 4), "p_valor": p_vals[i]}
            for i in range(len(nomes))
        ),
        key=lambda d: d["score"],
        reverse=True,
    )
    return {"metodo": metodo, "ranking": ranking}
