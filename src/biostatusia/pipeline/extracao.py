import cv2
import numpy as np
from io import BytesIO

from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis


def extrair_morfologia(gray_u8: np.ndarray) -> dict:
    thresh = cv2.adaptiveThreshold(
        gray_u8, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"circularidade": 0.0, "solidez": 0.0}
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularidade = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidez = area / hull_area if hull_area > 0 else 0.0
    return {
        "circularidade": round(float(circularidade), 4),
        "solidez": round(float(solidez), 4),
    }


def extrair_intensidade(img: np.ndarray) -> dict:
    flat = img.flatten()
    mean = np.mean(flat)
    std = np.std(flat)
    return {
        "snr": round(float(mean / (std + 1e-8)), 4),
        "assimetria": round(float(skew(flat)), 4),
        "curtose": round(float(kurtosis(flat)), 4),
    }


def extrair_glcm(gray_u8: np.ndarray) -> dict:
    glcm = graycomatrix(gray_u8, [1], [0], 256, symmetric=True, normed=True)
    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256])
    p = hist / (hist.sum() + 1e-10)
    entropia = float(-np.sum(p * np.log2(p + 1e-10)))
    return {
        "contraste": round(float(graycoprops(glcm, "contrast")[0, 0]), 4),
        "homogeneidade": round(float(graycoprops(glcm, "homogeneity")[0, 0]), 4),
        "energia": round(float(graycoprops(glcm, "energy")[0, 0]), 4),
        "entropia": round(entropia, 4),
    }


def extrair_todos(caminho: str, estrategia: dict | None = None) -> dict | None:
    """Extrai biomarcadores. Se `estrategia` for fornecida, aplica pré-processamento adaptativo."""
    if estrategia is not None:
        from biostatusia.pipeline.preprocessamento import preprocessar_adaptativo
        arr_norm = preprocessar_adaptativo(caminho, estrategia)
        if arr_norm is None:
            return None
        gray = (arr_norm * 255).astype(np.uint8)
        return {
            "morfologia": extrair_morfologia(gray),
            "textura_glcm": extrair_glcm(gray),
            "distribuicao_intensidade": extrair_intensidade(arr_norm),
        }

    try:
        if caminho.startswith("http"):
            import requests
            response = requests.get(caminho, headers={"User-Agent": "Mozilla/5.0"})
            from PIL import Image
            img_pil = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            from PIL import Image
            img_pil = Image.open(caminho).convert("RGB")
    except Exception:
        return None

    arr = np.array(img_pil).astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    img_u8 = (arr * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

    return {
        "morfologia": extrair_morfologia(gray),
        "textura_glcm": extrair_glcm(gray),
        "distribuicao_intensidade": extrair_intensidade(arr),
    }


def extrair_lote_imagem(arquivos: list[dict], estrategia: dict | None = None) -> list[dict]:
    """
    Processa uma lista de imagens e retorna registros com biomarcadores, no
    mesmo formato usado por `extrair_lote_temporal`/`extrair_lote_dicom`/
    `extrair_lote_volumetrico`: [{"caminho","label","categoria","biomarcadores"}, ...].

    `arquivos`: [{"caminho": ..., "label": 0|1|None, "categoria": ...}, ...]
    (mesmo formato devolvido por `listar_imagens`).
    """
    resultados = []
    for arq in arquivos:
        try:
            bio = extrair_todos(arq["caminho"], estrategia=estrategia)
            if bio is not None:
                resultados.append({
                    "caminho": arq["caminho"],
                    "label": arq.get("label"),
                    "categoria": arq.get("categoria", "INDEFINIDO"),
                    "biomarcadores": bio,
                })
        except Exception as e:
            resultados.append({
                "caminho": arq["caminho"],
                "erro": str(e),
                "categoria": arq.get("categoria", "INDEFINIDO"),
            })
    return resultados
