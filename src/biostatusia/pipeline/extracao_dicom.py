"""
Extrator F3 — Biomarcadores de Imagens DICOM 2D.
Estende os 9 biomarcadores existentes com features radiômicas DICOM-específicas.
"""
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis


def extrair_biomarcadores_dicom(sinal) -> dict | None:
    """Recebe SinalNormalizado (F3) e retorna dict de biomarcadores."""
    try:
        arr = sinal.dados  # float32 [0,1] (H, W)
        gray = (arr * 255).astype(np.uint8)
        meta = sinal.metadados

        bio = {
            "morfologia": _extrair_morfologia(gray),
            "textura_glcm": _extrair_glcm(gray),
            "distribuicao_intensidade": _extrair_intensidade(arr),
            "radiomico_dicom": _extrair_radiomico(arr, gray, meta),
        }
        return bio
    except Exception as e:
        return {"erro": str(e)}


def _extrair_morfologia(gray_u8: np.ndarray) -> dict:
    thresh = cv2.adaptiveThreshold(
        gray_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"circularidade": 0.0, "solidez": 0.0}
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidez = area / hull_area if hull_area > 0 else 0.0
    return {
        "circularidade": round(float(circ), 4),
        "solidez": round(float(solidez), 4),
    }


def _extrair_glcm(gray_u8: np.ndarray) -> dict:
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


def _extrair_intensidade(arr: np.ndarray) -> dict:
    flat = arr.flatten()
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    return {
        "snr": round(mean / (std + 1e-8), 4),
        "assimetria": round(float(skew(flat)), 4),
        "curtose": round(float(kurtosis(flat)), 4),
    }


def _extrair_radiomico(arr: np.ndarray, gray: np.ndarray, meta: dict) -> dict:
    """Features radiômicas específicas para imagens médicas."""
    h, w = arr.shape
    area_pixel = h * w

    # Densidade de brilho alto (proxy de densidade mamária ou opacidade RX)
    threshold_alto = 0.7
    densidade_alta = float(np.sum(arr > threshold_alto)) / area_pixel

    # Gradiente médio (nitidez de bordas — relevante para nódulos)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradiente = float(np.mean(np.sqrt(sobel_x ** 2 + sobel_y ** 2)))

    # Uniformidade (1 - coeficiente de variação)
    mean_ = float(np.mean(arr))
    std_ = float(np.std(arr))
    uniformidade = 1.0 - (std_ / (mean_ + 1e-8))

    # Hounsfield range (se disponível nos metadados)
    hu_range = None
    if "hounsfield_min" in meta and "hounsfield_max" in meta:
        hu_range = meta["hounsfield_max"] - meta["hounsfield_min"]

    resultado = {
        "densidade_alta_pct": round(densidade_alta * 100, 2),
        "gradiente_medio": round(gradiente, 4),
        "uniformidade": round(float(uniformidade), 4),
        "modalidade": meta.get("modalidade", "DICOM"),
        "pixel_spacing": meta.get("pixel_spacing_mm", [1.0, 1.0]),
    }
    if hu_range is not None:
        resultado["hounsfield_range"] = round(hu_range, 1)

    return resultado


def extrair_lote_dicom(arquivos: list[dict]) -> list[dict]:
    """Processa lista de DICOMs e retorna registros com biomarcadores."""
    from biostatusia.pipeline.io_sinais import carregar_sinal

    resultados = []
    for arq in arquivos:
        try:
            sinal = carregar_sinal(arq["caminho"])
            bio = extrair_biomarcadores_dicom(sinal)
            if bio and "erro" not in bio:
                resultados.append({
                    "caminho": arq["caminho"],
                    "label": arq.get("label"),
                    "categoria": arq.get("categoria", "INDEFINIDO"),
                    "biomarcadores": bio,
                    "tipo": sinal.tipo,
                })
        except Exception as e:
            resultados.append({
                "caminho": arq["caminho"],
                "erro": str(e),
                "categoria": arq.get("categoria", "INDEFINIDO"),
            })
    return resultados
