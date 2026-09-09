"""
Extrator F4 — Biomarcadores de Volumes 3D (TC, RM, PET/SPECT).
Features: estatísticas volumétricas, textura por plano, morfologia 3D.
"""
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis


def extrair_biomarcadores_volumetrico(sinal) -> dict | None:
    """Recebe SinalNormalizado (F4) e retorna dict de biomarcadores 3D."""
    try:
        volume = sinal.dados  # float32 (D, H, W)
        meta = sinal.metadados
        D, H, W = volume.shape

        bio = {
            "volumetrico_global": _stats_globais(volume),
            "textura_3d": _textura_por_plano(volume),
            "morfologia_3d": _morfologia_3d(volume, meta),
        }
        return bio
    except Exception as e:
        return {"erro": str(e)}


def _stats_globais(volume: np.ndarray) -> dict:
    flat = volume.flatten()
    return {
        "media_intensidade": round(float(np.mean(flat)), 6),
        "desvio_intensidade": round(float(np.std(flat)), 6),
        "mediana_intensidade": round(float(np.median(flat)), 6),
        "assimetria": round(float(skew(flat)), 4),
        "curtose": round(float(kurtosis(flat)), 4),
        "percentil_5": round(float(np.percentile(flat, 5)), 6),
        "percentil_95": round(float(np.percentile(flat, 95)), 6),
        "n_voxels_altos": int(np.sum(flat > 0.7)),
        "fracao_voxels_altos": round(float(np.mean(flat > 0.7)), 6),
    }


def _textura_por_plano(volume: np.ndarray) -> dict:
    """GLCM no slice central de cada plano ortogonal."""
    D, H, W = volume.shape
    resultados: dict = {}

    planos = {
        "axial":   volume[D // 2],
        "coronal": volume[:, H // 2, :],
        "sagital": volume[:, :, W // 2],
    }

    for nome, slc in planos.items():
        gray = (slc * 255).astype(np.uint8)
        glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
        hist = gray.flatten()
        p = np.bincount(hist, minlength=256).astype(float)
        p /= p.sum() + 1e-10
        entropia = float(-np.sum(p * np.log2(p + 1e-10)))
        resultados[nome] = {
            "contraste": round(float(graycoprops(glcm, "contrast")[0, 0]), 4),
            "homogeneidade": round(float(graycoprops(glcm, "homogeneity")[0, 0]), 4),
            "energia": round(float(graycoprops(glcm, "energy")[0, 0]), 4),
            "entropia": round(entropia, 4),
        }

    return resultados


def _morfologia_3d(volume: np.ndarray, meta: dict) -> dict:
    """Estima volume de lesão candidata via threshold de intensidade alta."""
    threshold = 0.65
    mask = volume > threshold
    n_voxels_lesao = int(mask.sum())

    voxel_size = meta.get("voxel_size_mm", [1.0, 1.0, 1.0])
    vol_voxel = float(np.prod(voxel_size[:3])) if len(voxel_size) >= 3 else 1.0
    volume_mm3 = n_voxels_lesao * vol_voxel

    # Sphericity aproximada pela razão de eixos do bounding box
    coords = np.argwhere(mask)
    sphericity = 0.0
    if len(coords) > 10:
        ranges = coords.max(axis=0) - coords.min(axis=0) + 1
        # Sphericity = (pi^(1/3) * (6V)^(2/3)) / A — aproximamos por razão de eixos
        min_r, max_r = float(ranges.min()), float(ranges.max())
        sphericity = round(min_r / (max_r + 1e-8), 4)

    return {
        "n_voxels_lesao_candidata": n_voxels_lesao,
        "volume_lesao_mm3": round(volume_mm3, 2),
        "sphericity_aprox": sphericity,
        "n_slices_total": volume.shape[0],
        "voxel_size_mm": voxel_size[:3] if len(voxel_size) >= 3 else [1.0, 1.0, 1.0],
    }


def extrair_lote_volumetrico(arquivos: list[dict]) -> list[dict]:
    """Processa lista de volumes e retorna registros com biomarcadores."""
    from biostatusia.pipeline.io_sinais import carregar_sinal

    resultados = []
    for arq in arquivos:
        try:
            sinal = carregar_sinal(arq["caminho"])
            bio = extrair_biomarcadores_volumetrico(sinal)
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
