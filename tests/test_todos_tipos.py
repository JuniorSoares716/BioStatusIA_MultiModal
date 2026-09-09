"""
Teste de pipeline puro (sem LLM) para CADA tipo de dado em escopo:
imagem comum, tabular, F1 (sinal temporal), F3 (DICOM 2D) e F4 (volume 3D).

Para os tipos sem fixture no repositório (F1/F3/F4) gera pequenos datasets
sintéticos rotulados (benign/malignant) para exercitar leitura → extração → AutoML.
Gera um relatório consolidado em reports/relatorio_tipos.md.
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from biostatusia.pipeline.classificador import treinar_vetores
from biostatusia.pipeline.inferencia import achatar_biomarcadores

FIX = RAIZ / "tests" / "test_platform_data"
rng = np.random.RandomState(42)
relato = []


def log(msg=""):
    print(msg)
    relato.append(msg)


def treina(vetores, labels, familia):
    """Roda AutoML se houver >=10 amostras e 2 classes; devolve resumo textual."""
    X = np.array(vetores, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(labels)
    if len(X) < 10 or len(set(y.tolist())) < 2:
        return f"AutoML NÃO executado ({len(X)} amostras, {len(set(y.tolist()))} classes)."
    res = treinar_vetores(X, y, scaling="standard", familia=familia)
    m = res["metricas"][res["melhor_modelo"]]
    return (f"AutoML OK — melhor: **{res['melhor_modelo']}** | "
            f"AUC={m['auc']} Sens={m['sensibilidade']} Espec={m['especificidade']} "
            f"F1={m['f1']} MCC={m['mcc']} Kappa={m['kappa']}")


# ── Geradores de fixtures sintéticas ─────────────────────────────────────────

def gerar_ecg_mat(path, maligno: bool):
    import scipy.io
    fs, dur = 250, 8
    t = np.linspace(0, dur, fs * dur)
    hr = 1.8 if maligno else 1.0            # ~108 vs 60 bpm
    ruido = 0.25 if maligno else 0.05
    ecg = np.sin(2 * np.pi * hr * t) + 0.3 * np.sin(2 * np.pi * 2 * hr * t)
    ecg += ruido * rng.randn(len(t))
    scipy.io.savemat(str(path), {"val": ecg.reshape(1, -1)})


def gerar_dicom(path, maligno: bool):
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage
    n = 64
    base = 900 if maligno else 400
    arr = (base + 120 * rng.randn(n, n)).clip(0, 4000).astype(np.uint16)
    if maligno:                              # lesão densa irregular
        arr[20:44, 20:50] = (1800 + 200 * rng.randn(24, 30)).clip(0, 4000).astype(np.uint16)
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = CTImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.ImplementationClassUID = generate_uid()
    ds = Dataset()
    ds.file_meta = fm
    ds.preamble = b"\x00" * 128
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.Rows, ds.Columns = n, n
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [1.0, 1.0]
    ds.WindowCenter = 1000
    ds.WindowWidth = 2000
    ds.PixelData = arr.tobytes()
    try:
        ds.save_as(str(path), enforce_file_format=True)   # pydicom >=3
    except TypeError:
        ds.save_as(str(path), write_like_original=False)  # pydicom <3


def gerar_nifti(path, maligno: bool):
    import nibabel as nib
    d, h, w = 24, 32, 32
    base = 0.7 if maligno else 0.3
    vol = (base + 0.1 * rng.randn(w, h, d)).astype(np.float32)
    if maligno:                              # nódulo esférico de alta intensidade
        zz, yy, xx = np.mgrid[0:w, 0:h, 0:d]
        mask = (xx - d // 2) ** 2 + (yy - h // 2) ** 2 + (zz - w // 2) ** 2 < 25
        vol[mask] = 1.5
    nib.save(nib.Nifti1Image(vol, affine=np.eye(4)), str(path))


def preparar(sub, n_por_classe, gerador, ext):
    """Cria n_por_classe arquivos por classe se ainda não existirem."""
    base = FIX / sub
    caminhos = []
    for cls, malig in (("benign", False), ("malignant", True)):
        d = base / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_por_classe):
            p = d / f"{cls}_{i:02d}{ext}"
            if not p.exists():
                gerador(p, malig)
            caminhos.append((p, 1 if malig else 0))
    return caminhos


# ── 1. IMAGEM COMUM (fixture real: Mini-BUSI) ────────────────────────────────

def testar_imagem():
    log("## 1. Imagem comum (radiômica) — Mini-BUSI")
    from biostatusia.pipeline.extracao import extrair_todos
    from biostatusia.pipeline.classificador import _vetor
    from biostatusia.pipeline.io_utils import listar_imagens
    regs = listar_imagens(RAIZ / "dataset_teste_busi")
    vetores, labels, n_ok = [], [], 0
    for r in regs:
        bio = extrair_todos(r["caminho"])
        if bio and r["label"] is not None:
            vetores.append(_vetor(bio)); labels.append(r["label"]); n_ok += 1
    log(f"- Imagens rotuladas processadas: **{n_ok}** (de {len(regs)})")
    if n_ok:
        log(f"- Exemplo de biomarcadores (1ª imagem): {list(extrair_todos(regs[0]['caminho'])['morfologia'].items())}")
        log(f"- {treina(vetores, labels, 'IMG')}")
    log("")


# ── 2. TABULAR (fixture real: WBCD-50) ───────────────────────────────────────

def testar_tabular():
    log("## 2. Tabular — WBCD-50")
    from biostatusia.pipeline.dados_tabulares import (
        carregar_csv, detectar_schema, extrair_features, analisar_tabular,
    )
    csv = RAIZ / "dataset_teste_csv" / "wbcd_50.csv"
    header, data = carregar_csv(str(csv))
    schema = detectar_schema(header, data)
    X, y, _ = extrair_features(data, schema)
    stats = analisar_tabular(X, y, schema)
    log(f"- Coluna-rótulo detectada: **{schema['label_name']}** | features: {schema['n_features']} | amostras: {stats['n_amostras']}")
    X = np.nan_to_num(X, nan=0.0)
    log(f"- {treina(X.tolist(), y.tolist(), 'TAB')}")
    log("")


# ── 3/4/5. F1 / F3 / F4 (fixtures sintéticas) ────────────────────────────────

def testar_familia(titulo, sub, n, gerador, ext, familia, extrator, exemplo_key):
    log(f"## {titulo}")
    from biostatusia.pipeline.io_sinais import carregar_sinal
    caminhos = preparar(sub, n, gerador, ext)
    vetores, labels, exemplo = [], [], None
    for p, lbl in caminhos:
        try:
            sinal = carregar_sinal(str(p))
            bio = extrator(sinal)
            if not bio or "erro" in bio:
                continue
            v = achatar_biomarcadores(bio)
            if v.size:
                vetores.append(v.tolist()); labels.append(lbl)
                if exemplo is None:
                    exemplo = (sinal.tipo, sinal.taxa_amostragem, bio)
        except Exception as e:
            log(f"- ERRO em {p.name}: {e}")
    log(f"- Sinais processados: **{len(vetores)}** | tipo detectado: {exemplo[0] if exemplo else '—'}")
    if exemplo:
        chaves = list(exemplo[2].get(exemplo_key, {}).keys()) if isinstance(exemplo[2].get(exemplo_key), dict) else list(exemplo[2].keys())
        log(f"- Grupos de features: {list(exemplo[2].keys())}")
    # alinhar largura dos vetores
    if vetores:
        largura = min(len(v) for v in vetores)
        vetores = [v[:largura] for v in vetores]
        log(f"- Dimensão do vetor de features: {largura}")
        log(f"- {treina(vetores, labels, familia)}")
    log("")


def main():
    from biostatusia.pipeline.extracao_temporal import extrair_biomarcadores_temporal
    from biostatusia.pipeline.extracao_dicom import extrair_biomarcadores_dicom
    from biostatusia.pipeline.extracao_volumetrica import extrair_biomarcadores_volumetrico

    log("# Relatório — Pipeline por Tipo de Dado (sem LLM)\n")
    log("Valida leitura → extração de biomarcadores → AutoML (`treinar_vetores`) para cada família.\n")
    testar_imagem()
    testar_tabular()
    testar_familia("3. F1 — Sinal Temporal (ECG sintético)", "f1_ecg", 7,
                   gerar_ecg_mat, ".mat", "F1", extrair_biomarcadores_temporal, "tempo")
    testar_familia("4. F3 — DICOM 2D (TC sintético)", "f3_dicom", 7,
                   gerar_dicom, ".dcm", "F3", extrair_biomarcadores_dicom, "morfologia")
    testar_familia("5. F4 — Volume 3D (NIfTI sintético)", "f4_volume", 7,
                   gerar_nifti, ".nii", "F4", extrair_biomarcadores_volumetrico, "estatisticas_globais")

    saida = RAIZ / "reports" / "relatorio_tipos.md"
    saida.parent.mkdir(exist_ok=True)
    saida.write_text("\n".join(relato), encoding="utf-8")
    print(f"\nRelatório salvo em {saida}")


if __name__ == "__main__":
    main()
