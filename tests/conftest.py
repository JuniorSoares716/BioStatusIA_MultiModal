"""Fixtures compartilhadas da suíte pytest do BioStatusIA."""
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_rng = np.random.RandomState(7)

# Scripts de validação manuais (não são testes pytest) — excluídos da coleta automática.
collect_ignore = [
    "test_direct.py",
    "test_large_image_dataset.py",
    "test_todos_tipos.py",
    "run_tests.py",
]


@pytest.fixture(scope="session")
def raiz():
    return RAIZ


# ── Geradores de fixtures sintéticas (reutilizáveis nos testes) ───────────────

def _gerar_ecg_mat(path, maligno=False):
    import scipy.io
    fs, dur = 250, 6
    t = np.linspace(0, dur, fs * dur)
    hr = 1.8 if maligno else 1.0
    ecg = np.sin(2 * np.pi * hr * t) + (0.25 if maligno else 0.05) * _rng.randn(len(t))
    scipy.io.savemat(str(path), {"val": ecg.reshape(1, -1)})


def _gerar_dicom(path, maligno=False):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage
    n = 64
    base = 900 if maligno else 400
    arr = (base + 120 * _rng.randn(n, n)).clip(0, 4000).astype(np.uint16)
    if maligno:
        arr[20:44, 20:50] = (1800 + 200 * _rng.randn(24, 30)).clip(0, 4000).astype(np.uint16)
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
        ds.save_as(str(path), enforce_file_format=True)
    except TypeError:
        ds.save_as(str(path), write_like_original=False)


def _gerar_nifti(path, maligno=False):
    import nibabel as nib
    d, h, w = 24, 32, 32
    base = 0.7 if maligno else 0.3
    vol = (base + 0.1 * _rng.randn(w, h, d)).astype(np.float32)
    if maligno:
        zz, yy, xx = np.mgrid[0:w, 0:h, 0:d]
        mask = (xx - d // 2) ** 2 + (yy - h // 2) ** 2 + (zz - w // 2) ** 2 < 25
        vol[mask] = 1.5
    nib.save(nib.Nifti1Image(vol, affine=np.eye(4)), str(path))


@pytest.fixture
def gerar_ecg():
    return _gerar_ecg_mat


@pytest.fixture
def gerar_dicom():
    return _gerar_dicom


@pytest.fixture
def gerar_nifti():
    return _gerar_nifti


# ── Dados auxiliares ─────────────────────────────────────────────────────────

@pytest.fixture
def metricas_completas():
    """Dict de métricas por modelo no formato do AutoML (3 modelos, AUC decrescente)."""
    def m(auc, sens=0.9):
        return {
            "acuracia": 0.9, "sensibilidade": sens, "especificidade": 0.88,
            "precisao": 0.9, "recall": sens, "f1": 0.9, "auc": auc,
            "mcc": 0.8, "kappa": 0.79, "ece": 0.05,
            "latencia_inferencia_ms": 0.5, "tempo_treino_s": 0.1,
        }
    return {"RandomForest": m(0.94), "SVM": m(0.90), "KNN": m(0.85)}


@pytest.fixture
def dados_binarios():
    """(X, y) separável: 60 amostras, 9 features, 2 classes."""
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(0, 1, (30, 9)), rng.normal(2, 1, (30, 9))])
    y = np.array([0] * 30 + [1] * 30)
    return X, y


@pytest.fixture
def dataset_rotulado(tmp_path):
    """Pasta com subpastas benign/ e malignant/ contendo imagens (vazias — só p/ detecção)."""
    for cls in ("benign", "malignant"):
        d = tmp_path / cls
        d.mkdir()
        for i in range(3):
            (d / f"{cls}_{i}.png").write_bytes(b"")
    return tmp_path


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Cliente de teste Flask com banco isolado em arquivo temporário.

    Isola cada teste de rota do `biostatusia.db` real (evita poluir o banco e
    torna os testes herméticos/determinísticos)."""
    monkeypatch.setattr("biostatusia.database.DB_PATH", tmp_path / "test.db")
    from biostatusia.app import app
    app.config.update(TESTING=True)
    return app.test_client()
