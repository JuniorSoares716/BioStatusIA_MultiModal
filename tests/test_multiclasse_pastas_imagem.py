"""Testes para a detecção de rótulo multi-classe por pasta em datasets de
imagem — antes, só a convenção binária (benign/malignant, yes/no) era
reconhecida; qualquer estrutura com N classes nomeadas (ex.: glioma/
meningioma/notumor/pituitary) ficava toda como INDEFINIDO."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.io_utils import listar_imagens


def _cria_imagens(pasta: Path, n: int):
    pasta.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (pasta / f"img_{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)


def test_detecta_4_classes_nomeadas_sem_split():
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    for nome, n in [("glioma", 10), ("meningioma", 10), ("notumor", 10), ("pituitary", 10)]:
        _cria_imagens(tmp / nome, n)

    registros = listar_imagens(tmp)
    c = Counter(r["categoria"] for r in registros)
    assert c.get("INDEFINIDO", 0) == 0
    assert set(c.keys()) == {"GLIOMA", "MENINGIOMA", "NOTUMOR", "PITUITARY"}
    assert len(set(r["label"] for r in registros)) == 4


def test_detecta_multiclasse_dentro_de_training_testing(tmp_path):
    """Regressão do caso real: Training/glioma, Testing/glioma etc. devem
    ser unificados na MESMA classe (o split não deve virar uma classe)."""
    for split in ("Training", "Testing"):
        for nome, n in [("glioma", 5), ("meningioma", 5), ("notumor", 5), ("pituitary", 5)]:
            _cria_imagens(tmp_path / split / nome, n)

    registros = listar_imagens(tmp_path)
    assert len(registros) == 40
    c = Counter(r["categoria"] for r in registros)
    assert c == {"GLIOMA": 10, "MENINGIOMA": 10, "NOTUMOR": 10, "PITUITARY": 10}
    assert c.get("INDEFINIDO", 0) == 0
    # cada categoria deve ter um único label consistente entre Training e Testing
    labels_por_categoria = {}
    for r in registros:
        labels_por_categoria.setdefault(r["categoria"], set()).add(r["label"])
    assert all(len(s) == 1 for s in labels_por_categoria.values())


def test_pastas_binarias_vazias_nao_atrapalham_multiclasse_real(tmp_path):
    """Regressão exata do caso reportado: benign/malignant vazias
    coexistindo com a estrutura multi-classe real."""
    (tmp_path / "benign").mkdir()
    (tmp_path / "malignant").mkdir()
    for split in ("Training", "Testing"):
        for nome in ("glioma", "meningioma", "notumor", "pituitary"):
            _cria_imagens(tmp_path / split / nome, 3)

    registros = listar_imagens(tmp_path)
    assert len(registros) == 24  # 4 classes x 3 imagens x 2 splits
    assert Counter(r["categoria"] for r in registros).get("INDEFINIDO", 0) == 0


def test_convencao_binaria_tem_prioridade_sobre_multiclasse(tmp_path):
    """Quando a base já tem uma convenção binária clara com imagens de
    verdade, usa BENIGNO/MALIGNO (mais informativo clinicamente) em vez de
    tratar como classes multi-nomeadas genéricas."""
    _cria_imagens(tmp_path / "benign", 5)
    _cria_imagens(tmp_path / "malignant", 5)

    registros = listar_imagens(tmp_path)
    c = Counter(r["categoria"] for r in registros)
    assert c == {"BENIGNO": 5, "MALIGNO": 5}


def test_muitas_pastas_distintas_nao_viram_classes_falso_positivo(tmp_path):
    """Proteção contra falso positivo: uma pasta por paciente (dezenas de
    nomes distintos) não deve virar um problema de classificação de
    dezenas de classes."""
    for i in range(30):
        _cria_imagens(tmp_path / f"paciente_{i:03d}", 2)

    registros = listar_imagens(tmp_path)
    assert Counter(r["categoria"] for r in registros) == Counter(INDEFINIDO=60)


def test_unica_pasta_continua_indefinida():
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    _cria_imagens(tmp / "todas_as_imagens", 10)
    registros = listar_imagens(tmp)
    assert all(r["categoria"] == "INDEFINIDO" for r in registros)
