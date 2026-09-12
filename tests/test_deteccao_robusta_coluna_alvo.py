"""Testes para a detecção robusta de coluna-alvo — cobre dois casos reais
que causavam 'N/A' silencioso mesmo com uma coluna-alvo perfeitamente
válida presente:
1. Coluna-alvo no MEIO do arquivo (não primeira nem última, nome não
   reconhecido) — ex.: dataset UCI Parkinsons ("status").
2. Arquivo ORDENADO/AGRUPADO por classe — uma amostra do início nunca veria
   a diversidade de classes que só aparece mais adiante no arquivo — ex.:
   MIT-BIH ECG (18 mil linhas de uma classe, depois as outras)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.dados_tabulares import detectar_schema, candidatos_coluna_alvo


def test_detecta_coluna_alvo_no_meio_do_arquivo():
    """Regressão do caso real UCI Parkinsons: 'status' é a 17ª de 23
    colunas, não a primeira nem a última, e não está em LABEL_KEYWORDS."""
    header = ["name", "f1", "f2", "status", "f3", "f4", "f5"]
    data = [[f"p{i}", str(i), str(i * 2), "1" if i % 3 == 0 else "0", str(i), str(i), str(i)]
            for i in range(100)]

    schema = detectar_schema(header, data)
    assert schema["label_name"] == "status"


def test_nao_auto_seleciona_quando_ha_2_candidatas_no_meio():
    """Quando há AMBIGUIDADE real (2+ candidatas), não escolhe sozinho —
    deixa para o portão de confirmação perguntar ao usuário."""
    header = ["name", "cond_a", "f1", "cond_b", "f2"]
    data = [[f"p{i}", "sim" if i % 2 else "nao", str(i), "x" if i % 2 else "y", str(i)]
            for i in range(50)]

    schema = detectar_schema(header, data)
    assert schema["label_idx"] is None  # ambíguo — não deveria auto-escolher


def test_detecta_coluna_alvo_em_arquivo_ordenado_por_classe():
    """Regressão do caso real MIT-BIH: arquivo com milhares de linhas de
    uma única classe seguidas, diversidade só aparece bem mais adiante."""
    header = [str(i) for i in range(10)]
    # 500 linhas de classe "0", depois 50 de "1", 30 de "2" -- tudo na ultima coluna
    data = []
    for i in range(500):
        data.append([str(i)] * 9 + ["0"])
    for i in range(50):
        data.append([str(i)] * 9 + ["1"])
    for i in range(30):
        data.append([str(i)] * 9 + ["2"])

    schema = detectar_schema(header, data)
    assert schema["label_name"] == "9"  # última coluna (índice 9, header[9]="9")


def test_candidatos_coluna_alvo_encontra_classe_so_no_final_do_arquivo():
    header = [str(i) for i in range(5)]
    data = [[str(i)] * 4 + ["0"] for i in range(1000)] + [[str(i)] * 4 + ["1"] for i in range(200)]

    candidatos = candidatos_coluna_alvo(header, data)
    nomes = [c["nome"] for c in candidatos]
    assert "4" in nomes  # a última coluna deveria ser encontrada mesmo com a classe 1 só no final
    cand = next(c for c in candidatos if c["nome"] == "4")
    assert cand["n_classes"] == 2


def test_deteccao_continua_rapida_em_dataset_largo():
    """A saída antecipada (early exit) deve manter a detecção rápida mesmo
    com muitas colunas de alta cardinalidade (features contínuas) — não
    deveria escanear a base inteira para colunas que claramente não são
    candidatas."""
    n_linhas, n_cols = 5000, 50
    header = [f"f{i}" for i in range(n_cols)] + ["target"]
    data = []
    for i in range(n_linhas):
        linha = [str(i * 0.137 + j) for j in range(n_cols)]  # todas continuas, alta cardinalidade
        linha.append("1" if i % 4 == 0 else "0")
        data.append(linha)

    t0 = time.perf_counter()
    schema = detectar_schema(header, data)
    dt = time.perf_counter() - t0
    assert schema["label_name"] == "target"
    assert dt < 5.0, f"deteccao muito lenta: {dt:.2f}s"
