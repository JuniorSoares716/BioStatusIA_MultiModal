"""Tier 1 — leitura e schema de dados tabulares."""
import numpy as np

from biostatusia.pipeline.dados_tabulares import (
    carregar_csv, detectar_schema, extrair_features,
)


def test_schema_por_nome():
    header = ["a", "b", "diagnosis"]
    data = [["1", "2", "M"], ["3", "4", "B"], ["5", "6", "M"]]
    s = detectar_schema(header, data)
    assert s["label_name"] == "diagnosis"
    assert s["feature_names"] == ["a", "b"]
    assert s["n_features"] == 2


def test_schema_fallback_ultima_coluna():
    # Sem nome-rótulo reconhecido → usa última coluna com 2–10 valores únicos
    header = ["a", "b", "c"]
    data = [["1", "2", "x"], ["3", "4", "y"], ["5", "6", "x"]]
    s = detectar_schema(header, data)
    assert s["label_idx"] == 2


def test_separador_ponto_e_virgula(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a;b;label\n1;2;M\n3;4;B\n", encoding="utf-8")
    header, data = carregar_csv(str(p))
    assert header == ["a", "b", "label"]
    assert len(data) == 2


def test_fallback_encoding_latin1(tmp_path):
    p = tmp_path / "d.csv"
    p.write_bytes("a,b,label\n1,2,Cão\n3,4,B\n".encode("latin-1"))
    res = carregar_csv(str(p))
    assert res is not None
    header, data = res
    assert len(data) == 2


def test_extrair_features_preserva_nan():
    header = ["a", "b", "label"]
    data = [["1", "nan", "M"], ["3", "4", "B"], ["5", "6", "M"]]
    s = detectar_schema(header, data)
    X, y, _ = extrair_features(data, s)
    # coluna b é numérica no schema, mas 'nan' vira NaN em extrair_features
    assert np.isnan(X[0, 1])
    assert y is not None and len(y) == 3
