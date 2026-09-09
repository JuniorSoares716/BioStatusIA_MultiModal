"""Testes para codificação one-hot de colunas categóricas de baixa
cardinalidade (antes descartadas silenciosamente por não serem numéricas)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.pipeline.dados_tabulares import detectar_schema, extrair_features
from biostatusia.pipeline.classificador import treinar_vetores


def _dataset_healthcare_like(n=300, seed=1):
    import random
    rng = random.Random(seed)
    header = ["Patient_ID", "Age", "Gender", "Blood Type", "Admission Type", "Doctor", "Medical Condition"]
    generos = ["Male", "Female"]
    tipos_sangue = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    admissao = ["Emergency", "Elective", "Urgent"]
    medicos = [f"Dr{i}" for i in range(200)]  # alta cardinalidade — não deve virar feature
    condicoes = ["Diabetes", "Hypertension", "Asthma"]
    data = []
    for i in range(n):
        data.append([
            f"P{i}", str(20 + i % 60), rng.choice(generos), rng.choice(tipos_sangue),
            rng.choice(admissao), rng.choice(medicos), rng.choice(condicoes),
        ])
    return header, data


def test_colunas_categoricas_de_baixa_cardinalidade_sao_detectadas():
    header, data = _dataset_healthcare_like()
    schema = detectar_schema(header, data)
    assert set(schema["categorical_names"]) == {"Gender", "Blood Type", "Admission Type"}


def test_coluna_de_alta_cardinalidade_e_excluida():
    header, data = _dataset_healthcare_like()
    schema = detectar_schema(header, data)
    assert "Doctor" not in schema["categorical_names"]
    assert "Patient_ID" not in schema["categorical_names"]


def test_feature_names_onehot_tem_formato_coluna_igual_valor():
    header, data = _dataset_healthcare_like()
    schema = detectar_schema(header, data)
    assert "Gender=Male" in schema["feature_names"]
    assert "Gender=Female" in schema["feature_names"]


def test_x_tem_o_numero_certo_de_colunas():
    header, data = _dataset_healthcare_like()
    schema = detectar_schema(header, data)
    X, y, _ = extrair_features(data, schema)
    assert X.shape[1] == schema["n_features"]
    assert X.shape[1] == len(schema["feature_names"])
    # 1 numérica (Age) + 2 (Gender) + 8 (Blood Type) + 3 (Admission Type) = 14
    assert X.shape[1] == 14


def test_linha_onehot_tem_exatamente_uma_marcacao_por_coluna_categorica():
    header, data = _dataset_healthcare_like()
    schema = detectar_schema(header, data)
    X, y, _ = extrair_features(data, schema)
    # Gender: 2 colunas (soma 1), Blood Type: 8 colunas (soma 1), Admission Type: 3 colunas (soma 1)
    idx_gender = [i for i, n in enumerate(schema["feature_names"]) if n.startswith("Gender=")]
    idx_blood = [i for i, n in enumerate(schema["feature_names"]) if n.startswith("Blood Type=")]
    idx_adm = [i for i, n in enumerate(schema["feature_names"]) if n.startswith("Admission Type=")]
    assert np.all(X[:, idx_gender].sum(axis=1) == 1)
    assert np.all(X[:, idx_blood].sum(axis=1) == 1)
    assert np.all(X[:, idx_adm].sum(axis=1) == 1)


def test_sinal_apenas_categorico_agora_e_aprendido(monkeypatch, tmp_path):
    """Regressão-chave: uma coluna categórica com sinal real e nenhuma
    numérica útil — antes desta correção, o modelo não tinha como ver esse
    sinal (só usava colunas numéricas)."""
    monkeypatch.setattr("biostatusia.pipeline.classificador.MODEL_DIR", tmp_path)
    import random
    rng = random.Random(5)
    header = ["id", "idade_ruido", "Gender", "diagnostico"]
    data = []
    for i in range(400):
        genero = rng.choice(["Male", "Female"])
        acerta = rng.random() < 0.95
        if genero == "Male":
            diag = "positivo" if acerta else "negativo"
        else:
            diag = "negativo" if acerta else "positivo"
        data.append([f"p{i}", str(rng.randint(20, 80)), genero, diag])

    schema = detectar_schema(header, data)
    X, y, _ = extrair_features(data, schema)
    res = treinar_vetores(X, y, familia="tabular", selecao_features=False)
    melhor = res["melhor_modelo"]
    assert res["metricas"][melhor]["acuracia"] > 0.85  # bem acima do acaso (50%)
