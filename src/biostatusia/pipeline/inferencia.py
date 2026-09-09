"""
Inferência individual — isola o vencedor do pódio (AutoML) e classifica um
único exemplar novo (ex.: um novo ECG ou Raio-X submetido pelo médico).

Contrato de persistência: cada família (F1/F3/F4/TAB) grava seu campeão em
`models/vencedor_<familia>.pkl`, contendo o modelo já treinado, o scaler
ajustado, os nomes das features e as métricas de validação. A inferência
recarrega esse artefato e devolve (categoria, probabilidade) sem re-treinar.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models"


def _caminho(familia: str) -> Path:
    fam = (familia or "GEN").replace("/", "_")
    return MODEL_DIR / f"vencedor_{fam}.pkl"


def salvar_modelo_vencedor(nome: str, modelo, scaler, familia: str,
                           feature_names: list[str] | None,
                           metricas: dict) -> dict:
    """Grava o campeão do pódio em disco. Retorna metadados do artefato."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artefato = {
        "nome": nome,
        "modelo": modelo,
        "scaler": scaler,
        "feature_names": feature_names or [],
        "familia": familia,
        "metricas": metricas,
    }
    destino = _caminho(familia)
    with open(destino, "wb") as f:
        pickle.dump(artefato, f)
    return {"arquivo": str(destino), "modelo": nome, "familia": familia}


def carregar_modelo_vencedor(familia: str) -> dict | None:
    caminho = _caminho(familia)
    if not caminho.exists():
        return None
    with open(caminho, "rb") as f:
        return pickle.load(f)


def achatar_biomarcadores(bio: dict) -> np.ndarray:
    """Achata um dicionário aninhado de biomarcadores num vetor numérico 1D."""
    v: list[float] = []
    for grupo in bio.values():
        if isinstance(grupo, dict):
            for val in grupo.values():
                if isinstance(val, (int, float)):
                    v.append(float(val))
                elif isinstance(val, list) and all(isinstance(x, (int, float)) for x in val):
                    v.extend(float(x) for x in val[:5])
        elif isinstance(grupo, (int, float)):
            v.append(float(grupo))
    return np.asarray(v, dtype=np.float32)


def carregar_vencedor_mais_recente() -> dict | None:
    """Fallback: o vencedor persistido mais recente, qualquer família."""
    if not MODEL_DIR.exists():
        return None
    candidatos = sorted(MODEL_DIR.glob("vencedor_*.pkl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidatos:
        return None
    with open(candidatos[0], "rb") as f:
        return pickle.load(f)


def prever_exemplar(vetor: np.ndarray, familia: str, permitir_fallback: bool = True) -> dict:
    """
    Classifica um único exemplar usando o vencedor persistido da família.
    `vetor`: 1D (n_features,) ou 2D (1, n_features).
    Retorna dict com categoria, probabilidade e o modelo usado — ou um aviso.

    Funciona tanto para o vencedor binário quanto multi-classe: usa o vetor
    de probabilidade inteiro (não só a coluna 1), e reporta a classe de
    maior probabilidade — sem assumir que a classe 1 é sempre "a positiva".
    """
    artefato = carregar_modelo_vencedor(familia)
    usou_fallback = False
    if artefato is None and permitir_fallback:
        artefato = carregar_vencedor_mais_recente()
        usou_fallback = artefato is not None
    if artefato is None:
        return {"disponivel": False, "aviso": f"Nenhum modelo treinado para a família {familia}."}

    X = np.asarray(vetor, dtype=np.float32).reshape(1, -1)
    esperado = len(artefato["feature_names"]) or (
        artefato["scaler"].n_features_in_ if artefato.get("scaler") is not None else X.shape[1]
    )
    if X.shape[1] != esperado:
        # Ajuste defensivo: trunca ou preenche com zeros para casar a dimensionalidade.
        ajustado = np.zeros((1, esperado), dtype=np.float32)
        n = min(X.shape[1], esperado)
        ajustado[0, :n] = X[0, :n]
        X = ajustado

    scaler = artefato.get("scaler")
    X_s = scaler.transform(X) if scaler is not None else X
    modelo = artefato["modelo"]
    proba = modelo.predict_proba(X_s)[0]
    classes_brutas = list(getattr(modelo, "classes_", range(len(proba))))
    # sklearn expõe .classes_ como np.int64/np.str_ etc — não são serializáveis
    # em JSON diretamente; converte para tipos nativos do Python.
    classes = [c.item() if hasattr(c, "item") else c for c in classes_brutas]
    idx_top = int(np.argmax(proba))
    classe_prevista = classes[idx_top]
    prob_top = float(proba[idx_top])

    resultado = {
        "disponivel": True,
        "modelo": artefato["nome"],
        "familia": familia,
        "familia_modelo": artefato.get("familia", familia),
        "usou_fallback": usou_fallback,
        "n_classes": len(classes),
        "classe_prevista": classe_prevista,
        "probabilidade": round(prob_top, 4),
        "distribuicao_probabilidade": {str(c): round(float(p), 4) for c, p in zip(classes, proba)},
        "metricas_validacao": artefato.get("metricas", {}),
    }

    if len(classes) == 2 and set(classes) == {0, 1}:
        # Caso binário clássico (0/1) — mantém os campos e o rótulo usados
        # antes, para não quebrar quem já lê "probabilidade_positiva"/"categoria".
        prob_pos = float(proba[classes.index(1)])
        resultado["probabilidade_positiva"] = round(prob_pos, 4)
        resultado["categoria"] = "MALIGNO/POSITIVO" if prob_pos >= 0.5 else "BENIGNO/NEGATIVO"
    else:
        resultado["categoria"] = str(classe_prevista)

    return resultado
