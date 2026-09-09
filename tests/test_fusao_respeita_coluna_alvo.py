"""Testes para: a fusão multimodal precisa respeitar a coluna-alvo escolhida
explicitamente pelo usuário (mesmo quando outra modalidade, ex. imagem, já
tem um rótulo próprio válido) — e nunca quebrar silenciosamente com colunas
multi-classe (a fusão hoje só suporta binário)."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.app import _processar_fusao_multimodal


def _montar_base_com_imagem_rotulada(csv_conteudo: str, n=40):
    tmp = Path(tempfile.mkdtemp())
    pasta_run = tmp / "run"
    pasta_run.mkdir()
    registros_img = []
    for i in range(n):
        registros_img.append({
            "caminho": f"/x/paciente_{i:03d}.png",
            "label": i % 2,  # imagem JÁ tem rótulo binário próprio (benign/malignant)
            "categoria": "MALIGNO" if i % 2 else "BENIGNO",
            "biomarcadores": {
                "morfologia": {"circularidade": 0.5, "solidez": 0.6},
                "textura_glcm": {"contraste": 1.0, "homogeneidade": 0.4, "energia": 0.2, "entropia": 3.0},
                "distribuicao_intensidade": {"snr": 5.0, "assimetria": 0.1, "curtose": 2.5},
            },
        })
    (pasta_run / "biomarcadores.json").write_text(json.dumps(registros_img))
    (tmp / "clinico.csv").write_text(csv_conteudo)
    return tmp, pasta_run


def test_coluna_binaria_escolhida_tem_prioridade_sobre_rotulo_da_imagem():
    # diagnostico segue um padrão BEM diferente do rótulo alternado da imagem
    csv = "image_id,idade,diagnostico\n" + "\n".join(
        f"paciente_{i:03d},{40+i},{'sim' if i < 5 else 'nao'}" for i in range(40)
    )
    tmp, pasta_run = _montar_base_com_imagem_rotulada(csv)

    import biostatusia.pipeline.fusao_multimodal as fm_mod
    y_capturado = {}
    original = fm_mod.treinar_fusao_tardia

    def fake(modalidades, y, *a, **kw):
        y_capturado["y"] = y.copy()
        return original(modalidades, y, *a, **kw)

    fm_mod.treinar_fusao_tardia = fake
    try:
        _processar_fusao_multimodal(tmp, pasta_run, "RandomForest", "LogisticRegression",
                                     coluna_alvo_escolhida="diagnostico")
    finally:
        fm_mod.treinar_fusao_tardia = original

    y = y_capturado["y"]
    # Se a fusão tivesse usado o rótulo alternado da imagem (0,1,0,1,...) em vez
    # da coluna "diagnostico" (5 "sim" seguidos de "nao"), este padrão não bateria.
    assert y[:5].tolist() == [1, 1, 1, 1, 1]
    assert y[5:10].tolist() == [0, 0, 0, 0, 0]


def test_sem_escolha_explicita_ainda_usa_fallback_binario_de_qualquer_modalidade():
    # Comportamento antigo preservado quando NADA foi escolhido explicitamente.
    csv = "image_id,idade,diagnostico\n" + "\n".join(
        f"paciente_{i:03d},{40+i},{'sim' if i % 2 else 'nao'}" for i in range(40)
    )
    tmp, pasta_run = _montar_base_com_imagem_rotulada(csv)
    res = _processar_fusao_multimodal(tmp, pasta_run, "RandomForest", "LogisticRegression")
    assert res.get("fusao_multimodal", {}).get("metricas")


def test_coluna_multiclasse_agora_funciona_na_fusao():
    csv = "image_id,idade,Medical Condition\n" + "\n".join(
        f"paciente_{i:03d},{40+i},{['Diabetes','Hypertension','Asthma','Obesity'][i%4]}" for i in range(40)
    )
    tmp, pasta_run = _montar_base_com_imagem_rotulada(csv)
    res = _processar_fusao_multimodal(tmp, pasta_run, "RandomForest", "LogisticRegression",
                                       coluna_alvo_escolhida="Medical Condition")
    fm = res.get("fusao_multimodal", {})
    assert fm.get("metricas")  # a fusão agora treina de verdade com 4 classes
    assert fm.get("n_classes") == 4
    assert fm.get("melhor_modelo") is not None
