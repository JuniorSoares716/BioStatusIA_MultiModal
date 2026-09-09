"""Testes para biostatusia.validacao — validação de entrada do usuário."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biostatusia.validacao import (
    validar_kaggle_id, validar_caminho_manual, validar_tipo_sinal,
    validar_arquivo_upload, validar_entrada_analise,
)


def test_kaggle_id_valido_passa():
    r = validar_kaggle_id("paultimothymooney/chest-xray-pneumonia")
    assert r["valido"] is True
    assert r["valor"] == "paultimothymooney/chest-xray-pneumonia"
    assert r["corrigido_automaticamente"] is False


def test_kaggle_id_vazio_e_permitido():
    r = validar_kaggle_id("")
    assert r["valido"] is True
    assert r["valor"] == ""


def test_kaggle_url_dataset_e_autocorrigida():
    r = validar_kaggle_id("https://www.kaggle.com/datasets/koryto/countryinfo")
    assert r["valido"] is True
    assert r["valor"] == "koryto/countryinfo"
    assert r["corrigido_automaticamente"] is True


def test_kaggle_url_dataset_com_barra_final_e_autocorrigida():
    r = validar_kaggle_id("kaggle.com/datasets/owner/data/")
    assert r["valido"] is True
    assert r["valor"] == "owner/data"


def test_kaggle_url_notebook_da_erro_claro():
    r = validar_kaggle_id(
        "https://www.kaggle.com/code/paulohernane/"
        "melanoma-cancer-detection-with-transfer-learning/input"
    )
    assert r["valido"] is False
    assert "notebook" in r["erro"].lower()
    assert r["exemplo_correto"]


def test_kaggle_url_competicao_da_erro_claro():
    r = validar_kaggle_id("https://www.kaggle.com/competitions/titanic")
    assert r["valido"] is False
    assert "competi" in r["erro"].lower()


def test_kaggle_dominio_sem_padrao_dataset_da_erro():
    # Caso de borda: "www.kaggle.com/blah" bate no regex ingênuo de "algo/algo",
    # mas não é um ID válido — precisa ser pego pela checagem de "kaggle.com".
    r = validar_kaggle_id("www.kaggle.com/blah")
    assert r["valido"] is False


def test_kaggle_formato_generico_invalido():
    r = validar_kaggle_id("lung cancer dataset")
    assert r["valido"] is False
    assert r["exemplo_correto"]


def test_caminho_manual_inexistente():
    r = validar_caminho_manual("/caminho/que/definitivamente/nao/existe/xyz123")
    assert r["valido"] is False
    assert "não foi encontrado" in r["erro"]


def test_caminho_manual_vazio_e_permitido():
    r = validar_caminho_manual("")
    assert r["valido"] is True


def test_caminho_manual_existente_passa(tmp_path):
    d = tmp_path / "dados"
    d.mkdir()
    r = validar_caminho_manual(str(d))
    assert r["valido"] is True


def test_tipo_sinal_valido():
    assert validar_tipo_sinal("ECG")["valido"] is True
    assert validar_tipo_sinal("auto")["valido"] is True


def test_tipo_sinal_invalido():
    r = validar_tipo_sinal("RAIOX")
    assert r["valido"] is False
    assert r["exemplo_correto"]


def test_arquivo_extensao_suportada():
    assert validar_arquivo_upload("exame.csv")["valido"] is True
    assert validar_arquivo_upload("volume.nii.gz")["valido"] is True
    assert validar_arquivo_upload("")["valido"] is True  # nenhum arquivo enviado


def test_arquivo_extensao_nao_suportada():
    r = validar_arquivo_upload("laudo.pdf")
    assert r["valido"] is False
    assert r["exemplo_correto"]


def test_arquivo_tamanho_excede_limite():
    r = validar_arquivo_upload("exame.csv", tamanho_bytes=5 * 1024 ** 3)  # 5 GB > limite de 4 GB
    assert r["valido"] is False


def test_entrada_analise_prioriza_primeira_falha_na_ordem_certa():
    # arquivo inválido deve ser reportado antes de checar kaggle_id, já que
    # é a primeira fonte tentada pelo /analisar.
    r = validar_entrada_analise(
        arquivo_nome="laudo.pdf", arquivo_tamanho=100,
        caminho_manual="", kaggle_id="link errado",
        tipo_sinal="auto",
    )
    assert r["campo"] == "arquivo"


def test_entrada_analise_tudo_valido_normaliza_kaggle():
    r = validar_entrada_analise(
        arquivo_nome=None, arquivo_tamanho=None,
        caminho_manual="", kaggle_id="https://www.kaggle.com/datasets/koryto/countryinfo",
        tipo_sinal="auto",
    )
    assert r["valido"] is True
    assert r["kaggle_id_normalizado"] == "koryto/countryinfo"
    assert r["corrigido_automaticamente"] is True


def test_entrada_analise_nenhuma_fonte_e_sinalizado():
    r = validar_entrada_analise(
        arquivo_nome=None, arquivo_tamanho=None,
        caminho_manual="", kaggle_id="", tipo_sinal="auto",
    )
    assert r["valido"] is True
    assert r["nenhuma_fonte_informada"] is True
