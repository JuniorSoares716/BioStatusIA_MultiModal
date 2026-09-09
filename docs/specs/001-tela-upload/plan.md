# Plan — Tela 1: Upload & Detecção de Modo

> Ref. spec: `./spec.md`

## 1. Visão técnica
`GET /` renderiza o formulário. `POST /analisar` resolve o caminho do dataset, detecta o
modo, dispara a crew adequada, consolida o `pipeline_data`, persiste e redireciona.

## 2. Rotas e handlers
| Rota | Método | Handler (`app.py`) | Entrada | Saída |
|------|--------|--------------------|---------|-------|
| `/` | GET | `tela1()` | — | `tela1_upload.html` |
| `/analisar` | POST | `analisar()` | `arquivo` \| `caminho_manual` \| `kaggle_id` (+ `tipo_sinal`) | redirect `→ /resultados/<id>` |

## 3. Módulos envolvidos
- `pipeline/io_utils.py`: `eh_imagem/eh_tabular/eh_sinal_temporal/eh_dicom/eh_volumetrico`,
  `listar_imagens`, `encontrar_csv`, `label_pasta`, `criar_pasta_run`.
- `app.py::detectar_estrutura()`: classificação em 9 modos.
- `crew.py`: `BioStatusIACrew`, `BioStatusIACrewTabular`, `BioStatusIACrewSinal`,
  `BioStatusIACrewImagem3D`.
- Consolidação: `_consolidar_imagem()`, `_consolidar_sinal()`.
- `database.py`: `salvar()`, `salvar_resultado()`.

## 4. Contratos de dados
`pipeline_data` (dict → JSON em `resultados_pipeline.pipeline_json`) contém sempre `modo`,
`familia`, `n_imagens`. Campos adicionais por modo: `estatisticas`, `estrategia_preproc`,
`features_engenharia`, `biomarcadores` / `biomarcadores_sinal`, e — quando treinado —
`metricas`, `roc_data`, `confusion_matrix`, `melhor_modelo`, `comparacao_ab`, `shap`.

## 5. Fluxo de execução
1. Resolver origem: upload (salva em `static/uploads/`, extrai ZIP) → caminho manual →
   fallback KaggleHub.
2. `detectar_estrutura(base_path)` → modo.
3. `invalido` → HTTP 400. Senão `criar_pasta_run()`.
4. Despacho:
   - `tabular` → `BioStatusIACrewTabular` + `dados_tabulares` + `treinar_vetores` (se
     rotulado e ≥10 amostras).
   - `sinal_temporal` → `BioStatusIACrewSinal` → `_consolidar_sinal(..., "F1")`.
   - `imagem_dicom_2d` → `BioStatusIACrewImagem3D` → `_consolidar_sinal(..., "F3")`.
   - `volume_3d` → `BioStatusIACrewImagem3D` → `_consolidar_sinal(..., "F4")`.
   - imagem (`imagem_unica/imagens_soltas/dataset_rotulado/multimodal`) → `BioStatusIACrew`
     → `_consolidar_imagem()`; `multimodal` anexa análise tabular.
5. `salvar()` + `salvar_resultado()` → redirect `/resultados/<id>`.

## 6. Decisões de design
- Detecção baseada em extensão + inventário recursivo, não em conteúdo, por ser barata e
  determinística.
- Pasta de `.dcm`: ≥10 arquivos ⇒ série 3D (`volume_3d`), senão 2D — heurística de
  `detectar_estrutura()`.
- `multimodal` (imagem+tabular) tratado pela crew de imagem, com bloco tabular anexado.

## 7. Riscos e mitigações
- ZIP malicioso/enorme → limite de 4 GB; extração em subpasta isolada.
- Ollama indisponível → `kickoff` em `try/except`; o laudo recebe a mensagem de erro mas o
  pipeline estatístico segue e a Tela 2 ainda popula.
- Caminho manual inexistente → checagem `base_path.exists()` → HTTP 400.

## 8. Aderência à constituição
Art. 1/2 (9 modos, entrada flexível), Art. 4 (consolidação e I/O em `app.py`), Art. 5
(treino condicionado nos ramos), Art. 10 (limite 4 GB preservado).
