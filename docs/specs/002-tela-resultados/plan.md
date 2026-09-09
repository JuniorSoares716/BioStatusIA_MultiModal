# Plan — Tela 2: Resultados (4 abas)

> Ref. spec: `./spec.md`

## 1. Visão técnica
`tela2()` busca o resultado por id, converte o laudo Markdown→HTML, carrega o histórico e
renderiza `tela2_resultados.html`, passando `pipeline_json` (string) que o JS do template
consome para montar tabelas e gráficos Plotly.

## 2. Rotas e handlers
| Rota | Método | Handler (`app.py`) | Entrada | Saída |
|------|--------|--------------------|---------|-------|
| `/resultados/<id>` | GET | `tela2(resultado_id)` | id | HTML renderizado |
| `/api/historico` | GET | `api_historico()` | — | JSON (≤30) |
| `/api/exemplos/<id>` | GET | `api_exemplos(resultado_id)` | id | JSON (≤50) |

## 3. Módulos envolvidos
- `database.py`: `buscar_resultado()`, `listar()`, `listar_resultados_completo()`.
- `markdown` (lib): conversão do laudo para HTML.
- Template `tela2_resultados.html`: 4 abas + JS de renderização (Plotly).

## 4. Contratos de dados
Render recebe: `dados` (linha do resultado), `pipeline` (dict), `pipeline_json` (str),
`laudo_html` (str), `historico` (lista). Chaves consumidas do `pipeline`: `estatisticas`,
`estrategia_preproc`, `features_engenharia`, `biomarcadores`/`biomarcadores_sinal`,
`tabular_stats`, `schema_tabular`, `metricas`, `roc_data`, `confusion_matrix`,
`melhor_modelo`, `comparacao_ab`, `shap`, `dados_viz`, `canais`.

## 5. Fluxo de execução
1. `buscar_resultado(id)`; se `None` → 404.
2. `md.markdown(dados["laudo"])` → `laudo_html`.
3. `listar()` → histórico do prontuário.
4. `render_template(...)`.
5. No cliente: JS lê `pipeline_json` e monta cada aba; abas 1/2/4 sempre; gráficos de AutoML
   condicionais a `metricas`.

## 6. Decisões de design
- Consolidação de sinal reusa o formato de `estatisticas` da imagem
  (`_estatisticas_sinal`) para reaproveitar tabela e boxplot em qualquer família.
- Laudo Markdown fica **apenas** na Aba 4 (dossiê), mantendo a Aba 3 exclusivamente
  quantitativa.

## 7. Riscos e mitigações
- `pipeline_json` grande (muitos biomarcadores) → consolidação limita listas (ex.: 50
  exemplos, `dados_viz` ≤2000 pontos).
- Ausência de treino → template esconde blocos de AutoML sem quebrar as demais abas.

## 8. Aderência à constituição
Art. 4 (busca/serialização em `app.py`/`database.py`), Art. 7 (aviso ético no template),
Art. 9 (template é a verdade; nenhum HTML de runtime é editado).
