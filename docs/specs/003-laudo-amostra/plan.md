# Plan — Laudo de Amostra (Aba 4 — Seção A)

> Ref. spec: `./spec.md`

## 1. Visão técnica
`laudo_amostra()` tem dois caminhos: (1) `resultado_id` do banco — monta contexto a partir do
`pipeline_data` e do exemplar escolhido; (2) upload avulso — salva o arquivo, detecta tipo,
extrai biomarcadores, roda inferência individual. Em ambos, chama a crew interativa e
persiste o laudo.

## 2. Rotas e handlers
| Rota | Método | Handler (`app.py`) | Entrada | Saída |
|------|--------|--------------------|---------|-------|
| `/laudo_amostra` | POST | `laudo_amostra()` | `arquivo` \| `resultado_id` (+ `exemplo_idx`) | JSON `{laudo_html, laudo_id}` |

## 3. Módulos envolvidos
- Detecção/extração: `io_utils`, `pipeline/extracao.py`, `pipeline/extracao_temporal.py`,
  `pipeline/extracao_dicom.py`, `pipeline/dados_tabulares.py`, `pipeline/io_sinais.py`.
- Inferência: `pipeline/inferencia.py` (`achatar_biomarcadores`, `prever_exemplar`,
  `carregar_modelo_vencedor`, `carregar_vencedor_mais_recente`).
- Agente: `crew.py::BioStatusIACrewInterativo` + `config/agents.yaml`
  (`radiologista_ia_interativo`) + `config/tasks.yaml` (`tarefa_laudo_interativo`).
- Persistência: `database.py::salvar_laudo_interativo()`.

## 4. Contratos de dados
`biomarcadores_ctx` (dict) é o contexto passado ao agente via `biomarcadores_trecho` (JSON).
Campos: `fonte`, `familia`, `tipo_sinal`, `modo`, `biomarcadores`/`exemplo_biomarcadores`,
`inferencia_modelo` (categoria + `probabilidade_positiva` + `metricas_validacao`), e, no caso
de banco, `metricas_modelo`. Resposta: `{ laudo_html, laudo_id }`.

## 5. Fluxo de execução
1. Ler `resultado_id` / `arquivo`.
2. **Caso banco**: `buscar_resultado` → montar contexto + exemplar (`exemplo_idx`).
3. **Caso upload**: salvar em `static/uploads/`, `detectar_estrutura`, ramificar por tipo,
   extrair biomarcadores, `prever_exemplar(vetor, familia)`.
4. `BioStatusIACrewInterativo().crew().kickoff(inputs=...)`.
5. `salvar_laudo_interativo` → JSON `{laudo_html, laudo_id}`.

## 6. Decisões de design
- Não reexecuta o pipeline completo: a Seção A é intencionalmente "barata" e pontual.
- Inferência com **ajuste defensivo de dimensionalidade** (trunca/preenche) e **fallback**
  para o vencedor mais recente, para nunca falhar por incompatibilidade de features.

## 7. Riscos e mitigações
- Vetor de features incompatível com o modelo → ajuste defensivo em `prever_exemplar`.
- LLM indisponível → `try/except` devolve mensagem de erro como laudo (não derruba a rota).
- Extração falha → `erro_extracao` anexado, laudo ainda é gerado com o que houver.

## 8. Aderência à constituição
Art. 8 (agente sem tools, `max_iter=4`), Art. 7 (5ª seção = aviso ético), Art. 3/4 (contrato
de sinal, I/O concentrada em `app.py`/`database.py`).
