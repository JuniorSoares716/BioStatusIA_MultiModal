# Tasks — Laudo de Amostra (Aba 4 — Seção A)

> Ref. spec: `./spec.md`  |  Ref. plan: `./plan.md`
> Status: `[ ]` pendente · `[~]` em andamento · `[x]` feito

## Tarefas

- [x] T-1 — Rota `POST /laudo_amostra` aceitando `arquivo` OU `resultado_id` · verifica: RF-1, CA-1 · `app.py`
- [x] T-2 — Caso banco: montar contexto + exemplar via `exemplo_idx` · verifica: RF-1, CA-2 · `app.py`
- [x] T-3 — Caso upload: detectar tipo e extrair biomarcadores por família · verifica: RF-2, CA-3 · `app.py`, `pipeline/extracao*.py`
- [x] T-4 — Inferência individual com vencedor persistido + fallback · verifica: RF-3, CA-3 · `pipeline/inferencia.py`
- [x] T-5 — Crew interativa com 5 seções obrigatórias · verifica: RF-4, RF-5, CA-5 · `crew.py`, `config/tasks.yaml`
- [x] T-6 — Persistir laudo e retornar `{laudo_html, laudo_id}` · verifica: RF-6, CA-4 · `database.py`
- [x] T-7 — Aviso ético como 5ª seção · verifica: CA-5 · `config/tasks.yaml`

## Melhorias sugeridas (backlog)
- [ ] B-1 — Exibir explicitamente `inferencia_modelo` (categoria/probabilidade) no HTML da Seção A.
- [ ] B-2 — Teste unitário de `prever_exemplar` com ajuste de dimensionalidade.

## Verificação final
- [x] Todos os CA cobertos por ao menos uma task.
- [x] Art. 8 (sem tools, `max_iter=4`) e Art. 7 (aviso ético) respeitados.
