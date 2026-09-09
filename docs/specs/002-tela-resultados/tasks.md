# Tasks — Tela 2: Resultados (4 abas)

> Ref. spec: `./spec.md`  |  Ref. plan: `./plan.md`
> Status: `[ ]` pendente · `[~]` em andamento · `[x]` feito

## Tarefas

- [x] T-1 — Handler `tela2()` com 404 para id inexistente · verifica: CA-1 · `app.py`
- [x] T-2 — Aba 1 (biomarcadores, boxplot, schema, correlações) · verifica: RF-1, CA-2 · `templates/tela2_resultados.html`
- [x] T-3 — Aba 2 (análise da base, estratégia, engenharia de features + estado vazio) · verifica: RF-2 · `tela2_resultados.html`
- [x] T-4 — Aba 3 (pódio, ROC, matriz confusão, radar, McNemar) — só AutoML · verifica: RF-3, CA-6 · `tela2_resultados.html`
- [x] T-5 — Aba 4 (dossiê + histórico + Seção A + Seção B) · verifica: RF-4 · `tela2_resultados.html`
- [x] T-6 — `GET /api/historico` (≤30) · verifica: CA-4 · `app.py`, `database.py`
- [x] T-7 — `GET /api/exemplos/<id>` (≤50) · verifica: CA-5 · `app.py`
- [x] T-8 — Aviso ético visível na página · verifica: CA-3 · `tela2_resultados.html`

## Melhorias sugeridas (backlog)
- [ ] B-1 — Exportar a Aba 3 (pódio/ROC) como PNG/CSV.
- [ ] B-2 — Teste de fumaça renderizando um `pipeline_json` de exemplo por família.

## Verificação final
- [x] Todos os CA cobertos por ao menos uma task.
- [x] Art. 7 (aviso ético) e Art. 9 (template como verdade) respeitados.
