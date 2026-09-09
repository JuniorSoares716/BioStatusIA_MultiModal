# Tasks — Laudo Populacional (Aba 4 — Seção B)

> Ref. spec: `./spec.md`  |  Ref. plan: `./plan.md`
> Status: `[ ]` pendente · `[~]` em andamento · `[x]` feito

## Tarefas

- [x] T-1 — Rota `GET /laudo_populacional/<id>` com 404 · verifica: CA-1 · `app.py`
- [x] T-2 — Montagem determinística do laudo (sem LLM) · verifica: RF-1, CA-2 · `pipeline/relatorios.py`
- [x] T-3 — Correlações de Pearson |r| ≥ 0.7 sobre biomarcadores de sinal · verifica: RF-2, CA-4 · `pipeline/relatorios.py`, `pipeline/inferencia.py`
- [x] T-4 — Pódio ordenado por AUC com campeão marcado · verifica: RF-3, CA-3 · `pipeline/relatorios.py`
- [x] T-5 — Seções de distribuição, balanceamento e SHAP (quando houver) · verifica: RF-4 · `pipeline/relatorios.py`
- [x] T-6 — Resposta `{laudo_html, laudo_md, podio}` · verifica: RF-5, CA-6 · `app.py`
- [x] T-7 — Aviso ético ao final (`_AVISO_ETICO`) · verifica: RF-6, CA-5 · `pipeline/relatorios.py`

## Melhorias sugeridas (backlog)
- [ ] B-1 — Nomear correlações com os nomes reais dos biomarcadores (hoje usa `f0..fn`).
- [ ] B-2 — Teste de determinismo: mesmo `pipeline_data` ⇒ `laudo_md` idêntico.

## Verificação final
- [x] Todos os CA cobertos por ao menos uma task.
- [x] Art. 6 (pódio por AUC) e Art. 7 (aviso ético) respeitados.
