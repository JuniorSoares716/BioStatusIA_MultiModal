# Tasks — Tela 1: Upload & Detecção de Modo

> Ref. spec: `./spec.md`  |  Ref. plan: `./plan.md`
> Status: `[ ]` pendente · `[~]` em andamento · `[x]` feito

## Tarefas

- [x] T-1 — Formulário de upload com drag & drop + caminho manual · verifica: RF-1 · `templates/tela1_upload.html`
- [x] T-2 — Salvar upload e extrair ZIP em `static/uploads/` · verifica: RF-2 · `app.py::analisar()`
- [x] T-3 — `detectar_estrutura()` cobrindo os 9 modos + `invalido` · verifica: CA-1..CA-7 · `app.py`
- [x] T-4 — Despacho por modo para as 4 crews · verifica: RF-4 · `app.py::analisar()`, `crew.py`
- [x] T-5 — Consolidação `_consolidar_imagem` / `_consolidar_sinal` · verifica: RF-4 · `app.py`
- [x] T-6 — Treino condicionado (≥10 amostras, ≥2 classes) nos ramos tabular/multimodal · verifica: CA-2 · `pipeline/classificador.py`
- [x] T-7 — Persistência + redirect para Tela 2 · verifica: CA-8 · `database.py`, `app.py`
- [x] T-8 — Campo `tipo_sinal` opcional para F1 · verifica: RF-6 · `app.py::analisar()`
- [x] T-9 — HTTP 400 com lista de tipos aceitos em `invalido` · verifica: CA-7 · `app.py::analisar()`

## Melhorias sugeridas (backlog)
- [ ] B-1 — Testes automatizados de `detectar_estrutura()` por modo (fixtures em `tests/`).
- [ ] B-2 — Feedback de progresso na UI durante o `kickoff` (hoje é síncrono).

## Verificação final
- [x] Todos os CA cobertos por ao menos uma task.
- [x] Imports do pacote OK após reorganização (`python -c "import biostatusia"`).
