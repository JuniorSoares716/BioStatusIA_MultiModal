# Plan — Laudo Populacional (Aba 4 — Seção B)

> Ref. spec: `./spec.md`

## 1. Visão técnica
`laudo_populacional()` carrega o resultado, opcionalmente calcula correlações sobre os
biomarcadores de sinal, e delega a `pipeline/relatorios.py` a montagem determinística do
Markdown e do pódio.

## 2. Rotas e handlers
| Rota | Método | Handler (`app.py`) | Entrada | Saída |
|------|--------|--------------------|---------|-------|
| `/laudo_populacional/<id>` | GET | `laudo_populacional(resultado_id)` | id | JSON `{laudo_html, laudo_md, podio}` |

## 3. Módulos envolvidos
- `database.py::buscar_resultado()`.
- `pipeline/relatorios.py`: `construir_laudo_populacional()`, `construir_podio()`,
  `correlacoes_biomarcadores()`.
- `pipeline/inferencia.py::achatar_biomarcadores()` (para vetorizar biomarcadores de sinal).
- `markdown` (lib) com extensão `tables`.

## 4. Contratos de dados
Entrada: `pipeline_data` persistido. Saída JSON: `laudo_html` (Markdown→HTML),
`laudo_md` (Markdown cru) e `podio` (lista de dicts com `posicao`, `modelo`, `campeao`, `auc`,
`sensibilidade`, `especificidade`, `f1`, `mcc`, `kappa`).

## 5. Fluxo de execução
1. `buscar_resultado(id)`; se `None` → 404.
2. Se houver `biomarcadores_sinal`: achatar vetores, alinhar largura mínima, nomear `f0..fn`,
   calcular `correlacoes_biomarcadores` e anexar ao pipeline.
3. `construir_laudo_populacional(pipeline)` → `laudo_md`.
4. `construir_podio(metricas, melhor)` → `podio`.
5. Retornar JSON com `md.markdown(laudo_md, extensions=["tables"])`.

## 6. Decisões de design
- **Determinístico e sem LLM** por princípio: reprodutibilidade acadêmica e auditabilidade.
- Correlações limitadas a |r| ≥ 0.7 e top-15/top-10 para manter o laudo legível.
- Pódio ordenado por AUC, coerente com o critério de adoção do AutoML (Art. 6).

## 7. Riscos e mitigações
- Poucas amostras (<3) → `correlacoes_biomarcadores` retorna lista vazia (guarda interna).
- `metricas` ausente → seções condicionais são omitidas sem quebrar o laudo.

## 8. Aderência à constituição
Art. 4 (`relatorios.py` = funções puras; I/O no handler), Art. 6 (pódio por AUC), Art. 7
(aviso ético ao final via `_AVISO_ETICO`).
