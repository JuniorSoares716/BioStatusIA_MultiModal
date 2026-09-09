# Spec — Laudo Populacional (Aba 4 — Seção B)

> **ID:** 004  |  **Status:** Implementada
> **Rota:** `GET /laudo_populacional/<id>`  |  **Template:** seção da Aba 4

## 1. Objetivo (por quê)
Fornecer um relatório analítico **determinístico** (sem LLM) do dataset inteiro: distribuição
de classes, pódio do AutoML, correlações fortes entre biomarcadores, balanceamento e features
mais influentes (SHAP) — montado a partir do `pipeline_data` já persistido.

## 2. Usuários e contexto
Pesquisador avaliando a base como um todo (não um exemplar). Precisa de números reproduzíveis
e auditáveis.

## 3. Histórias de usuário
- Como pesquisador, quero um laudo populacional reproduzível (mesmo dado ⇒ mesmo laudo), para
  citar em trabalho acadêmico.
- Como cientista de dados, quero ver o pódio dos modelos e as correlações fortes entre
  biomarcadores em um relatório pronto.

## 4. Requisitos funcionais
- RF-1: Montar o laudo a partir do `pipeline_data` persistido, **sem** chamar o LLM.
- RF-2: Enriquecer com correlações de Pearson (|r| ≥ 0.7) calculadas sobre os biomarcadores
  de sinal, quando houver.
- RF-3: Construir o pódio ordenando por AUC (desc), marcando o campeão.
- RF-4: Incluir, quando disponíveis: distribuição de classes, balanceamento
  (SMOTE/ADASYN) e top features SHAP.
- RF-5: Retornar `{ laudo_html, laudo_md, podio }`.
- RF-6: Incluir o aviso ético ao final.

## 5. Requisitos não-funcionais
- RNF-1: Determinístico e sem dependência de rede/LLM.
- RNF-2: Markdown com extensão de tabelas para renderizar o pódio.

## 6. Regras de negócio / restrições da constituição
- Art. 6 (critério do campeão: AUC + sensibilidade ≥ 0.8), Art. 7 (aviso ético), Art. 4
  (I/O concentrada; `relatorios.py` são funções puras).

## 7. Critérios de aceite
- CA-1: `id` inexistente → HTTP 404.
- CA-2: O mesmo `pipeline_data` produz o mesmo `laudo_md` (determinismo).
- CA-3: Havendo `metricas`, o pódio vem ordenado por AUC com um campeão marcado.
- CA-4: Havendo biomarcadores de sinal, correlações |r| ≥ 0.7 aparecem (até 10).
- CA-5: O laudo termina com o aviso ético.
- CA-6: Resposta contém `laudo_html`, `laudo_md` e `podio`.

## 8. Fora de escopo
Laudo por exemplar (spec 003). Geração de laudo com LLM (esta via é determinística por
princípio).
