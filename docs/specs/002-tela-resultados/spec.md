# Spec — Tela 2: Resultados (4 abas)

> **ID:** 002  |  **Status:** Implementada
> **Rotas:** `GET /resultados/<id>`, `GET /api/historico`, `GET /api/exemplos/<id>`
> **Template:** `tela2_resultados.html`

## 1. Objetivo (por quê)
Apresentar, de forma navegável, todo o resultado de uma análise: estatísticas e
biomarcadores, decisões de pré-processamento e engenharia de features, comparação de modelos
(AutoML) e os laudos — a partir do `pipeline_data` persistido.

## 2. Usuários e contexto
Médico/pesquisador após submeter dados na Tela 1. Consome resultados; não reprocessa.

## 3. Histórias de usuário
- Como usuário, quero ver os biomarcadores e sua distribuição por classe, para entender o
  dado.
- Como usuário, quero entender qual pré-processamento foi aplicado e por quê.
- Como cientista de dados, quero comparar os 6 modelos (pódio, ROC, matriz de confusão,
  McNemar), para escolher o campeão com confiança.
- Como médico, quero ler o laudo do radiologista IA e o histórico do prontuário em um só
  lugar.

## 4. Requisitos funcionais
- RF-1: **Aba 1 — Estatísticas & Biomarcadores**: tabela de biomarcadores, boxplot, schema
  tabular, métricas univariadas e correlações. Sempre visível.
- RF-2: **Aba 2 — Pré-processamento & Engenharia de Features**: análise da base, estratégia
  adaptativa com justificativas e ranking de importância de features (estado vazio quando não
  há features derivadas). Sempre visível.
- RF-3: **Aba 3 — AutoML**: comparação dos 6 modelos, ROC, matriz de confusão, radar,
  McNemar. Gráficos só quando há treino. **Sem** laudo Markdown nesta aba.
- RF-4: **Aba 4 — Laudo**: dossiê do `radiologista_ia`, histórico do prontuário (SQLite),
  Seção A (Laudo de Amostra, ver 003) e Seção B (Laudo Populacional, ver 004).
- RF-5: `GET /api/historico` retorna os últimos 30 resultados para o seletor da Aba 4.
- RF-6: `GET /api/exemplos/<id>` retorna exemplos individuais do dataset da análise.

## 5. Requisitos não-funcionais
- RNF-1: Gráficos via Plotly.js (CDN); estilo via Tailwind (CDN).
- RNF-2: A página recebe `pipeline_json` serializado + `laudo_html` (Markdown→HTML) +
  `historico`.
- RNF-3: Funciona para qualquer família (imagem/tabular/F1/F3/F4) reusando os mesmos
  componentes de tabela/boxplot.

## 6. Regras de negócio / restrições da constituição
- Art. 4 (I/O só em `app.py`/`database.py`), Art. 7 (aviso ético visível), Art. 9 (o HTML do
  template é a verdade visual — não editar HTML de runtime).

## 7. Critérios de aceite
- CA-1: `GET /resultados/<id>` inexistente → HTTP 404 "Resultado não encontrado".
- CA-2: Abas 1, 2 e 4 aparecem sempre; gráficos de AutoML só quando há `metricas`.
- CA-3: O aviso ético aparece na saída visual.
- CA-4: `/api/historico` devolve ≤30 itens ordenados do mais recente.
- CA-5: `/api/exemplos/<id>` devolve ≤50 exemplos com `idx`, `nome`, `categoria`,
  `biomarcadores`.
- CA-6: A Aba 3 não exibe laudo Markdown (o laudo vive na Aba 4).

## 8. Fora de escopo
Geração dos laudos de amostra e populacional (specs 003 e 004). Reprocessamento de dados.
