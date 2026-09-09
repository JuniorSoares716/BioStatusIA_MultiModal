# Constituição do BioStatusIA

Princípios invioláveis. Toda `spec`, `plan` e alteração de código deve respeitá-los.
Derivados das diretrizes do projeto (`CLAUDE.md`). Um artigo só muda por decisão explícita
registrada aqui.

## Art. 1 — Entrada flexível
O sistema aceita **qualquer tipo de dado em escopo** e detecta o tipo automaticamente em
`app.py::detectar_estrutura()`. A análise estatística **sempre** roda sobre o que for
recebido; o pipeline se adapta ao que é viável. Entradas fora de escopo caem em `invalido`
com mensagem clara — nunca falham em silêncio.

## Art. 2 — Escopo v3 fixo (F1/F3/F4 + tabular + imagem comum)
As famílias em escopo são **F1** (sinais temporais), **F3** (DICOM 2D) e **F4** (volume 3D),
mais dados tabulares e imagem comum. Áudio (F2) e vídeo (F5) foram **removidos** e não podem
ser reintroduzidos. Os **9 modos** de `detectar_estrutura()` são mantidos.

## Art. 3 — Contrato SinalNormalizado
`SinalNormalizado` (`pipeline/io_sinais.py`) é o contrato único entre leitores e extratores.
Não pode ser quebrado. Toda leitura passa por `carregar_sinal()`, que despacha para o leitor
correto e devolve esse dataclass.

## Art. 4 — Pipeline = funções puras
Cada etapa do `pipeline/` recebe e devolve dados, sem I/O. Toda I/O (banco, HTML, arquivos)
fica concentrada em `app.py` e `database.py`. Agentes comunicam-se por filesystem
(`static/runs/run_<timestamp>/`): as tools persistem JSONs completos e retornam apenas
resumo textual ao LLM.

## Art. 5 — Treino condicionado
Classificadores só são treinados em contexto rotulado e com **≥10 amostras e ≥2 classes**.
Fora disso, registra-se `aviso_classificador` — nunca se treina às cegas.

## Art. 6 — Rigor clínico do AutoML
O AutoML (`pipeline/avaliacao_modelos.py`) usa 6 modelos, 5-fold CV estratificado + holdout
20%, balanceamento SMOTE/ADASYN **apenas no treino** (sem vazamento), e métricas enriquecidas
(sensibilidade, especificidade, MCC, Kappa, ECE, McNemar, SHAP). Critério de adoção do
campeão: **maior AUC com preferência por sensibilidade ≥ 0.8** (minimiza falso-negativo).

## Art. 7 — Aviso ético obrigatório
O aviso de que os laudos **não substituem avaliação de médico habilitado** é obrigatório em:
toda saída visual da Tela 2, todo laudo do `radiologista_ia`, todo laudo do
`radiologista_ia_interativo` (5ª seção), e todo laudo do `bioestatistico`. Nenhuma mudança
pode removê-lo.

## Art. 8 — Agentes de laudo sem ferramentas
`radiologista_ia` e `radiologista_ia_interativo` **nunca** recebem tools. `max_iter` não
aumenta sem medir tempo de resposta. `Process.sequential` não vira `hierarchical` sem revisar
`manager_llm`.

## Art. 9 — Templates HTML são a fonte da verdade visual
`templates/tela1_upload.html` e `templates/tela2_resultados.html` são a verdade visual.
HTML gerado em runtime nunca é editado à mão.

## Art. 10 — Ferramental e limites
Dependências via `uv add` (nunca `pip install`). `MAX_CONTENT_LENGTH = 4 GB` não pode ser
reduzido (volumes 3D). Não versionar `.env`, `biostatusia.db`, `models/`,
`static/uploads/`.

## Art. 11 — SDD primeiro
Mudança de página/fluxo começa pela `spec.md`, passa pela `plan.md`, vira `tasks.md` e só
então vira código. O código serve à spec.
