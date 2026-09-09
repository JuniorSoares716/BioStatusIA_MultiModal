# Spec — Laudo de Amostra (Aba 4 — Seção A)

> **ID:** 003  |  **Status:** Implementada
> **Rota:** `POST /laudo_amostra`  |  **Template:** seção da Aba 4 em `tela2_resultados.html`

## 1. Objetivo (por quê)
Gerar um laudo clínico **focado em uma única amostra** — seja um arquivo avulso enviado na
hora, seja um exemplar de uma análise já existente no banco — sem reexecutar o pipeline
completo. Complementa a inferência do vencedor do pódio com a interpretação do agente.

## 2. Usuários e contexto
Médico querendo um segundo olhar sobre um exame específico (um ECG, um Raio-X) fora do fluxo
populacional.

## 3. Histórias de usuário
- Como médico, quero enviar um arquivo avulso e receber um laudo focado, para avaliar um caso
  pontual.
- Como médico, quero escolher um exemplar de uma análise anterior e obter o laudo daquele
  exemplar específico.
- Como médico, quero que, havendo modelo treinado da família, o laudo traga a classificação
  inferida (categoria + probabilidade).

## 4. Requisitos funcionais
- RF-1: Aceitar `multipart` com **`arquivo`** (upload) **ou** **`resultado_id`** (análise
  existente); opcionalmente `exemplo_idx`.
- RF-2: Para upload, detectar tipo e extrair biomarcadores conforme a família (imagem→F3,
  sinal→F1, DICOM→F3, tabular→stats).
- RF-3: Havendo vetor de biomarcadores, rodar inferência individual com o vencedor persistido
  (`prever_exemplar`), com fallback para o vencedor mais recente.
- RF-4: Disparar `BioStatusIACrewInterativo` (agente `radiologista_ia_interativo`) com o
  contexto montado.
- RF-5: O laudo deve conter as **5 seções obrigatórias**: Achado Principal, Severidade (1–5),
  Comparação com Referência, Recomendação Imediata e Aviso Ético.
- RF-6: Persistir em `laudos_interativos` e retornar `{ laudo_html, laudo_id }`.

## 5. Requisitos não-funcionais
- RNF-1: Agente `radiologista_ia_interativo` sem ferramentas, `max_iter=4`.
- RNF-2: Erros de extração/ inferência não abortam o laudo — são anexados ao contexto
  (`erro_extracao`, `inferencia_erro`).

## 6. Regras de negócio / restrições da constituição
- Art. 7 (aviso ético — 5ª seção), Art. 8 (agente de laudo sem tools, `max_iter` fixo),
  Art. 3/4 (contrato de sinal e I/O concentrada).

## 7. Critérios de aceite
- CA-1: Sem `arquivo` nem `resultado_id` → HTTP 400.
- CA-2: `resultado_id` inexistente → HTTP 404.
- CA-3: Upload de imagem → família F3, biomarcadores extraídos e (se houver modelo)
  `inferencia_modelo` presente no contexto.
- CA-4: Resposta contém `laudo_html` e `laudo_id`; registro criado em `laudos_interativos`.
- CA-5: O laudo apresenta as 5 seções obrigatórias, incluindo o aviso ético.

## 8. Fora de escopo
Seleção de ROI/slice/frame (removida na v3). Laudo populacional (spec 004).
