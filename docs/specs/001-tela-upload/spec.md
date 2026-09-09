# Spec — Tela 1: Upload & Detecção de Modo

> **ID:** 001  |  **Status:** Implementada
> **Rotas:** `GET /`, `POST /analisar`  |  **Template:** `tela1_upload.html`

## 1. Objetivo (por quê)
Permitir que o usuário submeta qualquer dado biomédico em escopo (imagem, tabular, sinal
temporal, DICOM, volume 3D ou pastas/ZIP com combinações) e ter o tipo detectado
automaticamente, disparando o pipeline e a crew corretos sem exigir configuração manual.

## 2. Usuários e contexto
Pesquisador/médico no início do fluxo. Não sabe (nem deveria precisar saber) qual "modo"
técnico se aplica ao seu arquivo — o sistema decide.

## 3. Histórias de usuário
- Como pesquisador, quero arrastar um arquivo ou ZIP e clicar em analisar, para não ter de
  classificar manualmente o tipo do dado.
- Como usuário, quero informar um caminho local de pasta, para analisar dados que já estão
  no disco sem re-upload.
- Como usuário sem dados próprios, quero rodar com um dataset de exemplo (KaggleHub), para
  testar a plataforma.

## 4. Requisitos funcionais
- RF-1: Aceitar três formas de entrada: upload (arquivo/ZIP), caminho manual, ou vazio
  (fallback KaggleHub BUSI, ou `kaggle_id` informado).
- RF-2: ZIPs são extraídos automaticamente para `static/uploads/<nome>/`.
- RF-3: `detectar_estrutura()` classifica a entrada em um dos **9 modos em escopo** ou
  `invalido`.
- RF-4: Cada modo dispara a crew correspondente (imagem, tabular, sinal F1, imagem médica
  F3/F4) e consolida um `pipeline_data`.
- RF-5: Ao final, persistir em `analises` + `resultados_pipeline` e redirecionar para
  `GET /resultados/<id>`.
- RF-6: Para `sinal_temporal`, honrar o campo opcional `tipo_sinal` do formulário.

## 5. Requisitos não-funcionais
- RNF-1: `MAX_CONTENT_LENGTH = 4 GB` (volumes 3D).
- RNF-2: Detecção de pasta é recursiva (`rglob`) e reconhece rótulos por subpastas
  (`benign/`, `malignant/` e sinônimos).
- RNF-3: Falha de entrada retorna HTTP 400 com mensagem explicando os tipos aceitos.

## 6. Regras de negócio / restrições da constituição
- Art. 1 (entrada flexível), Art. 2 (9 modos fixos, sem F2/F5), Art. 5 (treino condicionado),
  Art. 10 (limite 4 GB).

## 7. Critérios de aceite
- CA-1: Dado um `.png`, o modo detectado é `imagem_unica` e a crew de imagem roda.
- CA-2: Dado um `.csv`, o modo é `tabular` e a crew tabular roda.
- CA-3: Dada pasta com subpastas `benign/` e `malignant/`, o modo é `dataset_rotulado`.
- CA-4: Dado `.edf/.mat/.dat`, o modo é `sinal_temporal` (F1).
- CA-5: Dado `.dcm` único → `imagem_dicom_2d`; pasta com ≥10 `.dcm` → `volume_3d`.
- CA-6: Dado `.nii/.nii.gz/.mha` → `volume_3d` (F4).
- CA-7: Entrada não reconhecida → `invalido` + HTTP 400 com lista de tipos aceitos.
- CA-8: Após sucesso, o usuário é redirecionado para a Tela 2 com um `resultado_id` válido.

## 8. Fora de escopo
Renderização de resultados (ver 002). Geração de laudos interativos (ver 003/004).
Reintrodução de áudio/vídeo.
