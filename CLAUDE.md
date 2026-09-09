# BioStatusIA v2 — Diretrizes do Projeto para Claude

## O que é este projeto

Sistema de Apoio à Decisão Clínica (CDSS) para análise automatizada de sinais biomédicos com foco em **3 famílias de sinal + dados tabulares** (áudio F2 e vídeo F5 foram **removidos do escopo** na v3). O pipeline é **flexível e detecta automaticamente o tipo de entrada** em escopo e adapta o processamento ao que for viável. Combina **radiômica e extração multi-domínio** (OpenCV/scikit-image/MNE/wfdb/pydicom/nibabel), **AutoML com 6 classificadores e avaliação enriquecida** (Sensibilidade, Especificidade, MCC, Kappa, ECE, McNemar, SHAP, balanceamento SMOTE/ADASYN), **IA Multi-Agente** (5 Crews CrewAI + Ollama) e uma **interface web Flask** com design "clínico moderno" (prototipado no Stitch): tela de upload, tela de resultados em 4 abas (com laudo populacional e laudo de amostra) e página de histórico dedicada.

### Escopo v3 (F1/F3/F4 + tabular)

| Família | Tipos | Leitura |
|---|---|---|
| **F1 — Sinais Temporais** | ECG, EEG, EMG, EOG, PPG, PA, Espirometria | `mne`, `wfdb`, `scipy` |
| **F3 — Imagem DICOM 2D** | Raio-X, Mamografia, Ultrassom estático | `pydicom` |
| **F4 — Volume 3D** | TC, RM, PET/SPECT | `nibabel`, `SimpleITK` |
| **Tabular** | Features clínicas multimodais | parser próprio |

Contexto acadêmico: projeto de mestrado em IA na Saúde — Fortaleza, CE.

---

## Princípio fundamental — entrada flexível

> **O sistema aceita QUALQUER tipo de dado.** Sempre executa análise estatística sobre o que receber e adapta o pipeline ao que é viável.

Existem **9 modos em escopo** detectados automaticamente em `app.py::detectar_estrutura()` (v3 — sem `audio_biomedico` nem `video_medico`; entradas fora de escopo caem em `invalido`):

| Modo | O que é | Crew disparada |
|---|---|---|
| `imagem_unica` | Uma imagem `.png/.jpg/.bmp/.tif` | `BioStatusIACrew` |
| `imagens_soltas` | Pasta/ZIP com imagens sem rótulos | `BioStatusIACrew` |
| `dataset_rotulado` | Pasta com subpastas `benign/` e `malignant/` | `BioStatusIACrew` |
| `tabular` | CSV/TXT/TSV único ou pasta só com tabulares | `BioStatusIACrewTabular` |
| `multimodal` | Pasta/ZIP com imagens **e** CSV/TXT juntos | `BioStatusIACrew` + tabular |
| `sinal_temporal` | Arquivos `.dat/.hea/.edf/.bdf/.mat/.xml/.c3d` | `BioStatusIACrewSinal` |
| `imagem_dicom_2d` | Arquivo `.dcm` único (Raio-X, Mamografia, Ultrassom estático) | `BioStatusIACrewImagem3D` |
| `volume_3d` | `.nii/.nii.gz/.mha` ou pasta com ≥10 `.dcm` (TC, RM, PET/SPECT) | `BioStatusIACrewImagem3D` |
| `multimodal_expandido` | Mix de imagens/sinais/tabular | `BioStatusIACrew` + sub-crews |

A análise estatística sempre roda. Treino de classificadores só ocorre com rótulos e ≥10 amostras com 2 classes. Laudo IA roda apenas quando há sinal ou imagem disponível.

---

## 3 Famílias de Sinal (v3)

| Família | Tipos suportados | Biblioteca de leitura |
|---|---|---|
| **F1 — Sinais Temporais** | ECG, EEG, EMG, EOG, PPG, PA, Espirometria | `mne`, `wfdb`, `scipy` |
| **F3 — Imagem DICOM 2D** | Raio-X, Mamografia, Ultrassom estático | `pydicom` |
| **F4 — Volume 3D** | TC, RM, PET/SPECT | `nibabel`, `SimpleITK` |

### `SinalNormalizado` — dataclass de saída unificada

```python
@dataclass
class SinalNormalizado:
    familia: str          # "F1" | "F3" | "F4"
    tipo: str             # "ECG" | "EEG" | "Raio-X" | "TC" | ...
    dados: np.ndarray     # array bruto
    taxa_amostragem: float
    canais: list[str]
    metadados: dict
    caminho_original: str
    dados_viz: list       # ≤2000 pontos para Plotly (downsampled)
```

Toda leitura passa por `pipeline/io_sinais.py::carregar_sinal()` que despacha para o leitor correto e retorna `SinalNormalizado`.

---

## Estado atual do projeto (v3)

| Componente | Status |
|---|---|
| Motor de radiômica (`pipeline/extracao.py`) | Feito |
| Extratores F1/F3/F4 (`pipeline/extracao_*.py`) | Feito |
| Leitores F1/F3/F4 (`pipeline/leitura_*.py`) | Feito |
| Dispatcher universal (`pipeline/io_sinais.py`) | Feito |
| Ingestão validada com Pydantic (`pipeline/ingestao.py`) | Feito (v3) |
| AutoML 6 modelos + métricas enriquecidas + MCC/Kappa/SHAP/SMOTE (`pipeline/avaliacao_modelos.py`) | Feito |
| Inferência individual — vencedor do pódio (`pipeline/inferencia.py`) | Feito (v3) |
| Laudo populacional determinístico (`pipeline/relatorios.py`) | Feito (v3) |
| Ponto de extensão CNN (`pipeline/deep_learning.py`) | Scaffold (v3) |
| 5 Crews CrewAI | Feito |
| 8 agentes (`config/agents.yaml`) | Feito |
| 9 tasks (`config/tasks.yaml`) | Feito |
| 8 tools CrewAI (`tools/`) | Feito |
| Banco SQLite com 3 tabelas + migração v2 | Feito |
| Servidor Flask com rotas de análise + laudo populacional + laudo individual + histórico | Feito |
| Tela 1 — upload (9 modos em escopo) — design Stitch | Feito |
| Tela 2 — 4 abas — design Stitch (Estatísticas/AutoML/Laudo em bento) | Feito |
| Tela 3 — histórico de análises (`/historico`) — design Stitch | Feito (v3.1) |
| Métricas clínicas completas no `treinar_vetores` (imagem/tabular) | Feito (v3.1) |
| SDD em `docs/specs/` (constituição + spec/plan/tasks por página) | Feito (v3.1) |
| Dashboard HTML estático CLI (`main.py`) | Mantido para retrocompatibilidade |

---

## Fluxo completo (v2)

```
[Tela 1 — Upload]
  Usuário envia: imagem, ZIP, CSV, .edf, .dcm, .nii, caminho local...
        │
        ▼
[detectar_estrutura()]     ←── app.py
  9 modos: imagem_unica | imagens_soltas | dataset_rotulado | tabular |
           multimodal | sinal_temporal | imagem_dicom_2d |
           volume_3d | multimodal_expandido
  (validação de cabeçalho opcional via pipeline/ingestao.py — Pydantic)
        │
        ├──[F1]──────► BioStatusIACrewSinal
        │                analista_sinais_fisiologicos (Temporal)
        │                → radiologista_ia
        │
        ├──[F3/F4]──► BioStatusIACrewImagem3D
        │                especialista_imagem_medica (DICOM 2D / Volume 3D)
        │                → radiologista_ia
        │
        ├──[Img]────► BioStatusIACrew (original)
        │                engenheiro_pdi → analista_tecnico
        │                → cientista_dados → radiologista_ia
        │
        └──[CSV]────► BioStatusIACrewTabular
                         bioestatistico → laudo tabular
        │
        ▼
[avaliacao_modelos.py]   ←── quando há rótulos e ≥10 amostras
  6 classificadores, CV estratificada repetida (5×3) sem vazamento, métricas enriquecidas (MCC, Kappa, ECE),
  balanceamento SMOTE/ADASYN, McNemar A/B, SHAP no vencedor
        │
        ▼
[inferencia.py]  ←── persiste o vencedor do pódio (models/vencedor_<familia>.pkl)
  Laudo individual recarrega o campeão e classifica novo exemplar
        │
        ▼
[Persistência]  ←── database.py
  Tabela `analises`
  Tabela `resultados_pipeline` (família_sinal, sinal_tipo)
  Tabela `laudos_interativos` (novo v2)
        │
        ▼
[Tela 2 — Resultados (4 abas)]
  Aba 1: Estatísticas + Biomarcadores
  Aba 2: Pré-processamento + Estratégia + Engenharia de Features
  Aba 3: AutoML (apenas — pódio, ROC, radar, matriz de confusão, McNemar)
  Aba 4: Laudo (Dossiê Radiologista IA + Laudo Populacional lado a lado + Laudo de Amostra)
  Tela 3: Histórico de análises (/historico) — página dedicada
```

---

## Stack (v2)

| Componente | Tecnologia | Versão |
|---|---|---|
| Python | 3.11 | `>=3.10,<3.13` |
| Gerenciador de pacotes | `uv` | Sempre `uv sync` / `uv add` |
| LLM | Ollama `qwen2.5:3b` | Configurado em `.env` |
| Agentes | CrewAI | `>=0.203.1,<1.0.0` |
| Servidor web | Flask | `>=3.0.0` |
| Banco de dados | SQLite (built-in) | `biostatusia.db` na raiz |
| Visão Computacional | OpenCV, scikit-image | — |
| Sinais Fisiológicos | MNE-Python, wfdb | — |
| DICOM | pydicom | — |
| Volumes 3D | nibabel, SimpleITK | — |
| Classificadores ML | scikit-learn | `>=1.3.0` (6 modelos) |
| Balanceamento / Interpretabilidade | imbalanced-learn, shap | — |
| Deep Learning (opcional) | torch, torchvision | scaffold `deep_learning.py` |
| Gráficos | Plotly.js (via CDN) | 2.32 |
| Frontend | Tailwind CSS (via CDN) | latest |

Ao adicionar dependências: `uv add <pacote>` — nunca `pip install`.

`MAX_CONTENT_LENGTH = 4 * 1024 * 1024 * 1024` (4 GB — necessário para volumes 3D grandes, como TC/RM completos).

---

## Estrutura de arquivos (v3.1)

```
BioStatusIA/
├── CLAUDE.md
├── README.md
├── docs/
│   ├── documentacao_notion.md       # Documentação completa no formato Notion
│   ├── stitch_prompt.md             # Prompt mestre usado para gerar as telas no Stitch
│   └── specs/                       # Spec-Driven Development (SDD) — fonte da verdade
│       ├── README.md                #   índice + fluxo spec→plan→tasks
│       ├── constitution.md          #   princípios invioláveis do projeto
│       ├── templates/               #   modelos spec/plan/tasks
│       ├── 001-tela-upload/         #   spec.md · plan.md · tasks.md
│       ├── 002-tela-resultados/
│       ├── 003-laudo-amostra/
│       └── 004-laudo-populacional/
├── .env                              # MODEL, API_BASE, PYTHONUTF8 — não commitar
├── pyproject.toml
├── biostatusia.db                    # SQLite — não versionar
├── models/                           # Modelos .pkl treinados — não versionar
├── dataset_teste_busi/               # Mini-BUSI (imagens)
├── dataset_teste_csv/                # WBCD-50 (tabular)
├── tests/                            # Scripts de validação + fixtures (test_platform_data)
├── reports/                          # Saídas .md dos scripts de teste
├── legacy/                           # HTMLs antigos sem uso no runtime
│
└── src/biostatusia/
    ├── app.py                        # Flask: detectar_estrutura, rotas, consolidação
    ├── main.py                       # Pipeline CLI (retrocompatibilidade)
    ├── crew.py                       # 5 Crews CrewAI
    ├── database.py                   # SQLite: 3 tabelas + migração v2
    │
    ├── pipeline/                     # Funções puras (chamadas pelas tools)
    │   ├── __init__.py
    │   ├── io_utils.py               # Extensões + predicados (F1/F3/F4/tabular)
    │   ├── io_sinais.py              # SinalNormalizado + carregar_sinal()
    │   ├── ingestao.py               # (v3) Validação Pydantic + leitura de cabeçalho
    │   ├── analise_base.py           # analisar_base + decidir_estrategia
    │   ├── preprocessamento.py       # preprocessar + preprocessar_adaptativo
    │   ├── segmentacao.py
    │   ├── extracao.py               # extrair_todos (imagens + estratégia)
    │   ├── classificador.py          # treinar() + treinar_vetores() — métricas clínicas completas (+persiste vencedor)
    │   ├── dados_tabulares.py        # CSV/TXT: schema, features, stats
    │   ├── avaliacao_modelos.py      # 6 modelos, CV, MCC/Kappa/ECE, SMOTE, SHAP, McNemar
    │   ├── inferencia.py             # (v3) Vencedor do pódio + previsão de exemplar único
    │   ├── relatorios.py             # (v3) Laudo populacional (pódio, correlações)
    │   ├── deep_learning.py          # (v3) Scaffold ExtratorCNN (torch opcional)
    │   ├── leitura_temporal.py       # Lê .dat/.edf/.mat/.xml/.c3d → SinalNormalizado
    │   ├── leitura_dicom.py          # Lê .dcm único ou série → SinalNormalizado
    │   ├── leitura_volumetrica.py    # Lê .nii/.nii.gz/.mha → SinalNormalizado
    │   ├── extracao_temporal.py      # Features F1: RMS, HRV, bandas EEG, FVC...
    │   ├── extracao_dicom.py         # Features F3: 9 biomarcadores + DICOM-específicos
    │   └── extracao_volumetrica.py   # Features F4: stats 3D, GLCM por plano, morfologia
    │
    ├── config/
    │   ├── agents.yaml               # 8 agentes
    │   └── tasks.yaml                # 9 tasks
    │
    ├── tools/                        # CrewAI BaseTool — wrappers sobre pipeline/
    │   ├── analise_base_tool.py      # FerramentaAnaliseBase
    │   ├── extracao_tool.py          # FerramentaExtrairBiomarcadores (lote)
    │   ├── treino_tool.py            # FerramentaTreinarClassificador
    │   ├── tabular_tool.py           # FerramentaAnaliseTabular
    │   ├── custom_tool.py            # FerramentaAnaliseImagem (single, legado)
    │   ├── sinais_temporais_tool.py  # FerramentaExtrairSinalTemporal
    │   ├── dicom_tool.py             # FerramentaExtrairDICOM
    │   └── volumetrico_tool.py       # FerramentaExtrairVolume3D
    │
    ├── static/
    │   ├── uploads/                  # Arquivos enviados — não versionar
    │   └── runs/                     # Workspaces dos kickoffs — não versionar
    │
    └── templates/
        ├── tela1_upload.html         # Upload com drag & drop (9 modos) — design Stitch
        ├── tela2_resultados.html     # Resultados — 4 abas — design Stitch
        └── tela3_historico.html      # Histórico de análises — design Stitch
```

> **Reorganização (v3.1):** scripts de teste na raiz foram movidos para `tests/` (caminhos re-ancorados na raiz do repo), suas saídas `.md` para `reports/`, e HTMLs legados sem uso no runtime para `legacy/`. O `main.py` (CLI) mantém na raiz os templates/relatórios que lê e gera. As telas foram reconstruídas no **design "clínico moderno"** prototipado no Stitch (`docs/stitch_prompt.md`).

---

## Como rodar

### Servidor web (modo principal)
```powershell
uv run flask --app src/biostatusia/app.py run --port 5000
```
Abrir `http://localhost:5000`.

### CLI (modo retrocompatível, sem UI)
```powershell
uv run biostatsia
```

---

## Tela 1 — Upload

**Rota**: `GET /` e `POST /analisar`

Aceita três formas de entrada:
1. **Upload de arquivo** (drag & drop ou seletor): imagem, ZIP, CSV, EDF, DICOM, NIfTI, MP4
2. **Caminho manual** (texto): qualquer pasta ou arquivo no disco
3. **Vazio**: usa cache do KaggleHub (dataset BUSI)

- Arquivos vão para `src/biostatusia/static/uploads/`
- ZIPs são extraídos automaticamente
- `MAX_CONTENT_LENGTH = 4 GB`

---

## Tela 2 — Resultados (4 abas)

**Rota**: `GET /resultados/<int:resultado_id>`

Layout no **design "clínico moderno"** (Stitch): barra de topo teal com navegação (Dashboard/Histórico), banner ético âmbar, cabeçalho de contexto (dataset, família, amostras, badge do melhor modelo) e as 4 abas.

| Aba | Conteúdo | Quando aparece |
|---|---|---|
| **Aba 1 — Estatísticas & Biomarcadores** | *Imagem/sinal* (bento): Biomarcadores por Amostra (tabela dinâmica em JS), boxplot por classe, Resumo do Dataset, estatísticas descritivas. *Tabular*: variáveis clínicas, 31 métricas univariadas, testes de normalidade/hipótese, heatmap de correlação | Sempre |
| **Aba 2 — Pré-processamento & Engenharia de Features** | Análise da base, estratégia adaptativa, justificativas + seção **Engenharia de Features** (gráfico de importância + médias, com estado vazio quando não há features derivadas) | Sempre (seção de features sempre visível) |
| **Aba 3 — AutoML** | Bento: pódio dos 6 modelos (tabela ranqueada com troféu), curva ROC, matriz de confusão em blocos, radar comparativo, teste A/B McNemar e cards de MCC/Kappa/ECE/latência. **Apenas AutoML** — sem laudo Markdown | Sempre; gráficos só se treinado |
| **Aba 4 — Laudo** | **Dossiê do Radiologista IA** e **Laudo Populacional** lado a lado, + **Laudo de Amostra** (upload/seleção) abaixo | Sempre |

### Aba 4 — Laudo (detalhe)

A Aba 4 concentra os laudos em duas linhas. Na primeira, lado a lado: o **Dossiê do Radiologista IA** (laudo Markdown da análise atual, com botão de impressão e Aviso Ético) e o **Laudo Populacional** (botão Gerar → relatório determinístico da base). Na segunda, o **Laudo de Amostra**: o usuário faz upload de um arquivo avulso (imagem, sinal, DICOM, volume) **ou** seleciona uma análise anterior do banco, e recebe um laudo focado do `radiologista_ia_interativo` (5 seções obrigatórias). O antigo bloco "Histórico do Prontuário (SQLite)" saiu da aba — agora há a **página dedicada `/historico`** (Tela 3). Não há seleção de ROI/slice/frame por família.

Rotas de laudo em `app.py`:
- `POST /laudo_amostra` — recebe `multipart` com `arquivo` (upload) **ou** `resultado_id` (análise existente), dispara `BioStatusIACrewInterativo`, salva em `laudos_interativos`, retorna `{"laudo_html": ..., "laudo_id": ...}`
- `GET /laudo_populacional/<int:resultado_id>` — relatório determinístico do dataset (distribuição, correlações, pódio), montado a partir do `pipeline_json` sem LLM; retorna `{"laudo_html": ..., "laudo_md": ..., "podio": ...}`
- `GET /api/historico` — últimos resultados para o seletor da Aba 4
- `GET /api/exemplos/<int:resultado_id>` — exemplos individuais do dataset da análise

Página de histórico (Tela 3):
- `GET /historico` — renderiza `tela3_historico.html` com `listar_resultados_completo()` (tabela com busca, badges de categoria, atalho para reabrir cada resultado e cartões-resumo). Os links "Histórico" das telas apontam para cá.

> **Nota:** as rotas `POST /laudo_interativo`, `GET /slice/<id>/<idx>` e `GET /frame/<id>/<idx>` de versões anteriores **não existem mais** — foram substituídas por `/laudo_amostra` e `/laudo_populacional`.

---

## AutoML — `pipeline/avaliacao_modelos.py`

### 6 modelos concorrentes

| Modelo | Parâmetros |
|---|---|
| `LogisticRegression` | `max_iter=1000` |
| `KNeighborsClassifier` | `n_neighbors=5` |
| `SVC` (RBF) | `probability=True` |
| `RandomForestClassifier` | `n_estimators=100, random_state=42` |
| `GradientBoostingClassifier` | `n_estimators=100, random_state=42` |
| `MLPClassifier` | `hidden_layer_sizes=(64,32), max_iter=500` |

### Métricas enriquecidas por fold

- `acuracia`, `sensibilidade` (recall positivo), `especificidade` (recall negativo)
- `precisao`, `recall`, `f1`, `auc`
- `ece` — Expected Calibration Error (10 bins)
- `latencia_inferencia_ms` — tempo médio por predição
- `tempo_treino_s` — tempo de treino

### Teste de McNemar (A/B)

- Compara os dois melhores modelos (maior AUC)
- Retorna: chi², p-value, `diferenca_significativa` (p < 0.05)

### Protocolo de validação (v3.2 — sem vazamento)

- Split 80/20 **antes** de qualquer escala/balanceamento; `StandardScaler` ajustado **só no treino** (T1).
- CV **estratificada repetida** `RepeatedStratifiedKFold(5×3)`; escala e `balancear()` (SMOTE/ADASYN) reajustados **dentro de cada fold** (T2).
- `metricas_cv[modelo][metrica]` traz `media`, `desvio` e **`ic95`** (IC 95% t-Student) (T4).

### Critério de seleção

Melhor modelo = **maior AUC entre os modelos com `sensibilidade ≥ 0.8`** (minimiza falso-negativo em contexto clínico); sem nenhum atingir o piso, cai para o de maior AUC. O critério efetivo é exposto no campo `criterio_selecao` (T3).

### Treinador padrão (`pipeline/classificador.py`)

Os fluxos de **imagem** e **tabular** treinam via `treinar_vetores()` (holdout 80/20, sem CV/SMOTE/SHAP; desde a v3.2 o `scaler` é ajustado **só no treino** para eliminar vazamento — T1). Desde a v3.1 ele calcula o **mesmo conjunto de métricas clínicas** do AutoML por fold — acurácia, sensibilidade, especificidade, precisão, recall, F1, AUC, MCC, Kappa, ECE, latência e tempo de treino — de modo que a tabela do AutoML e o Laudo Populacional exibem todas as colunas. Métricas ausentes (análises salvas antes da v3.1) aparecem como `—` na UI e no laudo (`relatorios.py::_fmt_metrica`); reprocessar o dataset preenche tudo.

---

## Agentes CrewAI (v3) — 5 Crews, 8 agentes

### Crew 1: `BioStatusIACrew` — fluxo completo de imagem (original)

| Ordem | Agente | Tool | Output |
|---|---|---|---|
| 1 | `engenheiro_pdi` | `FerramentaAnaliseBase` | `analise_base.json` |
| 2 | `analista_tecnico` | `FerramentaExtrairBiomarcadores` | `biomarcadores.json` |
| 3 | `cientista_dados` | `FerramentaTreinarClassificador` | `metricas.json` |
| 4 | `radiologista_ia` | — | Laudo Markdown |

### Crew 2: `BioStatusIACrewTabular`

| Agente | Tool |
|---|---|
| `bioestatistico` | `FerramentaAnaliseTabular` |

### Crew 3: `BioStatusIACrewSinal` — F1

| Ordem | Agente | Tools |
|---|---|---|
| 1 | `analista_sinais_fisiologicos` | `FerramentaExtrairSinalTemporal` |
| 2 | `radiologista_ia` | — |

### Crew 4: `BioStatusIACrewImagem3D` — F3 + F4

| Ordem | Agente | Tools |
|---|---|---|
| 1 | `especialista_imagem_medica` | `FerramentaExtrairDICOM` + `FerramentaExtrairVolume3D` |
| 2 | `radiologista_ia` | — |

### Crew 5: `BioStatusIACrewInterativo` — Laudo Interativo

| Agente | Tools | Restrição |
|---|---|---|
| `radiologista_ia_interativo` | — (sem tools) | `max_iter=4`, sem tool calls |

Laudo interativo tem **5 seções obrigatórias**: Achado Principal, Severidade 1–5, Comparação com Referência, Recomendação Imediata, Aviso Ético.

### Comunicação entre agentes (via filesystem)

Cada execução cria `static/runs/run_<timestamp>/`. Tools persistem JSONs completos; retornam apenas resumo em texto ao LLM. O agente seguinte lê os JSONs do anterior.

---

## Banco de dados (v2)

### Tabela `analises` (inalterada)
```sql
id        INTEGER PRIMARY KEY AUTOINCREMENT
data_hora TEXT NOT NULL
imagem    TEXT NOT NULL
categoria TEXT NOT NULL   -- "BENIGNO" | "MALIGNO" | "INDEFINIDO" | "TABULAR" | "SINAL"
laudo     TEXT
```

### Tabela `resultados_pipeline` (migração v2 — novas colunas)
```sql
id            INTEGER PRIMARY KEY AUTOINCREMENT
data_hora     TEXT NOT NULL
dataset_path  TEXT
n_imagens     INTEGER
pipeline_json TEXT               -- payload completo para Tela 2
melhor_modelo TEXT
analise_id    INTEGER            -- FK → analises.id
familia_sinal TEXT               -- "F1" | "F3" | "F4" | ""
sinal_tipo    TEXT               -- "ECG" | "EEG" | "Raio-X" | ...
```

### Tabela `laudos_interativos` (nova v2)
```sql
id            INTEGER PRIMARY KEY AUTOINCREMENT
resultado_id  INTEGER            -- FK → resultados_pipeline.id
trecho_inicio REAL               -- segundos ou índice de início
trecho_fim    REAL               -- segundos ou índice de fim
roi_json      TEXT               -- JSON com x,y,w,h para ROI de imagem
canal         TEXT               -- canal selecionado (ECG lead, EEG canal...)
slice_idx     INTEGER            -- índice de slice para volumes 3D
laudo_foco    TEXT               -- laudo interativo gerado
data_hora     TEXT NOT NULL
```

Migração v2 é idempotente (`_migrar_v2()` usa `try/except OperationalError` para `ALTER TABLE`).

---

## Pipeline adaptativo de leitura e extração

### F1 — Sinais Temporais (`leitura_temporal.py` → `extracao_temporal.py`)

**Leitores:**
- `.dat/.hea` → `wfdb.rdrecord()`
- `.edf/.bdf` → `mne.io.read_raw_edf/bdf()`
- `.mat` → `scipy.io.loadmat()`
- `.xml` → ElementTree (Espirometria Schiller/Cosmed)
- `.c3d` → `bioread` (Movimento/PSG)

**Features extraídas:**
- Domínio do tempo: RMS, média, desvio, skewness, kurtosis, pico-a-pico, SNR_dB
- Domínio da frequência: PSD (Welch), centroide espectral, 4 bandas de potência, freq dominante
- ECG específico: R-peaks (scipy), FC_bpm, RMSSD, SDNN, pNN50
- EEG específico: bandas delta/theta/alpha/beta/gamma relativas, ratio alpha/beta
- EMG específico: RMS envelope, frequência mediana
- Espirometria: FVC, FEV1, FEV1/FVC, PEF

### F3 — DICOM 2D (`leitura_dicom.py` → `extracao_dicom.py`)

- 9 biomarcadores originais (morfologia, textura GLCM, distribuição)
- DICOM-específicos: densidade alta (%), gradiente médio, uniformidade, modalidade, pixel_spacing, hounsfield_range
- Janelamento HU automático (WindowCenter/Width do header DICOM)
- SOPClassUID para identificar modalidade

### F4 — Volume 3D (`leitura_volumetrica.py` → `extracao_volumetrica.py`)

- nibabel para `.nii/.nii.gz` (transpõe X,Y,Z → Z,Y,X axial-first)
- SimpleITK para `.mha`
- Stats globais: média, desvio, mediana, skewness, kurtosis, P5/P95, n_voxels_altos
- GLCM por plano ortogonal (axial/coronal/sagital)
- Morfologia 3D: volume da lesão (mm³), esfericidade, axes do bounding box
- `slice_para_png_base64(volume, idx)` → base64 de slice axial (helper de visualização; a antiga rota `/slice/` foi removida na v3)

---

## Biomarcadores — regra de interpretação (imagens)

| Métrica | Categoria | Sinal de Malignidade |
|---|---|---|
| Circularidade | Morfologia | Baixa (< 0.7) |
| Solidez | Morfologia | Baixa (margens irregulares) |
| Entropia | Textura (GLCM) | Alta (tecido heterogêneo) |
| Homogeneidade | Textura (GLCM) | Baixa |
| Energia | Textura (GLCM) | Baixa |
| Contraste | Textura (GLCM) | Alto |
| SNR | Intensidade | Qualidade do sinal — não diagnóstico |
| Assimetria | Intensidade | Complementar |
| Curtose | Intensidade | Complementar |

Regra geral: **baixa solidez + alta entropia → suspeito de malignidade**.

---

## Datasets de validação (Kaggle)

| Família | Dataset | Slug no Kaggle | Formato |
|---|---|---|---|
| F1 — ECG | ECG Heartbeat Categorization | `shayanfazeli/heartbeat` | CSV (MIT-BIH) |
| F3 — DICOM/Raio-X | Chest X-Ray Images (Pneumonia) | `paultimothymooney/chest-xray-pneumonia` | JPEG (NORMAL/PNEUMONIA) |

Para volumes 3D (F4), datasets como BRATS são muito pesados para KaggleHub — usar download direto.

---

## Nomes de pastas reconhecidos como rótulo

```python
PASTAS_BENIGNAS = {"benign", "benigno", "normal", "negative", "0"}
PASTAS_MALIGNAS = {"malignant", "malign", "maligno", "abnormal", "positive", "1"}
```

Busca **recursiva** — funciona com estruturas aninhadas como `Dataset_BUSI_with_GT/benign/`.

---

## Detecção de coluna-rótulo em CSV/TXT

`pipeline/dados_tabulares.py::detectar_schema()` identifica:
1. Por nome: `label`, `class`, `diagnosis`, `target`, `outcome`, `y`, `categoria`, `resultado`
2. Se não achar por nome: última coluna com 2–10 valores únicos
3. Rótulos binários mapeados para 0/1 via `MALIGNANT_KEYWORDS`

Separador detectado automaticamente (`,`, `;`, `\t`, `|`). Encoding: UTF-8 com fallback Latin-1.

---

## Convenções de código

- **Português** para variáveis de domínio (`laudo`, `categoria`, `biomarcadores`, `solidez`). Inglês para infraestrutura (`conn`, `path`, `model`).
- **Sem comentários** que expliquem o "o quê" — só o "por quê" quando não óbvio.
- **Pipeline = funções puras**: cada etapa recebe e devolve dados, sem I/O.
- Toda I/O (DB, HTML, arquivos) concentrada em `app.py` e `database.py`.
- **Templates HTML são a fonte da verdade visual** — nunca editar HTML gerado em runtime. Ao restilizar, preservar IDs e handlers usados pelo JS (ex.: `chart-radar`, `chart-roc`, `chart-boxplot`, `tbody-banco`, `btn-gerar-laudo-*`, `painel-laudo-*`).
- **Design "clínico moderno"** (Stitch): paleta teal Material, fonte Inter, ícones Material Symbols, banner ético âmbar, badges clínicos. Referência em `docs/stitch_prompt.md`.
- **SDD primeiro**: mudança de página/fluxo começa pela `spec.md` em `docs/specs/<feature>/`, passa por `plan.md`, vira `tasks.md` e só então vira código. Respeitar `docs/specs/constitution.md`.
- **Pydantic** para schemas de ferramentas CrewAI.

---

## Problema conhecido: encoding no Windows

CrewAI emite emojis nos logs do EventBus. Terminal com `charmap` (cp1252) gera `[EventBus Error]`. Cosméticos — não afetam execução.

```
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
```

---

## O que NÃO fazer

- Não usar `pip install` — usar `uv add <pacote>`.
- Não commitar `.env`, `biostatusia.db`, `models/`, `src/biostatusia/static/uploads/`.
- Não editar HTMLs gerados em runtime.
- Não trocar `Process.sequential` por `Process.hierarchical` sem revisar `manager_llm`.
- Não aumentar `max_iter` sem medir tempo de resposta.
- Não adicionar ferramentas ao `radiologista_ia` nem ao `radiologista_ia_interativo`.
- Não reintroduzir áudio (F2) nem vídeo (F5) — foram removidos do escopo na v3.
- Manter os 9 modos em escopo em `detectar_estrutura()` (F1/F3/F4 + tabular + imagem comum).
- Não treinar classificadores fora do contexto rotulado.
- Não reduzir `MAX_CONTENT_LENGTH` — volumes 3D precisam de 4 GB.
- Não quebrar a `SinalNormalizado` — é o contrato entre leitores e extratores.
- Não remover o Aviso Ético de nenhuma saída visual ou laudo.
- Não reintroduzir os links "Pacientes" e "Relatórios" na navegação (removidos — só Dashboard/Histórico).
- Não mover os templates/relatórios que o `main.py` (CLI) lê/gera na raiz sem atualizar os caminhos nele.
- Não editar a skill em cache — mudanças de skill vão em Configurações; os artefatos SDD ficam em `docs/specs/`.

---

## Aviso ético obrigatório

Este sistema é uma **ferramenta de suporte à decisão clínica**. Nenhuma mudança de código deve remover o aviso de que os laudos **não substituem a avaliação de um médico habilitado**. Esse aviso é obrigatório em:
- Toda saída visual (banner ético nas Telas 1, 2 e 3)
- Todo laudo do `radiologista_ia`
- Todo laudo do `radiologista_ia_interativo` (5ª seção obrigatória)
- Todo laudo do `bioestatistico`
