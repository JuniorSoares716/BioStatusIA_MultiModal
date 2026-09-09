# BioStatusIA v3

> Sistema de Apoio à Decisão Clínica (CDSS) para análise automatizada de sinais biomédicos em **3 famílias de sinal (F1/F3/F4) + dados tabulares**, com IA Multi-Agente local, AutoML de 6 modelos com avaliação enriquecida e laudos (populacional e de amostra).

Pipeline orquestrado por agentes CrewAI rodando sobre LLM local (Ollama), com classificação AutoML (6 modelos, 5-fold CV, métricas clínicas enriquecidas) e interface web Flask com design clínico moderno — tela de upload, tela de resultados em 4 abas e página de histórico dedicada.

Projeto acadêmico de mestrado em IA na Saúde — Fortaleza, CE.

> **Metodologia:** o projeto adota **Spec-Driven Development (SDD)** — as especificações vivem em `docs/specs/` (constituição + spec/plan/tasks por página) e são a fonte da verdade para a evolução do sistema.

> **Escopo v3:** áudio biomédico (F2) e vídeo médico (F5) foram **removidos do escopo**. O foco atual é F1 (sinais temporais), F3 (DICOM 2D), F4 (volume 3D) e dados tabulares.

---

## Princípio fundamental

**O sistema aceita qualquer tipo de dado em escopo.** Ao receber uma entrada, ele:

1. Detecta automaticamente a estrutura e a família de sinal (9 modos)
2. Carrega e normaliza sinais via `SinalNormalizado` (dataclass universal)
3. Extrai biomarcadores específicos da família detectada
4. Treina 6 classificadores concorrentes com avaliação enriquecida (quando há rótulos e ≥10 amostras com 2 classes)
5. Gera laudo preliminar via agentes IA especializados por família
6. Disponibiliza laudo populacional (determinístico, nível da base) e laudo de amostra avulsa

A análise estatística sempre roda. O treino de classificadores só ocorre com rótulos e ≥10 amostras. O laudo IA roda apenas quando há sinal ou imagem disponível.

---

## 3 Famílias de Sinal + Tabular

| Família | Tipos | Formatos de Entrada | Leitura |
|---|---|---|---|
| **F1 — Sinais Temporais** | ECG, EEG, EMG, EOG, PPG, PA, Espirometria | `.dat/.hea`, `.edf`, `.bdf`, `.mat`, `.xml`, `.c3d` | `mne`, `wfdb`, `scipy` |
| **F3 — Imagem DICOM 2D** | Raio-X, Mamografia, Ultrassom estático | `.dcm` (arquivo único) | `pydicom` |
| **F4 — Volume 3D** | TC, RM, PET/SPECT | `.nii`, `.nii.gz`, `.mha`, pasta com ≥10 `.dcm` | `nibabel`, `SimpleITK` |
| **Tabular** | Features clínicas multimodais | `.csv`, `.txt`, `.tsv` | parser próprio |
| **Imagem comum** | `.png/.jpg/.bmp/.tif` | radiômica OpenCV/scikit-image | — |

---

## 9 Modos detectados automaticamente

| Modo | Entrada | Crew disparada |
|---|---|---|
| `imagem_unica` | Uma imagem `.png/.jpg/.bmp/.tif` | `BioStatusIACrew` |
| `imagens_soltas` | Pasta/ZIP com imagens sem rótulos | `BioStatusIACrew` |
| `dataset_rotulado` | Pasta com subpastas `benign/` + `malignant/` | `BioStatusIACrew` |
| `tabular` | CSV/TXT/TSV com features clínicas | `BioStatusIACrewTabular` |
| `multimodal` | Imagens **e** CSV juntos | `BioStatusIACrew` + tabular |
| `sinal_temporal` | ECG / EEG / EMG / ... (F1) | `BioStatusIACrewSinal` |
| `imagem_dicom_2d` | DICOM único (Raio-X, Mamografia...) | `BioStatusIACrewImagem3D` |
| `volume_3d` | NIfTI / MHA / série DICOM ≥10 arquivos | `BioStatusIACrewImagem3D` |
| `multimodal_expandido` | Mix de imagens / sinais / tabular | `BioStatusIACrew` + sub-crews |

Entradas fora de escopo ou vazias caem em `invalido`.

Nomes de pastas reconhecidos como rótulo — Benignas: `benign`, `benigno`, `normal`, `negative`, `0` | Malignas: `malignant`, `malign`, `maligno`, `abnormal`, `positive`, `1` (busca recursiva).

---

## Arquitetura — pipeline agentificado (v3)

**5 Crews, 8 agentes, 8 tools** — todos em processo `sequential`.

### Crews de Sinal e Imagem

```
BioStatusIACrewSinal (F1)
  analista_sinais_fisiologicos
    └─ FerramentaExtrairSinalTemporal → biomarcadores_temporal.json
  radiologista_ia → laudo Markdown

BioStatusIACrewImagem3D (F3/F4)
  especialista_imagem_medica
    ├─ FerramentaExtrairDICOM        → biomarcadores_dicom.json
    └─ FerramentaExtrairVolume3D     → biomarcadores_volumetrico.json
  radiologista_ia → laudo Markdown
```

### Crew de Imagem Original (radiômica)

```
BioStatusIACrew
  engenheiro_pdi       → analise_base.json
  analista_tecnico     → biomarcadores.json
  cientista_dados      → metricas.json
  radiologista_ia      → laudo Markdown
```

### Crew Tabular e Laudo Interativo

```
BioStatusIACrewTabular
  bioestatistico → FerramentaAnaliseTabular → laudo Markdown

BioStatusIACrewInterativo
  radiologista_ia_interativo (sem tool, max_iter=4)
  → laudo de amostra avulsa (5 seções obrigatórias)
```

Os agentes se comunicam via JSONs em `static/runs/run_<timestamp>/` — o LLM recebe resumos em texto, não dados brutos.

### 8 Agentes (`config/agents.yaml`)

`engenheiro_pdi`, `analista_tecnico`, `cientista_dados`, `radiologista_ia`, `bioestatistico`, `analista_sinais_fisiologicos`, `especialista_imagem_medica`, `radiologista_ia_interativo`.

### 8 Tools (`tools/`)

`analise_base_tool`, `extracao_tool`, `treino_tool`, `tabular_tool`, `custom_tool` (legado), `sinais_temporais_tool`, `dicom_tool`, `volumetrico_tool`.

---

## AutoML — 6 Modelos Concorrentes (`pipeline/avaliacao_modelos.py`)

Treino em 5-fold StratifiedKFold:

| Modelo | Tipo |
|---|---|
| Regressão Logística | Linear |
| KNN | Baseado em distância |
| SVM RBF | Kernel não-linear |
| Random Forest | Ensemble (100 árvores) |
| Gradient Boosting | Boosting (100 estimadores) |
| MLP Neural Network | Rede neural (64×32) |

**Métricas por modelo:** acurácia, sensibilidade, especificidade, precisão, recall, F1, AUC, MCC, Kappa, ECE (Expected Calibration Error), latência de inferência (ms), tempo de treino (s).

> **Nota (v3.1):** o treinador padrão usado pelos fluxos de imagem e tabular (`pipeline/classificador.py::treinar_vetores`) também passou a calcular o **conjunto completo** dessas métricas — sensibilidade, especificidade, MCC, Kappa, ECE, latência e tempo de treino — de modo que a tabela do AutoML e o Laudo Populacional exibem todas as colunas preenchidas. Análises salvas antes dessa mudança mantêm o esquema antigo (métricas ausentes aparecem como `—`); basta reprocessar o dataset para preencher tudo.

**Balanceamento:** SMOTE / ADASYN quando há desbalanceamento de classes.

**Interpretabilidade:** SHAP no modelo vencedor.

**Teste de McNemar** entre os 2 melhores modelos — chi², p-value, significância estatística.

**Critério de seleção:** maior AUC, com preferência por sensibilidade ≥ 0.80 (minimiza falso-negativo).

O modelo vencedor é persistido em `models/vencedor_<familia>.pkl` (`pipeline/inferencia.py`) para classificação de amostras individuais.

---

## 🧪 Validation Benchmark com Bases Reais do Kaggle (< 1GB)

O **BioStatusIA v3** conta com um script de avaliação experimental (`scripts/executar_kaggle_reais.py`) que realiza o download automático de datasets biomédicos reais do Kaggle via `kagglehub`, testa o AutoML de 6 modelos (5-fold CV) e gera laudos clínicos reais via Ollama (`qwen2.5:3b`).

### Execução do Benchmark

```bash
uv run python scripts/executar_kaggle_reais.py
```

### Resultados Obtidos nas Bases Reais do Kaggle

| # | Base Real (Kaggle) | Família | Modo Detectado | N° Feats | Top Feature (SHAP) | Modelo Vencedor | AUC | Sensib. | Espec. |
|---|---|:---:|:---:|:---:|---|:---:|:---:|:---:|:---:|
| 01 | **Breast Cancer Wisconsin** | `Tabular` | `tabular` | **31** | `radius_mean / texture_mean` | **RandomForest** | **0.99** | 0.90 | 0.97 |
| 02 | **BUSI Breast Ultrasound** | `Imagem2D` | `dataset_rotulado` | **12** | `Entropia GLCM (0.342)` | **KNN** | **0.89** | 1.00 | 0.67 |
| 03 | **MIT-BIH & PTB ECG Signals** | `F1` | `tabular` | **188** | `Amplitude RMS` | **RandomForest** | **0.89** | 0.87 | 0.85 |
| 04 | **Stroke Prediction Clinical** | `Tabular` | `tabular` | **6** | `age / hypertension` | **RandomForest** | **0.89** | 0.87 | 0.85 |
| 05 | **PIMA Diabetes Metabolic** | `Tabular` | `tabular` | **8** | `Glucose / Pregnancies` | **SVM** | **0.82** | 0.50 | 0.90 |
| 06 | **Brain Tumor MRI Slices** | `F4` | `dataset_rotulado` | **12** | `Entropia GLCM (0.342)` | **SVM** | **0.56** | 0.33 | 1.00 |
| 07 | **COVID-19 Chest X-Ray** | `F3` | `dataset_rotulado` | **12** | `Contraste GLCM` | **SVM** | **0.78** | 0.00 | 0.33 |
| 08 | **Brain MRI Oncology** | `F4` | `dataset_rotulado` | **12** | `Solidez Margens` | **LogisticRegression** | **1.00** | 1.00 | 0.67 |

- 📊 **Relatório Master Comparativo:** [reports/relatorio_kaggle_reais_comparativo.md](file:///c:/Users/Braudel/.gemini/antigravity-ide/scratch/BioStatusIA/reports/relatorio_kaggle_reais_comparativo.md)
- 🩺 **Insights Médicos por Base:** [reports/insights_kaggle_reais/](file:///c:/Users/Braudel/.gemini/antigravity-ide/scratch/BioStatusIA/reports/insights_kaggle_reais/)

---

## Biomarcadores extraídos por família

### F1 — Sinais Temporais
- Tempo: RMS, média, desvio, skewness, kurtosis, pico-a-pico, SNR_dB
- Frequência: PSD (Welch), centroide espectral, 4 bandas de potência, freq dominante
- ECG: FC_bpm, RMSSD, SDNN, pNN50, R-peaks
- EEG: bandas delta/theta/alpha/beta/gamma, ratio alpha/beta
- EMG: RMS envelope, frequência mediana
- Espirometria: FVC, FEV1, FEV1/FVC, PEF

### F3 — DICOM 2D
- 9 biomarcadores radiômicos (morfologia, GLCM, distribuição de intensidade)
- Densidade alta (%), gradiente médio, uniformidade, modalidade DICOM
- Pixel spacing, hounsfield_range, janelamento HU automático

### F4 — Volume 3D
- Estatísticas globais: média, desvio, mediana, skewness, kurtosis, P5/P95, voxels_altos
- GLCM por plano ortogonal (axial, coronal, sagital)
- Morfologia 3D: volume da lesão (mm³), esfericidade, bounding box axes

### Imagem comum (radiômica)
| Grupo | Métrica | Sinal de Malignidade |
|---|---|---|
| Morfologia | Circularidade | Baixa (<0.7) |
| Morfologia | Solidez | Baixa (margens irregulares) |
| Textura (GLCM) | Contraste | Alto |
| Textura (GLCM) | Homogeneidade | Baixa |
| Textura (GLCM) | Energia | Baixa |
| Textura (GLCM) | Entropia | Alta (tecido heterogêneo) |
| Distribuição | SNR | Qualidade do sinal (não diagnóstico) |
| Distribuição | Assimetria | Complementar |
| Distribuição | Curtose | Complementar |

Regra geral: **baixa solidez + alta entropia → suspeito de malignidade**.

---

## Pré-processamento adaptativo (imagens)

O `engenheiro_pdi` analisa a base antes da extração e escolhe a estratégia:

| Condição | Estratégia |
|---|---|
| Ruído médio > 0.05 | Non-Local Means |
| Ruído ≤ 0.05 | Gaussian blur 5×5 |
| Outliers > 10% (IQR) | Normalização percentil 1–99% |
| Outliers ≤ 10% | Min-max [0,1] |
| Contraste médio < 30 | CLAHE |
| Contraste ≥ 30 | Sem equalização |
| Tamanhos heterogêneos | Resize 256×256 obrigatório |

---

## Interface web (design clínico moderno)

A interface segue um **design system "clínico moderno"** (paleta teal Material, tipografia Inter, ícones Material Symbols, cartões arredondados, badges clínicos e banner ético em âmbar). Barra de navegação com **Dashboard** (upload) e **Histórico**.

### Telas

- **Tela 1 — Upload** (`/`): drag & drop para qualquer tipo de dado em escopo, campo de caminho local, seletor de tipo de sinal, integração KaggleHub e faixa dos tipos suportados.
- **Tela 2 — Resultados** (`/resultados/<id>`): cabeçalho de contexto (dataset, família, amostras, melhor modelo) + 4 abas.
- **Tela 3 — Histórico** (`/historico`): lista de todas as análises anteriores com busca, badges de categoria, atalho para reabrir cada resultado e cartões-resumo (total, famílias distintas, última análise).

### Tela 2 — 4 abas

| Aba | Conteúdo | Quando aparece |
|---|---|---|
| **1 — Estatísticas & Biomarcadores** | *Modo imagem/sinal* (bento): biomarcadores por amostra, boxplot por classe, resumo do dataset, estatísticas descritivas. *Modo tabular*: variáveis clínicas, 31 métricas univariadas por feature, testes de normalidade/hipótese e heatmap de correlação | Sempre |
| **2 — Pré-processamento** | Estratégia adaptativa com justificativas + engenharia de features (ranking de importância) | Sempre |
| **3 — AutoML** | Pódio dos 6 modelos (tabela ranqueada com troféu no vencedor), curva ROC, matriz de confusão em blocos, radar comparativo, teste A/B de McNemar e cards de MCC/Kappa/ECE/latência | Sempre; gráficos só se treinado |
| **4 — Laudo** | Dossiê do Radiologista IA + **Laudo Populacional** (lado a lado) e **Laudo de Amostra**: upload de arquivo avulso **ou** seleção de análise anterior → laudo do Radiologista IA | Sempre |

O **Laudo de Amostra** gera 5 seções obrigatórias: Achado Principal, Severidade 1–5, Comparação com Referência, Recomendação Imediata e Aviso Ético.

O **Laudo Populacional** (`/laudo_populacional/<id>`) é um relatório determinístico do dataset completo — distribuição, correlações de biomarcadores e pódio final do AutoML — montado a partir do `pipeline_json` persistido, sem depender do LLM.

---

## API Flask — Rotas

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Tela 1 — Upload |
| `POST` | `/analisar` | Dispara o pipeline → redireciona para `/resultados/<id>` |
| `GET` | `/resultados/<id>` | Tela 2 — 4 abas |
| `GET` | `/historico` | Tela 3 — Histórico de análises |
| `GET` | `/api/historico` | JSON com os últimos resultados (seletor da Aba 4) |
| `GET` | `/api/exemplos/<id>` | Exemplos individuais do dataset da análise |
| `GET` | `/laudo_populacional/<id>` | Laudo populacional determinístico (JSON com HTML/MD + pódio) |
| `POST` | `/laudo_amostra` | Laudo de amostra avulsa (upload) ou de análise existente (resultado_id) |

---

## Stack tecnológica (v3)

| Camada | Tecnologia |
|---|---|
| Backend | Python 3.10–3.12, Flask ≥3.0 |
| LLM local | Ollama (`qwen2.5:3b`) |
| Agentes | CrewAI ≥0.203 (5 crews, 8 agentes) |
| Sinais fisiológicos | MNE-Python, wfdb, scipy |
| Imagens DICOM | pydicom |
| Volumes 3D | nibabel, SimpleITK |
| Visão computacional | OpenCV (headless), scikit-image |
| AutoML | scikit-learn (6 modelos) |
| Balanceamento / Interpretabilidade | imbalanced-learn, SHAP |
| Deep Learning (opcional) | scaffold `deep_learning.py` (torch opcional) |
| Banco de dados | SQLite (3 tabelas) |
| Gerenciador de pacotes | `uv` |
| Frontend | HTML5 + Tailwind CSS, fonte Inter, ícones Material Symbols (CDN) |
| Design das telas | Prototipadas no Stitch (design "clínico moderno") — ver `docs/stitch_prompt.md` |
| Gráficos | Plotly.js 2.32 (CDN) |

> **Nota:** Tailwind CSS e Plotly.js são carregados via CDN. É necessário acesso à internet no primeiro carregamento das telas — sem ele, o layout e os gráficos não renderizam.

---

## Instalação

### Pré-requisitos

- Python 3.10–3.12
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com) com `qwen2.5:3b`:
  ```bash
  ollama pull qwen2.5:3b
  ```

### Setup

```bash
git clone https://github.com/keikorr/BioStatusIA.git
cd BioStatusIA
uv sync
```

### Configuração `.env`

```env
MODEL=ollama/qwen2.5:3b
API_BASE=http://localhost:11434
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
```

---

## Como rodar

### Servidor web (modo principal)

```bash
uv run flask --app src/biostatusia/app.py run --port 5000
```

Abra `http://localhost:5000` — Tela 1 com drag & drop para qualquer tipo de dado em escopo.

> **Pré-requisitos de execução:** o Ollama precisa estar rodando (`http://localhost:11434`) com o modelo `qwen2.5:3b` disponível, e a máquina precisa de acesso à internet para os assets de CDN. Sem o Ollama, as etapas de laudo IA falham; sem internet, as telas não renderizam corretamente.

### CLI retrocompatível (sem UI)

```bash
uv run biostatsia
```

### Datasets de validação (Kaggle)

| Família | Dataset | Kaggle slug |
|---|---|---|
| F1 (ECG) | ECG Heartbeat Categorization | `shayanfazeli/heartbeat` |
| F3 (Raio-X) | Chest X-Ray Images (Pneumonia) | `paultimothymooney/chest-xray-pneumonia` |

Para volumes 3D (F4), datasets como BRATS são pesados demais para o KaggleHub — usar download direto.

### Datasets de teste local

- **Mini-BUSI** (imagens): `dataset_teste_busi/`
- **WBCD-50** (50 amostras tabulares): `dataset_teste_csv/wbcd_50.csv`

---

## Testes

### Suíte automatizada (`pytest`) — 55 testes

Suíte rápida (~20s, **sem Ollama nem internet**) cobrindo as três camadas:

```bash
uv sync
uv run pytest
```

| Camada | Arquivo | Cobertura |
|---|---|---|
| **Unit — pipeline** | `test_deteccao_modo.py` | os 9 modos de `detectar_estrutura()` + F2/F5 → `invalido` + predicados `io_utils` |
| | `test_tabular.py` | schema por nome/fallback, separador, encoding, NaN |
| | `test_classificador.py` | `treinar_vetores` com todas as métricas, campeão por AUC, guarda de <10 amostras, SMOTE seguro |
| | `test_relatorios.py` | `_fmt_metrica(None)→—`, pódio ordenado, laudo determinístico, correlações |
| | `test_inferencia.py` | achatar, roundtrip salvar/carregar, ajuste dimensional defensivo |
| | `test_contrato_sinais.py` | `SinalNormalizado` por família (F1/F3/F4) |
| **Integração — Flask** | `test_rotas.py` | rotas GET, 404/400, banco isolado, `/laudo_amostra` com crew mockada |
| **Crews (sem LLM)** | `test_crews.py` | fiação das 5 crews + guardrails (radiologistas sem tools, `max_iter`) |
| | `test_tools.py` | contrato de filesystem das tools (cadeia de JSONs) |
| **Guardrails** | `test_guardrails.py` | aviso ético nas 3 telas, 5 seções do laudo, nav sem Pacientes/Relatórios |

Testes que exigem Ollama/Kaggle podem ser marcados com `@pytest.mark.slow` e excluídos com `uv run pytest -m "not slow"`.

### Scripts de validação manuais (`reports/`)

Além da suíte, há scripts que rodam o pipeline de ponta a ponta e gravam relatórios:

| Comando | Requer | O que valida |
|---|---|---|
| `uv run python tests/test_todos_tipos.py` | — | Leitura → extração → AutoML para **cada** tipo (imagem, tabular, F1, F3, F4) |
| `uv run python tests/test_direct.py` | — | Pipeline puro (base, extração, AutoML, tabular, sinal) |
| `uv run python tests/run_tests.py` | Ollama | Crews completas ponta a ponta |
| `uv run python tests/test_large_image_dataset.py` | Internet / KaggleHub | Pipeline de imagem sobre dataset grande do Kaggle |

Saídas em `reports/relatorio_tipos.md`, `reports/resultados_teste_direto.md`, `reports/resultados_teste.md` e `reports/resultados_large_dataset.md`.

---

## Spec-Driven Development (SDD)

A evolução do sistema segue **SDD**: primeiro a especificação, depois o código. Artefatos em `docs/specs/`:

- `constitution.md` — princípios invioláveis (escopo v3, contrato `SinalNormalizado`, rigor do AutoML, aviso ético, etc.)
- `README.md` — índice e fluxo (spec → plan → tasks → implementação)
- `templates/` — modelos reutilizáveis
- Uma pasta por página/fluxo (`001-tela-upload`, `002-tela-resultados`, `003-laudo-amostra`, `004-laudo-populacional`), cada uma com `spec.md`, `plan.md` e `tasks.md`

---

## Estrutura do projeto

```
BioStatusIA/
├── README.md
├── CLAUDE.md                          # Diretrizes do projeto para Claude
├── docs/
│   ├── documentacao_notion.md         # Documentação completa no formato Notion
│   ├── stitch_prompt.md               # Prompt mestre usado para gerar as telas no Stitch
│   └── specs/                         # Spec-Driven Development (SDD)
│       ├── README.md                  # Índice + fluxo SDD
│       ├── constitution.md            # Princípios invioláveis do projeto
│       ├── templates/                 # Modelos spec/plan/tasks
│       ├── 001-tela-upload/           # spec.md · plan.md · tasks.md
│       ├── 002-tela-resultados/
│       ├── 003-laudo-amostra/
│       └── 004-laudo-populacional/
├── pyproject.toml
├── uv.lock
├── .env                               # MODEL, API_BASE (não versionado)
├── biostatusia.db                     # SQLite (gerado em runtime, não versionado)
├── models/                            # Modelos .pkl treinados (não versionado)
├── dataset_teste_busi/                # Mini-BUSI para testes
├── dataset_teste_csv/                 # WBCD-50 tabular
├── tests/                             # Scripts de validação (test_direct, run_tests, ...)
├── reports/                           # Saídas .md dos scripts de teste
├── legacy/                            # HTMLs antigos sem uso no runtime
│
└── src/biostatusia/
    ├── app.py                         # Flask — 9 modos, 8 rotas (inclui /historico)
    ├── main.py                        # CLI retrocompatível
    ├── crew.py                        # 5 Crews CrewAI
    ├── database.py                    # SQLite: 3 tabelas + migração v2
    │
    ├── pipeline/                      # Funções puras (chamadas pelas tools)
    │   ├── io_utils.py                # Extensões + predicados (F1/F3/F4/tabular)
    │   ├── io_sinais.py               # SinalNormalizado + carregar_sinal()
    │   ├── ingestao.py                # Validação Pydantic + leitura de cabeçalho
    │   ├── analise_base.py            # analisar_base + decidir_estrategia
    │   ├── preprocessamento.py        # preprocessar + preprocessar_adaptativo
    │   ├── segmentacao.py
    │   ├── extracao.py                # extrair_todos (imagens, com estratégia)
    │   ├── classificador.py           # treinar() + treinar_vetores() (métricas clínicas completas)
    │   ├── dados_tabulares.py         # CSV/TXT: schema, features, stats
    │   ├── avaliacao_modelos.py       # 6 modelos, CV, MCC/Kappa/ECE, SMOTE, SHAP, McNemar
    │   ├── inferencia.py              # Vencedor do pódio + previsão de amostra única
    │   ├── relatorios.py              # Laudo populacional (pódio, correlações)
    │   ├── deep_learning.py           # Scaffold ExtratorCNN (torch opcional)
    │   ├── leitura_temporal.py        # Lê .dat/.edf/.mat/.xml/.c3d
    │   ├── leitura_dicom.py           # Lê .dcm único ou série
    │   ├── leitura_volumetrica.py     # Lê .nii/.nii.gz/.mha
    │   ├── extracao_temporal.py       # Features F1
    │   ├── extracao_dicom.py          # Features F3
    │   └── extracao_volumetrica.py    # Features F4
    │
    ├── config/
    │   ├── agents.yaml                # 8 agentes
    │   └── tasks.yaml                 # 9 tasks
    │
    ├── tools/                         # 8 tools CrewAI
    │   ├── analise_base_tool.py
    │   ├── extracao_tool.py
    │   ├── treino_tool.py
    │   ├── tabular_tool.py
    │   ├── custom_tool.py             # legado
    │   ├── sinais_temporais_tool.py
    │   ├── dicom_tool.py
    │   └── volumetrico_tool.py
    │
    ├── static/
    │   ├── uploads/                   # Arquivos enviados
    │   └── runs/                      # Workspaces dos kickoffs (JSONs)
    │
    └── templates/
        ├── tela1_upload.html          # Upload universal (9 modos)
        ├── tela2_resultados.html      # Resultados — 4 abas
        └── tela3_historico.html       # Histórico de análises
```

> **Reorganização (v3.1):** scripts de teste foram movidos da raiz para `tests/` (com caminhos re-ancorados na raiz do repositório), suas saídas `.md` para `reports/`, e HTMLs antigos sem uso no runtime para `legacy/`. O `main.py` (CLI) mantém na raiz os templates/relatórios que ele lê e gera.

---

## Banco de dados (3 tabelas)

### `analises`
| Coluna | Tipo | Descrição |
|---|---|---|
| `id` | INTEGER PK | — |
| `data_hora` | TEXT | ISO 8601 |
| `imagem` | TEXT | caminho do arquivo analisado |
| `categoria` | TEXT | `BENIGNO` / `MALIGNO` / `INDEFINIDO` / `TABULAR` / `SINAL` |
| `laudo` | TEXT | Markdown gerado pelo agente |

### `resultados_pipeline`
| Coluna | Tipo | Descrição |
|---|---|---|
| `id` | INTEGER PK | — |
| `data_hora` | TEXT | ISO 8601 |
| `dataset_path` | TEXT | caminho do dataset |
| `n_imagens` | INTEGER | amostras processadas |
| `pipeline_json` | TEXT | payload completo para Tela 2 |
| `melhor_modelo` | TEXT | nome do vencedor AutoML |
| `analise_id` | INTEGER | FK → analises |
| `familia_sinal` | TEXT | `F1` / `F3` / `F4` / `""` |
| `sinal_tipo` | TEXT | `ECG`, `Raio-X`, etc. |

### `laudos_interativos`
| Coluna | Tipo | Descrição |
|---|---|---|
| `id` | INTEGER PK | — |
| `resultado_id` | INTEGER | FK → resultados_pipeline |
| `trecho_inicio` | REAL | segundos ou índice de início |
| `trecho_fim` | REAL | segundos ou índice de fim |
| `roi_json` | TEXT | JSON `{x,y,w,h}` para ROI de imagem |
| `canal` | TEXT | canal selecionado (lead ECG, etc.) |
| `slice_idx` | INTEGER | índice de slice para volumes 3D |
| `laudo_foco` | TEXT | laudo focado gerado |
| `data_hora` | TEXT | ISO 8601 |

Migração v2 (`database.py::_migrar_v2()`) é idempotente (`try/except OperationalError` em `ALTER TABLE`).

---

## Trade-offs conhecidos

- **Pipeline completo**: 4–6 chamadas LLM local (~30s cada) = 3–8 min por análise completa.
- **Non-Local Means**: ~10× mais lento que Gaussian. Em datasets >500 imagens com ruído alto, extração notavelmente mais lenta.
- **Volumes 3D**: `MAX_CONTENT_LENGTH = 4 GB` — uploads grandes consomem memória durante extração de slices.
- **Dependência de CDN**: Tailwind e Plotly vêm de CDN — sem internet, as telas quebram.
- **Encoding no Windows**: CrewAI emite emojis nos logs — `[EventBus Error]` com `charmap` (cp1252). Cosméticos. Mitigação: `.env` com `PYTHONUTF8=1`.

---

## Aviso ético

Este sistema é uma **ferramenta de suporte à decisão clínica**. Os laudos gerados pelos agentes IA **não substituem a avaliação de um médico habilitado**. Esse aviso é obrigatório em toda saída visual, em todos os laudos e na 5ª seção do Laudo de Amostra.

---

## Licença

MIT
