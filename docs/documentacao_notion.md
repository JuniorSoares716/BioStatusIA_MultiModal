# 📖 Dossiê de Documentação: BioStatusIA v3

> **Sistema de Apoio à Decisão Clínica (CDSS)** para análise automatizada de sinais biomédicos em **3 famílias (F1/F3/F4) + dados tabulares**, com IA Multi-Agente local (CrewAI + Ollama), AutoML de 6 modelos com avaliação enriquecida e laudos (populacional e de amostra).

Projeto acadêmico de mestrado em IA na Saúde — Fortaleza, CE.

> **Escopo v3:** áudio biomédico (F2) e vídeo médico (F5) foram **removidos do escopo**. O foco atual é F1 (sinais temporais), F3 (DICOM 2D), F4 (volume 3D) e dados tabulares.

---

## 🎯 1. Objetivos do Projeto (v3)

O **BioStatusIA v3** consolida o CDSS como plataforma de análise biomédica focada em três famílias de sinal e dados tabulares. Seus princípios fundamentais são:

1. **Aceitar qualquer tipo de dado em escopo** — imagem, sinal temporal (F1), DICOM 2D (F3), volume 3D (F4) ou tabular
2. **Auto-detecção do tipo de entrada** — 9 modos identificados automaticamente
3. **Pipeline agentificado especializado por família** — 5 crews, 8 agentes, 8 tools
4. **AutoML rigoroso** — 6 modelos, 5-fold CV, métricas clínicas enriquecidas (MCC, Kappa, ECE), balanceamento SMOTE/ADASYN, SHAP e teste de McNemar
5. **Laudos** — laudo populacional determinístico (nível da base) e laudo de amostra avulsa (Radiologista IA)

---

## 🛠️ 2. Requisitos do Sistema (v3)

### Requisitos Funcionais (RF)

*   **RF-01 (Detecção Universal):** Reconhecer e rotear automaticamente 9 modos de dados em 3 famílias de sinais biomédicos + tabular.
*   **RF-02 (SinalNormalizado):** Toda leitura de sinal deve produzir um `SinalNormalizado` — dataclass unificada com família, tipo, dados, taxa de amostragem, canais, metadados e dados de visualização (≤2000 pontos para Plotly).
*   **RF-03 (Análise Estatística Cega):** Calcular métricas sobre os dados brutos antes de qualquer limpeza, para evitar viés de seleção.
*   **RF-04 (Pré-Processamento Adaptativo):** Pipeline determinístico de decisão baseado em estatísticas detectadas (ruído, contraste, outliers, tamanho).
*   **RF-05 (AutoML Enriquecido):** 6 classificadores concorrentes, 5-fold StratifiedKFold, MCC, Kappa, ECE, latência de inferência, balanceamento SMOTE/ADASYN, SHAP no vencedor, teste de McNemar, seleção por AUC com preferência por sensibilidade ≥ 0.80.
*   **RF-06 (Laudo IA Especializado):** Cada família de sinal tem agente e task específicos — `radiologista_ia` interpreta achados considerando o contexto clínico do tipo de sinal.
*   **RF-07 (Laudo Populacional):** Endpoint `GET /laudo_populacional/<id>` monta relatório determinístico do dataset completo (distribuição, correlações, pódio) a partir do `pipeline_json` — sem depender do LLM.
*   **RF-08 (Laudo de Amostra):** Endpoint `POST /laudo_amostra` gera laudo do Radiologista IA para um arquivo avulso (upload) ou análise existente (resultado_id), com 5 seções obrigatórias.
*   **RF-09 (Interface em 4 Abas):** Estatísticas/Biomarcadores, Pré-processamento & Engenharia de Features, AutoML (apenas), Laudo (Dossiê Radiologista IA + Histórico + Laudo de Amostra + Laudo Populacional).
*   **RF-10 (Inferência Individual):** O modelo vencedor é persistido (`models/vencedor_<familia>.pkl`) e reutilizado para classificar novas amostras.

### Requisitos Não-Funcionais (RNF)

*   **RNF-01 (Privacidade Total):** Processamento 100% local — nenhum dado de saúde enviado para APIs externas de IA.
*   **RNF-02 (LLM Local):** `qwen2.5:3b` via Ollama — sem dependência de conectividade para inferência do LLM.
*   **RNF-03 (Capacidade para Volumes 3D):** `MAX_CONTENT_LENGTH = 4 GB` para suportar TC e RM completos.
*   **RNF-04 (Idempotência do Banco):** Migração v2 via `ALTER TABLE` com `try/except` — segura em re-execuções.
*   **RNF-05 (Retrocompatibilidade):** Pipeline CLI (`main.py`) e crews originais de imagem mantidos intactos.
*   **RNF-06 (Dependência de CDN):** Tailwind CSS e Plotly.js vêm de CDN — o primeiro carregamento das telas exige acesso à internet.

---

## 🧬 3. As 3 Famílias de Sinal + Tabular

### Família F1 — Sinais Temporais Fisiológicos

| Tipo | Biblioteca | Features Diagnósticas |
|---|---|---|
| ECG | `wfdb`, `mne` | FC_bpm, RMSSD, SDNN, pNN50, R-peaks |
| EEG | `mne` | Bandas δ/θ/α/β/γ, ratio α/β |
| EMG | `mne` | RMS envelope, frequência mediana |
| EOG / PPG / PA | `mne`, `scipy` | RMS, PSD, centroide espectral |
| Espirometria | ElementTree (XML) | FVC, FEV1, FEV1/FVC, PEF |

**Formatos:** `.dat/.hea` (WFDB), `.edf/.bdf` (EDF+/BDF), `.mat` (Matlab), `.xml` (Schiller/Cosmed), `.c3d`

### Família F3 — Imagem DICOM 2D

| Modalidade | SOPClassUID detectado | Interpretação Clínica |
|---|---|---|
| Raio-X de Tórax | `1.2.840.10008.5.1.4.1.1.1.1` | Consolidações, derrames, pneumotórax |
| Mamografia | `1.2.840.10008.5.1.4.1.1.1.2` | Densidades BI-RADS, microcalcificações |
| Ultrassom | `1.2.840.10008.5.1.4.1.1.6` | Ecogenicidade, sombreamento |

**Processamento:** Janelamento Hounsfield automático via `WindowCenter/Width`, Rescale Slope/Intercept, 9 biomarcadores radiômicos + DICOM-específicos

**Formato:** `.dcm` (arquivo único)

### Família F4 — Volumes 3D

| Formato | Biblioteca | Convenção de Eixos |
|---|---|---|
| `.nii` / `.nii.gz` | nibabel | Transpõe (X,Y,Z) → (Z,Y,X) axial-first |
| `.mha` | SimpleITK | Lido diretamente (Z,Y,X) |
| Série DICOM (≥10 `.dcm`) | pydicom | Ordenada por InstanceNumber, empilhada (D,H,W) |

**Features 3D:** stats globais (média/desvio/mediana/skewness/kurtosis/P5/P95/voxels_altos), GLCM por plano ortogonal (axial/coronal/sagital), morfologia 3D (volume em mm³, esfericidade, bounding box axes)

### Dados Tabulares

| Aspecto | Detalhe |
|---|---|
| Formatos | `.csv`, `.txt`, `.tsv` |
| Separador | Detectado automaticamente (`,`, `;`, `\t`, `|`) |
| Encoding | UTF-8 com fallback Latin-1 |
| Coluna-rótulo | Por nome (`label`, `class`, `diagnosis`, `target`, ...) ou última coluna com 2–10 valores únicos |

### Imagem comum (radiômica)

`.png/.jpg/.bmp/.tif` processados via OpenCV/scikit-image — 9 biomarcadores radiômicos (morfologia, GLCM, distribuição).

---

## 🤖 4. Arquitetura Multi-Agente (v3)

### 5 Crews CrewAI

```
Crew 1 — BioStatusIACrew (Imagem / radiômica)
  ├── engenheiro_pdi         [FerramentaAnaliseBase]          → analise_base.json
  ├── analista_tecnico       [FerramentaExtrairBiomarcadores] → biomarcadores.json
  ├── cientista_dados        [FerramentaTreinarClassificador] → metricas.json
  └── radiologista_ia        [sem tool]                       → Laudo Markdown

Crew 2 — BioStatusIACrewTabular (CSV/TXT)
  └── bioestatistico         [FerramentaAnaliseTabular]       → Laudo Markdown

Crew 3 — BioStatusIACrewSinal (F1)
  ├── analista_sinais_fisiologicos
  │     └── [FerramentaExtrairSinalTemporal] → biomarcadores_temporal.json
  └── radiologista_ia        [sem tool]        → Laudo Markdown

Crew 4 — BioStatusIACrewImagem3D (F3 + F4)
  ├── especialista_imagem_medica
  │     ├── [FerramentaExtrairDICOM]          → biomarcadores_dicom.json
  │     └── [FerramentaExtrairVolume3D]       → biomarcadores_volumetrico.json
  └── radiologista_ia        [sem tool]        → Laudo Markdown

Crew 5 — BioStatusIACrewInterativo (Laudo de Amostra)
  └── radiologista_ia_interativo [sem tool, max_iter=4]       → Laudo Interativo Markdown
```

### 8 Agentes (`config/agents.yaml`)

| Agente | Função | Crews |
|---|---|---|
| `engenheiro_pdi` | PDI + estratégia adaptativa | Crew 1 |
| `analista_tecnico` | Extração radiômica em lote | Crew 1 |
| `cientista_dados` | AutoML + seleção de modelo | Crew 1 |
| `radiologista_ia` | Laudo clínico geral | Crews 1, 3, 4 |
| `bioestatistico` | Análise e laudo tabular | Crew 2 |
| `analista_sinais_fisiologicos` | Features F1 | Crew 3 |
| `especialista_imagem_medica` | Features DICOM F3/F4 | Crew 4 |
| `radiologista_ia_interativo` | Laudo focado em amostra | Crew 5 |

### 8 Tools CrewAI (`tools/`)

| Tool | Wrapper | Output |
|---|---|---|
| `FerramentaAnaliseBase` | `analise_base.py` | `analise_base.json` |
| `FerramentaExtrairBiomarcadores` | `extracao.py` (lote) | `biomarcadores.json` |
| `FerramentaTreinarClassificador` | `classificador.py` | `metricas.json` |
| `FerramentaAnaliseTabular` | `dados_tabulares.py` | resumo texto |
| `FerramentaAnaliseImagem` | `extracao.py` (single, legado) | resumo texto |
| `FerramentaExtrairSinalTemporal` | `extracao_temporal.py` | `biomarcadores_temporal.json` |
| `FerramentaExtrairDICOM` | `extracao_dicom.py` | `biomarcadores_dicom.json` |
| `FerramentaExtrairVolume3D` | `extracao_volumetrica.py` | `biomarcadores_volumetrico.json` |

### Protocolo de comunicação via filesystem

```
static/runs/run_<timestamp>/
├── analise_base.json               ← escrito pelo engenheiro_pdi
├── biomarcadores.json              ← escrito pelo analista_tecnico
├── metricas.json                   ← escrito pelo cientista_dados
├── biomarcadores_temporal.json     ← escrito por analista_sinais_fisiologicos
├── biomarcadores_dicom.json        ← escrito por especialista_imagem_medica
└── biomarcadores_volumetrico.json  ← escrito por especialista_imagem_medica
```

Cada tool persiste o JSON completo → retorna resumo em texto ao LLM → próximo agente lê o JSON. O LLM nunca ingere megabytes de dados numéricos.

---

## 📊 5. AutoML Enriquecido (`pipeline/avaliacao_modelos.py`)

### 6 Modelos Concorrentes

| # | Modelo | Tipo | Justificativa Clínica |
|---|---|---|---|
| 1 | Regressão Logística | Linear | Baseline interpretável, coeficientes = importância de features |
| 2 | KNN (k=5) | Distância | Robusto em distribuições multimodais |
| 3 | SVM RBF | Kernel | Estado da arte em alta dimensão com poucas amostras |
| 4 | Random Forest (100) | Ensemble | Alta estabilidade, importância de features |
| 5 | Gradient Boosting (100) | Boosting | Melhor AUC em dados tabulares médicos |
| 6 | MLP (64×32) | Rede Neural | Mapeamento de padrões não-lineares complexos |

### Protocolo de Avaliação

```
Dataset com rótulos e ≥10 amostras (2 classes)
  ├── StandardScaler (fit apenas no treino)
  ├── Balanceamento SMOTE / ADASYN (quando desbalanceado)
  ├── 5-fold StratifiedKFold (random_state=42)
  │     Para cada fold × cada modelo:
  │       - treino + inferência cronometrados
  │       - predict_proba → AUC, ECE
  │       - classification_report → sensibilidade, especificidade, F1
  │       - MCC, Kappa
  ├── Holdout 20% (test set) → ROC, confusion_matrix
  ├── McNemar test (top-2 modelos)
  └── SHAP no modelo vencedor
```

### Métricas Clínicas (além das padrão)

| Métrica | Fórmula | Relevância Clínica |
|---|---|---|
| **Sensibilidade** | TP / (TP + FN) | Minimiza falso-negativo — detectar malignidade |
| **Especificidade** | TN / (TN + FP) | Minimiza falso-positivo — reduz biópsias desnecessárias |
| **MCC** | Matthews Correlation Coefficient | Robusto a desbalanceamento de classes |
| **Kappa** | Cohen's Kappa | Concordância além do acaso |
| **ECE** | Calibration error (10 bins) | Confiança do modelo = probabilidade real |
| **Latência (ms)** | tempo/amostra | Viabilidade clínica em tempo real |
| **McNemar** | χ² + p-value | Diferença estatisticamente significativa entre modelos |

### Critério de Seleção do Vencedor

```python
# Prefere modelo com sensibilidade >= 0.80
candidatos = [m for m in modelos if m["sensibilidade"] >= 0.80]
if candidatos:
    vencedor = max(candidatos, key=lambda m: m["auc"])
else:
    vencedor = max(modelos, key=lambda m: m["auc"])
```

O vencedor é persistido em `models/vencedor_<familia>.pkl` (`pipeline/inferencia.py`) para inferência de amostras individuais.

---

## 🎨 6. Interface Web — 4 Abas (Tela 2)

### Aba 1 — Estatísticas & Biomarcadores

- Tabela de biomarcadores (média ± desvio por categoria)
- Boxplot de distribuição por biomarcador (Plotly.js)
- Schema tabular (colunas, tipos, missing values) para modos CSV
- 31 métricas univariadas por feature (sidebar interativa)
- Heatmap de correlação

### Aba 2 — Pré-processamento & Engenharia de Features

Cards explicativos com decisão + justificativa para cada etapa:
- Denoising escolhido (Non-Local Means vs Gaussian) com ruído detectado
- Normalização escolhida (Percentil 1–99% vs Min-Max) com % de outliers
- Equalização (CLAHE vs nenhuma) com contraste detectado
- Resize (obrigatório vs não necessário) com tamanhos detectados

Seção **Engenharia de Features** (sempre visível):
- Gráfico de importância das features derivadas (Sobel, histograma, LBP, quadrantes)
- Cards com os valores médios da amostra
- Estado vazio explicativo quando a análise não possui features derivadas (ex.: modo tabular)

### Aba 3 — AutoML

**Apenas AutoML** (o laudo Markdown e o histórico foram movidos para a Aba 4):
- Pódio dos 6 modelos com AUC, sensibilidade, especificidade
- Gráfico comparativo de métricas (Plotly.js) e radar
- Curva ROC (Plotly.js)
- Matriz de Confusão (Plotly.js heatmap)
- Resultado do teste de McNemar

### Aba 4 — Laudo

No topo: **Dossiê e Laudo do Radiologista IA** (laudo Markdown da análise atual, com impressão e Aviso Ético) + **Histórico do Prontuário (SQLite)** — ambos movidos da antiga Aba 3.

**Seção A — Laudo de Amostra:** o usuário faz upload de um arquivo avulso **ou** seleciona uma análise anterior do banco, e recebe um laudo do Radiologista IA.

```
Toggle: [📁 Upload de Arquivo]  |  [🗄️ Selecionar Análise Anterior]
   └──► POST /laudo_amostra  (multipart com arquivo OU resultado_id)
              └──► BioStatusIACrewInterativo → laudo focado (5 seções)
```

**Laudo Interativo — 5 seções obrigatórias:**
1. **Achado Principal** — o que chama atenção neste trecho/região
2. **Severidade Estimada** — escala 1 (normal) a 5 (crítico) com justificativa
3. **Comparação com Referência** — valores normais esperados vs achado
4. **Recomendação Imediata** — próximo passo clínico sugerido
5. **Aviso Ético** — este laudo é de suporte e NÃO substitui avaliação médica

### Laudo Populacional (nível da base)

`GET /laudo_populacional/<id>` retorna um relatório determinístico do dataset completo — distribuição estatística, correlações de biomarcadores e pódio final do AutoML — montado a partir do `pipeline_json` persistido, sem depender do LLM.

---

## 💾 7. Banco de Dados SQLite (3 tabelas)

### `analises`
```sql
CREATE TABLE analises (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    data_hora TEXT NOT NULL,
    imagem    TEXT NOT NULL,
    categoria TEXT NOT NULL,  -- BENIGNO | MALIGNO | INDEFINIDO | TABULAR | SINAL
    laudo     TEXT
);
```

### `resultados_pipeline` (expandida na migração v2)
```sql
CREATE TABLE resultados_pipeline (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    data_hora     TEXT NOT NULL,
    dataset_path  TEXT,
    n_imagens     INTEGER,
    pipeline_json TEXT,           -- payload completo para Tela 2
    melhor_modelo TEXT,
    analise_id    INTEGER REFERENCES analises(id),
    familia_sinal TEXT,           -- F1 | F3 | F4 | ""
    sinal_tipo    TEXT            -- ECG | Raio-X | ...
);
```

### `laudos_interativos`
```sql
CREATE TABLE laudos_interativos (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    resultado_id  INTEGER REFERENCES resultados_pipeline(id),
    trecho_inicio REAL,
    trecho_fim    REAL,
    roi_json      TEXT,
    canal         TEXT,
    slice_idx     INTEGER,
    laudo_foco    TEXT,
    data_hora     TEXT NOT NULL
);
```

**Migração v2** (`database.py::_migrar_v2()`): usa `try/except OperationalError` para `ALTER TABLE` — idempotente em re-execuções.

---

## 🌐 8. API Flask — Rotas (v3)

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Tela 1 — Upload |
| `POST` | `/analisar` | Dispara pipeline → redireciona para `/resultados/<id>` |
| `GET` | `/resultados/<id>` | Tela 2 — 4 abas |
| `GET` | `/api/historico` | JSON com os últimos resultados (seletor da Aba 4) |
| `GET` | `/api/exemplos/<id>` | Exemplos individuais do dataset da análise |
| `GET` | `/laudo_populacional/<id>` | Laudo populacional determinístico (JSON com HTML/MD + pódio) |
| `POST` | `/laudo_amostra` | Laudo de amostra avulsa (upload) ou de análise existente (resultado_id) |

### Payload de `/laudo_amostra` (POST, multipart)

```
resultado_id = 42            # análise existente no banco
   — OU —
arquivo = <upload>           # arquivo avulso (imagem, sinal, DICOM, ...)
```

Resposta:
```json
{
    "laudo_html": "<h2>Achado Principal</h2>...",
    "laudo_id": 7
}
```

---

## 📦 9. Stack Tecnológica Completa (v3)

| Camada | Tecnologia | Versão |
|---|---|---|
| Python | 3.10–3.12 | `>=3.10,<3.13` |
| Gerenciador | `uv` | — |
| LLM | Ollama `qwen2.5:3b` | local |
| Agentes | CrewAI | `>=0.203.1,<1.0.0` |
| Servidor Web | Flask | `>=3.0.0` |
| Banco | SQLite (built-in) | 3 tabelas |
| Sinais F1 | MNE-Python, wfdb, scipy | — |
| DICOM F3 | pydicom | — |
| Volumes F4 | nibabel, SimpleITK | — |
| Visão | OpenCV (headless), scikit-image | — |
| AutoML | scikit-learn | `>=1.3.0` |
| Balanceamento / Interpretabilidade | imbalanced-learn, SHAP | — |
| Gráficos | Plotly.js | 2.32 (CDN) |
| Frontend | Tailwind CSS | latest (CDN) |
| Capacidade Upload | — | 4 GB (`MAX_CONTENT_LENGTH`) |

---

## 🗃️ 10. Datasets de Validação (Kaggle)

| Família | Dataset | Kaggle Slug | Formato | Classes |
|---|---|---|---|---|
| **F1 — ECG** | ECG Heartbeat Categorization (MIT-BIH) | `shayanfazeli/heartbeat` | CSV | 5 arritmias |
| **F3 — Raio-X** | Chest X-Ray Images (Pneumonia) | `paultimothymooney/chest-xray-pneumonia` | JPEG | NORMAL / PNEUMONIA |

**Nota F4 (Volumes 3D):** Datasets de TC/RM (ex: BRATS, LUNA16) são muito pesados para KaggleHub — usar download direto via site oficial ou `kaggle datasets download` na CLI.

**Estrutura esperada para validação com o pipeline:**
```
dataset_raio_x/
├── NORMAL/          ← automaticamente reconhecido como "normal"
└── PNEUMONIA/       ← mapeado para malignant_keywords → rótulo 1
```

---

## 🔄 11. Fluxo Completo v3 (Diagrama)

```
ENTRADA (qualquer em escopo)
      │
      ▼
detectar_estrutura()   ←── app.py (9 modos)
      │
      ├── imagem_unica / imagens_soltas / dataset_rotulado / multimodal
      │         └──► BioStatusIACrew (4 agentes)
      │               engenheiro_pdi → analista_tecnico → cientista_dados → radiologista_ia
      │
      ├── tabular (CSV/TXT)
      │         └──► BioStatusIACrewTabular (1 agente)
      │               bioestatistico → laudo tabular
      │
      ├── sinal_temporal (F1)
      │         └──► BioStatusIACrewSinal (2 agentes)
      │               analista_sinais_fisiologicos → radiologista_ia
      │
      ├── imagem_dicom_2d / volume_3d (F3/F4)
      │         └──► BioStatusIACrewImagem3D (2 agentes)
      │               especialista_imagem_medica → radiologista_ia
      │
      └── multimodal_expandido
                └──► BioStatusIACrew + sub-crews
      │
      ▼
avaliacao_modelos.py    ←── se há rótulos + ≥10 amostras (2 classes)
      │   6 modelos × 5-fold CV
      │   MCC + Kappa + ECE + SMOTE/ADASYN + McNemar + SHAP
      ▼
inferencia.py           ←── persiste vencedor em models/vencedor_<familia>.pkl
      ▼
database.py::salvar_resultado()
      │   analises + resultados_pipeline (familia_sinal, sinal_tipo)
      ▼
Tela 2 — 4 Abas
      │
      ├── GET /laudo_populacional/<id> → relatório determinístico da base
      │
      └── Aba 4 — Seção A: upload/seleção de amostra
                  │
                  └──► POST /laudo_amostra
                              └──► BioStatusIACrewInterativo
                                    radiologista_ia_interativo
                                    → 5 seções + aviso ético
                                    → laudos_interativos (DB)
```

---

## ⚠️ 12. Trade-offs e Limitações Conhecidas

| Aspecto | Limitação | Mitigação |
|---|---|---|
| Tempo de pipeline | 4–6 chamadas LLM × ~30s = 3–8 min | Pipeline assíncrono no roadmap |
| Non-Local Means | ~10× mais lento que Gaussian | Apenas ativado com ruído > 0.05 |
| Volumes 3D | Carregamento completo em RAM | `MAX_CONTENT_LENGTH = 4 GB` |
| Dependência de CDN | Tailwind/Plotly via CDN — sem internet as telas quebram | Garantir conectividade no 1º load |
| LLM indisponível | Sem Ollama rodando, laudos IA falham | Iniciar `ollama serve` + `ollama pull qwen2.5:3b` |
| Encoding Windows | CrewAI EventBus emite emojis → `[EventBus Error]` no terminal | `.env` com `PYTHONUTF8=1` |
| F4 Kaggle | Datasets de TC são muito grandes para KaggleHub | Download direto |

---

## 🔧 13. Configuração do Ambiente

### `.env` completo

```env
MODEL=ollama/qwen2.5:3b
API_BASE=http://localhost:11434
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
```

### Dependências principais (`pyproject.toml`)

```
crewai[tools]>=0.203.1,<1.0.0
flask>=3.0.0
scikit-learn>=1.3.0
opencv-python-headless
scikit-image
numpy>=1.21.0,<2.0.0
scipy
mne
wfdb
pydicom
nibabel
simpleitk
imbalanced-learn
shap
markdown
python-dotenv
```

### Como rodar

```bash
# Instalar dependências
uv sync

# Iniciar Ollama (em outro terminal) + baixar modelo
ollama serve
ollama pull qwen2.5:3b

# Iniciar servidor web
uv run flask --app src/biostatusia/app.py run --port 5000

# Ou usar o CLI retrocompatível
uv run biostatsia
```

---

## 🔒 14. Aviso Ético Obrigatório

> **Este sistema é uma ferramenta de suporte à decisão clínica (CDSS). Os laudos gerados pelos agentes de IA são preliminares e baseados em análise computacional automática. Eles NÃO substituem, em nenhuma circunstância, a avaliação clínica de um médico habilitado. Todo resultado deve ser interpretado por um profissional de saúde qualificado antes de qualquer decisão diagnóstica ou terapêutica.**

Este aviso é **obrigatório** em:
- Toda saída visual da Tela 2
- Todo laudo do `radiologista_ia` (task: `tarefa_laudo`, `tarefa_laudo_sinal`)
- 5ª seção do `radiologista_ia_interativo` (task: `tarefa_laudo_interativo`)
- Todo laudo do `bioestatistico` (task: `tarefa_laudo_tabular`)

---

## 📝 15. Histórico de Versões

| Versão | Data | Principais Mudanças |
|---|---|---|
| v1.0 | 2025-12 | CDSS original: ultrassom mamário, 5 modos, 2 crews, 5 agentes, 5 tools |
| v1.5 | 2026-01 | AutoML 6 modelos, interface em abas premium, sidebar interativa de estatísticas |
| v2.0 | 2026-06 | 5 famílias de sinal, 10 modos, 6 crews, 11 agentes, 10 tools, Laudo Interativo, avaliação enriquecida |
| **v3.0** | **2026-07** | **Escopo reduzido a F1/F3/F4 + tabular (áudio/vídeo removidos); 9 modos, 5 crews, 8 agentes, 8 tools; ingestão Pydantic; AutoML enriquecido (MCC/Kappa/SMOTE/ADASYN/SHAP); inferência individual (vencedor do pódio); laudo populacional determinístico + laudo de amostra; scaffold CNN** |
