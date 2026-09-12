import csv
from pathlib import Path

import numpy as np

LABEL_KEYWORDS = {
    "label", "class", "diagnosis", "diagnostico", "target",
    "categoria", "outcome", "y", "result", "resultado",
}
MALIGNANT_KEYWORDS = {
    "m", "malignant", "maligno", "malign", "positive",
    "1", "true", "yes", "sim", "abnormal",
}


def _carregar_xlsx(caminho_p: Path) -> tuple[list[str], list[list[str]]] | None:
    """Lê a primeira aba de uma planilha .xlsx e converte para o mesmo
    formato (header, linhas-como-string) usado pelo resto do pipeline
    tabular — assim, detecção de schema, rótulo, imputação etc. funcionam
    sem nenhuma mudança, seja o arquivo .csv ou .xlsx.

    Carrega os bytes em memória (BytesIO) em vez de passar o caminho direto
    para o openpyxl: a biblioteca recusa arquivos cujo NOME não termina em
    .xlsx/.xlsm/etc, mesmo que o conteúdo seja um .xlsx válido — o que
    quebraria justamente o caso que queremos suportar (arquivo com a
    extensão "errada", mas conteúdo Excel de verdade)."""
    try:
        from openpyxl import load_workbook
    except ImportError:
        raise RuntimeError(
            "Este arquivo é uma planilha Excel, mas a biblioteca 'openpyxl' "
            "não está instalada no servidor. Instale com "
            "'pip install openpyxl' ou exporte a planilha como .csv."
        )

    import io
    with open(caminho_p, "rb") as f:
        buffer = io.BytesIO(f.read())

    wb = load_workbook(filename=buffer, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    linhas_brutas = list(ws.iter_rows(values_only=True))
    wb.close()

    linhas = [
        [("" if v is None else str(v)) for v in linha]
        for linha in linhas_brutas
        if any(v is not None and str(v).strip() for v in linha)
    ]
    if len(linhas) < 2:
        return None

    header_candidato = [h.strip() for h in linhas[0]]
    if _parece_linha_de_dados(header_candidato):
        header = [str(i) for i in range(len(header_candidato))]
        data = linhas
    else:
        header = header_candidato
        data = linhas[1:]
    return header, data


def _eh_zip_binario(caminho_p: Path) -> bool:
    """Detecta pelo CONTEÚDO (não pela extensão) se o arquivo é um .xlsx de
    verdade — arquivos .xlsx são, por baixo, um ZIP (assinatura 'PK'). Cobre
    o caso comum de um arquivo com a extensão errada: um Excel binário
    salvo/exportado como .csv, ou um texto puro salvo como .xlsx."""
    try:
        with open(caminho_p, "rb") as f:
            assinatura = f.read(4)
        return assinatura[:2] == b"PK"
    except OSError:
        return False


def carregar_csv(caminho: str) -> tuple[list[str], list[list[str]]] | None:
    """Carrega uma tabela (CSV/TXT delimitado OU planilha .xlsx). O formato
    real é detectado pelo CONTEÚDO do arquivo, não pela extensão — um
    arquivo ``.csv`` que na verdade é um Excel binário (ou um ``.xlsx`` que
    na verdade é texto puro) é lido corretamente de qualquer forma."""
    caminho_p = Path(caminho)
    if not caminho_p.exists() or not caminho_p.is_file():
        return None

    if _eh_zip_binario(caminho_p):
        return _carregar_xlsx(caminho_p)

    try:
        with open(caminho_p, "r", encoding="utf-8-sig") as f:
            sample = f.read(4096)
    except UnicodeDecodeError:
        with open(caminho_p, "r", encoding="latin-1") as f:
            sample = f.read(4096)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimitador = dialect.delimiter
    except csv.Error:
        delimitador = ","

    try:
        with open(caminho_p, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter=delimitador)
            rows = [row for row in reader if any(c.strip() for c in row)]
    except UnicodeDecodeError:
        with open(caminho_p, "r", encoding="latin-1") as f:
            reader = csv.reader(f, delimiter=delimitador)
            rows = [row for row in reader if any(c.strip() for c in row)]

    if len(rows) < 2:
        return None

    header_candidato = [h.strip() for h in rows[0]]
    if _parece_linha_de_dados(header_candidato):
        # CSV sem cabeçalho (comum em datasets de benchmark, ex.: ECG5000):
        # a "primeira linha" é, na verdade, uma amostra numérica de verdade.
        # Gera nomes de coluna genéricos e preserva TODAS as linhas como dados.
        header = [str(i) for i in range(len(header_candidato))]
        data = rows
    else:
        header = header_candidato
        data = rows[1:]
    return header, data


def _parece_linha_de_dados(valores: list[str]) -> bool:
    """True se a "linha de cabeçalho" candidata parece, na verdade, ser uma
    amostra numérica (CSV sem cabeçalho) — ex.: ['-1.63', '0.42', ..., '1']."""
    if not valores:
        return False
    numericos = 0
    for v in valores:
        v = v.strip()
        if not v:
            continue
        try:
            float(v.replace(",", "."))
            numericos += 1
        except ValueError:
            pass
    return numericos / len(valores) >= 0.9


def candidatos_coluna_alvo(header: list[str], data: list[list[str]],
                            n_amostragem_deteccao: int = 200) -> list[dict]:
    """
    Lista TODAS as colunas do CSV/planilha que são candidatas plausíveis a
    coluna-alvo (classe/rótulo) — qualquer coluna com entre 2 e 10 valores
    únicos (mesmo critério usado no fallback de `detectar_schema`).

    Para cada candidata, devolve a contagem REAL de linhas por classe (no
    dataset inteiro, não só numa amostra) — para o usuário ver exatamente
    quais classes existem e quantas amostras cada uma tem antes de escolher
    quais usar.

    Escaneia a base INTEIRA por coluna (não uma amostra do início) — com
    saída antecipada assim que uma coluna ultrapassa 10 valores distintos,
    o que mantém isso rápido mesmo em bases largas, já que a maioria das
    colunas numéricas contínuas estoura o limite nas primeiras linhas. Uma
    amostra do início sozinha erraria em datasets ORDENADOS por classe
    (comuns em benchmarks — ex.: MIT-BIH, onde as primeiras ~18 mil linhas
    são todas de uma única classe e a diversidade só aparece bem mais
    adiante no arquivo).
    """
    candidatos = []
    for i, nome in enumerate(header):
        contagem: dict[str, int] = {}
        estourou = False
        for row in data:
            if i < len(row) and row[i].strip():
                v = row[i].strip()
                contagem[v] = contagem.get(v, 0) + 1
                if len(contagem) > 10:
                    estourou = True
                    break
        if estourou or not (2 <= len(contagem) <= 10):
            continue

        classes_ordenadas = sorted(contagem.items(), key=lambda kv: (-kv[1], kv[0]))
        candidatos.append({
            "indice": i,
            "nome": nome,
            "n_classes": len(contagem),
            "classes": [{"valor": v, "n": n} for v, n in classes_ordenadas],
            "exemplos": [v for v, _ in classes_ordenadas[:6]],
        })
    return candidatos


def filtrar_por_classes(header: list[str], data: list[list[str]],
                         coluna_idx: int, classes_permitidas: list[str]) -> list[list[str]]:
    """
    Mantém só as linhas cujo valor na coluna `coluna_idx` está em
    `classes_permitidas` — usado quando o usuário escolhe explicitamente
    quais classes comparar dentro de uma coluna com mais de 2 valores
    (o BioStatusIA hoje só treina classificação binária).
    """
    permitidas = set(classes_permitidas)
    return [row for row in data if coluna_idx < len(row) and row[coluna_idx].strip() in permitidas]


LIMITE_CARDINALIDADE_ONEHOT = 20


def detectar_schema(header: list[str], data: list[list[str]],
                     label_idx_forcado: int | None = None) -> dict:
    """Identifica coluna-rótulo, colunas numéricas e colunas categóricas de
    baixa cardinalidade (codificadas via one-hot).

    `label_idx_forcado`: quando informado (ex.: escolha explícita do
    usuário no modal de confirmação), usa essa coluna como rótulo
    diretamente, ignorando a heurística automática abaixo."""
    n_cols = len(header)

    if label_idx_forcado is not None and 0 <= label_idx_forcado < n_cols:
        label_idx = label_idx_forcado
    else:
        label_idx = None
        for i, name in enumerate(header):
            if name.lower().strip() in LABEL_KEYWORDS:
                label_idx = i
                break

        def _cardinalidade_ate(idx: int, limite: int = 10) -> int | None:
            """Nº de valores únicos da coluna `idx`, escaneando a base
            inteira — mas com saída antecipada assim que ultrapassa
            `limite` (a maioria das colunas numéricas contínuas estoura o
            limite nas primeiras dezenas de linhas, então isso continua
            rápido mesmo em bases grandes). Devolve None se ultrapassar.
            Usa a base inteira, não uma amostra do início — necessário
            porque muitos CSVs de benchmark vêm ORDENADOS por classe (ex.:
            MIT-BIH: as primeiras ~18 mil linhas são todas de uma única
            classe; uma amostra pequena do início nunca veria as outras)."""
            unicos: set[str] = set()
            for row in data:
                if idx < len(row) and row[idx].strip():
                    unicos.add(row[idx].strip())
                    if len(unicos) > limite:
                        return None
            return len(unicos) if unicos else None

        if label_idx is None and n_cols > 1:
            for idx in (n_cols - 1, 0):  # prioridade: última coluna, depois primeira
                card = _cardinalidade_ate(idx)
                if card is not None and 2 <= card <= 10:
                    label_idx = idx
                    break

        if label_idx is None and n_cols > 1:
            # Nem nome reconhecido nem primeira/última coluna serviram — a
            # coluna-alvo real pode estar no meio do arquivo (ex.: dataset
            # da UCI onde "status" não é a última coluna). Escaneia TODAS as
            # colunas; se só UMA for uma candidata plausível (2 a 10 valores
            # únicos), usa ela automaticamente — sem ambiguidade, não
            # precisa nem passar pelo modal de confirmação. Se houver 2+
            # candidatas, deixa como está (None) e a checagem de
            # `candidatos_coluna_alvo()` no portão de confirmação assume,
            # pedindo para o usuário escolher.
            candidatas_completas = [i for i in range(n_cols)
                                    if (c := _cardinalidade_ate(i)) is not None and 2 <= c <= 10]
            if len(candidatas_completas) == 1:
                label_idx = candidatas_completas[0]

    numeric_cols: list[int] = []
    for i in range(n_cols):
        if i == label_idx:
            continue
        eh_numerica = True
        amostras_parseadas = 0
        for row in data[:30]:
            if i < len(row) and row[i].strip():
                try:
                    float(row[i].replace(",", "."))
                    amostras_parseadas += 1
                except ValueError:
                    eh_numerica = False
                    break
        if eh_numerica and amostras_parseadas > 0:
            numeric_cols.append(i)

    # Colunas categóricas de baixa cardinalidade (ex.: Gender, Blood Type,
    # Admission Type) — viram variáveis via one-hot encoding, em vez de
    # serem descartadas silenciosamente por não serem numéricas. Excluídas:
    # a coluna-rótulo, colunas já numéricas, e colunas de alta cardinalidade
    # (nomes, IDs, texto livre) — que só inflariam a dimensionalidade sem
    # sinal real (ex.: "Doctor" com centenas de nomes distintos).
    categorical_cols: list[int] = []
    categorical_categorias: dict[int, list[str]] = {}
    amostra_cardinalidade = data[:300]
    for i in range(n_cols):
        if i == label_idx or i in numeric_cols:
            continue
        valores_amostra = [row[i].strip() for row in amostra_cardinalidade if i < len(row) and row[i].strip()]
        if not valores_amostra or len(set(valores_amostra)) > LIMITE_CARDINALIDADE_ONEHOT:
            continue  # já claramente alta cardinalidade (ou vazia) — descarta sem escanear tudo
        valores_completos = [row[i].strip() for row in data if i < len(row) and row[i].strip()]
        unicos = sorted(set(valores_completos))
        if 2 <= len(unicos) <= LIMITE_CARDINALIDADE_ONEHOT:
            categorical_cols.append(i)
            categorical_categorias[i] = unicos

    onehot_feature_names = [
        f"{header[i]}={cat}" for i in categorical_cols for cat in categorical_categorias[i]
    ]

    return {
        "label_idx": label_idx,
        "label_name": header[label_idx] if label_idx is not None else None,
        "feature_indices": numeric_cols,
        "feature_names": [header[i] for i in numeric_cols] + onehot_feature_names,
        "n_features": len(numeric_cols) + len(onehot_feature_names),
        "n_amostras": len(data),
        "categorical_indices": categorical_cols,
        "categorical_categorias": categorical_categorias,
        "categorical_names": [header[i] for i in categorical_cols],
        "n_features_numericas": len(numeric_cols),
        "n_features_categoricas_onehot": len(onehot_feature_names),
    }


def extrair_features(data: list[list[str]], schema: dict) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Converte data em (X, y, label_map) preservando valores ausentes como NaN.

    X inclui tanto as colunas numéricas quanto as colunas categóricas de
    baixa cardinalidade (`schema["categorical_indices"]`), codificadas via
    one-hot — nessa ordem: numéricas primeiro, depois as one-hot, alinhado
    com `schema["feature_names"]`."""
    feature_idx = schema["feature_indices"]
    label_idx = schema["label_idx"]
    categorical_idx = schema.get("categorical_indices", [])
    categorical_categorias = schema.get("categorical_categorias", {})

    X_rows: list[list[float]] = []
    y_raw: list[str] = []

    for row in data:
        features = []
        for i in feature_idx:
            val = row[i].strip() if i < len(row) else ""
            if not val or val.lower() in ["?", "nan", "null", "none", "na", "-"]:
                features.append(np.nan)
            else:
                try:
                    features.append(float(val.replace(",", ".")))
                except ValueError:
                    features.append(np.nan)

        # One-hot das colunas categóricas — valor ausente ou categoria nunca
        # vista na detecção do schema vira tudo zero (não precisa de
        # imputação separada; "nenhuma categoria marcada" já representa isso).
        for i in categorical_idx:
            val = row[i].strip() if i < len(row) else ""
            for cat in categorical_categorias.get(i, []):
                features.append(1.0 if val == cat else 0.0)

        # Só adiciona se o rótulo for válido (se aplicável)
        if label_idx is not None:
            lbl = row[label_idx].strip() if label_idx < len(row) else ""
            if lbl:
                X_rows.append(features)
                y_raw.append(lbl)
        else:
            X_rows.append(features)

    X = np.array(X_rows, dtype=np.float64)
    label_map: dict = {}

    if label_idx is None or not y_raw:
        return X, None, label_map

    unique_labels = sorted(set(y_raw))
    if len(unique_labels) == 2:
        for lbl in unique_labels:
            label_map[lbl] = 1 if lbl.lower().strip() in MALIGNANT_KEYWORDS else 0
        if len(set(label_map.values())) == 1:
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
    else:
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}

    y = np.array([label_map[lbl] for lbl in y_raw], dtype=np.int64)
    return X, y, label_map


def analisar_tabular(X: np.ndarray, y: np.ndarray | None, schema: dict) -> dict:
    """Estatísticas descritivas completas das features tabulares brutas (31 métricas científicas)."""
    import scipy.stats as stats

    if X.size == 0:
        return {"n_amostras": 0, "n_features": 0, "features": {}}

    n_amostras = int(X.shape[0])
    n_features = int(X.shape[1])
    feature_names = schema["feature_names"]

    stats_res: dict = {
        "n_amostras": n_amostras,
        "n_features": n_features,
        "features": {},
        "label_name": schema.get("label_name"),
    }

    # Correlação multivariada global (Pearson, Spearman, Covariância) com tratamento de NaN
    if n_features > 1:
        pearson_mat = np.zeros((n_features, n_features))
        spearman_mat = np.zeros((n_features, n_features))
        
        for r in range(n_features):
            for c in range(n_features):
                if r == c:
                    pearson_mat[r, c] = 1.0
                    spearman_mat[r, c] = 1.0
                else:
                    col_r = X[:, r]
                    col_c = X[:, c]
                    valid_mask = ~np.isnan(col_r) & ~np.isnan(col_c)
                    if valid_mask.sum() >= 3:
                        p_coef, _ = stats.pearsonr(col_r[valid_mask], col_c[valid_mask])
                        pearson_mat[r, c] = float(p_coef) if not np.isnan(p_coef) else 0.0
                        
                        s_coef, _ = stats.spearmanr(col_r[valid_mask], col_c[valid_mask])
                        spearman_mat[r, c] = float(s_coef) if not np.isnan(s_coef) else 0.0
                    else:
                        pearson_mat[r, c] = 0.0
                        spearman_mat[r, c] = 0.0
        
        try:
            # np.ma.cov lida com arrays mascarados (ignorando NaNs)
            cov_mat = np.ma.cov(np.ma.masked_invalid(X), rowvar=False)
            cov_list = cov_mat.tolist() if isinstance(cov_mat, np.ndarray) else [[float(cov_mat)]]
        except Exception:
            cov_list = []
            
        stats_res["correlacoes"] = {
            "pearson": pearson_mat.tolist(),
            "spearman": spearman_mat.tolist(),
            "covariancia": cov_list
        }
    else:
        stats_res["correlacoes"] = {
            "pearson": [[1.0]],
            "spearman": [[1.0]],
            "covariancia": [[float(np.nanvar(X))]] if n_amostras > 1 else [[0.0]]
        }

    for i, name in enumerate(feature_names):
        col = X[:, i]
        valid_col = col[~np.isnan(col)]
        n_valid = len(valid_col)
        n_missing = int(np.isnan(col).sum())
        
        if n_valid == 0:
            continue
            
        # 1. Medidas de Tendência Central
        media = float(np.nanmean(col))
        mediana = float(np.nanmedian(col))
        try:
            mode_res = stats.mode(valid_col, keepdims=True)
            moda = float(mode_res.mode[0]) if len(mode_res.mode) > 0 else float(valid_col[0])
        except Exception:
            moda = media
            
        # 2. Medidas de Dispersão
        v_min = float(np.nanmin(col))
        v_max = float(np.nanmax(col))
        amplitude = v_max - v_min
        variancia = float(np.nanvar(col)) if n_valid > 1 else 0.0
        desvio = float(np.nanstd(col)) if n_valid > 1 else 0.0
        cv = desvio / (media + 1e-8)
        
        # 3. Medidas de Posição (Quartis, Percentis, Decis)
        q25, q50, q75 = np.nanpercentile(col, [25, 50, 75])
        p10, p90 = np.nanpercentile(col, [10, 90])
        decis = np.nanpercentile(col, range(10, 100, 10)).tolist()
        iqr_val = q75 - q25
        
        # 4. Medidas de Forma
        try:
            skew_val = float(stats.skew(col, nan_policy='omit'))
            kurt_val = float(stats.kurtosis(col, nan_policy='omit'))
        except Exception:
            skew_val = 0.0
            kurt_val = 0.0
            
        # 5. Outliers (Z-score e IQR)
        lim_inf = q25 - 1.5 * iqr_val
        lim_sup = q75 + 1.5 * iqr_val
        outliers_iqr = int(((valid_col < lim_inf) | (valid_col > lim_sup)).sum())
        
        if desvio > 0:
            z_scores = (valid_col - media) / desvio
            outliers_z = int((np.abs(z_scores) > 2.5).sum())
        else:
            outliers_z = 0
            
        # 6. Métricas de Distribuição (SNR, Entropia, Densidade)
        snr = media / (desvio + 1e-8)
        hist_counts, _ = np.histogram(valid_col, bins="auto")
        entropia = float(stats.entropy(hist_counts + 1e-8))
        
        n_bins = len(hist_counts)
        max_entropy = np.log(n_bins) if n_bins > 1 else 1.0
        uniformidade = float(1.0 - (entropia / max_entropy)) if max_entropy > 0 else 1.0
        
        # 7. Estatísticas de Normalidade
        try:
            shapiro_stat, shapiro_p = stats.shapiro(valid_col) if n_valid >= 3 else (0.0, 1.0)
            ks_stat, ks_p = stats.kstest(valid_col, 'norm', args=(media, desvio + 1e-8))
        except Exception:
            shapiro_stat, shapiro_p = 0.0, 1.0
            ks_stat, ks_p = 0.0, 1.0
            
        # 8. Estatísticas por Grupo & Testes de Hipótese (se houver rótulo)
        stats_por_grupo = {}
        testes_hipotese = {}
        if y is not None:
            unique_groups = sorted(set(y.tolist()))
            groups_data = []
            for g in unique_groups:
                g_data = col[y == g]
                g_valid = g_data[~np.isnan(g_data)]
                groups_data.append(g_valid)
                stats_por_grupo[int(g)] = {
                    "media": float(np.nanmean(g_data)) if len(g_valid) > 0 else 0.0,
                    "desvio": float(np.nanstd(g_data)) if len(g_valid) > 0 else 0.0,
                }
            
            # Executa testes estatísticos baseados no número de grupos
            try:
                if len(unique_groups) == 2 and len(groups_data[0]) >= 3 and len(groups_data[1]) >= 3:
                    # Teste t de Student para 2 grupos
                    t_stat, t_p = stats.ttest_ind(groups_data[0], groups_data[1], equal_var=False)
                    testes_hipotese["teste_t"] = {
                        "estatistica": float(t_stat) if not np.isnan(t_stat) else 0.0,
                        "p_valor": float(t_p) if not np.isnan(t_p) else 1.0,
                        "diferenca_significativa": bool(t_p < 0.05) if not np.isnan(t_p) else False
                    }
                elif len(unique_groups) > 2 and all(len(g) >= 3 for g in groups_data):
                    # ANOVA para múltiplos grupos
                    f_stat, f_p = stats.f_oneway(*groups_data)
                    testes_hipotese["anova"] = {
                        "estatistica": float(f_stat) if not np.isnan(f_stat) else 0.0,
                        "p_valor": float(f_p) if not np.isnan(f_p) else 1.0,
                        "diferenca_significativa": bool(f_p < 0.05) if not np.isnan(f_p) else False
                    }
            except Exception:
                pass

        # Compilação final de métricas univariadas da feature
        stats_res["features"][name] = {
            # Tendência Central
            "media": round(media, 4),
            "mediana": round(mediana, 4),
            "moda": round(moda, 4),
            
            # Dispersão
            "min": round(v_min, 4),
            "max": round(v_max, 4),
            "amplitude": round(amplitude, 4),
            "variancia": round(variancia, 4),
            "desvio": round(desvio, 4),
            "coeficiente_variacao": round(cv, 4),
            "iqr": round(iqr_val, 4),
            
            # Posição
            "quartis": [round(float(q25), 4), round(float(q50), 4), round(float(q75), 4)],
            "percentis": {
                "P10": round(float(p10), 4),
                "P25": round(float(q25), 4),
                "P50": round(float(q50), 4),
                "P75": round(float(q75), 4),
                "P90": round(float(p90), 4)
            },
            "decis": [round(float(d), 4) for d in decis],
            
            # Forma
            "assimetria": round(skew_val, 4),
            "curtose": round(kurt_val, 4),
            
            # Outliers
            "outliers_iqr": outliers_iqr,
            "outliers_zscore": outliers_z,
            "boxplot_limites": {
                "lim_inf": round(float(lim_inf), 4),
                "lim_sup": round(float(lim_sup), 4)
            },
            
            # Métricas de Distribuição
            "snr": round(snr, 4),
            "entropia": round(entropia, 4),
            "uniformidade": round(uniformidade, 4),
            
            # Normalidade
            "normalidade": {
                "shapiro": {
                    "estatistica": round(float(shapiro_stat), 4),
                    "p_valor": round(float(shapiro_p), 4),
                    "eh_normal": bool(shapiro_p > 0.05)
                },
                "ks": {
                    "estatistica": round(float(ks_stat), 4),
                    "p_valor": round(float(ks_p), 4),
                    "eh_normal": bool(ks_p > 0.05)
                }
            },
            
            # Estatísticas de Grupo & Testes de Hipótese
            "por_grupo": stats_por_grupo,
            "testes_hipotese": testes_hipotese,
            "n_nulos": n_missing
        }

    if y is not None:
        unique, counts = np.unique(y, return_counts=True)
        stats_res["distribuicao_labels"] = {int(u): int(c) for u, c in zip(unique, counts)}

    return stats_res


def decidir_estrategia_tabular(analise: dict) -> dict:
    """Propõe e justifica estratégias de pré-processamento baseando-se na análise do dado bruto."""
    n_total = analise.get("n_amostras", 0)
    
    # Mapeia se há valores nulos globais e outliers
    tem_nulos = False
    outliers_globais = 0
    shapiro_normais = 0
    n_features = len(analise.get("features", {}))
    
    for fname, fstats in analise.get("features", {}).items():
        if fstats.get("n_nulos", 0) > 0:
            tem_nulos = True
        outliers_globais += fstats.get("outliers_iqr", 0)
        if fstats.get("normalidade", {}).get("shapiro", {}).get("eh_normal"):
            shapiro_normais += 1

    estrategia = {
        "imputacao": "none",
        "escalamento": "standard",
        "justificativas": [],
    }

    if tem_nulos:
        estrategia["imputacao"] = "median"
        estrategia["justificativas"].append(
            "Células vazias ou com caracteres de nulo detectadas na base -> Aplicada Imputação por Mediana."
        )
    else:
        estrategia["justificativas"].append(
            "Nenhum valor nulo ou vazio detectado na base de dados -> Imputação desnecessária."
        )

    # Se mais de 10% da base total acumulada for de outliers ou se poucas colunas forem normais
    pct_outliers = (outliers_globais / (n_total * n_features + 1e-8))
    tamanho_insuficiente = (shapiro_normais / (n_features + 1e-8)) < 0.5
    
    if pct_outliers > 0.08 or tamanho_insuficiente:
        estrategia["escalamento"] = "robust"
        estrategia["justificativas"].append(
            f"Taxa acumulada de outliers ({pct_outliers:.1%}) ou comportamento não-gaussiano dominante -> Aplicado RobustScaler para mitigar distorções."
        )
    else:
        estrategia["escalamento"] = "standard"
        estrategia["justificativas"].append(
            f"Distribuição de features regular com baixa taxa de outliers ({pct_outliers:.1%}) -> Aplicado StandardScaler padrão."
        )

    return estrategia


def preprocessar_tabular_amostras(X: np.ndarray, estrategia: dict) -> np.ndarray:
    """Aplica a imputação de nulos decidida na estratégia.

    ATENÇÃO — uso apenas para estatística/exibição (EDA), NUNCA para alimentar
    o treino de um classificador: como ajusta o imputer no array inteiro, usar
    a saída desta função como entrada de treino/teste causa vazamento de dados
    (a mediana/média de imputação "vê" as amostras de teste). Para treinar um
    modelo, use `imputar_treino_teste()`, que ajusta o imputer apenas no fold
    de treino.
    """
    from sklearn.impute import SimpleImputer

    X_preproc = X.copy()
    if X_preproc.size > 0:
        if estrategia.get("imputacao") == "median" and np.isnan(X_preproc).any():
            imputer = SimpleImputer(strategy="median")
            X_preproc = imputer.fit_transform(X_preproc)
        elif np.isnan(X_preproc).any():
            # Fallback de segurança se houver nulos perdidos mas imputacao == "none"
            imputer = SimpleImputer(strategy="mean")
            X_preproc = imputer.fit_transform(X_preproc)

    return X_preproc


def imputar_treino_teste(X_train: np.ndarray, X_test: np.ndarray,
                          metodo: str = "auto") -> tuple[np.ndarray, np.ndarray, dict]:
    """Imputa valores ausentes SEM vazamento de dados: o imputer é ajustado
    (fit) apenas no fold de treino, e as mesmas estatísticas (mediana/média do
    treino) são aplicadas (transform) tanto no treino quanto no teste.

    `metodo`: "median", "mean", "none" (força não imputar) ou "auto" (usa
    mediana se houver nulos, robusta a outliers — comportamento padrão seguro
    para chamadores que não decidiram uma estratégia explícita, ex.: o modo
    multimodal, que hoje não fazia nenhuma imputação e podia quebrar com NaN).
    """
    from sklearn.impute import SimpleImputer

    info = {"metodo": metodo, "aplicado": False}
    tem_nulos_treino = X_train.size > 0 and np.isnan(X_train).any()
    tem_nulos_teste = X_test.size > 0 and np.isnan(X_test).any()
    if metodo == "none" or (not tem_nulos_treino and not tem_nulos_teste):
        return X_train, X_test, info

    estrategia_sklearn = "median" if metodo in ("median", "auto") else "mean"
    imputer = SimpleImputer(strategy=estrategia_sklearn)
    # Ajusta SOMENTE no treino — estatísticas de teste nunca entram aqui.
    X_train_imp = imputer.fit_transform(X_train) if X_train.size > 0 else X_train
    X_test_imp = imputer.transform(X_test) if X_test.size > 0 else X_test
    info["aplicado"] = True
    info["estrategia_sklearn"] = estrategia_sklearn
    info["ajustado_apenas_no_treino"] = True
    return X_train_imp, X_test_imp, info


def selecionar_features_shap(X_train: np.ndarray, y_train: np.ndarray,
                              feature_names: list[str] | None = None,
                              n_features: int | None = None,
                              max_amostras_shap: int = 200) -> tuple[np.ndarray, dict]:
    """
    Seleção de features guiada por importância SHAP (BioStatusIA).

    Substitui a seleção RFE/PCA anterior (`selecionar_features` em
    avaliacao_modelos.py), que existia no código mas nunca era chamada em
    lugar nenhum do pipeline — a Seção III-B do artigo alegava "SHAP-ranked
    feature selection" sem que isso estivesse de fato implementado.

    Como SHAP precisa de um modelo já treinado, o procedimento é:
    1) Treina um modelo "sonda" rápido (RandomForest) usando SOMENTE o fold
       de treino recebido — nunca visita o conjunto de teste, então não há
       vazamento (o índice das features escolhidas é depois aplicado
       igualmente a treino e teste pelo chamador).
    2) Calcula SHAP (TreeExplainer) numa subamostra do treino (até
       `max_amostras_shap`, por custo computacional).
    3) Rankeia as features por |SHAP| médio e mantém as `n_features` mais
       importantes (mesmo critério de tamanho que a seleção antiga:
       max(2, min(n_features_originais, n_amostras // 3, 20))).

    Retorna (indices_selecionados, info). Falha de forma segura — devolve
    TODOS os índices originais, sem alterar nada — se o `shap` não estiver
    instalado, se já houver poucas features, ou se qualquer etapa levantar
    exceção (ex.: classe única no fold, amostras insuficientes).
    """
    n_orig = X_train.shape[1]
    info: dict = {"aplicado": False, "n_original": n_orig, "metodo": "shap_randomforest"}

    n_alvo = n_features or max(2, min(n_orig, X_train.shape[0] // 3, 20))
    if n_orig <= n_alvo:
        info["motivo"] = "dimensionalidade já baixa"
        return np.arange(n_orig), info
    if len(set(y_train.tolist())) < 2:
        info["motivo"] = "apenas uma classe no fold de treino"
        return np.arange(n_orig), info

    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier

        sonda = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )
        sonda.fit(X_train, y_train)

        n_sample = min(max_amostras_shap, len(X_train))
        rng = np.random.RandomState(42)
        idx_amostra = rng.choice(len(X_train), size=n_sample, replace=False)
        X_amostra = X_train[idx_amostra]

        explainer = shap.TreeExplainer(sonda)
        valores_shap = explainer.shap_values(X_amostra)
        n_classes_sonda = len(sonda.classes_)
        if isinstance(valores_shap, list):
            if n_classes_sonda == 2:
                valores_shap = np.asarray(valores_shap[1])  # binário: comportamento original
            else:
                # Multi-classe: média da magnitude entre TODAS as classes — não
                # existe uma única "classe positiva" para basear a seleção.
                valores_shap = np.mean([np.abs(np.asarray(v)) for v in valores_shap], axis=0)
        else:
            valores_shap = np.asarray(valores_shap)
            if valores_shap.ndim == 3:
                if n_classes_sonda == 2:
                    valores_shap = valores_shap[:, :, 1]  # binário: comportamento original
                else:
                    valores_shap = np.abs(valores_shap).mean(axis=-1)

        importancia_media = np.abs(valores_shap).mean(axis=0)
        if importancia_media.shape[0] != n_orig:
            info["motivo"] = (
                f"SHAP devolveu {importancia_media.shape[0]} valores para "
                f"{n_orig} features — incompatibilidade, mantendo todas."
            )
            return np.arange(n_orig), info

        ordem = np.argsort(importancia_media)[::-1]
        selecionadas = np.sort(ordem[:n_alvo])

        nomes = feature_names if (feature_names and len(feature_names) == n_orig) \
            else [f"f{i}" for i in range(n_orig)]

        info["aplicado"] = True
        info["n_selecionadas"] = int(n_alvo)
        info["indices_selecionados"] = selecionadas.tolist()
        info["ranking"] = sorted(
            [{"feature": nomes[i], "importancia_shap": round(float(importancia_media[i]), 5)}
             for i in range(n_orig)],
            key=lambda d: d["importancia_shap"], reverse=True,
        )
        info["features_descartadas"] = [
            nomes[i] for i in range(n_orig) if i not in set(selecionadas.tolist())
        ]
        return selecionadas, info
    except Exception as e:
        info["erro"] = str(e)
        return np.arange(n_orig), info
