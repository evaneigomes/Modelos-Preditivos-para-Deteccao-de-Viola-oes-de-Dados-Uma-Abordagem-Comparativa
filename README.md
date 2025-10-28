# MVP2 — Privacidade e Segurança

## Comparação entre Deep Learning e Machine Learning na Previsão de Violações de Dados

**Discente:** Evanei Gomes dos Santos

**Docente:** Prof. Dr. André Luiz Marques Serrano

**Curso:** PPEE/UnB — Engenharia Elétrica

**Entrega:** Notebook público no Google Colab (link ao final)

---

## Resumo

Este trabalho compara modelos de **Deep Learning** (LSTM, TCN) e de **Machine Learning**/estatísticos (Prophet, SARIMA, XGBoost) na previsão mensal de **violações de dados** por setor organizacional, usando a base **Privacy Rights Clearinghouse – Data Breach Chronology** (2010–2023). A avaliação usa **MAPE (%)** como métrica principal (complementada por **MAE** e **RMSE**). Em síntese: modelos de **redes neurais** tendem a apresentar **melhor acurácia** em setores com **padrões temporais mais complexos**, enquanto **XGBoost** mostra competitividade em séries mais **agregadas** e com **padrões suaves**. As conclusões e números setoriais detalhados constam nas seções de **Resultados** e **Conclusão**.

---

## 1. Definição do Problema

Violações de dados geram impactos financeiros e reputacionais. O objetivo é **prever a quantidade mensal** de violações por **tipo de organização** (ex.: saúde, governo, financeiro), para apoiar decisões de prevenção e resposta.

**Hipóteses:**

1. As séries mensais por setor contêm sinal suficiente para **prever picos** de incidentes.
2. **DL (LSTM/TCN)** terá vantagem em séries **não lineares** e com **interações temporais** de longo alcance, enquanto **XGBoost/Prophet/SARIMA** podem ser mais competitivos em **setores agregados** ou com sazonalidade mais clara.

**Escopo:** Comparativo entre famílias de modelos, priorizando **acurácia preditiva (MAPE)** e **consistência**.

---


### Carregamento via URL (Google Sheets/CSV)

```python
import pandas as pd
from urllib.parse import urlparse

# Cole aqui a URL pública do Google Sheets (ou um CSV remoto)
URL = "https://docs.google.com/spreadsheets/d/SEU_ID/export?format=xlsx"

# Caso a URL seja do tipo .../edit?usp=sharing, troque por .../export?format=xlsx
if "edit" in URL and "export" not in URL:
    URL = URL.split("/edit")[0] + "/export?format=xlsx"

# Leitura
try:
    df_raw = pd.read_excel(URL)
except Exception:
    # Fallback: CSV
    df_raw = pd.read_csv(URL)

# Visualização inicial
print(df_raw.head())
```

### Seleção de colunas e padronização de nomes

```python
# Esperado: uma coluna de data e colunas de setores (BSF, BSO, BSR, EDU, GOV, MED, NGO, UNKN, Total Geral)
# Ajuste o nome da coluna de data conforme o seu arquivo (ex.: 'Date Breach')
DATE_COL = 'Date Breach'
setores = ['BSF','BSO','BSR','EDU','GOV','MED','NGO','UNKN','Total Geral']

# Garantir tipos corretos
import numpy as np
import pandas as pd

df = df_raw.copy()
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)

# Forçar inteiros não-negativos nos setores
for c in setores:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).clip(lower=0).astype(int)
```

---

## 3. Metodologia

**Arquitetura analítica** (alto nível):

1. **Preparação** (limpeza, padronização de datas, filtro temporal 2010–2023, **reamostragem mensal**;
2. **Remoção de outliers por IQR**, cálculo do **expoente de Hurst** para checar persistência/aleatoriedade);
3. **Divisão treino/teste** com **janela temporal fixa** (hold-out nos últimos meses), opcionalmente com **validação cruzada temporal** quando cabível;
4. **Modelagem**: Prophet, SARIMA, XGBoost, LSTM, TCN;
5. **Otimização** por *grid search* e ajustes finos por família;
6. **Avaliação** com MAPE (principal), MAE e RMSE;
7. **Comparação** e **análise crítica** por setor;
8. **Conclusão** e próximos passos.

**Boas práticas de reprodutibilidade:** notebook público **Colab**; dataset carregado **via URL**; células de texto explicando decisões; gráficos no próprio notebook.

---

### Filtro temporal, agregação mensal e limpeza básica

```python
# Recorte 2010–2023 e índice temporal
inicio, fim = pd.Timestamp('2010-01-01'), pd.Timestamp('2023-12-31')
mask = (df[DATE_COL] >= inicio) & (df[DATE_COL] <= fim)
df = df.loc[mask].copy()

# Agregação mensal (soma de incidentes por mês)
df['month'] = df[DATE_COL].dt.to_period('M').dt.to_timestamp()
df_mensal = (df.groupby('month')[setores]
               .sum()
               .asfreq('MS')
               .fillna(0)
               .astype(int))

df_mensal.head()
```

### Tratamento de outliers por IQR (winsorização por setor)

```python
def winsorize_iqr(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    return s.clip(lower=max(0, low), upper=max(high, 0))

for c in setores:
    df_mensal[c] = winsorize_iqr(df_mensal[c])
```

### Expoente de Hurst (diagnóstico)

```python
import numpy as np

def hurst_exponent(ts):
    # Implementação simples de R/S (aproximação)
    ts = np.asarray(ts, dtype=float)
    N = len(ts)
    if N < 20:
        return np.nan
    lags = np.floor(np.logspace(1, np.log10(N/2), num=20)).astype(int)
    tau = []
    for lag in lags:
        diffs = ts[lag:] - ts[:-lag]
        tau.append(np.sqrt(np.std(diffs)))
    # Ajuste log-log
    x = np.log(lags)
    y = np.log(tau)
    H = np.polyfit(x, y, 1)[0]
    return max(min(H, 1.0), 0.0)

for c in setores:
    H = hurst_exponent(df_mensal[c].values)
    print(f"Hurst[{c}]: {H:.2f}")
```

### Split temporal (treino vs. teste)

```python
TEST_SIZE = 24  # meses

split_idx = len(df_mensal) - TEST_SIZE
train_idx = df_mensal.index[:split_idx]
test_idx  = df_mensal.index[split_idx:]

print(train_idx[0], '→', train_idx[-1], '| test:', test_idx[0], '→', test_idx[-1])
```

---


> Abaixo, trechos compactos por família de modelos. Nos notebooks finais, essas funções são chamadas em *loops* por setor, com *grid/tuning* quando aplicável.

### Métricas e utilitários

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

RESULTS = []  # coleciona dicionários de resultados
```

### Prophet

```python
from prophet import Prophet

def run_prophet(serie, test_size=24):
    y = serie.reset_index().rename(columns={'month':'ds', serie.name:'y'})
    train, test = y.iloc[:-test_size], y.iloc[-test_size:]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(train)
    future = m.make_future_dataframe(periods=test_size, freq='MS')
    fcst = m.predict(future).tail(test_size)
    y_true = test['y'].values
    y_pred = fcst['yhat'].values
    return {
        'Setor': serie.name,
        'Modelo': 'Prophet',
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE (%)': mape(y_true, y_pred)
    }
```

### SARIMA (statsmodels)

```python
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_sarima(serie, test_size=24, p=1,d=0,q=1,P=1,D=1,Q=1,s=12):
    y_train = serie.iloc[:-test_size]
    y_test  = serie.iloc[-test_size:]
    model = SARIMAX(y_train, order=(p,d,q), seasonal_order=(P,D,Q,s), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=test_size).predicted_mean
    y_true = y_test.values
    y_pred = pred.values
    return {
        'Setor': serie.name,
        'Modelo': 'SARIMA',
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE (%)': mape(y_true, y_pred)
    }
```

### XGBoost (lags)

```python
from xgboost import XGBRegressor

def make_lagged_df(serie, lags=12):
    df = pd.DataFrame({'y': serie.values}, index=serie.index)
    for L in range(1, lags+1):
        df[f'lag_{L}'] = df['y'].shift(L)
    return df.dropna()

def run_xgb(serie, test_size=24, lags=12):
    df_l = make_lagged_df(serie, lags)
    X, y = df_l.drop(columns=['y']).values, df_l['y'].values
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    model = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'Setor': serie.name,
        'Modelo': 'XGBoost',
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAPE (%)': mape(y_test, y_pred)
    }
```

### LSTM (Keras)

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def make_supervised(arr, look_back=12):
    X, y = [], []
    for i in range(len(arr) - look_back):
        X.append(arr[i:i+look_back])
        y.append(arr[i+look_back])
    X, y = np.array(X), np.array(y)
    return X[..., np.newaxis], y  # (samples, timesteps, features=1)

def run_lstm(serie, test_size=24, look_back=12, epochs=200, batch_size=32):
    values = serie.values.astype(float)
    scaler = MinMaxScaler()
    vals = scaler.fit_transform(values.reshape(-1,1)).ravel()

    X, y = make_supervised(vals, look_back)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    model = Sequential([
        LSTM(64, input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')
    model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)

    y_pred = model.predict(X_test).ravel()
    # inversão para escala original
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1,1)).ravel()

    return {
        'Setor': serie.name,
        'Modelo': 'LSTM',
        'MAE': mean_absolute_error(y_test_inv, y_pred_inv),
        'RMSE': np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)),
        'MAPE (%)': mape(y_test_inv, y_pred_inv)
    }
```

### TCN (Temporal Convolutional Network)

```python
from tcn import TCN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Reutiliza make_supervised do LSTM (mesmo formato)

def run_tcn(serie, test_size=24, look_back=12, epochs=200, batch_size=32):
    values = serie.values.astype(float)
    scaler = MinMaxScaler()
    vals = scaler.fit_transform(values.reshape(-1,1)).ravel()

    X, y = make_supervised(vals, look_back)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    model = Sequential([
        TCN(nb_filters=64, kernel_size=3, dilations=[1,2,4,8], dropout_rate=0.1, return_sequences=False, input_shape=(look_back,1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')
    model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)

    y_pred = model.predict(X_test).ravel()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1,1)).ravel()

    return {
        'Setor': serie.name,
        'Modelo': 'TCN',
        'MAE': mean_absolute_error(y_test_inv, y_pred_inv),
        'RMSE': np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)),
        'MAPE (%)': mape(y_test_inv, y_pred_inv)
    }
```

### Loop por setor e consolidação de resultados

```python
RESULTS = []
for c in setores:
    serie = df_mensal[c]
    try:
        RESULTS.append(run_prophet(serie, test_size=TEST_SIZE))
    except Exception as e:
        print(f"Prophet falhou em {c}:", e)
    try:
        RESULTS.append(run_sarima(serie, test_size=TEST_SIZE))
    except Exception as e:
        print(f"SARIMA falhou em {c}:", e)
    try:
        RESULTS.append(run_xgb(serie, test_size=TEST_SIZE))
    except Exception as e:
        print(f"XGB falhou em {c}:", e)
    try:
        RESULTS.append(run_lstm(serie, test_size=TEST_SIZE))
    except Exception as e:
        print(f"LSTM falhou em {c}:", e)
    try:
        RESULTS.append(run_tcn(serie, test_size=TEST_SIZE))
    except Exception as e:
        print(f"TCN falhou em {c}:", e)

import pandas as pd

df_results = pd.DataFrame(RESULTS)
print(df_results.head())
```

---


### Tabelas, heatmap e melhores por setor

```python
# Tabela geral
summary = (df_results
           .groupby(['Setor','Modelo'])[['MAE','RMSE','MAPE (%)']]
           .mean()
           .reset_index())

# Melhor por setor com base no MAPE
melhores = summary.loc[summary.groupby('Setor')['MAPE (%)'].idxmin()].reset_index(drop=True)
print("
🏆 Melhor MAPE por setor:
", melhores)

# Heatmap simples (matriz Setor x Modelo com MAPE)
pivot_mape = summary.pivot(index='Setor', columns='Modelo', values='MAPE (%)')
print("
Matriz MAPE (Setor x Modelo):
", pivot_mape.round(2))
```

### Salvamento padronizado dos CSVs (sem timestamp)

```python
# Salva resultados no /content com nomes estáveis
summary.to_csv('/content/resultados_comparados.csv', index=False)
melhores.to_csv('/content/melhor_modelo_por_setor.csv', index=False)
pivot_mape.to_csv('/content/heatmap_mape.csv')

print("
💾 Arquivos salvos em /content:")
print("- resultados_comparados.csv")
print("- melhor_modelo_por_setor.csv")
print("- heatmap_mape.csv")
```
---

## 7. Resultados e Discussão

### 7.1 Síntese por Setor (evidências dos artefatos)

* **Total Geral:** **XGBoost** com MAPE ≈ **5,97%** (*alta precisão*). Redes **TCN/LSTM** também com bom desempenho (≈ **10–12%**), porém acima do XGBoost nesta série agregada.
* **UNKN (Desconhecido):** **XGBoost** ≈ **10,03%** (*boa/alta precisão*); **LSTM** ≈ **11,95%** próximo do patamar de boa precisão.
* **MED (Saúde):** **TCN** ≈ **23,52%** (*razoável*), **Prophet** ≈ **26,54%** (razoável).
* **BSF (Financeiro):** **LSTM** ≈ **21,14%** (limite entre *razoável* e *boa* dependendo de arredondamento/intervalo).
* **BSO (Outros Negócios):** **TCN** ≈ **19,39%** (*boa previsão*).
* **EDU (Educação):** **ARIMA** ≈ **36,74%** (*razoável*).
* **BSR (Varejo):** **MAPE ≥ 70%** em todas as famílias (previsão **imprecisa**; alta volatilidade).

> *Leitura crítica:* Em séries **agregadas** (Total Geral) e com **magnitude maior**, o **XGBoost** se destacou; já em setores com **padrões temporais complexos**, **LSTM/TCN** tendem a **superar** Prophet/SARIMA e, em alguns casos, competir com XGBoost. Diferenças de **split**, **tratamento de outliers** e **escolhas de hiperparâmetros** explicam variações pontuais.

### 7.2 Ações práticas

* **Planejamento tático:** adotar **XGBoost** para séries **macro**/agregadas; recorrer a **LSTM/TCN** para **setores específicos** com dinâmica mais **não linear**.
* **Setores críticos:** **BSR (Varejo)** exige estratégias alternativas (ex.: modelos hierárquicos, *external regressors*, *ensembles* especializados e *regime switching*).

### 7.3 Visualizações previstas no notebook

* **Heatmap de MAPE** (modelos × setores).
* **Boxplots** de distribuição de erros por modelo.
* **Curvas reais vs. previstas** para setores representativos.

---

## 8. Conclusão

1. **Não há “um modelo vencedor” universal**: a eficácia depende do **setor** e da **estrutura** da série.
2. **XGBoost** foi **muito competitivo** em séries **agregadas** (Total Geral, UNKN).
3. **LSTM/TCN** mostraram **vantagem** em setores **complexos** (MED, BSO; e casos como BSF/UNKN em diferentes preparações).
4. **Prophet/SARIMA** mantiveram-se como **baselines interpretáveis**, úteis para diagnóstico e *benchmarking*.
5. Para **produção**, sugere-se um **comitê de modelos** (meta-ensemble) com seleção por setor, além de **monitoramento de drift** e **retreinamento** periódico.

**Limitações:** disponibilidade e qualidade de rótulos por setor; possíveis mudanças de política de notificação ao longo dos anos; sensibilidade a *outliers*; e variações por *split* temporal.
**Próximos passos:** *ensembles* DL+ML, variáveis exógenas (ex.: eventos regulatórios), validação com 2024–2025, e avaliação de **intervalos de previsão**.

---

## 9. Checklist (MVP2)

**Definição do problema**: descrita nas seções 1–2.
**Hipóteses**: claras na Seção 1.
**Restrições/seleção de dados**: período 2010–2023; datas válidas; reamostragem mensal.
**Descrição do dataset**: atributos e setores (Seção 2).
**Split treino/teste**: hold-out temporal (últimos meses).
**Cross-validation**: quando aplicável, **TimeSeriesSplit**; caso contrário, justificar hold-out.
**Transformações**: padronização de datas, reamostragem mensal, IQR, Hurst.
**Feature selection**: para XGBoost, seleção implícita por importância; para DL, janela (*look_back*) e engenharia de *lags*.
**Modelagem**: justificativas por família (Seção 5).
**Tuning**: *grid search* por modelo/setor.
**Métricas**: MAPE/MAE/RMSE (Seção 6) + interpretação do MAPE.
**Resultados**: síntese e gráficos previstos (Seção 7).
**Melhor solução**: seleção por **setor**; sugerido **comitê** por contexto.

---

### Organização dos artefatos de saída

```python
# Organização final dos artefatos gerados no notebook
# (ajuste conforme desejar)
ARQUIVOS = {
    'comparacao_geral': '/content/resultados_comparados.csv',
    'melhores_por_setor': '/content/melhor_modelo_por_setor.csv',
    'heatmap_mape': '/content/heatmap_mape.csv'
}

for k, v in ARQUIVOS.items():
    print(f"{k:>20}: {v}")
```
