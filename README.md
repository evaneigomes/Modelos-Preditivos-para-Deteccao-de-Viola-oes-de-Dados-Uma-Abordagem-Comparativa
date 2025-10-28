# MVP2 — Ciência de Dados

## Comparação entre Deep Learning e Machine Learning na Previsão de Violações de Dados (2010–2023)

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

## 2. Dataset

**Fonte:** Privacy Rights Clearinghouse — *Data Breach Chronology*
**Período:** 2010–2023 (anterior a 2010 desconsiderado por baixa consistência)
**Periodicidade:** **Mensal (ME)** por tipo de organização

**Atributos principais:** `Date Breach` (data do incidente), colunas por setor como `BSF` (Serviços Financeiros), `BSO` (Outros Negócios), `BSR` (Varejo), `EDU` (Educação), `GOV` (Governo), `MED` (Saúde), `NGO` (ONGs), `UNKN` (Desconhecido) e `Total Geral`.

---

## 3. Metodologia

**Arquitetura analítica** (alto nível):

1. **EDA** e diagnóstico de séries;
2. **Preparação** (limpeza, padronização de datas, filtro temporal 2010–2023, **reamostragem mensal**, remoção de **outliers por IQR**, cálculo do **expoente de Hurst** para checar persistência/aleatoriedade);
3. **Divisão treino/teste** com **janela temporal fixa** (hold-out nos últimos meses), opcionalmente com **validação cruzada temporal** quando cabível;
4. **Modelagem**: Prophet, SARIMA, XGBoost, LSTM, TCN;
5. **Otimização** por *grid search* e ajustes finos por família;
6. **Avaliação** com MAPE (principal), MAE e RMSE;
7. **Comparação** e **análise crítica** por setor;
8. **Conclusão** e próximos passos.

**Boas práticas de reprodutibilidade:** notebook público **Colab**; dataset carregado **via URL**; células de texto explicando decisões; gráficos no próprio notebook.

---

## 4. Preparação dos Dados

* **Ajuste de datas**: padronização; descarte de datas incompletas; `YYYY-MM` → dia 1.
* **Filtro temporal**: 2010-01 a 2023-12.
* **Agregação**: reamostragem **mensal (ME)** por setor.
* **Outliers**: remoção por **IQR** (Q1−1,5×IQR; Q3+1,5×IQR).
* **Expoente de Hurst**: diagnóstico de persistência/reversão/aleatoriedade.

> *Observação:* As transformações foram aplicadas **após** consolidação por setor, mantendo consistência entre modelos.

---

## 5. Modelos e Treinamento

**Modelos avaliados:**

* **Prophet** (aditivo; sazonalidade; *changepoints*);
* **SARIMA** (componentes sazonais);
* **XGBoost Regressor** (lags, janelas deslizantes e *boosting* gradiente);
* **LSTM** (memória de longo prazo, *look_back*);
* **TCN** (convoluções causais dilatadas).

**Esboço de hiperparâmetros** (ajustados por setor):

* **SARIMA**: varredura em (p, d, q) × (P, D, Q, s=12).
* **Prophet**: sazonalidade anual/mensal; *changepoints* e Fourier.
* **XGBoost**: `n_estimators`, `max_depth`, `learning_rate`, *lags* e *window size*.
* **LSTM/TCN**: `look_back` (ex.: 6/12/18), `epochs`, `batch_size`; *early stopping* opcional.

**Treino/Teste:** divisão temporal com **teste nos últimos 24 meses** (padrão), mantendo séries alinhadas entre modelos. Ajustes finos por setor quando necessário.

---

## 6. Avaliação e Métricas

* **MAPE (%)** — métrica principal (interpretação proporcional entre setores).
* **MAE** — erro absoluto (escala original).
* **RMSE** — penaliza erros grandes.

**Classificação do MAPE (Lewis, 1982):**
<10: *Altamente preciso*; 10–19,99: *Boa previsão*; 20–49,99: *Razoável*; ≥50: *Imprecisa*.

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

### 7.2 Insight prático

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

## 10. Reprodutibilidade

* **Notebook Colab (público):** *[inserir link do seu notebook vfinal]*
* **Carregamento do dataset**: via **URL** no próprio notebook (sem configuração externa).
* **Saídas**: CSVs por modelo (ex.: `resultados_prophet.csv`, `resultados_sarima.csv`, `resultados_xgb.csv`, `resultados_lstm.csv`, `resultados_tcn.csv`), imagens (heatmap e comparações).

---

## 11. Referências

* Privacy Rights Clearinghouse — *Data Breach Chronology*.
* Literatura de séries temporais e cibersegurança (detalhes no artigo base).
* Materiais de apoio da disciplina (requisitos de entrega e checklist).

---

> **Observação final ao avaliador:** Este documento é o “relatório” em formato Markdown. As células de texto do *notebook Colab* repetem as seções explicativas, seguidas das células de código que reproduzem cada etapa (preparo, treino, avaliação e gráficos). O *link Colab* e as figuras/CSVs finais serão apontados ao final da execução completa do notebook vfinal.
