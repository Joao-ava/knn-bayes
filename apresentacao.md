# Comparação de Algoritmos de Aprendizado de Máquina para Classificação e Regressão

**Disciplina:** Inteligência Artificial Computacional  
**Universidade:** Universidade de Fortaleza (UNIFOR)  
**Professora:** Dra. Cynthia Moreira Maia  
**Período:** 2026.1 - AV2

---

## Autores

- **Taís Moreira Rodrigues** (23204071) - Coordenação, KNN, Naive Bayes Univariado, Introdução
- **João Alex Vieira de Almeida** (2320484) - Algoritmos de Regressão, Métricas, Banco de Dados
- **Jean de Souza Morais** (2323225) - Validação Cruzada, Análise de Resultados, Conclusões

---

## Sumário

1. [Introdução](#1-introdução)
2. [Algoritmos de Aprendizado de Máquina](#2-algoritmos-de-aprendizado-de-máquina)
3. [Experimentos](#3-experimentos)
   - [3.1 Banco de Dados](#31-banco-de-dados)
   - [3.2 Métricas de Avaliação](#32-métricas-de-avaliação)
   - [3.3 Resultados](#33-resultados)
4. [Conclusões](#4-conclusões)
5. [Referências](#5-referências)

---

## 1. Introdução

### 1.1 Contexto

Este trabalho apresenta a implementação e avaliação de algoritmos clássicos de aprendizado de máquina, tanto para problemas de **classificação** quanto para **regressão**. O foco principal é comparar o desempenho de diferentes abordagens, analisando trade-offs entre acurácia, tempo de processamento e generalização dos modelos.

Os algoritmos estudados representam duas filosofias diferentes de aprendizado:

- **KNN (K-Nearest Neighbors)**: Abordagem baseada em instâncias, não paramétrica
- **Naive Bayes**: Abordagem probabilística baseada no Teorema de Bayes

### 1.2 Motivação e Objetivos

A escolha desses algoritmos justifica-se por:

1. **Representatividade**: São algoritmos fundamentais em cursos de Machine Learning
2. **Praticidade**: Implementação relativamente direta em linguagens de programação
3. **Interpretabilidade**: Oferecem resultados e comportamentos facilmente explicáveis
4. **Versatilidade**: Podem ser aplicados tanto a problemas de classificação quanto regressão

**Objetivo Geral:** Implementar e comparar algoritmos de aprendizado supervisionado (KNN e Naive Bayes) em problemas de classificação e regressão, avaliando seu desempenho através de métricas apropriadas e validação cruzada.

**Objetivos Específicos:**
- Implementar KNN com distâncias euclidiana e Manhattan
- Implementar Naive Bayes univariado e multivariado
- Aplicar validação cruzada k-fold em todos os modelos
- Comparar desempenhos através de métricas apropriadas

### 1.3 Estrutura do Projeto

O projeto foi estruturado em módulos independentes para garantir modularidade e reutilização de código:

- `knn.py`: Implementação do algoritmo KNN
- `naive_bayes.py`: Implementação dos classificadores Naive Bayes
- `regression.py`: Implementação de regressão linear múltipla
- `metrics.py`: Implementação de métricas de avaliação
- `kfold.py`: Validação cruzada k-fold
- `data_preprocessing.py`: Pré-processamento e normalização de dados
- `train.py`: Pipeline de treinamento e validação dos modelos
- `data.py`: Carregamento dos datasets

### 1.4 Metodologia

#### 1.4.1 Validação Cruzada K-Fold

A validação cruzada k-fold é uma técnica robusta para avaliar o desempenho de modelos de machine learning. O processo segue os seguintes passos:

1. Dividir o dataset em k partições (folds) de tamanho aproximadamente igual
2. Para cada iteração i de 1 a k:
   - Usar a partição i como conjunto de teste
   - Usar as demais k-1 partições como conjunto de treinamento
   - Treinar o modelo e avaliar as métricas
3. Calcular média e desvio padrão das métricas em todas as iterações

**Vantagens:**
- Utiliza todos os dados tanto para treinamento quanto teste
- Reduz variância nos estimadores de desempenho
- Independente de splits aleatórios

No trabalho, utilizamos **k=5**, comum em problemas de medium-sized datasets.

#### 1.4.2 Pré-processamento de Dados

O pipeline de pré-processamento envolve as seguintes etapas:

1. **Tratamento de valores faltantes**: Preenchimento com a mediana da coluna
2. **Normalização Z-score**: Transformação para média 0 e desvio padrão 1
   $$z = \frac{x - \mu}{\sigma}$$
3. **Seleção por correlação** (apenas para regressão): Seleção das top-1000 features mais correlacionadas com a target
4. **Embaralhamento**: Randomização da ordem dos dados para evitar viés

#### 1.4.3 Pipeline de Treinamento

```
Dataset → Pré-processamento → K-Fold Split
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
              Para cada fold                  Modelo i
                    ↓                               ↓
              Separar Train/Test → Treinar → Predizer → Calcular Métricas
                                        ↓
                            Agregar média e desvio padrão
                                        ↓
                            Salvar resultados (JSON)
```

---

## 2. Algoritmos de Aprendizado de Máquina

### 2.1 K-Nearest Neighbors (KNN)

#### 2.1.1 Conceito

O KNN é um algoritmo de aprendizado baseado em instâncias que implementa o princípio: *"a classe de um ponto é determinada pela classe da maioria de seus vizinhos mais próximos"*.

#### 2.1.2 Características

- **Tipo**: Não paramétrico, baseado em instâncias
- **Aprendizado**: Lazy learning (armazena dados, não aprende explicitamente)
- **Hiperparâmetro principal**: k (número de vizinhos)
- **Métrica de distância**: Euclidiana ou Manhattan

#### 2.1.3 Funcionamento

**Fase de Treinamento:**
```python
fit(X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train
```

O treinamento é trivial: apenas armazena os dados de treinamento.

**Fase de Predição:**

Para cada ponto de teste $x_{test}$:

1. Calcular distância para todos os pontos de treinamento
2. Ordenar por distância e selecionar os k primeiros (vizinhos mais próximos)
3. **Classificação**: Retorna a classe mais frequente entre os k vizinhos
4. **Regressão**: Retorna a média dos valores dos k vizinhos

#### 2.1.4 Métricas de Distância

**Distância Euclidiana:**
$$d(x_1, x_2) = \sqrt{\sum_{i=1}^{n} (x_{1i} - x_{2i})^2}$$

**Distância Manhattan (L1):**
$$d(x_1, x_2) = \sum_{i=1}^{n} |x_{1i} - x_{2i}|$$

#### 2.1.5 Implementação

```python
class KNN:
    def __init__(self, k=5, metric='euclidean', task='classification'):
        self.k = k
        self.metric = metric
        self.task = task
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def euclidiana_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def calculate_prediction(self, x):
        if self.metric == 'euclidean':
            distances = [self.euclidiana_distance(x, x_train) 
                        for x_train in self.X_train]
        else:  # manhattan
            distances = [self.manhattan_distance(x, x_train) 
                        for x_train in self.X_train]
        
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        if self.task == "classification":
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            return unique[np.argmax(counts)]
        else:  # regression
            return np.mean(k_nearest_labels)
    
    def predict(self, X_test):
        return np.array([self.calculate_prediction(x) for x in X_test])
```

#### 2.1.6 Vantagens e Desvantagens

| Vantagens | Desvantagens |
|-----------|-------------|
| Simples de implementar | Lento em predição (O(n*m)) |
| Não faz suposições sobre dados | Sensível a escala de features |
| Funciona para regressão e classificação | Requer muito armazenamento |
| | Curse of dimensionality |

### 2.2 Naive Bayes

#### 2.2.1 Conceito

Naive Bayes é um classificador probabilístico baseado no Teorema de Bayes que assume **independência condicional** entre as features dado a classe.

#### 2.2.2 Teorema de Bayes

$$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$$

Onde:
- $P(C|X)$: Probabilidade posterior (classe dado as features)
- $P(X|C)$: Verossimilhança
- $P(C)$: Probabilidade a priori
- $P(X)$: Evidência

#### 2.2.3 Suposição de Independência

Pela suposição de independência condicional:

$$P(X_1, X_2, ..., X_n | C) = \prod_{i=1}^{n} P(X_i | C)$$

Logo:
$$P(C|X) \propto P(C) \prod_{i=1}^{n} P(X_i | C)$$

Em escala logarítmica (para evitar underflow):
$$\log P(C|X) = \log P(C) + \sum_{i=1}^{n} \log P(X_i | C)$$

#### 2.2.4 Variantes Implementadas

##### 2.2.4.1 Naive Bayes Univariado

Assume que cada feature segue uma distribuição normal independente.

**Função de Densidade de Probabilidade (PDF) Normal:**
$$P(X_i|C) = \frac{1}{\sqrt{2\pi\sigma_c^2}} \exp\left(-\frac{(X_i - \mu_c)^2}{2\sigma_c^2}\right)$$

Onde:
- $\mu_c$: Média da feature na classe c
- $\sigma_c^2$: Variância da feature na classe c

**Treinamento:**
```python
def fit(self, x_train, y_train):
    self._classes = np.unique(y_train)
    nclasses = len(self._classes)
    nfeatures = x_train.shape[1]
    
    self.mean = np.zeros((nclasses, nfeatures))
    self.var = np.zeros((nclasses, nfeatures))
    self.prior = np.zeros(nclasses)
    
    for idx, c in enumerate(self._classes):
        X_c = x_train[y_train == c]
        self.mean[idx, :] = X_c.mean(axis=0)
        self.var[idx, :] = X_c.var(axis=0)
        self.prior[idx] = np.mean(y_train == c)
```

**Predição:**
```python
def _predict(self, x):
    posteriors = []
    for idx, c in enumerate(self._classes):
        prior = np.log(self.prior[idx])
        posterior = np.sum(np.log(self._pdf(idx, x)))
        posteriors.append(prior + posterior)
    return self._classes[np.argmax(posteriors)]
```

##### 2.2.4.2 Naive Bayes Multivariado

Modela a distribuição conjunta das features como uma distribuição normal multivariada.

**PDF Multivariada:**
$$P(X|C) = \frac{1}{\sqrt{(2\pi)^d |\Sigma_c|}} \exp\left(-\frac{1}{2}(X - \mu_c)^T \Sigma_c^{-1} (X - \mu_c)\right)$$

Onde:
- $d$: Número de dimensões (features)
- $\Sigma_c$: Matriz de covariância da classe c
- $|\Sigma_c|$: Determinante da matriz de covariância

**Implementação em escala logarítmica:**
```python
def _log_pdf(self, class_idx, X):
    d = len(self.means[class_idx])
    det_cov = np.linalg.det(self.cov[class_idx])
    inv_cov = np.linalg.inv(self.cov[class_idx])
    
    log_norm_const = -0.5 * (d * np.log(2 * np.pi) + np.log(det_cov))
    diff = X - self.means[class_idx]
    log_exponent = -0.5 * (diff.T @ inv_cov @ diff)
    
    return log_norm_const + log_exponent
```

#### 2.2.5 Vantagens e Desvantagens

| Vantagens | Desvantagens |
|-----------|-------------|
| Rápido e eficiente | Suposição de independência raramente é verdadeira |
| Funciona bem com dados pequenos | Sensível a features irrelevantes |
| Probabilístico e interpretável | Requer cálculo de matriz inversa (multivariado) |
| Treina rapidamente | |

### 2.3 Regressão Linear Múltipla

#### 2.3.1 Conceito

A Regressão Linear Múltipla (RLM) modela a relação linear entre uma variável dependente (target) e múltiplas variáveis independentes (features).

#### 2.3.2 Formulação Matemática

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p + \epsilon$$

Em forma matricial:
$$\mathbf{y} = \mathbf{X}\mathbf{\beta} + \boldsymbol{\epsilon}$$

Onde:
- $\mathbf{X}$: Matriz de features (n × p+1)
- $\mathbf{\beta}$: Vetor de coeficientes
- $\boldsymbol{\epsilon}$: Vetor de erros

#### 2.3.3 Estimação de Coeficientes

Utilizando o método dos Mínimos Quadrados Ordinários (OLS):

$$\hat{\mathbf{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

As predições são calculadas como:
$$\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{\beta}}$$

---

## 3. Experimentos

### 3.1 Banco de Dados

#### 3.1.1 Dataset de Classificação: Fraud Detection

##### Descrição

Dataset de **detecção de fraude** em transações financeiras. Objetivo: classificar se uma transação é legítima (0) ou fraudulenta (1).

##### Características

- **Tamanho**: Aproximadamente 284,807 transações
- **Features**: 30 features numéricas (PCA-transformadas e tempo)
- **Target**: Binária (Fraude: 1, Legítima: 0)
- **Desbalanceamento**: Altamente desbalanceado (~0.17% de fraudes)
- **Formato**: Parquet (arquivo comprimido)

##### Features Disponíveis

As features incluem componentes principais (PCA) de dados de transação:
- `V1` a `V28`: Componentes principais
- `Time`: Tempo decorrido da primeira transação
- `Amount`: Valor da transação
- `Class`: Label (0 ou 1)

##### Desafios

1. **Desbalanceamento de classe**: Poucas amostras positivas
2. **Alta dimensionalidade**: 30 features para análise
3. **Distribuição não-linear**: Possível relação complexa entre features

#### 3.1.2 Dataset de Regressão: Santander Transaction

##### Descrição

Dataset de **satisfação de clientes** do banco Santander. Objetivo: prever um score de satisfação contínuo.

##### Características

- **Tamanho**: 76,020 observações
- **Features**: 371 features numéricas
- **Target**: Contínua (score de satisfação)
- **Formato**: Parquet

##### Tratamento

Para este dataset:
- Selecionamos as **top 1,000 features** mais correlacionadas com a target
- Objetivo: Reduzir dimensionalidade e ruído
- Aplicação: Seleção por correlação de Pearson

---

### 3.2 Métricas de Avaliação

#### 3.2.1 Métricas de Classificação

##### 3.2.1.1 Acurácia

Proporção de predições corretas sobre o total de amostras.

$$\text{Acurácia} = \frac{TP + TN}{TP + TN + FP + FN}$$

Onde:
- TP (True Positive): Corretamente classificado como positivo
- TN (True Negative): Corretamente classificado como negativo
- FP (False Positive): Incorretamente classificado como positivo
- FN (False Negative): Incorretamente classificado como negativo

**Implementação:**
```python
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
```

**Interpretação:** Métrica simples, mas pode ser enganosa em dados desbalanceados.

##### 3.2.1.2 Precisão

Proporção de predições positivas corretas sobre todas as predições positivas.

$$\text{Precisão} = \frac{TP}{TP + FP}$$

**Interpretação:** "De todas as predições positivas, quantas estavam corretas?"

**Implementação:**
```python
def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0
```

##### 3.2.1.3 Recall (Sensibilidade)

Proporção de amostras positivas corretamente identificadas.

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Interpretação:** "De todas as amostras positivas, quantas foram identificadas?"

**Implementação:**
```python
def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0
```

##### 3.2.1.4 F1-Score

Média harmônica entre Precisão e Recall. Balanceia ambas as métricas.

$$\text{F1-Score} = 2 \times \frac{\text{Precisão} \times \text{Recall}}{\text{Precisão} + \text{Recall}}$$

**Implementação:**
```python
def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0
```

**Vantagem:** Métrica equilibrada, preferível em dados desbalanceados.

#### 3.2.2 Métricas de Regressão

##### 3.2.2.1 R² Score (Coeficiente de Determinação)

Proporção da variância na variável dependente que é explicada pelas variáveis independentes.

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$

Onde:
- $SS_{res}$: Soma dos quadrados residuais
- $SS_{tot}$: Soma total dos quadrados

**Range:** $R^2 \in (-\infty, 1]$
- $R^2 = 1$: Predição perfeita
- $R^2 = 0$: Modelo não melhor que a média
- $R^2 < 0$: Modelo pior que a média

**Implementação:**
```python
def r2_score(y, y_pred):
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
```

##### 3.2.2.2 R² Score Ajustado

Versão ajustada do R² que penaliza o número de features, prevenindo overfitting.

$$R^2_{adj} = 1 - (1 - R^2) \times \frac{n - 1}{n - p - 1}$$

Onde:
- $n$: Número de amostras
- $p$: Número de features

**Vantagem:** Melhor para comparar modelos com diferentes números de features.

**Implementação:**
```python
def r2_score_adjusted(y, y_pred, p=1000):
    n = len(y)
    r2 = r2_score(y, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

---

## 4. Conclusões

### 4.1 Análise Comparativa

#### 4.1.1 KNN: Euclidiano vs Manhattan

##### Para Classificação

| Aspecto | Euclidiano | Manhattan |
|---------|-----------|-----------|
| Acurácia | 89.08% | 89.00% |
| F1-Score | 0.1203 | 0.1311 |
| Tempo Predição | 8.28s | 7.63s |

**Conclusão:** Manhattan é ligeiramente mais rápido e tem F1-Score melhor.

##### Para Regressão

| Aspecto | Euclidiano | Manhattan |
|---------|-----------|-----------|
| R² | -0.0128 | 0.0556 |
| Tempo Predição | 13.66s | 12.12s |

**Conclusão:** Manhattan é superior em ambos os aspectos.

#### 4.1.2 Naive Bayes: Univariado vs Multivariado

##### Comparação

| Métrica | Univariado | Multivariado | Diferença |
|---------|-----------|-------------|-----------|
| Acurácia | 81.81% | 82.17% | +0.36% |
| Precisão | 0.2321 | 0.2316 | -0.05% |
| Recall | 0.3474 | 0.3383 | -0.91% |
| F1-Score | 0.2757 | 0.2707 | -0.50% |
| Tempo Fit | 0.488 ms | 0.838 ms | +71.7% |
| Tempo Predição | 0.017 s | 0.054 s | +2.97x |

**Conclusão:** 
- Univariado é mais eficiente em tempo
- Multivariado oferece marginal melhoria em acurácia
- Tradeoff: complexidade vs desempenho

#### 4.1.3 KNN vs Naive Bayes (Classificação)

| Aspecto | KNN | Naive Bayes |
|---------|-----|------------|
| Acurácia | 89.04% | 82.00% |
| F1-Score | 0.1257 | 0.2732 |
| Tempo Predição | 7.96s | 0.036s |
| Interpretabilidade | Média | Alta |
| Requisitos Memória | Alto | Baixo |

**Análise Crítica:**
- KNN tem acurácia alta, mas F1-Score baixo
- NB é melhor para detectar positivos (fraudes)
- NB é ~220x mais rápido
- Para produção: NB é mais adequado

#### 4.1.4 Regressão: KNN vs Linear

| Aspecto | KNN | Linear |
|---------|-----|--------|
| R² | 0.0278 | -23.08 |
| Tempo Predição | 12.87s | 0.001s |
| Complexidade | O(n*m) | O(m) |
| Generalização | Média | Pior |

**Conclusão:** KNN é superior, mas ambos apresentam resultados ruins no dataset.

### 4.2 Discussão

#### 4.2.1 Desempenho de Classificação

##### Acurácia Enganosa

O dataset de detecção de fraude é altamente desbalanceado (~0.17% fraudes). Nesse contexto:

- **Acurácia alta (89%) não é meaningful**
- Um modelo ingênuo que sempre prediz "não-fraude" teria 99.83% de acurácia
- **F1-Score é métrica mais apropriada**: KNN (0.125) vs NB (0.273)

##### Por que Naive Bayes Supera KNN?

1. **KNN é sensível a desbalanceamento**:
   - Vizinhos mais próximos provavelmente são negativos
   - Predições tendenciosas para classe majoritária

2. **NB é probabilístico**:
   - Aprende distribuição de cada classe independentemente
   - Melhor recall (consegue identificar fraudes)

3. **Complexidade do espaço**:
   - Há regiões de fraude dispersas em espaço alto-dimensional
   - NB generaliza melhor essa distribuição

##### Implicações Práticas

Para um sistema de detecção de fraude:

- **Custo de FP (alarme falso)**: Baixo - cliente apenas verifica transação
- **Custo de FN (fraude não detectada)**: Alto - prejuízo financeiro
- **Recomendação**: Usar NB com threshold ajustado para maximizar recall

#### 4.2.2 Desempenho de Regressão

##### Por que R² é Tão Baixo?

Os resultados ruins (R² ≈ 0 ou negativo) indicam:

1. **Dataset complexo**:
   - 1000 features selecionadas ainda insuficientes
   - Possíveis relações não-lineares
   - Muito ruído

2. **Características do problema**:
   - Satisfação de cliente é multifatorial
   - Features podem não ser causais
   - Possível falta de features importantes

3. **Limitações dos modelos**:
   - KNN: Curse of dimensionality em 1000 dims
   - Linear: Assume linearidade que não existe

##### Possíveis Melhorias

1. **Feature Engineering**:
   - Criar features interativas
   - Aplicar transformações não-lineares
   - Seleção mais sofisticada que correlação

2. **Modelos Alternativos**:
   - Random Forest, Gradient Boosting (não-lineares)
   - Redes neurais (parametrizadas melhor)
   - SVM com kernel não-linear

3. **Hiperparâmetros**:
   - Tunar k para KNN (testamos k=25)
   - Regularização para Linear

##### Conclusão Regressão

- Dataset regressão é **significativamente mais desafiador** que classificação
- KNN ligeiramente melhor que Linear
- Ambos inadequados para o problema

#### 4.2.3 Trade-offs Observados

##### Velocidade vs Acurácia

```
Naive Bayes:  Rápido (17ms) ────┐ Melhor F1
              Acurácia 82%        │ 220x mais rápido
              
KNN:          Lento (8283ms) ───┘ Acurácia 89%
              Mas F1-Score pior   (mas enganoso)
```

##### Complexidade vs Desempenho

```
NB Univariado:   Simples ────┐ Desempenho ligeiramente pior
               Rápido (17ms)  │
               
NB Multivariado: Complexo ───┘ Marginal melhoria (0.36%)
               Lento (54ms)    Overhead: 3x mais tempo
```

#### 4.2.4 Lições Aprendidas

1. **Acurácia não é tudo**: Em dados desbalanceados, F1-Score é mais significativo

2. **Diferentes problemas, diferentes soluções**:
   - Classificação: Naive Bayes é "campeão"
   - Regressão: Ambos modelos são inadequados

3. **Importância de métricas apropriadas**:
   - Escolher métricas segundo o problema
   - Validação cruzada é essencial

4. **Scalability e implementação**:
   - KNN não scale bem (O(n*m))
   - NB é production-ready
   - Linear Regression é baseline rápido

### 4.3 Resumo dos Achados

#### Classificação (Fraud Detection)

✅ **Naive Bayes é vencedor:**
- F1-Score superior (0.273 vs 0.125 do KNN)
- 220x mais rápido
- Melhor recall (prioridade: detectar fraudes)
- Recomendado para produção

✅ **Manhattan melhor que Euclidiana para KNN:**
- Ligeiramente mais rápido
- Melhor F1-Score
- Recomendação: usar Manhattan em práticas futuras

❌ **KNN não adequado para o problema:**
- Apesar de acurácia alta, falha em detectar fraudes
- Computacionalmente caro
- Sensível ao desbalanceamento

#### Regressão (Santander Transaction)

⚠️ **Todos os modelos apresentam desempenho inadequado:**
- KNN Manhattan: R² = 0.0556 (melhor, mas ruim)
- KNN Euclidiano: R² = -0.0128
- Linear: R² = -23.08

📊 **Implicações:**
- Dataset altamente complexo
- Relações não-lineares provável
- Modelos lineares/simples são insuficientes

### 4.4 Recomendações

#### Curto Prazo

1. **Sistema de Detecção de Fraude**:
   - Implementar Naive Bayes Multivariado em produção
   - Ajustar threshold para maximizar recall
   - Monitorar taxa de falsos positivos

2. **Pipeline de Regressão**:
   - Explorar feature engineering mais sofisticado
   - Testar modelos não-lineares (Random Forest, Gradient Boosting)
   - Aumentar volume de features relevantes

#### Longo Prazo

1. **Pesquisa**:
   - Aplicar ensemble methods (votação de múltiplos modelos)
   - Investigar deep learning para ambos problemas
   - Estudar explicabilidade (SHAP, LIME)

2. **Produção**:
   - Implementar A/B testing para novos modelos
   - Monitorar drift de dados
   - Retreinamento contínuo

### 4.5 Contribuições Académicas

Este trabalho demonstrou:

1. ✅ Implementação correta de KNN e Naive Bayes do zero
2. ✅ Importância da validação cruzada k-fold
3. ✅ Seleção apropriada de métricas para cada contexto
4. ✅ Trade-offs entre velocidade, acurácia e interpretabilidade
5. ✅ Limitações de modelos simples em dados complexos

### 4.6 Trabalhos Futuros

1. **Extensões dos Modelos**:
   - Weight-KNN: vizinhos com peso inverso à distância
   - Gaussian Mixture Models como alternativa a NB
   - Kernel Methods (SVM)

2. **Melhorias no Pipeline**:
   - Hyperparameter tuning automático (Grid Search, Bayesian Optimization)
   - Ensemble methods (Voting, Stacking)
   - Cross-validation estratificada para desbalanceamento

3. **Novos Problemas**:
   - Aplicar metodologia em datasets de domínio real
   - Integração com ferramentas de ML (scikit-learn, TensorFlow)
   - Análise de interpretabilidade dos modelos

---

## 5. Referências

### Livros

1. **Murphy, K. P.** (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
   - Referência completa para ML probabilístico

2. **James, G., Witten, D., Hastie, T., & Tibshirani, R.** (2013). An Introduction to Statistical Learning. Springer.
   - Practical guide para algoritmos de ML

3. **Bishop, C. M.** (2006). Pattern Recognition and Machine Learning. Springer.
   - Fundação teórica de Pattern Recognition

### Artigos

4. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). The Elements of Statistical Learning (2nd ed.). Springer.
   - Comprehensive reference para supervised learning

5. **Fix, E., & Hodges, J. L.** (1951). Discriminatory Analysis. Nonparametric Discrimination: Consistency Properties.
   - Artigo original do KNN

6. **Rish, I.** (2001). An empirical study of the naive Bayes classifier. In IJCAI workshop on empirical methods in AI.
   - Análise empírica de Naive Bayes

### Recursos Online

7. **Scikit-learn Documentation**: https://scikit-learn.org/
   - Implementação de referência de ML algorithms

8. **Andrew Ng - Machine Learning Specialization** (Coursera)
   - Cursos sobre ML principles and applications

### Datasets

9. **Fraud Detection Dataset**: Kaggle Credit Card Fraud Detection
   - https://www.kaggle.com/mlg-ulb/creditcardfraud

10. **Santander Customer Transaction Prediction**: Kaggle
    - https://www.kaggle.com/c/santander-customer-transaction-prediction

### Padrões de Implementação

11. **PEP 8**: Style Guide for Python Code
    - Guia de estilo usado no desenvolvimento

12. **Clean Code**: Robert C. Martin
    - Principles aplicados na estrutura do código

---

## Apêndice A: Estrutura do Código

```
projeto/
├── app/
│   ├── main.py                  # Script simples de demonstração
│   ├── train.py                 # Pipeline de treinamento e validação
│   ├── data.py                  # Carregamento de datasets
│   ├── data_preprocessing.py    # Pré-processamento de dados
│   ├── knn.py                   # Implementação KNN
│   ├── naive_bayes.py           # Implementação Naive Bayes
│   ├── regression.py            # Implementação Regressão Linear
│   ├── metrics.py               # Métricas de avaliação
│   ├── kfold.py                 # K-Fold validation
│   └── metrics_knn.py           # Métricas específicas KNN
├── data/
│   ├── fraud_detection.pq       # Dataset classificação
│   └── santander_transaction.pq # Dataset regressão
├── figures/
│   └── regression_plotter.py    # Visualizações
├── classification_table.json    # Resultados classificação
├── regression_table.json        # Resultados regressão
└── apresentacao.md              # Este documento

```

---

**Versão:** 1.0  
**Data:** Maio de 2026  
**Última Atualização:** 03/05/2026  

---

## Declaração de Autoria

Nós, Taís Moreira Rodrigues, João Alex Vieira de Almeida e Jean de Souza Morais, declaramos que este trabalho é original e foi desenvolvido em conformidade com as normas de integridade acadêmica da Universidade de Fortaleza.

Realizamos o treinamento de modelos Bayes, KNN e Regressão Linear Múltipla nos conjuntos de dados utilizando K Folds para pegar as médias das métricas.

OBRIGADO!

Boas férias 😎🎅🥳


