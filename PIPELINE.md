
Claro! Aqui está o arquivo `README.md` traduzido para o português:

---

# MH-AutoML

## Visão Geral do Pipeline

![Fluxo da MH-AutoML](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/fluxo-MH-AutoML.png)

## Etapas

### Informações dos Dados

Nesta etapa, informações básicas sobre o conjunto de dados são extraídas e resumidas. Isso inclui:
- Informação do sistema
- Número de linhas e colunas
- Tipos de dados de cada coluna
- Informação de balanceamento de dados
- Análise de valores duplicados e ausentes
- Tipos de características (Permissões, APICalls)

Exemplo de saída da ferramenta na etapa de Exploração de dados.

### System Information

| Operating System Version | Total RAM Memory Usage (GB) | Available RAM Memory (GB) | Used RAM Memory (GB) |
|--------------------------|-----------------------------|---------------------------|----------------------|
| Windows-10-10.0.22631-SP0| 31.73                   | 17.59                 | 14.14            |

### Data Information

| Rows | Columns |
|------|---------|
| 15036| 56      |

### Data Type

| Data Type | Count |
|-----------|-------|
| float64   | 51    |

### Data Balancing

| Label | Percentage |
|-------|------------|
| 0   | 63.01%     |
| 1   | 36.99%     |

### Data Small

| Number of duplicate data | Number of null values |
|--------------------------|-----------------------|
| 10                        | 56                    |

### Features Info

| Permissions found | API_Calls found |
|-------------------|-----------------|
| 50                | 5               |

---

### Pré-processamento

O pré-processamento envolve a limpeza e transformação dos dados para torná-los adequados para o treinamento do modelo. Tarefas comuns nesta etapa incluem:
- Limpeza de dados
- Codificação de variáveis categóricas
Como artefato a ferramenta gera um mapa de calor das "sujeiras" do dataset onde, as colunas amarelas indicam colunas sem relevância, ou seja são compostas de valores nulos ou somente com valores =0.
![enter image description here](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/missing_values_heatmap.png)



### Engenharia de Características

#### 1. Gráfico de Componentes Principais (PCA)
![Gráfico PCA](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/PCA.png)

**Descrição:**
O gráfico de componentes principais (PCA) é uma técnica de redução de dimensionalidade que transforma um conjunto de observações de variáveis possivelmente correlacionadas em um conjunto de valores de variáveis linearmente não correlacionadas chamadas componentes principais. Este gráfico ajuda a visualizar a variação dos dados em um espaço de menor dimensão, geralmente em duas ou três dimensões, facilitando a identificação de padrões e a interpretação dos dados.

**Utilização:**
- Redução de dimensionalidade
- Visualização de clusters ou agrupamentos nos dados
- Identificação de variáveis mais relevantes

#### 2. Gráficos de Importância das Características (LASSO e ANOVA)
![Importância das Características](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/significance.png)

**Descrição:**
Os gráficos de importância das características gerados pelos métodos LASSO e ANOVA mostram a relevância de cada característica para a previsão do modelo. LASSO (Least Absolute Shrinkage and Selection Operator) realiza seleção e regularização de características para melhorar a precisão do modelo e a interpretabilidade. ANOVA (Analysis of Variance) é usada para comparar as médias e variâncias entre grupos.

**Utilização:**
- Identificação das características mais influentes
- Redução de dimensionalidade eliminando características irrelevantes
- Melhoria da interpretabilidade do modelo

#### 3. Gráfico de Distribuição de Treino e Teste
![Distribuição de Treino e Teste](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/train_test_distribution.png)

**Descrição:**
Este gráfico mostra a distribuição dos dados de treino e teste para cada classe, seja malware ou benigno. Ele ajuda a visualizar se as amostras de treino e teste estão bem balanceadas em termos de quantidade e distribuição das classes.

**Utilização:**
- Verificação do balanceamento das classes entre treino e teste
- Identificação de possíveis desbalanceamentos que podem afetar o desempenho do modelo
- Garantia de que o modelo está sendo treinado e testado de forma justa

### Seleção e otimização modelos
A seleção do modelo é feita pelo método de votação denominado Ensemble Vote Classifier que seleciona o melhor modelo a partir de uma métrica pré-definida como Recall no caso da MH-AutoML. Para otimização é utilizado a biblioteca Optuna, uma estrutura de otimização de hiper-parâmetros de código aberto. Os modelos utilizados pela ferramenta são:
- RandomForestClassifier
- DecisionTreeClassifier
- ExtraTreesClassifier
- KNN
- LightGBM
- CatBoost

Artefatos gerados nessa etapa são:
- Dados brutos CSVs dos estudos completo feito pelo optuna para cada modelo 
-  Ranking dos 5 melhores modelos
- Gráfico de histórico de otimização de hiper-parâmetros
![enter image description here](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/histori_hiperparametros.png)
- Gráfico de hiper-parâmetros mais importantes
![enter image description here](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/hiperparametros.png)


### Avaliação

O modelo selecionado é avaliado no conjunto de teste para avaliar seu desempenho. Esta etapa envolve:
- Cálculo de métricas de avaliação (por exemplo, precisão, precisão, recall, F1 score, MCC, matriz de confusão, curva ROC). Além disso é gerado o desempenho para cada etapa do pipeline com relação a Tempo de execução e consumo de memória RAM.
![enter image description here](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/performance_metrics.jpg)



### XAI (AI Explicável)

#### 1. Análise de Importância das Características com LIME
![Importância das Características](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/lime_feature_importance.jpg)

**Descrição:**
O gráfico gerado pela LIME (Local Interpretable Model-agnostic Explanations) mostra a importância das características para uma amostra específica. LIME cria um modelo interpretable localmente ao redor da previsão de uma amostra, permitindo entender quais características influenciaram mais essa previsão.

**Utilização:**
- Explicação das previsões do modelo em nível local
- Identificação de características mais importantes para amostras específicas
- Aumento da confiança e interpretabilidade do modelo

#### 2. Valores SHAP (SHapley Additive exPlanations) - Summary Plot
![Valores SHAP](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/shap_summary_plot.png)

**Descrição:**
O gráfico de valores SHAP (SHapley Additive exPlanations) summary plot resume o impacto de cada característica em todas as previsões. Ele mostra a distribuição dos valores SHAP para cada característica, indicando quanto cada característica contribui para a previsão final do modelo.

**Utilização:**
- Explicação global do modelo
- Identificação de características com maior impacto nas previsões
- Visualização da distribuição do impacto das características

#### 3. Explicações LIME (Local Interpretable Model-agnostic Explanations)
![Explicações LIME](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/Lime.png)

**Descrição:**
O gráfico de explicações LIME mostra como as características influenciam a previsão para uma amostra específica. LIME ajusta um modelo simples e interpretable ao redor da previsão de uma amostra, destacando as características que mais contribuíram para a previsão.

**Utilização:**
- Explicação detalhada de previsões individuais
- Aumento da interpretabilidade em casos específicos
- Identificação de padrões locais nas previsões do modelo

---

### 9. MLcicle

A etapa final é empacotar o modelo treinado e seus artefatos para implantação. Cada execução gera um experimento que pode ser comparado com os dados de outras execuções, alem de possibilitar o versinamento dos modelos gerados e a comparação dos mesmos por diversar méticas e parametros.
- Comparação entre execuções
![enter image description here](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/comp_experimentos.png)
- Comparação de modelos
![enter image description here](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/comp_modelos.png)
## Artefatos

Ao longo do pipeline, vários artefatos são gerados e armazenados, como descrito acima. Esses artefatos estão organizados em diretórios de acordo com sua etapa respectivamente.
![enter image description here](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/artefatos_dir.png) 


Esses artefatos são essenciais para garantir a reprodutibilidade e rastreabilidade de todo o processo de machine learning.
