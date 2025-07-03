# MH-AutoML: Malware Hunter AutoML

  __  __ _   _         _         _        __  __ _     
 |  \/  | | | |       / \  _   _| |_ ___ |  \/  | |    
 | |\/| | |_| |_____ / _ \| | | | __/ _ \| |\/| | |    
 | |  | |  _  |_____/ ___ \ |_| | || (_) | |  | | |___ 
 |_|  |_|_| |_|    /_/   \_\__,_|\__\___/|_|  |_|_____|
                                                                                                                                                        
          <<<<<<< Malware Hunter AutoML>>>>>>>

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura MVC](#arquitetura-mvc)
- [Instalação](#instalação)
- [Uso](#uso)
- [Pipeline de Machine Learning](#pipeline-de-machine-learning)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Configurações](#configurações)
- [Resultados e Artefatos](#resultados-e-artefatos)
- [Exemplos de Uso](#exemplos-de-uso)
- [Troubleshooting](#troubleshooting)
- [Contribuição](#contribuição)
- [Licença](#licença)

## 🎯 Visão Geral

O **MH-AutoML (Malware Hunter AutoML)** é um sistema automatizado de Machine Learning projetado especificamente para detecção de malware Android. O sistema implementa um pipeline completo de ML que integra análise de dados, pré-processamento, engenharia de features, otimização de modelos e interpretabilidade.

### Características Principais

- 🔄 **Pipeline Automatizado**: Processo completo de ML sem intervenção manual
- 🏗️ **Arquitetura MVC**: Separação clara de responsabilidades
- 📊 **Análise Exploratória**: Informações detalhadas sobre os dados
- 🧹 **Pré-processamento Robusto**: Limpeza e transformação automática
- ⚙️ **Otimização Automática**: Seleção de hiperparâmetros com Optuna
- 🎯 **Múltiplos Algoritmos**: LightGBM, CatBoost, Random Forest, etc.
- 📈 **Interpretabilidade**: SHAP e LIME para explicação de modelos
- 📋 **Relatórios Automáticos**: Documentação completa dos resultados
- 🌐 **Interface Web**: MLflow para acompanhamento de experimentos

## 🏗️ Arquitetura MVC

O sistema segue o padrão arquitetural **Model-View-Controller (MVC)**:

### Controller (`controller/core.py`)
- **Responsabilidade**: Orquestração do pipeline completo
- **Funções**:
  - Coordenação entre módulos
  - Gerenciamento de fluxo de dados
  - Controle de logging e MLflow
  - Execução sequencial das etapas

### Model (`model/`)
- **Responsabilidade**: Lógica de negócio e processamento de dados
- **Módulos**:
  - `preprocessing/`: Análise, limpeza e transformação de dados
  - `feature_engineering/`: Seleção e redução de features
  - `optimization/`: Otimização de hiperparâmetros
  - `interpretability/`: Interpretação de modelos
  - `tools/`: Utilitários e validações

### View (`view/main.py`)
- **Responsabilidade**: Interface de usuário (CLI)
- **Funções**:
  - Parsing de argumentos de linha de comando
  - Apresentação de resultados
  - Interface de entrada de dados

## 🚀 Instalação Rápida (Linux e Windows)

### 1. Clone o repositório
```bash
git clone <repository-url>
cd MHAutoML/src
```

### 2. Instale as dependências principais
```bash
pip install -r requirements.txt
```

### 3. Instale as dependências para relatórios PDF

#### No **Windows**:
```bash
pip install -r requirements_pdf.txt
# ReportLab funciona nativamente, não é necessário instalar dependências extras.
```

#### No **Linux**:
```bash
pip install -r requirements_pdf.txt
# Instale dependências do sistema para WeasyPrint:
sudo apt-get install build-essential python3-dev python3-pip python3-setuptools python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
```

### 4. Instale o sistema em modo desenvolvimento
```bash
python setup.py sdist bdist_wheel
pip install -e .
```

### 5. Execute o sistema (Linux ou Windows)
```bash
python view/main.py -d Datasets/dataset_sujo.csv -l class
```

> **Dica:** Para rodar testes, use:
> - Linux: `python -m pytest test/`
> - Windows: `python -m pytest test\\`

## 💻 Uso

### Comando Básico

```bash
python view/main.py -d <dataset_path> -l <label_column>
```

### Parâmetros Disponíveis

| Parâmetro | Descrição | Padrão | Opções |
|-----------|-----------|--------|--------|
| `-d, --dataset` | Caminho para o dataset | Obrigatório | CSV, Excel |
| `-l, --label-column` | Nome da coluna target | Obrigatório | String |
| `--log-level` | Nível de logging | info | debug, info |
| `--remove-duplicates` | Remover duplicatas | True | True/False |
| `--remove-missing-values` | Remover valores faltantes | True | True/False |
| `--remove-outliers` | Remover outliers | True | True/False |
| `--one-hot-encoder` | Aplicar one-hot encoding | True | True/False |
| `--label-encode` | Aplicar label encoding | True | True/False |
| `--balance-classes` | Balanceamento de classes | None | SMOTE, RUS |
| `--feature-selection` | Método de seleção de features | LASSO | PCA, LASSO, ANOVA |

### Exemplos de Uso

```bash
# Uso básico
python view/main.py -d Datasets/dataset_sujo.csv -l class

# Com balanceamento de classes
python view/main.py -d Datasets/dataset_sujo.csv -l class --balance-classes SMOTE

# Com seleção de features específica
python view/main.py -d Datasets/dataset_sujo.csv -l class --feature-selection PCA

# Com logging detalhado
python view/main.py -d Datasets/dataset_sujo.csv -l class --log-level debug
```

## 🔄 Pipeline de Machine Learning

O sistema executa um pipeline completo de ML com as seguintes etapas:

### 1. Data Info (`model/preprocessing/data_info.py`)
**Objetivo**: Análise exploratória inicial dos dados

**Funcionalidades**:
- Informações do sistema (OS, RAM, etc.)
- Estatísticas básicas do dataset
- Tipos de dados das colunas
- Análise de balanceamento de classes
- Detecção de duplicatas e valores faltantes
- Identificação de features Android (permissões, API calls)

**Saídas**:
- Tabelas formatadas com informações
- Logs detalhados da análise

### 2. Data Cleaning (`model/preprocessing/data_cleaning.py`)
**Objetivo**: Limpeza e preparação dos dados

**Funcionalidades**:
- Remoção de duplicatas (com detecção de assinaturas criptográficas)
- Tratamento de valores faltantes
- Remoção de outliers
- Conversão de tipos de dados
- Geração de heatmaps de valores faltantes

**Saídas**:
- Dataset limpo
- Visualizações de limpeza
- Relatórios de transformações

### 3. Data Transformation (`model/preprocessing/data_transformation.py`)
**Objetivo**: Transformação de dados para ML

**Funcionalidades**:
- Label encoding para variáveis categóricas
- One-hot encoding
- Normalização de dados
- Preparação para algoritmos de ML

**Saídas**:
- Dataset transformado
- Mapeamentos de encoding

### 4. Feature Engineering (`model/feature_engineering/data_reduction.py`)
**Objetivo**: Seleção e redução de features

**Métodos Disponíveis**:
- **LASSO**: Seleção baseada em regularização L1
- **PCA**: Redução de dimensionalidade
- **ANOVA**: Seleção baseada em testes estatísticos

**Funcionalidades**:
- Balanceamento de classes (SMOTE/RUS)
- Seleção automática de features
- Visualizações de importância
- Redução de dimensionalidade

**Saídas**:
- Features selecionadas
- Gráficos de importância
- Informações de transformação

### 5. Model Optimization (`model/optimization/hyperparameters_methods.py`)
**Objetivo**: Otimização automática de hiperparâmetros

**Algoritmos Suportados**:
- LightGBM
- CatBoost
- Random Forest
- Decision Tree
- Extra Trees
- K-Nearest Neighbors

**Funcionalidades**:
- Otimização com Optuna
- Cross-validation
- Múltiplas métricas de avaliação
- Ranking de modelos

**Saídas**:
- Melhores hiperparâmetros
- Ranking de algoritmos
- Relatórios de otimização

### 6. Interpretability (`model/interpretability/interpretability.py`)
**Objetivo**: Interpretação e explicação dos modelos

**Métodos**:
- **SHAP**: Explicações globais e locais
- **LIME**: Explicações locais
- **Feature Importance**: Importância de features

**Funcionalidades**:
- Explicações para diferentes tipos de modelos
- Visualizações interativas
- Relatórios HTML
- Gráficos de importância

**Saídas**:
- Explicações SHAP
- Relatórios LIME
- Gráficos de importância

## 📁 Estrutura do Projeto

```
src/
├── controller/
│   ├── __init__.py
│   └── core.py                 # Controller principal
├── model/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_info.py        # Análise de dados
│   │   ├── data_cleaning.py    # Limpeza de dados
│   │   └── data_transformation.py # Transformação
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   └── data_reduction.py   # Seleção de features
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── hyperparameters_methods.py # Otimização
│   │   ├── grid_search.py      # Grid search
│   │   └── hyperparameter_optimization.py
│   ├── interpretability/
│   │   ├── __init__.py
│   │   └── interpretability.py # Interpretação
│   └── tools/
│       ├── __init__.py
│       ├── dataset_validation.py
│       ├── mlflow_manager.py
│       └── report_generator.py
├── view/
│   ├── __init__.py
│   ├── main.py                 # Interface CLI
│   └── logo.py                 # Logo do sistema
├── test/                       # Testes unitários
├── scripts/
│   ├── check_html_feature_names.py
│   ├── debug_shap_values.py
│   └── list_artifacts.py
├── data/
│   └── dataset.csv
├── Datasets/
│   ├── *.csv
├── docs/
│   ├── ARTIFACTS_SUMMARY.md
│   ├── ARTIFACTS_LIST.md
│   ├── CONVERSATION_SUMMARY.md
│   ├── PDF_REPORT_GUIDE.md
│   ├── RELATORIO_FINAL_FEATURE_NAMES.md
│   ├── RESULTADO_TESTE_SHAP_FORCE_PLOT.md
│   ├── DIAGRAMS.md
│   ├── class_diagram.puml
│   ├── component_diagram.puml
│   ├── sequence_diagram.puml
│   └── desafios_tecnicos_completado.tex
├── results/
│   └── catboost_info/
├── requirements.txt            # Dependências
├── requirements_pdf.txt        # Dependências para relatórios PDF
├── setup.py                    # Configuração de instalação
└── README.md                   # Esta documentação
```

## ⚙️ Configurações

### Configurações do Sistema

O sistema pode ser configurado através de parâmetros de linha de comando:

```python
# Exemplo de configuração avançada
core = Core(
    dataset_url="path/to/dataset.csv",
    label="target_column",
    log_level="info",
    remove_duplicates=True,
    remove_missing_values=True,
    remove_outliers=True,
    one_hot_encoder=True,
    do_label_encode=True,
    balance_classes="SMOTE",  # ou "RUS" ou None
    feature_selection_method="LASSO"  # ou "PCA" ou "ANOVA"
)
```

### Configurações de Logging

```python
# Níveis de logging disponíveis
log_levels = ["debug", "info", "warning", "error", "critical"]
```

### Configurações de MLflow

O sistema configura automaticamente:
- Tracking de experimentos
- Logging de métricas
- Artefatos de modelo
- Interface web em http://localhost:5000

## 📊 Resultados e Artefatos

### Arquivos Gerados

O sistema gera diversos artefatos na pasta `results/`:

#### Modelos e Dados
- `best_model_YYYYMMDD_HHMMSS.pkl` - Modelo treinado
- `treino_YYYYMMDD_HHMMSS.csv` - Dados de treino processados

#### Relatórios de Performance
- `performance_metrics.jpg` - Gráficos de métricas
- `performance_summary.csv` - Resumo de performance
- `Hyperparameters_Results.csv` - Resultados de otimização
- `Models_Ranking.csv` - Ranking de algoritmos

#### Visualizações
- `missing_values_heatmap.png` - Heatmap de valores faltantes
- `clean_missing_values_heatmap.png` - Heatmap após limpeza
- `lasso_feature_importance.png` - Importância de features LASSO
- `train_test_distribution.png` - Distribuição train/test

#### Interpretabilidade
- `shap_interpretability_YYYYMMDD_HHMMSS.html` - Explicações SHAP
- `lime_interpretability_YYYYMMDD_HHMMSS.html` - Explicações LIME
- `lime_feature_importance_YYYYMMDD_HHMMSS.png` - Importância LIME

#### Otimização
- `optuna_optimization_history.html` - Histórico de otimização
- `optuna_param_importance.html` - Importância de parâmetros
- `optuna_parallel_coordinate.html` - Coordenadas paralelas
- `optuna_slice_plot.html` - Gráficos de fatia

#### Relatórios
- `report_YYYYMMDD_HHMMSS.html` - Relatório completo

### Métricas de Avaliação

O sistema calcula e reporta:

- **Accuracy**: Acurácia geral
- **Precision**: Precisão por classe
- **Recall**: Recall por classe
- **F1-Score**: F1-score por classe
- **ROC-AUC**: Área sob a curva ROC
- **MCC**: Coeficiente de correlação de Matthews

### Exemplo de Saída

```
INFO: Top ranked algorithms:
Classifier: LightGBM, Value: 0.8937
Classifier: CatBoost, Value: 0.8834
Classifier: DecisionTreeClassifier, Value: 0.8507
Classifier: RandomForestClassifier, Value: 0.8441
Classifier: ExtraTreesClassifier, Value: 0.8283

INFO: Classification Report:
              precision    recall  f1-score   support
           0       0.93      0.97      0.95      2534
           1       0.95      0.88      0.92      1526
    accuracy                           0.94      4060
   macro avg       0.94      0.93      0.93      4060
weighted avg       0.94      0.94      0.94      4060
```

## 🎯 Exemplos de Uso

### Exemplo 1: Detecção Básica de Malware

```bash
# Dataset com features de malware Android
python view/main.py -d Datasets/drebin215.csv -l class
```

### Exemplo 2: Análise com Balanceamento

```bash
# Usando SMOTE para balancear classes desbalanceadas
python view/main.py -d Datasets/dataset_sujo.csv -l class --balance-classes SMOTE
```

### Exemplo 3: Redução de Dimensionalidade

```