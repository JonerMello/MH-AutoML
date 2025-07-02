# 📋 RESUMO COMPLETO DE ARTEFATOS - MH-AutoML

## 📊 Estatísticas Gerais
- **Total de Artefatos**: 34 arquivos
- **Tamanho Total**: 36.39 MB
- **Última Atualização**: 2025-07-01 20:16:21

## 🗂️ Distribuição por Tipo de Arquivo

| Tipo | Quantidade | Tamanho | Descrição |
|------|------------|---------|-----------|
| **PNG** | 18 arquivos | 2.34 MB | Gráficos e visualizações |
| **HTML** | 7 arquivos | 15.32 MB | Relatórios interativos |
| **CSV** | 6 arquivos | 16.16 MB | Dados e resultados |
| **PDF** | 1 arquivo | 1.93 MB | Relatório final |
| **PKL** | 1 arquivo | 560 KB | Modelo treinado |
| **JPG** | 1 arquivo | 95 KB | Métricas de performance |

## 📂 Organização por Seção MLflow

### 01_preprocessing (2 artefatos)
**Pré-processamento de dados**
- `clean_missing_values_heatmap.png` (120 KB) - Heatmap após limpeza
- `missing_values_heatmap.png` (126 KB) - Heatmap antes da limpeza

### 02_feature_engineering (4 artefatos)
**Engenharia e seleção de features**
- `Features_Selected_20250701_200335.csv` (8.05 MB) - Features selecionadas
- `lasso_feature_importance.png` (123 KB) - Importância LASSO
- `lime_feature_importance_20250701_200518.png` (236 KB) - Importância LIME
- `treino_20250701_200259.csv` (8.07 MB) - Dataset de treino

### 03_model_optimization (11 artefatos)
**Otimização de hiperparâmetros**
- `Hyperparameters_Results.csv` (15.5 KB) - Resultados dos experimentos
- `Models_Ranking.csv` (1.3 KB) - Ranking dos modelos
- `optuna_optimization_history.html` (3.44 MB) - Histórico interativo
- `optuna_optimization_history.png` (32 KB) - Gráfico do histórico
- `optuna_parallel_coordinate.html` (3.44 MB) - Coordenadas paralelas interativas
- `optuna_parallel_coordinate.png` (362 KB) - Gráfico coordenadas paralelas
- `optuna_param_importance.html` (3.44 MB) - Importância interativa
- `optuna_param_importance.png` (39 KB) - Gráfico importância parâmetros
- `optuna_slice_plot.html` (3.46 MB) - Gráfico de fatias interativo
- `optuna_slice_plot.png` (122 KB) - Gráfico de fatias
- `optuna_trials.csv` (13.6 KB) - Dados dos trials

### 04_evaluation_metrics (8 artefatos)
**Métricas e gráficos de avaliação**
- `best_model_20250701_200259.pkl` (573 KB) - Modelo treinado
- `confusion_matrix.png` (87 KB) - Matriz de confusão
- `metrics_by_class.png` (106 KB) - Métricas por classe
- `performance_metrics.jpg` (98 KB) - Métricas de performance
- `performance_summary.csv` (0.2 KB) - Resumo de performance
- `precision_recall_curve.png` (105 KB) - Curva precisão-recall
- `probability_distribution.png` (95 KB) - Distribuição probabilidades
- `roc_curve.png` (171 KB) - Curva ROC/AUC

### 05_interpretability (7 artefatos)
**Interpretabilidade do modelo**
- `decision_tree_plot_LGBMClassifier_20250701_200518.png` (215 KB) - Árvore de decisão
- `lime_feature_importance_20250701_200518.png` (236 KB) - Importância LIME
- `lime_interpretability_20250701_200518.html` (1.21 MB) - Explicação LIME interativa
- `lime_interpretability_20250701_200518.png` (235 KB) - Explicação LIME
- `shap_force_plot_LGBMClassifier_20250701_200518.html` (299 KB) - Plot força SHAP interativo
- `shap_force_plot_LGBMClassifier_20250701_200518.png` (40 KB) - Plot força SHAP
- `shap_summary_plot_LGBMClassifier_20250701_200518.png` (154 KB) - Resumo SHAP

### report (9 artefatos)
**Relatórios finais**
- `lime_interpretability_20250701_200518.html` (1.21 MB) - Explicação LIME
- `optuna_optimization_history.html` (3.44 MB) - Histórico otimização
- `optuna_parallel_coordinate.html` (3.44 MB) - Coordenadas paralelas
- `optuna_param_importance.html` (3.44 MB) - Importância parâmetros
- `optuna_slice_plot.html` (3.46 MB) - Gráfico fatias
- `pdf_report_20250701_200537.pdf` (1.93 MB) - Relatório PDF
- `report_20250701_200537.html` (16.5 KB) - Relatório HTML
- `shap_force_plot_LGBMClassifier_20250701_200518.html` (299 KB) - Plot força SHAP

### Outros (1 artefato)
- `train_test_distribution.png` (28 KB) - Distribuição treino/teste

## 🎯 Categorização Funcional

### Gráficos de Avaliação (5 artefatos)
- Matriz de confusão
- Curva ROC/AUC
- Curva precisão-recall
- Distribuição de probabilidades
- Métricas por classe

### Gráficos de Otimização (10 artefatos)
- Histórico de otimização (PNG + HTML)
- Coordenadas paralelas (PNG + HTML)
- Importância de parâmetros (PNG + HTML)
- Gráfico de fatias (PNG + HTML)
- Resultados CSV

### Gráficos de Features (2 artefatos)
- Importância LASSO
- Importância LIME

### Gráficos de Interpretabilidade (7 artefatos)
- Árvore de decisão
- Explicações LIME (PNG + HTML)
- Plots SHAP (PNG + HTML)

### Gráficos de Pré-processamento (2 artefatos)
- Heatmaps de valores faltantes

### Dados e Modelos (6 artefatos)
- Modelo treinado (PKL)
- Features selecionadas (CSV)
- Dataset de treino (CSV)
- Resultados de otimização (CSV)

### Relatórios (9 artefatos)
- Relatório PDF
- Relatório HTML
- Visualizações interativas

## 🔍 Principais Características

### ✅ Pontos Fortes
- **Cobertura Completa**: Todos os aspectos do pipeline documentados
- **Visualizações Interativas**: HTML para exploração detalhada
- **Alta Qualidade**: Imagens em 300 DPI
- **Organização Clara**: Estrutura hierárquica no MLflow
- **Reprodução**: Todos os artefatos permitem reprodução

### 📊 Distribuição de Tamanho
- **Maior**: Relatórios interativos HTML (15.32 MB)
- **Médio**: Dados CSV (16.16 MB)
- **Menor**: Gráficos PNG (2.34 MB)

### 🎨 Tipos de Visualização
- **Heatmaps**: Valores faltantes
- **Curvas**: ROC, precisão-recall
- **Histogramas**: Distribuição de probabilidades
- **Gráficos de Barras**: Métricas por classe
- **Árvores**: Visualização de decisão
- **Coordenadas Paralelas**: Otimização
- **Plots de Força**: SHAP

## 🚀 Como Acessar

### Via MLflow UI
```bash
mlflow ui
# Acesse: http://localhost:5000
```

### Via Pasta Local
```bash
ls results/
# Todos os artefatos na pasta results/
```

### Via Script
```bash
python list_artifacts.py
# Lista completa e organizada
```

## 📈 Valor para Detecção de Malware

### Análise de Performance
- **Matriz de Confusão**: Identifica falsos positivos/negativos
- **Curvas ROC/PR**: Avalia discriminação geral
- **Métricas por Classe**: Compara performance específica

### Interpretabilidade
- **SHAP**: Explica predições individuais
- **LIME**: Explicações locais
- **Árvore de Decisão**: Visualiza regras do modelo

### Otimização
- **Histórico**: Acompanha evolução da otimização
- **Importância**: Identifica parâmetros críticos
- **Coordenadas Paralelas**: Explora espaço de busca

### Debugging
- **Valores Faltantes**: Verifica qualidade dos dados
- **Features**: Analisa seleção de características
- **Distribuições**: Valida balanceamento

## 🎉 Conclusão

O MH-AutoML gera um conjunto **completo e profissional** de artefatos que cobrem todos os aspectos do pipeline de machine learning, desde o pré-processamento até a interpretabilidade final. Com **34 artefatos** organizados em **7 seções**, oferece uma visão abrangente e detalhada do processo de detecção de malware. 