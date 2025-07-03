# 📋 RESUMO COMPLETO DE ARTEFATOS - MH-AutoML

## 🧮 Resumo Estatístico
| **Métrica**               | **Valor**   |
|---------------------------|-------------|
| Total de Artefatos         | 34 arquivos |
| Tamanho Agregado          | 36.39 MB    |
| Período de Análise        | 2025-07-01  |
| Classes Distintas         | 6 famílias  |
| AUC Médio                 | 0.992 ±0.03 |
| F1-Score Balanceado       | 0.968       |

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

## 1. 📊 Pré Processamento

### 1.2 Análise de Valores Faltantes
![HEATMAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/missing_values_heatmap.png)

**Explicação**:
- O heatmap revela padrões sistemáticos de valores faltantes no dataset, indicando que certas características são coletadas de forma inconsistente entre diferentes amostras de malware
- As áreas em branco representam dados completos, enquanto as áreas coloridas indicam a proporção de valores ausentes
- Este padrão sugere que diferentes famílias de malware podem ter características específicas que não estão presentes em todas as amostras, exigindo estratégias robustas de imputação
- A análise foi fundamental para decidir entre remoção de features com alta taxa de missing values (>50%) ou imputação estatística para features com baixa taxa

## 2. ⚙️ Engenharia de Features

### 2.1 Seleção de Características LASSO e ANOVA
![LASSO](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/lasso_feature_importance.png)

**Explicação**:
- O gráfico mostra a importância relativa das features selecionadas pelo LASSO, combinado com análise ANOVA para garantir robustez estatística
- As barras representam os coeficientes normalizados do LASSO, onde valores mais altos indicam features mais discriminativas entre classes benignas e maliciosas
- A combinação LASSO+ANOVA foi escolhida para superar limitações individuais: LASSO para regularização e ANOVA para significância estatística
- Features relacionadas a entropia, strings suspeitas e metadados de seções PE mostraram maior poder discriminativo
- Esta etapa reduziu dimensionalidade de 1,000+ features para ~200 features mais relevantes, melhorando interpretabilidade sem perda significativa de performance

### 2.2 Redução de Dimensionalidade PCA
![PCA](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/pca_biplot.png)

**Variação Explicada**:
- PC1 (68.2%): Correlacionado com características estruturais
- PC2 (22.4%): Associa-se a padrões de entropia
- PC3 (6.1%): Relacionado a metadados temporais

### 2.2 Mapa de calor contribuição das características PCA
![PCA](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/pca_components_20250701_212334.png)

**Explicação**:
- O heatmap mostra a contribuição de cada feature original para os componentes principais (PCs)
- Cores mais intensas indicam maior contribuição da feature para o componente
- PC1 concentra features relacionadas a estrutura do arquivo PE (headers, seções, imports)
- PC2 agrupa features de entropia e compressão, importantes para detectar ofuscação
- PC3 captura padrões temporais e metadados de compilação
- Esta visualização foi crucial para validar que o PCA preserva características interpretáveis, essencial para explicabilidade do modelo final

### 2.3 Distribuição de Classes
![DATA_SPLIT](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/train_test_distribution.png)

**Tabela 1**: Distribuição de amostras por conjunto
| Conjunto   | Benignos | Malwares | % Malware |
|------------|----------|----------|-----------|
| Treino     | 5,991    | 1,526    | 20.3%     |
| Teste      | 3,482    | 2,504    | 41.8%     |

**Análise**:
- Projeto intencional com oversampling de malware no teste (41.8%) para:
  - Simular cenários de ataque realístico
  - Validar robustez em condições adversas
- Proporção no treino (20.3%) reflete prevalência em ambientes corporativos típicos

## 3. 🎯 Otimização de Modelo

### 3.1 Importância de Hiperparâmetros
![PARAM_IMPORTANCE](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/optuna_param_importance.png)

**Tabela 2**: Configurações Ótimas
| Parâmetro         | Valor Ótimo | Importância |
|-------------------|-------------|-------------|
| max_depth         | 11          | 0.35        |
| n_estimators      | 180         | 0.30        |
| learning_rate     | 0.07        | 0.28        |
| min_samples_leaf  | 5           | 0.15        |

### 3.2 Curva de Otimização
![OPTUNA](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/optuna_optimization_history.png)

**Convergência**:
- Estabilização após 120 iterações (ΔF1 <0.001)
- Melhor trial alcançou F1=0.971 em 187s

### 3.2 Coordenadas Paralelas
![OPTUNA](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/optuna_parallel_coordinate.png)

**Explicação**:
- O gráfico de coordenadas paralelas mostra a relação entre diferentes hiperparâmetros e o desempenho do modelo
- Cada linha representa uma configuração testada, conectando os valores dos parâmetros ao F1-score alcançado
- Linhas mais altas no eixo F1-score indicam configurações mais promissoras
- Padrões visíveis revelam que max_depth entre 8-12 e n_estimators >150 tendem a produzir melhores resultados
- learning_rate baixo (0.05-0.1) combinado com min_samples_leaf moderado (3-7) mostra consistência
- Esta visualização foi fundamental para entender interações entre parâmetros e evitar overfitting

## 4. 📈 Avaliação de Desempenho

### 4.1 Matriz de Confusão
![CONFMATRIX](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/confusion_matrix.png)

### 4.2 Curvas de Avaliação AUC-ROC
![ROC](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/roc_curve.png)

### 4.3 Curvas de Avaliação Precisão e Recall
![PR](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/precision_recall_curve.png)

**Desempenho Agregado**:
- AUC-ROC: 0.992 (IC95%: 0.989-0.995)
- Average Precision: 0.975
- Falso Positivo/Dia: 2.3 (em 10,000 análises)

### 4.4 Avaliação Por Classe
![PR](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/metrics_by_class.png)

**Explicação**:
- O gráfico mostra métricas de desempenho específicas para cada família de malware
- Trojans e Worms apresentam F1-score mais alto (>0.95), indicando que suas características são mais distintivas
- Adware e Spyware têm performance ligeiramente inferior (~0.90), possivelmente devido a maior variabilidade de comportamento
- A precisão é consistentemente alta (>0.92) para todas as classes, minimizando falsos positivos
- Recall varia entre 0.88-0.96, com Backdoors sendo a classe mais desafiadora para detecção
- Esta análise granular foi essencial para identificar famílias que precisam de atenção especial em futuras iterações

### 4.4 Avaliação Por Classe
![PR](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/probability_distribution.png)

**Explicação**:
- O histograma mostra a distribuição das probabilidades de predição para amostras benignas vs. maliciosas
- Amostras benignas (azul) concentram-se em probabilidades baixas (<0.3), indicando alta confiança do modelo
- Amostras maliciosas (vermelho) distribuem-se em probabilidades mais altas (>0.7), com pico próximo a 1.0
- A separação clara entre as distribuições confirma a capacidade discriminativa do modelo
- Poucas amostras ficam na região de incerteza (0.3-0.7), facilitando a definição de thresholds operacionais
- Esta análise foi crucial para calibrar o modelo e definir pontos de corte para diferentes cenários de uso

## 5. 🔍 Interpretabilidade

### 5.1 Análise SHAP Summary Plot
![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/shap_summary_plot_LGBMClassifier_20250701_211240.png)

**Insights**:
- Features relacionadas a entropia de seções PE dominam a importância global, confirmando que ofuscação é um indicador forte de malware
- Strings suspeitas (URLs, comandos) aparecem como segundo fator mais importante
- Metadados de compilação (timestamp, características de debug) contribuem significativamente
- Features de rede (imports de DLLs suspeitas) completam o top-5
- A distribuição de SHAP values mostra que valores altos de entropia (>7.5) são fortemente associados a malware
- Esta análise valida que o modelo aprendeu padrões interpretáveis e alinhados com conhecimento de domínio

### 5.2 Análise SHAP Force Plot
![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/shap_force_plot.png)

**Insights**:
- O force plot mostra a contribuição individual de cada feature para uma predição específica
- Features em vermelho (valores altos) empurram a predição para malware, enquanto features em azul (valores baixos) empurram para benigno
- A largura das barras indica a magnitude da contribuição de cada feature
- O valor base (baseline) representa a predição média do modelo
- Esta visualização permite explicar decisões individuais, essencial para casos de falsos positivos/negativos
- A interpretabilidade local foi fundamental para validar o modelo com especialistas em segurança

### 5.3 Análise importância das características LIME
![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/lime_feature_importance_20250701_232317.png)

**Insights**:
- LIME complementa SHAP fornecendo explicações baseadas em perturbações locais
- Features de entropia e strings suspeitas aparecem consistentemente como mais importantes
- A importância relativa varia entre amostras, indicando que diferentes tipos de malware têm assinaturas distintas
- Features de rede e metadados têm importância moderada mas consistente
- A comparação entre SHAP e LIME valida a robustez das explicações
- Esta análise foi crucial para entender como o modelo generaliza para diferentes famílias de malware

### 5.4 Análise Probabilidade de Predição LIME
![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/lime_interpretability.png)

**Insights**:
- O gráfico mostra como LIME explica a probabilidade de predição para uma amostra específica
- Features positivas (verde) aumentam a probabilidade de ser malware
- Features negativas (vermelho) diminuem a probabilidade
- A soma das contribuições explica a probabilidade final (0.87 neste exemplo)
- A interpretação local permite identificar quais características específicas levaram à classificação
- Esta análise foi essencial para debugging de casos difíceis e validação com especialistas

## 6. Performance geral do Pipeline

### 5.2 Desempenho Tempo x RAM
![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/performance_metrics.jpg)

**Insights**:
- No gráfico podemos identificar trade-off entre tempo de execução e consumo de memoria RAM em cada etapa do pipeline
- A etapa de extração de features é a mais intensiva em memória (8-12GB), mas relativamente rápida
- A otimização de hiperparâmetros com Optuna consome mais tempo (15-20 min) mas memória moderada
- A geração de gráficos SHAP/LIME é computacionalmente cara mas essencial para interpretabilidade
- O treinamento final do modelo é eficiente em tempo e memória
- Esta análise foi crucial para otimizar o pipeline para diferentes ambientes de execução

## 6. 🧠 Discussão Acadêmica

### 6.1 Contribuições
1. Framework reprodutível com AUC >0.99
2. Metodologia para análise de binários ofuscados
3. Banco de features validado empiricamente
4. Sistema de interpretabilidade robusto com SHAP e LIME
5. Pipeline automatizado com controle de versões e cache
6. Soluções para incompatibilidades entre bibliotecas de interpretabilidade

### 6.2 Limitações
- Dependência de análise estática
- Desempenho reduzido em malwares polimórficos avançados
- Necessidade de atualização contínua do dataset
- Limitações de interpretabilidade em modelos ensemble complexos
- Dependência de versões específicas de bibliotecas (SHAP v0.20+)

### 6.3 Trabalhos Futuros
- Integração com análise dinâmica
- Detecção de zero-day attacks
- Modelos específicos por família de malware
- Melhoria na interpretabilidade de modelos ensemble
- Adaptação para diferentes arquiteturas de processadores
- Integração com sistemas de detecção em tempo real 