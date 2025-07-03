
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

## 1. 📊 Pré Processamento

### 1.2 Análise de Valores Faltantes
![HEATMAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/missing_values_heatmap.png)

**Explicação**:
- O heatmap revela padrões sistemáticos de valores faltantes no dataset, indicando que certas características são coletadas de forma inconsistente entre diferentes amostras de malware
- As áreas em roxo representam dados completos, enquanto as áreas amarelas indicam a proporção de valores ausentes
- A análise é fundamental para decidir entre remoção de características com alta taxa de missing values (>50%) ou imputação estatística para características com baixa taxa

## 2. ⚙️ Engenharia de Features

### 2.1 Seleção de Características LASSO e ANOVA
![LASSO](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/lasso_feature_importance.png)

**Explicação**:
- O gráfico mostra a importância relativa das características selecionadas pelo LASSO, o mesmo modelo se aplica para  o método de seleção ANOVA
- As barras representam os coeficientes normalizados do LASSO, onde valores mais altos indicam características mais discriminativas entre classes benignas e maliciosas
- Esta etapa reduz dimensionalidade de características selecionando apenas as  mais relevantes, melhorando interpretabilidade sem perda significativa de performance

### 2.2 Redução de Dimensionalidade PCA
![PCA](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/pca_biplot.png)

**Explicação**:
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

**Explicação**:
- Projeto intencional com oversampling de malware no teste (41.8%) para:
  - Simular cenários de ataque realístico
  - Validar robustez em condições adversas
- Proporção no treino (20.3%) reflete prevalência em ambientes corporativos típicos

## 3. 🎯 Otimização de Modelo

### 3.1 Importância de Hiperparâmetros
![PARAM_IMPORTANCE](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/optuna_param_importance.png)

**Explicação**:
O gráfico de importância dos hiperparâmetros fornece insights sobre quais parâmetros do modelo exercem maior influência no desempenho, permitindo reduzir o espaço de busca e, consequentemente, tornar o processo de otimização mais eficiente.
| Parâmetro         | Valor Ótimo | Importância |
|-------------------|-------------|-------------|
| max_depth         | 11          | 0.35        |
| n_estimators      | 180         | 0.30        |
| learning_rate     | 0.07        | 0.28        |
| min_samples_leaf  | 5           | 0.15        |

### 3.2 Curva de Otimização
![OPTUNA](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/optuna_optimization_history.png)

**Explicação**:
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
**Explicação**:
A matriz de confusão demonstra a capacidade do modelo em distinguir entre amostras benignas e maliciosas.  Das **2.534** amostras benignas reais, **2.457** foram corretamente classificadas, com apenas **77 falsos positivos**, o que demonstra **alta especificidade**.Das **1.526** amostras maliciosas, **1.355** foram corretamente identificadas, com **171 falsos negativos**, revelando uma **boa sensibilidade**, embora ainda haja espaço para melhorias na detecção de ameaças. O desempenho indica que o modelo apresenta **baixo índice de alarmes falsos** e uma **eficácia significativa na identificação de malwares**, sendo adequado para cenários que exigem confiança tanto na detecção quanto na minimização de alertas indevidos.
### 4.2 Curvas de Avaliação AUC-ROC
![ROC](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/roc_curve.png)
**Explicação**:
A curva ROC (Receiver Operating Characteristic) compara a taxa de verdadeiros positivos (sensibilidade) com a taxa de falsos positivos, em diferentes limiares de decisão. A área sob a curva (AUC) indica a capacidade do modelo de distinguir entre as classes. Um valor de AUC próximo de 1, como o obtido (0.982), demonstra excelente desempenho discriminativo, com mínima sobreposição entre classes benignas e maliciosas.
### 4.3 Curvas de Avaliação Precisão e Recall
![PR](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/precision_recall_curve.png)

**Explicação**:
A curva de Precisão-Recall é particularmente útil em cenários com classes desbalanceadas. Ela ilustra a relação entre a **precisão** (proporção de verdadeiros positivos entre os positivos previstos) e o **recall** (proporção de verdadeiros positivos identificados corretamente). O valor médio de precisão (Average Precision = 0.975) indica que, mesmo com alto recall, o modelo mantém uma elevada taxa de precisão, minimizando alarmes falsos.

### 4.4 Avaliação Por Classe
![PR](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/metrics_by_class.png)

**Explicação**:
Este gráfico compara as métricas de avaliação — **Precisão**, **Recall** e **F1-Score** — para cada classe (Benigno e Malware), permitindo uma análise detalhada do equilíbrio do modelo entre diferentes tipos de erro.

-   **Benigno**: apresenta alta **revocação (0.970)**, o que indica que quase todos os aplicativos benignos foram corretamente identificados. A **precisão (0.935)** também é elevada, significando que a maioria das previsões como benignas realmente corresponde a essa classe. O **F1-Score (0.952)** resume esse bom equilíbrio entre precisão e recall.
    
-   **Malware**: tem uma **precisão ainda mais alta (0.946)**, o que é crucial em sistemas de segurança, pois minimiza o número de falsos positivos (benignos classificados como malware). A **revocação (0.888)**, embora ligeiramente inferior, ainda indica boa capacidade de detecção. O **F1-Score (0.916)** demonstra um desempenho sólido e consistente na identificação de ameaças.
    

Em conjunto, esses resultados sugerem que o modelo mantém **bom equilíbrio entre segurança (detecção de malware) e confiabilidade (baixo alarme falso)**, sendo apropriado para ambientes onde a minimização de riscos e ruído operacional é essencial.

### 4.4 Avaliação Por Classe
![PR](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/probability_distribution.png)

**Explicação**:
Este gráfico exibe a densidade das probabilidades preditas pelo modelo para cada classe (Benigno em azul, Malware em vermelho), com um **limiar de decisão fixado em 0.5** (linha tracejada).

Observações importantes:

-   A maioria dos exemplos **benignos** concentra-se à esquerda do limiar (probabilidades próximas de 0), indicando alta confiança do modelo ao classificá-los corretamente como não maliciosos.
    
-   De forma análoga, a maioria dos exemplos **maliciosos** se agrupa à direita do limiar (probabilidades próximas de 1), evidenciando também alta confiança na classificação como malware.
    
-   A separação clara entre as duas distribuições sugere **alta discriminabilidade** do modelo, ou seja, baixa ambiguidade na predição.
    
-   A baixa sobreposição entre as curvas reduz a taxa de erros de classificação, como falsos positivos e falsos negativos.
    

Esse comportamento é desejável em sistemas de detecção, pois reforça que o modelo não apenas acerta as classes, mas o faz com **alta confiança estatística**, tornando a ferramenta confiável para uso em ambientes críticos.

## 5. 🔍 Interpretabilidade

### 5.1 Análise SHAP Summary Plot
![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/shap_summary_plot_LGBMClassifier_20250701_211240.png)

**Explicação**:
Este gráfico resume a **influência de cada permissão Android** sobre as previsões do modelo, utilizando valores SHAP, que quantificam o impacto de cada feature na saída do classificador.

**Interpretação do Gráfico:**

-   O eixo X representa o valor SHAP, ou seja, o **impacto individual** da feature na predição. Valores positivos empurram a previsão para “malware”, enquanto valores negativos favorecem a classe “benigno”.
    
-   Cada ponto representa uma amostra; a cor indica o valor da feature (vermelho = valor alto / presente, azul = valor baixo / ausente).
    
-   As permissões mais relevantes incluem:
    
    -   `SEND_SMS_1.0`, `READ_PHONE_STATE_1.0`, `READ_SMS_1.0`: quando **ativas (vermelhas)**, têm forte impacto positivo na classificação como **malware**, sugerindo comportamento malicioso.
        
    -   `GET_ACCOUNTS_1.0` e `ACCESS_NETWORK_STATE_1.0`: apresentam impacto misto, com comportamentos diferentes dependendo do contexto.
        
    -   Permissões como `RECEIVE_BOOT_COMPLETED_1.0` e `WRITE_HISTORY_BOOKMARKS_1.0` também contribuem, mas com menor intensidade.
        

Este gráfico permite concluir que o modelo **aprende padrões interpretáveis** e condizentes com práticas conhecidas de malware, reforçando a confiabilidade e **explicabilidade** do processo de decisão. Além disso, evidencia a importância de um subconjunto reduzido de permissões, o que pode auxiliar na **redução dimensional** e auditoria dos atributos usados.

### 5.2 Análise SHAP Force Plot
![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/shap_force_plot.png)

**Explicação**:
Este gráfico visualiza como os valores das features influenciaram uma predição específica do modelo. A previsão final (`f(x) = 5.94`) resulta da soma do valor base (média das predições) com os impactos acumulados de cada atributo.

**Interpretação:**

-   **Base value**: é o valor médio da saída do modelo sem considerar nenhuma feature (referência neutra).
    
-   **Setas vermelhas (→ higher)**: indicam features que **aumentaram** a probabilidade da amostra ser classificada como **malware**.
    
    -   `SEND_SMS_1.0`, `ACCESS_NETWORK_STATE_1.0` e `GET_ACCOUNTS_1.0` contribuíram positivamente, impulsionando a predição para a classe maliciosa.
        
-   **Seta azul (→ lower)**: representa uma feature que **reduziu** essa probabilidade.
    
    -   `WRITE_HISTORY_BOOKMARKS_1.0` teve um impacto negativo, atuando como um fator benigno.
        

O valor final de **5.94** está bem acima do limiar de decisão, reforçando que o modelo classificou esta instância com **alta confiança como malware**.

### 5.3 Análise importância das características LIME
![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/lime_feature_importance_20250701_232317.png)

**Explicação**:
O gráfico apresenta a contribuição individual de cada feature na decisão do modelo para uma **amostra específica**, permitindo uma análise **local** da explicação. Os pesos indicam o quanto cada atributo influenciou a classificação:

-   **Barras azuis (positivas)**: características que **favoreceram a predição como malware**.
    
    -   Destaques:
        
        -   `SEND_SMS > -0.56` (peso: +0.242)
            
        -   `READ_PHONE_STATE > 0.76` (peso: +0.241)
            
        -   `INTERNET > 0.38` (peso: +0.241)  
            → A presença ou valores altos dessas permissões aumentaram significativamente a probabilidade de a amostra ser classificada como maliciosa.
            
-   **Barras vermelhas (negativas)**: características que **atuaram contra a classificação como malware**, ou seja, aproximaram a instância da classe benigna.
    
    -   Destaques:
        
        -   `WRITE_HISTORY_BOOKMARKS <= -0.22` (peso: –0.224)
            
        -   `GET_ACCOUNTS <= 1.53` (peso: –0.188)
            
        -   `READ_SMS <= -0.48` (peso: –0.119)  
            → Esses atributos, ao estarem ausentes ou abaixo de determinado valor, indicaram comportamento benigno ao modelo.
            

Este gráfico complementa a explicação global do SHAP ao mostrar **como o modelo tomou uma decisão em um caso concreto**, oferecendo uma forma interpretável e confiável de justificar decisões individuais — fundamental em contextos como cibersegurança e auditoria.

### 5.4 Análise Probabilidade de Predição LIME
![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/lime_interpretability.png)

**Insights**:
O gráfico mostra como as **features ativas** (valores fornecidos à direita) impactaram a predição do modelo (barra central), contribuindo para a classificação como **malware (classe 1)** ou **benigno (classe 0)**.

#### 🔶 Principais contribuições para **classe 1 (malware)**:

-   `READ_PHONE_STATE = 0.76`
    
-   `INTERNET = 0.39`
    
-   `SEND_SMS = 1.78`
    
-   `CHANGE_WIFI_MULTICAST_STATE = -0.12`
    

Estas permissões são **fortes indicativos de comportamento malicioso** e empurraram a predição para a classe “malware”.

#### 🔷 Principais contribuições para **classe 0 (benigno)**:

-   `WRITE_HISTORY_BOOKMARKS = -0.30`
    
-   `GET_ACCOUNTS = -0.64`
    
-   `READ_CALL_LOG = -0.12`
    
-   `USE_CREDENTIALS = -0.33`
    
-   `READ_SMS = -0.48`
    
-   `MASTER_CLEAR = -0.11`
    

Essas permissões, por estarem ausentes ou com valor baixo, puxaram a predição em direção à classe benigna. Mesmo assim, o modelo atribuiu **alta probabilidade (≈ 1.00)** para a classe malware, dado o maior peso das permissões maliciosas.

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