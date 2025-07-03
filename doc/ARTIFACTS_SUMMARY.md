  

# 📋 RESUMO  DE ARTEFATOS - MH-AutoML

  

## 🧮 Resumo

Este documento apresenta uma análise detalhada dos principais artefatos gerados pelo pipeline MH-AutoML, uma solução desenvolvida para a detecção de malware em aplicações Android. Os resultados e as configurações aqui descritas são baseados em um conjunto de dados de teste, composto por **15.036 amostras e 51 características**.

O dataset reflete um cenário realista de distribuição de classes, onde a proporção de amostras benignas é naturalmente maior que a de malwares, apresentando **63.01% de amostras benignas e 36.99% de amostras maliciosas**. Essa característica do conjunto de dados é fundamental para avaliar a capacidade do modelo em lidar com a assimetria comum em problemas de detecção de ameaças.

Além das visualizações gráficas que ilustram cada etapa do processo, o pipeline também gera arquivos CSV e um modelo serializado, que registram escolhas e resultados cruciais. Ao longo das seções seguintes, serão abordadas as etapas cruciais do processo de Machine Learning:

-   **Pré-processamento dos dados:** Com destaque para a análise de valores faltantes.
    
-   **Engenharia de características:** Incluindo a seleção de features via LASSO e ANOVA, e a redução de dimensionalidade com PCA. Além disso, arquivos como `Features_Selected_20250701_232221.csv` e `treino_20250701_232146.csv` fornecem o registro exato das características selecionadas e dos dados utilizados no treinamento.
    
-   **Otimização do modelo:** Detalhando a importância dos hiperparâmetros e a curva de otimização. O arquivo `Hyperparameters_Results.csv` documenta todas as tentativas de otimização e as métricas de desempenho correspondentes, enquanto `Models_Ranking.csv` oferece uma classificação dos modelos testados.
    
-   **Avaliação de desempenho:** Apresentando métricas como matriz de confusão, curvas ROC, Precisão-Recall e avaliação por classe.
    
-   **Interpretabilidade do modelo:** Utilizando técnicas avançadas como SHAP e LIME, além da análise da árvore de decisão, para garantir a transparência e a confiabilidade das previsões.
    
-   **Performance geral do pipeline:** Avaliando o consumo de tempo e memória RAM por etapa.
    

Cada artefato visual e estatístico será explicado para oferecer uma compreensão completa do fluxo de trabalho, das escolhas técnicas e dos resultados alcançados pelo sistema MH-AutoML, incluindo o `best_model_20250701_232146.pkl`, que representa o modelo final otimizado.



## 1. 📊 Pré Processamento

  

### 1.2 Análise de Valores Faltantes

![HEATMAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/missing_values_heatmap.png)

  

**Explicação**:


A figura abaixo apresenta um *heatmap* de valores ausentes no conjunto de dados, em que as regiões em roxo indicam dados completos, enquanto as regiões em amarelo representam valores faltantes. Essa visualização é útil para identificar padrões sistemáticos de ausência, indicando que certas características foram coletadas de forma inconsistente entre as amostras.

É possível observar que algumas colunas apresentam uma proporção significativa de valores ausentes, o que pode comprometer a performance de modelos de aprendizado de máquina. Dessa forma:

- **Características com mais de 50% de valores ausentes** devem ser consideradas para remoção;
- **Características com taxa moderada de ausência** podem ser tratadas com técnicas de imputação.

  

## 2. ⚙️ Engenharia de Features

  

### 2.1 Seleção de Características LASSO e ANOVA

![LASSO](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/lasso_feature_importance.png)

  

**Explicação LASSO**:

O gráfico acima apresenta a importância das características extraída a partir dos coeficientes do modelo LASSO (*Least Absolute Shrinkage and Selection Operator*), que aplica regularização L1 durante o treinamento. Nesse contexto:

- As **barras azuis** representam características com coeficientes positivos, que estão positivamente associadas à classe-alvo (por exemplo, "malware");
- As **barras vermelhas** indicam características com coeficientes negativos, associadas à classe oposta (por exemplo, "benigno").

O valor absoluto do coeficiente indica o peso da influência da característica no modelo. Características com coeficiente zero foram automaticamente eliminadas pelo LASSO, contribuindo para a seleção de um subconjunto mais relevante de atributos e promovendo a interpretabilidade do modelo.

Destacam-se, por exemplo, permissões como `SEND_SMS`, `READ_PHONE_STATE`, `INTERNET` e `READ_SMS`, que apresentaram os maiores coeficientes positivos, sugerindo forte associação com o comportamento malicioso de aplicações Android. Esse tipo de análise é essencial para entender o papel individual de cada atributo no processo de classificação automatizada.
### Importância de Características com ANOVA

O gráfico acima representa os *F-values* obtidos pela aplicação do teste estatístico ANOVA (*Analysis of Variance*) para avaliação univariada de importância de cada característica em relação à variável-alvo.

Características com valores de F mais altos possuem maior poder discriminativo entre as classes. Diferente do LASSO, que realiza penalizações e seleção multivariada, a ANOVA avalia cada atributo de forma independente, oferecendo uma visão estatística útil para filtragem inicial de atributos.

Essa abordagem é especialmente útil em etapas preliminares de seleção de características, sendo frequentemente combinada com métodos de regularização ou árvores de decisão para análises mais robustas.
  

### 2.2 Redução de Dimensionalidade PCA

![PCA](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/pca_biplot.png)

  

**Explicação**:

O gráfico apresentado é um **Biplot de Análise de Componentes Principais (PCA)**. Este tipo de visualização tem como objetivo reduzir a dimensionalidade de um conjunto de dados complexo (neste caso, as permissões de aplicativos) para duas dimensões (Componente Principal 1 e Componente Principal 2), que capturam a maior parte da variabilidade dos dados.

O gráfico exibe simultaneamente:

1.  **As amostras (pontos):** Cada ponto representa um aplicativo, colorido de acordo com sua classe. Roxo (valor 0) indica um aplicativo **Benigno**, e amarelo (valor 1) indica **Malware**.
    
2.  **As variáveis originais (vetores):** As setas vermelhas partindo da origem são os vetores de carga, representando as permissões do sistema (features). A direção e o comprimento de cada vetor indicam como cada permissão influencia os dois componentes principais.
    

**Principais destaques:**

A análise do gráfico revela como as permissões (features) se relacionam com as classes de aplicativos (Benigno vs. Malware):

-   **Separação de Classes:** Existe uma tendência clara de separação entre as classes. Aplicativos **Malware (amarelo)** estão concentrados predominantemente no quadrante superior direito, indicando que possuem valores positivos tanto para a Componente Principal 1 quanto para a Componente Principal 2. Aplicativos **Benignos (roxo)** estão mais dispersos, mas com uma forte concentração no lado esquerdo do gráfico (valores negativos de Componente Principal 1).
    
-   **Influência das Permissões:** Os vetores de carga nos mostram as permissões mais influentes:
    
    -   **Fortemente associadas a Malware:** Permissões como `SEND_SMS`, `READ_PHONE_STATE`, `INSTALL_PACKAGES` e `WRITE_SMS` têm vetores que apontam para a mesma direção da concentração de malware (direita e para cima). Isso indica que essas permissões são fortes indicadores de um aplicativo malicioso.
        
    -   **Correlação entre Permissões:** Os vetores para `SEND_SMS`, `RECEIVE_SMS`, `READ_SMS` e `WRITE_SMS` estão muito próximos, apontando na mesma direção. Isso sugere que essas permissões são altamente correlacionadas, ou seja, se um aplicativo solicita uma delas, é muito provável que solicite as outras também.
        

**Interpretação:**

O biplot demonstra que as duas primeiras componentes principais conseguem capturar características importantes que distinguem aplicativos benignos de maliciosos.

A **Componente Principal 1** (eixo horizontal) parece ser um forte diferenciador geral. Valores positivos nesta componente, influenciados principalmente por permissões como `READ_PHONE_STATE` e `INSTALL_PACKAGES`, estão fortemente associados a malware.

A **Componente Principal 2** (eixo vertical), influenciada principalmente pela permissão `SEND_SMS`, ajuda a refinar essa separação, especialmente para o cluster de malware.

A sobreposição entre os pontos roxos e amarelos indica que apenas com estas duas componentes não é possível separar perfeitamente as duas classes, mas a tendência é evidente. O gráfico fornece insights valiosos para a engenharia de features, mostrando que um modelo de machine learning provavelmente atribuirá grande importância a permissões como `SEND_SMS` e `READ_PHONE_STATE` para detectar atividades maliciosas. Em resumo, a visualização confirma que o comportamento de solicitação de certas permissões é um fator crucial para a classificação de malware.

  

### 2.2 Mapa de calor contribuição das características PCA

![PCA](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/pca_components_20250701_212334.png)

  

**Explicação**:

O gráfico apresentado é um **mapa de calor (heatmap)** que detalha as contribuições (ou "cargas") de cada feature original (permissões de aplicativos, listadas no eixo Y) para cada uma das dez primeiras Componentes Principais (PCs), de PC 1 a PC 10 (listadas no eixo X).

Esta visualização nos permite entender a "composição" de cada componente principal. A intensidade e a cor de cada célula indicam a força e a direção da influência de uma feature sobre um componente:

-   **Cor Vermelha (valores positivos):** Indica uma contribuição positiva. Quando o valor de uma feature com carga positiva aumenta, o escore do componente principal também tende a aumentar.
    
-   **Cor Azul (valores negativos):** Indica uma contribuição negativa. Quando o valor de uma feature com carga negativa aumenta, o escore do componente principal tende a diminuir.
    
-   **Cor Clara/Branca (valores próximos de zero):** Indica que a feature tem pouca ou nenhuma influência sobre aquele componente específico.
    

**Principais destaques:**

O mapa de calor revela que cada componente principal é dominado por um pequeno conjunto de features correlacionadas, representando um tipo específico de comportamento ou funcionalidade do aplicativo.

-   **Componente Principal 1 (PC 1):** É fortemente dominada por duas features com cargas positivas muito altas: `ACCESS_NETWORK_STATE_1.0` e `INTERNET_1.0`. Todas as outras features têm uma contribuição quase nula para este componente.
    
-   **Componente Principal 2 (PC 2):** É majoritariamente definida pela permissão `SEND_SMS_1.0`, que possui a maior carga positiva. Outras permissões relacionadas a SMS (`RECEIVE_SMS` e `READ_SMS`) também contribuem positivamente, mas com menor intensidade.
    
-   **Componente Principal 3 (PC 3):** É primariamente influenciada positivamente pela permissão `READ_PHONE_STATE_1.0`.
    
-   **Componente Principal 4 (PC 4):** Possui uma forte carga negativa da feature `USE_CREDENTIALS_1.0`.
    
-   **Outros Componentes:** Cada componente subsequente é definido por outras features. Por exemplo, `PC 5` é influenciado por `WRITE_SETTINGS_1.0`, e `PC 8` é negativamente influenciado por `RESTART_PACKAGES_1.0`.
    

**Interpretação:**

Este mapa de calor é fundamental para interpretar o que cada componente principal representa em termos práticos. Em vez de analisar dezenas de permissões, podemos entender a variabilidade dos dados através de 10 "conceitos" ou "padrões de comportamento" independentes.

-   **PC 1 pode ser interpretado como o "Componente de Acesso à Internet"**. Aplicativos com pontuação alta neste componente são aqueles que fazem uso intensivo da rede.
    
-   **PC 2 representa a "Componente de Funcionalidade SMS"**. Aplicativos que enviam, recebem ou leem SMS terão uma pontuação alta neste componente. (Conectando com a análise anterior, esta era uma característica chave para identificar malware).
    
-   **PC 3 pode ser visto como a "Componente de Identificação do Dispositivo"**, já que `READ_PHONE_STATE` permite o acesso a informações como o IMEI do telefone.
    
-   **PC 4 representa um "Componente de Gerenciamento de Contas/Credenciais"**, e sua carga negativa sugere um comportamento distinto para apps que usam essa permissão.
    

Em suma, o gráfico "traduz" os componentes matemáticos abstratos (PC 1, PC 2, etc.) em comportamentos de aplicativos compreensíveis. Essa análise é extremamente útil para a engenharia de features e para a interpretabilidade de modelos de machine learning. Por exemplo, se um modelo de classificação identifica que PC 2 é um forte preditor de malware, este mapa de calor nos diz explicitamente que a capacidade de **enviar SMS** é a razão por trás dessa predição.

  

### 2.3 Distribuição de Classes

![DATA_SPLIT](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/train_test_distribution.png)

  
**Explicação**:
A figura abaixo apresenta a distribuição das classes (_Benign_  e  _Malware_) nos conjuntos de treinamento e teste. A visualização mostra o número de amostras de cada classe em ambos os conjuntos, permitindo avaliar se a divisão foi balanceada e representativa.

- **Treinamento**:  
  - Benign: **5991**  
  - Malware: **3482**  
  - Total: **9473** amostras  

- **Teste**:  
  - Benign: **2534**  
  - Malware: **1526**  
  - Total: **4060** amostras  

**Proporção (Benign:Malware)**: ~1.7:1 em ambos os conjuntos.  

**Principais Pontos**:  
✔ Divisão estratificada (proporção preservada).  
✔ Conjunto de treinamento maior para aprendizado adequado.
    
Essa distribuição é crucial para garantir que o modelo seja treinado e avaliado em dados que refletem a proporção real das classes, evitando viés. 

  

## 3. 🎯 Otimização de Modelo

  

### 3.1 Importância de Hiperparâmetros

![PARAM_IMPORTANCE](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/optuna_param_importance.png)

  

**Explicação**:
O gráfico apresenta a importância relativa dos hiperparâmetros no desempenho do modelo, conforme avaliado pelo framework Optuna. Os parâmetros são ordenados por sua influência no valor objetivo, onde valores mais altos indicam maior impacto.

**Principais destaques:**

Os hiperparâmetros mais significativos são:

-   **neighbors**  (importância ~0.35) - o parâmetro com maior influência
    
-   **nav_depth**  (importância ~0.25)
    
-   **leaf_size**  (importância ~0.20)
    
-   **nples_split**  (importância ~0.15)
    
-   **classifier**  (importância ~0.10)
    

**Interpretação:**  
Os parâmetros na extremidade direita do gráfico (com importância acima de 0.10) merecem atenção especial durante o ajuste fino do modelo, pois pequenas variações nestes podem impactar significativamente os resultados. Por outro lado, parâmetros com importância próxima de zero (como 'ic_params' e 'eaf_nodes') têm efeito mínimo e podem ser mantidos com valores padrão para simplificar o processo de otimização.
O gráfico de importância dos hiperparâmetros fornece insights sobre quais parâmetros do modelo exercem maior influência no desempenho, permitindo reduzir o espaço de busca e, consequentemente, tornar o processo de otimização mais eficiente.


  

### 3.2 Curva de Otimização

![OPTUNA](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/optuna_optimization_history.png)

  

**Explicação**:

O gráfico de histórico de otimização documenta a evolução do desempenho do modelo ao longo de sucessivas tentativas de ajuste de hiperparâmetros, revelando importantes padrões:

**Trajetória de Melhoria**

-   O processo iniciou com um desempenho modesto (0.65), característico de configurações não otimizadas
    
-   Observa-se uma rápida ascensão na fase inicial (0.65 → 0.80 em 5 tentativas), demonstrando a eficácia da abordagem de otimização
    
-   A fase intermediária (tentativas 5-15) apresenta ganhos incrementais, típico de processos de refinamento
    
-   O platô alcançado (0.90) após 15 tentativas sugere a exploração adequada do espaço de parâmetros
    

**Dinâmica de Convergência**

1.  **Fase Exploratória**  (0-5 tentativas): Melhorias rápidas indicam que o algoritmo encontrou rapidamente regiões promissoras no espaço de parâmetros
    
2.  **Fase de Refinamento**  (5-15 tentativas): Aperfeiçoamento gradual sugere ajustes finos nas combinações de parâmetros
    
3.  **Fase de Estabilização**  (pós-15 tentativas): Manutenção do desempenho máximo indica possível esgotamento das melhorias viáveis
    

**Interpretação Técnica**

-   A diferença entre os valores inicial (0.65) e final (0.90) representa um ganho de ~38% no desempenho
    
-   A estabilização precoce (em ~15 tentativas) pode sugerir:
    
    -   Eficiência do algoritmo de otimização
        
    -   Espaço de parâmetros bem definido
        
    -   Potencial para melhorias através da expansão do espaço de busca
        


Esta análise demonstra a efetividade do processo de otimização, com ganhos significativos de desempenho alcançados de forma eficiente, ao mesmo tempo que aponta oportunidades para refinamentos adicionais.

  

### 3.2 Coordenadas Paralelas

![OPTUNA](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/optuna_parallel_coordinate.png)

  

**Explicação**:

O gráfico de coordenadas paralelas revela as relações entre múltiplos hiperparâmetros e o desempenho do modelo (0.89 a 0.97), destacando combinações ótimas através de padrões visíveis nas linhas superiores - alguns parâmetros mostram forte correlação com bons resultados (valores concentrados em faixas específicas), enquanto outros apresentam variação aleatória, indicando menor influência, sugerindo que a otimização deve focar nos intervalos associados às melhores execuções e reduzir a busca em parâmetros menos impactantes.

  

## 4. 📈 Avaliação de Desempenho

  

### 4.1 Matriz de Confusão

![CONFMATRIX](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/confusion_matrix.png)

**Explicação**:

A matriz de confusão demonstra a capacidade do modelo em distinguir entre amostras benignas e maliciosas. Das **2.534** amostras benignas reais, **2.457** foram corretamente classificadas, com apenas **77 falsos positivos**, o que demonstra **alta especificidade**.Das **1.526** amostras maliciosas, **1.355** foram corretamente identificadas, com **171 falsos negativos**, revelando uma **boa sensibilidade**, embora ainda haja espaço para melhorias na detecção de ameaças. O desempenho indica que o modelo apresenta **baixo índice de alarmes falsos** e uma **eficácia significativa na identificação de malwares**, sendo adequado para cenários que exigem confiança tanto na detecção quanto na minimização de alertas indevidos.

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

  

-  **Benigno**: apresenta alta **revocação (0.970)**, o que indica que quase todos os aplicativos benignos foram corretamente identificados. A **precisão (0.935)** também é elevada, significando que a maioria das previsões como benignas realmente corresponde a essa classe. O **F1-Score (0.952)** resume esse bom equilíbrio entre precisão e recall.

-  **Malware**: tem uma **precisão ainda mais alta (0.946)**, o que é crucial em sistemas de segurança, pois minimiza o número de falsos positivos (benignos classificados como malware). A **revocação (0.888)**, embora ligeiramente inferior, ainda indica boa capacidade de detecção. O **F1-Score (0.916)** demonstra um desempenho sólido e consistente na identificação de ameaças.

  

Em conjunto, esses resultados sugerem que o modelo mantém **bom equilíbrio entre segurança (detecção de malware) e confiabilidade (baixo alarme falso)**, sendo apropriado para ambientes onde a minimização de riscos e ruído operacional é essencial.

  

### 4.4 Avaliação Por Classe

![PR](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/probability_distribution.png)

  

**Explicação**:

Este gráfico exibe a densidade das probabilidades preditas pelo modelo para cada classe (Benigno em azul, Malware em vermelho), com um **limiar de decisão fixado em 0.5** (linha tracejada).

  

Observações importantes:

  

- A maioria dos exemplos **benignos** concentra-se à esquerda do limiar (probabilidades próximas de 0), indicando alta confiança do modelo ao classificá-los corretamente como não maliciosos.

- De forma análoga, a maioria dos exemplos **maliciosos** se agrupa à direita do limiar (probabilidades próximas de 1), evidenciando também alta confiança na classificação como malware.

- A separação clara entre as duas distribuições sugere **alta discriminabilidade** do modelo, ou seja, baixa ambiguidade na predição.

- A baixa sobreposição entre as curvas reduz a taxa de erros de classificação, como falsos positivos e falsos negativos.

  

Esse comportamento é desejável em sistemas de detecção, pois reforça que o modelo não apenas acerta as classes, mas o faz com **alta confiança estatística**, tornando a ferramenta confiável para uso em ambientes críticos.

  

## 5. 🔍 Interpretabilidade

  

### 5.1 Análise SHAP Summary Plot

![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/shap_summary_plot_LGBMClassifier_20250701_211240.png)

  

**Explicação**:

Este gráfico resume a **influência de cada permissão Android** sobre as previsões do modelo, utilizando valores SHAP, que quantificam o impacto de cada feature na saída do classificador.

  

**Interpretação do Gráfico:**

  

- O eixo X representa o valor SHAP, ou seja, o **impacto individual** da feature na predição. Valores positivos empurram a previsão para “malware”, enquanto valores negativos favorecem a classe “benigno”.

- Cada ponto representa uma amostra; a cor indica o valor da feature (vermelho = valor alto / presente, azul = valor baixo / ausente).

- As permissões mais relevantes incluem:

-  `SEND_SMS_1.0`, `READ_PHONE_STATE_1.0`, `READ_SMS_1.0`: quando **ativas (vermelhas)**, têm forte impacto positivo na classificação como **malware**, sugerindo comportamento malicioso.

-  `GET_ACCOUNTS_1.0` e `ACCESS_NETWORK_STATE_1.0`: apresentam impacto misto, com comportamentos diferentes dependendo do contexto.

- Permissões como `RECEIVE_BOOT_COMPLETED_1.0` e `WRITE_HISTORY_BOOKMARKS_1.0` também contribuem, mas com menor intensidade.

  

Este gráfico permite concluir que o modelo **aprende padrões interpretáveis** e condizentes com práticas conhecidas de malware, reforçando a confiabilidade e **explicabilidade** do processo de decisão. Além disso, evidencia a importância de um subconjunto reduzido de permissões, o que pode auxiliar na **redução dimensional** e auditoria dos atributos usados.

  

### 5.2 Análise SHAP Force Plot

![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/shap_force_plot.png)

  

**Explicação**:

Este gráfico visualiza como os valores das features influenciaram uma predição específica do modelo. A previsão final (`f(x) = 5.94`) resulta da soma do valor base (média das predições) com os impactos acumulados de cada atributo.

  

**Interpretação:**

  

-  **Base value**: é o valor médio da saída do modelo sem considerar nenhuma feature (referência neutra).

-  **Setas vermelhas (→ higher)**: indicam features que **aumentaram** a probabilidade da amostra ser classificada como **malware**.

-  `SEND_SMS_1.0`, `ACCESS_NETWORK_STATE_1.0` e `GET_ACCOUNTS_1.0` contribuíram positivamente, impulsionando a predição para a classe maliciosa.

-  **Seta azul (→ lower)**: representa uma feature que **reduziu** essa probabilidade.

-  `WRITE_HISTORY_BOOKMARKS_1.0` teve um impacto negativo, atuando como um fator benigno.

  

O valor final de **5.94** está bem acima do limiar de decisão, reforçando que o modelo classificou esta instância com **alta confiança como malware**.

  

### 5.3 Análise importância das características LIME

![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/lime_feature_importance_20250701_232317.png)

  

**Explicação**:

O gráfico apresenta a contribuição individual de cada feature na decisão do modelo para uma **amostra específica**, permitindo uma análise **local** da explicação. Os pesos indicam o quanto cada atributo influenciou a classificação:

  

-  **Barras azuis (positivas)**: características que **favoreceram a predição como malware**.

- Destaques:

-  `SEND_SMS > -0.56` (peso: +0.242)

-  `READ_PHONE_STATE > 0.76` (peso: +0.241)

-  `INTERNET > 0.38` (peso: +0.241)

→ A presença ou valores altos dessas permissões aumentaram significativamente a probabilidade de a amostra ser classificada como maliciosa.

-  **Barras vermelhas (negativas)**: características que **atuaram contra a classificação como malware**, ou seja, aproximaram a instância da classe benigna.

- Destaques:

-  `WRITE_HISTORY_BOOKMARKS <= -0.22` (peso: –0.224)

-  `GET_ACCOUNTS <= 1.53` (peso: –0.188)

-  `READ_SMS <= -0.48` (peso: –0.119)

→ Esses atributos, ao estarem ausentes ou abaixo de determinado valor, indicaram comportamento benigno ao modelo.

  

Este gráfico complementa a explicação global do SHAP ao mostrar **como o modelo tomou uma decisão em um caso concreto**, oferecendo uma forma interpretável e confiável de justificar decisões individuais — fundamental em contextos como cibersegurança e auditoria.

  

### 5.4 Análise Probabilidade de Predição LIME

![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/lime_interpretability.png)

  

**Insights**:

O gráfico mostra como as **features ativas** (valores fornecidos à direita) impactaram a predição do modelo (barra central), contribuindo para a classificação como **malware (classe 1)** ou **benigno (classe 0)**.

  

#### 🔶 Principais contribuições para **classe 1 (malware)**:

  

-  `READ_PHONE_STATE = 0.76`

-  `INTERNET = 0.39`

-  `SEND_SMS = 1.78`

-  `CHANGE_WIFI_MULTICAST_STATE = -0.12`

  

Estas permissões são **fortes indicativos de comportamento malicioso** e empurraram a predição para a classe “malware”.

  

#### 🔷 Principais contribuições para **classe 0 (benigno)**:

  

-  `WRITE_HISTORY_BOOKMARKS = -0.30`

-  `GET_ACCOUNTS = -0.64`

-  `READ_CALL_LOG = -0.12`

-  `USE_CREDENTIALS = -0.33`

-  `READ_SMS = -0.48`

-  `MASTER_CLEAR = -0.11`

  

Essas permissões, por estarem ausentes ou com valor baixo, puxaram a predição em direção à classe benigna. Mesmo assim, o modelo atribuiu **alta probabilidade (≈ 1.00)** para a classe malware, dado o maior peso das permissões maliciosas.

### 5.4 Análise da Árvore de Decisão

![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/decision_tree_plot_ExtraTreesClassifier_20250701_232317.png)
  A imagem exibe a estrutura de uma única **árvore de decisão**, que é a primeira árvore de um modelo de `ExtraTreesClassifier`. Este tipo de modelo de machine learning cria uma "floresta" de múltiplas árvores e combina seus resultados, mas a visualização de uma única árvore nos permite entender a lógica de classificação que o modelo aprendeu.

A árvore funciona como um fluxograma de decisões:

-   **Nós (caixas):** Cada nó interno representa uma pergunta sobre uma feature específica (uma permissão do Android).
    
-   **Ramos (setas):** As setas indicam o caminho a seguir com base na resposta ("sim" ou "não", ou, mais precisamente, `True` ou `False` para a condição).
    
-   **Folhas (nós finais):** Os nós na base da árvore representam a decisão final de classificação.
    

Dentro de cada nó, temos as seguintes informações:

-   **Condição de divisão:** A regra usada para dividir os dados (ex: `SEND_SMS_1.0 <= 0.852`).
    
-   **`entropy`:** Uma medida de impureza do nó. Um valor de 0 significa que o nó é "puro" (todas as amostras pertencem à mesma classe).
    
-   **`samples`:** O número de aplicativos que chegaram a este nó.
    
-   **`value`:** A distribuição das classes neste nó (ex: `[5991, 3482]` significa 5991 amostras da classe 0 e 3482 da classe 1).
    
-   **`class`:** A classe majoritária no nó. A cor do nó (laranja para classe 0 - Benigno, azul para classe 1 - Malware) reflete essa maioria.
    

**Principais destaques:**

A análise da estrutura da árvore revela uma clara hierarquia na importância das permissões para a detecção de malware.

-   **Feature Mais Importante:** A primeira decisão, no topo da árvore (nó raiz), é baseada na permissão `SEND_SMS_1.0`. Isso significa que, de todas as permissões disponíveis, esta é a que melhor consegue separar os aplicativos benignos dos maliciosos no primeiro passo.
    
-   **Caminho Crítico para Malware:** O caminho à direita, onde `SEND_SMS_1.0` é `True` (maior que 0.852), leva a um nó que é predominantemente **Malware (azul)**, com 1890 amostras de malware contra 343 benignas. Isso indica que a solicitação da permissão para enviar SMS é um fortíssimo indicador de malícia.
    
-   **Hierarquia de Permissões:** Para os aplicativos que _não_ solicitam `SEND_SMS` (o caminho da esquerda), a próxima pergunta mais importante é sobre a permissão `READ_PHONE_STATE_1.0`. Se a resposta for sim, a suspeita aumenta. Se for não, o modelo continua a perguntar sobre outras permissões como `GET_ACCOUNTS_1.0`.
    
-   **Pureza das Folhas:** O objetivo da árvore é terminar em folhas que sejam o mais "puras" possível (predominantemente de uma única cor). Vemos que a árvore consegue isolar grupos de malware (folhas azuis) e de aplicativos benignos (folhas laranjas) com razoável sucesso.
    

**Interpretação:**

Esta árvore de decisão torna o processo de classificação do modelo transparente e interpretável. Ela essencialmente cria um conjunto de regras "se-então" para identificar malware.

A lógica do modelo pode ser lida como um processo de triagem:

1.  **Primeiro, verifique se o aplicativo envia SMS.** Se sim, a probabilidade de ser malware é muito alta.
    
2.  **Se não, verifique se ele lê o estado do telefone.** Esta é a segunda bandeira vermelha mais importante.
    
3.  **Se não, verifique se ele acessa as contas do usuário.** E assim por diante.
    

As permissões que aparecem nos níveis superiores da árvore (`SEND_SMS`, `READ_PHONE_STATE`, `GET_ACCOUNTS`) são as mais impactantes para o modelo. Esta conclusão está perfeitamente alinhada com as análises de PCA anteriores, que também destacaram a importância dessas mesmas features. A árvore, no entanto, nos dá uma visão mais direta e processual, mostrando a ordem e a lógica exata das decisões que levam a uma classificação final de **Benigno** ou **Malware**.

## 6. Performance geral do Pipeline

  

### 5.2 Desempenho Tempo x RAM

![SHAP](https://raw.githubusercontent.com/JonerMello/MH-AutoML/main/doc/artifacts/performance_metrics.jpg)

  


A imagem apresentada é um gráfico de barras e linhas que detalha as métricas de performance (tempo de execução em segundos e consumo de memória RAM em MB) para cada etapa de um pipeline de Machine Learning (ML).

**Análise do Gráfico:**

O gráfico possui dois eixos Y:

-   **Eixo Y Esquerdo (Azul):** Representa o "Elapsed Time (seconds)" (Tempo Decorrido em segundos) para cada etapa. As barras azuis mostram esses valores.
    
-   **Eixo Y Direito (Verde):** Representa o "Memory Usage (MB)" (Uso de Memória em MB). A linha verde com marcadores circulares mostra esses valores.
    

As etapas do pipeline de ML estão no **Eixo X**, rotuladas como "Step Name":

1.  **Data Info**
    
2.  **Preprocessing**
    
3.  **Feature Engineering**
    
4.  **Hyperparameter**
    
5.  **Interpretability**
    
6.  **Evaluation**
    

Vamos analisar cada etapa:

-   **Data Info:**
    
    -   **Tempo de Execução:** 5.57 segundos (barra azul)
        
    -   **Uso de Memória:** 4.99 MB (marcador verde)
        
    -   Esta etapa inicial para obter informações sobre os dados é relativamente rápida e consome pouca memória.
        
-   **Preprocessing (Pré-processamento):**
    
    -   **Tempo de Execução:** 4.50 segundos (barra azul)
        
    -   **Uso de Memória:** 98.17 MB (marcador verde)
        
    -   Embora o tempo de execução seja menor que o de "Data Info", o consumo de memória aumenta significativamente, o que é comum em etapas de limpeza, tratamento de valores ausentes ou normalização de dados.
        
-   **Feature Eng (Engenharia de Características):**
    
    -   **Tempo de Execução:** 0.92 segundos (barra azul)
        
    -   **Uso de Memória:** 23.74 MB (marcador verde)
        
    -   Esta é a etapa mais rápida em termos de tempo de execução e possui um consumo de memória moderado. A criação ou transformação de características pode ser eficiente em alguns casos.
        
-   **Hyperparameter (Otimização de Hiperparâmetros):**
    
    -   **Tempo de Execução:** 38.95 segundos (barra azul)
        
    -   **Uso de Memória:** 39.57 MB (marcador verde)
        
    -   Esta é a etapa mais demorada do pipeline, consumindo quase 39 segundos. Isso é esperado, pois a otimização de hiperparâmetros (por exemplo, busca em grade, busca aleatória) envolve a treinamento e avaliação de múltiplos modelos. O consumo de memória é relativamente baixo em comparação com o pré-processamento.
        
-   **Interpretability (Interpretabilidade):**
    
    -   **Tempo de Execução:** 4.51 segundos (barra azul)
        
    -   **Uso de Memória:** 354.16 MB (marcador verde)
        
    -   Embora o tempo de execução seja razoável, esta etapa apresenta o **maior consumo de memória RAM** de todo o pipeline, atingindo mais de 354 MB. Isso pode indicar o uso de algoritmos complexos para explicar as previsões do modelo, que podem exigir a manipulação de grandes estruturas de dados.
        
-   **Evaluation (Avaliação):**
    
    -   **Tempo de Execução:** 8.19 segundos (barra azul)
        
    -   **Uso de Memória:** 138.94 MB (marcador verde)
        
    -   A etapa final, responsável por avaliar o desempenho do modelo, leva um tempo considerável e tem um consumo de memória elevado, embora menor que a etapa de interpretabilidade. Isso pode envolver o cálculo de diversas métricas e a geração de relatórios.
        

**Pontos Chave e Insights:**

-   **Gargalo de Tempo:** A etapa de "Hyperparameter" é o principal gargalo em termos de tempo de execução.
    
-   **Gargalo de Memória:** A etapa de "Interpretability" é o principal gargalo em termos de consumo de memória RAM.
    
-   **Trade-offs:** É interessante observar que as etapas mais demoradas nem sempre são as que mais consomem memória, e vice-versa. Por exemplo, "Hyperparameter" é lenta mas não é a que mais usa memória, enquanto "Interpretability" consome muita memória em um tempo moderado.
    
-   **Otimização:** Este tipo de gráfico é crucial para identificar áreas onde a otimização pode ser mais eficaz. Por exemplo, se a memória for um problema, focar em reduzir o consumo na etapa de "Interpretability" seria prioritário. Se o tempo for crítico, otimizar a "Hyperparameter" seria essencial.
    

Em resumo, a imagem fornece uma visão clara e quantitativa do desempenho de cada componente do pipeline de ML, permitindo que os desenvolvedores e engenheiros de ML identifiquem e abordem ineficiências em termos de tempo e uso de recursos.

  

## 7. 📝 Considerações Finais

O pipeline de Machine Learning "MH-AutoML" demonstra uma robustez notável na detecção de malwares em aplicações Android, com base nas permissões solicitadas. A análise detalhada dos artefatos gerados em cada etapa oferece insights valiosos sobre o comportamento do modelo e a importância das características.

**Pontos Fortes do Pipeline:**

-   **Alta Performance:** As métricas de avaliação, como AUC (0.992 ±0.03) e F1-Score Balanceado (0.968), indicam um desempenho excepcional na classificação, minimizando tanto falsos positivos quanto falsos negativos. A matriz de confusão e as curvas ROC/Precision-Recall corroboram a capacidade discriminativa do modelo.
    
-   **Interpretabilidade Aprofundada:** O uso de ferramentas como SHAP e LIME é crucial. O SHAP Summary Plot e o Force Plot revelam que permissões como `SEND_SMS`, `READ_PHONE_STATE` e `INTERNET` são os principais indicadores de comportamento malicioso, o que é consistente com as expectativas de segurança de aplicativos. A análise LIME complementa, fornecendo explicações localizadas para decisões específicas, essencial para a confiança em cenários críticos como cibersegurança. A visualização da árvore de decisão de `ExtraTreesClassifier` também oferece uma interpretação clara das regras de classificação aprendidas pelo modelo.
    
-   **Engenharia de Features Eficiente:** As etapas de seleção de características (LASSO e ANOVA) e redução de dimensionalidade (PCA) foram bem aplicadas. O biplot do PCA demonstrou uma separação clara entre as classes benignas e maliciosas, reforçando que as permissões são características discriminativas. O heatmap das componentes principais "traduz" essas componentes em padrões de comportamento de aplicativos, como "Acesso à Internet" (PC 1) e "Funcionalidade SMS" (PC 2).
    
-   **Otimização de Hiperparâmetros Eficaz:** O processo de otimização de hiperparâmetros, utilizando Optuna, demonstrou eficiência ao alcançar um platô de desempenho elevado em poucas tentativas. A análise de importância dos hiperparâmetros ("neighbors", "nav_depth", "leaf_size", "nples_split" e "classifier") direciona o ajuste fino, e as coordenadas paralelas mostram as combinações que levam aos melhores resultados.
    
-   **Gestão de Dados:** A análise de valores faltantes e a distribuição de classes nos conjuntos de treinamento e teste (`~1.7:1` Benigno:Malware) indicam uma preparação de dados cuidadosa, com divisão estratificada para evitar vieses.
    

**Desafios e Oportunidades de Otimização:**

-   **Gargalo de Memória na Interpretabilidade:** Conforme a análise de desempenho do pipeline, a etapa de "Interpretability" consome a maior quantidade de memória RAM (354.16 MB). Embora seja fundamental para a explicabilidade, otimizações nesta fase (e.g., amostragem, técnicas mais eficientes) podem ser exploradas para reduzir o consumo de recursos, especialmente em ambientes com restrição de memória.
    
-   **Tempo de Execução na Otimização de Hiperparâmetros:** A etapa de "Hyperparameter" é a mais demorada (38.95 segundos). Para grandes conjuntos de dados ou otimizações mais extensas, métodos como otimização bayesiana mais avançada ou a paralelização da busca podem ser considerados para acelerar o processo.
    
    

Em suma, o MH-AutoML apresenta um framework robusto e bem-sucedido para a detecção de malware, com um equilíbrio notável entre alta performance preditiva e transparência em suas decisões. As considerações sobre o consumo de recursos indicam áreas potenciais para otimização contínua, garantindo que o pipeline não apenas seja eficaz, mas também eficiente em sua execução.