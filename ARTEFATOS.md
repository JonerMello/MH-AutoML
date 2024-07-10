
# Artefatos apêndice SBSeg24/SF: #243362: MH-AutoML: Transparência, Interpretabilidade e Desempenho na Detecção de Malware Android 

Neste trabalho apresentamos a MH-AutoML, uma ferramenta de AutoML especializada para o domínio de detecção de malware Android. Diferentemente de outras ferramentas de AutoML, a MH-AutoML incorpora recursos importantes de transparência e interpretabilidade em todos os estágios do pipeline. A ferramenta incorpora também métodos de seleção de características específicos ao domínio em questão, bem como otimizações de hiperparâmetros que levam a resultados muitos bons, como modelos preditivos com 95% de recall, a um custo computacional relativamente baixo.

## 1. Selos Considerados
Os autores julgam como considerados no processo de avaliação os seguintes selos:

**Selo D - Artefatos Disponíveis**:
Justificativa: Repositório público  disponível no GitHub com documentação da ferramenta e módulos.

**Selo F - Artefatos Funcionais**:
Justificativa: Artefatos funcionais e testados em Ubuntu 22.04.04 e Windows 10.

**Selo R - Artefatos Reprodutíveis**:
Justificativa: São disponibilizados scripts para reprodução dos experimentos detalhados no artigo.

**Selo S- Artefatos Sustentáveis**:
Justificativa: Código inteligível e acompanhado com boa documentação. A documentação cobre quesitos básicos de engenharia de software, analise, projeto, desenvolvimento e teste.


## 2. Informações básicas
Os códigos utilizados para a execução ferramenta MH-AutoML, estão disponibilizados no repositório GitHub [https://github.com/SBSegSF24/MH-AutoML](https://github.com/SBSegSF24/MH-AutoML). Nesse repositório há um README.md contendo informações sobre o fluxo de execução da ferramenta, configuração, parâmetros de entrada e instalação nos seguintes ambientes :

-*OS* (testado em  Ubuntu 22.04 com Python 3.8.10 e Windows com 10 3.9.13)

-*Containers* Docker (testado em Docker version 24.0.7, build 24.0.7-ubuntu 20.04.1)


## 3. Instalação 

Testamos o código da ferramenta com as seguintes versões Python:

- Python 3.8.10

- Python 3.9.13

O código da MH-autoML possui dependências com diversos pacotes e bibliotecas Python.
Entre elas, as principais são:
numpy 1.22, pandas 1.4.4, scikit-learn 1.1.1. e mlflow 2.11.3, shap 0.46.0, lime 0.2.0.1, lightgbm 4.4.0, catboost 1.2.5.
Ademais, a lista extensa das dependências encontra-se no arquivo **setup.py** no GitHub.

Instruções completas de instalação e execução está disponíveis no **README.md** no GitHub.

## 4. Datasets

O diretório **Datasets** contém conjuntos de dados como:

- adroit.csv

- androcrawl.csv

- android_permissions.csv

- dataset_sujo.csv

- defensedroid_prs.csv

- drebin215.csv

- kronodroid_emulador.csv

- kronodroid_real.csv


## 5. Ambiente de testes

A ferramenta foi testada nos seguintes ambientes: 

-   CPU: 4+ cores

-   RAM: 16GB+

-   Disco: 2GB+ de espaço disponível

-   MS-Windows 10 64 bit ou Ubuntu 20.04.

## 6. Teste mínimo

A execução do teste mínimo funcional está documentada no **README.md**.

## 7. Experimentos

Há instruções de reprodução dos experimentos no **README.md** do GitHub.
