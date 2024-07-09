
# MH-AutoML_V2

O ** framework MH-AutoML** está em fase de desenvolvimento, com foco especial em transparência, interpretabilidade, gerenciamento de experimentos e versionamento de modelos. O framework abrange todas as fases de um pipeline padrão de aprendizado de máquina, incluindo pré-processamento, engenharia de características, seleção de modelo, otimização de modelo e interpretabilidade do modelo alem do gerenciamento do ciclo de vidas de ML.
![**Arquitetura do framework**](https://github.com/Malware-Hunter/MotoDroidV2/blob/main/imgs/fluxo-MH-AutoML.png)

## 1. Selos Considerados
Os autores julgam como considerados no processo de avaliação os seguintes selos:

**Selo D - Artefatos Disponíveis**:
Justificativa: Repositório  anônimo disponível no GitHub público com documentação da ferramenta e módulos.

**Selo F - Artefatos Funcionais**:
Justificativa: Artefatos funcionais e testados em Ubuntu 22.04.04, Windows 10.

**Selo R - Artefatos Reprodutíveis**:
Justificativa: São disponibilizados scripts para reprodução dos experimentos detalhados no artigo.[script1]().

**Selo S- Artefatos Sustentáveis**:
Justificativa: Código inteligível e acompanhado com boa documentação. A versão preliminar encontra-se em: [Fixa técnica](https://github.com/SBSegSF24/MH-AutoML/tree/main/Documenta%C3%A7%C3%A3o). A documentação cobre quesitos básicos de engenharia de software, analise, projeto, desenvolvimento e teste.


## 2. Informações básicas
Os códigos utilizados para a execução ferramenta MH-AutoML, estão disponibilizados no repositório GitHub https://github.com/SBSegSF24/MH-AutoML/tree/main. Nesse repositório encontram-se um README.md contendo informações sobre o fluxo de execução da ferramenta, configuração, parâmetros de entrada e instalação nos seguintes ambientes :

-*OS* (testado em  Ubuntu 22.04 com Python 3.8.10 e Windows com 10 3.9.13)

-*Containers* Docker (testado em Docker version 24.0.7, build 24.0.7-ubuntu 20.04.1)


### 2.1. Dependências
Testamos o código da ferramenta com as seguintes versões Python:
- Python 3.8.10
- Python 3.9.13

O código da MH-autoML possui dependências com diversos pacotes e bibliotecas Python.
Entre elas, as principais são:
numpy 1.22, pandas 1.4.4, scikit-learn 1.1.1. e mlflow 2.11.3, shap 0.46.0, lime 0.2.0.1, lightgbm 4.4.0, catboost 1.2.5.
Ademais, a lista extensa das dependências encontra-se no arquivo [setup.py.](https://github.com/SBSegSF24/MH-AutoML/blob/main/src/setup.py)



### Pré-Configurações

Clone o repositório:
git clone https://github.com/Malware-Hunter/MotoDroidV2.git 

###  Instalação

Para instalar os pacotes necessários você deve estar em ***src***
```bash
pip  install  .
```

### Execução demo:
Teste funcional rápido utilizando o dataset dataset_sujo.csv, com 15000 amostras e 51 características, o dataset tem valores ausentes, strings, NaN gerado aleatoriamente. O teste mínimo leva 2 minutos num computador Core i7 com 32GB RAM.
Para executar a ferramenta você deve chamar a classe principal **main.py**.

```bash

python3  view/main.py  -d  Datasets/dataset_sujo.csv  -l  class

```

### Instalação via docker

- 1 Gerando a imagem docker.

```bash

sudo  docker  build  -t  mh-automl-image  .

```

- 2 Executando a imagem docker.

```bash

sudo  docker  run  -it  --name  mh-automl-container  mh-automl-image  /bin/bash

```

- 3 Executando a ferramenta no docker.

  

```bash

python3.8  view/main.py  -d  Datasets/dataset_sujo.csv  -l  class

```

###  Significado das flags

- -d Dataset a ser utilizado

- -l Nome da coluna de classificação
A instalação manual e em outros ambientes está detalhada no README.md do repositório GitHub.

## 4. Datasets
O diretório [Datasets](https://github.com/SBSegSF24/MH-AutoML/tree/main/src/Datasets) contém  os datasets como:
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
-   Storage: 2GB+ de espaço disponível
-   MS-Windows 10 64 bit ou Ubuntu 20.04.


## 7. Experimentos
Para reproduzir os experimentos executados no artigo utilize o seguinte comando:
