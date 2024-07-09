# MH-AutoML_V2
O ** framework MH-AutoML** está em fase de desenvolvimento, com foco especial em transparência, interpretabilidade, gerenciamento de experimentos e versionamento de modelos. O framework abrange todas as fases de um pipeline padrão de aprendizado de máquina, incluindo pré-processamento, engenharia de características, seleção de modelo, otimização de modelo e interpretabilidade do modelo alem do gerenciamento do ciclo de vidas de ML.
![**Arquitetura do framework**](https://github.com/Malware-Hunter/MotoDroidV2/blob/main/imgs/fluxo-MH-AutoML.png)

# 📦 Dependências
- python ==3.5 á 3.9
- pandas==1.3.3
- numpy==1.20.1
- scipy==1.7.1
- scikit-learn==1.1.1
- lightgbm==4.0.6
- catboost==1.0.1
- optuna==2.10.1
- plotly==5.3.1
- shap==0.39.0
- lime==0.2.0.1
- mlflow==2.11.3

# 📋 Pré-Configurações

Clone o repositório:

    git clone https://github.com/Malware-Hunter/MotoDroidV2.git


# ⚙️ Instalação
Para instalar os pacotes necessários você deve estar em ***src*** 

```bash
pip install .
```

# 🚀 Execução demo:
Para executar a ferramenta voce deve chamar a classe principal main.py.
```bash
 python3 view/main.py -d Datasets/dataset_sujo.csv -l class
```

# ⚙️ Instalação via docker
- 1 Gerando a imagem docker.
```bash
sudo docker build -t mh-automl-image .
```
- 2 Executando a imagem docker.
```bash
sudo docker run -it --name mh-automl-container mh-automl-image /bin/bash
```
- 3 Executando a ferramenta no docker.

```bash
python3.8 view/main.py -d Datasets/dataset_sujo.csv -l class
```
## 🏷️ Significado das flags 
- -d Dataset a ser utilizado
- -l  Nome da coluna de classificação 

# 🖥️ Requisitos de hardware recomendados:
- CPU: 4+ cores
- RAM: 16GB+
- Storage: 2GB+ de expaço disponivel
- MS-Windows 10 64 bit ou Ubuntu 20.04.

# 🚀 Executando com os datasets do estudo


### adroit.csv
```bash
python3 main.py -d ../Datasets/adroit.csv -l class
```

### androcrawl.csv
```bash
python3 main.py -d ../Datasets/androcrawl.csv -l class
```

### android_permissions.csv
```bash
python3 main.py -d ../Datasets/android_permissions.csv -l class
```

### dataset_sujo.csv
```bash
python3 main.py -d ../Datasets/dataset_sujo.csv -l class
```

### defensedroid_prs.csv
```bash
python3 main.py -d ../Datasets/defensedroid_prs.csv -l class
```

### drebin215.csv
```bash
python3 main.py -d ../Datasets/drebin215.csv -l class
```

### kronodroid_emulador.csv
```bash
python3 main.py -d ../Datasets/kronodroid_emulador.csv -l class
```

### kronodroid_real.csv
```bash
python3 main.py -d ../Datasets/kronodroid_real.csv -l class
```
