# MH-AutoML_V2
O ** framework MH-AutoML** está em fase de desenvolvimento, com foco especial em transparência, interpretabilidade, gerenciamento de experimentos e versionamento de modelos. O framework abrange todas as fases de um pipeline padrão de aprendizado de máquina, incluindo pré-processamento, engenharia de características, seleção de modelo, otimização de modelo e interpretabilidade do modelo alem do gerenciamento do ciclo de vidas de ML.
![**Arquitetura do framework**](https://github.com/Malware-Hunter/MotoDroidV2/blob/main/imgs/fluxo-MH-AutoML.png)

# 📦 Dependências
- Python 3.8.10
- Python 3.9.13
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

# Download

Clone o repositório:
```bash
git clone https://github.com/SBSegSF24/MH-AutoML.git 
```
#  Instalação e execução local da demo:
Teste funcional rápido utilizando o dataset dataset_sujo.csv, com 15000 amostras e 51 características, o dataset tem valores ausentes, strings, NaN gerado aleatoriamente. O teste mínimo leva 2 minutos num computador Core i7 com 32GB RAM.
Para executar a ferramenta você deve chamar a classe principal **./run_demo.sh**.

```bash
./run_demo.sh
```


# ⚙️ Instalação e execução via docker
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
- Python 3.8.10


# 🚀 Executando com os datasets do estudo

Para executar os experimento basta executar a ferramenta com os mesmos datasets utilizado no estudo, utilize o parâmetro `-d` para alterar o dataset utilizado. Abaixo está um exemplo de execução:

### Exemplo de execução
```bash
python3.8 view/main.py -d /Datasets/adroit.csv -l class
```

Substitua `/Datasets/adroit.csv` pelo caminho do dataset desejado. Aqui está a lista dos demais datasets disponíveis:

- androcrawl.csv
- android_permissions.csv
- dataset_sujo.csv
- defensedroid_prs.csv
- drebin215.csv
- kronodroid_emulador.csv
- kronodroid_real.csv
