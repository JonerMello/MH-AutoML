
# MH-AutoML: Transparência, Interpretabilidade e Desempenho na Detecção de Malware Android 

Neste trabalho apresentamos a MH-AutoML, uma ferramenta de AutoML especializada para o domínio de detecção de malware Android. Diferentemente de outras ferramentas de AutoML, a MH-AutoML incorpora recursos importantes de transparência e interpretabilidade em todos os estágios do pipeline. A ferramenta incorpora também métodos de seleção de características específicos ao domínio em questão, bem como otimizações de hiperparâmetros que levam a resultados muitos bons, como modelos preditivos com 95% de recall, a um custo computacional relativamente baixo.

![**Arquitetura do framework**](https://raw.githubusercontent.com/Lost-User-24/MH-AutoML/main/pipeline/fluxo-MH-AutoML.png)

# 📦 Dependências
- Python 3.8.10 Ubuntu 20.04.04
- Python 3.9.13 Windows 10
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
#  Instalação e execução demo:
Teste funcional rápido utilizando o dataset dataset_sujo.csv, com 15000 amostras e 51 características, o dataset tem valores ausentes, strings, NaN gerado aleatoriamente. O teste mínimo leva 5 minutos num computador Core i7 com 32GB RAM.
Para executar a ferramenta você deve chamar a classe principal **./run_demo.sh**.

- 1 Prepare seu ambiente executando ./config.sh
*Caso tenha problema de permissão negada utilize o comando:*
```bash
chmod +x config.sh
chmod +x run_demo.sh
chmod +x run_reproducao_mh_automl.sh
```

O comando config.sh vai BAIXAR E INSTALAR o python3.8.10 em seu ambiente para que a ferramenta execute sem algum problema de compatibilidade.

```bash
./config.sh
```
- 2 Carrege as dependencias e execute a ferramenta com ./run_demo.sh
```bash
./run_demo.sh
```

# ⚙️ Instalação e execução via docker
- 1 Gerando a imagem docker.
```bash
sudo docker build -t mhautoml:latest .
```
- 2 Executando a imagem docker.
```bash
sudo docker run -v $(readlink -f . ):/mhautoml -it mhautoml
```

## 🏷️ Significado das flags 
- -d Dataset a ser utilizado
- -l  Nome da coluna de classificação 

# 🖥️ Requisitos recomendados:
- CPU: 4+ cores
- RAM: 16GB+
- Storage: 2GB+ de expaço disponivel
- MS-Windows 10 ou 11 64 bit ou Ubuntu 20.04.
- Python 3.8.10

# Ambientes testados
Hardware: Intel(R) Core(TM) i7-1185G7, 32GB RAM. Software: Ubuntu 22.04.4 LTS, Linux version 5.15.153.1-microsoft-standard-WSL2, Docker version 24.0.7 (build ced0996), Python 3.8.10

Hardware: Intel Core i7-10700, 8 cores, 16 GB RAM. Software: Ubuntu 24.02 LTS, Python 3.12.3,Docker 26.1.4

Hardware: Intel(R) Core(TM) i7-1185G7, 32GB RAM. Software: Windows 11 Pro compilação 22631.3880, Python 3.9.13

# 🚀 Reprodução do estudo

Para executar os experimento com a ferramenta MH-AutoML basta executar o script run_experimento.sh, esse script executa a ferramenta para cada dataset utilizado no experimento. Abaixo está o comando de execução:
*Para execução completa pode levar mais de 2 horas nas segintes configurações:*
Hardware: Intel(R) Core(TM) i7-1185G7, 32GB RAM. Software: Ubuntu 22.04.4 LTS, Linux version 5.15.153.1-microsoft-standard-WSL2, Docker version 24.0.7 (build ced0996), Python 3.8.10

```bash
./run_reproducao_mh_automl.sh
```
Para executar os experimentos com [AutoGluon](https://github.com/autogluon/autogluon), [AutoPytorch](https://github.com/automl/Auto-PyTorch), [Auto-Sklearn](https://github.com/automl/auto-sklearn), [TPOT](https://github.com/EpistasisLab/tpot), [MLJAR](https://github.com/mljar/mljar-supervised), [HyperGBM](https://github.com/DataCanvasIO/HyperGBM) e [LightAutoML](https://github.com/sb-ai-lab/LightAutoML) em todos os datasets, use o script run_tools_docker.sh, o script irá fazer a instalação e execução de todas ferramentas necessárias no Docker.

```bash
sudo ./run_tools_docker.sh <output_directory>
```

Para uma demonstração no dataset Adroit, use o script run_tools_docker.sh com o argumento "--d":

```bash
sudo ./run_tools_docker.sh <output_directory> --d 
```

*Para execução completa pode levar aproximadamente 28 horas e para a demonstração aproximadamente 3 horas nas segintes configurações:*
Hardware: Intel(R) Core(TM) i7-1185G7, 32GB RAM. Software: Ubuntu 22.04.4 LTS, Docker version 24.0.7 (build ced0996), Python 3.10.12




