FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

USER root

# Atualiza os pacotes e instala dependências necessárias
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.8 python3.8-venv python3.8-dev python3.8-distutils python3-pip
RUN apt-get -y install unzip

# Configura o bash como shell padrão
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Configura fuso horário para evitar prompts interativos
RUN ln -fs /usr/share/zoneinfo/America/Sao_Paulo /etc/localtime
RUN apt-get install -y tzdata
RUN dpkg-reconfigure --frontend noninteractive tzdata

# Copia o código para o diretório de trabalho
WORKDIR /MH-AutoML
COPY . ./

WORKDIR /MH-AutoML/src

# Verifica e instala o pip3 se necessário
RUN which pip || apt-get install -y python3-pip

# Instala e configura o pipenv
RUN pip install pipenv
RUN pipenv lock
#RUN pipenv install -r requirements.txt
RUN pip install setuptools wheel

# Executa comandos de build necessários (se aplicável)
RUN python3.8 setup.py sdist bdist_wheel
# Instala o pacote Python
RUN python3.8 -m pip install .
# Exponha a porta 5000 para o MLflow
EXPOSE 5000

# Comando para executar o script Python principal
CMD ["python3.8", "view/main.py", "-d", "../Datasets/dataset_sujo.csv", "-l", "class"]
