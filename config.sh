#!/bin/bash

# Verificar se o Python 3.8 já está instalado
if command -v python3.8.10 &>/dev/null; then
    echo "Python 3.8 já está instalado."
else
    # Mensagem de alerta para o usuário
    read -p "Uma nova versão do Python (3.8.10) será instalada. Deseja continuar? (s/n): " resposta

    if [[ "$resposta" =~ ^[Ss]$ ]]; then
        # Atualizar os pacotes do sistema
        sudo apt update

        # Instalar as dependências necessárias
        sudo apt install -y \
            build-essential \
            zlib1g-dev \
            libncurses5-dev \
            libgdbm-dev \
            libnss3-dev \
            libssl-dev \
            libreadline-dev \
            libffi-dev \
            libsqlite3-dev \
            wget \
            libbz2-dev \
            liblzma-dev

        # Baixar o código fonte do Python 3.8.10
        cd /tmp
        wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz

        # Extrair o arquivo baixado
        tar -xf Python-3.8.10.tgz

        # Navegar até o diretório extraído
        cd Python-3.8.10

        # Configurar o script de instalação
        ./configure --enable-optimizations

        # Compilar e instalar o Python
        make -j $(nproc)
        sudo make altinstall

        # Verificar a instalação
        python3.8 --version
    else
        echo "Instalação do Python 3.8.10 cancelada pelo usuário."
        exit 1
    fi
fi

# Criar e ativar um ambiente virtual para Python 3.8.10
python3.8 -m venv env_python3.8
source env_python3.8/bin/activate
cd src
python3.8 -m pip install .
