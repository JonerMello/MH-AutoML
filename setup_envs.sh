#!/bin/bash

# Função para criar um ambiente virtual e instalar pacotes
setup_venv() {
    local tool_name=$1
    local python_version=${2:-python3.9}

    echo "Criando ambiente virtual para $tool_name usando $python_version..."
    $python_version -m venv "${tool_name}_venv"

    echo "Ativando ambiente virtual..."
    source "${tool_name}_venv/bin/activate"

    echo "Instalando pip no ambiente virtual..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py

    echo "Instalando dependências para $tool_name..."

    case "$tool_name" in
        autoPyTorch)
            pip install Cython==0.29.36
            pip install scipy==1.9
            pip uninstall -y imbalanced-learn mlxtend yellowbrick
            pip install scikit-learn==0.24.2 --no-build-isolation
            pip install dask[dataframe]
            pip install ConfigSpace==0.7.1
            pip install --force-reinstall numpy==1.26.4
            pip install autoPyTorch
            ;;
        
        hypergbm)
            pip install build-essential swig python3-dev
            pip install --force-reinstall numpy==1.26.4
            pip install hypergbm
            ;;
        
        lightautoml)
            #pip install build-essential swig python3-dev
            pip install --force-reinstall numpy==1.26.4
            pip install lightautoml
            ;;
        
        autogluon)
            #pip install build-essential swig python3-dev
            #pip install --force-reinstall numpy==1.26.4
            pip install autogluon
            ;;
           
        tpot)
            #pip install build-essential swig python3-dev
            #pip install --force-reinstall numpy==1.26.4
            pip install tpot
            ;;
        
        mljar-supervised)
            #pip install build-essential swig python3-dev
            pip install mljar-supervised
            ;;
        
        *)
            echo "Ferramenta desconhecida: $tool_name"
            ;;
    esac

    #echo "Desativando e removendo o ambiente virtual..."
    deactivate
}

# Função para instalar pacotes do auto-sklearn fora do ambiente virtual
setup_autosklearn() {
    echo "Instalando dependências para auto-sklearn fora do ambiente virtual..."
    apt-get update
    pip install build-essential swig python3-dev
    pip install --force-reinstall numpy==1.26.4
    pip install auto-sklearn
}

# Cria e configura ambientes virtuais para todas as ferramentas, exceto auto-sklearn
setup_venv "autoPyTorch"
setup_autosklearn
setup_venv "hypergbm"
setup_venv "lightautoml"
setup_venv "mljar-supervised"
setup_venv "tpot"
setup_venv "autogluon"
