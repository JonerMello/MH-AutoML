#!/bin/bash

# Função para instalar Autogluon e executar o script
run_autogluon() {
    pip install autogluon
    python3 autogluon_test.py datasets
}

# Função para instalar TPOT e executar o script
run_tpot() {
    pip install tpot
    python3 tpot_test.py datasets
}

# Função para instalar AutoPyTorch e executar o script
run_autopytorch() {
    pip install autoPyTorch
    python3 autopytorch_test.py datasets
}

# Função para instalar AutoSklearn e executar o script
run_autosklearn() {
    pip install auto-sklearn
    python3 autosklearn_test.py datasets
}

# Função para instalar HyperGBM e executar o script
run_hypergbm() {
    pip install hypergbm
    python3 hypergbm_test.py datasets
}

# Função para instalar LightAutoML e executar o script
run_lightautoml() {
    pip install lightautoml
    python3 lightautoml_test.py datasets
}

# Função para instalar MLJAR e executar o script
run_mljar() {
    pip install mljar-supervised
    python3 mljar_test.py datasets
}

# Seleciona a ferramenta de AutoML com base na variável de ambiente
case "$AUTOML_TOOL" in
    "autogluon")
        run_autogluon
        ;;
    "tpot")
        run_tpot
        ;;
    "autopytorch")
        run_autopytorch
        ;;
    "autosklearn")
        run_autosklearn
        ;;
    "hypergbm")
        run_hypergbm
        ;;
    "lightautoml")
        run_lightautoml
        ;;
    "mljar")
        run_mljar
        ;;
    *)
        echo "Ferramenta de AutoML desconhecida: $AUTOML_TOOL"
        exit 1
        ;;
esac

