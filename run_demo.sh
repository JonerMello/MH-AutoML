#!/bin/bash


# Instale os pacotes necessários
cd src
pip install .

# Execute a ferramenta
python3 view/main.py -d Datasets/dataset_sujo.csv -l class

