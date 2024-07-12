#!/bin/bash

# Lista de datasets disponíveis
datasets=(
    "Datasets/adroit.csv"
    "Datasets/androcrawl.csv"
    "Datasets/android_permissions.csv"
    "Datasets/dataset_sujo.csv"
    "Datasets/defensedroid_prs.csv"
    "Datasets/drebin215.csv"
    "Datasets/kronodroid_emulador.csv"
    "Datasets/kronodroid_real.csv"
)

# Loop para rodar o experimento com cada dataset
for dataset in "${datasets[@]}"; do
    echo "Running experiment with $dataset..."
    python3.8 src/view/main.py -d "$dataset" -l class
done

echo "All experiments completed."

