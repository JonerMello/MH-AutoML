#!/bin/bash

# Função para ativar o ambiente virtual e executar o script Python
run_in_venv() {
    local tool_name=$1
    local script_name=$2
    local use_dataset_csv=$3
    local output_dir=$4

    if [ "$tool_name" == "auto-sklearn" ]; then
        echo "Executando $tool_name fora do ambiente virtual..."

        echo "Definindo o parâmetro do dataset..."
        local dataset_param="./Datasets"
        if [ "$use_dataset_csv" == "--d" ]; then
            dataset_param="./Datasets/adroit.csv"
        fi

        echo "Executando o script $script_name com o parâmetro -i $dataset_param e -o $output_dir..."
        python3.9 src/"$script_name" -i "$dataset_param" -o "$output_dir"
    else
        echo "Ativando ambiente virtual para $tool_name..."
        source "${tool_name}_venv/bin/activate"

        echo "Definindo o parâmetro do dataset..."
        local dataset_param="./Datasets"
        if [ "$use_dataset_csv" == "--d" ]; then
            dataset_param="./Datasets/adroit.csv"
        fi

        echo "Executando o script $script_name com o parâmetro -i $dataset_param e -o $output_dir..."
        python3.9 src/"$script_name" -i "$dataset_param" -o "$output_dir"

        echo "Desativando o ambiente virtual..."
        deactivate
    fi
}

# Obtém os parâmetros da linha de comando
dataset_option=""
output_dir="./output"  # Diretório de saída padrão

while [[ $# -gt 0 ]]; do
    case $1 in
        --d)
            dataset_option="--d"
            shift
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Executa todos os testes
echo "Iniciando execução dos testes..."

run_in_venv "autoPyTorch" "autopytorch_test.py" "$dataset_option" "$output_dir"
run_in_venv "auto-sklearn" "autosklearn_test.py" "$dataset_option" "$output_dir"
run_in_venv "hypergbm" "hypergbm_test.py" "$dataset_option" "$output_dir"
run_in_venv "lightautoml" "lightautoml_test.py" "$dataset_option" "$output_dir"
run_in_venv "mljar-supervised" "mljar_test.py" "$dataset_option" "$output_dir"
run_in_venv "autogluon" "autogluon_test.py" "$dataset_option" "$output_dir"
run_in_venv "tpot" "tpot_test.py" "$dataset_option" "$output_dir"

echo "Todos os testes concluídos."

