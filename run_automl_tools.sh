#!/bin/bash

# Diretório onde os resultados serão salvos
OUTPUT_DIR=$1

# Verifica se o diretório de saída foi fornecido
if [ -z "$OUTPUT_DIR" ]; then
    echo "Por favor, forneça o diretório de saída como argumento."
    exit 1
fi

# Criar o diretório de saída se não existir
mkdir -p "$OUTPUT_DIR"

# Lista de ferramentas de AutoML
AUTOML_TOOLS=("autogluon" "tpot" "autopytorch" "autosklearn" "hypergbm" "lightautoml" "mljar")

# Iterar sobre cada ferramenta de AutoML
for TOOL in "${AUTOML_TOOLS[@]}"; do
    # Construir a imagem Docker para cada ferramenta usando Dockerfile.automl
    docker build -f Dockerfile.automl --build-arg AUTOML_TOOL="$TOOL" -t automl_image_$TOOL .

    # Verificar se a construção da imagem foi bem-sucedida
    if [ $? -ne 0 ]; then
        echo "Falha ao construir a imagem Docker para $TOOL"
        exit 1
    fi

    # Executar o contêiner com montagem de volumes para salvar resultados
    docker run --rm -e AUTOML_TOOL="$TOOL" -v "$(pwd)/$OUTPUT_DIR:/app/output" automl_image_$TOOL

    # Verificar se o contêiner foi executado com sucesso
    if [ $? -ne 0 ]; then
        echo "Falha ao executar o contêiner para $TOOL"
        exit 1
    fi

    echo "O contêiner para $TOOL foi executado e os resultados estão em $OUTPUT_DIR"
done

