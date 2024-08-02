#!/bin/bash

# Nome da imagem e do contêiner
IMAGE_NAME="my_automl_image"
CONTAINER_NAME="my_automl_container"

# Verificar se o primeiro argumento é um diretório local para salvar os resultados
LOCAL_OUTPUT_DIR=$1
shift

# Verificar a presença do parâmetro --d
use_demo_image=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --d)
            use_demo_image=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Se LOCAL_OUTPUT_DIR não for especificado, use o diretório padrão ./output
LOCAL_OUTPUT_DIR=${LOCAL_OUTPUT_DIR:-$(pwd)/output}

# Criar diretório local de output se não existir
if [ ! -d "$LOCAL_OUTPUT_DIR" ]; then
    echo "Criando diretório de output em $LOCAL_OUTPUT_DIR..."
    mkdir -p "$LOCAL_OUTPUT_DIR"
fi

# Determinar o Dockerfile a ser usado
if [ "$use_demo_image" = true ]; then
    DOCKERFILE="Dockerfile.demo"
    IMAGE_NAME="my_demo_image"
else
    DOCKERFILE="Dockerfile.automl"
fi

# Construir a imagem Docker usando o Dockerfile apropriado
echo "Construindo a imagem Docker usando $DOCKERFILE..."
sudo docker build -t $IMAGE_NAME -f $DOCKERFILE .

# Verificar se há um contêiner em execução com o mesmo nome e parar e remover se existir
if [ "$(sudo docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Parando e removendo o contêiner existente..."
    sudo docker stop $CONTAINER_NAME
    sudo docker rm $CONTAINER_NAME
fi

# Executar o contêiner com volume montado
echo "Executando o contêiner..."
sudo docker run --name $CONTAINER_NAME -v "$LOCAL_OUTPUT_DIR":/app/output $IMAGE_NAME

# Verificar logs do contêiner
echo "Logs do contêiner:"
sudo docker logs $CONTAINER_NAME

