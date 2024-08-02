import os
import time
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from autoPyTorch.api.tabular_classification import TabularClassificationTask

# Função para formatar o tempo em segundos para hh:mm:ss
def formatar_tempo(tempo_segundos):
    horas = int(tempo_segundos / 3600)
    minutos = int((tempo_segundos % 3600) / 60)
    segundos = int(tempo_segundos % 60)
    return f"{horas:02d}:{minutos:02d}:{segundos:02d}"

# Função para rodar o AutoPyTorch em um arquivo CSV específico
def run_autopytorch(dataset_file):
    dataset = pd.read_csv(dataset_file)
    X = dataset.drop('class', axis=1)
    y = dataset['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    
    # Inicia a contagem do tempo de execução total da ferramenta
    start_time = time.time()
    classifier = TabularClassificationTask(ensemble_nbest=1)
    classifier.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test.copy(),
        y_test=y_test.copy(),
        optimize_metric='accuracy',
        memory_limit=1024 * 16
    )
    
    # Para a contagem
    end_time = time.time()
    
    # Calcula o tempo total e formata para hh:mm:ss
    total_time = formatar_tempo(end_time - start_time)

    # Faz a predição com o dataset de teste usando o melhor modelo
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    results = {
        'dataset': dataset_file,
        'tempo': total_time,
        'acuracia': accuracy,
        'precisao': precision,
        'recall': recall,
        'f1': f1
    }

    return results

# Função principal para processar todos os arquivos CSV na pasta
def processar_todos_csv_pasta(pasta, output_path):
    resultados = []
    
    # Itera sobre todos os arquivos na pasta
    for root, dirs, files in os.walk(pasta):
        for file in files:
            if file.endswith(".csv"):
                dataset_file = os.path.join(root, file)
                print(f"Processando arquivo: {dataset_file}")
                resultado = run_autopytorch(dataset_file)
                if resultado:
                    resultados.append(resultado)

    # Salva todos os resultados em um único DataFrame
    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv(os.path.join(output_path, "resultados_autopytorch.csv"), index=False)
    print(f"Resultados salvos em CSV no diretório {output_path}.")

# Função para processar um único arquivo CSV
def processar_arquivo_csv(arquivo_csv, output_path):
    resultado = run_autopytorch(arquivo_csv)
    if resultado:
        resultados_df = pd.DataFrame([resultado])
        resultados_df.to_csv(os.path.join(output_path, "resultados_autopytorch.csv"), index=False)
        print(f"Resultado salvo em CSV no diretório {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa arquivos CSV com AutoPyTorch.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Caminho para a pasta com datasets ou um único arquivo CSV.")
    parser.add_argument("-o", "--output", type=str, default="/output", help="Caminho para o diretório onde os resultados serão salvos.")
  
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    if os.path.isdir(args.input):
        processar_todos_csv_pasta(args.input, args.output)
    elif os.path.isfile(args.input) and args.input.endswith(".csv"):
        processar_arquivo_csv(args.input, args.output)
    else:
        print("O caminho fornecido não é um arquivo CSV válido ou uma pasta.")
