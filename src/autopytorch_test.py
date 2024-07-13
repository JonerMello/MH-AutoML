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
def processar_todos_csv_pasta(pasta):
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
    resultados_df.to_csv("/app/output/resultados_autopytorch.csv", index=False)
    print("Resultados salvos em CSV no diretório /app/output.")

# Função para processar um único arquivo CSV
def processar_arquivo_csv(arquivo_csv):
    resultado = run_autopytorch(arquivo_csv)
    if resultado:
        resultados_df = pd.DataFrame([resultado])
        resultados_df.to_csv("/app/output/resultados_autopytorch.csv", index=False)
        print("Resultado salvo em CSV no diretório /app/output.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa arquivos CSV com AutoPyTorch.")
    parser.add_argument("input_path", type=str, help="Caminho para a pasta com datasets ou um único arquivo CSV.")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input_path):
        processar_todos_csv_pasta(args.input_path)
    elif os.path.isfile(args.input_path) and args.input_path.endswith(".csv"):
        processar_arquivo_csv(args.input_path)
    else:
        print("O caminho fornecido não é um arquivo CSV válido ou uma pasta.")
