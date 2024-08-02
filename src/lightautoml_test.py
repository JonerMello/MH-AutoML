import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime
import timeit
import argparse

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


def run_lightautoml(dataset_file):
    dataset = pd.read_csv(dataset_file)

    train, test = train_test_split(dataset, test_size=0.33, random_state=1)

    start_time = timeit.default_timer()
    
    try:
        task = Task('binary')

        roles = {'target': 'class'}

        automl = TabularAutoML(task=task, timeout=6000000000, cpu_limit=4, reader_params={'random_state': 1})
        automl.fit_predict(train, roles=roles, verbose=0)
        
        predictions = automl.predict(test)

        m, s = divmod(timeit.default_timer() - start_time, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)

        predictions_binary = (predictions.data[:, 0] > 0.5).astype(int)

        # Calcule as métricas
        accuracy = accuracy_score(test['class'], predictions_binary)
        precision = precision_score(test['class'], predictions_binary)
        recall = recall_score(test['class'], predictions_binary)
        f1 = f1_score(test['class'], predictions_binary)

        # Salve os resultados em um DataFrame
        results = {
            "dataset": dataset_file,
            "execution_time": time_str,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        return results

    except Exception as e:
        print(f"Error running AutoML on dataset {dataset_file}: {e}")
        return None

# Função principal para processar todos os arquivos CSV na pasta
def processar_todos_csv_pasta(pasta, output_path):
    resultados = []
    
    # Itera sobre todos os arquivos na pasta
    for root, dirs, files in os.walk(pasta):
        for file in files:
            if file.endswith(".csv"):
                dataset_file = os.path.join(root, file)
                print(f"Processando arquivo: {dataset_file}")
                resultado = run_lightautoml(dataset_file)
                if resultado:
                    resultados.append(resultado)

    # Salva todos os resultados em um único DataFrame
    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv(os.path.join(output_path, "resultados_lightautoml.csv"), index=False)
    print(f"Resultados salvos em CSV no diretório {output_path}.")

# Função para processar um único arquivo CSV
def processar_arquivo_csv(arquivo_csv, output_path):
    resultado = run_lightautoml(arquivo_csv)
    if resultado:
        resultados_df = pd.DataFrame([resultado])
        resultados_df.to_csv(os.path.join(output_path, "resultados_lightautoml.csv"), index=False)
        print(f"Resultado salvo em CSV no diretório {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa arquivos CSV com Lightautoml.")
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
