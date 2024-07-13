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

def processar_todos_csv_pasta(pasta):
    resultados = []

    for root, dirs, files in os.walk(pasta):
        for file in files:
            if file.endswith(".csv"):
                dataset_file = os.path.join(root, file)
                print(f"Processando arquivo: {dataset_file}")
                resultado = run_lightautoml(dataset_file)
                if resultado:
                    resultados.append(resultado)

    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv("/app/output/resultados_lightautoml.csv", index=False)
    print("Resultados salvos em CSV no diretório /app/output.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa arquivos CSV com LightAutoML.")
    parser.add_argument("input_path", type=str, help="Caminho para a pasta com datasets ou um único arquivo CSV.")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input_path):
        processar_todos_csv_pasta(args.input_path)
    elif os.path.isfile(args.input_path) and args.input_path.endswith(".csv"):
        resultado = run_lightautoml(args.input_path)
        if resultado:
            resultados_df = pd.DataFrame([resultado])
            resultados_df.to_csv("/app/output/resultados_lightautoml.csv", index=False)
            print("Resultado salvo em CSV no diretório /app/output.")
    else:
        print("O caminho fornecido não é um arquivo CSV válido ou uma pasta.")
