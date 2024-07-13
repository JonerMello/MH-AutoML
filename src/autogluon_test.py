import os
import pandas as pd
import sklearn.metrics
import timeit
import argparse
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from datetime import datetime

# Função para rodar o AutoGluon em um arquivo CSV específico
def run_autogluon(dataset_file):
    try:
        dataset_df = pd.read_csv(dataset_file)
        df_train, df_test = train_test_split(dataset_df, test_size=0.33, random_state=1)
        
        start_time = timeit.default_timer()
        predictor = TabularPredictor(label='class').fit(train_data=df_train)
        predictions = predictor.predict(df_test, as_pandas=False)

        m, s = divmod(timeit.default_timer() - start_time, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)

        y_test = df_test['class']

        results = {
            "dataset": dataset_file,
            "tempo": time_str,
            "acuracia": sklearn.metrics.accuracy_score(y_test, predictions),
            "precisao": sklearn.metrics.precision_score(y_test, predictions),
            "recall": sklearn.metrics.recall_score(y_test, predictions),
            "f1": sklearn.metrics.f1_score(y_test, predictions)
        }

        return results

    except Exception as e:
        print(f'Erro Autogluon dataset {dataset_file}: {e}')
        return None

# Função principal para processar todos os arquivos CSV na pasta
def processar_todos_csv_pasta(pasta):
    resultados = []
    
    # Itera sobre todos os arquivos na pasta
    for root, dirs, files in os.walk(pasta):
        for file in files:
            if file.endswith(".csv"):
                dataset_file = os.path.join(root, file)
                print(f"Processando arquivo: {dataset_file}")
                resultado = run_autogluon(dataset_file)
                if resultado:
                    resultados.append(resultado)

    # Salva todos os resultados em um único DataFrame
    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv("/app/output/resultados_autogluon.csv", index=False)
    print("Resultados salvos em CSV no diretório /app/output.")

# Função para processar um único arquivo CSV
def processar_arquivo_csv(arquivo_csv):
    resultado = run_autogluon(arquivo_csv)
    if resultado:
        resultados_df = pd.DataFrame([resultado])
        resultados_df.to_csv("/app/output/resultados_autogluon.csv", index=False)
        print("Resultado salvo em CSV no diretório /app/output.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa arquivos CSV com AutoGluon.")
    parser.add_argument("input_path", type=str, help="Caminho para a pasta com datasets ou um único arquivo CSV.")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input_path):
        processar_todos_csv_pasta(args.input_path)
    elif os.path.isfile(args.input_path) and args.input_path.endswith(".csv"):
        processar_arquivo_csv(args.input_path)
    else:
        print("O caminho fornecido não é um arquivo CSV válido ou uma pasta.")
