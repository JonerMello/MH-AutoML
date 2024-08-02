import os
import pandas as pd
import timeit
from sklearn.model_selection import train_test_split
import sklearn.metrics
import argparse
from hypergbm import make_experiment


def run_hypergbm(dataset_file):
    dataset = pd.read_csv(dataset_file)
    X = dataset.drop('class', axis=1)
    y = dataset['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    start_time = timeit.default_timer()

    try:
        experiment = make_experiment(pd.concat([X_train, y_train], axis=1), target='class')
        estimator = experiment.run()
        elapsed_time = timeit.default_timer() - start_time
        h, m, s = map(int, [elapsed_time // 3600, (elapsed_time % 3600) // 60, elapsed_time % 60])
        time_str = "%02d:%02d:%02d" % (h, m, s)

        y_pred = estimator.predict(X_test)

        accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
        f1 = sklearn.metrics.f1_score(y_test, y_pred, average='macro')  # Use 'weighted' for imbalanced classes
        precision = sklearn.metrics.precision_score(y_test, y_pred, average='macro')
        recall = sklearn.metrics.recall_score(y_test, y_pred, average='macro')

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "dataset": dataset_file,
            "execution_time": time_str
        }

        return results

    except Exception as e:
        print(f"Erro na ferramenta HyperGBM no dataset {dataset_file}: {e}")
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
                resultado = run_hypergbm(dataset_file)
                if resultado:
                    resultados.append(resultado)

    # Salva todos os resultados em um único DataFrame
    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv(os.path.join(output_path, "resultados_hypergbm.csv"), index=False)
    print(f"Resultados salvos em CSV no diretório {output_path}.")

# Função para processar um único arquivo CSV
def processar_arquivo_csv(arquivo_csv, output_path):
    resultado = run_hypergbm(arquivo_csv)
    if resultado:
        resultados_df = pd.DataFrame([resultado])
        resultados_df.to_csv(os.path.join(output_path, "resultados_hypergbm.csv"), index=False)
        print(f"Resultado salvo em CSV no diretório {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa arquivos CSV com Hypergbm.")
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
