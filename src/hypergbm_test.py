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

def processar_todos_csv_pasta(pasta):
    resultados = []

    for root, dirs, files in os.walk(pasta):
        for file in files:
            if file.endswith(".csv"):
                dataset_file = os.path.join(root, file)
                print(f"Processando arquivo: {dataset_file}")
                resultado = run_hypergbm(dataset_file)
                if resultado:
                    resultados.append(resultado)

    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv("/app/output/resultados_hypergbm.csv", index=False)
    print("Resultados salvos em CSV no diretório /app/output.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa arquivos CSV com HyperGBM.")
    parser.add_argument("input_path", type=str, help="Caminho para a pasta com datasets ou um único arquivo CSV.")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input_path):
        processar_todos_csv_pasta(args.input_path)
    elif os.path.isfile(args.input_path) and args.input_path.endswith(".csv"):
        resultado = run_hypergbm(args.input_path)
        if resultado:
            resultados_df = pd.DataFrame([resultado])
            resultados_df.to_csv("/app/output/resultados_hypergbm.csv", index=False)
            print("Resultado salvo em CSV no diretório /app/output.")
    else:
        print("O caminho fornecido não é um arquivo CSV válido ou uma pasta.")
