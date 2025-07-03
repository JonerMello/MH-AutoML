import os
import mlflow
from mlflow.data.pandas_dataset import PandasDataset

class MLflowManager:
    def __init__(self, experiment_name="MH-AutoML"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)
        
    def start_run(self, run_name=None):
        return mlflow.start_run(run_name=run_name, log_system_metrics=True)
    
    def log_metrics(self, metrics_dict):
        for key, value in metrics_dict.items():
            mlflow.log_metric(key, value)
    
    def log_params(self, params_dict):
        mlflow.log_params(params_dict)
    
    def log_model(self, model, artifact_path, registered_model_name):
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
    
    def log_artifact(self, file_path, artifact_path=None):
        if file_path and os.path.exists(file_path):
            mlflow.log_artifact(file_path, artifact_path)
    
    def log_dataset(self, dataset_df, source):
        dataset = mlflow.data.from_pandas(dataset_df, source=source)
        mlflow.log_input(dataset, context="training")
    
    def end_run(self):
        mlflow.end_run()