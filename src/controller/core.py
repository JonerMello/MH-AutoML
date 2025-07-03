import os
import pandas as pd
import numpy as np
import csv
import logging
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import time
import mlflow
import threading
import webbrowser
from mlflow import MlflowClient
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
import shap
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import subprocess
import base64
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
# Configure matplotlib to suppress categorical units warning
plt.rcParams['figure.max_open_warning'] = 0
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import (precision_recall_curve, average_precision_score, 
                            roc_auc_score, confusion_matrix, fbeta_score)

import plotly.io as pio
import plotly.graph_objs as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,mean_squared_error,matthews_corrcoef
from halo import Halo
from model.tools.dataset_validation import DatasetValidation
from model.preprocessing.data_cleaning import DataCleaning
from model.preprocessing.data_info import DataInfo
from model.preprocessing.data_transformation import DataTransformation
from model.feature_engineering.data_reduction import FeatureSelection
from model.optimization.hyperparameters_methods import Hyperparameters
from model.interpretability.interpretability import Interpretability
from model.tools.report_generator import ReportGenerator
from model.tools.pdf_report_generator import PDFReportGenerator
import io
from sklearn.pipeline import FeatureUnion
from model.optimization.grid_search import GridSearch
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
import joblib
import datetime
from sklearn.utils import estimator_html_repr
import codecs
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from colorama import Fore, Style
import warnings
from mlflow.models import infer_signature
import psutil
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor


from optuna.exceptions import ExperimentalWarning
from typing import Optional, Tuple, Dict, Any

# Configure warnings globally to suppress matplotlib categorical units message
warnings.filterwarnings("ignore", message=".*categorical units.*")
warnings.filterwarnings("ignore", message=".*Using categorical units.*")

class Core:
    """
    Classe principal responsável por orquestrar o pipeline de AutoML do sistema MH-AutoML.
    Gerencia o fluxo de dados, execução das etapas do pipeline, logging, geração de artefatos e integração com MLflow.
    """
    # Desativar todos os UserWarnings temporariamente
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    mlflow.set_system_metrics_sampling_interval(1)
    def __init__(self,
                 dataset_url: str,
                 label: str,
                 log_level: str = 'info',
                 remove_duplicates: bool = True,
                 remove_missing_values: bool = True,
                 remove_outliers: bool = True,
                 one_hot_encoder: bool = True,
                 do_label_encode: bool = True,
                 balance_classes: Optional[str] = None,
                 feature_selection_method: str = 'LASSO'):
        """
        Inicializa o controlador principal do pipeline.
        Args:
            dataset_url (str): Caminho para o dataset.
            label (str): Nome da coluna alvo.
            log_level (str): Nível de logging ('debug', 'info', etc).
            remove_duplicates (bool): Remover duplicatas.
            remove_missing_values (bool): Remover valores faltantes.
            remove_outliers (bool): Remover outliers.
            one_hot_encoder (bool): Aplicar one-hot encoding.
            do_label_encode (bool): Aplicar label encoding.
            balance_classes (Optional[str]): Estratégia de balanceamento ('SMOTE', 'RUS', None).
            feature_selection_method (str): Método de seleção de features ('LASSO', 'PCA', 'ANOVA').
        """
        self.dataset_url = dataset_url
        self.label = label
        self.use_pca = False 
        self.use_anova = False
        self.use_lasso = False
        self.log_level = log_level
        self.remove_duplicates = remove_duplicates
        self.remove_missing_values = remove_missing_values
        self.remove_outliers = remove_outliers
        self.one_hot_encoder = one_hot_encoder
        self.do_label_encode = do_label_encode
        self.balance_classes = balance_classes
        self.feature_selection_method = feature_selection_method
        self.measurements = []
        self._configure_logging()
        self._configure_warnings()
        self._configure_mlflow()
        self.logger = logging.getLogger(__name__)
    
    def _configure_logging(self):
        """Configura o nível de logging do sistema."""
        log_level_mapping = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        if self.log_level not in log_level_mapping:
            raise ValueError(f"Invalid log level: {self.log_level}")
        logging.basicConfig(level=log_level_mapping[self.log_level], format='%(levelname)s: %(message)s')
    
    def _configure_warnings(self):
        """Desativa warnings desnecessários para execução limpa."""
        warnings.filterwarnings('ignore', category=UserWarning)
        try:
            from optuna.exceptions import ExperimentalWarning
            warnings.filterwarnings("ignore", category=ExperimentalWarning)
        except ImportError:
            pass
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        # Suppress matplotlib categorical units warning
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        plt.rcParams['figure.max_open_warning'] = 0
        
        # Suppress specific matplotlib warnings about categorical units
        warnings.filterwarnings("ignore", message=".*categorical units.*")
        warnings.filterwarnings("ignore", message=".*Using categorical units.*")
    
    def _configure_mlflow(self):
        """Configura parâmetros globais do MLflow."""
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        try:
            mlflow.set_system_metrics_sampling_interval(1)
        except Exception:
            pass

    def create_results_folder(self, folder_name="results"):
        """
        Create a folder with the specified name if it doesn't exist.

        Args:
            folder_name (str): The name of the folder to create.

        Returns:
            str: The path to the created folder.
        """
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return folder_name 

    def plot_train_test_relation(self, y_train: np.ndarray, y_test: np.ndarray) -> None:
        """
        Plota a distribuição das classes nos conjuntos de treino e teste.
        Args:
            y_train (np.ndarray): Labels de treino.
            y_test (np.ndarray): Labels de teste.
        """
        import matplotlib.pyplot as plt  # Import local para evitar conflitos globais
        fig, ax = plt.subplots(figsize=(10, 6))  # type: ignore
        class_labels = {0: 'Benign', 1: 'Malware'}
        train_counts = np.bincount(y_train)
        test_counts = np.bincount(y_test)
        max_classes = max(len(train_counts), len(test_counts))
        train_counts = np.pad(train_counts, (0, max_classes - len(train_counts)), mode='constant')
        test_counts = np.pad(test_counts, (0, max_classes - len(test_counts)), mode='constant')
        index = np.arange(max_classes)
        bar_width = 0.35
        bars1 = ax.bar(index, train_counts, bar_width, label='Training Set')  # type: ignore
        bars2 = ax.bar(index, test_counts, bar_width, bottom=train_counts, label='Test Set')  # type: ignore
        for i, label in class_labels.items():
            ax.text(i, train_counts[i] / 2, f"{train_counts[i]}", ha='center', va='center_baseline', color='white', fontweight='bold')  # type: ignore
            ax.text(i, train_counts[i] + test_counts[i] / 2, f"{test_counts[i]}", ha='center', va='center_baseline', color='white', fontweight='bold')  # type: ignore
        ax.set_xlabel('Class')  # type: ignore
        ax.set_ylabel('Count')  # type: ignore
        ax.set_title('Distribution of Classes in Training and Test Sets')  # type: ignore
        ax.set_xticks(index)  # type: ignore
        ax.set_xticklabels([class_labels.get(i, '') for i in range(max_classes)])  # type: ignore
        ax.legend()  # type: ignore
        results_folder = "results"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        img_filename = f'train_test_distribution.png'
        img_filepath = os.path.join(results_folder, img_filename)
        plt.savefig(img_filepath, bbox_inches='tight')
        plt.close()
    

    def display_data_info(self, dataset_df):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        colored_message = f"[{Fore.GREEN}{now}{Style.RESET_ALL}] Displaying data info..."
        logging.info(colored_message)
        data_info = DataInfo(self.label, dataset_df)
        data_info.display_dataframe_info()
        return data_info

    def preprocess(self, dataset_df, label):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        colored_message = f"[{Fore.GREEN}{now}{Style.RESET_ALL}] Preprocessing data..."
        logging.info(colored_message)

        # Check the data types of each column in the DataFrame
        data_types = dataset_df.dtypes

        # Check if dataset_df contains numerical columns
        contains_numerical = data_types != 'int64'

        if contains_numerical.any():
            one_hot_encoder = True
        else:
            one_hot_encoder = False

        preprocessor = Pipeline(steps=[
            ('Data Cleaning', DataCleaning(remove_duplicates=self.remove_duplicates, remove_missing_values=self.remove_missing_values, remove_outliers=self.remove_outliers)),
            ('Data Transformation', DataTransformation(label=self.label, one_hot_encoder=one_hot_encoder, do_label_encode=self.do_label_encode))
        ])
        
        transformation = preprocessor.fit_transform(dataset_df)

        # Assuming transformation is a tuple or list
        if isinstance(transformation, tuple) or isinstance(transformation, list):
            X, y = transformation[0].astype(np.int8), transformation[1].astype(np.int8)
        else:
            raise ValueError("Unexpected data structure returned by the preprocessor.")
       

        return X, y, preprocessor
  

    def enable_pca(self):
        self.use_pca = True

    def enable_anova(self):
        self.use_anova = True

    def enable_lasso(self):
        self.use_lasso = True


    def log_optuna_visualizations(self, study):
        """Generate Optuna visualizations and save them to results folder"""
        try:
            import tempfile
            import plotly.io as pio
            results_folder = self.create_results_folder()
        
            # Create temporary directory for plots
            with tempfile.TemporaryDirectory() as tmp_dir:
                # 1. Parameter Importance Plot (Plotly version)
                fig_param = optuna.visualization.plot_param_importances(study)
                param_path = os.path.join(results_folder, "optuna_param_importance.html")
                fig_param.write_html(param_path)
            
                # 2. Optimization History Plot (Plotly version)
                fig_history = optuna.visualization.plot_optimization_history(study)
                history_path = os.path.join(results_folder, "optuna_optimization_history.html")
                fig_history.write_html(history_path)
            
                # 3. Save matplotlib versions as PNG for compatibility
                try:
                    # Parameter Importance (Matplotlib)
                    fig_matplotlib1 = optuna.visualization.matplotlib.plot_param_importances(study)
                    param_png_path = os.path.join(results_folder, "optuna_param_importance.png")
                    fig_matplotlib1.figure.savefig(param_png_path)
                    plt.close(fig_matplotlib1.figure)
                
                    # Optimization History (Matplotlib)
                    fig_matplotlib2 = optuna.visualization.matplotlib.plot_optimization_history(study)
                    history_png_path = os.path.join(results_folder, "optuna_optimization_history.png")
                    fig_matplotlib2.figure.savefig(history_png_path)
                    plt.close(fig_matplotlib2.figure)
                except Exception as e:
                    self.logger.warning(f"Matplotlib visualization failed: {str(e)}")
            
                # 4. Additional visualizations (Plotly)
                try:
                    # Slice Plot
                    fig_slice = optuna.visualization.plot_slice(study)
                    slice_path = os.path.join(results_folder, "optuna_slice_plot.html")
                    fig_slice.write_html(slice_path)
                
                    # Parallel Coordinate Plot
                    fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
                    parallel_path = os.path.join(results_folder, "optuna_parallel_coordinate.html")
                    fig_parallel.write_html(parallel_path)
                    
                    # Generate PNG versions of HTML plots (optional - can be disabled if causing issues)
                    # NOTE: Set to True only if you want PNG versions. May cause blocking on some systems.
                    generate_pngs = False  # Set to True to enable PNG generation (may cause blocking)
                    
                    if generate_pngs:
                        try:
                            # Convert parallel coordinate plot to PNG
                            parallel_png_path = os.path.join(results_folder, "optuna_parallel_coordinate.png")
                            fig_parallel.write_image(parallel_png_path, width=1200, height=800)
                        except Exception as png_error:
                            self.logger.warning(f"Parallel coordinate PNG conversion failed: {str(png_error)}")
                        
                        # Generate slice plot PNG with simplified approach
                        try:
                            slice_png_path = os.path.join(results_folder, "optuna_slice_plot.png")
                            # Try matplotlib approach first (more reliable)
                            self._create_slice_plot_matplotlib(study, slice_png_path)
                        except Exception as slice_png_error:
                            self.logger.warning(f"Matplotlib slice plot failed: {str(slice_png_error)}")
                            # Fallback to Plotly (may cause blocking)
                            try:
                                fig_slice.write_image(slice_png_path, width=1200, height=800)
                            except Exception as fallback_error:
                                self.logger.warning(f"Plotly slice plot PNG failed: {str(fallback_error)}")
                    else:
                        self.logger.info("PNG generation disabled for Optuna plots")
                        
                except Exception as e:
                    self.logger.warning(f"Advanced visualization failed: {str(e)}")
            
                # 5. Save study data
                trials_df = study.trials_dataframe()
                trials_path = os.path.join(results_folder, "optuna_trials.csv")
                trials_df.to_csv(trials_path, index=False)
            
                return {
                    'param_importance_html': param_path,
                    'optimization_history_html': history_path,
                    'param_importance_png': param_png_path if os.path.exists(param_png_path) else None,
                    'optimization_history_png': history_png_path if os.path.exists(history_png_path) else None,
                    'slice_plot': slice_path if os.path.exists(slice_path) else None,
                    'parallel_coordinate': parallel_path if os.path.exists(parallel_path) else None,
                    'trials_data': trials_path
                }
            
        except Exception as e:
            self.logger.error(f"Failed to generate Optuna visualizations: {str(e)}")
            return None

    def feature_selection(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Executa a seleção de features conforme o método configurado.
        Args:
            X (np.ndarray): Dados de entrada.
            y (np.ndarray): Labels.
        Returns:
            Tuple: Dados transformados, pipeline, informações de seleção e nomes das features.
        """
        import numpy as np
        import pandas as pd
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        colored_message = f"[{Fore.GREEN}{now}{Style.RESET_ALL}] Performing feature selection"
        logging.info(colored_message)
        results_folder = self.create_results_folder()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.plot_train_test_relation(np.asarray(y_train), np.asarray(y_test))
        feature_selection_steps = []
        if self.balance_classes:
            feature_selection_steps.append(('balancing', FeatureSelection(balance_classes=self.balance_classes)))
        if self.use_pca:
            feature_selection_steps.append(('PCA', FeatureSelection(pca=True, num_components=int(0.3 * X.shape[1]))))
        elif self.use_anova:
            feature_selection_steps.append(('ANOVA', FeatureSelection(anova=True, k_features=int(0.3 * X.shape[1]))))
        elif self.use_lasso:
            feature_selection_steps.append(('LASSO', FeatureSelection(lasso=True, alpha=0.00001)))
        else:
            colored_message = f"[{Fore.YELLOW}No feature selection method enabled. Using original features.{Style.RESET_ALL}]"
            self.logger.warning(colored_message)
            # Retorna os dados originais sem transformação
            if hasattr(X_train, 'columns') and not isinstance(X_train, list):
                feature_names = list(X_train.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(np.asarray(X_train).shape[1])]
            return np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test), None, {
                'method': None,
                'feature_names': feature_names,
                'original_features': feature_names,
                'transformer': None
            }, feature_names
        feature_selection_pipeline = Pipeline(feature_selection_steps) if feature_selection_steps else None
        if feature_selection_pipeline:
            X_train_selected = feature_selection_pipeline.fit_transform(X_train, y_train)
            X_test_selected = feature_selection_pipeline.transform(X_test)
            fs_step = None
            selected_features_info = {}
            for step_name, step_obj in feature_selection_pipeline.named_steps.items():
                if isinstance(step_obj, FeatureSelection) and step_name in ['PCA', 'ANOVA', 'LASSO']:
                    fs_step = step_obj
                    if self.use_pca and hasattr(fs_step, 'pca') and getattr(fs_step, 'pca', None) is not None:
                        pca_obj = getattr(fs_step, 'pca', None)
                        selected_features_info = {
                            'type': 'pca',
                            'components': getattr(pca_obj, 'components_', None),
                            'explained_variance': getattr(pca_obj, 'explained_variance_ratio_', None),
                            'n_components': getattr(pca_obj, 'n_components_', None)
                        }
                    elif self.use_anova and hasattr(fs_step, 'anova_scores'):
                        selected_features_info = {
                            'type': 'anova',
                            'scores': getattr(fs_step, 'anova_scores', None),
                            'k_features': getattr(fs_step, 'k', None)
                        }
                    elif self.use_lasso and hasattr(fs_step, 'lasso_coef'):
                        selected_features_info = {
                            'type': 'lasso',
                            'coefficients': getattr(fs_step, 'lasso_coef', None),
                            'alpha': getattr(fs_step, 'alpha', None),
                            'selected_features': getattr(fs_step, 'feature_names', [])
                        }
                    break
            # Obtém os nomes das features transformadas
            if fs_step and hasattr(fs_step, 'get_feature_names'):
                feature_names = fs_step.get_feature_names()
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(np.asarray(X_train_selected).shape[1])]
            else:
                if hasattr(X_train, 'columns') and not isinstance(X_train, list):
                    feature_names = list(X_train.columns)
                else:
                    feature_names = [f"feature_{i}" for i in range(np.asarray(X_train_selected).shape[1])]
            # Adiciona informações extras para PCA
            if self.use_pca and hasattr(fs_step, 'pca') and getattr(fs_step, 'pca', None) is not None:
                pca_obj = getattr(fs_step, 'pca', None)
                components = getattr(pca_obj, 'components_', None)
                if components is not None:
                    feature_importance = np.abs(components)
                    top_features_per_component = []
                    if hasattr(X_train, 'columns') and not isinstance(X_train, list):
                        original_feature_names = list(X_train.columns)
                    else:
                        original_feature_names = [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
                    for i in range(components.shape[0]):
                        top_features_idx = np.argsort(feature_importance[i])[-3:][::-1]
                        top_features = [(original_feature_names[j], components[i][j]) for j in top_features_idx]
                        top_features_per_component.append(top_features)
                    selected_features_info['top_features'] = top_features_per_component
            # Cria DataFrame com nomes das features
            transformed_feature_names_df = pd.DataFrame(X_train_selected, columns=feature_names)
            # Adiciona pca_info se PCA foi aplicado
            pca_info = None
            if self.use_pca and fs_step is not None and hasattr(fs_step, 'get_transformation_info'):
                transf_info = fs_step.get_transformation_info()
                pca_info = transf_info.get('pca_info', None)
            feature_selection_info = {
                'method': 'pca' if self.use_pca else 'anova' if self.use_anova else 'lasso' if self.use_lasso else None,
                'feature_names': feature_names,
                'original_features': list(X_train.columns) if hasattr(X_train, 'columns') and not isinstance(X_train, list) else None,
                'transformer': fs_step,
                'selected_features_info': selected_features_info,
                'pca_info': pca_info
            }
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            feature_selection_filename = f'Features_Selected_{current_datetime}.csv'
            features_selected = os.path.join(results_folder, feature_selection_filename)
            transformed_feature_names_df.to_csv(features_selected, index=False)
            self.logger.info(f"See the selected features at: {features_selected}")
        else:
            X_train_selected, X_test_selected = np.asarray(X_train), np.asarray(X_test)
            if hasattr(X_train, 'columns') and not isinstance(X_train, list):
                feature_names = list(X_train.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(np.asarray(X_train_selected).shape[1])]
            feature_selection_info = {
                'method': None,
                'feature_names': feature_names,
                'original_features': feature_names,
                'transformer': None,
                'selected_features_info': {}
            }
        return np.asarray(X_train_selected), np.asarray(X_test_selected), np.asarray(y_train), np.asarray(y_test), feature_selection_pipeline, feature_selection_info, feature_names

    #Condicao de parada optuna
    def early_stopping_callback(self, study, trial):
        if trial.number > 5 and study.best_value > 0.98:
            study.stop()

    #configuracoes optuna timeout 1200 n_trial 20
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Otimiza hiperparâmetros usando Optuna.
        Args:
            X (np.ndarray): Dados de entrada.
            y (np.ndarray): Labels.
        Returns:
            Tuple: (study, best_model)
        """
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        colored_message = f"[{Fore.GREEN}{now}{Style.RESET_ALL}] Optimizing hyperparameters..."
        logging.info(colored_message)
        hyperparameters = Hyperparameters(X, y)
        # INFO = 20, pois optuna.logging.INFO não é exportado
        optuna.logging.set_verbosity(20)
        optuna.logging.disable_default_handler()
        study = optuna.create_study(study_name="distributed-study", direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(hyperparameters, timeout=1200, gc_after_trial=True, n_trials=20, show_progress_bar=True, catch=(ValueError,), callbacks=[self.early_stopping_callback])
        best_model = study.best_trial.user_attrs.get("final_model", None)
        return study, best_model

    def evaluate_model(self, model, X_test, y_test):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        colored_message = f"[{Fore.GREEN}{now}{Style.RESET_ALL}] Evaluating model..."
        logging.info(colored_message)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidade da classe positiva
     
        report = classification_report(y_test, y_pred, target_names=["Benign", "Malware"], output_dict=True)

        self.logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
        
        # Gerar gráficos de avaliação
        self._generate_evaluation_plots(y_test, y_pred, y_pred_proba)
        
        return report, y_test, y_pred

    def _generate_evaluation_plots(self, y_test, y_pred, y_pred_proba):
        """Gerar gráficos de matriz de confusão e ROC/AUC"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
            
            # 1. Matriz de Confusão
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Benign', 'Malware'], 
                       yticklabels=['Benign', 'Malware'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            confusion_matrix_path = os.path.join('results', 'confusion_matrix.png')
            plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Confusion matrix saved: {confusion_matrix_path}")
            
            # 2. Curva ROC/AUC
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            roc_curve_path = os.path.join('results', 'roc_curve.png')
            plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"ROC curve saved: {roc_curve_path}")
            
            # 3. Curva Precisão-Recall
            plt.figure(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            plt.plot(recall, precision, color='green', lw=2, 
                    label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
            plt.axhline(y=1, color='navy', lw=2, linestyle='--', label='Perfect Precision')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            precision_recall_path = os.path.join('results', 'precision_recall_curve.png')
            plt.savefig(precision_recall_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Precision-Recall curve saved: {precision_recall_path}")
            
            # 4. Distribuição de Probabilidades
            plt.figure(figsize=(10, 6))
            
            # Separar probabilidades por classe real
            benign_probs = y_pred_proba[y_test == 0]
            malware_probs = y_pred_proba[y_test == 1]
            
            plt.hist(benign_probs, bins=30, alpha=0.7, label='Benign', color='blue', density=True)
            plt.hist(malware_probs, bins=30, alpha=0.7, label='Malware', color='red', density=True)
            plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Density')
            plt.title('Distribution of Predicted Probabilities')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            prob_dist_path = os.path.join('results', 'probability_distribution.png')
            plt.savefig(prob_dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Probability distribution saved: {prob_dist_path}")
            
            # 5. Métricas por Classe (Bar Plot)
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            # Calcular métricas por classe
            precision_per_class = precision_score(y_test, y_pred, average=None)
            recall_per_class = recall_score(y_test, y_pred, average=None)
            f1_per_class = f1_score(y_test, y_pred, average=None)
            
            classes = ['Benign', 'Malware']
            metrics = ['Precision', 'Recall', 'F1-Score']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(classes))
            width = 0.25
            
            ax.bar(x - width, [precision_per_class[0], precision_per_class[1]], width, label='Precision', alpha=0.8)
            ax.bar(x, [recall_per_class[0], recall_per_class[1]], width, label='Recall', alpha=0.8)
            ax.bar(x + width, [f1_per_class[0], f1_per_class[1]], width, label='F1-Score', alpha=0.8)
            
            ax.set_xlabel('Classes')
            ax.set_ylabel('Score')
            ax.set_title('Performance Metrics by Class')
            ax.set_xticks(x)
            ax.set_xticklabels(classes)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Adicionar valores nas barras
            for i, (prec, rec, f1) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
                ax.text(i - width, prec + 0.01, f'{prec:.3f}', ha='center', va='bottom')
                ax.text(i, rec + 0.01, f'{rec:.3f}', ha='center', va='bottom')
                ax.text(i + width, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            metrics_by_class_path = os.path.join('results', 'metrics_by_class.png')
            plt.savefig(metrics_by_class_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Metrics by class saved: {metrics_by_class_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not generate evaluation plots: {e}")

    def start_mlflow_ui(self):
        # Usando subprocess para iniciar o servidor MLflow de forma independente
        subprocess.Popen(["mlflow", "ui"])

    def open_browser(self):
        time.sleep(5)  # Espera 5 segundos para o servidor iniciar
        webbrowser.open_new_tab("http://localhost:5000")

    def measure_performance(self):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # em MB
        return start_time, start_memory

    def save_performance(self, step_name, start_time, start_memory):
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # em MB
        elapsed_time = end_time - start_time
        memory_usage = end_memory - start_memory

        self.measurements.append({
            'Step Name': step_name,
            'Elapsed Time (seconds)': f"{elapsed_time:.2f}",
            'Memory Usage (MB)': f"{memory_usage:.2f}"
        })

    def export_to_csv(self, filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Step Name', 'Elapsed Time (seconds)', 'Memory Usage (MB)'])
            writer.writeheader()
            for measurement in self.measurements:
                writer.writerow(measurement)

    def plot_performance(self, filename='performance_metrics.jpg'):
        step_names = [m['Step Name'] for m in self.measurements]
        elapsed_times = [float(m['Elapsed Time (seconds)']) for m in self.measurements]
        memory_usages = [float(m['Memory Usage (MB)']) for m in self.measurements]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_title('Performance Metrics per Step')
        ax1.set_xlabel('Step Name')
        ax1.set_ylabel('Elapsed Time (seconds)', color='tab:blue')
        bars = ax1.bar(step_names, elapsed_times, color='tab:blue', alpha=0.6, label='Elapsed Time (seconds)')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        for bar in bars:
            yval = bar.get_height()
            ax1.annotate(f'{yval:.2f}', 
                         xy=(bar.get_x() + bar.get_width() / 2, yval), 
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords='offset points', 
                         ha='center', va='bottom', fontsize=10, color='blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Memory Usage (MB)', color='tab:green')
        line = ax2.plot(step_names, memory_usages, color='tab:green', marker='o', label='Memory Usage (MB)')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        for i, txt in enumerate(memory_usages):
            ax2.annotate(f'{txt:.2f}', 
                         xy=(step_names[i], memory_usages[i]), 
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords='offset points', 
                         ha='center', va='bottom', fontsize=10, color='green')

        # Adicionar a legenda
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        fig.tight_layout()

        # Salvar o gráfico em um arquivo JPG
        plt.savefig(filename, format='jpg')

        # Fechar a figura para liberar memória
        plt.close()



    def print_optimized_pipeline(self, pipeline=None):
        """
        Exibe a estrutura do pipeline de forma absolutamente clara,
        sem duplicações ou ambiguidades
        """
        try:
            pipeline = pipeline if pipeline is not None else getattr(self, 'pipeline', None)
        
            if pipeline is None:
                print("⚠️ Nenhum pipeline disponível para visualização")
                return
            
            if not hasattr(pipeline, 'steps'):
                print("⚠️ O objeto fornecido não é um Pipeline válido")
                return
            
            print("\n" + "="*40)
            print("PIPELINE RESUME".center(40))
            print("="*40)
        
            # Dicionário para rastrear componentes já impressos
            printed = {}
        
            self._print_optimized_component(pipeline, printed)
        
            print("="*40 + "\n")
        
        except Exception as e:
            print(f"\n❌ Erro ao exibir a estrutura do pipeline: {str(e)}\n")

    def _print_optimized_component(self, component, printed, indent=0, is_last=False, step_name=None):
        """Imprime um componente do pipeline de forma otimizada"""
        # Identificador único baseado no objeto e nível de indentação
        component_id = f"{id(component)}-{indent}"
    
        if component_id in printed:
            return
        
        printed[component_id] = True
    
        prefix = "    " * indent
        connector = "└─" if is_last else "├─"
    
        # Nome de exibição
        display_name = step_name if step_name else component.__class__.__name__
    
        # Imprime o nome do componente
        print(f"{prefix}{connector} {display_name}")
    
        # Se for um Pipeline ou VotingClassifier, imprime parâmetros específicos
        if hasattr(component, 'steps') or hasattr(component, 'estimators'):
            params = component.get_params()
        
            # Filtra apenas parâmetros relevantes
            relevant_params = {
                #'verbose': params.get('verbose'),
                'voting': params.get('voting'),
                'flatten_transform': params.get('flatten_transform')
            }
        
            # Imprime parâmetros não nulos
            param_items = [(k, v) for k, v in relevant_params.items() if v is not None]
            for i, (param, value) in enumerate(param_items):
                param_connector = "└─" if i == len(param_items) - 1 else "├─"
                print(f"{prefix}    {param_connector} {param}: {value}")
    
        # Imprime parâmetros específicos do componente
        self._print_component_params(component, indent)
    
        # Processa subcomponentes
        if hasattr(component, 'steps'):
            for i, (name, obj) in enumerate(component.steps):
                self._print_optimized_component(
                    obj, printed, indent + 1,
                    is_last=(i == len(component.steps) - 1),
                    step_name=f"{name} ({obj.__class__.__name__})"
                )
        elif hasattr(component, 'estimators'):
            for name, estimator in component.estimators:
                self._print_optimized_component(
                    estimator, printed, indent + 1,
                    step_name=f"{name} ({estimator.__class__.__name__})"
                )

    def _print_component_params(self, component, indent):
        """Imprime apenas os parâmetros mais relevantes de cada componente"""
        prefix = "    " * (indent + 1)
        params = component.get_params()
    
        # Mapeamento de parâmetros relevantes por tipo de componente
        param_mapping = {
            # Pré-processamento
            'DataCleaning': ['remove_duplicates', 'remove_missing_values', 'remove_outliers'],
            'DataTransformation': ['do_label_encode', 'label', 'one_hot_encoder'],
            'FeatureSelection': ['alpha', 'k_features', 'n_components'],
            # Balanceamento
            'SMOTE': ['balancing', 'k_neighbors'],
            'RandomUnderSampler': ['balancing'],
            # Classificadores
            'RandomForestClassifier': [
                'n_estimators',
                'max_depth',
                'min_samples_split',
                'min_samples_leaf',
                'max_features',
                'random_state'
            ],
            'DecisionTreeClassifier': [
                'max_depth',
                'min_samples_split',
                'min_samples_leaf',
                'max_features',
                'random_state'
            ],
            'ExtraTreesClassifier': [
                'n_estimators',
                'max_depth',
                'min_samples_split',
                'max_features',
                'random_state'
            ],
            'LGBMClassifier': [
                'learning_rate',
                'n_estimators',
                'num_leaves',
                'max_depth',
                'min_child_samples',
                'subsample',
                'colsample_bytree',
                'random_state'
            ],
            'CatBoostClassifier': [
                'iterations',
                'learning_rate',
                'depth',
                'l2_leaf_reg',
                'random_strength',
                'random_state'
            ],
    
            # Meta-classificadores
            'StackingClassifier': ['stack_method', 'passthrough'],
    
            
        }
    
        component_type = component.__class__.__name__
        relevant_params = param_mapping.get(component_type, [])
    
        # Filtra e imprime os parâmetros
        param_items = [(p, params[p]) for p in relevant_params if p in params and params[p] is not None]
        for i, (param, value) in enumerate(param_items):
            param_connector = "└─" if i == len(param_items) - 1 else "├─"
            print(f"{prefix}{param_connector} {param}: {value}")

    # Função para logar artefatos se o arquivo existir e não for None
    def log_artifact_if_exists(self,file_path, artifact_path):
        if file_path and os.path.exists(file_path):
            mlflow.log_artifact(file_path, artifact_path)
    
    def _create_slice_plot_matplotlib(self, study, output_path):
        """Create a simple slice plot using matplotlib as fallback"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get trials data
            trials_df = study.trials_dataframe()
            
            if trials_df.empty:
                return
            
            # Get numeric parameters only
            numeric_params = []
            for col in trials_df.columns:
                if col.startswith('params_') and trials_df[col].dtype in ['float64', 'int64']:
                    param_name = col.replace('params_', '')
                    numeric_params.append(param_name)
            
            if not numeric_params:
                return
            
            # Create subplots for each parameter
            n_params = min(len(numeric_params), 4)  # Limit to 4 parameters
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, param in enumerate(numeric_params[:n_params]):
                if i >= 4:
                    break
                    
                param_col = f'params_{param}'
                values = trials_df[param_col].dropna()
                scores = trials_df['value'].dropna()
                
                if len(values) > 0:
                    ax = axes[i]
                    ax.scatter(values, scores, alpha=0.6)
                    ax.set_xlabel(param)
                    ax.set_ylabel('Score')
                    ax.set_title(f'{param} vs Score')
                    ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(n_params, 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Matplotlib slice plot creation failed: {str(e)}")


    def run(self):

        validator = DatasetValidation(self.dataset_url, self.label)
        results_folder= self.create_results_folder()
        
        if not validator.validate_dataset():
            return None 

        # Iniciando o MLflow UI em segundo plano
        self.start_mlflow_ui()
        self.open_browser()
        mlflow.set_experiment("MH-AutoML")

        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #report_filename = f'report_{current_datetime}.pdf'
        
        # Step 1: Data info
        start_time_step1, start_memory_step1 = self.measure_performance()
        dataset_df = validator.load_data()   
        display_data = self.display_data_info(dataset_df)
        self.save_performance('Data info', start_time_step1, start_memory_step1)

      

        # Step 2: Preprocessing
        start_time_step2, start_memory_step2 = self.measure_performance()
        X, y, preprocessor = self.preprocess(dataset_df, self.label)
        self.save_performance('Preprocessing', start_time_step2, start_memory_step2)


        # Step 3: Dimensionality Reduction
        start_time_step3, start_memory_step3 = self.measure_performance()
        if dataset_df.shape[1] > 50:
            # Habilitar o método de seleção de características com base no valor passado
            if self.feature_selection_method == 'PCA':
                self.enable_pca()
            elif self.feature_selection_method == 'LASSO':
                self.enable_lasso()
            elif self.feature_selection_method == 'ANOVA':
                self.enable_anova()
        else:
            colored_message_not_features = f"[{Fore.YELLOW}The number of features in the dataset is less than or equal to 50, there is no need to reduce it.{Style.RESET_ALL}]"
            self.logger.warning(colored_message_not_features)
           
        X_train_selected, X_test_selected, y_train, y_test, selection_pipeline, feature_selection_info, feature_names = self.feature_selection(X, y)
       
        self.save_performance('Feature Eng', start_time_step3, start_memory_step3)
        

        # Step 4: Hyperparameter Optimization
        start_time_step4, start_memory_step4 = self.measure_performance()
        hyperparameters = Hyperparameters(X_train_selected, y_train)
        study, best_model = self.optimize_hyperparameters(X_train_selected, y_train)   
        # Generate and get paths to Optuna visualizations
        optuna_visualizations = self.log_optuna_visualizations(study)
        formatted_ranking, df_results, select_results = hyperparameters.calculate_metrics_and_save_results(study, X_train_selected, X_test_selected, y_train, y_test)
        steps = [('Preprocessor', preprocessor),("Feature engineering",selection_pipeline), ("Ensemble Classifier", best_model)]
        pipeline = Pipeline(steps)
        model_name = best_model.estimators[0][0]
        model_instance = best_model.estimators[0][1]
        model_params = model_instance.get_params()

        self.logger.info(f"Best Model: {model_name}, Best Parameters: {model_params}")
        # Fit the best model with the training data using X_train_pca and y_train
        best_model.fit(X_train_selected, y_train)
        y_pred = best_model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)


        self.save_performance('Hyperparameter', start_time_step4, start_memory_step4)

        # Step 5: Interpretability:
        start_time_step5, start_memory_step5 = self.measure_performance()
        interpretability = Interpretability(
            best_model, 
            X_train_selected, 
            X_test_selected, 
            y_train,
            feature_selection_info,
            results_folder
        )

        # Gera as explicações do modelo
        shap_exp_filepath, exp_filepath, lime_exp_filepath = interpretability.explain_model()
        # Gerar PDP para as 5 features mais importantes (padrão)
       

        #if self.use_lasso:
            # SHAP explanation - usando feature_names que já temos
            #shap_values = Interpretability.explanation(study.best_trial, X_train_selected, y_train, X_test_selected, feature_names)
    
            # LIME explanation - também usando feature_names
            #lime_explanation = interpretability.lime_explanation(study.best_trial, X_train_selected, y_train, X_test_selected, feature_names)

        self.save_performance('Interpretability', start_time_step5, start_memory_step5)

        # Step 6: Evaluation and report
        start_time_step6, start_memory_step6 = self.measure_performance()
     
        # Concatenar os dados de treino e a variável alvo em um DataFrame do Pandas
        df = pd.DataFrame(data=X_train_selected, columns=feature_names)  # Usa feature_names diretamente
        df['class'] = y_train

        # Salvar o DataFrame em um arquivo CSV
        treino_filename = f'treino_{current_datetime}.csv'
        filepath_treino = os.path.join(results_folder, treino_filename)
        df.to_csv(filepath_treino, index=False)
        self.logger.info(f"Dados de treino salvos em {filepath_treino}")

        #######################################################################################

        model_filename = f'best_model_{current_datetime}.pkl'
        model_filepath = os.path.join(results_folder, model_filename)
        with open(model_filepath, 'wb') as model_file:
            pickle.dump(best_model, model_file)

        print("\n")
        self.logger.info(f"See your Best Model at: {model_filepath}")
        report, y_test, y_pred = self.evaluate_model(best_model, X_test_selected, y_test)
        print("\n")

        
        #self.logger.info("Pipeline Configs")
        #self.logger.info(pipeline)
        self.print_optimized_pipeline(pipeline)
 
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f'report_{current_datetime}.html'
        # Concatenando o horário atual com a URL do conjunto de dados
        run_name = f"{current_datetime}_{self.dataset_url}"



        # geração do relatório HTML:
        report_generator = ReportGenerator(results_folder)
        report_filename = report_generator.generate_report(
            pipeline=pipeline,
            display_data=display_data,
            study=study,
            best_model=best_model,
            model_name=model_name,
            model_params=model_params,
            select_results=select_results,
            report=report,
            y_test=y_test,
            y_pred=y_pred,
            feature_names=feature_names,
            feature_selection_info=feature_selection_info,
            shap_exp_filepath=shap_exp_filepath,
            exp_filepath=exp_filepath,
            lime_exp_filepath=lime_exp_filepath
        )
        
        # geração do relatório PDF:
        # Generate PDF report
        pdf_generator = PDFReportGenerator(results_folder)
        pdf_filename = pdf_generator.generate_pdf_report(
            pipeline=pipeline,
            display_data=display_data,
            study=study,
            best_model=best_model,
            model_name=model_name,
            model_params=model_params,
            select_results=select_results,
            report=report,
            y_test=y_test,
            y_pred=y_pred,
            feature_names=feature_names,
            feature_selection_info=feature_selection_info,
            shap_exp_filepath=shap_exp_filepath,
            exp_filepath=exp_filepath,
            lime_exp_filepath=lime_exp_filepath
        )
        self.logger.info(f"PDF report generated: {pdf_filename}")

        self.save_performance('Evaluation', start_time_step6, start_memory_step6)
        self.export_to_csv('results/performance_summary.csv')
        self.plot_performance('results/performance_metrics.jpg')
                ###########################################################################

        dataset: PandasDataset = mlflow.data.from_pandas(dataset_df, source=self.dataset_url)
        # Registrando os melhores hiperparâmetros no MLflow
        with mlflow.start_run(run_name=run_name,log_system_metrics=True) as run:
            mlflow.set_tag("experiment_type", "Auto_ML")           
            mlflow.log_input(dataset, context="training")
            mlflow.log_params(model_params)
            # Registre as métricas do relatório de classificação

            # Métricas básicas 
            mlflow.log_metric("accuracy", report["accuracy"])
            mlflow.log_metric("precision", report["macro avg"]["precision"])
            mlflow.log_metric("recall", report["macro avg"]["recall"])
            mlflow.log_metric("f1", report["macro avg"]["f1-score"])
            mlflow.log_metric("mcc", matthews_corrcoef(y_test, y_pred))

            # Métricas específicas para malware (classe positiva)
            #mlflow.log_metric("malware_precision", report["1"]["precision"])
            #mlflow.log_metric("malware_recall", report["1"]["recall"])
            #mlflow.log_metric("malware_f1", report["1"]["f1-score"])

            # Métricas para classe benigna
            #mlflow.log_metric("benign_precision", report["0"]["precision"]) 
            #mlflow.log_metric("benign_recall", report["0"]["recall"])

            # Métricas avançadas de detecção
            mlflow.log_metric("false_positive_rate", confusion_matrix(y_test, y_pred)[0,1] / (confusion_matrix(y_test, y_pred)[0,1] + confusion_matrix(y_test, y_pred)[0,0]))
            mlflow.log_metric("false_negative_rate", confusion_matrix(y_test, y_pred)[1,0] / (confusion_matrix(y_test, y_pred)[1,0] + confusion_matrix(y_test, y_pred)[1,1]))
            mlflow.log_metric("detection_rate", confusion_matrix(y_test, y_pred)[1,1] / sum(confusion_matrix(y_test, y_pred)[1]))

            # Métricas para conjuntos desbalanceados
            mlflow.log_metric("average_precision", average_precision_score(y_test, y_pred))
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred))

            # F2-Score (dá mais peso ao recall - importante para detecção)
            mlflow.log_metric("f2_score", fbeta_score(y_test, y_pred, beta=2))

            # Taxa de descoberta verdadeira (True Positive Rate)
            mlflow.log_metric("true_positive_rate", recall_score(y_test, y_pred, pos_label=1))

            # Razão de falsos alarmes
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            mlflow.log_metric("false_alarm_ratio", fp / (fp + tp))
            #mlflow.log_metric(key="train_loss", value=train_loss, step=epoch, timestamp=now)


            #mlflow.log_artifact(self.dataset_url, artifact_path="datasets")

            # Log de artefatos em cada diretório
            
            self.log_artifact_if_exists(estimator_html_repr(pipeline), "00_Data_info")
            self.log_artifact_if_exists("results/missing_values_heatmap.png", "01_preprocessing")
            self.log_artifact_if_exists("results/clean_missing_values_heatmap.png", "01_preprocessing")
            self.log_artifact_if_exists("results/pca_biplot.png", "02_feature_engineering")
            self.log_artifact_if_exists("results/lasso_feature_importance.png", "02_feature_engineering")
            self.log_artifact_if_exists("results/anova_feature_importance.png", "02_feature_engineering")
            self.log_artifact_if_exists("results/train_test_distribution.png", "02_feature_engineering")
            
            # Log PCA components plot if it exists
            import glob
            pca_components_files = glob.glob("results/pca_components_*.png")
            for pca_file in pca_components_files:
                self.log_artifact_if_exists(pca_file, "02_feature_engineering")
            self.log_artifact_if_exists("results/Hyperparameters_Results.csv", "03_model_optimization")
            self.log_artifact_if_exists("results/Models_Ranking.csv", "03_model_optimization")


            if optuna_visualizations:
                if optuna_visualizations.get('param_importance_png'):
                    self.log_artifact_if_exists(optuna_visualizations['param_importance_png'], "03_model_optimization")
                if optuna_visualizations.get('optimization_history_png'):
                    self.log_artifact_if_exists(optuna_visualizations['optimization_history_png'], "03_model_optimization")
                if optuna_visualizations.get('param_importance_html'):
                    self.log_artifact_if_exists(optuna_visualizations['param_importance_html'], "03_model_optimization")
                if optuna_visualizations.get('optimization_history_html'):
                    self.log_artifact_if_exists(optuna_visualizations['optimization_history_html'], "03_model_optimization")
                if optuna_visualizations.get('slice_plot'):
                    self.log_artifact_if_exists(optuna_visualizations['slice_plot'], "03_model_optimization")
                if optuna_visualizations.get('parallel_coordinate'):
                    self.log_artifact_if_exists(optuna_visualizations['parallel_coordinate'], "03_model_optimization")
                if optuna_visualizations.get('trials_data'):
                    self.log_artifact_if_exists(optuna_visualizations['trials_data'], "03_model_optimization")

            self.log_artifact_if_exists('results/performance_metrics.jpg', "04_evaluation_metrics")
            self.log_artifact_if_exists(model_filepath, "04_evaluation_metrics")
            self.log_artifact_if_exists('results/confusion_matrix.png', "04_evaluation_metrics")
            self.log_artifact_if_exists('results/roc_curve.png', "04_evaluation_metrics")
            self.log_artifact_if_exists('results/precision_recall_curve.png', "04_evaluation_metrics")
            self.log_artifact_if_exists('results/probability_distribution.png', "04_evaluation_metrics")
            self.log_artifact_if_exists('results/metrics_by_class.png', "04_evaluation_metrics")

            self.log_artifact_if_exists("results/shap_summary_plot.png", "05_interpretability")
           
            # Log SHAP summary plot específico do modelo se existir
            shap_summary_files = glob.glob("results/shap_summary_plot_*.png")
            for shap_file in shap_summary_files:
                self.log_artifact_if_exists(shap_file, "05_interpretability")
           
            # Verificação adicional para os arquivos específicos
            if lime_exp_filepath and os.path.exists(lime_exp_filepath):
                mlflow.log_artifact(lime_exp_filepath, artifact_path="05_interpretability")

            if exp_filepath and os.path.exists(exp_filepath):
                mlflow.log_artifact(exp_filepath, artifact_path="05_interpretability")

            if shap_exp_filepath and os.path.exists(shap_exp_filepath):
                mlflow.log_artifact(shap_exp_filepath, artifact_path="05_interpretability")



     
            # Log do relatório se existir
            if report_filename and os.path.exists(report_filename):
                mlflow.log_artifact(report_filename, artifact_path="report")

            # Log do relatório PDF se existir
            if pdf_filename and os.path.exists(pdf_filename):
                mlflow.log_artifact(pdf_filename, artifact_path="report")

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="MH_Best_Model",
                registered_model_name=model_name,
            )

        run = mlflow.get_run(mlflow.last_active_run().info.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        # Ensure the dataset source file exists before loading
        dataset_path = dataset_source.uri
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.abspath(os.path.join("Datasets", os.path.basename(dataset_path)))

        if os.path.exists(dataset_path):
            dataset_source.load()
        else:
            raise FileNotFoundError(f"No such file or directory: '{dataset_path}'")

        mlflow.end_run() 

        print("acess : http://localhost:5000")

        
        self.logger.info("Done!")
