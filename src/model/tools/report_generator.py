import os
import io
import base64
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import set_config
from sklearn.utils import estimator_html_repr
import optuna

class ReportGenerator:
    def __init__(self, results_folder="results"):
        """Initialize the report generator with output directory"""
        self.results_folder = results_folder
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def generate_report(self, pipeline, display_data, study, best_model, model_name, model_params, 
                       select_results, report, y_test, y_pred, feature_names, feature_selection_info,
                       shap_exp_filepath, exp_filepath, lime_exp_filepath):
        """Generate a comprehensive HTML report of the AutoML pipeline results"""
        
        # Create timestamped report filename
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = os.path.join(self.results_folder, f'report_{current_datetime}.html')

        with open(report_filename, 'w', encoding='utf-8') as report_file:
            # HTML Header with modern styling
            report_file.write(f"""
            <html>
            <head>
                <title>AutoML Analysis Report</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    h1, h2, h3 {{
                        color: #2c3e50;
                        border-bottom: 2px solid #f0f0f0;
                        padding-bottom: 10px;
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #f2f2f2;
                        font-weight: bold;
                    }}
                    .metric-box {{
                        background-color: #f8f9fa;
                        border: 1px solid #e9ecef;
                        border-radius: 5px;
                        padding: 15px;
                        margin: 10px 0;
                    }}
                    .image-container {{
                        text-align: center;
                        margin: 20px 0;
                    }}
                    .image-container img {{
                        max-width: 100%;
                        height: auto;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                    }}
                    section {{
                        margin: 30px 0;
                        padding: 20px;
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    pre {{
                        background-color: #f8f9fa;
                        border: 1px solid #e9ecef;
                        border-radius: 5px;
                        padding: 15px;
                        overflow-x: auto;
                        white-space: pre-wrap;
                    }}
                    a {{
                        color: #007bff;
                        text-decoration: none;
                    }}
                    a:hover {{
                        text-decoration: underline;
                    }}
                    ul {{
                        list-style-type: none;
                        padding: 0;
                    }}
                    li {{
                        margin: 10px 0;
                        padding: 10px;
                        background-color: #f8f9fa;
                        border-radius: 5px;
                    }}
                </style>
            </head>
            <body>
                <h1>AutoML Analysis Report</h1>
                <p><em>Generated on: {current_datetime}</em></p>
            """)

            # Pipeline Configuration Section
            report_file.write("""
            <section>
                <h2>Pipeline Configuration</h2>
                <div class="metric-box">
            """)
            if pipeline:
                report_file.write("<h4>Pipeline Steps</h4>")
                for i, step in enumerate(pipeline.steps):
                    report_file.write(f"<p><b>Step {i+1}:</b> {step[0]} - {step[1].__class__.__name__}</p>")
            report_file.write("</div></section>")

            # Data Analysis Section
            report_file.write("""
            <section>
                <h2>Data Analysis</h2>
                <div class="metric-box">
            """)
            if display_data is not None and hasattr(display_data, 'dataset'):
                dataset = display_data.dataset
                report_file.write("<h4>Dataset Information</h4>")
                report_file.write(f"<p><b>Shape:</b> {dataset.shape}</p>")
                report_file.write(f"<p><b>Columns:</b> {len(dataset.columns)}</p>")
                report_file.write(f"<p><b>Memory Usage:</b> {dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB</p>")
                
                # Data types
                report_file.write("<h4>Data Types</h4>")
                report_file.write("<table><tr><th>Column</th><th>Type</th><th>Non-Null Count</th></tr>")
                for col, dtype in dataset.dtypes.items():
                    non_null = dataset[col].count()
                    report_file.write(f"<tr><td>{col}</td><td>{dtype}</td><td>{non_null}</td></tr>")
                report_file.write("</table>")
                
                # Missing values
                missing_values = dataset.isnull().sum()
                if missing_values.sum() > 0:
                    report_file.write("<h4>Missing Values</h4>")
                    report_file.write("<table><tr><th>Column</th><th>Missing Count</th><th>Percentage</th></tr>")
                    for col, count in missing_values[missing_values > 0].items():
                        percentage = (count / len(dataset)) * 100
                        report_file.write(f"<tr><td>{col}</td><td>{count}</td><td>{percentage:.2f}%</td></tr>")
                    report_file.write("</table>")
                else:
                    report_file.write("<p><em>No missing values found in the dataset.</em></p>")
                
                # Display DataInfo results if available
                if hasattr(display_data, 'info_table_result') and display_data.info_table_result is not None:
                    report_file.write("<h4>Dataset Overview</h4>")
                    report_file.write(display_data.info_table_result.to_html(index=False))
                
                if hasattr(display_data, 'data_types_result') and display_data.data_types_result is not None:
                    report_file.write("<h4>Data Type Distribution</h4>")
                    report_file.write(display_data.data_types_result.to_html(index=False))
                
                if hasattr(display_data, 'balance_info_result') and display_data.balance_info_result is not None:
                    report_file.write("<h4>Class Distribution</h4>")
                    report_file.write(display_data.balance_info_result.to_html(index=False))
                
                if hasattr(display_data, 'duplicates_missing_result') and display_data.duplicates_missing_result is not None:
                    report_file.write("<h4>Data Quality</h4>")
                    report_file.write(display_data.duplicates_missing_result.to_html(index=False))
                
                if hasattr(display_data, 'features_info_result') and display_data.features_info_result is not None:
                    report_file.write("<h4>Features Information</h4>")
                    report_file.write(display_data.features_info_result.to_html(index=False))
            report_file.write("</div></section>")

            # 01_preprocessing Section
            report_file.write("""
            <section>
                <h2>01_preprocessing</h2>
                <div class="metric-box">
                    <p>Data preprocessing artifacts and visualizations:</p>
                </div>
            """)
            # Missing values heatmaps
            for file in os.listdir(self.results_folder):
                if file == "clean_missing_values_heatmap.png":
                    report_file.write(f"<div class='image-container'><img src='{file}' alt='Clean Missing Values Heatmap'><p>Clean Missing Values Heatmap</p></div>")
                elif file == "missing_values_heatmap.png":
                    report_file.write(f"<div class='image-container'><img src='{file}' alt='Missing Values Heatmap'><p>Missing Values Heatmap</p></div>")
            report_file.write("</section>")

            # 02_feature_engineering Section
            report_file.write("""
            <section>
                <h2>02_feature_engineering</h2>
                <div class="metric-box">
                    <p>Feature engineering results and visualizations:</p>
                </div>
            """)
            # Feature selection method
            method = feature_selection_info.get('method', None)
            if method:
                report_file.write(f"<h4>Feature Selection Method: {method.upper()}</h4>")
                if method == 'lasso':
                    report_file.write("<p><b>Method used:</b> LASSO (L1 regularization)</p>")
                    selected_features_info = feature_selection_info.get('selected_features_info', {})
                    selected_features = selected_features_info.get('selected_features') or selected_features_info.get('features', [])
                    report_file.write("<h4>Selected Features</h4>")
                    if selected_features:
                        report_file.write("<table><tr><th>Feature</th></tr>")
                        for feat in selected_features:
                            feat_clean = str(feat).replace('_1.0', '')
                            report_file.write(f"<tr><td>{feat_clean}</td></tr>")
                        report_file.write("</table>")
                    else:
                        report_file.write("<p><em>No features were selected by this method.</em></p>")
                elif method == 'anova':
                    report_file.write("<p><b>Method used:</b> ANOVA F-test</p>")
                    selected_features_info = feature_selection_info.get('selected_features_info', {})
                    selected_features = selected_features_info.get('selected_features') or selected_features_info.get('features', [])
                    scores = selected_features_info.get('scores', [])
                    report_file.write("<h4>Selected Features and F-scores</h4>")
                    if selected_features:
                        report_file.write("<table><tr><th>Feature</th><th>F-score</th></tr>")
                        for feat, score in zip(selected_features, scores):
                            feat_clean = str(feat).replace('_1.0', '')
                            report_file.write(f"<tr><td>{feat_clean}</td><td>{score:.4f}</td></tr>")
                        report_file.write("</table>")
                    else:
                        report_file.write("<p><em>No features were selected by this method.</em></p>")
                elif method == 'pca':
                    report_file.write("<p><b>Method used:</b> Principal Component Analysis (PCA)</p>")
                    pca_info = feature_selection_info.get('pca_info', {})
                    if pca_info and 'components' in pca_info and 'feature_names' in pca_info:
                        report_file.write("<h4>PCA Components</h4>")
                        components = pca_info['components']
                        original_features = pca_info['feature_names']
                        report_file.write("<table><tr><th>Component</th>" + ''.join([f"<th>{str(f).replace('_1.0','')}</th>" for f in original_features]) + "</tr>")
                        for i, row in enumerate(components):
                            report_file.write(f"<tr><td>PC{i+1}</td>" + ''.join([f"<td>{val:.2f}</td>" for val in row]) + "</tr>")
            report_file.write("</table>")

            # Feature engineering visualizations
            for file in os.listdir(self.results_folder):
                if file == "lasso_feature_importance.png":
                    report_file.write(f"<div class='image-container'><img src='{file}' alt='LASSO Feature Importance'><p>LASSO Feature Importance</p></div>")
                elif file == "train_test_distribution.png":
                    report_file.write(f"<div class='image-container'><img src='{file}' alt='Train/Test Distribution'><p>Train/Test Distribution</p></div>")
            report_file.write("</section>")

            # 03_model_optimization Section
            report_file.write("""
            <section>
                <h2>03_model_optimization</h2>
                <div class="metric-box">
                    <p>Hyperparameter tuning and model selection results:</p>
                </div>
            """)
            # Best model info
            report_file.write("<h4>Best Model</h4>")
            if hasattr(best_model, 'estimators_'):
                report_file.write("<ul>")
                for name, est in best_model.named_estimators_.items():
                    report_file.write(f"<li><b>{name}</b>: {est.__class__.__name__} - Params: {est.get_params()}</li>")
                report_file.write("</ul>")
            else:
                report_file.write(f"<p><b>{best_model.__class__.__name__}</b> - Params: {best_model.get_params()}</p>")
            
            # Optimization artifacts
            for file in os.listdir(self.results_folder):
                if file == "Hyperparameters_Results.csv":
                    report_file.write(f"<p><a href='{file}' download>Download Hyperparameters Results (CSV)</a></p>")
                elif file == "Models_Ranking.csv":
                    report_file.write(f"<p><a href='{file}' download>Download Models Ranking (CSV)</a></p>")
                elif file == "optuna_optimization_history.html":
                    report_file.write(f"<p><a href='{file}' target='_blank'>View Optuna Optimization History (HTML)</a></p>")
                elif file == "optuna_optimization_history.png":
                    report_file.write(f"<div class='image-container'><img src='{file}' alt='Optuna Optimization History'><p>Optuna Optimization History</p></div>")
                elif file == "optuna_parallel_coordinate.html":
                    report_file.write(f"<p><a href='{file}' target='_blank'>View Optuna Parallel Coordinate (HTML)</a></p>")
                elif file == "optuna_parallel_coordinate.png":
                    report_file.write(f"<div class='image-container'><img src='{file}' alt='Optuna Parallel Coordinate'><p>Optuna Parallel Coordinate</p></div>")
                elif file == "optuna_param_importance.html":
                    report_file.write(f"<p><a href='{file}' target='_blank'>View Optuna Parameter Importance (HTML)</a></p>")
                elif file == "optuna_param_importance.png":
                    report_file.write(f"<div class='image-container'><img src='{file}' alt='Optuna Parameter Importance'><p>Optuna Parameter Importance</p></div>")
                elif file == "optuna_slice_plot.html":
                    report_file.write(f"<p><a href='{file}' target='_blank'>View Optuna Slice Plot (HTML)</a></p>")
                elif file == "optuna_slice_plot.png":
                    report_file.write(f"<div class='image-container'><img src='{file}' alt='Optuna Slice Plot'><p>Optuna Slice Plot</p></div>")
                elif file == "optuna_trials.csv":
                    report_file.write(f"<p><a href='{file}' download>Download Optuna Trials (CSV)</a></p>")
            report_file.write("</section>")

            # 04_evaluation_metrics Section
            report_file.write("""
            <section>
                <h2>04_evaluation_metrics</h2>
                <div class="metric-box">
                    <p>Model evaluation metrics and performance results:</p>
                </div>
            """)
            # Classification report
            if report is not None:
                report_file.write("<h4>Classification Report</h4>")
                report_file.write(f"<pre>{report}</pre>")
            
            # Best model file
            for file in os.listdir(self.results_folder):
                if file.startswith('best_model_') and file.endswith('.pkl'):
                    report_file.write(f"<p><a href='{file}' download>Download Best Model (PKL)</a></p>")
                elif file == "performance_metrics.jpg":
                    report_file.write(f"<div class='image-container'><img src='{file}' alt='Performance Metrics'><p>Performance Metrics</p></div>")
            report_file.write("</section>")

            # 05_interpretability Section
            report_file.write("""
            <section>
                <h2>05_interpretability</h2>
                <div class="metric-box">
                    <p>Model interpretability artifacts and explanations:</p>
                </div>
            """)
            # SHAP artifacts
            for file in os.listdir(self.results_folder):
                if file.startswith('shap_force_plot_') and file.endswith('.html'):
                    report_file.write(f"<p><a href='{file}' target='_blank'>View SHAP Force Plot (HTML)</a></p>")
                elif file.startswith('shap_force_plot_') and file.endswith('.png'):
                    report_file.write(f"<div class='image-container'><img src='{file}' alt='SHAP Force Plot'><p>SHAP Force Plot</p></div>")
                elif file.startswith('shap_summary_plot') and file.endswith('.png'):
                    report_file.write(f"<div class='image-container'><img src='{file}' alt='SHAP Summary Plot'><p>SHAP Summary Plot</p></div>")
            
            # LIME artifacts
            for file in os.listdir(self.results_folder):
                if file.startswith('lime_feature_importance_') and file.endswith('.png'):
                    report_file.write(f"<div class='image-container'><img src='{file}' alt='LIME Feature Importance'><p>LIME Feature Importance</p></div>")
                elif file.startswith('lime_interpretability_') and file.endswith('.html'):
                    report_file.write(f"<p><a href='{file}' target='_blank'>View LIME Interpretability (HTML)</a></p>")
                elif file.startswith('lime_interpretability_') and file.endswith('.png'):
                    report_file.write(f"<div class='image-container'><img src='{file}' alt='LIME Interpretability'><p>LIME Interpretability</p></div>")
            report_file.write("</section>")

            # Footer
            report_file.write(f"""
            <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #eee; text-align: center; color: #7f8c8d;">
                <p>AutoML Report generated by MH-AutoML System</p>
                <p>{current_datetime}</p>
            </footer>
            """)

            report_file.write("</body></html>")

        return report_filename