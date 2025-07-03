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
import subprocess
import tempfile
from pathlib import Path
import logging

class PDFReportGenerator:
    def __init__(self, results_folder="results"):
        """Initialize the PDF report generator with output directory"""
        self.results_folder = results_folder
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        self.logger = logging.getLogger(__name__)

    def html_to_image(self, html_path, output_path, width=1200, height=800):
        """Convert HTML file to PNG image using headless browser"""
        try:
            # Try using playwright (more reliable)
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page(viewport={'width': width, 'height': height})
                    page.goto(f'file://{os.path.abspath(html_path)}')
                    page.wait_for_load_state('networkidle')
                    page.screenshot(path=output_path, full_page=True)
                    browser.close()
                return True
            except ImportError:
                self.logger.warning("Playwright not available, trying alternative methods")
                
            # Fallback: Try using selenium
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument(f"--window-size={width},{height}")
                
                driver = webdriver.Chrome(options=chrome_options)
                driver.get(f'file://{os.path.abspath(html_path)}')
                driver.save_screenshot(output_path)
                driver.quit()
                return True
            except Exception as e:
                self.logger.warning(f"Selenium conversion failed: {e}")
                
            # Final fallback: Try using wkhtmltoimage
            try:
                subprocess.run([
                    'wkhtmltoimage', 
                    '--width', str(width),
                    '--height', str(height),
                    '--format', 'png',
                    html_path, 
                    output_path
                ], check=True, capture_output=True)
                return True
            except Exception as e:
                self.logger.warning(f"wkhtmltoimage conversion failed: {e}")
                
        except Exception as e:
            self.logger.error(f"HTML to image conversion failed: {e}")
            return False

    def generate_pdf_report(self, pipeline, display_data, study, best_model, model_name, model_params,
                           select_results, report, y_test, y_pred, feature_names, feature_selection_info,
                           shap_exp_filepath, exp_filepath, lime_exp_filepath):
        """Generate a comprehensive PDF report of the AutoML pipeline results"""

        # Create timestamped report filename
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = os.path.join(self.results_folder, f'pdf_report_{current_datetime}.pdf')
        
        # First, convert HTML files to images
        self._convert_html_to_images()
        
        # Generate HTML content for PDF conversion
        html_content = self._generate_html_content(
            pipeline, display_data, study, best_model, model_name, model_params,
            select_results, report, y_test, y_pred, feature_names, feature_selection_info,
            current_datetime
        )
        
        # Convert HTML to PDF
        try:
            self._html_to_pdf(html_content, pdf_filename, display_data)
            return pdf_filename
        except Exception as e:
            self.logger.error(f"PDF generation failed: {e}")
            # Fallback: generate simple PDF with ReportLab
            try:
                self._generate_simple_pdf_report(html_content, pdf_filename, display_data)
                self.logger.info(f"PDF report generated with ReportLab: {pdf_filename}")
                return pdf_filename
            except Exception as e2:
                self.logger.error(f"ReportLab PDF generation failed: {e2}")
                # Final fallback: save as HTML
                html_filename = os.path.join(self.results_folder, f'pdf_report_{current_datetime}.html')
                with open(html_filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.logger.info(f"HTML report saved as fallback: {html_filename}")
                return html_filename

    def _convert_html_to_images(self):
        """Convert all HTML files to PNG images for PDF inclusion"""
        html_files = [
            "optuna_parallel_coordinate.html",
            "optuna_param_importance.html", 
            "optuna_slice_plot.html",
            "optuna_optimization_history.html"
        ]
        
        # Add LIME and SHAP HTML files
        for file in os.listdir(self.results_folder):
            if file.startswith('lime_interpretability_') and file.endswith('.html'):
                html_files.append(file)
            elif file.startswith('shap_force_plot_') and file.endswith('.html'):
                html_files.append(file)
        
        for html_file in html_files:
            html_path = os.path.join(self.results_folder, html_file)
            if os.path.exists(html_path):
                png_file = html_file.replace('.html', '.png')
                png_path = os.path.join(self.results_folder, png_file)
                
                if not os.path.exists(png_path):  # Only convert if PNG doesn't exist
                    self.logger.info(f"Converting {html_file} to PNG...")
                    self.html_to_image(html_path, png_path)

    def _generate_html_content(self, pipeline, display_data, study, best_model, model_name, model_params,
                              select_results, report, y_test, y_pred, feature_names, feature_selection_info,
                              current_datetime):
        """Generate HTML content optimized for PDF conversion with all artifacts"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MH-AutoML - Professional Malware Detection Report</title>
            <style>
                @page {{
                    size: A4;
                    margin: 2cm;
                }}
                body {{
                    font-family: 'Arial', sans-serif;
                    line-height: 1.4;
                    color: #333;
                    font-size: 11pt;
                    background-color: #f8f9fa;
                }}
                .container {{
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                    page-break-after: avoid;
                }}
                h1 {{
                    font-size: 28pt;
                    text-align: center;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    margin: -20px -20px 30px -20px;
                    border-radius: 5px;
                }}
                h2 {{
                    font-size: 20pt;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    margin-top: 30px;
                    page-break-before: auto;
                    color: #2c3e50;
                }}
                h3 {{
                    font-size: 16pt;
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                    margin-top: 25px;
                }}
                h4 {{
                    font-size: 14pt;
                    color: #7f8c8d;
                    margin-top: 20px;
                }}
                .section {{
                    margin: 25px 0;
                    padding: 20px;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                    background-color: #ffffff;
                }}
                .artifact-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .artifact-item {{
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    text-align: center;
                }}
                .artifact-item img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .artifact-title {{
                    font-weight: bold;
                    color: #495057;
                    margin: 10px 0 5px 0;
                    font-size: 12pt;
                }}
                .artifact-description {{
                    font-size: 10pt;
                    color: #6c757d;
                    line-height: 1.3;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                    font-size: 10pt;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .metric-highlight {{
                    background-color: #e8f5e8;
                    padding: 10px;
                    border-radius: 5px;
                    border-left: 4px solid #28a745;
                    margin: 10px 0;
                }}
                .warning {{
                    background-color: #fff3cd;
                    padding: 10px;
                    border-radius: 5px;
                    border-left: 4px solid #ffc107;
                    margin: 10px 0;
                }}
                .info {{
                    background-color: #d1ecf1;
                    padding: 10px;
                    border-radius: 5px;
                    border-left: 4px solid #17a2b8;
                    margin: 10px 0;
                }}
                .executive-summary {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .executive-summary h3 {{
                    color: white;
                    border-left: none;
                    padding-left: 0;
                }}
                .performance-metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    border: 1px solid #dee2e6;
                }}
                .metric-value {{
                    font-size: 24pt;
                    font-weight: bold;
                    color: #3498db;
                }}
                .metric-label {{
                    font-size: 10pt;
                    color: #6c757d;
                    margin-top: 5px;
                }}
                .page-break {{
                    page-break-before: always;
                }}
                .footer {{
                    text-align: center;
                    font-size: 9pt;
                    color: #6c757d;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #dee2e6;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔬 MH-AutoML</h1>
                <h1 style="font-size: 18pt; background: none; color: #2c3e50; margin-top: -20px;">Professional Malware Detection Report</h1>
                
                <div class="executive-summary">
                    <h3>📊 Executive Summary</h3>
                    <p><strong>Generation Date:</strong> {current_datetime}</p>
                    <p><strong>Selected Model:</strong> {model_name}</p>
                    <p><strong>Overall Accuracy:</strong> {report.get('accuracy', 0):.3f}</p>
                    <p><strong>F1-Score:</strong> {report.get('macro avg', {}).get('f1-score', 0):.3f}</p>
                </div>

                <div class="info">
                    <h4>🔬 Report Overview</h4>
                    <p>This report presents the results of an automated machine learning analysis conducted using the MH-AutoML system for Android malware detection. The analysis includes comprehensive data preprocessing, feature engineering, model optimization, and interpretability insights to ensure reliable and explainable malware detection.</p>
                </div>

                <div class="page-break"></div>
                
                <h2>🔄 Pipeline Overview</h2>
                <div class="section">
                    <div class="info">
                        <h4>📋 Analysis Steps</h4>
                        <p>The MH-AutoML pipeline consists of the following automated steps:</p>
                        <ul>
                            <li><strong>0. Data Info:</strong> Dataset validation and information analysis</li>
                            <li><strong>1. Preprocessing:</strong> Data cleaning, missing value treatment, and encoding</li>
                            <li><strong>2. Feature Engineering:</strong> Dimensionality reduction and feature selection using PCA, LASSO, and ANOVA</li>
                            <li><strong>3. Model Optimization:</strong> Hyperparameter tuning with Optuna for best performance</li>
                            <li><strong>4. Interpretability:</strong> SHAP and LIME explanations for model decisions</li>
                            <li><strong>5. Evaluation:</strong> Comprehensive model assessment and reporting</li>
                        </ul>
                    </div>
                </div>

                <h2>📈 1. Performance Metrics</h2>
                <div class="section">
                    <div class="performance-metrics">
                        <div class="metric-card">
                            <div class="metric-value">{report.get('accuracy', 0):.3f}</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{report.get('macro avg', {}).get('precision', 0):.3f}</div>
                            <div class="metric-label">Precision</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{report.get('macro avg', {}).get('recall', 0):.3f}</div>
                            <div class="metric-label">Recall</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{report.get('macro avg', {}).get('f1-score', 0):.3f}</div>
                            <div class="metric-label">F1-Score</div>
                        </div>
                    </div>
                </div>

                <h2>📊 2. Evaluation Charts</h2>
                <div class="section">
                    <div class="artifact-grid">
                        {self._get_evaluation_artifacts()}
                    </div>
                </div>

                <div class="page-break"></div>
                
                <h2>🔧 3. Hyperparameter Optimization</h2>
                <div class="section">
                    <h3>Best Model Parameters</h3>
                    <div class="info">
                        <strong>Model:</strong> {best_model.__class__.__name__}<br>
                        <strong>Parameters:</strong> {str(model_params)[:200]}...
                    </div>
                    
                    <h3>Optimization Visualizations</h3>
                    <div class="artifact-grid">
                        {self._get_optimization_artifacts()}
                    </div>
                </div>

                <div class="page-break"></div>
                
                <h2>🎯 4. Feature Engineering</h2>
                <div class="section">
                    <div class="artifact-grid">
                        {self._get_feature_engineering_artifacts()}
                    </div>
                </div>

                <h2>🧠 5. Model Interpretability</h2>
                <div class="section">
                    <div class="artifact-grid">
                        {self._get_interpretability_artifacts()}
                    </div>
                </div>

                <div class="page-break"></div>
                
                <h2>📋 6. Data Preprocessing</h2>
                <div class="section">
                    <div class="artifact-grid">
                        {self._get_preprocessing_artifacts()}
                    </div>
                </div>

                <h2>📊 7. Dataset Information</h2>
                <div class="section">
                    {self._get_dataset_info(display_data)}
                </div>

                <div class="page-break"></div>
                
                <h2>📈 8. Detailed Classification Report</h2>
                <div class="section">
                    {self._get_detailed_classification_report(report)}
                </div>

                <h2>🔍 9. Feature Analysis</h2>
                <div class="section">
                    {self._get_feature_analysis(feature_names, feature_selection_info)}
                </div>

                <div class="footer">
                    <p><strong>MH-AutoML</strong> - Automated Machine Learning System for Malware Detection</p>
                    <p>Report automatically generated on {current_datetime}</p>
                    <p>© 2025 MH-AutoML - All rights reserved</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content

    def _get_evaluation_artifacts(self):
        """Get evaluation artifacts HTML"""
        artifacts = []
        evaluation_files = [
            ('confusion_matrix.png', 'Confusion Matrix', 'Confusion matrix shows the model\'s classification performance. True positives (correctly identified malware) and true negatives (correctly identified benign apps) are crucial for security. False positives (benign apps flagged as malware) and false negatives (undetected malware) represent different types of security risks.'),
            ('roc_curve.png', 'ROC/AUC Curve', 'ROC curve shows the model\'s ability to distinguish between malware and benign applications across different classification thresholds. Higher AUC indicates better discrimination capability.'),
            ('precision_recall_curve.png', 'Precision-Recall Curve', 'Precision-recall curve is especially important for malware detection where the positive class (malware) is often minority. High precision reduces false positives, while high recall minimizes undetected malware.'),
            ('probability_distribution.png', 'Probability Distribution', 'Probability distribution shows how the model assigns malware probabilities to applications. Good separation between benign and malicious probability distributions indicates confident predictions.'),
            ('metrics_by_class.png', 'Metrics by Class', 'Metrics by class compares specific performance for benign vs. malicious applications. This allows adjusting the model to prioritize malware detection or avoid false positives based on security requirements.')
        ]
        
        for filename, title, description in evaluation_files:
            filepath = os.path.join(self.results_folder, filename)
            if os.path.exists(filepath):
                artifacts.append(f"""
                    <div class="artifact-item">
                        <img src="{filepath}" alt="{title}">
                        <div class="artifact-title">{title}</div>
                        <div class="artifact-description">{description}</div>
                    </div>
                """)
        
        return '\n'.join(artifacts) if artifacts else '<p>No evaluation charts found.</p>'

    def _get_optimization_artifacts(self):
        """Get optimization artifacts HTML"""
        artifacts = []
        optimization_files = [
            ('optuna_optimization_history.png', 'Optimization History', 'Optimization history shows how model performance improved during hyperparameter tuning. Each trial represents a different configuration tested for malware detection accuracy.'),
            ('optuna_param_importance.png', 'Parameter Importance', 'Parameter importance shows which hyperparameters most influence malware detection performance. This helps focus optimization efforts on the most critical parameters.'),
            ('optuna_parallel_coordinate.png', 'Parallel Coordinates', 'Parallel coordinates plot shows the relationship between different hyperparameter combinations and their impact on malware detection performance.'),
            ('optuna_slice_plot.png', 'Slice Plot', 'Slice plot shows the sensitivity of model performance to individual hyperparameter values. This helps identify optimal ranges for each parameter in malware detection.')
        ]
        
        for filename, title, description in optimization_files:
            filepath = os.path.join(self.results_folder, filename)
            if os.path.exists(filepath):
                artifacts.append(f"""
                    <div class="artifact-item">
                        <img src="{filepath}" alt="{title}">
                        <div class="artifact-title">{title}</div>
                        <div class="artifact-description">{description}</div>
                    </div>
                """)
        
        return '\n'.join(artifacts) if artifacts else '<p>No optimization charts found.</p>'

    def _get_feature_engineering_artifacts(self):
        """Get feature engineering artifacts HTML"""
        artifacts = []
        feature_files = [
            ('lasso_feature_importance.png', 'LASSO Importance', 'LASSO feature importance shows which application characteristics are most relevant for malware detection. Features with higher importance are more critical for distinguishing between benign and malicious applications.'),
            ('train_test_distribution.png', 'Train/Test Distribution', 'Train/test distribution shows the balance of malware and benign samples between training and test sets. This ensures representative data distribution for reliable model evaluation.')
        ]
        
        for filename, title, description in feature_files:
            filepath = os.path.join(self.results_folder, filename)
            if os.path.exists(filepath):
                artifacts.append(f"""
                    <div class="artifact-item">
                        <img src="{filepath}" alt="{title}">
                        <div class="artifact-title">{title}</div>
                        <div class="artifact-description">{description}</div>
                    </div>
                """)
        
        return '\n'.join(artifacts) if artifacts else '<p>No feature engineering charts found.</p>'

    def _get_interpretability_artifacts(self):
        """Get interpretability artifacts HTML"""
        artifacts = []
        
        # Find SHAP and LIME files
        for file in os.listdir(self.results_folder):
            if file.startswith('shap_force_plot_') and file.endswith('.png'):
                filepath = os.path.join(self.results_folder, file)
                artifacts.append(f"""
                    <div class="artifact-item">
                        <img src="{filepath}" alt="SHAP Force Plot">
                        <div class="artifact-title">SHAP Force Plot</div>
                        <div class="artifact-description">SHAP force plot shows detailed contribution of each feature to a specific malware prediction. Red bars push toward malware classification, blue bars push toward benign classification.</div>
                    </div>
                """)
            elif file.startswith('shap_summary_plot_') and file.endswith('.png'):
                filepath = os.path.join(self.results_folder, file)
                artifacts.append(f"""
                    <div class="artifact-item">
                        <img src="{filepath}" alt="SHAP Summary">
                        <div class="artifact-title">SHAP Summary Plot</div>
                        <div class="artifact-description">SHAP summary plot shows global feature importance for malware detection. Features with higher absolute SHAP values are more critical for distinguishing between benign and malicious applications.</div>
                    </div>
                """)
            elif file.startswith('lime_interpretability_') and file.endswith('.png'):
                filepath = os.path.join(self.results_folder, file)
                artifacts.append(f"""
                    <div class="artifact-item">
                        <img src="{filepath}" alt="LIME Explanation">
                        <div class="artifact-title">LIME Explanation</div>
                        <div class="artifact-description">LIME explanation provides local interpretability for a specific prediction. It shows which features contributed most to classifying this particular application as malware or benign.</div>
                    </div>
                """)
            elif file.startswith('lime_feature_importance_') and file.endswith('.png'):
                filepath = os.path.join(self.results_folder, file)
                artifacts.append(f"""
                    <div class="artifact-item">
                        <img src="{filepath}" alt="LIME Feature Importance">
                        <div class="artifact-title">LIME Feature Importance</div>
                        <div class="artifact-description">LIME feature importance shows the weight of each feature in the model's decision for a specific case. Positive values favor malware classification, negative values favor benign classification.</div>
                    </div>
                """)
            elif file.startswith('decision_tree_plot_') and file.endswith('.png'):
                filepath = os.path.join(self.results_folder, file)
                artifacts.append(f"""
                    <div class="artifact-item">
                        <img src="{filepath}" alt="Decision Tree">
                        <div class="artifact-title">Decision Tree</div>
                        <div class="artifact-description">Decision tree visualization shows the model's decision rules. Each node represents a feature split that helps distinguish between malware and benign applications.</div>
                    </div>
                """)
        
        return '\n'.join(artifacts) if artifacts else '<p>No interpretability charts found.</p>'

    def _get_preprocessing_artifacts(self):
        """Get preprocessing artifacts HTML"""
        artifacts = []
        preprocessing_files = [
            ('missing_values_heatmap.png', 'Missing Values (Before)', 'This heatmap shows the distribution of missing values in the original dataset. It helps identify which features may have incomplete data, which is crucial for reliable malware detection.'),
            ('clean_missing_values_heatmap.png', 'Missing Values (After)', 'This heatmap shows the distribution of missing values after data cleaning. It demonstrates the effectiveness of preprocessing in preparing the dataset for modeling.')
        ]
        
        for filename, title, description in preprocessing_files:
            filepath = os.path.join(self.results_folder, filename)
            if os.path.exists(filepath):
                artifacts.append(f"""
                    <div class="artifact-item">
                        <img src="{filepath}" alt="{title}">
                        <div class="artifact-title">{title}</div>
                        <div class="artifact-description">{description}</div>
                    </div>
                """)
        
        return '\n'.join(artifacts) if artifacts else '<p>No preprocessing charts found.</p>'

    def _get_dataset_info(self, display_data):
        """Get dataset information HTML"""
        # Se for DataInfo ou outro objeto, tenta converter para dict
        if hasattr(display_data, '__dict__'):
            display_data = vars(display_data)
        if not isinstance(display_data, dict):
            return '<p>Dataset information not available.</p>'
        
        return f"""
            <div class="info">
                <h4>📊 General Information</h4>
                <p><strong>Total Samples:</strong> {display_data.get('total_samples', 'N/A')}</p>
                <p><strong>Total Features:</strong> {display_data.get('total_features', 'N/A')}</p>
                <p><strong>Numeric Features:</strong> {display_data.get('numeric_features', 'N/A')}</p>
                <p><strong>Categorical Features:</strong> {display_data.get('categorical_features', 'N/A')}</p>
            </div>
            
            <div class="metric-highlight">
                <h4>🎯 Class Distribution</h4>
                <p><strong>Class 0 (Benign):</strong> {display_data.get('class_0_count', 'N/A')} ({display_data.get('class_0_percentage', 'N/A')}%)</p>
                <p><strong>Class 1 (Malware):</strong> {display_data.get('class_1_count', 'N/A')} ({display_data.get('class_1_percentage', 'N/A')}%)</p>
            </div>
        """

    def _get_detailed_classification_report(self, report):
        """Get detailed classification report HTML"""
        if not report:
            return '<p>Classification report not available.</p>'
        
        html = '<h4>📋 Classification Report</h4>'
        html += '<table>'
        html += '<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>'
        
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                html += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{metrics.get('precision', 0):.3f}</td>
                        <td>{metrics.get('recall', 0):.3f}</td>
                        <td>{metrics.get('f1-score', 0):.3f}</td>
                        <td>{metrics.get('support', 0)}</td>
                    </tr>
                """
        
        html += '</table>'
        return html

    def _get_feature_analysis(self, feature_names, feature_selection_info):
        """Get feature analysis HTML"""
        html = '<h4>🔍 Feature Analysis</h4>'
        
        if feature_names:
            html += f'<p><strong>Selected Features:</strong> {len(feature_names)}</p>'
            html += '<div class="info">'
            html += '<strong>Top 10 Features:</strong><br>'
            for i, feature in enumerate(feature_names[:10]):
                html += f'{i+1}. {feature}<br>'
            html += '</div>'
        
        if feature_selection_info:
            html += f'<p><strong>Selection Method:</strong> {feature_selection_info.get("method", "N/A")}</p>'
            html += f'<p><strong>Original Features:</strong> {feature_selection_info.get("original_features", "N/A")}</p>'
            html += f'<p><strong>Selected Features:</strong> {feature_selection_info.get("selected_features", "N/A")}</p>'
        
        return html

    def _html_to_pdf(self, html_content, pdf_path, display_data=None):
        """Convert HTML content to PDF using only ReportLab fallback (Windows compatible)"""
        try:
            self._generate_simple_pdf_report(html_content, pdf_path, display_data)
            return
        except Exception as e:
            self.logger.warning(f"ReportLab fallback failed: {e}")
            # If all methods fail, raise exception
            raise Exception("ReportLab PDF generation failed")

    def _generate_simple_pdf_report(self, html_content, pdf_path, display_data=None):
        """Generate a comprehensive PDF report using ReportLab as fallback"""
        try:
            # Defina a função utilitária logo no início
            def ensure_table_has_data(table):
                if len(table) == 1:
                    table.append(["No data available"] * len(table[0]))
                return table

            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from reportlab.platypus import Image
            
            # Create PDF document
            doc = SimpleDocTemplate(pdf_path, pagesize=A4, 
                                  leftMargin=1*inch, rightMargin=1*inch,
                                  topMargin=1*inch, bottomMargin=1*inch)
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1,  # Center
                textColor=colors.darkblue,
                fontName='Helvetica-Bold'
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                spaceBefore=20,
                textColor=colors.darkblue,
                fontName='Helvetica-Bold'
            )
            
            subheading_style = ParagraphStyle(
                'CustomSubHeading',
                parent=styles['Heading3'],
                fontSize=14,
                spaceAfter=8,
                spaceBefore=15,
                textColor=colors.darkgreen,
                fontName='Helvetica-Bold'
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=6,
                leading=14
            )
            
            # Title page
            story.append(Paragraph("AutoML Analysis Report", title_style))
            story.append(Spacer(1, 20))
            
            # Add timestamp
            timestamp_style = ParagraphStyle(
                'Timestamp',
                parent=styles['Normal'],
                fontSize=12,
                alignment=1,
                textColor=colors.grey
            )
            current_time = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
            story.append(Paragraph(f"Generated on: {current_time}", timestamp_style))
            story.append(Paragraph("MH-AutoML System", timestamp_style))
            story.append(PageBreak())
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", heading_style))
            summary_text = """
            This report presents the results of an automated machine learning analysis conducted using the MH-AutoML system. 
            The analysis includes comprehensive data preprocessing, feature engineering, model optimization, and interpretability insights.
            """
            story.append(Paragraph(summary_text, normal_style))
            story.append(Spacer(1, 15))
            
            # Pipeline Configuration
            story.append(Paragraph("Pipeline Configuration", heading_style))
            story.append(Paragraph("The AutoML pipeline consists of the following steps:", normal_style))
            story.append(Spacer(1, 10))
            
            # Add pipeline steps if available (following MH-AutoML real structure)
            pipeline_steps = [
                "0. Data Info - Dataset validation and information",
                "1. Preprocessing - Data cleaning and transformation",
                "2. Feature Engineering - Dimensionality reduction and selection", 
                "3. Model Optimization - Hyperparameter tuning with Optuna",
                "4. Interpretability - SHAP and LIME explanations",
                "5. Evaluation - Model assessment and reporting"
            ]
            
            for step in pipeline_steps:
                story.append(Paragraph(f"• {step}", normal_style))
            
            story.append(Spacer(1, 15))
            
            # Data Info Section (Step 0)
            story.append(Paragraph("0. Data Info", heading_style))
            story.append(Paragraph("Dataset validation and information analysis:", normal_style))
            story.append(Spacer(1, 10))
            
            # System Information
            story.append(Paragraph("System Information:", subheading_style))
            import platform
            import psutil
            
            system_info = [
                ["Operating System", platform.system() + " " + platform.release()],
                ["Python Version", platform.python_version()],
                ["Architecture", platform.architecture()[0]],
                ["Total RAM", f"{psutil.virtual_memory().total / (1024**3):.2f} GB"],
                ["Available RAM", f"{psutil.virtual_memory().available / (1024**3):.2f} GB"],
                ["CPU Cores", str(psutil.cpu_count())],
                ["CPU Usage", f"{psutil.cpu_percent()}%"]
            ]
            
            # Create system info table
            system_table = Table(system_info, colWidths=[2*inch, 3*inch])
            system_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(system_table)
            story.append(Spacer(1, 15))
            
            # Dataset Information (usando DataInfo)
            story.append(Paragraph("Dataset Information:", subheading_style))
            if display_data is not None and hasattr(display_data, 'info_table_result') and display_data.info_table_result is not None:
                info_df = display_data.info_table_result
                dataset_info = [["Info", "Value"]]
                for col in info_df.columns:
                    dataset_info.append([col, str(info_df[col].iloc[0])])
            else:
                dataset_info = [["Info", "Value"], ["No data available", ""]]
            dataset_table = Table(dataset_info, colWidths=[2*inch, 3*inch])
            dataset_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(dataset_table)
            story.append(Spacer(1, 15))

            # Data Types Analysis (usando DataInfo)
            story.append(Paragraph("Data Types Analysis:", subheading_style))
            if display_data is not None and hasattr(display_data, 'data_types_result') and display_data.data_types_result is not None:
                dtypes_df = display_data.data_types_result
                data_types_info = [list(map(str, dtypes_df.columns.tolist()))] + [list(map(str, row)) for row in dtypes_df.values.tolist()]
            else:
                data_types_info = [["Data Type", "Count"], ["No data available", ""]]
            dtypes_table = Table(data_types_info, colWidths=[2*inch, 3*inch])
            dtypes_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(dtypes_table)
            story.append(Spacer(1, 15))

            # Class Balance Analysis (usando DataInfo)
            story.append(Paragraph("Class Balance Analysis:", subheading_style))
            if display_data is not None and hasattr(display_data, 'balance_info_result') and display_data.balance_info_result is not None:
                balance_df = display_data.balance_info_result
                class_balance_info = [list(map(str, balance_df.columns.tolist()))] + [list(map(str, row)) for row in balance_df.values.tolist()]
            else:
                class_balance_info = [["Label", "Percentage"], ["No data available", ""]]
            balance_table = Table(class_balance_info, colWidths=[2*inch, 3*inch])
            balance_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(balance_table)
            story.append(Spacer(1, 15))

            # Android Features Analysis (usando DataInfo)
            story.append(Paragraph("Android Features Analysis:", subheading_style))
            if display_data is not None and hasattr(display_data, 'features_info_result') and display_data.features_info_result is not None:
                features_df = display_data.features_info_result
                android_features_info = [list(map(str, features_df.columns.tolist()))] + [list(map(str, row)) for row in features_df.values.tolist()]
            else:
                android_features_info = [["Feature Type", "Count"], ["No data available", ""]]
            android_table = Table(android_features_info, colWidths=[2*inch, 3*inch])
            android_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(android_table)
            story.append(Spacer(1, 15))
            
            # 1. Data Preprocessing
            story.append(PageBreak())
            story.append(Paragraph("1. Data Preprocessing - Cleaning and preparation", heading_style))
            story.append(Paragraph(
                "In this step, the system performs data cleaning, removes missing values, duplicates, and handles outliers, as well as applies necessary encodings. Proper preprocessing ensures the quality and reliability of the data for malware detection.",
                normal_style))
            story.append(Spacer(1, 10))
            
            # Add preprocessing visualizations
            preprocessing_images = []
            for file in os.listdir(self.results_folder):
                if file.endswith(('.png', '.jpg', '.jpeg')) and ("missing_values_heatmap" in file or "clean_missing_values_heatmap" in file):
                    image_path = os.path.join(self.results_folder, file)
                    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                        preprocessing_images.append((file, image_path))
            
            if preprocessing_images:
                story.append(Paragraph("Preprocessing Visualizations:", subheading_style))
                for filename, image_path in preprocessing_images:
                    try:
                        img = Image(image_path, width=5*inch, height=3*inch)
                        story.append(img)
                        # Explicação específica para o gráfico missing_values_heatmap.png
                        if filename == "missing_values_heatmap.png":
                            story.append(Paragraph(
                                "Figure: missing_values_heatmap.png - This heatmap shows the distribution of missing values in the original dataset. It helps identify which features may have incomplete data, which is crucial for reliable malware detection.",
                                normal_style))
                        elif filename == "clean_missing_values_heatmap.png":
                            story.append(Paragraph(
                                "Figure: clean_missing_values_heatmap.png - This heatmap shows the distribution of missing values after data cleaning. It demonstrates the effectiveness of preprocessing in preparing the dataset for modeling.",
                                normal_style))
                        else:
                            story.append(Paragraph(f"Figure: {filename}", normal_style))
                        story.append(Spacer(1, 10))
                    except Exception as e:
                        self.logger.warning(f"Could not add preprocessing image {filename}: {e}")

            # 2. Feature Engineering
            story.append(PageBreak())
            story.append(Paragraph("2. Feature Engineering - Selection and transformation", heading_style))
            story.append(Paragraph(
                "Feature engineering includes selection and transformation of variables, using techniques such as PCA, LASSO, and ANOVA for dimensionality reduction and selection of the best variables for malware detection.",
                normal_style))
            story.append(Spacer(1, 10))
            
            # Add feature engineering visualizations
            feature_images = []
            for file in os.listdir(self.results_folder):
                if file.endswith(('.png', '.jpg', '.jpeg')) and ("pca_" in file or "lasso_" in file or "anova_" in file or "train_test_distribution" in file):
                    image_path = os.path.join(self.results_folder, file)
                    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                        feature_images.append((file, image_path))
            
            if feature_images:
                story.append(Paragraph("Feature Engineering Visualizations:", subheading_style))
                for filename, image_path in feature_images:
                    try:
                        img = Image(image_path, width=5*inch, height=3*inch)
                        story.append(img)
                        # Explicações específicas para cada tipo de gráfico de feature engineering
                        if "lasso_feature_importance" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - LASSO feature importance shows which application characteristics are most relevant for malware detection. Features with higher importance are more critical for distinguishing between benign and malicious applications.",
                                normal_style))
                        elif "train_test_distribution" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - Train/test distribution shows the balance of malware and benign samples between training and test sets. This ensures representative data distribution for reliable model evaluation.",
                                normal_style))
                        elif "pca_" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - PCA visualization shows dimensionality reduction results. This helps identify the most important components for malware detection while reducing computational complexity.",
                                normal_style))
                        elif "anova_" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - ANOVA analysis shows statistical significance of features in distinguishing between malware and benign applications. Higher F-scores indicate more discriminative features.",
                                normal_style))
                        else:
                            story.append(Paragraph(f"Figure: {filename}", normal_style))
                        story.append(Spacer(1, 10))
                    except Exception as e:
                        self.logger.warning(f"Could not add feature engineering image {filename}: {e}")

            # 3. Model Optimization
            story.append(PageBreak())
            story.append(Paragraph("3. Model Optimization - Hyperparameter tuning", heading_style))
            story.append(Paragraph(
                "Model optimization through hyperparameter tuning using Optuna, seeking the best possible performance for malware detection accuracy and reliability.",
                normal_style))
            story.append(Spacer(1, 10))
            
            # Add model optimization visualizations
            optimization_images = []
            for file in os.listdir(self.results_folder):
                if file.endswith(('.png', '.jpg', '.jpeg')) and "optuna_" in file:
                    image_path = os.path.join(self.results_folder, file)
                    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                        optimization_images.append((file, image_path))
            
            if optimization_images:
                story.append(Paragraph("Model Optimization Visualizations:", subheading_style))
                for filename, image_path in optimization_images:
                    try:
                        img = Image(image_path, width=5*inch, height=3*inch)
                        story.append(img)
                        # Explicações específicas para cada tipo de gráfico de otimização
                        if "optuna_optimization_history" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - Optimization history shows how model performance improved during hyperparameter tuning. Each trial represents a different configuration tested for malware detection accuracy.",
                                normal_style))
                        elif "optuna_param_importance" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - Parameter importance shows which hyperparameters most influence malware detection performance. This helps focus optimization efforts on the most critical parameters.",
                                normal_style))
                        elif "optuna_parallel_coordinate" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - Parallel coordinates plot shows the relationship between different hyperparameter combinations and their impact on malware detection performance.",
                                normal_style))
                        elif "optuna_slice_plot" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - Slice plot shows the sensitivity of model performance to individual hyperparameter values. This helps identify optimal ranges for each parameter in malware detection.",
                                normal_style))
                        else:
                            story.append(Paragraph(f"Figure: {filename}", normal_style))
                        story.append(Spacer(1, 10))
                    except Exception as e:
                        self.logger.warning(f"Could not add optimization image {filename}: {e}")

            # 4. Model Evaluation
            story.append(PageBreak())
            story.append(Paragraph("4. Model Evaluation - Performance assessment", heading_style))
            story.append(Paragraph(
                "Model evaluation with metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix to assess malware detection performance.",
                normal_style))
            story.append(Spacer(1, 10))
            
            # Add model evaluation visualizations (confusion matrix, performance metrics)
            evaluation_images = []
            for file in os.listdir(self.results_folder):
                if file.endswith(('.png', '.jpg', '.jpeg')) and ("confusion_matrix" in file or "performance_metrics" in file or "roc_curve" in file or "precision_recall_curve" in file or "probability_distribution" in file or "metrics_by_class" in file):
                    image_path = os.path.join(self.results_folder, file)
                    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                        evaluation_images.append((file, image_path))
            
            if evaluation_images:
                story.append(Paragraph("Model Evaluation Visualizations:", subheading_style))
                for filename, image_path in evaluation_images:
                    try:
                        img = Image(image_path, width=5*inch, height=3*inch)
                        story.append(img)
                        # Explicações específicas para cada tipo de gráfico de avaliação
                        if "confusion_matrix" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - Confusion matrix shows the model's classification performance. True positives (correctly identified malware) and true negatives (correctly identified benign apps) are crucial for security. False positives (benign apps flagged as malware) and false negatives (undetected malware) represent different types of security risks.",
                                normal_style))
                        elif "roc_curve" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - ROC curve shows the model's ability to distinguish between malware and benign applications across different classification thresholds. Higher AUC indicates better discrimination capability.",
                                normal_style))
                        elif "precision_recall_curve" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - Precision-recall curve is especially important for malware detection where the positive class (malware) is often minority. High precision reduces false positives, while high recall minimizes undetected malware.",
                                normal_style))
                        elif "probability_distribution" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - Probability distribution shows how the model assigns malware probabilities to applications. Good separation between benign and malicious probability distributions indicates confident predictions.",
                                normal_style))
                        elif "metrics_by_class" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - Metrics by class compares specific performance for benign vs. malicious applications. This allows adjusting the model to prioritize malware detection or avoid false positives based on security requirements.",
                                normal_style))
                        else:
                            story.append(Paragraph(f"Figure: {filename}", normal_style))
                        story.append(Spacer(1, 10))
                    except Exception as e:
                        self.logger.warning(f"Could not add evaluation image {filename}: {e}")
            
            # 5. Interpretability
            story.append(PageBreak())
            story.append(Paragraph("5. Interpretability - SHAP and LIME analysis", heading_style))
            story.append(Paragraph(
                "Model interpretability analysis using SHAP and LIME to explain model decisions and identify the most important variables for malware detection.",
                normal_style))
            story.append(Spacer(1, 10))
            
            # Add interpretability visualizations
            interpretability_images = []
            for file in os.listdir(self.results_folder):
                if file.endswith(('.png', '.jpg', '.jpeg')) and ("shap_" in file or "lime_" in file or "decision_tree_plot" in file):
                    image_path = os.path.join(self.results_folder, file)
                    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                        interpretability_images.append((file, image_path))
            
            if interpretability_images:
                story.append(Paragraph("Interpretability Visualizations:", subheading_style))
                for filename, image_path in interpretability_images:
                    try:
                        img = Image(image_path, width=5*inch, height=3*inch)
                        story.append(img)
                        # Explicações específicas para cada tipo de gráfico de interpretabilidade
                        if "shap_force_plot" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - SHAP force plot shows detailed contribution of each feature to a specific malware prediction. Red bars push toward malware classification, blue bars push toward benign classification.",
                                normal_style))
                        elif "shap_summary_plot" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - SHAP summary plot shows global feature importance for malware detection. Features with higher absolute SHAP values are more critical for distinguishing between benign and malicious applications.",
                                normal_style))
                        elif "lime_interpretability" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - LIME explanation provides local interpretability for a specific prediction. It shows which features contributed most to classifying this particular application as malware or benign.",
                                normal_style))
                        elif "lime_feature_importance" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - LIME feature importance shows the weight of each feature in the model's decision for a specific case. Positive values favor malware classification, negative values favor benign classification.",
                                normal_style))
                        elif "decision_tree_plot" in filename:
                            story.append(Paragraph(
                                f"Figure: {filename} - Decision tree visualization shows the model's decision rules. Each node represents a feature split that helps distinguish between malware and benign applications.",
                                normal_style))
                        else:
                            story.append(Paragraph(f"Figure: {filename}", normal_style))
                        story.append(Spacer(1, 10))
                    except Exception as e:
                        self.logger.warning(f"Could not add interpretability image {filename}: {e}")
            
            # Artifacts Summary
            story.append(PageBreak())
            story.append(Paragraph("Generated Artifacts", heading_style))
            story.append(Paragraph("Complete list of all generated files:", normal_style))
            story.append(Spacer(1, 10))
            
            # Categorize artifacts following MH-AutoML MLflow structure
            artifact_categories = {
                "00_Data_info": [],
                "01_preprocessing": [],
                "02_feature_engineering": [],
                "03_model_optimization": [],
                "04_evaluation_metrics": [],
                "05_interpretability": [],
                "Reports": []
            }
            
            for file in os.listdir(self.results_folder):
                # 01_preprocessing
                if "clean_missing_values_heatmap" in file or "missing_values_heatmap" in file:
                    artifact_categories["01_preprocessing"].append(file)
                
                # 03_model_optimization (deve vir antes de feature_engineering para evitar conflitos)
                elif ("optuna_" in file or "Hyperparameters_Results" in file or 
                      "Models_Ranking" in file or "optuna_trials" in file):
                    artifact_categories["03_model_optimization"].append(file)
                
                # 02_feature_engineering
                elif ("Features_Selected_" in file or "treino_" in file or 
                      "pca_" in file or "lasso_" in file or "anova_" in file or 
                      "train_test_distribution" in file or 
                      (file.endswith('.csv') and not any(x in file for x in ['optuna', 'Hyperparameters', 'Models_Ranking']))):
                    artifact_categories["02_feature_engineering"].append(file)
                
                # 04_evaluation_metrics
                elif (file.endswith('.pkl') or "performance_" in file or 
                      "best_model_" in file):
                    artifact_categories["04_evaluation_metrics"].append(file)
                
                # 05_interpretability
                elif ("shap_" in file or "lime_" in file):
                    artifact_categories["05_interpretability"].append(file)
                
                # Reports (HTML files that are not optuna)
                elif file.endswith('.html') and not file.startswith('optuna'):
                    artifact_categories["Reports"].append(file)
                
                # Default categorization for remaining files
                elif file.endswith(('.png', '.jpg', '.jpeg')):
                    if "optuna" in file:
                        artifact_categories["03_model_optimization"].append(file)
                    elif "shap" in file or "lime" in file:
                        artifact_categories["05_interpretability"].append(file)
                    elif "pca" in file or "lasso" in file or "anova" in file:
                        artifact_categories["02_feature_engineering"].append(file)
                    elif "missing" in file or "clean" in file:
                        artifact_categories["01_preprocessing"].append(file)
                    else:
                        artifact_categories["02_feature_engineering"].append(file)
            
            # Add artifacts to PDF organized by pipeline sections
            # Map categories to readable names
            category_names = {
                "00_Data_info": "Data Information",
                "01_preprocessing": "1. Data Preprocessing",
                "02_feature_engineering": "2. Feature Engineering", 
                "03_model_optimization": "3. Model Optimization",
                "04_evaluation_metrics": "4. Model Evaluation",
                "05_interpretability": "5. Interpretability",
                "Reports": "Reports"
            }
            
            for category, files in artifact_categories.items():
                if files:
                    readable_name = category_names.get(category, category)
                    story.append(Paragraph(f"{readable_name}:", subheading_style))
                    
                    # Sort files by type (images first, then data files, then others)
                    image_files = [f for f in sorted(files) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    data_files = [f for f in sorted(files) if f.endswith(('.csv', '.pkl'))]
                    other_files = [f for f in sorted(files) if not f.endswith(('.png', '.jpg', '.jpeg', '.csv', '.pkl'))]
                    
                    # Display images with special formatting
                    if image_files:
                        story.append(Paragraph("📊 Visualizations:", normal_style))
                        for file in image_files:
                            story.append(Paragraph(f"  • {file}", normal_style))
                        story.append(Spacer(1, 3))
                    
                    # Display data files
                    if data_files:
                        story.append(Paragraph("📁 Data Files:", normal_style))
                        for file in data_files:
                            story.append(Paragraph(f"  • {file}", normal_style))
                        story.append(Spacer(1, 3))
                    
                    # Display other files
                    if other_files:
                        story.append(Paragraph("📄 Other Files:", normal_style))
                        for file in other_files:
                            story.append(Paragraph(f"  • {file}", normal_style))
                        story.append(Spacer(1, 3))
                    
                    story.append(Spacer(1, 5))
            
            # Footer note
            story.append(Spacer(1, 20))
            note_style = ParagraphStyle(
                'Note',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.grey,
                alignment=1
            )
            story.append(Paragraph("Note: This is a simplified PDF version. For the complete interactive report with all visualizations and detailed formatting, please refer to the HTML version.", note_style))
            
            # Build PDF
            doc.build(story)
            return True
            
        except ImportError:
            self.logger.warning("ReportLab not available for fallback PDF generation")
            return False 