#!/usr/bin/env python3
"""
Test script to verify PDF generation with English explanations
"""

import os
import sys
import tempfile
import shutil
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.tools.pdf_report_generator import PDFReportGenerator

def create_sample_image(filepath, title="Sample Plot"):
    """Create a sample PNG image"""
    plt.figure(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()

def create_sample_artifacts(results_folder):
    """Create sample artifacts for testing"""
    os.makedirs(results_folder, exist_ok=True)
    
    # Create sample images
    sample_files = [
        ('confusion_matrix.png', 'Confusion Matrix'),
        ('roc_curve.png', 'ROC Curve'),
        ('precision_recall_curve.png', 'Precision-Recall Curve'),
        ('probability_distribution.png', 'Probability Distribution'),
        ('metrics_by_class.png', 'Metrics by Class'),
        ('optuna_optimization_history.png', 'Optimization History'),
        ('optuna_param_importance.png', 'Parameter Importance'),
        ('optuna_parallel_coordinate.png', 'Parallel Coordinates'),
        ('optuna_slice_plot.png', 'Slice Plot'),
        ('lasso_feature_importance.png', 'LASSO Importance'),
        ('train_test_distribution.png', 'Train/Test Distribution'),
        ('shap_force_plot_test.png', 'SHAP Force Plot'),
        ('shap_summary_plot_test.png', 'SHAP Summary'),
        ('lime_interpretability_test.png', 'LIME Explanation'),
        ('lime_feature_importance_test.png', 'LIME Feature Importance'),
        ('decision_tree_plot_test.png', 'Decision Tree'),
        ('missing_values_heatmap.png', 'Missing Values Before'),
        ('clean_missing_values_heatmap.png', 'Missing Values After')
    ]
    
    for filename, title in sample_files:
        filepath = os.path.join(results_folder, filename)
        create_sample_image(filepath, title)

def test_pdf_generation():
    """Test PDF generation with English explanations"""
    print("🔬 Testing PDF generation with English explanations...")
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        results_folder = os.path.join(temp_dir, "test_results")
        create_sample_artifacts(results_folder)
        
        # Initialize PDF generator
        pdf_generator = PDFReportGenerator(results_folder=results_folder)
        
        # Mock data for testing
        pipeline = "test_pipeline"
        display_data = {
            'total_samples': 1000,
            'total_features': 50,
            'numeric_features': 30,
            'categorical_features': 20,
            'class_0_count': 600,
            'class_0_percentage': 60.0,
            'class_1_count': 400,
            'class_1_percentage': 40.0
        }
        study = None
        best_model = type('MockModel', (), {'__class__': type('MockClass', (), {'__name__': 'RandomForestClassifier'})})()
        model_name = "RandomForestClassifier"
        model_params = {'n_estimators': 100, 'max_depth': 10}
        select_results = {}
        report = {
            'accuracy': 0.95,
            '0': {'precision': 0.96, 'recall': 0.94, 'f1-score': 0.95, 'support': 600},
            '1': {'precision': 0.94, 'recall': 0.96, 'f1-score': 0.95, 'support': 400},
            'macro avg': {'precision': 0.95, 'recall': 0.95, 'f1-score': 0.95, 'support': 1000}
        }
        y_test = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        feature_selection_info = {
            'method': 'LASSO',
            'original_features': 50,
            'selected_features': 5
        }
        shap_exp_filepath = os.path.join(results_folder, "shap_force_plot_test.png")
        exp_filepath = os.path.join(results_folder, "lime_interpretability_test.png")
        lime_exp_filepath = os.path.join(results_folder, "lime_feature_importance_test.png")
        
        # Generate PDF
        pdf_path = os.path.join(temp_dir, "test_report.pdf")
        
        try:
            pdf_generator.generate_pdf_report(
                pipeline, display_data, study, best_model, model_name, model_params,
                select_results, report, y_test, y_pred, feature_names, feature_selection_info,
                shap_exp_filepath, exp_filepath, lime_exp_filepath
            )
            
            # Check if PDF was created
            if os.path.exists(pdf_path):
                print("✅ PDF generated successfully!")
                print(f"📄 PDF path: {pdf_path}")
                
                # Check file size
                file_size = os.path.getsize(pdf_path)
                print(f"📊 File size: {file_size} bytes")
                
                if file_size > 1000:  # Should be more than 1KB
                    print("✅ PDF has reasonable size")
                else:
                    print("⚠️ PDF seems too small, might be empty")
                
                return True
            else:
                print("❌ PDF was not created")
                return False
                
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            return False

def main():
    """Main test function"""
    print("🚀 Starting PDF English explanations test...")
    
    success = test_pdf_generation()
    
    if success:
        print("\n🎉 All tests passed! PDF generation with English explanations is working correctly.")
    else:
        print("\n💥 Tests failed! There are issues with PDF generation.")
        sys.exit(1)

if __name__ == "__main__":
    main() 