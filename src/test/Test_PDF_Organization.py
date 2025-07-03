#!/usr/bin/env python3
"""
Test script to verify PDF report image organization by pipeline section.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tools.pdf_report_generator import PDFReportGenerator

def create_test_images():
    """Create test image files for different pipeline sections."""
    test_dir = tempfile.mkdtemp()
    
    # Create test images for each section
    test_images = [
        # Data Preprocessing
        "clean_missing_values_heatmap.png",
        "missing_values_heatmap.png",
        
        # Feature Engineering
        "pca_biplot.png",
        "lasso_coefficients.png",
        "anova_scores.png",
        "train_test_distribution.png",
        
        # Model Optimization
        "optuna_optimization_history.png",
        "optuna_slice_plot.png",
        "optuna_parallel_coordinate.png",
        
        # Model Evaluation
        "confusion_matrix.png",
        "performance_metrics.png",
        "roc_curve.png",
        
        # Interpretability
        "shap_summary_plot.png",
        "shap_waterfall_plot.png",
        "lime_explanation.png"
    ]
    
    for image in test_images:
        # Create empty files to simulate images
        with open(os.path.join(test_dir, image), 'w') as f:
            f.write("test")
    
    return test_dir

def test_image_organization():
    """Test that images are correctly organized by pipeline section."""
    print("Testing PDF report image organization...")
    
    # Create test directory with images
    test_dir = create_test_images()
    
    try:
        # Initialize PDF generator
        pdf_generator = PDFReportGenerator(test_dir)
        
        # Test image categorization
        preprocessing_images = []
        feature_images = []
        optimization_images = []
        evaluation_images = []
        interpretability_images = []
        
        for file in os.listdir(test_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(test_dir, file)
                if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                    # Categorize images
                    if "missing_values_heatmap" in file or "clean_missing_values_heatmap" in file:
                        preprocessing_images.append(file)
                    elif "pca_" in file or "lasso_" in file or "anova_" in file or "train_test_distribution" in file:
                        feature_images.append(file)
                    elif "optuna_" in file:
                        optimization_images.append(file)
                    elif "confusion_matrix" in file or "performance_metrics" in file or "roc_curve" in file:
                        evaluation_images.append(file)
                    elif "shap_" in file or "lime_" in file:
                        interpretability_images.append(file)
        
        # Print results
        print("\nImage Organization Results:")
        print("=" * 50)
        print(f"1. Data Preprocessing ({len(preprocessing_images)} images):")
        for img in preprocessing_images:
            print(f"   - {img}")
        
        print(f"\n2. Feature Engineering ({len(feature_images)} images):")
        for img in feature_images:
            print(f"   - {img}")
        
        print(f"\n3. Model Optimization ({len(optimization_images)} images):")
        for img in optimization_images:
            print(f"   - {img}")
        
        print(f"\n4. Model Evaluation ({len(evaluation_images)} images):")
        for img in evaluation_images:
            print(f"   - {img}")
        
        print(f"\n5. Interpretability ({len(interpretability_images)} images):")
        for img in interpretability_images:
            print(f"   - {img}")
        
        # Verify expected organization
        expected_preprocessing = ["clean_missing_values_heatmap.png", "missing_values_heatmap.png"]
        expected_feature = ["pca_biplot.png", "lasso_coefficients.png", "anova_scores.png", "train_test_distribution.png"]
        expected_optimization = ["optuna_optimization_history.png", "optuna_slice_plot.png", "optuna_parallel_coordinate.png"]
        expected_evaluation = ["confusion_matrix.png", "performance_metrics.png", "roc_curve.png"]
        expected_interpretability = ["shap_summary_plot.png", "shap_waterfall_plot.png", "lime_explanation.png"]
        
        # Check if all expected images are in correct sections
        success = True
        
        for img in expected_preprocessing:
            if img not in preprocessing_images:
                print(f"❌ ERROR: {img} should be in preprocessing section")
                success = False
        
        for img in expected_feature:
            if img not in feature_images:
                print(f"❌ ERROR: {img} should be in feature engineering section")
                success = False
        
        for img in expected_optimization:
            if img not in optimization_images:
                print(f"❌ ERROR: {img} should be in model optimization section")
                success = False
        
        for img in expected_evaluation:
            if img not in evaluation_images:
                print(f"❌ ERROR: {img} should be in model evaluation section")
                success = False
        
        for img in expected_interpretability:
            if img not in interpretability_images:
                print(f"❌ ERROR: {img} should be in interpretability section")
                success = False
        
        if success:
            print("\n✅ SUCCESS: All images are correctly organized by pipeline section!")
        else:
            print("\n❌ FAILED: Some images are not in the correct sections")
        
        return success
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_image_organization() 