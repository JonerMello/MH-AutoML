#!/usr/bin/env python3
"""
Quick test for PDF generation with new image organization.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tools.pdf_report_generator import PDFReportGenerator

class MockModel:
    """Mock model for testing."""
    def __init__(self):
        self.estimators_ = {}
        self.named_estimators_ = {}
    
    def get_params(self):
        return {'param1': 'value1'}

def create_test_environment():
    """Create a test environment with sample files."""
    test_dir = tempfile.mkdtemp()
    
    # Create sample images for each section
    test_images = [
        "clean_missing_values_heatmap.png",
        "pca_biplot.png", 
        "optuna_optimization_history.png",
        "confusion_matrix.png",
        "shap_summary_plot.png"
    ]
    
    for image in test_images:
        with open(os.path.join(test_dir, image), 'w') as f:
            f.write("test")
    
    # Create sample CSV files
    csv_files = [
        "Features_Selected_20250630_202514.csv",
        "Hyperparameters_Results.csv",
        "Models_Ranking.csv"
    ]
    
    for csv in csv_files:
        with open(os.path.join(test_dir, csv), 'w') as f:
            f.write("test")
    
    return test_dir

def test_pdf_generation():
    """Test PDF generation with new organization."""
    print("Testing PDF generation with new image organization...")
    
    test_dir = create_test_environment()
    
    try:
        # Initialize PDF generator
        pdf_generator = PDFReportGenerator(test_dir)
        
        # Generate PDF with minimal parameters
        mock_model = MockModel()
        success = pdf_generator.generate_pdf_report(
            pipeline=None,
            display_data=None,
            study=None,
            best_model=mock_model,
            model_name="TestModel",
            model_params={'param1': 'value1'},
            select_results=None,
            report="Test Report",
            y_test=[],
            y_pred=[],
            feature_names=[],
            feature_selection_info={},
            shap_exp_filepath=None,
            exp_filepath=None,
            lime_exp_filepath=None
        )
        
        if success:
            # Check if PDF or HTML was generated
            pdf_files = [f for f in os.listdir(test_dir) if f.endswith('.pdf')]
            html_files = [f for f in os.listdir(test_dir) if f.endswith('.html')]
            
            if pdf_files:
                pdf_path = os.path.join(test_dir, pdf_files[0])
                print(f"✅ SUCCESS: PDF generated successfully at {pdf_path}")
                print(f"   File size: {os.path.getsize(pdf_path)} bytes")
                return True
            elif html_files:
                html_path = os.path.join(test_dir, html_files[0])
                print(f"✅ SUCCESS: HTML generated as fallback at {html_path}")
                print(f"   File size: {os.path.getsize(html_path)} bytes")
                return True
            else:
                print("❌ FAILED: No report file generated")
                return False
        else:
            print("❌ FAILED: PDF generation failed")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        # Clean up
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_pdf_generation() 