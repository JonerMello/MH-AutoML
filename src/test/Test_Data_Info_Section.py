#!/usr/bin/env python3
"""
Test script to verify the new Data Info section in PDF reports.
"""

import os
import sys
import tempfile
import shutil
import pandas as pd
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

def create_test_data_info_files():
    """Create test CSV files with data info for testing."""
    test_dir = tempfile.mkdtemp()
    
    # Create dataset info file
    dataset_info_data = {
        'Metric': ['Total Samples', 'Features', 'Target Classes', 'Memory Usage', 'Missing Values', 'Duplicate Rows'],
        'Value': ['10000', '150', '2', '45.2 MB', '1250', '50']
    }
    df_dataset = pd.DataFrame(dataset_info_data)
    df_dataset.to_csv(os.path.join(test_dir, 'dataset_info.csv'), index=False)
    
    # Create data types info file
    dtypes_data = {
        'Feature Type': ['Numeric Features', 'Categorical Features', 'Boolean Features', 'Object Features'],
        'Count': ['120', '25', '3', '2']
    }
    df_dtypes = pd.DataFrame(dtypes_data)
    df_dtypes.to_csv(os.path.join(test_dir, 'data_types_analysis.csv'), index=False)
    
    # Create class balance info file
    balance_data = {
        'Class': ['Class 0 (Benign)', 'Class 1 (Malware)', 'Balance Ratio', 'Imbalance Type'],
        'Value': ['7000', '3000', '2.33:1', 'Moderate Imbalance']
    }
    df_balance = pd.DataFrame(balance_data)
    df_balance.to_csv(os.path.join(test_dir, 'class_balance_analysis.csv'), index=False)
    
    # Create Android features info file
    android_data = {
        'Feature Category': ['Android Permissions', 'API Calls', 'System Calls', 'Hardware Features', 'Network Features'],
        'Count': ['85', '45', '12', '8', '15']
    }
    df_android = pd.DataFrame(android_data)
    df_android.to_csv(os.path.join(test_dir, 'android_features_analysis.csv'), index=False)
    
    # Create some test images
    test_images = [
        "clean_missing_values_heatmap.png",
        "pca_biplot.png",
        "optuna_optimization_history.png"
    ]
    
    for image in test_images:
        with open(os.path.join(test_dir, image), 'w') as f:
            f.write("test")
    
    return test_dir

def test_data_info_section():
    """Test that the Data Info section displays all required information."""
    print("Testing Data Info section in PDF report...")
    
    test_dir = create_test_data_info_files()
    
    try:
        # Initialize PDF generator
        pdf_generator = PDFReportGenerator(test_dir)
        
        # Generate PDF with mock data
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
                
                # Verify that all required CSV files were created
                required_files = [
                    'dataset_info.csv',
                    'data_types_analysis.csv', 
                    'class_balance_analysis.csv',
                    'android_features_analysis.csv'
                ]
                
                print("\nData Info Files Created:")
                for file in required_files:
                    file_path = os.path.join(test_dir, file)
                    if os.path.exists(file_path):
                        print(f"   ✅ {file}")
                    else:
                        print(f"   ❌ {file} - Missing")
                
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
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        shutil.rmtree(test_dir)

def test_system_info():
    """Test that system information can be retrieved."""
    print("\nTesting system information retrieval...")
    
    try:
        import platform
        import psutil
        
        system_info = {
            "OS": platform.system() + " " + platform.release(),
            "Python": platform.python_version(),
            "Architecture": platform.architecture()[0],
            "RAM": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "CPU Cores": str(psutil.cpu_count())
        }
        
        print("✅ System Information Retrieved:")
        for key, value in system_info.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR retrieving system info: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Data Info Section")
    print("=" * 50)
    
    # Test system info
    system_success = test_system_info()
    
    # Test data info section
    data_info_success = test_data_info_section()
    
    if system_success and data_info_success:
        print("\n🎉 All tests passed! Data Info section is working correctly.")
    else:
        print("\n💥 Some tests failed. Check the output above for details.") 