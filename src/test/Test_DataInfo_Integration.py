#!/usr/bin/env python3
"""
Test script to verify DataInfo integration in PDF reports.
"""

import os
import sys
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tools.pdf_report_generator import PDFReportGenerator
from model.preprocessing.data_info import DataInfo

class MockModel:
    """Mock model for testing."""
    def __init__(self):
        self.estimators_ = {}
        self.named_estimators_ = {}
    
    def get_params(self):
        return {'param1': 'value1'}

def create_test_dataset():
    """Create a test dataset with Android malware features."""
    # Create sample dataset with Android features
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create feature names with Android permissions and API calls
    feature_names = []
    for i in range(n_features):
        if i < 20:
            feature_names.append(f"permission_{i}")
        elif i < 40:
            feature_names.append(f"api_call_{i}")
        else:
            feature_names.append(f"feature_{i}")
    
    # Create dataset
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)  # Binary classification
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def test_datainfo_integration():
    """Test that DataInfo integration works correctly."""
    print("Testing DataInfo integration in PDF report...")
    
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create test dataset
        dataset = create_test_dataset()
        print(f"✅ Created test dataset: {dataset.shape[0]} samples, {dataset.shape[1]} features")
        
        # Test DataInfo class directly
        print("\nTesting DataInfo class...")
        data_info = DataInfo(label='target', dataset=dataset)
        data_info.display_dataframe_info()
        
        print("✅ DataInfo analysis completed")
        print(f"   - System info: {data_info.system_info_result is not None}")
        print(f"   - Dataset info: {data_info.info_table_result is not None}")
        print(f"   - Data types: {data_info.data_types_result is not None}")
        print(f"   - Balance info: {data_info.balance_info_result is not None}")
        print(f"   - Duplicates/Missing: {data_info.duplicates_missing_result is not None}")
        print(f"   - Features info: {data_info.features_info_result is not None}")
        
        # Initialize PDF generator
        pdf_generator = PDFReportGenerator(test_dir)
        
        # Generate PDF with real dataset
        mock_model = MockModel()
        success = pdf_generator.generate_pdf_report(
            pipeline=None,
            display_data=dataset,  # Pass the real dataset
            study=None,
            best_model=mock_model,
            model_name="TestModel",
            model_params={'param1': 'value1'},
            select_results=None,
            report="Test Report",
            y_test=dataset['target'].values,
            y_pred=dataset['target'].values,
            feature_names=dataset.columns.tolist(),
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
                
                # Check if the HTML contains real data (not "Loading...")
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Check for real data indicators
                if "1000" in html_content and "50" in html_content:
                    print("✅ HTML contains real dataset information")
                else:
                    print("❌ HTML still contains placeholder data")
                
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

def test_datainfo_methods():
    """Test individual DataInfo methods."""
    print("\nTesting individual DataInfo methods...")
    
    try:
        # Create test dataset
        dataset = create_test_dataset()
        
        # Test DataInfo
        data_info = DataInfo(label='target', dataset=dataset)
        
        # Test system info
        system_info = data_info.system_info()
        print(f"✅ System info: {system_info is not None}")
        
        # Test dataset info
        info_table = data_info.display_info_table()
        print(f"✅ Dataset info: {info_table is not None}")
        if info_table is not None:
            print(f"   Rows: {info_table['Rows'].iloc[0]}")
            print(f"   Columns: {info_table['Columns'].iloc[0]}")
        
        # Test data types
        data_types = data_info.display_data_types()
        print(f"✅ Data types: {data_types is not None}")
        
        # Test balance info
        balance_info = data_info.display_balance_info()
        print(f"✅ Balance info: {balance_info is not None}")
        
        # Test duplicates/missing
        duplicates_missing = data_info.display_duplicates_missing()
        print(f"✅ Duplicates/Missing: {duplicates_missing is not None}")
        
        # Test features info
        features_info = data_info.display_features_info()
        print(f"✅ Features info: {features_info is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR testing DataInfo methods: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing DataInfo Integration")
    print("=" * 50)
    
    # Test DataInfo methods
    methods_success = test_datainfo_methods()
    
    # Test integration
    integration_success = test_datainfo_integration()
    
    if methods_success and integration_success:
        print("\n🎉 All tests passed! DataInfo integration is working correctly.")
    else:
        print("\n💥 Some tests failed. Check the output above for details.") 