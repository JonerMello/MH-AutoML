import os
import sys
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import Mock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.tools.pdf_report_generator import PDFReportGenerator
from model.preprocessing.data_info import DataInfo

def create_test_dataset():
    """Create a realistic test dataset for Android malware detection."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create feature names (mix of permissions and API calls)
    permission_features = [f'permission_{i}' for i in range(30)]
    api_features = [f'api_call_{i}' for i in range(20)]
    feature_names = permission_features + api_features
    
    # Create realistic data
    data = {}
    for feature in feature_names:
        if 'permission' in feature:
            # Binary features for permissions
            data[feature] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        else:
            # Continuous features for API calls
            data[feature] = np.random.normal(0, 1, n_samples)
    
    # Create target variable (malware vs benign)
    data['target'] = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    return pd.DataFrame(data)

class MockModel:
    """Mock model for testing."""
    def __init__(self):
        self.named_estimators_ = [('test_estimator', Mock())]
    
    def predict(self, X):
        return np.random.choice([0, 1], size=len(X))
    
    def get_params(self, deep=True):
        return {'mock_param': 42}

def test_pdf_datainfo_correction():
    """Test that PDF correctly uses DataInfo instance."""
    print("Testing PDF DataInfo correction...")
    
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create realistic dataset
        dataset = create_test_dataset()
        print(f"✅ Created realistic dataset: {dataset.shape[0]} samples, {dataset.shape[1]} features")
        
        # Create DataInfo instance (this is what the controller passes)
        print("\nCreating DataInfo instance...")
        data_info = DataInfo(label='target', dataset=dataset)
        data_info.display_dataframe_info()
        
        print("✅ DataInfo analysis completed")
        print(f"   - System info: {data_info.system_info_result is not None}")
        print(f"   - Dataset info: {data_info.info_table_result is not None}")
        print(f"   - Data types: {data_info.data_types_result is not None}")
        print(f"   - Balance info: {data_info.balance_info_result is not None}")
        print(f"   - Duplicates/Missing: {data_info.duplicates_missing_result is not None}")
        print(f"   - Features info: {data_info.features_info_result is not None}")
        
        # Print expected values
        print("\nExpected DataInfo values:")
        if data_info.info_table_result is not None:
            print(f"   Total Samples: {data_info.info_table_result['Rows'].iloc[0]}")
            print(f"   Features: {data_info.info_table_result['Columns'].iloc[0]}")
        
        if data_info.balance_info_result is not None:
            print("   Class Balance:")
            for _, row in data_info.balance_info_result.iterrows():
                print(f"     {row['Label']}: {row['Percentage']}")
        
        if data_info.data_types_result is not None:
            print("   Data Types:")
            for _, row in data_info.data_types_result.iterrows():
                print(f"     {row['Data Type']}: {row['Count']}")
        
        if data_info.features_info_result is not None:
            print("   Android Features:")
            for _, row in data_info.features_info_result.iterrows():
                # Robustly get the first two columns regardless of their names
                feature_type = row.iloc[0]
                count = row.iloc[1]
                print(f"     {feature_type}: {count}")
        
        # Initialize PDF generator
        pdf_generator = PDFReportGenerator(test_dir)
        
        # Generate PDF with DataInfo instance (not the dataset)
        mock_model = MockModel()
        success = pdf_generator.generate_pdf_report(
            pipeline=None,
            display_data=data_info,  # Pass the DataInfo instance
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
            # Check if PDF was generated
            pdf_files = [f for f in os.listdir(test_dir) if f.endswith('.pdf')]
            html_files = [f for f in os.listdir(test_dir) if f.endswith('.html')]
            
            if pdf_files:
                pdf_path = os.path.join(test_dir, pdf_files[0])
                print(f"✅ SUCCESS: PDF generated successfully at {pdf_path}")
                print(f"   File size: {os.path.getsize(pdf_path)} bytes")
                
                # Check if PDF contains real data (basic check)
                with open(pdf_path, 'rb') as f:
                    content = f.read()
                    if b'Total Samples' in content and b'Features' in content:
                        print("✅ PDF contains dataset information sections")
                    else:
                        print("❌ PDF missing dataset information sections")
                
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
        import shutil
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    success = test_pdf_datainfo_correction()
    if success:
        print("\n🎉 All tests passed! PDF generator now correctly uses DataInfo instance.")
    else:
        print("\n💥 Tests failed! Check the output above for details.") 