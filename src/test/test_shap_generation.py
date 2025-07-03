#!/usr/bin/env python3
"""
Test script to verify SHAP generation functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from model.interpretability.interpretability import Interpretability
import tempfile

def test_shap_generation():
    """Test SHAP generation with a simple dataset"""
    print("🧪 Testing SHAP generation...")
    
    # Create a simple test dataset
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create feature selection info
    feature_selection_info = {
        'method': 'lasso',
        'feature_names': [f'feature_{i}' for i in range(n_features)],
        'original_features': [f'feature_{i}' for i in range(n_features)]
    }
    
    # Create temporary results folder
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Initialize Interpretability
        interpretability = Interpretability(
            best_model=model,
            X_train_selected=X_train,
            X_test_selected=X_test,
            y_train=y_train,
            feature_selection_info=feature_selection_info,
            results_folder=temp_dir
        )
        
        # Test SHAP generation
        print("🔍 Testing SHAP generation...")
        shap_file = interpretability._generate_shap_explanation("test_timestamp")
        
        if shap_file:
            print(f"✅ SHAP file generated: {shap_file}")
            print(f"📁 File exists: {os.path.exists(shap_file)}")
            
            # Check if files were created
            files_in_dir = os.listdir(temp_dir)
            print(f"📋 Files in directory: {files_in_dir}")
            
            return True
        else:
            print("❌ SHAP generation failed")
            return False

if __name__ == "__main__":
    success = test_shap_generation()
    if success:
        print("\n🎉 SHAP generation test passed!")
    else:
        print("\n💥 SHAP generation test failed!") 