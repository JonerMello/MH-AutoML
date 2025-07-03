#!/usr/bin/env python3
"""
Test script to verify SHAP force plot feature names are displayed correctly.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.interpretability.interpretability import Interpretability
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def test_shap_feature_names():
    """Test SHAP force plot to ensure feature names are displayed correctly."""
    
    print("🧪 Testing SHAP force plot feature names...")
    
    # Create a simple dataset with meaningful feature names
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Create meaningful feature names
    feature_names = ['age', 'income', 'education', 'credit_score', 'employment_years']
    
    # Convert to DataFrame to preserve feature names
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create results folder
    results_folder = "test_shap_feature_names_results"
    os.makedirs(results_folder, exist_ok=True)
    
    # Create feature selection info (simulating no feature selection)
    feature_selection_info = {
        'method': None,
        'feature_names': feature_names,
        'original_features': feature_names,
        'transformer': None,
        'selected_features_info': {}
    }
    
    # Create Interpretability object
    interpretability = Interpretability(
        best_model=model,
        X_train_selected=np.array(X_train),
        X_test_selected=np.array(X_test),
        y_train=np.array(y_train),
        feature_selection_info=feature_selection_info,
        results_folder=results_folder
    )
    
    # Set feature names
    interpretability.feature_names = feature_names
    
    # Generate SHAP explanations
    print("🔍 Generating SHAP explanations...")
    shap_file = interpretability._generate_shap_explanation(datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    if shap_file:
        print(f"✅ SHAP file generated: {shap_file}")
        
        # Check if the HTML file contains the feature names
        with open(shap_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for feature names in the HTML content
        found_features = []
        for feature in feature_names:
            if feature in content:
                found_features.append(feature)
        
        print(f"📊 Found {len(found_features)}/{len(feature_names)} feature names in HTML:")
        for feature in found_features:
            print(f"   ✅ {feature}")
        
        missing_features = [f for f in feature_names if f not in found_features]
        if missing_features:
            print(f"❌ Missing feature names:")
            for feature in missing_features:
                print(f"   ❌ {feature}")
        else:
            print("🎉 All feature names found in SHAP force plot!")
            
        return len(found_features) == len(feature_names)
    else:
        print("❌ Failed to generate SHAP file")
        return False

if __name__ == "__main__":
    success = test_shap_feature_names()
    if success:
        print("\n✅ Test passed! Feature names are correctly displayed in SHAP force plot.")
    else:
        print("\n❌ Test failed! Feature names are not correctly displayed in SHAP force plot.") 