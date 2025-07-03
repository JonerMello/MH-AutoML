#!/usr/bin/env python3
"""
Test script to verify SHAP force plot PNG generation.
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

def test_shap_png_generation():
    """Test SHAP PNG generation to ensure it's not blank."""
    
    print("🧪 Testing SHAP PNG generation...")
    
    # Create a simple dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create results folder
    results_folder = "test_shap_results"
    os.makedirs(results_folder, exist_ok=True)
    
    # Create feature selection info (simulating no feature selection)
    feature_selection_info = {
        'applied_method': None,
        'selected_features_info': None
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shap_file = interpretability._generate_shap_explanation(timestamp)
    
    # Check if files were created
    shap_force_png_path = os.path.join(results_folder, f'shap_force_plot_RandomForestClassifier_{timestamp}.png')
    shap_summary_png_path = os.path.join(results_folder, f'shap_summary_plot_RandomForestClassifier_{timestamp}.png')
    
    print(f"📁 Results folder: {results_folder}")
    print(f"📄 SHAP HTML: {shap_file}")
    print(f"🖼️ SHAP Force PNG: {shap_force_png_path}")
    print(f"📊 SHAP Summary PNG: {shap_summary_png_path}")
    
    # Check if files exist
    files_exist = {
        'shap_html': os.path.exists(shap_file) if shap_file else False,
        'shap_force_png': os.path.exists(shap_force_png_path),
        'shap_summary_png': os.path.exists(shap_summary_png_path)
    }
    
    print("\n📋 File existence check:")
    for file_type, exists in files_exist.items():
        status = "✅" if exists else "❌"
        print(f"  {status} {file_type}: {exists}")
    
    # Check file sizes to ensure they're not empty
    if files_exist['shap_force_png']:
        size_force_png = os.path.getsize(shap_force_png_path)
        print(f"\n📏 SHAP Force PNG size: {size_force_png:,} bytes")
        
        if size_force_png > 1000:  # Should be more than 1KB for a valid image
            print("✅ SHAP Force PNG has reasonable size - not blank!")
        else:
            print("⚠️ SHAP Force PNG might be blank or corrupted")
    
    if files_exist['shap_summary_png']:
        size_summary_png = os.path.getsize(shap_summary_png_path)
        print(f"📏 SHAP Summary PNG size: {size_summary_png:,} bytes")
        
        if size_summary_png > 1000:
            print("✅ SHAP Summary PNG has reasonable size!")
        else:
            print("⚠️ SHAP Summary PNG might be blank or corrupted")
    
    # List all files in results folder
    print(f"\n📂 All files in {results_folder}:")
    for file in sorted(os.listdir(results_folder)):
        file_path = os.path.join(results_folder, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  📄 {file} ({size:,} bytes)")
    
    print("\n🎯 Test completed!")
    return files_exist

if __name__ == "__main__":
    test_shap_png_generation() 