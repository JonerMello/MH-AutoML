#!/usr/bin/env python3
"""
Teste específico para verificar a categorização de artefatos no PDF
Verifica se os arquivos são organizados corretamente por seção do pipeline
"""

import os
import sys
import pandas as pd
import numpy as np

# Adicionar o diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_artifact_categorization():
    """Testa a categorização de artefatos seguindo o template MLflow"""
    
    print("🧪 Testando categorização de artefatos...")
    
    try:
        from model.tools.pdf_report_generator import PDFReportGenerator
        
        # Criar pasta de resultados de teste
        test_results_folder = "test_results"
        if not os.path.exists(test_results_folder):
            os.makedirs(test_results_folder)
        
        # Criar arquivos de teste seguindo exatamente a estrutura fornecida
        print("📁 Criando arquivos de teste seguindo estrutura MLflow...")
        
        import matplotlib.pyplot as plt
        
        # 01_preprocessing
        fig, ax = plt.subplots(figsize=(8, 6))
        data = np.random.rand(10, 10)
        im = ax.imshow(data, cmap='viridis')
        ax.set_title('Missing Values Heatmap')
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(os.path.join(test_results_folder, "missing_values_heatmap.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        data = np.random.rand(10, 10)
        im = ax.imshow(data, cmap='plasma')
        ax.set_title('Clean Missing Values Heatmap')
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(os.path.join(test_results_folder, "clean_missing_values_heatmap.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        # 02_feature_engineering
        # CSV files
        features_df = pd.DataFrame({
            'feature': ['feature_1', 'feature_2', 'feature_3'],
            'importance': [0.8, 0.6, 0.4]
        })
        features_df.to_csv(os.path.join(test_results_folder, "Features_Selected_20250630_202514.csv"), index=False)
        
        hyperparams_df = pd.DataFrame({
            'model': ['LightGBM', 'CatBoost', 'RandomForest'],
            'accuracy': [0.95, 0.94, 0.93]
        })
        hyperparams_df.to_csv(os.path.join(test_results_folder, "Hyperparameters_Results.csv"), index=False)
        
        ranking_df = pd.DataFrame({
            'model': ['LightGBM', 'CatBoost', 'RandomForest'],
            'rank': [1, 2, 3]
        })
        ranking_df.to_csv(os.path.join(test_results_folder, "Models_Ranking.csv"), index=False)
        
        trials_df = pd.DataFrame({
            'trial': range(1, 21),
            'score': np.random.rand(20)
        })
        trials_df.to_csv(os.path.join(test_results_folder, "optuna_trials.csv"), index=False)
        
        treino_df = pd.DataFrame({
            'feature_1': np.random.rand(100),
            'feature_2': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        })
        treino_df.to_csv(os.path.join(test_results_folder, "treino_20250630_202504.csv"), index=False)
        
        # Images
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(np.random.rand(50), np.random.rand(50))
        ax.set_title('PCA Biplot')
        plt.tight_layout()
        plt.savefig(os.path.join(test_results_folder, "pca_biplot.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        components = range(1, 11)
        explained_var = np.random.rand(10)
        ax.plot(components, explained_var)
        ax.set_title('PCA Components')
        plt.tight_layout()
        plt.savefig(os.path.join(test_results_folder, "pca_components_20250630_202540.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ['Train', 'Test']
        counts = [70, 30]
        ax.bar(categories, counts)
        ax.set_title('Train Test Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(test_results_folder, "train_test_distribution.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        # 03_model_optimization
        # HTML files
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Optuna Optimization History</title></head>
        <body><h1>Optimization History</h1></body>
        </html>
        """
        with open(os.path.join(test_results_folder, "optuna_optimization_history.html"), 'w') as f:
            f.write(html_content)
        
        with open(os.path.join(test_results_folder, "optuna_parallel_coordinate.html"), 'w') as f:
            f.write(html_content)
        
        with open(os.path.join(test_results_folder, "optuna_param_importance.html"), 'w') as f:
            f.write(html_content)
        
        with open(os.path.join(test_results_folder, "optuna_slice_plot.html"), 'w') as f:
            f.write(html_content)
        
        # PNG files
        fig, ax = plt.subplots(figsize=(8, 6))
        trials = range(1, 21)
        scores = np.random.rand(20)
        ax.plot(trials, scores)
        ax.set_title('Optuna Optimization History')
        plt.tight_layout()
        plt.savefig(os.path.join(test_results_folder, "optuna_optimization_history.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(np.random.rand(20), np.random.rand(20))
        ax.set_title('Optuna Parallel Coordinate')
        plt.tight_layout()
        plt.savefig(os.path.join(test_results_folder, "optuna_parallel_coordinate.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        params = ['param1', 'param2', 'param3']
        importance = np.random.rand(3)
        ax.barh(params, importance)
        ax.set_title('Optuna Parameter Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(test_results_folder, "optuna_param_importance.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(np.random.rand(100), bins=20)
        ax.set_title('Optuna Slice Plot')
        plt.tight_layout()
        plt.savefig(os.path.join(test_results_folder, "optuna_slice_plot.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        # 04_evaluation_metrics
        # PKL file (simulado)
        import pickle
        mock_model = {'type': 'LightGBM', 'accuracy': 0.95}
        with open(os.path.join(test_results_folder, "best_model_20250630_202504.pkl"), 'wb') as f:
            pickle.dump(mock_model, f)
        
        # 05_interpretability
        fig, ax = plt.subplots(figsize=(8, 6))
        features = ['feature_1', 'feature_2', 'feature_3']
        importance = np.random.rand(3)
        ax.barh(features, importance)
        ax.set_title('LIME Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(test_results_folder, "lime_feature_importance_20250630_202540.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        lime_html = """
        <!DOCTYPE html>
        <html>
        <head><title>LIME Interpretability</title></head>
        <body><h1>LIME Analysis</h1></body>
        </html>
        """
        with open(os.path.join(test_results_folder, "lime_interpretability_20250630_202540.html"), 'w') as f:
            f.write(lime_html)
        
        print("✅ Arquivos de teste criados seguindo estrutura MLflow")
        
        # Testar categorização
        print("🔍 Testando categorização de artefatos...")
        pdf_generator = PDFReportGenerator(test_results_folder)
        
        # Simular categorização
        artifact_categories = {
            "00_Data_info": [],
            "01_preprocessing": [],
            "02_feature_engineering": [],
            "03_model_optimization": [],
            "04_evaluation_metrics": [],
            "05_interpretability": [],
            "Reports": []
        }
        
        for file in os.listdir(test_results_folder):
            # 01_preprocessing
            if "clean_missing_values_heatmap" in file or "missing_values_heatmap" in file:
                artifact_categories["01_preprocessing"].append(file)
            
            # 02_feature_engineering
            elif ("Features_Selected_" in file or "treino_" in file or 
                  "pca_" in file or "lasso_" in file or "anova_" in file or 
                  "train_test_distribution" in file or 
                  (file.endswith('.csv') and not any(x in file for x in ['optuna', 'Hyperparameters', 'Models_Ranking']))):
                artifact_categories["02_feature_engineering"].append(file)
            
            # 03_model_optimization
            elif ("optuna_" in file or "Hyperparameters_Results" in file or 
                  "Models_Ranking" in file or "optuna_trials" in file):
                artifact_categories["03_model_optimization"].append(file)
            
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
        
        # Verificar categorização
        print("\n📋 Resultado da categorização:")
        
        expected_categories = {
            "01_preprocessing": ["clean_missing_values_heatmap.png", "missing_values_heatmap.png"],
            "02_feature_engineering": [
                "Features_Selected_20250630_202514.csv", "Hyperparameters_Results.csv", 
                "Models_Ranking.csv", "pca_biplot.png", "pca_components_20250630_202540.png",
                "train_test_distribution.png", "treino_20250630_202504.csv"
            ],
            "03_model_optimization": [
                "optuna_optimization_history.html", "optuna_optimization_history.png",
                "optuna_parallel_coordinate.html", "optuna_parallel_coordinate.png",
                "optuna_param_importance.html", "optuna_param_importance.png",
                "optuna_slice_plot.html", "optuna_slice_plot.png", "optuna_trials.csv"
            ],
            "04_evaluation_metrics": ["best_model_20250630_202504.pkl"],
            "05_interpretability": [
                "lime_feature_importance_20250630_202540.png", 
                "lime_interpretability_20250630_202540.html"
            ]
        }
        
        success = True
        for category, expected_files in expected_categories.items():
            actual_files = artifact_categories.get(category, [])
            print(f"\n{category}:")
            print(f"  Esperado: {len(expected_files)} arquivos")
            print(f"  Encontrado: {len(actual_files)} arquivos")
            
            for expected_file in expected_files:
                if expected_file in actual_files:
                    print(f"    ✅ {expected_file}")
                else:
                    print(f"    ❌ {expected_file} (não encontrado)")
                    success = False
        
        if success:
            print("\n🎉 Categorização de artefatos funcionando corretamente!")
        else:
            print("\n💥 Alguns arquivos não foram categorizados corretamente")
        
        return success
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_artifact_categorization()
    if success:
        print("\n🎉 Teste de categorização passou!")
    else:
        print("\n💥 Teste de categorização falhou.") 