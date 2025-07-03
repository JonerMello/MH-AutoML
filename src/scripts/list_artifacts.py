#!/usr/bin/env python3
"""
Script para listar todos os artefatos gerados pelo MH-AutoML
"""

import os
import glob
from datetime import datetime
from pathlib import Path

def list_artifacts():
    """Lista todos os artefatos gerados pelo MH-AutoML"""
    
    print("📋 LISTA COMPLETA DE ARTEFATOS - MH-AutoML")
    print("=" * 60)
    print(f"📅 Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Diretório: {os.getcwd()}")
    print()
    
    # Estrutura de seções do MLflow
    sections = {
        "00_Data_info": "Informações sobre o dataset",
        "01_preprocessing": "Pré-processamento de dados", 
        "02_feature_engineering": "Engenharia e seleção de features",
        "03_model_optimization": "Otimização de hiperparâmetros",
        "04_evaluation_metrics": "Métricas e gráficos de avaliação",
        "05_interpretability": "Interpretabilidade do modelo",
        "MH_Best_Model": "Modelo final registrado",
        "report": "Relatórios finais"
    }
    
    # Mapeamento de extensões para tipos
    extension_types = {
        '.png': 'Imagem',
        '.jpg': 'Imagem', 
        '.jpeg': 'Imagem',
        '.html': 'Relatório Interativo',
        '.csv': 'Dados',
        '.pkl': 'Modelo',
        '.pdf': 'Relatório PDF',
        '.yaml': 'Configuração',
        '.json': 'Dados JSON',
        '.txt': 'Texto',
        '.log': 'Log'
    }
    
    total_artifacts = 0
    total_size = 0
    
    # Verificar pasta results
    results_path = Path("results")
    if not results_path.exists():
        print("❌ Pasta 'results' não encontrada!")
        return
    
    print("📊 ARTEFATOS ENCONTRADOS:")
    print("-" * 60)
    
    # Listar todos os arquivos na pasta results
    all_files = list(results_path.rglob("*"))
    all_files = [f for f in all_files if f.is_file()]
    
    # Agrupar por tipo de arquivo
    files_by_extension = {}
    for file_path in all_files:
        ext = file_path.suffix.lower()
        if ext not in files_by_extension:
            files_by_extension[ext] = []
        files_by_extension[ext].append(file_path)
    
    # Mostrar estatísticas por extensão
    print("📈 ESTATÍSTICAS POR TIPO DE ARQUIVO:")
    print("-" * 40)
    
    for ext, files in sorted(files_by_extension.items()):
        file_type = extension_types.get(ext, 'Outro')
        total_size_ext = sum(f.stat().st_size for f in files)
        total_artifacts += len(files)
        total_size += total_size_ext
        
        print(f"📄 {ext.upper()} ({file_type}): {len(files)} arquivos - {total_size_ext / 1024:.1f} KB")
    
    print()
    print("📋 LISTA DETALHADA DE ARQUIVOS:")
    print("-" * 60)
    
    # Agrupar por categoria funcional
    categories = {
        "Gráficos de Avaliação": ["confusion_matrix", "roc_curve", "precision_recall_curve", 
                                 "probability_distribution", "metrics_by_class"],
        "Gráficos de Otimização": ["optuna_", "hyperparameters", "models_ranking"],
        "Gráficos de Features": ["pca_", "lasso_", "anova_", "feature_importance"],
        "Gráficos de Interpretabilidade": ["shap_", "lime_", "decision_tree"],
        "Gráficos de Pré-processamento": ["missing_values", "clean_"],
        "Dados e Modelos": [".csv", ".pkl", "treino_", "features_selected"],
        "Relatórios": [".html", ".pdf", "report_"],
        "Performance": ["performance_", "best_model_"]
    }
    
    for category, patterns in categories.items():
        category_files = []
        for file_path in all_files:
            file_name = file_path.name.lower()
            if any(pattern.lower() in file_name for pattern in patterns):
                category_files.append(file_path)
        
        if category_files:
            print(f"\n🎯 {category.upper()}:")
            for file_path in sorted(category_files):
                size = file_path.stat().st_size
                print(f"   📄 {file_path.name} ({size / 1024:.1f} KB)")
    
    # Mostrar arquivos não categorizados
    categorized_files = set()
    for patterns in categories.values():
        for file_path in all_files:
            file_name = file_path.name.lower()
            if any(pattern.lower() in file_name for pattern in patterns):
                categorized_files.add(file_path)
    
    uncategorized = [f for f in all_files if f not in categorized_files]
    if uncategorized:
        print(f"\n🔍 OUTROS ARQUIVOS:")
        for file_path in sorted(uncategorized):
            size = file_path.stat().st_size
            print(f"   📄 {file_path.name} ({size / 1024:.1f} KB)")
    
    print()
    print("📊 RESUMO FINAL:")
    print("-" * 40)
    print(f"📁 Total de artefatos: {total_artifacts}")
    print(f"💾 Tamanho total: {total_size / (1024*1024):.2f} MB")
    print(f"📅 Última atualização: {datetime.fromtimestamp(max(f.stat().st_mtime for f in all_files)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verificar se há MLflow runs
    mlruns_path = Path("mlruns")
    if mlruns_path.exists():
        print(f"\n🔗 MLflow runs encontrados em: {mlruns_path}")
        print("   Execute 'mlflow ui' para visualizar no navegador")
    
    print()
    print("✅ Listagem concluída!")

def categorize_by_mlflow_section():
    """Categoriza artefatos por seção do MLflow"""
    
    print("\n🗂️ CATEGORIZAÇÃO POR SEÇÃO MLFLOW:")
    print("-" * 50)
    
    mlflow_sections = {
        "00_Data_info": ["dataset_info", "data_analysis"],
        "01_preprocessing": ["missing_values", "clean_", "preprocessing"],
        "02_feature_engineering": ["pca_", "lasso_", "anova_", "feature_", "treino_", "features_selected"],
        "03_model_optimization": ["optuna_", "hyperparameters", "models_ranking"],
        "04_evaluation_metrics": ["confusion_matrix", "roc_curve", "precision_recall", "probability_distribution", 
                                 "metrics_by_class", "performance_", "best_model_"],
        "05_interpretability": ["shap_", "lime_", "decision_tree"],
        "MH_Best_Model": ["model.pkl", "conda.yaml", "python_env.yaml", "MLmodel"],
        "report": ["report_", ".html", ".pdf"]
    }
    
    results_path = Path("results")
    if not results_path.exists():
        return
    
    all_files = list(results_path.rglob("*"))
    all_files = [f for f in all_files if f.is_file()]
    
    for section, patterns in mlflow_sections.items():
        section_files = []
        for file_path in all_files:
            file_name = file_path.name.lower()
            if any(pattern.lower() in file_name for pattern in patterns):
                section_files.append(file_path)
        
        if section_files:
            print(f"\n📂 {section}:")
            for file_path in sorted(section_files):
                size = file_path.stat().st_size
                print(f"   📄 {file_path.name} ({size / 1024:.1f} KB)")

if __name__ == "__main__":
    list_artifacts()
    categorize_by_mlflow_section() 