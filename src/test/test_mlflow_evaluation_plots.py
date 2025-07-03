#!/usr/bin/env python3
"""
Teste para verificar se os gráficos de avaliação estão sendo logados no MLflow
"""

import os
import sys
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionar o diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar MLflow
mlflow.set_tracking_uri('file:./mlruns')

def test_mlflow_evaluation_plots():
    """Testa o log dos gráficos de avaliação no MLflow"""
    print("🧪 Testando log de gráficos de avaliação no MLflow...")
    
    # Criar dados de teste
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Treinar modelo
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Fazer predições
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Criar pasta results se não existir
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    try:
        # Gerar gráficos
        print("📊 Gerando gráficos de avaliação...")
        
        # 1. Matriz de Confusão
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malware'], 
                   yticklabels=['Benign', 'Malware'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        confusion_matrix_path = os.path.join(results_folder, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Curva ROC/AUC
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_curve_path = os.path.join(results_folder, 'roc_curve.png')
        plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Gráficos gerados com sucesso")
        
        # Testar log no MLflow
        print("📝 Testando log no MLflow...")
        
        with mlflow.start_run(run_name="test_evaluation_plots") as run:
            # Log dos gráficos
            if os.path.exists(confusion_matrix_path):
                mlflow.log_artifact(confusion_matrix_path, artifact_path="04_evaluation_metrics")
                print(f"✅ Matriz de confusão logada: 04_evaluation_metrics/confusion_matrix.png")
            
            if os.path.exists(roc_curve_path):
                mlflow.log_artifact(roc_curve_path, artifact_path="04_evaluation_metrics")
                print(f"✅ Curva ROC/AUC logada: 04_evaluation_metrics/roc_curve.png")
            
            # Log de métricas
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("accuracy", (y_pred == y_test).mean())
            
            print(f"✅ Métricas logadas: roc_auc={roc_auc:.3f}")
        
        print(f"\n🎉 Teste do MLflow concluído com sucesso!")
        print(f"📁 Run ID: {run.info.run_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste do MLflow: {e}")
        return False

if __name__ == "__main__":
    success = test_mlflow_evaluation_plots()
    if success:
        print("\n✅ Todos os testes passaram!")
    else:
        print("\n💥 Testes falharam!")
        sys.exit(1) 