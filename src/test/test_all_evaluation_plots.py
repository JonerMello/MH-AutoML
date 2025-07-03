#!/usr/bin/env python3
"""
Teste completo para verificar todos os gráficos de avaliação
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve, 
                           average_precision_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionar o diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_all_evaluation_plots():
    """Testa a geração de todos os gráficos de avaliação"""
    print("🧪 Testando todos os gráficos de avaliação...")
    
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
    
    generated_files = []
    
    try:
        # 1. Matriz de Confusão
        print("📊 Gerando matriz de confusão...")
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
        generated_files.append(('confusion_matrix.png', confusion_matrix_path))
        print(f"✅ Matriz de confusão salva: {confusion_matrix_path}")
        
        # 2. Curva ROC/AUC
        print("📈 Gerando curva ROC/AUC...")
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
        generated_files.append(('roc_curve.png', roc_curve_path))
        print(f"✅ Curva ROC/AUC salva: {roc_curve_path}")
        
        # 3. Curva Precisão-Recall
        print("📊 Gerando curva precisão-recall...")
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        plt.plot(recall, precision, color='green', lw=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        plt.axhline(y=1, color='navy', lw=2, linestyle='--', label='Perfect Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        precision_recall_path = os.path.join(results_folder, 'precision_recall_curve.png')
        plt.savefig(precision_recall_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(('precision_recall_curve.png', precision_recall_path))
        print(f"✅ Curva precisão-recall salva: {precision_recall_path}")
        
        # 4. Distribuição de Probabilidades
        print("📊 Gerando distribuição de probabilidades...")
        plt.figure(figsize=(10, 6))
        
        # Separar probabilidades por classe real
        benign_probs = y_pred_proba[y_test == 0]
        malware_probs = y_pred_proba[y_test == 1]
        
        plt.hist(benign_probs, bins=30, alpha=0.7, label='Benign', color='blue', density=True)
        plt.hist(malware_probs, bins=30, alpha=0.7, label='Malware', color='red', density=True)
        plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Distribution of Predicted Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        prob_dist_path = os.path.join(results_folder, 'probability_distribution.png')
        plt.savefig(prob_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(('probability_distribution.png', prob_dist_path))
        print(f"✅ Distribuição de probabilidades salva: {prob_dist_path}")
        
        # 5. Métricas por Classe
        print("📊 Gerando métricas por classe...")
        # Calcular métricas por classe
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        classes = ['Benign', 'Malware']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(classes))
        width = 0.25
        
        ax.bar(x - width, [precision_per_class[0], precision_per_class[1]], width, label='Precision', alpha=0.8)
        ax.bar(x, [recall_per_class[0], recall_per_class[1]], width, label='Recall', alpha=0.8)
        ax.bar(x + width, [f1_per_class[0], f1_per_class[1]], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, (prec, rec, f1) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            ax.text(i - width, prec + 0.01, f'{prec:.3f}', ha='center', va='bottom')
            ax.text(i, rec + 0.01, f'{rec:.3f}', ha='center', va='bottom')
            ax.text(i + width, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        metrics_by_class_path = os.path.join(results_folder, 'metrics_by_class.png')
        plt.savefig(metrics_by_class_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(('metrics_by_class.png', metrics_by_class_path))
        print(f"✅ Métricas por classe salvas: {metrics_by_class_path}")
        
        # Verificar se todos os arquivos foram criados
        print(f"\n📋 Resumo dos arquivos gerados:")
        for filename, filepath in generated_files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"   ✅ {filename} ({size} bytes)")
            else:
                print(f"   ❌ {filename} (não encontrado)")
        
        print(f"\n🎉 Teste concluído com sucesso!")
        print(f"📁 Arquivos gerados em: {results_folder}")
        print(f"📊 Total de gráficos: {len(generated_files)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao gerar gráficos: {e}")
        return False

if __name__ == "__main__":
    success = test_all_evaluation_plots()
    if success:
        print("\n✅ Todos os testes passaram!")
    else:
        print("\n💥 Testes falharam!")
        sys.exit(1) 