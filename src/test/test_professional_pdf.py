#!/usr/bin/env python3
"""
Teste para verificar o relatório PDF profissional do MH-AutoML
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionar o diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_professional_pdf():
    """Testa a geração do relatório PDF profissional"""
    print("🧪 Testando relatório PDF profissional...")
    
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
    
    # Gerar alguns gráficos de teste
    print("📊 Gerando gráficos de teste...")
    
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
    plt.savefig(os.path.join(results_folder, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Curva ROC
    from sklearn.metrics import roc_curve, auc
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
    plt.savefig(os.path.join(results_folder, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Curva Precisão-Recall
    from sklearn.metrics import precision_recall_curve, average_precision_score
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
    plt.savefig(os.path.join(results_folder, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Distribuição de Probabilidades
    plt.figure(figsize=(10, 6))
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
    plt.savefig(os.path.join(results_folder, 'probability_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Métricas por Classe
    from sklearn.metrics import precision_score, recall_score, f1_score
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
    
    for i, (prec, rec, f1) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
        ax.text(i - width, prec + 0.01, f'{prec:.3f}', ha='center', va='bottom')
        ax.text(i, rec + 0.01, f'{rec:.3f}', ha='center', va='bottom')
        ax.text(i + width, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'metrics_by_class.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráficos de teste gerados")
    
    # Preparar dados para o relatório
    report = classification_report(y_test, y_pred, target_names=["Benign", "Malware"], output_dict=True)
    
    # Dados simulados
    display_data = {
        'total_samples': n_samples,
        'total_features': n_features,
        'numeric_features': n_features,
        'categorical_features': 0,
        'class_0_count': np.sum(y == 0),
        'class_0_percentage': np.sum(y == 0) / len(y) * 100,
        'class_1_count': np.sum(y == 1),
        'class_1_percentage': np.sum(y == 1) / len(y) * 100
    }
    
    # Testar geração do PDF
    try:
        from model.tools.pdf_report_generator import PDFReportGenerator
        
        print("📄 Gerando relatório PDF profissional...")
        
        pdf_generator = PDFReportGenerator(results_folder)
        pdf_filename = pdf_generator.generate_pdf_report(
            pipeline=None,
            display_data=display_data,
            study=None,
            best_model=model,
            model_name="RandomForestClassifier",
            model_params=model.get_params(),
            select_results=None,
            report=report,
            y_test=y_test,
            y_pred=y_pred,
            feature_names=[f"feature_{i}" for i in range(n_features)],
            feature_selection_info={'method': 'LASSO', 'original_features': n_features, 'selected_features': n_features},
            shap_exp_filepath=None,
            exp_filepath=None,
            lime_exp_filepath=None
        )
        
        if pdf_filename and os.path.exists(pdf_filename):
            size = os.path.getsize(pdf_filename)
            print(f"✅ Relatório PDF profissional gerado: {pdf_filename}")
            print(f"📏 Tamanho: {size / 1024:.1f} KB")
            
            # Verificar se é PDF ou HTML
            if pdf_filename.endswith('.pdf'):
                print("🎉 PDF gerado com sucesso!")
            else:
                print("📄 HTML gerado como fallback")
            
            return True
        else:
            print("❌ Falha na geração do relatório")
            return False
            
    except Exception as e:
        print(f"❌ Erro na geração do relatório: {e}")
        return False

if __name__ == "__main__":
    success = test_professional_pdf()
    if success:
        print("\n✅ Teste do relatório PDF profissional concluído com sucesso!")
    else:
        print("\n💥 Teste falhou!")
        sys.exit(1) 