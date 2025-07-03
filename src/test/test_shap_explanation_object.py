import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap

# Criar dados de teste
np.random.seed(42)
X = np.random.randn(100, 5)
y = (X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1 > 0).astype(int)

# Nomes das features
feature_names = ['age', 'income', 'education', 'credit_score', 'employment_years']

# Treinar modelo
rf = RandomForestClassifier(n_estimators=5, random_state=42)
rf.fit(X, y)

# Criar explainer
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X[0:1])
expected_value = explainer.expected_value

print("=== Teste do Objeto Explanation ===")
print(f"Feature names: {feature_names}")
print(f"SHAP values shape: {shap_values.shape}")
print(f"Expected value: {expected_value}")

# Testar diferentes formas de criar o objeto Explanation
print("\n1. Testando com Explanation object:")
try:
    explanation = shap.Explanation(
        values=shap_values[..., 1],  # Usar classe positiva
        base_values=expected_value[1],
        data=X[0:1],
        feature_names=feature_names
    )
    print("✅ Explanation object criado com sucesso")
    print(f"   - Feature names no objeto: {explanation.feature_names}")
    
    # Testar force plot
    force_plot = shap.plots.force(explanation[0])
    print("✅ Force plot gerado com sucesso")
    
except Exception as e:
    print(f"❌ Erro: {e}")

print("\n2. Testando método alternativo (sem Explanation object):")
try:
    # Método direto com feature_names
    force_plot2 = shap.plots.force(expected_value[1], shap_values[..., 1], feature_names=feature_names)
    print("✅ Force plot direto gerado com sucesso")
except Exception as e:
    print(f"❌ Erro: {e}")

print("\n3. Verificando se feature_names são passados corretamente:")
# Verificar se os feature_names estão sendo incluídos no objeto
if 'explanation' in locals():
    print(f"   - Explanation tem feature_names: {hasattr(explanation, 'feature_names')}")
    if hasattr(explanation, 'feature_names'):
        print(f"   - Feature names: {explanation.feature_names}") 