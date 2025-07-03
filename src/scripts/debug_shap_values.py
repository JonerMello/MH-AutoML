import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import shap

# Criar dados de teste
np.random.seed(42)
X = np.random.randn(100, 5)
y = (X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1 > 0).astype(int)

# Testar RandomForest
print("=== Testando RandomForest ===")
rf = RandomForestClassifier(n_estimators=5, random_state=42)
rf.fit(X, y)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X[0:1])
expected_value = explainer.expected_value

print(f"Tipo de shap_values: {type(shap_values)}")
print(f"Tipo de expected_value: {type(expected_value)}")
if isinstance(shap_values, list):
    print(f"shap_values é lista com {len(shap_values)} elementos")
    for i, sv in enumerate(shap_values):
        print(f"  shap_values[{i}]: shape {sv.shape}, tipo {type(sv)}")
else:
    print(f"shap_values shape: {shap_values.shape}")

if isinstance(expected_value, list):
    print(f"expected_value é lista com {len(expected_value)} elementos")
    for i, ev in enumerate(expected_value):
        print(f"  expected_value[{i}]: {ev}, tipo {type(ev)}")
else:
    print(f"expected_value: {expected_value}")

# Testar DecisionTree
print("\n=== Testando DecisionTree ===")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)

explainer = shap.TreeExplainer(dt)
shap_values = explainer.shap_values(X[0:1])
expected_value = explainer.expected_value

print(f"Tipo de shap_values: {type(shap_values)}")
print(f"Tipo de expected_value: {type(expected_value)}")
if isinstance(shap_values, list):
    print(f"shap_values é lista com {len(shap_values)} elementos")
    for i, sv in enumerate(shap_values):
        print(f"  shap_values[{i}]: shape {sv.shape}, tipo {type(sv)}")
else:
    print(f"shap_values shape: {shap_values.shape}")

if isinstance(expected_value, list):
    print(f"expected_value é lista com {len(expected_value)} elementos")
    for i, ev in enumerate(expected_value):
        print(f"  expected_value[{i}]: {ev}, tipo {type(ev)}")
else:
    print(f"expected_value: {expected_value}")

# Testar chamada direta do shap.plots.force
print("\n=== Testando shap.plots.force diretamente ===")
try:
    if isinstance(shap_values, list):
        shap_plot = shap.plots.force(expected_value[1], shap_values[1])
        print("✅ shap.plots.force(expected_value[1], shap_values[1]) funcionou!")
    else:
        shap_plot = shap.plots.force(expected_value, shap_values)
        print("✅ shap.plots.force(expected_value, shap_values) funcionou!")
except Exception as e:
    print(f"❌ Erro: {e}") 