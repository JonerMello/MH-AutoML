import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

# Tenta importar LightGBM e CatBoost
try:
    from lightgbm import LGBMClassifier
    has_lgbm = True
except ImportError:
    has_lgbm = False
try:
    from catboost import CatBoostClassifier
    has_catboost = True
except ImportError:
    has_catboost = False

# Adiciona o diretório src ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.interpretability.interpretability import Interpretability

def test_shap_force_plot_for_model(model_class, model_name, **model_kwargs):
    print(f"\n🧪 Testando SHAP force plot para: {model_name}")
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    feature_names = ['age', 'income', 'education', 'credit_score', 'employment_years']
    X_df = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
    model = model_class(**model_kwargs)
    model.fit(X_train, y_train)
    results_folder = f"test_shap_force_plot_{model_name}_results"
    os.makedirs(results_folder, exist_ok=True)
    feature_selection_info = {
        'method': None,
        'feature_names': feature_names,
        'original_features': feature_names,
        'transformer': None,
        'selected_features_info': {}
    }
    interpretability = Interpretability(
        best_model=model,
        X_train_selected=np.array(X_train),
        X_test_selected=np.array(X_test),
        y_train=np.array(y_train),
        feature_selection_info=feature_selection_info,
        results_folder=results_folder
    )
    interpretability.feature_names = feature_names
    try:
        shap_file = interpretability._generate_shap_explanation(datetime.now().strftime("%Y%m%d_%H%M%S"))
        if shap_file and os.path.exists(shap_file):
            print(f"✅ SHAP force plot gerado com sucesso para {model_name}: {shap_file}")
            return True
        else:
            print(f"❌ Falha ao gerar SHAP force plot para {model_name}")
            return False
    except Exception as e:
        print(f"❌ Erro ao gerar SHAP force plot para {model_name}: {e}")
        return False

def main():
    results = {}
    results['RandomForestClassifier'] = test_shap_force_plot_for_model(RandomForestClassifier, 'RandomForestClassifier', n_estimators=10, random_state=42)
    results['DecisionTreeClassifier'] = test_shap_force_plot_for_model(DecisionTreeClassifier, 'DecisionTreeClassifier', random_state=42)
    results['ExtraTreesClassifier'] = test_shap_force_plot_for_model(ExtraTreesClassifier, 'ExtraTreesClassifier', n_estimators=10, random_state=42)
    if has_lgbm:
        results['LGBMClassifier'] = test_shap_force_plot_for_model(LGBMClassifier, 'LGBMClassifier', n_estimators=10, random_state=42)
    else:
        print("⚠️ LightGBM não instalado, pulando LGBMClassifier.")
    if has_catboost:
        results['CatBoostClassifier'] = test_shap_force_plot_for_model(CatBoostClassifier, 'CatBoostClassifier', iterations=10, verbose=0, random_seed=42)
    else:
        print("⚠️ CatBoost não instalado, pulando CatBoostClassifier.")
    print("\nResumo dos resultados:")
    for model, success in results.items():
        print(f"{model}: {'✅' if success else '❌'}")

if __name__ == "__main__":
    main() 