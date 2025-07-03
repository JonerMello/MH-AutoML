import os
import datetime
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import is_classifier
from model.optimization.hyperparameters_methods import Hyperparameters
from typing import Optional, Tuple, Dict, Any
import time
from tqdm import tqdm

class Interpretability:
    def __init__(self, best_model, X_train_selected: np.ndarray, X_test_selected: np.ndarray, 
                 y_train: np.ndarray, feature_selection_info: Dict[str, Any], 
                 results_folder: str):
        """
        Initialize the Interpretability class with model and data information.
        
        Args:
            best_model: The trained model to interpret
            X_train_selected: Transformed training features
            X_test_selected: Transformed test features
            y_train: Training labels
            feature_selection_info: Dictionary with feature selection information from FeatureSelection
            results_folder: Path to save interpretation results
        """
        self.best_model = best_model
        self.X_train_selected = X_train_selected
        self.X_test_selected = X_test_selected
        self.y_train = y_train
        self.feature_selection_info = feature_selection_info
        self.results_folder = results_folder
        
        # Determine applied method and feature names
        self.applied_method = feature_selection_info.get('method')
        self.feature_names = feature_selection_info.get('feature_names', [])
        self.original_features = feature_selection_info.get('original_features', [])
        
        # Create results folder if it doesn't exist
        os.makedirs(results_folder, exist_ok=True)

    def explain_model(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Generate model explanations based on the feature selection method used.
        
        Returns:
            Tuple containing paths to SHAP explanation, LIME explanation HTML, and LIME importance plot
        """
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Loader animado para interpretação do modelo
        print("🔄 Interpreting model...")
        with tqdm(total=100, desc="Model Interpretation", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            # Simular progresso da interpretação
            pbar.update(20)  # Início da interpretação
            time.sleep(0.1)
            pbar.update(30)  # SHAP analysis
            time.sleep(0.1)
            pbar.update(30)  # LIME analysis
            time.sleep(0.1)
            pbar.update(20)  # Finalização
        print("✅ Model interpretation completed!")
        
        if self.applied_method == 'pca':
            return self._explain_pca_model(current_datetime)
        elif self.applied_method in ['anova', 'lasso']:
            return self._explain_feature_selected_model(current_datetime)
        else:
            return self._explain_original_features(current_datetime)

    def _explain_pca_model(self, timestamp: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Generate explanations for PCA-transformed models.
    
        Args:
            timestamp: Current timestamp for file naming
        
        Returns:
            Tuple of file paths for SHAP, LIME HTML, and LIME plot
        """
        try:
            # 1. First explain the PCA components themselves
            pca_explanation = self._explain_pca_components()
            pca_plot_path = os.path.join(self.results_folder, f'pca_components_{timestamp}.png')
            pca_explanation['plot'].savefig(pca_plot_path)
            plt.close()
        except Exception as e:
            print(f"Could not generate PCA component explanation: {e}")
            pca_plot_path = None
    
        # 2. Then explain the model on PCA components
        shap_file = self._generate_shap_explanation(timestamp, is_pca=True)
        lime_html, lime_plot = self._generate_lime_explanation(timestamp, is_pca=True)
    
        return shap_file, lime_html, lime_plot

    def _explain_pca_components(self) -> Dict[str, Any]:
        """
        Explain the meaning of PCA components.
    
        Returns:
            Dictionary with component explanation data and plot
        """
        if 'pca_info' not in self.feature_selection_info:
            raise ValueError("PCA info not available in feature_selection_info")
        
        pca_info = self.feature_selection_info['pca_info']
        components = pca_info['components']
        original_features = pca_info['feature_names']
        pca_features = self.feature_names
    
        # Create component importance DataFrame
        components_df = pd.DataFrame(
            components,
            columns=original_features,
            index=pca_features
        )
    
        # Plot component heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(components_df, cmap='coolwarm', center=0, annot=True, fmt=".2f")
        plt.title('Feature Contributions to PCA Components')
        plt.tight_layout()
    
        return {
            'components_df': components_df,
            'explained_variance': pca_info['explained_variance'],
            'plot': plt
        }

    def _explain_feature_selected_model(self, timestamp: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Generate explanations for models with feature selection (ANOVA/LASSO).
        
        Args:
            timestamp: Current timestamp for file naming
            
        Returns:
            Tuple of file paths for SHAP, LIME HTML, and LIME plot
        """
        # Generate feature importance plot for selected features
        self._plot_feature_selection_importance(timestamp)
        
        # Generate standard explanations
        shap_file = self._generate_shap_explanation(timestamp)
        lime_html, lime_plot = self._generate_lime_explanation(timestamp)
        
        return shap_file, lime_html, lime_plot

    def _explain_original_features(self, timestamp: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Generate explanations for models without feature transformation.
        
        Args:
            timestamp: Current timestamp for file naming
            
        Returns:
            Tuple of file paths for SHAP, LIME HTML, and LIME plot
        """
        shap_file = self._generate_shap_explanation(timestamp)
        lime_html, lime_plot = self._generate_lime_explanation(timestamp)
        return shap_file, lime_html, lime_plot

    def _generate_shap_explanation(self, timestamp: str, is_pca: bool = False) -> Optional[str]:
        """
        Generate SHAP explanations for the model.
    
        Args:
            timestamp: Current timestamp for file naming
            is_pca: Whether explaining PCA-transformed data
        
        Returns:
            Path to saved SHAP explanation file
        """
        try:
            print(f"🔍 Starting SHAP explanation generation...")
            sample_idx = 0  # Explain first sample
            estimators = []
            if hasattr(self.best_model, 'named_estimators_'):
                # VotingClassifier: pode ter múltiplos estimadores
                estimators = list(self.best_model.named_estimators_.values())
                print(f"📊 Found {len(estimators)} estimators in VotingClassifier")
            elif hasattr(self.best_model, 'estimators_'):
                # Modelos ensemble simples (RandomForest, ExtraTrees, etc.)
                estimators = self.best_model.estimators_
                print(f"📊 Found {len(estimators)} estimators in ensemble model")
            else:
                estimators = [self.best_model]
                print(f"📊 Using single estimator: {type(self.best_model).__name__}")
            
            shap_files = []
            for i, estimator in enumerate(estimators):
                print(f"🔧 Processing estimator {i+1}/{len(estimators)}: {type(estimator).__name__}")
                
                # Select appropriate explainer
                if isinstance(estimator, (RandomForestClassifier, ExtraTreesClassifier, 
                                        DecisionTreeClassifier, LGBMClassifier, CatBoostClassifier)):
                    print(f"🌳 Using TreeExplainer for {type(estimator).__name__}")
                    explainer = shap.TreeExplainer(estimator, feature_perturbation="interventional")
                else:
                    print(f"🔧 Using KernelExplainer for {type(estimator).__name__}")
                    background = shap.sample(self.X_train_selected, 100)
                    explainer = shap.KernelExplainer(estimator.predict_proba, background)
                
                print(f"📈 Calculating SHAP values...")
                # Calculate SHAP values for both plots
                shap_values_single = explainer.shap_values(self.X_test_selected[sample_idx:sample_idx+1])
                shap_values_all = explainer.shap_values(self.X_test_selected)
                
                # Handle multi-class case
                if isinstance(shap_values_single, list):
                    shap_values_single = shap_values_single[1]  # Use values for positive class
                    shap_values_all = shap_values_all[1]
                    print(f"📊 Multi-class case detected, using positive class values")
                
                expected_value = (explainer.expected_value[0] 
                                if isinstance(explainer.expected_value, list) 
                                else explainer.expected_value)
                
                # Se for PCA, renomear os componentes para incluir as top features originais
                feature_names = self.feature_names
                if is_pca and 'selected_features_info' in self.feature_selection_info and self.feature_selection_info['selected_features_info'] and 'top_features' in self.feature_selection_info['selected_features_info']:
                    top_features = self.feature_selection_info['selected_features_info']['top_features']
                    if isinstance(top_features, dict):
                        feature_names = [f"PC{i+1} ({', '.join(top_features.get(f'PC_{i+1}', []))})" for i in range(len(self.feature_names))]
                    elif isinstance(top_features, list):
                        feature_names = [str(f) for f in top_features]
                    else:
                        feature_names = self.feature_names
                
                print(f"🎨 Generating SHAP force plot...")
                try:
                    # Handle different formats of shap_values and expected_value
                    if isinstance(shap_values_single, list):
                        # List format (some models return list of arrays)
                        shap_values_to_use = shap_values_single[1] if len(shap_values_single) > 1 else shap_values_single[0]
                        expected_value_to_use = explainer.expected_value[1] if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > 1 else explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
                    else:
                        # Array format (most models return numpy array)
                        if shap_values_single.ndim == 3:
                            # Shape (1, n_features, n_classes) - use positive class (index 1)
                            shap_values_to_use = shap_values_single[..., 1]
                            expected_value_to_use = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
                        else:
                            # Single class or other format
                            shap_values_to_use = shap_values_single
                            expected_value_to_use = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else explainer.expected_value
                    
                    # Generate force plot with correct parameters and feature names
                    # Use direct method with feature_names parameter
                    shap_plot = shap.plots.force(expected_value_to_use, shap_values_to_use, feature_names=feature_names)
                    
                except Exception as force_error:
                    print(f"⚠️ Error with force plot: {force_error}")
                    # Fallback: try with class 0
                    try:
                        if isinstance(shap_values_single, list):
                            shap_values_to_use = shap_values_single[0]
                            expected_value_to_use = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
                        else:
                            if shap_values_single.ndim == 3:
                                shap_values_to_use = shap_values_single[..., 0]
                                expected_value_to_use = explainer.expected_value[0]
                            else:
                                shap_values_to_use = shap_values_single
                                expected_value_to_use = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else explainer.expected_value
                        
                        # Use direct method with feature_names parameter for fallback
                        shap_plot = shap.plots.force(expected_value_to_use, shap_values_to_use, feature_names=feature_names)
                    except Exception as force_error2:
                        print(f"❌ SHAP force plot failed again: {force_error2}")
                        shap_plot = None
                if shap_plot is not None:
                    shap_html_path = os.path.join(self.results_folder, f'shap_force_plot_{type(estimator).__name__}_{timestamp}.html')
                    shap.save_html(shap_html_path, shap_plot)
                    print(f"💾 SHAP HTML saved: {shap_html_path}")
                    shap_files.append(shap_html_path)
                # Gera também um summary_plot (bar) mais limpo
                try:
                    shap_summary_path = os.path.join(self.results_folder, f'shap_summary_plot_{type(estimator).__name__}_{timestamp}.png')
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 8))
                    # Seguindo o padrão da documentação SHAP com feature_names
                    shap.summary_plot(shap_values_all, self.X_test_selected, feature_names=feature_names, max_display=10, show=False)
                    plt.tight_layout()
                    plt.savefig(shap_summary_path, dpi=200, bbox_inches='tight')
                    plt.close()
                    print(f"📊 SHAP summary plot saved: {shap_summary_path}")
                except Exception as summary_error:
                    print(f"⚠️ Could not generate SHAP summary plot: {summary_error}")
                
                # Generate PNG version of SHAP force plot
                try:
                    print(f"🖼️ Generating SHAP PNG...")
                    shap_png_path = os.path.join(self.results_folder, f'shap_force_plot_{type(estimator).__name__}_{timestamp}.png')
                    
                    # Use matplotlib to create a custom force plot visualization
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as patches
                    
                    # Create a horizontal bar plot similar to force plot using SHAP values
                    plt.figure(figsize=(12, max(6, len(feature_names) * 0.4)))
                    
                    # Get SHAP values for the first sample
                    if isinstance(shap_values_single, list):
                        # Use the same logic as the force plot
                        shap_values_sample = shap_values_single[1].flatten() if len(shap_values_single) > 1 else shap_values_single[0].flatten()
                    else:
                        if shap_values_single.ndim == 3:
                            # Shape (1, n_features, n_classes) - use positive class (index 1)
                            shap_values_sample = shap_values_single[..., 1].flatten()
                        else:
                            shap_values_sample = shap_values_single.flatten()
                    
                    # Ensure feature_names are strings and handle numpy arrays
                    clean_feature_names = []
                    for name in feature_names:
                        if isinstance(name, np.ndarray):
                            clean_feature_names.append(str(name.item()) if name.size == 1 else str(name))
                        else:
                            clean_feature_names.append(str(name))
                    
                    # Sort features by absolute SHAP value
                    feature_importance = list(zip(clean_feature_names, shap_values_sample))
                    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    # Take top features
                    top_features = feature_importance[:min(15, len(feature_importance))]
                    features = [item[0] for item in top_features]
                    weights = [float(item[1]) for item in top_features]  # Ensure weights are float
                    
                    # Create horizontal bar plot with colors based on weight
                    colors = ['red' if w < 0 else 'blue' for w in weights]
                    bars = plt.barh(range(len(features)), weights, color=colors, alpha=0.7)
                    
                    # Add feature names as y-axis labels
                    plt.yticks(range(len(features)), features)
                    
                    # Add value labels on bars
                    for i, (bar, weight) in enumerate(zip(bars, weights)):
                        plt.text(weight + (0.01 if weight >= 0 else -0.01), 
                                bar.get_y() + bar.get_height()/2, 
                                f'{weight:.3f}', 
                                va='center', 
                                ha='left' if weight >= 0 else 'right',
                                fontsize=9)
                    
                    plt.title(f'SHAP Force Plot - {type(estimator).__name__}', fontsize=14, fontweight='bold')
                    plt.xlabel('SHAP Value (Red: Negative, Blue: Positive)', fontsize=12)
                    plt.ylabel('Features', fontsize=12)
                    plt.grid(axis='x', alpha=0.3)
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                    
                    # Add model info annotation
                    plt.text(0.02, 0.98, f'Model: {type(estimator).__name__}', 
                            transform=plt.gca().transAxes, 
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    plt.tight_layout()
                    plt.savefig(shap_png_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"💾 SHAP PNG saved: {shap_png_path}")
                        
                except Exception as png_error:
                    print(f"❌ Error generating SHAP PNG: {png_error}")
                    # Final fallback: create a simple text-based visualization
                    try:
                        plt.figure(figsize=(10, 6))
                        plt.text(0.5, 0.5, f'SHAP Force Plot\n{type(estimator).__name__}\n\nHTML version available for interactive view', 
                                ha='center', va='center', fontsize=14, 
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                        plt.axis('off')
                        plt.savefig(shap_png_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"💾 SHAP PNG (fallback) saved: {shap_png_path}")
                    except Exception as fallback_error:
                        print(f"❌ Final fallback failed: {fallback_error}")
                
                # Se o estimador for um modelo de árvore, plote a árvore (primeira árvore para ensembles)
                tree_models = (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, LGBMClassifier, CatBoostClassifier)
                if isinstance(estimator, tree_models):
                    from sklearn import tree
                    import matplotlib.pyplot as plt
                    tree_plot_path = os.path.join(self.results_folder, f'decision_tree_plot_{type(estimator).__name__}_{timestamp}.png')
                    
                    if isinstance(estimator, (RandomForestClassifier, ExtraTreesClassifier)):
                        # Para ensembles, plote a primeira árvore
                        base_tree = estimator.estimators_[0]
                        plt.figure(figsize=(20, 10))
                        tree.plot_tree(base_tree, filled=True, feature_names=feature_names, class_names=True, rounded=True, fontsize=10)
                        plt.title(f"{type(estimator).__name__} - First Tree")
                        plt.savefig(tree_plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"🌳 Decision tree plot saved: {tree_plot_path}")
                    elif isinstance(estimator, LGBMClassifier):
                        # Para LightGBM, use plot_tree do próprio pacote
                        try:
                            import lightgbm as lgb
                            # Usar a API correta do LightGBM para plotar árvores
                            ax = lgb.plot_tree(estimator, tree_index=0, figsize=(20, 10), 
                                             show_info=["split_gain", "internal_value", "internal_count", "leaf_count"],
                                             dpi=300)
                            plt.title(f"LGBMClassifier - First Tree")
                            plt.savefig(tree_plot_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            print(f"🌳 Decision tree plot saved: {tree_plot_path}")
                        except Exception as lgb_error:
                            print(f"⚠️ Could not plot LightGBM tree (Graphviz required): {lgb_error}")
                            # Fallback: gerar um gráfico de importância de features em vez da árvore
                            try:
                                plt.figure(figsize=(12, 8))
                                lgb.plot_importance(estimator, max_num_features=20, figsize=(12, 8))
                                plt.title(f"LGBMClassifier - Feature Importance")
                                plt.savefig(tree_plot_path, dpi=300, bbox_inches='tight')
                                plt.close()
                                print(f"📊 Feature importance plot saved (fallback): {tree_plot_path}")
                            except Exception as fallback_error:
                                print(f"❌ LightGBM plotting failed completely: {fallback_error}")
                    elif isinstance(estimator, CatBoostClassifier):
                        # Para CatBoost, use o método plot_tree do próprio pacote
                        try:
                            from catboost import CatBoost
                            plt.figure(figsize=(20, 10))
                            estimator.plot_tree(tree_idx=0, pool=None)
                            plt.title(f"CatBoostClassifier - First Tree")
                            plt.savefig(tree_plot_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            print(f"🌳 Decision tree plot saved: {tree_plot_path}")
                        except Exception as cb_error:
                            print(f"⚠️ Could not plot CatBoost tree: {cb_error}")
                    else:
                        # DecisionTreeClassifier
                        plt.figure(figsize=(20, 10))
                        tree.plot_tree(estimator, filled=True, feature_names=feature_names, class_names=True, rounded=True, fontsize=10)
                        plt.title(f"Decision Tree - {type(estimator).__name__}")
                        plt.savefig(tree_plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"🌳 Decision tree plot saved: {tree_plot_path}")
                
            # Retorna o primeiro arquivo (ou todos, se desejar)
            result = shap_files[0] if shap_files else None
            print(f"✅ SHAP generation completed. Result: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Error generating SHAP explanations: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _plot_feature_selection_importance(self, timestamp: str) -> Optional[str]:
        """
        Generate feature importance plot for selected features.
    
        Args:
            timestamp: Current timestamp for file naming
        
        Returns:
            Path to saved plot file
        """
        if self.applied_method not in ['anova', 'lasso']:
            return None
        
        try:
            if 'selected_features_info' not in self.feature_selection_info:
                print("Feature selection info not available")
                return None
            
            plt.figure(figsize=(12, 8))
            info = self.feature_selection_info['selected_features_info']
        
            if self.applied_method == 'anova' and 'scores' in info:
                plt.barh(info['features'], info['scores'])
                plt.title('ANOVA F-scores for Selected Features')
            elif self.applied_method == 'lasso' and 'coefficients' in info:
                plt.barh(info['features'], info['coefficients'])
                plt.title('LASSO Coefficients for Selected Features')
            else:
                #print(f"Missing data for {self.applied_method} feature importance plot")
                return None
            
            plt.xlabel('Importance Score')
            plt.tight_layout()
        
            plot_filename = f'{self.applied_method}_feature_importance_{timestamp}.png'
            plot_filepath = os.path.join(self.results_folder, plot_filename)
            plt.savefig(plot_filepath)
            plt.close()
        
            return plot_filepath
        
        except Exception as e:
            print(f"Error generating feature importance plot: {e}")
            return None

    @staticmethod
    def explanation(best_trial, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, feature_names: list) -> np.ndarray:
        """
        Static method to generate SHAP explanations for a model from a trial.
        """
        final_model = Hyperparameters(X_train, y_train).get_models_params(best_trial)
        final_model.fit(X_train, y_train)
        feature_names_clean = [str(name).replace('_1.0', '') for name in feature_names]
        if hasattr(final_model, 'named_estimators_'):
            estimator = next(iter(final_model.named_estimators_.values()))
        else:
            estimator = final_model
        if isinstance(estimator, (KNeighborsClassifier)):
            explainer = shap.KernelExplainer(
                estimator.predict, 
                shap.sample(X_train, 100), 
                feature_names=feature_names_clean
            )
        else:
            explainer = shap.TreeExplainer(
                estimator
            )
        shap_values = explainer.shap_values(X_test)
        # Garantir que o retorno é np.ndarray
        if isinstance(shap_values, list):
            shap_values = np.asarray(shap_values[1])
        else:
            shap_values = np.asarray(shap_values)
        # Generate summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feature_names_clean, show=False)
        return shap_values

    def _generate_lime_explanation(self, timestamp: str, is_pca: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate LIME explanations for the model.
    
        Args:
            timestamp: Current timestamp for file naming
            is_pca: Whether explaining PCA-transformed data
        
        Returns:
            Tuple containing paths to LIME HTML explanation and feature importance plot
        """
         
        try:
            # Se for PCA, renomear os componentes para incluir as top features originais
            if is_pca and 'selected_features_info' in self.feature_selection_info and 'top_features' in self.feature_selection_info['selected_features_info']:
                top_features = self.feature_selection_info['selected_features_info']['top_features']
                feature_names = [
                    f"PC{i+1} (" + ', '.join([name for name, _ in features]) + ")"
                    for i, features in enumerate(top_features)
                ]
            else:
                feature_names = self.feature_names
            cleaned_feature_names = [str(name).replace('_1.0', '') for name in feature_names]
        
            # Create LIME explainer
            explainer = LimeTabularExplainer(
                self.X_train_selected,
                mode="classification",
                training_labels=self.y_train,
                feature_names=cleaned_feature_names,
                verbose=False
            )
        
            # Explain first test sample
            explanation = explainer.explain_instance(
                self.X_test_selected[0],
                self.best_model.predict_proba,
                num_features=min(20, self.X_train_selected.shape[1])
            )
        
            # Save HTML explanation
            lime_html_filename = f'lime_interpretability_{timestamp}.html'
            lime_html_path = os.path.join(self.results_folder, lime_html_filename)
            explanation.save_to_file(lime_html_path)
            
            # Generate comprehensive LIME visualization as PNG
            try:
                lime_png_filename = f'lime_interpretability_{timestamp}.png'
                lime_png_path = os.path.join(self.results_folder, lime_png_filename)
                
                # Create subplots for different aspects of LIME explanation
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
                
                # Top subplot: Feature importance
                explanation.as_pyplot_figure()
                ax1.set_title('LIME Feature Importance')
                ax1.set_xlabel('Feature')
                ax1.set_ylabel('Weight')
                
                # Bottom subplot: Prediction probabilities
                probs = explanation.predict_proba
                classes = explanation.class_names if hasattr(explanation, 'class_names') else ['Class 0', 'Class 1']
                ax2.bar(classes, probs)
                ax2.set_title('Prediction Probabilities')
                ax2.set_ylabel('Probability')
                ax2.set_ylim(0, 1)
                
                plt.tight_layout()
                plt.savefig(lime_png_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as png_error:
                print(f"Error generating LIME PNG: {png_error}")
        
            # Save detailed feature importance plot with better formatting
            lime_plot_filename = f'lime_feature_importance_{timestamp}.png'
            lime_plot_path = os.path.join(self.results_folder, lime_plot_filename)
        
            try:
                # Get feature importance data
                feature_importance = explanation.as_list()
                
                # Create a more detailed feature importance plot
                plt.figure(figsize=(12, min(10, len(feature_importance) * 0.4)))
                
                # Extract feature names and weights
                features = [item[0] for item in feature_importance]
                weights = [item[1] for item in feature_importance]
                
                # Create horizontal bar plot with colors based on weight
                colors = ['red' if w < 0 else 'blue' for w in weights]
                bars = plt.barh(range(len(features)), weights, color=colors, alpha=0.7)
                
                # Add feature names as y-axis labels
                plt.yticks(range(len(features)), features)
                
                # Add value labels on bars
                for i, (bar, weight) in enumerate(zip(bars, weights)):
                    plt.text(weight + (0.01 if weight >= 0 else -0.01), 
                            bar.get_y() + bar.get_height()/2, 
                            f'{weight:.3f}', 
                            va='center', 
                            ha='left' if weight >= 0 else 'right',
                            fontsize=9)
                
                plt.title('LIME Feature Importance Analysis', fontsize=14, fontweight='bold')
                plt.xlabel('Feature Weight (Red: Negative, Blue: Positive)', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.grid(axis='x', alpha=0.3)
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                
                plt.tight_layout()
                plt.savefig(lime_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as plot_error:
                print(f"Error generating detailed LIME feature importance plot: {plot_error}")
                # Fallback to simple plot
                plt.figure(figsize=(12, min(10, len(explanation.as_list()) * 0.5)))
                explanation.as_pyplot_figure()
                plt.tight_layout()
                plt.savefig(lime_plot_path, dpi=300)
                plt.close()
        
            return lime_html_path, lime_plot_path
        
        except Exception as e:
            print(f"Error generating LIME explanations: {e}")
            return None, None

    def plot_global_feature_importance(self) -> Optional[str]:
        """
        Generate global feature importance plot for tree-based models
    
        Returns:
            Path to the saved image file if successful, None otherwise
        """
        try:
            if not hasattr(self.best_model, 'feature_importances_'):
                print("Model does not have feature_importances_ attribute")
                return None
            
            importances = self.best_model.feature_importances_
            if len(importances) == 0:
                print("No feature importances available")
                return None
            
            indices = np.argsort(importances)[::-1]
        
            plt.figure(figsize=(12, 8))
            plt.title("Global Feature Importance")
            plt.bar(range(len(importances)), 
                    importances[indices],
                    align="center")
        
            # Use feature names if available, otherwise use indices
            if hasattr(self, 'feature_names') and len(self.feature_names) == len(importances):
                plt.xticks(list(range(len(importances))),
                           [self.feature_names[i] for i in indices],
                           rotation=90)
            else:
                plt.xticks(list(range(len(importances))),
                           [str(i) for i in indices],
                           rotation=90)
        
            plt.tight_layout()
        
            # Ensure directory exists
            os.makedirs(self.results_folder, exist_ok=True)
        
            # Save plot
            plot_path = os.path.join(self.results_folder, 'global_feature_importance.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
        
            print(f"Feature importance plot saved to: {plot_path}")
            return plot_path
        
        except Exception as e:
            print(f"Error generating feature importance plot: {str(e)}")
            return None

    def partial_dependence_plots(self, features=None, n_samples=500):
        """Generate partial dependence plots for selected features"""
        from sklearn.inspection import PartialDependenceDisplay
    
        if features is None:
            features = self.feature_names[:5]  # Default to first 5 features
    
        fig, ax = plt.subplots(figsize=(12, 8))
        PartialDependenceDisplay.from_estimator(
            self.best_model,
            self.X_train_selected[:n_samples],
            features=features,
            feature_names=self.feature_names,
            ax=ax
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, 'partial_dependence_plots.png'))
        plt.close()

