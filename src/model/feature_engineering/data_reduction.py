import pandas as pd
import numpy as np
import logging
import os
from colorama import Fore, Style
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import f_classif, SelectFpr, SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

class FeatureSelection(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer for feature selection using PCA and ANOVA (Analysis of Variance).

    Parameters:
    - pca (bool, default=False): Whether to apply Principal Component Analysis (PCA) for dimensionality reduction.
    - num_components (int or None, optional): Number of principal components to keep if PCA is enabled.
    - anova (bool, default=False): Whether to apply ANOVA for feature selection.
    - k_features (int or None, optional): Number of top features to select if ANOVA is enabled.
    - balance_classes (bool, default=False): Whether to balance the classes using oversampling (SMOTE) or undersampling.

    Attributes:
    - pca (PCA or None): The PCA instance if PCA is enabled, otherwise None.
    - anova_selector (SelectKBest or None): The SelectKBest instance using ANOVA if anova is enabled, otherwise None.
    """

    def __init__(self, pca=False, num_components=None, anova=False, k_features=None, lasso=False, alpha=None, balance_classes=False):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO) 
        self.num_components = num_components
        self.pca = None
        self.anova = anova
        self.k_features = k_features
        self.lasso = lasso
        self.alpha = alpha
        self.balance_classes=balance_classes
        self.feature_names = None  # Initialize the list of column names as None
        if pca and num_components is not None:
            self.pca = PCA(n_components=num_components)
        if anova and k_features is not None:
            self.anova_selector = SelectFpr(score_func=f_classif, alpha=k_features)
        if lasso:
            lasso_alpha = self.alpha if self.alpha is not None else 0.0001
            self.lasso_selector = SelectFromModel(Lasso(alpha=lasso_alpha))
        # ... (código existente)
        self.applied_method = None  # Novo atributo para rastrear o método aplicado
        self.pca_components_info = None  # Para armazenar informações dos componentes PCA
        self.selected_features_info = None  # Para armazenar informações das features selecionadas

    def fit(self, X, y=None):
        if self.balance_classes =='RUS':
            result = self.balance_data(X, y)
            if isinstance(result, tuple) and len(result) == 2:
                X, y = result
        if self.balance_classes =='SMOTE':  
            result = self.balance_data_SMOTE(X, y)
            if isinstance(result, tuple) and len(result) == 2:
                X, y = result

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            self.original_features = X.columns.tolist()  # Manter cópia dos nomes originais

        if self.pca:
            X = self.pca.fit_transform(X)
            self.applied_method = 'pca'
            n_comp = self.num_components if self.num_components is not None else X.shape[1]
            self.pca_components_info = {
                'explained_variance': self.pca.explained_variance_ratio_,
                'components': self.pca.components_,
                'feature_names': self.feature_names
            }
            self.feature_names = [f'PC_{i+1}' for i in range(n_comp)]
            self.plot_pca_biplot(X, self.pca, self.original_features, y)
            # print(f"Selected features (pca): {self.feature_names}")

        elif self.anova:
            self.anova_selector.fit(X, y)
            self.applied_method = 'anova'
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns[self.anova_selector.get_support()].tolist()
            self.selected_features_info = {
                'method': 'anova',
                'features': self.feature_names,
                'selected_features': self.feature_names,
                'scores': self.anova_selector.scores_
            }
            self.plot_anova_features(self.anova_selector.scores_, 
                                    self.feature_names,
                                    title="ANOVA F-values Feature Importance")
            # print(f"Selected features (anova): {self.feature_names}")
        elif self.lasso:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            lasso_cv = LassoCV(cv=5).fit(X_scaled, y)
            best_alpha = lasso_cv.alpha_
            lasso = Lasso(alpha=best_alpha)
            lasso.fit(X_scaled, y)
            self.feature_names = X.columns[lasso.coef_ != 0].tolist()
            self.applied_method = 'lasso'
            self.selected_features_info = {
                'method': 'lasso',
                'features': self.feature_names,
                'selected_features': self.feature_names,
                'coefficients': lasso.coef_[lasso.coef_ != 0],
                'alpha': best_alpha
            }
            self.lasso_selector = SelectFromModel(lasso)
            self.lasso_selector.fit(X_scaled, y)
            self.plot_lasso_features(lasso.coef_, 
                                   X.columns,
                                   title="LASSO Coefficients Feature Importance")
            # print(f"Selected features (lasso): {self.feature_names}")
        
        return self

    def get_transformation_info(self):
        """Retorna informações detalhadas sobre a transformação aplicada"""
        return {
            'method': self.applied_method,
            'pca_info': self.pca_components_info if self.applied_method == 'pca' else None,
            'selected_features_info': self.selected_features_info if self.applied_method in ['anova', 'lasso'] else None,
            'feature_names': self.feature_names,
            'original_features': self.original_features
        }


    def transform(self, X):
        """
        Transform the input data by applying PCA and/or ANOVA.

        Parameters:
        - X (array-like or pd.DataFrame): Input features.

        Returns:
        - X_transformed (array-like): Transformed features.
        """
        if self.pca:
            X = self.pca.transform(X)
        if self.anova:
            X = self.anova_selector.transform(X)
        if self.lasso:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = self.lasso_selector.transform(X_scaled)
        return X
    
    def get_feature_names(self):
        """
        Returns the column names after transformation.

        Returns:
        - feature_names (list): List of column names after transformation.
        """
        return self.feature_names

    def balance_data_SMOTE(self, X, y):
        # Use numpy to calculate the class counts
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class = unique_classes[np.argmin(class_counts)]
        max_class_count = np.max(class_counts)
        max_class = unique_classes[np.argmax(class_counts)]  # Add max_class
        colored_message_not_feature = f"[{Fore.YELLOW}Classes before balancing: {dict(zip(unique_classes, class_counts))}{Style.RESET_ALL}]"
        self.logger.warning(colored_message_not_feature)


        if min_class != max_class:
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            unique_classes_resampled, class_counts_resampled = np.unique(y_resampled, return_counts=True)
            colored_message_not_feature = f"[{Fore.YELLOW}Classes after balancing SMOTE: {dict(zip(unique_classes_resampled, class_counts_resampled))}{Style.RESET_ALL}]"
            self.logger.warning(colored_message_not_feature)

            return X_resampled, y_resampled
        else:
            return X, y


    def balance_data(self, X, y):
        # Use numpy to calculate the class counts
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class = unique_classes[np.argmin(class_counts)]
        max_class_count = np.max(class_counts)
        max_class = unique_classes[np.argmax(class_counts)]  # Add max_class
        colored_message_not_feature = f"[{Fore.YELLOW}Classes before balancing: {dict(zip(unique_classes, class_counts))}{Style.RESET_ALL}]"
        self.logger.warning(colored_message_not_feature)
      

        if min_class != max_class:
            # Use RandomUnderSampler to perform undersampling
            under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = under_sampler.fit_resample(X, y)
            unique_classes_resampled, class_counts_resampled = np.unique(y_resampled, return_counts=True)
            colored_message_not_feature = f"[{Fore.YELLOW}Classes after balancing RandomUnderSampler: {dict(zip(unique_classes_resampled, class_counts_resampled))}{Style.RESET_ALL}]"
            self.logger.warning(colored_message_not_feature)


            return X_resampled, y_resampled
        else:
            return X, y


    def plot_pca_biplot(self, X_pca, pca, X_columns, y, sample_size=1000, save_path="results"):
        """
        Plot a biplot of PCA showing the data points and the loadings of the original features.

        Parameters:
        - X_pca (array-like): Transformed features after PCA.
        - pca (PCA): The trained PCA instance.
        - X_columns (list): List of column names of the original features.
        - y (array-like): Target variable.
        - sample_size (int): Number of points to sample for plotting.

        Returns:
        - None
        """
        # Amostrar aleatoriamente um subconjunto dos dados para o scatter plot
        sample_indices = np.random.choice(range(len(X_pca)), size=min(sample_size, len(X_pca)), replace=False)
        X_pca_sampled = X_pca[sample_indices]
        y_sampled = y[sample_indices]

        # Biplot
        plt.figure(figsize=(10, 8))
        for i in range(len(pca.components_)):
            # Vetores de carga das características originais
            plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color='r', alpha=0.5, width=0.005)
            plt.text(pca.components_[0, i]*1.15, pca.components_[1, i]*1.15, X_columns[i], color='g')

        # Scatter Plot das duas primeiras componentes principais com coloração de acordo com a classe
        plt.scatter(X_pca_sampled[:, 0], X_pca_sampled[:, 1], c=y_sampled, cmap='viridis', alpha=0.5)

        plt.title('Biplot das duas primeiras componentes principais com vetores de carga (amostra de 1000 pontos)')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.colorbar(label='Classe (0: Benigno, 1: Malware)')
        plt.grid(True)
        # Verificar se o diretório de salvamento existe
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        # Salvar a figura
        save_file_path = os.path.join(save_path, "pca_biplot.png")
        plt.savefig(save_file_path)
        

    def plot_lasso_feature_importance_01(self, selected_features, lasso_coef, save_path="results"):
        plt.bar(selected_features, lasso_coef)
        plt.xticks(rotation=90)
        plt.grid()
        plt.title("Feature Selection Based on Lasso")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        # Definir limites do eixo y com uma margem proporcional
        margin = 0.1 * (max(lasso_coef) - min(lasso_coef))  # 10% da amplitude dos coeficientes
        plt.ylim(min(lasso_coef) - margin, max(lasso_coef) + margin)
        # Salvar o gráfico em um arquivo JPG
        save_file_path_lasso = os.path.join(save_path, "lasso_feature_importance.png")
        plt.savefig(save_file_path_lasso)
        plt.close()  # Fechar o plot para liberar a memória


    def get_significant_features(self, f_statistic, p_value, feature_names, significance_threshold=0.05, save_path="results"):
        """
        Get significant features based on ANOVA results.

        Parameters:
        - f_statistic (array-like): Array of F-statistic values for each feature.
        - p_value (array-like): Array of p-values for each feature.
        - feature_names (list): List of feature names.
        - significance_threshold (float): Threshold for significance.

        Returns:
        - significant_features (list): List of significant feature names.
        """
        # Remove sufixos "_1.0" dos nomes das features
        cleaned_feature_names = [name.replace('_1.0', '') for name in feature_names]
        anova_results = pd.DataFrame({'Feature': cleaned_feature_names,
                                      'F-statistic': f_statistic,
                                      'P-value': p_value})

        significant_features = anova_results[anova_results['P-value'] < significance_threshold]
    
        # Selecionar as 10 principais características com base na estatística F
        top_significant_features = significant_features.nlargest(10, 'F-statistic')
    
        # Plotando as 10 características mais significativas
        plt.figure(figsize=(10, 6))
        plt.barh(top_significant_features['Feature'], top_significant_features['F-statistic'], color='skyblue')
        plt.xlabel('Estatística F')
        plt.title('10 Características Mais Significativas')
        plt.gca().invert_yaxis()  
        plt.tight_layout()  # Ajusta a plotagem para evitar cortar nomes

        # Verificar se o diretório de salvamento existe
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Salvar a figura
        save_file_path = os.path.join(save_path, "anova_significance.png")
        plt.savefig(save_file_path)

        return top_significant_features['Feature'].tolist()

    def plot_anova_features(self, scores, feature_names, title="ANOVA Feature Importance", top_n=20):
        """Plot ANOVA F-values for feature importance."""
        plt.figure(figsize=(10, 6))
    
        # Sort features by score
        indices = np.argsort(scores)[-top_n:]
    
        # Create plot
        plt.barh(range(len(indices)), scores[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('F-value')
        plt.title(title)
        plt.tight_layout()
    
        # Save plot
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/anova_feature_importance.png")
        plt.close()

    def plot_lasso_features(self, coefficients, feature_names, title="LASSO Feature Importance"):
        if coefficients is None or feature_names is None:
            return
        plt.figure(figsize=(10, 6))
        indices = np.argsort(np.abs(coefficients))[::-1]
        nonzero_indices = [i for i in indices if coefficients[i] != 0]
        if len(nonzero_indices) > 0:
            colors = ['red' if coefficients[i] < 0 else 'blue' for i in nonzero_indices]
            plt.barh(range(len(nonzero_indices)), 
                     [coefficients[i] for i in nonzero_indices], 
                     color=colors, 
                     align='center')
            plt.yticks(range(len(nonzero_indices)), 
                       [feature_names[i] for i in nonzero_indices])
            plt.xlabel('Coefficient Value')
            plt.title(title)
            plt.tight_layout()
            plt.axvline(x=0, color='black', linestyle='--')
            os.makedirs("results", exist_ok=True)
            plt.savefig(f"results/lasso_feature_importance.png")
            plt.close()


    def plot_lasso_feature_importance(self, save_path="results"):
        """
        Plot the feature importance based on LASSO coefficients.

        Parameters:
        - save_path (str): Directory to save the plot.

        Returns:
        - None
        """

        lasso_coef = self.lasso_selector.estimator_.coef_
        selected_features = self.feature_names

        plt.figure(figsize=(10, 6))
        plt.bar(selected_features, lasso_coef)
        plt.xticks(rotation=90)
        plt.grid()
        plt.title("Feature Selection Based on LASSO")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        # Define y-axis limits with a proportional margin
        margin = 0.1 * (max(lasso_coef) - min(lasso_coef))  # 10% of the coefficient range
        plt.ylim(min(lasso_coef) - margin, max(lasso_coef) + margin)

        # Save the plot
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file_path = os.path.join(save_path, "lasso_feature_importance.png")
        plt.savefig(save_file_path)
        plt.close()

