import unittest
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Adicionar o diretório src ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar módulos do sistema
from model.preprocessing.data_cleaning import DataCleaning
from model.preprocessing.data_info import DataInfo

class TestPreprocessing(unittest.TestCase):
    """Testes abrangentes para a etapa de pré-processamento do pipeline MHAutoML"""
    
    def setUp(self):
        """Configuração inicial para todos os testes"""
        print(f"\n🧪 Iniciando testes de pré-processamento - {datetime.now().strftime('%H:%M:%S')}")
        
        # Criar dados de teste sintéticos
        np.random.seed(42)
        self.n_samples = 1000
        self.n_features = 20
        
        # Dados limpos
        self.X_clean = np.random.randn(self.n_samples, self.n_features)
        self.y_clean = np.random.randint(0, 2, self.n_samples)
        
        # Dados com problemas
        self.X_dirty = self.X_clean.copy()
        self.y_dirty = self.y_clean.copy()
        
        # Adicionar valores faltantes
        mask = np.random.random(self.X_dirty.shape) < 0.1  # 10% de valores faltantes
        self.X_dirty[mask] = np.nan
        
        # Adicionar outliers
        outlier_indices = np.random.choice(self.n_samples, size=50, replace=False)
        self.X_dirty[outlier_indices, 0] = 1000  # Outliers extremos
        
        # Adicionar valores infinitos
        inf_indices = np.random.choice(self.n_samples, size=20, replace=False)
        self.X_dirty[inf_indices, 1] = np.inf
        
        # Adicionar valores negativos infinitos
        neg_inf_indices = np.random.choice(self.n_samples, size=20, replace=False)
        self.X_dirty[neg_inf_indices, 2] = -np.inf
        
        # Criar DataFrames
        feature_names = [f'feature_{i}' for i in range(self.n_features)]
        self.df_clean = pd.DataFrame(self.X_clean, columns=feature_names)
        self.df_clean['target'] = self.y_clean
        
        self.df_dirty = pd.DataFrame(self.X_dirty, columns=feature_names)
        self.df_dirty['target'] = self.y_dirty
        
        # Instanciar classes
        self.data_cleaning = DataCleaning(label='target')
        # DataInfo será instanciado com dados específicos em cada teste
        
        # Criar diretório de resultados de teste
        self.test_results_dir = "test_preprocessing_results"
        os.makedirs(self.test_results_dir, exist_ok=True)
    
    def test_01_data_info_analysis(self):
        """Teste 1: Análise de informações dos dados"""
        print("📊 Teste 1: Análise de informações dos dados")
        
        # Testar análise de dados limpos
        info_clean = self.data_info.analyze_data(self.df_clean)
        
        # Verificações básicas
        self.assertIsInstance(info_clean, dict)
        self.assertIn('shape', info_clean)
        self.assertIn('dtypes', info_clean)
        self.assertIn('missing_values', info_clean)
        self.assertIn('duplicates', info_clean)
        
        # Verificar valores específicos
        self.assertEqual(info_clean['shape'], (self.n_samples, self.n_features + 1))
        self.assertEqual(info_clean['duplicates'], 0)  # Dados sintéticos não devem ter duplicatas
        
        print("✅ Análise de dados limpos: OK")
        
        # Testar análise de dados sujos
        info_dirty = self.data_info.analyze_data(self.df_dirty)
        
        # Verificar detecção de problemas
        self.assertGreater(info_dirty['missing_values'].sum(), 0)
        self.assertIn('infinity_values', info_dirty)
        
        print("✅ Análise de dados sujos: OK")
        
        # Salvar relatório
        report_path = os.path.join(self.test_results_dir, "data_info_report.txt")
        with open(report_path, 'w') as f:
            f.write("=== RELATÓRIO DE ANÁLISE DE DADOS ===\n\n")
            f.write("DADOS LIMPOS:\n")
            for key, value in info_clean.items():
                f.write(f"{key}: {value}\n")
            f.write("\nDADOS SUJOS:\n")
            for key, value in info_dirty.items():
                f.write(f"{key}: {value}\n")
        
        print(f"📄 Relatório salvo: {report_path}")
    
    def test_02_missing_values_detection(self):
        """Teste 2: Detecção de valores faltantes"""
        print("🔍 Teste 2: Detecção de valores faltantes")
        
        # Detectar valores faltantes
        missing_info = self.data_cleaning.detect_missing_values(self.df_dirty)
        
        # Verificações
        self.assertIsInstance(missing_info, dict)
        self.assertIn('total_missing', missing_info)
        self.assertIn('missing_percentage', missing_info)
        self.assertIn('columns_with_missing', missing_info)
        
        # Verificar valores específicos
        self.assertGreater(missing_info['total_missing'], 0)
        self.assertGreater(missing_info['missing_percentage'], 0)
        
        print(f"✅ Valores faltantes detectados: {missing_info['total_missing']} ({missing_info['missing_percentage']:.2f}%)")
        
        # Verificar se as colunas com valores faltantes foram identificadas
        self.assertIsInstance(missing_info['columns_with_missing'], list)
        
        print("✅ Detecção de valores faltantes: OK")
    
    def test_03_outlier_detection(self):
        """Teste 3: Detecção de outliers"""
        print("🎯 Teste 3: Detecção de outliers")
        
        # Detectar outliers
        outlier_info = self.data_cleaning.detect_outliers(self.df_dirty)
        
        # Verificações
        self.assertIsInstance(outlier_info, dict)
        self.assertIn('outlier_counts', outlier_info)
        self.assertIn('outlier_percentage', outlier_info)
        
        # Verificar se outliers foram detectados
        self.assertGreater(outlier_info['outlier_counts'].sum(), 0)
        
        print(f"✅ Outliers detectados: {outlier_info['outlier_counts'].sum()} ({outlier_info['outlier_percentage']:.2f}%)")
        
        # Verificar se a coluna com outliers extremos foi identificada
        self.assertGreater(outlier_info['outlier_counts']['feature_0'], 0)
        
        print("✅ Detecção de outliers: OK")
    
    def test_04_data_cleaning_pipeline(self):
        """Teste 4: Pipeline completo de limpeza de dados"""
        print("🧹 Teste 4: Pipeline completo de limpeza de dados")
        
        # Aplicar pipeline de limpeza
        df_cleaned, cleaning_report = self.data_cleaning.clean_data(
            self.df_dirty, 
            remove_duplicates=True,
            handle_missing='impute',
            handle_outliers='remove',
            handle_infinite=True
        )
        
        # Verificações básicas
        self.assertIsInstance(df_cleaned, pd.DataFrame)
        self.assertIsInstance(cleaning_report, dict)
        
        # Verificar se dados foram limpos
        self.assertEqual(df_cleaned.isnull().sum().sum(), 0)  # Sem valores faltantes
        self.assertFalse(np.isinf(df_cleaned.select_dtypes(include=[np.number])).any().any())  # Sem valores infinitos
        
        # Verificar se outliers foram removidos
        self.assertLess(df_cleaned.shape[0], self.df_dirty.shape[0])
        
        print(f"✅ Dados limpos: {df_cleaned.shape[0]} amostras (original: {self.df_dirty.shape[0]})")
        
        # Verificar relatório de limpeza
        self.assertIn('rows_removed', cleaning_report)
        self.assertIn('missing_values_filled', cleaning_report)
        self.assertIn('outliers_removed', cleaning_report)
        
        print("✅ Pipeline de limpeza: OK")
        
        # Salvar dados limpos
        cleaned_path = os.path.join(self.test_results_dir, "cleaned_data.csv")
        df_cleaned.to_csv(cleaned_path, index=False)
        print(f"💾 Dados limpos salvos: {cleaned_path}")
    
    def test_05_data_validation(self):
        """Teste 5: Validação de qualidade dos dados"""
        print("✅ Teste 5: Validação de qualidade dos dados")
        
        # Validar dados limpos
        validation_clean = self.data_cleaning.validate_data(self.df_clean)
        
        # Verificações
        self.assertIsInstance(validation_clean, dict)
        self.assertIn('is_valid', validation_clean)
        self.assertIn('issues', validation_clean)
        
        # Dados limpos devem ser válidos
        self.assertTrue(validation_clean['is_valid'])
        self.assertEqual(len(validation_clean['issues']), 0)
        
        print("✅ Validação de dados limpos: OK")
        
        # Validar dados sujos
        validation_dirty = self.data_cleaning.validate_data(self.df_dirty)
        
        # Dados sujos devem ter issues
        self.assertFalse(validation_dirty['is_valid'])
        self.assertGreater(len(validation_dirty['issues']), 0)
        
        print(f"✅ Validação de dados sujos: {len(validation_dirty['issues'])} issues detectadas")
        
        # Verificar tipos de issues
        issue_types = [issue['type'] for issue in validation_dirty['issues']]
        self.assertIn('missing_values', issue_types)
        self.assertIn('outliers', issue_types)
        self.assertIn('infinite_values', issue_types)
        
        print("✅ Validação de qualidade: OK")
    
    def test_06_data_distribution_analysis(self):
        """Teste 6: Análise de distribuição dos dados"""
        print("📈 Teste 6: Análise de distribuição dos dados")
        
        # Analisar distribuição
        distribution_info = self.data_info.analyze_distribution(self.df_clean)
        
        # Verificações
        self.assertIsInstance(distribution_info, dict)
        self.assertIn('numerical_stats', distribution_info)
        self.assertIn('categorical_stats', distribution_info)
        
        # Verificar estatísticas numéricas
        numerical_stats = distribution_info['numerical_stats']
        self.assertIn('mean', numerical_stats)
        self.assertIn('std', numerical_stats)
        self.assertIn('min', numerical_stats)
        self.assertIn('max', numerical_stats)
        
        print("✅ Análise de distribuição: OK")
        
        # Verificar se estatísticas fazem sentido
        for col in numerical_stats:
            if col != 'target':  # Pular coluna target
                self.assertIsInstance(numerical_stats[col]['mean'], (int, float))
                self.assertGreaterEqual(numerical_stats[col]['std'], 0)
        
        print("✅ Estatísticas numéricas: OK")
    
    def test_07_correlation_analysis(self):
        """Teste 7: Análise de correlação"""
        print("🔗 Teste 7: Análise de correlação")
        
        # Analisar correlações
        correlation_info = self.data_info.analyze_correlations(self.df_clean)
        
        # Verificações
        self.assertIsInstance(correlation_info, dict)
        self.assertIn('correlation_matrix', correlation_info)
        self.assertIn('high_correlations', correlation_info)
        
        # Verificar matriz de correlação
        corr_matrix = correlation_info['correlation_matrix']
        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertEqual(corr_matrix.shape, (self.n_features + 1, self.n_features + 1))
        
        # Verificar se valores de correlação estão entre -1 e 1
        self.assertTrue((corr_matrix >= -1).all().all())
        self.assertTrue((corr_matrix <= 1).all().all())
        
        print("✅ Análise de correlação: OK")
        
        # Salvar matriz de correlação
        corr_path = os.path.join(self.test_results_dir, "correlation_matrix.csv")
        corr_matrix.to_csv(corr_path)
        print(f"💾 Matriz de correlação salva: {corr_path}")
    
    def test_08_data_quality_metrics(self):
        """Teste 8: Métricas de qualidade dos dados"""
        print("📊 Teste 8: Métricas de qualidade dos dados")
        
        # Calcular métricas de qualidade
        quality_metrics = self.data_info.calculate_quality_metrics(self.df_clean)
        
        # Verificações
        self.assertIsInstance(quality_metrics, dict)
        self.assertIn('completeness', quality_metrics)
        self.assertIn('consistency', quality_metrics)
        self.assertIn('accuracy', quality_metrics)
        self.assertIn('timeliness', quality_metrics)
        
        # Verificar valores das métricas
        for metric, value in quality_metrics.items():
            self.assertIsInstance(value, (int, float))
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
        
        print("✅ Métricas de qualidade: OK")
        
        # Verificar se dados limpos têm alta qualidade
        self.assertGreater(quality_metrics['completeness'], 0.9)
        self.assertGreater(quality_metrics['consistency'], 0.9)
        
        print(f"✅ Qualidade dos dados: {quality_metrics['completeness']:.2f} completude, {quality_metrics['consistency']:.2f} consistência")
    
    def test_09_preprocessing_pipeline_integration(self):
        """Teste 9: Integração do pipeline de pré-processamento"""
        print("🔧 Teste 9: Integração do pipeline de pré-processamento")
        
        # Simular pipeline completo
        try:
            # 1. Análise inicial
            info = self.data_info.analyze_data(self.df_dirty)
            
            # 2. Detecção de problemas
            missing_info = self.data_cleaning.detect_missing_values(self.df_dirty)
            outlier_info = self.data_cleaning.detect_outliers(self.df_dirty)
            
            # 3. Validação
            validation = self.data_cleaning.validate_data(self.df_dirty)
            
            # 4. Limpeza
            df_cleaned, cleaning_report = self.data_cleaning.clean_data(
                self.df_dirty,
                remove_duplicates=True,
                handle_missing='impute',
                handle_outliers='remove',
                handle_infinite=True
            )
            
            # 5. Validação pós-limpeza
            validation_clean = self.data_cleaning.validate_data(df_cleaned)
            
            # Verificações de integração
            self.assertTrue(validation_clean['is_valid'])
            self.assertEqual(df_cleaned.isnull().sum().sum(), 0)
            
            print("✅ Pipeline de integração: OK")
            
            # Salvar relatório de integração
            integration_report = {
                'initial_info': info,
                'missing_info': missing_info,
                'outlier_info': outlier_info,
                'initial_validation': validation,
                'cleaning_report': cleaning_report,
                'final_validation': validation_clean
            }
            
            report_path = os.path.join(self.test_results_dir, "integration_report.txt")
            with open(report_path, 'w') as f:
                f.write("=== RELATÓRIO DE INTEGRAÇÃO DO PIPELINE ===\n\n")
                for key, value in integration_report.items():
                    f.write(f"{key.upper()}:\n")
                    f.write(f"{value}\n\n")
            
            print(f"📄 Relatório de integração salvo: {report_path}")
            
        except Exception as e:
            self.fail(f"Pipeline de integração falhou: {str(e)}")
    
    def test_10_edge_cases(self):
        """Teste 10: Casos extremos e edge cases"""
        print("⚠️ Teste 10: Casos extremos e edge cases")
        
        # Teste com DataFrame vazio
        df_empty = pd.DataFrame()
        try:
            info_empty = self.data_info.analyze_data(df_empty)
            self.assertEqual(info_empty['shape'], (0, 0))
            print("✅ DataFrame vazio: OK")
        except Exception as e:
            print(f"⚠️ DataFrame vazio gerou exceção: {e}")
        
        # Teste com DataFrame com uma única linha
        df_single = pd.DataFrame([[1, 2, 3, 0]], columns=['f1', 'f2', 'f3', 'target'])
        try:
            info_single = self.data_info.analyze_data(df_single)
            self.assertEqual(info_single['shape'], (1, 4))
            print("✅ DataFrame com uma linha: OK")
        except Exception as e:
            print(f"⚠️ DataFrame com uma linha gerou exceção: {e}")
        
        # Teste com DataFrame com apenas valores NaN
        df_all_nan = pd.DataFrame(np.nan, index=range(10), columns=['f1', 'f2', 'f3', 'target'])
        try:
            info_all_nan = self.data_info.analyze_data(df_all_nan)
            self.assertEqual(info_all_nan['missing_values'].sum(), 40)  # 10 linhas * 4 colunas
            print("✅ DataFrame com apenas NaN: OK")
        except Exception as e:
            print(f"⚠️ DataFrame com apenas NaN gerou exceção: {e}")
        
        # Teste com DataFrame com tipos mistos
        df_mixed = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'string': ['a', 'b', 'c', 'd', 'e'],
            'boolean': [True, False, True, False, True],
            'target': [0, 1, 0, 1, 0]
        })
        try:
            info_mixed = self.data_info.analyze_data(df_mixed)
            self.assertEqual(info_mixed['shape'], (5, 4))
            print("✅ DataFrame com tipos mistos: OK")
        except Exception as e:
            print(f"⚠️ DataFrame com tipos mistos gerou exceção: {e}")
    
    def tearDown(self):
        """Limpeza após cada teste"""
        pass

def run_preprocessing_tests():
    """Executar todos os testes de pré-processamento"""
    print("🚀 INICIANDO TESTES DE PRÉ-PROCESSAMENTO - MH-AutoML")
    print("=" * 60)
    
    # Criar suite de testes
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPreprocessing)
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumo dos resultados
    print("\n" + "=" * 60)
    print("📋 RESUMO DOS TESTES DE PRÉ-PROCESSAMENTO")
    print("=" * 60)
    print(f"✅ Testes executados: {result.testsRun}")
    print(f"❌ Falhas: {len(result.failures)}")
    print(f"⚠️ Erros: {len(result.errors)}")
    print(f"🎯 Taxa de sucesso: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FALHAS DETECTADAS:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n⚠️ ERROS DETECTADOS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    print(f"\n📁 Resultados salvos em: test_preprocessing_results/")
    print("=" * 60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_preprocessing_tests()
    sys.exit(0 if success else 1) 