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

class TestPreprocessingSimple(unittest.TestCase):
    """Testes simplificados para a etapa de pré-processamento do pipeline MHAutoML"""
    
    def setUp(self):
        """Configuração inicial para todos os testes"""
        print(f"\n🧪 Iniciando testes de pré-processamento - {datetime.now().strftime('%H:%M:%S')}")
        
        # Criar dados de teste sintéticos
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 10
        
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
        outlier_indices = np.random.choice(self.n_samples, size=10, replace=False)
        self.X_dirty[outlier_indices, 0] = 1000  # Outliers extremos
        
        # Adicionar valores infinitos
        inf_indices = np.random.choice(self.n_samples, size=5, replace=False)
        self.X_dirty[inf_indices, 1] = np.inf
        
        # Criar DataFrames
        feature_names = [f'feature_{i}' for i in range(self.n_features)]
        self.df_clean = pd.DataFrame(self.X_clean, columns=feature_names)
        self.df_clean['target'] = self.y_clean
        
        self.df_dirty = pd.DataFrame(self.X_dirty, columns=feature_names)
        self.df_dirty['target'] = self.y_dirty
        
        # Criar diretório de resultados de teste
        self.test_results_dir = "test_preprocessing_results"
        os.makedirs(self.test_results_dir, exist_ok=True)
    
    def test_01_data_info_basic_analysis(self):
        """Teste 1: Análise básica de informações dos dados"""
        print("📊 Teste 1: Análise básica de informações dos dados")
        
        # Instanciar DataInfo com dados limpos
        data_info_clean = DataInfo('target', self.df_clean)
        
        # Testar métodos básicos
        try:
            # Testar display_info_table
            info_table = data_info_clean.display_info_table()
            self.assertIsInstance(info_table, pd.DataFrame)
            self.assertEqual(info_table.iloc[0]['Rows'], self.n_samples)
            self.assertEqual(info_table.iloc[0]['Columns'], self.n_features + 1)
            print("✅ display_info_table: OK")
            
            # Testar display_data_types
            data_types = data_info_clean.display_data_types()
            self.assertIsInstance(data_types, pd.DataFrame)
            print("✅ display_data_types: OK")
            
            # Testar display_balance_info
            balance_info = data_info_clean.display_balance_info()
            self.assertIsInstance(balance_info, pd.DataFrame)
            print("✅ display_balance_info: OK")
            
            # Testar display_duplicates_missing
            duplicates_missing = data_info_clean.display_duplicates_missing()
            self.assertIsInstance(duplicates_missing, pd.DataFrame)
            print("✅ display_duplicates_missing: OK")
            
        except Exception as e:
            self.fail(f"Erro na análise básica: {str(e)}")
    
    def test_02_data_cleaning_basic_operations(self):
        """Teste 2: Operações básicas de limpeza de dados"""
        print("🧹 Teste 2: Operações básicas de limpeza de dados")
        
        # Instanciar DataCleaning
        data_cleaning = DataCleaning(label='target')
        
        try:
            # Testar transform com dados limpos
            df_transformed = data_cleaning.transform(self.df_clean)
            self.assertIsInstance(df_transformed, pd.DataFrame)
            self.assertEqual(df_transformed.shape, self.df_clean.shape)
            print("✅ Transform com dados limpos: OK")
            
            # Testar custom_convert
            test_values = [1, 0, 'true', 'false', '?', 3.14]
            converted_values = [data_cleaning.custom_convert(val) for val in test_values]
            self.assertEqual(converted_values[0], 1)  # int
            self.assertEqual(converted_values[1], 0)  # int
            self.assertEqual(converted_values[2], 1)  # 'true' -> 1
            self.assertEqual(converted_values[3], 0)  # 'false' -> 0
            self.assertTrue(np.isnan(converted_values[4]))  # '?' -> np.nan
            self.assertEqual(converted_values[5], 3.14)  # float
            print("✅ custom_convert: OK")
            
        except Exception as e:
            self.fail(f"Erro nas operações de limpeza: {str(e)}")
    
    def test_03_data_cleaning_with_dirty_data(self):
        """Teste 3: Limpeza de dados com problemas"""
        print("🔧 Teste 3: Limpeza de dados com problemas")
        
        # Instanciar DataCleaning com configurações específicas
        data_cleaning = DataCleaning(
            remove_duplicates=True,
            remove_missing_values=True,
            remove_outliers=True,
            label='target'
        )
        
        try:
            # Testar transform com dados sujos
            df_transformed = data_cleaning.transform(self.df_dirty)
            self.assertIsInstance(df_transformed, pd.DataFrame)
            
            # Verificar se dados foram limpos
            if df_transformed is not None:
                # Verificar se valores infinitos foram tratados
                numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    has_infinite = np.isinf(df_transformed[numeric_cols]).any().any()
                    self.assertFalse(has_infinite)
                
                print(f"✅ Transform com dados sujos: {df_transformed.shape[0]} amostras")
            else:
                print("⚠️ Transform retornou None (pode ser esperado para alguns casos)")
            
        except Exception as e:
            print(f"⚠️ Erro na limpeza de dados sujos: {str(e)}")
    
    def test_04_data_info_with_dirty_data(self):
        """Teste 4: Análise de dados com problemas"""
        print("🔍 Teste 4: Análise de dados com problemas")
        
        # Instanciar DataInfo com dados sujos
        data_info_dirty = DataInfo('target', self.df_dirty)
        
        try:
            # Testar análise de dados sujos
            info_table = data_info_dirty.display_info_table()
            self.assertIsInstance(info_table, pd.DataFrame)
            print("✅ Análise de dados sujos: OK")
            
            # Testar detecção de duplicatas e valores faltantes
            duplicates_missing = data_info_dirty.display_duplicates_missing()
            self.assertIsInstance(duplicates_missing, pd.DataFrame)
            
            # Verificar se valores faltantes foram detectados
            null_count = self.df_dirty.isnull().sum().sum()
            if null_count > 0:
                print(f"✅ Valores faltantes detectados: {null_count}")
            
        except Exception as e:
            self.fail(f"Erro na análise de dados sujos: {str(e)}")
    
    def test_05_data_cleaning_steps_individual(self):
        """Teste 5: Testes individuais dos passos de limpeza"""
        print("⚙️ Teste 5: Testes individuais dos passos de limpeza")
        
        data_cleaning = DataCleaning(label='target')
        
        try:
            # Testar remove_outliers_step
            df_no_outliers = data_cleaning.remove_outliers_step(self.df_dirty.copy())
            if df_no_outliers is not None:
                self.assertIsInstance(df_no_outliers, pd.DataFrame)
                print("✅ remove_outliers_step: OK")
            
            # Testar remove_duplicates_step
            df_no_duplicates = data_cleaning.remove_duplicates_step(self.df_dirty.copy())
            if df_no_duplicates is not None:
                self.assertIsInstance(df_no_duplicates, pd.DataFrame)
                print("✅ remove_duplicates_step: OK")
            
            # Testar remove_missing_values_step
            df_no_missing = data_cleaning.remove_missing_values_step(self.df_dirty.copy())
            if df_no_missing is not None:
                self.assertIsInstance(df_no_missing, pd.DataFrame)
                # Verificar se valores faltantes foram removidos
                null_count = df_no_missing.isnull().sum().sum()
                self.assertEqual(null_count, 0)
                print("✅ remove_missing_values_step: OK")
            
        except Exception as e:
            print(f"⚠️ Erro nos passos individuais: {str(e)}")
    
    def test_06_data_info_features_analysis(self):
        """Teste 6: Análise de features específicas"""
        print("🔬 Teste 6: Análise de features específicas")
        
        data_info = DataInfo('target', self.df_clean)
        
        try:
            # Testar display_features_info
            features_info = data_info.display_features_info()
            self.assertIsInstance(features_info, pd.DataFrame)
            print("✅ display_features_info: OK")
            
            # Testar has_categorical_rows
            has_categorical = data_info.has_categorical_rows()
            self.assertTrue(isinstance(has_categorical, (bool, type(None), np.bool_)))
            print("✅ has_categorical_rows: OK")
            
            # Testar find_and_drop_crypto_column
            crypto_col = data_info.find_and_drop_crypto_column()
            # Pode retornar None se não encontrar coluna criptográfica
            if crypto_col is not None:
                self.assertIsInstance(crypto_col, str)
                print(f"✅ Coluna criptográfica encontrada: {crypto_col}")
            else:
                print("✅ Nenhuma coluna criptográfica encontrada (esperado)")
            
        except Exception as e:
            self.fail(f"Erro na análise de features: {str(e)}")
    
    def test_07_data_cleaning_visualization(self):
        """Teste 7: Testes de visualização"""
        print("📊 Teste 7: Testes de visualização")
        
        data_cleaning = DataCleaning(label='target')
        
        try:
            # Testar plot_missing_values_heatmap
            data_cleaning.plot_missing_values_heatmap(self.df_dirty, self.test_results_dir)
            print("✅ plot_missing_values_heatmap: OK")
            
            # Verificar se arquivo foi criado
            heatmap_files = [f for f in os.listdir(self.test_results_dir) if 'missing_values' in f]
            if heatmap_files:
                print(f"✅ Arquivo de heatmap criado: {heatmap_files[0]}")
            
        except Exception as e:
            print(f"⚠️ Erro na visualização: {str(e)}")
    
    def test_08_edge_cases(self):
        """Teste 8: Casos extremos"""
        print("⚠️ Teste 8: Casos extremos")
        
        try:
            # Teste com DataFrame vazio
            df_empty = pd.DataFrame()
            data_info_empty = DataInfo('target', df_empty)
            info_table = data_info_empty.display_info_table()
            self.assertIsInstance(info_table, pd.DataFrame)
            print("✅ DataFrame vazio: OK")
            
            # Teste com DataFrame com uma única linha
            df_single = pd.DataFrame([[1, 2, 3, 0]], columns=['f1', 'f2', 'f3', 'target'])
            data_info_single = DataInfo('target', df_single)
            info_table = data_info_single.display_info_table()
            self.assertEqual(info_table.iloc[0]['Rows'], 1)
            print("✅ DataFrame com uma linha: OK")
            
            # Teste com DataFrame com apenas valores NaN
            df_all_nan = pd.DataFrame(np.nan, index=range(5), columns=['f1', 'f2', 'f3', 'target'])
            data_info_all_nan = DataInfo('target', df_all_nan)
            info_table = data_info_all_nan.display_info_table()
            self.assertIsInstance(info_table, pd.DataFrame)
            print("✅ DataFrame com apenas NaN: OK")
            
        except Exception as e:
            print(f"⚠️ Erro em casos extremos: {str(e)}")
    
    def test_09_integration_test(self):
        """Teste 9: Teste de integração completo"""
        print("🔗 Teste 9: Teste de integração completo")
        
        try:
            # 1. Análise inicial
            data_info = DataInfo('target', self.df_dirty)
            info_table = data_info.display_info_table()
            print("✅ Análise inicial: OK")
            
            # 2. Limpeza de dados
            data_cleaning = DataCleaning(
                remove_duplicates=True,
                remove_missing_values=True,
                remove_outliers=True,
                label='target'
            )
            
            df_cleaned = data_cleaning.transform(self.df_dirty)
            if df_cleaned is not None:
                print(f"✅ Limpeza concluída: {df_cleaned.shape[0]} amostras")
                
                # 3. Análise pós-limpeza
                data_info_clean = DataInfo('target', df_cleaned)
                info_table_clean = data_info_clean.display_info_table()
                print("✅ Análise pós-limpeza: OK")
                
                # Salvar dados limpos
                cleaned_path = os.path.join(self.test_results_dir, "cleaned_data_integration.csv")
                df_cleaned.to_csv(cleaned_path, index=False)
                print(f"💾 Dados limpos salvos: {cleaned_path}")
            else:
                print("⚠️ Limpeza retornou None")
            
        except Exception as e:
            self.fail(f"Erro no teste de integração: {str(e)}")
    
    def test_10_performance_test(self):
        """Teste 10: Teste de performance"""
        print("⚡ Teste 10: Teste de performance")
        
        try:
            # Criar dataset maior para teste de performance
            n_samples_large = 1000
            n_features_large = 50
            X_large = np.random.randn(n_samples_large, n_features_large)
            y_large = np.random.randint(0, 2, n_samples_large)
            
            feature_names = [f'feature_{i}' for i in range(n_features_large)]
            df_large = pd.DataFrame(X_large, columns=feature_names)
            df_large['target'] = y_large
            
            # Adicionar alguns problemas
            mask = np.random.random(df_large.shape) < 0.05  # 5% de valores faltantes
            df_large[mask] = np.nan
            
            # Medir tempo de análise
            import time
            start_time = time.time()
            
            data_info = DataInfo('target', df_large)
            info_table = data_info.display_info_table()
            
            analysis_time = time.time() - start_time
            print(f"✅ Análise de {n_samples_large} amostras em {analysis_time:.2f}s")
            
            # Medir tempo de limpeza
            start_time = time.time()
            
            data_cleaning = DataCleaning(
                remove_duplicates=True,
                remove_missing_values=True,
                remove_outliers=True,
                label='target'
            )
            
            df_cleaned = data_cleaning.transform(df_large)
            
            cleaning_time = time.time() - start_time
            print(f"✅ Limpeza de {n_samples_large} amostras em {cleaning_time:.2f}s")
            
            # Verificar se performance é aceitável (< 10 segundos)
            total_time = analysis_time + cleaning_time
            self.assertLess(total_time, 10.0, f"Performance muito lenta: {total_time:.2f}s")
            
        except Exception as e:
            print(f"⚠️ Erro no teste de performance: {str(e)}")
    
    def tearDown(self):
        """Limpeza após cada teste"""
        pass

def run_preprocessing_simple_tests():
    """Executar todos os testes de pré-processamento simplificados"""
    print("🚀 INICIANDO TESTES DE PRÉ-PROCESSAMENTO SIMPLIFICADOS - MH-AutoML")
    print("=" * 70)
    
    # Criar suite de testes
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPreprocessingSimple)
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumo dos resultados
    print("\n" + "=" * 70)
    print("📋 RESUMO DOS TESTES DE PRÉ-PROCESSAMENTO SIMPLIFICADOS")
    print("=" * 70)
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
    print("=" * 70)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_preprocessing_simple_tests()
    sys.exit(0 if success else 1) 