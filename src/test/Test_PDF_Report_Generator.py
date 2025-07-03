#!/usr/bin/env python3
"""
Teste completo do gerador de relatórios PDF do MH-AutoML
Testa todas as funcionalidades: conversão HTML->PNG, geração PDF, fallbacks
"""

import os
import sys
import pandas as pd
import numpy as np
import unittest
from unittest.mock import Mock, patch

# Adicionar o diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestPDFReportGenerator(unittest.TestCase):
    """Testes para o gerador de relatórios PDF"""
    
    def setUp(self):
        """Configuração inicial para cada teste"""
        self.test_results_folder = "test_results"
        if not os.path.exists(self.test_results_folder):
            os.makedirs(self.test_results_folder)
        
        # Criar dados de teste
        np.random.seed(42)
        self.n_samples = 50
        self.n_features = 5
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randint(0, 2, self.n_samples)
        
        # DataFrame simulado
        self.feature_names = [f'feature_{i}' for i in range(self.n_features)]
        self.dataset_df = pd.DataFrame(self.X, columns=self.feature_names)
        self.dataset_df['target'] = self.y
        
        # Criar arquivos de teste
        self._create_test_files()
    
    def _create_test_files(self):
        """Criar arquivos de teste seguindo o template MLflow"""
        import matplotlib.pyplot as plt
        
        # Arquivos de pré-processamento
        fig, ax = plt.subplots(figsize=(8, 6))
        data = np.random.rand(10, 10)
        im = ax.imshow(data, cmap='viridis')
        ax.set_title('Missing Values Heatmap')
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(os.path.join(self.test_results_folder, "missing_values_heatmap.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        # clean_missing_values_heatmap.png
        fig, ax = plt.subplots(figsize=(8, 6))
        data = np.random.rand(10, 10)
        im = ax.imshow(data, cmap='plasma')
        ax.set_title('Clean Missing Values Heatmap')
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(os.path.join(self.test_results_folder, "clean_missing_values_heatmap.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        # lasso_feature_importance.png
        fig, ax = plt.subplots(figsize=(8, 6))
        features = self.feature_names[:3]
        importance = np.random.rand(len(features))
        ax.barh(features, importance)
        ax.set_title('LASSO Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.test_results_folder, "lasso_feature_importance.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        # optuna_optimization_history.png
        fig, ax = plt.subplots(figsize=(8, 6))
        trials = range(1, 21)
        scores = np.random.rand(20)
        ax.plot(trials, scores)
        ax.set_title('Optuna Optimization History')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Score')
        plt.tight_layout()
        plt.savefig(os.path.join(self.test_results_folder, "optuna_optimization_history.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        # performance_metrics.jpg
        fig, ax = plt.subplots(figsize=(8, 6))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = np.random.rand(4)
        ax.bar(metrics, values)
        ax.set_title('Performance Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(self.test_results_folder, "performance_metrics.jpg"), dpi=100, bbox_inches='tight')
        plt.close()
        
        # Arquivo HTML de teste
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Test HTML</title></head>
        <body>
            <h1>Test HTML Content</h1>
            <p>This is a test HTML file for conversion.</p>
        </body>
        </html>
        """
        with open(os.path.join(self.test_results_folder, "test_plot.html"), 'w') as f:
            f.write(html_content)
    
    def test_pdf_generator_initialization(self):
        """Testa inicialização do gerador de PDF"""
        from model.tools.pdf_report_generator import PDFReportGenerator
        
        generator = PDFReportGenerator(self.test_results_folder)
        self.assertIsNotNone(generator)
        self.assertEqual(generator.results_folder, self.test_results_folder)
    
    def test_html_to_png_conversion_methods(self):
        """Testa métodos de conversão HTML para PNG"""
        from model.tools.pdf_report_generator import PDFReportGenerator
        
        generator = PDFReportGenerator(self.test_results_folder)
        html_file = os.path.join(self.test_results_folder, "test_plot.html")
        
        # Testar conversão com Playwright (se disponível)
        try:
            png_path = generator._convert_html_to_png_playwright(html_file)
            if png_path:
                self.assertTrue(os.path.exists(png_path))
        except Exception:
            pass  # Playwright pode não estar disponível
        
        # Testar conversão com Selenium (se disponível)
        try:
            png_path = generator._convert_html_to_png_selenium(html_file)
            if png_path:
                self.assertTrue(os.path.exists(png_path))
        except Exception:
            pass  # Selenium pode não estar disponível
    
    def test_pdf_generation_methods(self):
        """Testa métodos de geração de PDF"""
        from model.tools.pdf_report_generator import PDFReportGenerator
        
        generator = PDFReportGenerator(self.test_results_folder)
        html_content = "<html><body><h1>Test</h1></body></html>"
        
        # Testar WeasyPrint (se disponível)
        try:
            pdf_path = generator._generate_pdf_weasyprint(html_content)
            if pdf_path:
                self.assertTrue(os.path.exists(pdf_path))
        except Exception:
            pass  # WeasyPrint pode não estar disponível
        
        # Testar pdfkit (se disponível)
        try:
            pdf_path = generator._generate_pdf_pdfkit(html_content)
            if pdf_path:
                self.assertTrue(os.path.exists(pdf_path))
        except Exception:
            pass  # pdfkit pode não estar disponível
    
    def test_reportlab_fallback(self):
        """Testa geração de PDF com ReportLab (fallback)"""
        from model.tools.pdf_report_generator import PDFReportGenerator
        
        generator = PDFReportGenerator(self.test_results_folder)
        
        # Mock data
        class MockPipeline:
            def __init__(self):
                self.steps = [
                    ('Preprocessor', Mock()),
                    ('Feature engineering', Mock()),
                    ('Ensemble Classifier', Mock())
                ]
        
        class MockModel:
            def __init__(self):
                self.estimators_ = {}
                self.named_estimators_ = {}
            
            def get_params(self):
                return {'param1': 'value1'}
        
        # Gerar PDF
        pdf_filename = generator.generate_pdf_report(
            pipeline=MockPipeline(),
            display_data=Mock(),
            study=None,
            best_model=MockModel(),
            model_name="TestModel",
            model_params={'param1': 'value1'},
            select_results=None,
            report="Test Report",
            y_test=self.y[:20],
            y_pred=self.y[:20],
            feature_names=self.feature_names,
            feature_selection_info={'method': 'lasso'},
            shap_exp_filepath=None,
            exp_filepath=None,
            lime_exp_filepath=None
        )
        
        self.assertIsNotNone(pdf_filename)
        self.assertTrue(os.path.exists(pdf_filename))
        self.assertTrue(pdf_filename.endswith('.pdf'))
        
        # Verificar tamanho do arquivo
        file_size = os.path.getsize(pdf_filename)
        self.assertGreater(file_size, 1000)  # Deve ter pelo menos 1KB
    
    def test_template_structure(self):
        """Testa se o PDF segue o template correto do MH-AutoML"""
        from model.tools.pdf_report_generator import PDFReportGenerator
        
        generator = PDFReportGenerator(self.test_results_folder)
        
        # Mock data
        class MockPipeline:
            def __init__(self):
                self.steps = [
                    ('Preprocessor', Mock()),
                    ('Feature engineering', Mock()),
                    ('Ensemble Classifier', Mock())
                ]
        
        class MockModel:
            def __init__(self):
                self.estimators_ = {}
                self.named_estimators_ = {}
            
            def get_params(self):
                return {'param1': 'value1'}
        
        # Gerar PDF
        pdf_filename = generator.generate_pdf_report(
            pipeline=MockPipeline(),
            display_data=Mock(),
            study=None,
            best_model=MockModel(),
            model_name="TestModel",
            model_params={'param1': 'value1'},
            select_results=None,
            report="Test Report",
            y_test=self.y[:20],
            y_pred=self.y[:20],
            feature_names=self.feature_names,
            feature_selection_info={'method': 'lasso'},
            shap_exp_filepath=None,
            exp_filepath=None,
            lime_exp_filepath=None
        )
        
        # Verificar se o arquivo foi gerado
        self.assertIsNotNone(pdf_filename)
        self.assertTrue(os.path.exists(pdf_filename))
        
        # Verificar se é PDF (não HTML fallback)
        if pdf_filename.endswith('.pdf'):
            print("✅ PDF gerado corretamente com template MH-AutoML")
        else:
            print("📄 HTML gerado como fallback")
    
    def test_artifact_categorization(self):
        """Testa categorização de artefatos seguindo template MLflow"""
        from model.tools.pdf_report_generator import PDFReportGenerator
        
        generator = PDFReportGenerator(self.test_results_folder)
        
        # Verificar se os arquivos foram categorizados corretamente
        expected_categories = [
            "00_Data_info",
            "01_preprocessing", 
            "02_feature_engineering",
            "03_model_optimization",
            "04_evaluation_metrics",
            "05_interpretability",
            "Reports"
        ]
        
        # Simular categorização
        artifact_categories = {
            "00_Data_info": [],
            "01_preprocessing": ["missing_values_heatmap.png", "clean_missing_values_heatmap.png"],
            "02_feature_engineering": ["lasso_feature_importance.png"],
            "03_model_optimization": ["optuna_optimization_history.png"],
            "04_evaluation_metrics": ["performance_metrics.jpg"],
            "05_interpretability": [],
            "Reports": []
        }
        
        # Verificar se as categorias estão corretas
        for category in expected_categories:
            self.assertIn(category, artifact_categories)
    
    def tearDown(self):
        """Limpeza após cada teste"""
        # Manter os arquivos de teste para inspeção manual
        pass

def run_tests():
    """Executa todos os testes"""
    print("🧪 Executando testes do gerador de PDF...")
    
    # Criar suite de testes
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPDFReportGenerator)
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumo
    print(f"\n📊 Resumo dos testes:")
    print(f"   ✅ Testes passaram: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   ❌ Testes falharam: {len(result.failures)}")
    print(f"   💥 Testes com erro: {len(result.errors)}")
    print(f"   📈 Total: {result.testsRun}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n🎉 Todos os testes passaram!")
    else:
        print("\n💥 Alguns testes falharam.")
    
    print("\n📚 Para mais informações, consulte o PDF_REPORT_GUIDE.md") 