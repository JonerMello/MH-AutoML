import pandas as pd
from model.tools.pdf_report_generator import PDFReportGenerator

# Crie um DataFrame de exemplo
df = pd.DataFrame({
    'permission_1': [1, 0, 1],
    'api_call_1': [0.5, 0.7, 0.2],
    'target': [0, 1, 0]
})

class DummyDisplayData:
    def __init__(self, dataset):
        self.dataset = dataset

display_data = DummyDisplayData(df)

pdf_generator = PDFReportGenerator("../results")
pdf_generator._generate_simple_pdf_report(
    html_content="",
    pdf_path="../results/test_minimal_report.pdf",
    display_data=display_data
)

print("✅ PDF gerado: ../results/test_minimal_report.pdf") 