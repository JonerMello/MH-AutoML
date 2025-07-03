# Testes do Sistema MH-AutoML

Este diretório contém todos os testes do sistema MH-AutoML, organizados por funcionalidade.

## 📋 Estrutura dos Testes

### 🔧 Testes de Funcionalidades Principais

#### `Test_Data_Cleaning.py`
- **Propósito**: Testa a limpeza de dados
- **Funcionalidades testadas**:
  - Remoção de valores faltantes
  - Remoção de duplicatas
  - Tratamento de outliers
- **Execução**: `python test/Test_Data_Cleaning.py`

#### `Test_Data_Transformation.py`
- **Propósito**: Testa transformações de dados
- **Funcionalidades testadas**:
  - One-hot encoding
  - Label encoding
  - Normalização
- **Execução**: `python test/Test_Data_Transformation.py`

#### `Test_DataCleaning_And_Transformation.py`
- **Propósito**: Teste integrado de limpeza e transformação
- **Funcionalidades testadas**:
  - Pipeline completo de pré-processamento
  - Integração entre módulos
- **Execução**: `python test/Test_DataCleaning_And_Transformation.py`

#### `Test_feature_selection.py`
- **Propósito**: Testa seleção de features
- **Funcionalidades testadas**:
  - PCA (Principal Component Analysis)
  - LASSO (Least Absolute Shrinkage and Selection Operator)
  - ANOVA (Analysis of Variance)
- **Execução**: `python test/Test_feature_selection.py`

#### `Test_Hyperparameters.py`
- **Propósito**: Testa otimização de hiperparâmetros
- **Funcionalidades testadas**:
  - Otimização com Optuna
  - Grid Search
  - Validação cruzada
- **Execução**: `python test/Test_Hyperparameters.py`

### 📄 Testes do Gerador de PDF

#### `Test_PDF_Report_Generator.py`
- **Propósito**: Teste completo do gerador de relatórios PDF
- **Funcionalidades testadas**:
  - Conversão HTML para PNG (Playwright, Selenium)
  - Geração de PDF (WeasyPrint, pdfkit, ReportLab)
  - Fallbacks e tratamento de erros
  - Categorização de artefatos
- **Execução**: `python test/Test_PDF_Report_Generator.py`

#### `Test_PDF_Quick.py`
- **Propósito**: Teste rápido do gerador de PDF
- **Funcionalidades testadas**:
  - Verificação básica de funcionalidade
  - Geração de PDF simples
  - Validação de arquivo gerado
- **Execução**: `python test/Test_PDF_Quick.py`

#### `Test_PDF_Template.py`
- **Propósito**: Teste específico do template MH-AutoML
- **Funcionalidades testadas**:
  - Estrutura de seções (0. Data Info, 1. Preprocessing, etc.)
  - Categorização MLflow (00_Data_info, 01_preprocessing, etc.)
  - Nomenclatura correta de imagens
  - Pipeline steps seguindo estrutura real
- **Execução**: `python test/Test_PDF_Template.py`

## 🚀 Como Executar os Testes

### Execução Individual
```bash
# Teste específico
python test/Test_PDF_Template.py

# Teste rápido
python test/Test_PDF_Quick.py

# Teste completo
python test/Test_PDF_Report_Generator.py
```

### Execução de Todos os Testes
```bash
# Executar todos os testes Python
python -m pytest test/ -v

# Ou executar cada teste individualmente
for test_file in test/Test_*.py; do
    python "$test_file"
done
```

## 📊 Resultados Esperados

### Testes de Funcionalidades
- ✅ Limpeza de dados funcionando
- ✅ Transformações aplicadas corretamente
- ✅ Seleção de features operacional
- ✅ Otimização de hiperparâmetros funcionando

### Testes do PDF
- ✅ PDF gerado com sucesso
- ✅ Template MH-AutoML seguido
- ✅ Categorização MLflow correta
- ✅ Fallbacks funcionando no Windows

## 🐛 Troubleshooting

### Problemas Comuns

#### Erro de Importação
```bash
# Adicionar diretório ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Dependências Faltando
```bash
# Instalar dependências de teste
pip install pytest matplotlib pandas numpy
```

#### Erro no Windows
```bash
# Usar caminhos com barras normais
python test\Test_PDF_Template.py
```

### Logs de Erro
- Verificar se todas as dependências estão instaladas
- Confirmar que o diretório `test_results` existe
- Verificar permissões de escrita no diretório

## 📈 Histórico de Testes

### Versão Atual
- ✅ Todos os testes passando
- ✅ Template PDF corrigido
- ✅ Estrutura MH-AutoML implementada
- ✅ Fallbacks funcionando

### Melhorias Futuras
- [ ] Testes de integração end-to-end
- [ ] Testes de performance
- [ ] Testes de stress
- [ ] Cobertura de código

## 📚 Documentação Relacionada

- [PDF_REPORT_GUIDE.md](../PDF_REPORT_GUIDE.md) - Guia do gerador de PDF
- [README.md](../README.md) - Documentação principal
- [DIAGRAMS.md](../DIAGRAMS.md) - Diagramas do sistema

## 🤝 Contribuição

Para adicionar novos testes:

1. Criar arquivo `Test_NovaFuncionalidade.py`
2. Seguir padrão de nomenclatura
3. Incluir documentação
4. Testar em diferentes ambientes
5. Atualizar este README

---

**Última atualização**: 30/06/2025
**Versão**: 1.0.0 