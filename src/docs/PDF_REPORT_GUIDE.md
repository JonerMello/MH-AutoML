# 📄 Guia de Relatórios PDF - MH-AutoML

## 🎯 Visão Geral

O sistema MH-AutoML agora gera automaticamente **relatórios PDF profissionais** além dos relatórios HTML. O sistema usa múltiplos métodos de geração com fallbacks para garantir compatibilidade em diferentes ambientes.

## ✨ Funcionalidades

### 🔄 Sistema de Fallback Inteligente
1. **WeasyPrint** (método principal - requer GTK no Windows)
2. **pdfkit/wkhtmltopdf** (método alternativo)
3. **ReportLab** (✅ **FUNCIONANDO NO WINDOWS** - método de fallback)

### 📊 Conteúdo do Relatório PDF
- **Página de título** com timestamp
- **Resumo executivo**
- **Configuração do pipeline**
- **Análise de dados**
- **Visualizações** (imagens PNG/JPG)
- **Métricas de performance**
- **Lista completa de artefatos**
- **Formatação profissional**

## 🚀 Como Usar

### Execução Automática
O PDF é gerado automaticamente quando você executa o pipeline:

```bash
python -m view.main --dataset Datasets/android_permissions.csv --feature_selection lasso
```

### Arquivos Gerados
- `report_YYYYMMDD_HHMMSS.html` - Relatório HTML completo
- `pdf_report_YYYYMMDD_HHMMSS.pdf` - Relatório PDF profissional

## 🔧 Instalação de Dependências

### Dependências Principais (já instaladas)
```bash
pip install weasyprint playwright pdfkit reportlab
playwright install chromium
```

### Para Windows (Recomendado)
```bash
# Instalar ReportLab (funciona nativamente no Windows)
pip install reportlab

# Instalar Playwright para conversão de imagens
pip install playwright
playwright install chromium
```

### Para Linux/macOS (PDF Completo)
```bash
# Instalar WeasyPrint com dependências
pip install weasyprint
# Ubuntu/Debian: sudo apt-get install build-essential python3-dev python3-pip python3-setuptools python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
```

## 📋 Estrutura do Relatório PDF

### 1. Página de Título
- Título do relatório
- Data e hora de geração
- Sistema MH-AutoML

### 2. Resumo Executivo
- Visão geral da análise
- Objetivos e metodologia

### 3. Configuração do Pipeline
- Etapas do pipeline AutoML
- Algoritmos utilizados

### 4. Análise de Dados
- Informações do dataset
- Visualizações de pré-processamento
- Gráficos de distribuição

### 5. Engenharia de Features
- Método de seleção de features
- Features selecionadas
- Gráficos de importância

### 6. Otimização do Modelo
- Hiperparâmetros otimizados
- Gráficos do Optuna
- Ranking de modelos

### 7. Métricas de Avaliação
- Relatório de classificação
- Matriz de confusão
- Curvas ROC/PR

### 8. Interpretabilidade
- Gráficos SHAP
- Análises LIME
- Explicações do modelo

### 9. Artefatos Gerados
- Lista completa de arquivos
- Categorização por tipo
- Links para arquivos

## 🎨 Formatação

### Estilos Profissionais
- **Fonte**: Helvetica/Arial
- **Cores**: Azul escuro para títulos, verde para subtítulos
- **Layout**: Margens de 1 polegada
- **Imagens**: Incluídas automaticamente com legendas

### Quebras de Página
- Título em página separada
- Seções organizadas logicamente
- Imagens não quebradas entre páginas

## 🔍 Solução de Problemas

### WeasyPrint no Windows
```
Erro: cannot load library 'gobject-2.0-0'
Solução: O sistema automaticamente usa ReportLab como fallback
```

### Imagens não aparecem
```
Problema: Conversão HTML para PNG falha
Solução: Verificar se Playwright está instalado
```

### PDF muito simples
```
Problema: Apenas texto, sem formatação
Solução: Verificar se ReportLab está instalado
```

## 📈 Comparação de Métodos

| Método | Windows | Linux | Qualidade | Velocidade |
|--------|---------|-------|-----------|------------|
| WeasyPrint | ❌ | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| pdfkit | ⚠️ | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| ReportLab | ✅ | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🎯 Benefícios

### Para Usuários
- **Portabilidade**: PDF funciona em qualquer dispositivo
- **Profissionalismo**: Relatórios prontos para apresentação
- **Completude**: Todas as informações incluídas
- **Compatibilidade**: Funciona em Windows, Linux, macOS

### Para Desenvolvedores
- **Robustez**: Múltiplos métodos de fallback
- **Flexibilidade**: Adapta-se ao ambiente
- **Manutenibilidade**: Código bem estruturado
- **Extensibilidade**: Fácil adicionar novos métodos

## 🔮 Próximas Melhorias

### Planejadas
- [ ] Suporte a tabelas complexas no PDF
- [ ] Gráficos interativos convertidos para estáticos
- [ ] Templates personalizáveis
- [ ] Compressão de imagens automática

### Em Desenvolvimento
- [ ] Suporte a múltiplos idiomas
- [ ] Exportação para outros formatos (DOCX, PPTX)
- [ ] Integração com sistemas de versionamento

## 📞 Suporte

### Logs de Debug
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Teste Manual
```bash
python test_pdf_generator.py
```

### Verificação de Dependências
```bash
python -c "import weasyprint, reportlab, playwright; print('Todas as dependências OK')"
```

---

**🎉 O sistema de relatórios PDF está pronto para uso em produção!** 