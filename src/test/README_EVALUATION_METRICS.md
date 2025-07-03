# Gráficos de Avaliação - Seção 04_evaluation_metrics

Este documento descreve todos os gráficos de avaliação que são gerados automaticamente e salvos na seção `04_evaluation_metrics` do MLflow.

## 📊 Gráficos Disponíveis

### 1. Matriz de Confusão (`confusion_matrix.png`)
- **Descrição**: Mostra a matriz de confusão com valores absolutos
- **Classes**: Benign (0) e Malware (1)
- **Visualização**: Heatmap com cores azuis
- **Informações**: Verdadeiros positivos, falsos positivos, verdadeiros negativos, falsos negativos

### 2. Curva ROC/AUC (`roc_curve.png`)
- **Descrição**: Curva ROC (Receiver Operating Characteristic) com valor AUC
- **Eixo X**: False Positive Rate (1 - Especificidade)
- **Eixo Y**: True Positive Rate (Sensibilidade)
- **Linha de referência**: Linha diagonal (AUC = 0.5)
- **Informações**: Valor AUC calculado automaticamente

### 3. Curva Precisão-Recall (`precision_recall_curve.png`)
- **Descrição**: Curva de precisão vs recall com Average Precision (AP)
- **Eixo X**: Recall (Sensibilidade)
- **Eixo Y**: Precisão
- **Linha de referência**: Linha horizontal em y=1 (precisão perfeita)
- **Informações**: Valor AP (Average Precision) calculado automaticamente

### 4. Distribuição de Probabilidades (`probability_distribution.png`)
- **Descrição**: Histograma das probabilidades preditas por classe real
- **Classes**: Benign (azul) e Malware (vermelho)
- **Linha de referência**: Limiar de decisão em 0.5
- **Informações**: Densidade de probabilidades para cada classe

### 5. Métricas por Classe (`metrics_by_class.png`)
- **Descrição**: Gráfico de barras com métricas de performance por classe
- **Métricas**: Precisão, Recall e F1-Score
- **Classes**: Benign e Malware
- **Informações**: Valores numéricos exibidos nas barras

## 🔧 Como são Gerados

Os gráficos são gerados automaticamente no método `evaluate_model()` da classe `Core`:

```python
def evaluate_model(self, model, X_test, y_test):
    # ... código existente ...
    
    # Gerar gráficos de avaliação
    self._generate_evaluation_plots(y_test, y_pred, y_pred_proba)
    
    return report, y_test, y_pred
```

## 📁 Localização no MLflow

Todos os gráficos são salvos em:
- **Pasta local**: `results/`
- **Seção MLflow**: `04_evaluation_metrics/`

## 🎯 Utilidade dos Gráficos

### Para Detecção de Malware:
1. **Matriz de Confusão**: Identificar falsos positivos e falsos negativos
2. **Curva ROC**: Avaliar capacidade discriminativa geral do modelo
3. **Curva PR**: Especialmente útil para dados desbalanceados
4. **Distribuição de Probabilidades**: Verificar separação entre classes
5. **Métricas por Classe**: Comparar performance entre benign e malware

### Para Análise de Performance:
- **AUC > 0.9**: Excelente discriminação
- **AUC 0.8-0.9**: Boa discriminação
- **AUC 0.7-0.8**: Discriminação aceitável
- **AUC < 0.7**: Discriminação pobre

## 🚀 Execução

Os gráficos são gerados automaticamente durante a execução do pipeline:

```bash
python -c "from controller.core import Core; core = Core('Datasets/android_permissions.csv', 'class'); core.run()"
```

## 📋 Teste dos Gráficos

Para testar a geração dos gráficos:

```bash
python test/test_all_evaluation_plots.py
```

## 📈 Métricas Adicionais Logadas

Além dos gráficos, as seguintes métricas são logadas no MLflow:

- `accuracy`: Acurácia geral
- `precision`: Precisão macro-média
- `recall`: Recall macro-média
- `f1`: F1-score macro-média
- `mcc`: Coeficiente de correlação de Matthews
- `roc_auc`: Área sob a curva ROC
- `average_precision`: Precisão média
- `f2_score`: F2-score (dá mais peso ao recall)
- `true_positive_rate`: Taxa de verdadeiros positivos
- `false_alarm_ratio`: Razão de falsos alarmes

## 🔍 Interpretação

### Para Detecção de Malware:
- **Alto Recall**: Importante para não perder malware
- **Alta Precisão**: Importante para reduzir falsos positivos
- **AUC Alto**: Indica boa capacidade discriminativa
- **Distribuição Separada**: Probabilidades bem separadas entre classes

### Sinais de Problemas:
- **AUC Baixo**: Modelo não consegue discriminar bem
- **Distribuição Sobreposta**: Probabilidades muito similares entre classes
- **Recall Baixo**: Muitos malwares não detectados
- **Precisão Baixa**: Muitos falsos positivos 