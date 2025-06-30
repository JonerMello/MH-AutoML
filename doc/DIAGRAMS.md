# MH-AutoML: Diagramas de Arquitetura

Este documento contém os diagramas de arquitetura do sistema MH-AutoML gerados em PlantUML.

## 📊 Diagramas Disponíveis

### 1. Diagrama de Classes (`class_diagram.puml`)
**Descrição**: Representa a estrutura completa das classes do sistema, suas relações e hierarquias.

**Características**:
- ✅ **Classes Principais**: Todas as classes do sistema organizadas por pacotes
- ✅ **Atributos e Métodos**: Detalhamento completo de cada classe
- ✅ **Relacionamentos**: Herança, composição e dependências
- ✅ **Arquitetura MVC**: Separação clara entre View, Controller e Model
- ✅ **Dependências Externas**: Bibliotecas e frameworks utilizados

**Pacotes Representados**:
- **View**: Interface CLI (Main)
- **Controller**: Orquestração (Core)
- **Model::Preprocessing**: Análise, limpeza e transformação
- **Model::FeatureEngineering**: Seleção de features
- **Model::Optimization**: Otimização de hiperparâmetros
- **Model::Interpretability**: Interpretação de modelos
- **Model::Tools**: Utilitários e validações
- **External Dependencies**: Bibliotecas externas

### 2. Diagrama de Sequência (`sequence_diagram.puml`)
**Descrição**: Mostra o fluxo de execução do sistema, desde o comando do usuário até a geração dos resultados.

**Características**:
- ✅ **Fluxo Completo**: Todas as etapas do pipeline
- ✅ **Interações**: Comunicação entre componentes
- ✅ **Condicionais**: Tratamento de erros e validações
- ✅ **Paralelismo**: Execução de tarefas simultâneas
- ✅ **Resultados**: Saídas e artefatos gerados

**Etapas Representadas**:
1. **Validação**: Verificação do dataset
2. **Análise**: Informações sobre os dados
3. **Limpeza**: Remoção de duplicatas e valores faltantes
4. **Transformação**: Encoding e normalização
5. **Seleção de Features**: LASSO, PCA ou ANOVA
6. **Otimização**: Hiperparâmetros com Optuna
7. **Interpretabilidade**: SHAP e LIME
8. **Relatórios**: Geração de documentação

### 3. Diagrama de Componentes (`component_diagram.puml`)
**Descrição**: Visão de alto nível da arquitetura do sistema, mostrando componentes e suas interações.

**Características**:
- ✅ **Arquitetura de Alto Nível**: Visão geral do sistema
- ✅ **Componentes**: Agrupamento lógico de funcionalidades
- ✅ **Interfaces**: Contratos entre componentes
- ✅ **Sistemas Externos**: MLflow, File System, Browser
- ✅ **Fluxo de Dados**: Direção das informações

**Camadas Representadas**:
- **View Layer**: Interface de usuário
- **Controller Layer**: Orquestração
- **Model Layer**: Lógica de negócio
- **External Systems**: Sistemas externos

## 🛠️ Como Usar os Diagramas

### Pré-requisitos
- PlantUML instalado ou plugin para IDE
- Java Runtime Environment (JRE)

### Visualização Online
1. Acesse [PlantUML Online Server](http://www.plantuml.com/plantuml/uml/)
2. Cole o conteúdo do arquivo `.puml`
3. O diagrama será gerado automaticamente

### Visualização Local
```bash
# Instalar PlantUML
java -jar plantuml.jar class_diagram.puml
java -jar plantuml.jar sequence_diagram.puml
java -jar plantuml.jar component_diagram.puml
```

### IDEs Suportadas
- **VS Code**: Plugin PlantUML
- **IntelliJ IDEA**: Plugin PlantUML
- **Eclipse**: Plugin PlantUML
- **PyCharm**: Plugin PlantUML

## 📋 Legenda dos Diagramas

### Relacionamentos (Diagrama de Classes)
- `-->` : Dependência/Associação
- `--|>` : Herança
- `*--` : Composição
- `o--` : Agregação

### Ativação (Diagrama de Sequência)
- `activate` : Início de execução
- `deactivate` : Fim de execução
- `alt` : Condicional
- `else` : Alternativa

### Interfaces (Diagrama de Componentes)
- `interface` : Contrato entre componentes
- `component` : Componente do sistema
- `package` : Agrupamento lógico

## 🔍 Análise dos Diagramas

### Arquitetura MVC
Os diagramas demonstram claramente a implementação do padrão MVC:

#### Model
- **Responsabilidade**: Lógica de negócio e processamento
- **Componentes**: Preprocessing, FeatureEngineering, Optimization, Interpretability, Tools
- **Características**: Baixo acoplamento, alta coesão

#### View
- **Responsabilidade**: Interface de usuário
- **Componente**: CLI Interface (Main)
- **Características**: Simples, focado na apresentação

#### Controller
- **Responsabilidade**: Orquestração e coordenação
- **Componente**: Core Controller
- **Características**: Centraliza o fluxo de execução

### Fluxo de Dados
1. **Entrada**: Dataset via CLI
2. **Validação**: Verificação de integridade
3. **Processamento**: Pipeline de ML
4. **Saída**: Modelos, relatórios e visualizações

### Pontos de Extensão
- **Novos Algoritmos**: Adicionar em AlgoLib
- **Novos Métodos de Feature Selection**: Estender FeatureSelection
- **Novos Métodos de Interpretabilidade**: Estender ModelExplanation

## 📈 Benefícios dos Diagramas

### Para Desenvolvedores
- ✅ **Entendimento Rápido**: Visão clara da arquitetura
- ✅ **Manutenção**: Identificação de dependências
- ✅ **Extensibilidade**: Pontos de modificação claros
- ✅ **Documentação**: Referência técnica atualizada

### Para Usuários
- ✅ **Transparência**: Entendimento do funcionamento
- ✅ **Confiança**: Arquitetura bem estruturada
- ✅ **Suporte**: Base para troubleshooting

### Para Stakeholders
- ✅ **Visão Geral**: Arquitetura de alto nível
- ✅ **Qualidade**: Padrões bem definidos
- ✅ **Escalabilidade**: Estrutura preparada para crescimento

## 🔄 Atualização dos Diagramas

### Quando Atualizar
- Adição de novas funcionalidades
- Modificação da arquitetura
- Mudanças nas dependências
- Refatoração significativa

### Como Atualizar
1. Modificar o arquivo `.puml` correspondente
2. Regenerar o diagrama
3. Atualizar esta documentação
4. Commit das mudanças

## 📚 Referências

- [PlantUML Documentation](https://plantuml.com/)
- [UML Diagrams](https://www.uml-diagrams.org/)
- [MVC Pattern](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller)
- [Software Architecture Patterns](https://martinfowler.com/articles/enterprisePatterns.html)

---

**Nota**: Estes diagramas são gerados automaticamente a partir do código fonte e devem ser mantidos atualizados conforme o sistema evolui. 