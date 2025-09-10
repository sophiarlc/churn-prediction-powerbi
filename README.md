# churn-prediction-powerbi
# Análise e Predição de Churn com Power BI!

Este projeto tem como objetivo **analisar e prever o churn de clientes** (cancelamento) através de dashboards interativos no Power BI e aplicação de técnicas de Machine Learning.

## Objetivos do Projeto
- Identificar os principais fatores que influenciam o churn de clientes.
- Criar dashboards interativos que facilitem a análise exploratória.
- Aplicar um modelo de predição de churn e avaliar sua performance.
- Propor insights e recomendações estratégicas para retenção.


## Tecnologias Utilizadas
- **Power BI** → Dashboards e relatórios interativos
- **SQL** → Transformação e manipulação de dados
- **Microsoft Fabric** → Criação de notebooks de Machine Learning e integração com o Power BI
- **Python (via Fabric Notebooks)** → Modelagem estatística
- **CSV/Excel** → Fonte de dados inicial

## Dataset

O dataset foi disponibilizado em planilhas Excel, contendo informações sobre:

- Dados demográficos (idade, gênero, estado, etc.)  
- Perfil de consumo (serviços contratados, cobranças, contratos)  
- Status do cliente (**Stayed** / **Churned**)  
- Categoria e motivo do churn  

Planilhas principais utilizadas:
- `vw_ChurnData` → Treinamento e avaliação do modelo  
- `vw_JoinData` → Previsões em novos clientes

## Etapas do Projeto

### 1. Entendimento do Problema
O **churn (rotatividade de clientes)** impacta diretamente na receita e crescimento das empresas.  
O desafio foi **entender os fatores de cancelamento** e **criar um modelo preditivo** para antecipar clientes em risco.  

### 2. Preparação dos Dados
- Importação no **Microsoft Fabric (Lakehouse)**.  
- **Transformações SQL** para limpeza e padronização.  
- Criação de colunas derivadas (tempo de contrato, taxa de cancelamento, etc.).  
- Envio para o **Power BI** para exploração inicial.

### 3. Criação dos Dashboards
Foram desenvolvidos **3 painéis no Power BI**:  

1. **Visão Geral do Churn - Sumário**  
   - Taxa de churn total
   - Total Clientes, Total Churn, Novos Clientes
   - Visualização: Demográfico, Geográfico, Informações de pagamento e conta, Serviços, Taxa de Permanência, Razão Churn, Contrato, Faixa Etária
  

2. **Razão do Churn**  
   - Razões das quais motivaram os cancelamentos, para conexão com a visualização Total Churn por Razão

3. **Predição**  
   - Clientes em risco
   - Perfil previsto de Churn

## 🤖 4. Machine Learning com Microsoft Fabric Notebooks
Para a predição de **Churn**, utilizei o **Microsoft Fabric Notebooks** integrado ao Power BI.  
A etapa de modelagem envolveu preparação de dados, treinamento de modelo e avaliação.

---

### Importação de bibliotecas e leitura dos dados

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Carregar os dados
file_path = r"/content/Prediction_Data.xlsx"
sheet_name = 'vw_ChurnData'
data = pd.read_excel(file_path, sheet_name=sheet_name)
print(data.head())
```

### Pré-processamento

```python
# Drop de colunas irrelevantes
data = data.drop(['Customer_ID', 'Churn_Category', 'Churn_Reason'], axis=1)

# Colunas categóricas a serem codificadas
columns_to_encode = [
    'Gender', 'Married', 'State', 'Value_Deal', 'Phone_Service', 'Multiple_Lines',
    'Internet_Service', 'Internet_Type', 'Online_Security', 'Online_Backup',
    'Device_Protection_Plan', 'Premium_Support', 'Streaming_TV', 'Streaming_Movies',
    'Streaming_Music', 'Unlimited_Data', 'Contract', 'Paperless_Billing',
    'Payment_Method'
]

label_encoders = {}
for column in columns_to_encode:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Codificação da variável alvo
data['Customer_Status'] = data['Customer_Status'].map({'Stayed': 0, 'Churned': 1})
```

### Treinamento do Modelo

```python
# Divisão treino/teste
X = data.drop('Customer_Status', axis=1)
y = data['Customer_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predições
y_pred = rf_model.predict(X_test)
```

### Avaliação do Modelo

```python
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Feature Importance

```python
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(15, 6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title('Feature Importances')
plt.xlabel('Relative Importance')
plt.ylabel('Feature Names')
plt.show()
```

Gráfico Gerado:

### Predição em Novos Dados

```python
# Leitura de novos dados
sheet_name = 'vw_JoinData'
new_data = pd.read_excel(file_path, sheet_name=sheet_name)

original_data = new_data.copy()
customer_ids = new_data['Customer_ID']

# Drop de colunas irrelevantes
new_data = new_data.drop(['Customer_ID', 'Customer_Status', 'Churn_Category', 'Churn_Reason'], axis=1)

# Encoding com os encoders salvos
for column in new_data.select_dtypes(include=['object']).columns:
    new_data[column] = label_encoders[column].transform(new_data[column])

# Predições
new_predictions = rf_model.predict(new_data)

# Inserindo previsões no dataset original
original_data['Customer_Status_Predicted'] = new_predictions

# Filtrar apenas clientes previstos como churn
original_data = original_data[original_data['Customer_Status_Predicted'] == 1]

# Salvar resultados
original_data.to_csv(r"/content/Predictions.csv", index=False)
```
📂 Saída gerada: Predictions.csv contendo os clientes previstos como Churned.

### 📊 Dashboards no Power BI

Após a modelagem e geração das previsões, os dados foram integrados ao Power BI para criação de dashboards interativos.
- Visão Geral do Churn
- Distribuição de clientes por Estado e Perfil
- Análise dos principais fatores de churn
- Clientes com maior risco de cancelamento (previsão do modelo)

### 📌 Referências

 **Baseado no vídeo** [Power BI End to End Churn Analysis Portfolio Project | Power BI + SQL + Machine Learning](https://www.youtube.com/watch?v=QFDslca5AX8)  
 **Documentação de referência**: [End-to-End Churn Analysis Portfolio Project - PivotalStats](https://pivotalstats.com/end-end-churn-analysis-portfolio-project/)
