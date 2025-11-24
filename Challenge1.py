import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # novo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sns.set_theme(style="whitegrid")
data = pd.read_csv("TMEDTREND_PUBLIC_250827.csv") #DATASET

print(" Dataset carregado com sucesso!")
print(data.head())
print("\nInformações gerais:")
print(data.info())
print("\nValores nulos por coluna:")
print(data.isnull().sum())

data = data.dropna() #Remove valores ausentes

# Conversão, caso exista colunas categóricas
if "Region" in data.columns:
    data["Region"] = data["Region"].astype("category").cat.codes

# Criar variável binária
q75 = data["Pct_Telehealth"].quantile(0.75)
data["High_Adoption"] = (data["Pct_Telehealth"] >= q75).astype(int)

print(f"\nDefinimos 'High_Adoption' como 1 se o Pct_Telehealth ≥ {q75:.2f}")

# Dados
plt.figure(figsize=(8, 5))
sns.histplot(data["Pct_Telehealth"], kde=True, bins=30)
plt.title("Distribuição do Uso de Telemedicina (%)")
plt.xlabel("Pct_Telehealth")
plt.ylabel("Frequência")
plt.tight_layout()
plt.show()

# Correlação
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação entre Variáveis")
plt.tight_layout()
plt.show()

# Separar as features e alvo
X = data.drop(columns=["Pct_Telehealth", "High_Adoption"])
y = data["High_Adoption"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =Padronização das variaveis
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treino do modelo
modelo = LogisticRegression(max_iter=2000, solver="liblinear")
modelo.fit(X_train_scaled, y_train)


y_pred = modelo.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print("\n=== Resultados ===")
print(f"Acurácia: {acc:.4f}")
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))
print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))

# Matriz de confusão
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão - Adoção de Telemedicina")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

print("Conclusão:")
print("O modelo conseguiu prever razoavelmente bem as regiões de alta adoção.")
print("Com isso, é possível identificar padrões regionais e demográficos")
print("que favorecem o uso da telemedicina, auxiliando na tomada de decisão.")
