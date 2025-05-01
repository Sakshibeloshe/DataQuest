import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# 🔹 Load the PCA-transformed dataset
df_pca = pd.read_csv("PCA_Train_Data.csv")

# 🔹 Define features and target
target_col = "Sepssis"  # Ensure the column name is correct
X = df_pca.drop(columns=[target_col])  # Features
y = df_pca[target_col]  # Target

# 🔹 Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.ensemble import RandomForestClassifier

# 🔹 Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 🔹 Make Predictions
y_pred_rf = rf.predict(X_test)

# 🔹 Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf, average="weighted")  # Weighted F1-score

# 🔹 Print Results with 10 Decimal Places
print("\n✅ Random Forest Accuracy:", f"{accuracy:.10f}")
print("\n✅ Random Forest F1-score:", f"{f1:.10f}")
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred_rf, digits=10))
print("\n📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
