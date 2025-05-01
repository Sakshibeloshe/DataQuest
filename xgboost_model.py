#XG BOOST
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🔹 Load the PCA-transformed dataset
df_pca = pd.read_csv("PCA_Train_Data.csv")

# 🔹 Define features and target
target_col = "Sepssis"  # Ensure the column name is correct
X = df_pca.drop(columns=[target_col])  # Features
y = df_pca[target_col]  # Target

# 🔹 Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from xgboost import XGBClassifier

# 🔹 Train XGBoost Model
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_train, y_train)

# 🔹 Make Predictions
y_pred_xgb = xgb.predict(X_test)

# 🔹 Evaluate Model Performance
print("\n✅ XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("\n📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
