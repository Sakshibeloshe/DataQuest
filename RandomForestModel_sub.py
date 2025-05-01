import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# 🔹 Load Processed Train & Test Data
df_train = pd.read_csv("Final_Train_Data.csv")  # Train Data
df_test = pd.read_csv("Final_Test_Data.csv")  # Test Data (for submission)

# 🔹 Ensure Correct Column Name
target_col = "Sepssis"
if target_col not in df_train.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

# 🔹 Check for Missing Values in Target Column
if df_train[target_col].isnull().sum() > 0:
    print(f"\n⚠ Warning: Missing values detected in '{target_col}', filling with most common value.")
    df_train[target_col].fillna(df_train[target_col].mode()[0], inplace=True)  # Fill NaN with mode

# 🔹 Convert Target Variable to Numeric (0 & 1)
df_train[target_col] = df_train[target_col].map({"Positive": 1, "Negative": 0}).astype(int)

# 🔹 Define Features & Target
X_train = df_train.drop(columns=[target_col])  # Features
y_train = df_train[target_col]  # Target

# 🔹 Ensure No NaN in Target
if y_train.isnull().sum() > 0:
    raise ValueError(f"❌ Target variable '{target_col}' still contains NaN values!")

# 🔹 Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ==========================
# 🔹 Predict on Test Data (For Submission)
# ==========================
X_test = df_test  # Test data (No target column)
y_pred_test = rf.predict(X_test)

# 🔹 Convert 0 → "Negative" and 1 → "Positive"
y_pred_test_labels = np.where(y_pred_test == 1, "Positive", "Negative")

# 🔹 Save Predictions in Submission Format
submission_df = pd.DataFrame(y_pred_test_labels, columns=["Sepssis"])
submission_df.to_csv("submission.csv", index=False)

print("\n✅ Submission file 'submission.csv' created successfully!")
