import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
import pickle
from collections import Counter

# 🔹 File paths
train_file = "Train_Data.csv"
test_file = "Test_Data.csv"

# 🔹 Load Train & Test datasets
df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

# 🔹 Ensure correct column name
target_col = "Sepssis"
if target_col not in df_train.columns:
    raise ValueError(f"Target column '{target_col}' not found in Train dataset.")

print(f"Target Column: {target_col}")

# 🔹 Define Feature Columns
feature_cols = df_train.columns.to_list()
feature_cols.remove(target_col)  # Remove target from feature list
print(f"Feature Columns: {feature_cols}")

# ===========================
# 🔹 Missing Values Handling
# ===========================
num_cols = df_train.select_dtypes(include=['number']).columns

# Fill missing values in Train & Test using Median
df_train[num_cols] = df_train[num_cols].fillna(df_train[num_cols].median())
df_test[num_cols] = df_test[num_cols].fillna(df_test[num_cols].median())

# ===========================
# 🔹 Convert Target Variable to Binary
# ===========================
df_train[target_col] = df_train[target_col].map({'Positive': 1, 'Negative': 0})

# ===========================
# 🔹 Outlier Detection (Train Data only)
# ===========================
lof = LocalOutlierFactor(n_neighbors=20)
outlier_flags = lof.fit_predict(df_train[num_cols])
df_train = df_train[outlier_flags == 1]  # Keep only non-outliers

# ===========================
# 🔹 Save Pre-Scaling & Pre-PCA Data
# ===========================
df_train.to_csv("Pre_Scaling_PCA_Train.csv", index=False)
df_test.to_csv("Pre_Scaling_PCA_Test.csv", index=False)
print("\n✅ Pre-Scaling & Pre-PCA datasets saved!")

# ===========================
# 🔹 Apply MinMax Scaling (Using Train Data Scalers)
# ===========================
scaler_dict = {}

for col in feature_cols:
    scaler = MinMaxScaler()
    df_train[[col]] = scaler.fit_transform(df_train[[col]])  # Train: Fit & Transform
    df_test[[col]] = scaler.transform(df_test[[col]])  # Test: Only Transform
    scaler_dict[col] = scaler  # Store Scalers

# Save Scaled Datasets
df_train.to_csv("Scaled_Train_Data.csv", index=False)
df_test.to_csv("Scaled_Test_Data.csv", index=False)

# Save Scalers for Future Use
with open("MinMax_Scalers.pkl", "wb") as f:
    pickle.dump(scaler_dict, f)

print("\n✅ MinMax Scaling applied & saved!")

# ===========================
# 🔹 Feature Importance using Random Forest
# ===========================
X_train = df_train.drop(columns=[target_col])
y_train = df_train[target_col]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

feature_importance = pd.DataFrame({"Feature": X_train.columns, "Importance": rf.feature_importances_})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print("\n--- Feature Importance ---")
print(feature_importance)

# ===========================
# 🔹 Apply Polynomial Features
# ===========================
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features_train = poly.fit_transform(df_train[feature_cols])
poly_features_test = poly.transform(df_test[feature_cols])

# Convert to DataFrames
poly_train_df = pd.DataFrame(poly_features_train, columns=poly.get_feature_names_out(feature_cols))
poly_test_df = pd.DataFrame(poly_features_test, columns=poly.get_feature_names_out(feature_cols))

# Concatenate with Original Data
df_train = pd.concat([df_train, poly_train_df], axis=1)
df_test = pd.concat([df_test, poly_test_df], axis=1)

# Save Final Scaled & PCA-ready Datasets
df_train.to_csv("Final_Train_Data.csv", index=False)
df_test.to_csv("Final_Test_Data.csv", index=False)

print("\n✅ Polynomial Features Added & Final Data Saved!")
