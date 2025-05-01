
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter



# Load dataset
file_path = "Train_Data.csv"  # Update if needed
df = pd.read_csv(file_path)

# Ensure correct column name
target_col = "Sepssis"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

print(f"Target Column: {target_col}")

# Feature columns
feature_cols = df.columns.to_list()
feature_cols.remove(target_col)
print(f"Feature Columns: {feature_cols}")

# 🔹 Check Missing Values
print("\n--- Missing Values Before Cleaning ---")
print(df.isnull().sum())

# 🔹 Fill Missing Values for Numerical Columns (Using Median)
num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 🔹 Convert Target Variable to Binary
df[target_col] = df[target_col].map({'Positive': 1, 'Negative': 0})

# 🔹 Outlier Detection & Removal using Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20)
outlier_flags = lof.fit_predict(df[num_cols])
df = df[outlier_flags == 1]  # Keep only non-outliers

# 🔹 Save the Cleaned Data Before Scaling & PCA
pre_scaling_pca_path = "Pre_Scaling_PCA_Data.csv"
df.to_csv(pre_scaling_pca_path, index=False)
print(f"\n✅ Pre-Scaling & Pre-PCA dataset saved as '{pre_scaling_pca_path}'")

import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# 🔹 Load the dataset
df = pd.read_csv("Pre_Scaling_PCA_Data.csv")  # Update with the correct file name

# 🔹 Define columns to scale
minmax_cols = feature_cols  # Update with your actual column names

# 🔹 Dictionary to store scalers for each column
scaler_dict = {}

# 🔹 Apply MinMaxScaler to each column separately
for col in minmax_cols:
    scaler = MinMaxScaler()
    df[[col]] = scaler.fit_transform(df[[col]])  # Apply scaling
    scaler_dict[col] = scaler  # Save the scaler object

print(scaler_dict)

scaling_path = "Scaling_Data.csv"
df.to_csv(scaling_path, index=False)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Feature Importance using Random Forest
X = df.drop(columns=[target_col])
y = df[target_col]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print("\n--- Feature Importance ---")
print(feature_importance)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(df[feature_cols]
                                   )  # Example selected features

# Convert to DataFrame
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(feature_cols))
df = pd.concat([df, poly_df], axis=1)

scaling_PCA_path = "Scaling_PCA_Data.csv"
df.to_csv(scaling_PCA_path, index=False)
