import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the scaled dataset
df = pd.read_csv("Scaled_Train_Data.csv")

# Define features and target
target_col = "Sepssis"  # Ensure this is the correct name
X = df.drop(columns=[target_col])  # Features
y = df[target_col]  # Target

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X)

# Convert back to DataFrame
df_pca = pd.DataFrame(X_pca, columns=[f"PCA_{i+1}" for i in range(X_pca.shape[1])])
df_pca[target_col] = y  # Add back target column

# Save the PCA transformed dataset
df_pca.to_csv("PCA_Train_Data.csv", index=False)
print(f"PCA reduced dataset saved as 'PCA_Train_Data.csv' with {X_pca.shape[1]} features.")

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Explained Variance")
plt.show()


def impute_selected_columns(df, columns_to_impute, imputation_method, return_imputed_values=False):
    """
    Imputes missing values in selected columns of a DataFrame using a single imputation method.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        columns_to_impute (list): A list of column names to impute.
        imputation_method (str): The imputation method to use ('mean', 'median', or 'mode').
        return_imputed_values (bool): If True, returns a dictionary of imputed values for each column.

    Returns:
        tuple: (pd.DataFrame, dict) if return_imputed_values is True, otherwise pd.DataFrame.
    """
    df_imputed = df.copy()  # Create a copy to avoid modifying the original DataFrame.
    imputed_values = {}
for column_name in columns_to_impute:
        if column_name not in df_imputed.columns:
            print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping.")
            continue  # Skip to the next column

        if df_imputed[column_name].isnull().any():  # Only impute if there are missing values.
            if imputation_method == 'mean':
                fill_value = df_imputed[column_name].mean()
            elif imputation_method == 'median':
                fill_value = df_imputed[column_name].median()
            elif imputation_method == 'mode':
                fill_value = df_imputed[column_name].mode()[0]
            else:
                print(f"Error: Invalid imputation method '{imputation_method}' for column '{column_name}'. Use 'mean', 'median', or 'mode'.")
                return df_imputed, imputed_values if return_imputed_values else df_imputed

            df_imputed[column_name] = df_imputed[column_name].fillna(fill_value)
            imputed_values[column_name] = fill_value

    if return_imputed_values:
        return df_imputed, imputed_values
    else:
        return df_imputed