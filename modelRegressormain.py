# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna

# Load the dataset
df = pd.read_csv('cleaned_dataset.csv')

# Display the shape, info, and statistical summary of the dataset
print(df.shape)
print(df.info())
print(df.describe())

# Clean the data by dropping specified columns
def clean_data(df):
    df = df.drop(columns=['country', 'fireplace', 'monthlycharges', 'locality', 'propertyid', 'constructionyear'])
    return df

df_clean = clean_data(df.copy())
df_clean.head()

# Encode categorical variables
columns_to_encode = ['district', 'floodingzone', 'subtypeofproperty', 'typeofsale', 'peb', 'province', 'region']
data_to_encode = df_clean[columns_to_encode]

one = OneHotEncoder()
encoded_data = one.fit_transform(data_to_encode)
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=one.get_feature_names_out(columns_to_encode))

# Combine the encoded variables with the rest of the dataset
df_final = pd.concat([df_clean.drop(columns=columns_to_encode), encoded_df], axis=1)
print(df_final.shape)
print(df_final.info())
df_final.head()

# Calculate and display the correlation between features and target variable
correlations = df_final.drop(columns=['price']).corrwith(df_final['price'])
print(correlations)

# Split the data into training and testing sets
y = np.array(df_final['price'])
X = np.array(df_final.drop(columns=['price']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Clean the training data by filling missing values with the median
def clean_data(X_train_df):
    for col in X_train_df.columns:
        X_train_df[col] = X_train_df[col].fillna(X_train_df[col].median())
    return X_train_df

X_train_df = pd.DataFrame(X_train.tolist() if len(X_train.shape) > 2 else X_train)
X_train_df_clean = clean_data(X_train_df.copy())
print(X_train_df_clean.head())

# Clean the test data by filling missing values with the median
def clean_data(X_test_df):
    for col in X_test_df.columns:
        if X_test_df[col].isnull().any():
            X_test_df[col].fillna(X_test_df[col].median(), inplace=True)
    return X_test_df

X_test_df = pd.DataFrame(X_test.tolist() if len(X_test.shape) > 2 else X_test)
X_test_df_clean = clean_data(X_test_df.copy())
print(X_test_df_clean.head())

# Clean the target training data by filling missing values with the median
def clean_data(y_train_df):
    y_train_df = y_train_df.fillna({0: y_train_df[0].median()})
    return y_train_df

y_train_df = pd.DataFrame(y_train.tolist() if len(y_train.shape) > 2 else y_train)
y_train_df_clean = clean_data(y_train_df.copy())
y_train_df_clean.head()

# Clean the target test data by filling missing values with 0
def clean_data(y_test_df):
    y_test_df = y_test_df.fillna({0: 0})
    return y_test_df

y_test_df = pd.DataFrame(y_test.tolist() if len(y_test.shape) > 2 else y_test)
y_test_df_clean = clean_data(y_test_df.copy())
print(y_test_df_clean.shape)

# Train a CatBoost Regressor model
from catboost import CatBoostRegressor

model = CatBoostRegressor(random_state=42)
model.fit(X_train_df_clean, y_train_df_clean)
y_pred = model.predict(X_test_df_clean)

# Calculate and display evaluation metrics
mae = mean_absolute_error(y_test_df_clean, y_pred)
mse = mean_squared_error(y_test_df_clean, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_df_clean, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R^2 Score: {r2}")
