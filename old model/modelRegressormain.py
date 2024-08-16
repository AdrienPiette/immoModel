import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
from typing import List
import pickle

# Load dataset
df = pd.read_csv('dataset_Aalst_none.csv')


'''def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values using KNNImputer.
    
    Args:
    - df (pd.DataFrame): DataFrame with missing values.
    
    Returns:
    - pd.DataFrame: DataFrame with imputed values.
    """
    imputer = KNNImputer(n_neighbors=5)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df'''

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features using OneHotEncoder.
    
    Args:
    - df (pd.DataFrame): The DataFrame with categorical columns to encode.
    - columns_to_encode (List[str]): List of column names to encode.
    
    Returns:
    - pd.DataFrame: DataFrame with encoded features.
    
    """
    # Define columns to encode
    columns_to_encode = ['District','Furnished', 'SubtypeOfProperty', 'PEB', 'Province', 'Region',
                     'SwimmingPool', 'Terrace', 'Kitchen', 'Garden', 'TypeOfProperty', 'StateOfBuilding', 'TypeOfSale' ]
    
    one = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = one.fit_transform(df[columns_to_encode])
    encoded_df = pd.DataFrame(encoded_data, columns=one.get_feature_names_out(columns_to_encode))

    with open('onehotencoder.pkl', 'wb') as f:
        pickle.dump(one, f)

    return pd.concat([df.drop(columns=columns_to_encode), encoded_df], axis=1)


'''# Impute missing values
df = impute_data(df)'''

# Encode categorical features
df_final = encode_categorical_features(df)


# Prepare features and target variable
y = df_final['Price']
X = df_final.drop(columns=['Price'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_evaluate_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Trains a CatBoostRegressor model, evaluates it, and prints the performance metrics.
    
    Args:
    - X_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training target values.
    - X_test (np.ndarray): Test features.
    - y_test (np.ndarray): Test target values.
    
    """
    model = CatBoostRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)                          
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R^2 Score: {r2}")

    print(df_final.columns)
    X_train.to_csv('X_train.csv', index=False)
    model.save_model('catboost_model.cbm')

# Train and evaluate the model
train_and_evaluate_model(X_train, y_train, X_test, y_test)

'''# Train and evaluate the model

train_and_evaluate_model(X_train_imputed, y_train, X_test_imputed, y_test)

def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> None:

    """
    Performs GridSearchCV to find the best hyperparameters for the model.
    
    Args:
    - X_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training target values.

    """
    param_grid = {
        'iterations': [100, 300, 500, 700],
        'learning_rate': [0.001, 0.01, 0.05],
        'depth': [3, 5, 7, 10, 12],
        'l2_leaf_reg': [0.1, 1, 3, 5, 10]
    }
    
    grid_search = GridSearchCV(
        estimator=CatBoostRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    
    best_model = grid_search.best_estimator_
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    
    print("Cross-validated scores (MAE): ", -cv_scores)
    print("Mean CV score (MAE): ", -cv_scores.mean())
    
    # Evaluate the best model on the test set

    y_pred = best_model.predict(X_test_imputed)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE (Best Model): {mae}")
    print(f"MSE (Best Model): {mse}")
    print(f"RMSE (Best Model): {rmse}")
    print(f"R^2 Score (Best Model): {r2}")

    best_model.save_model('catboost_model.cbm') 

# Perform grid search and evaluate the best model

perform_grid_search(X_train_imputed, y_train)'''