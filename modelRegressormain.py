import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(file_path)
    print(df.shape)
    print(df.info())
    print(df.describe())
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by dropping irrelevant columns.

    Args:
        df (pd.DataFrame): The original dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.drop(columns=['country', 'fireplace', 'monthlycharges', 'locality', 'propertyid', 'constructionyear'])
    return df

def encode_categorical(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Encode categorical variables using OneHotEncoder.

    Args:
        df (pd.DataFrame): Dataframe containing categorical variables.
        columns (list): List of columns to be encoded.

    Returns:
        pd.DataFrame: Dataframe with encoded variables.
    """
    one = OneHotEncoder()
    encoded_data = one.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=one.get_feature_names_out(columns))
    df_final = pd.concat([df.drop(columns=columns), encoded_df], axis=1)
    return df_final

def calculate_correlations(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the correlation between features and the target variable.

    Args:
        df (pd.DataFrame): Dataframe with features and target variable.

    Returns:
        pd.Series: Correlation values.
    """
    correlations = df.drop(columns=['price']).corrwith(df['price'])
    print(correlations)
    return correlations

def split_data(df: pd.DataFrame, target: str, test_size: float, random_state: int) -> tuple:
    """
    Split the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): The dataframe to split.
        target (str): The target column name.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: Split data (X_train, X_test, y_train, y_test).
    """
    y = np.array(df[target])
    X = np.array(df.drop(columns=[target]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    return X_train, X_test, y_train, y_test

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with the median of each column.

    Args:
        df (pd.DataFrame): The dataframe with missing values.

    Returns:
        pd.DataFrame: Dataframe with imputed values.
    """
    for col in df.columns:
        df[col] = df[col].fillna(df[col].median())
    return df

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> CatBoostRegressor:
    """
    Train a CatBoost Regressor model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.

    Returns:
        CatBoostRegressor: Trained model.
    """
    model = CatBoostRegressor(random_state=42, verbose=0)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: CatBoostRegressor, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluate the model and print evaluation metrics.

    Args:
        model (CatBoostRegressor): The trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test target.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R^2 Score: {r2}")

def main():
    """
    Main function to execute the data processing, model training, and evaluation pipeline.
    """
    # Load and clean dataset
    df = load_dataset('cleaned_dataset.csv')
    df_clean = clean_data(df.copy())

    # Encode categorical variables
    columns_to_encode = ['district', 'floodingzone', 'subtypeofproperty', 'typeofsale', 'peb', 'province', 'region']
    df_final = encode_categorical(df_clean, columns_to_encode)

    # Calculate correlations
    calculate_correlations(df_final)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df_final, 'price', test_size=0.2, random_state=42)

    # Impute missing values
    X_train_df_clean = impute_missing_values(pd.DataFrame(X_train))
    X_test_df_clean = impute_missing_values(pd.DataFrame(X_test))
    y_train_df_clean = impute_missing_values(pd.DataFrame(y_train))
    y_test_df_clean = impute_missing_values(pd.DataFrame(y_test))

    # Train model
    model = train_model(X_train_df_clean.to_numpy(), y_train_df_clean.to_numpy().ravel())

    # Evaluate model
    evaluate_model(model, X_test_df_clean.to_numpy(), y_test_df_clean.to_numpy().ravel())

if __name__ == "__main__":
    main()

