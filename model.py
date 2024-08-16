import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load your dataset
df = pd.read_csv('cleaned_data.csv')

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_encode = ['District', 'Furnished', 'SubtypeOfProperty', 'PEB', 'Province', 'Region',
                         'SwimmingPool', 'Terrace', 'Kitchen', 'Garden', 'TypeOfProperty', 
                         'StateOfBuilding', 'TypeOfSale']
    
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = one_hot_encoder.fit_transform(df[columns_to_encode])
    encoded_df = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out(columns_to_encode))
    
    # Save the OneHotEncoder for future use
    with open('onehotencoder.pkl', 'wb') as f:
        pickle.dump(one_hot_encoder, f)
    
    return pd.concat([df.drop(columns=columns_to_encode), encoded_df], axis=1)

# Encode categorical features
df_encoded = encode_categorical_features(df)

# Separate features and target variable
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    # Define hyperparameters to be tuned
    param = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'verbose': 0  # Suppress CatBoost's output
    }
    
    # Initialize the model with the parameters from the trial
    model = CatBoostRegressor(**param)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, predictions)
    
    # Return MSE as the metric to minimize
    return mse

# Create a study object and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Number of trials can be adjusted

# Print the best parameters and best score
print("Best parameters: ", study.best_params)
print("Best MSE: ", study.best_value)

# Train final model with the best parameters
best_params = study.best_params
final_model = CatBoostRegressor(**best_params)
final_model.fit(X_train, y_train)

# Make final predictions and evaluate
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_mae = mean_absolute_error(y_test, final_predictions)
final_r2 = r2_score(y_test, final_predictions)

print('Final Mean Squared Error:', final_mse)
print('Final Mean Absolute Error:', final_mae)
print('Final R-squared:', final_r2)

# Save the final model
final_model.save_model('catboost_model_optuna.cbm')
print("Final model has been saved.")
