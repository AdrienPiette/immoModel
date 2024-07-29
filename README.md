# 🏡 Real Estate Price Prediction

## Project Overview
🎯 **Objective:** Predict real estate prices using various machine learning techniques.
📊 **Data Source:** Cleaned real estate dataset with various features.
🔧 **Techniques Used:**
- Data Cleaning
- Feature Encoding
- Imputation of Missing Values
- Model Training & Evaluation

## Workflow
The project workflow consists of the following steps:

1. **Data Preparation**
   - 🧹 **Data Cleaning:** Dropped irrelevant columns and handled missing values.
   - 🏷️ **Feature Encoding:** Transformed categorical variables using OneHotEncoder.
   - 🔄 **Imputation:** Applied KNN Imputation to handle missing values.
   
   ![Data Preparation](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExbTZ1cThpN2w3dGY4cXN0bm01dWF3amdnMmQ4MzB4OGkxa2FzenB6NSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/hW8zWb9GKoN6YRreJM/giphy.webp)

2. **Model Training**
   - 🧪 Split the dataset into training and test sets (80% train, 20% test).
   - 📈 Trained a CatBoostRegressor model on the training set.
   - 🔍 Performed Grid Search to optimize hyperparameters.

   ![Model Training](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExY3J0eGV4dzhqcnFpNnQ4OXJrbnJyaHN3d3c3bWQ0N2E2a2l4eTF5aCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/d2ZfqZY5eSCR0rza/giphy.gif)

3. **Model Evaluation**
   - 🧮 Evaluated the model using various performance metrics.
   - 🎯 Fine-tuned the model based on evaluation results.

   ![Model Evaluation](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaHVraHBycjhsaTV5M25xZ2c1OTg4MXE3YXRhbTB6YXczOTV2anR2dSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Lk5BzpifzeI3KYm7n0/giphy.gif)

## Data Preparation
The data cleaning process involved:
- 🗑️ Dropping irrelevant columns such as `country`, `fireplace`, `monthlycharges`, `locality`, `propertyid`, `constructionyear`, `furnished`, and `roomcount`.
- 🔄 Encoding categorical features (`district`, `floodingzone`, `subtypeofproperty`, `typeofsale`, `peb`, `province`, `region`) using OneHotEncoder.
- 🔧 Imputing missing values using KNNImputer.

## Model Training
The CatBoostRegressor model was trained on the processed dataset. The training process involved:
- 📊 Splitting the data into training and test sets.
- 🔄 Imputing missing values in the training and test sets.
- 🔍 Performing Grid Search to find the best hyperparameters.

## Model Evaluation
The model was evaluated using the following metrics:
- 📏 **Mean Absolute Error (MAE)**
- 📏 **Mean Squared Error (MSE)**
- 📏 **Root Mean Squared Error (RMSE)**
- 📏 **R^2 Score**

**Best Model Parameters:**

**Parameter Grid:**

- iterations: [100, 300, 500, 700],
- learning_rate: [0.001, 0.01, 0.05],
- depth: [3, 5, 7, 10, 12],
- l2_leaf_reg: [0.1, 1, 3, 5, 10]


## Results

The model demonstrated high performance with low error metrics:

- 📏 MAE: 47856.00886071866
- 📏 MSE: 4370635560.1221075
- 📏 RMSE: 66110.78248003201
- 📈 R^2 Score: 0.7846459907700529

## Conclusion & Future Work
The CatBoostRegressor model successfully predicts real estate prices with high accuracy. Future work will focus on:
- 🔍 Exploring additional feature engineering techniques.
- 🧪 Testing other regression algorithms.
- 🚀 Deploying the model for real-world applications.

## Installation
To run the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/real-estate-price-prediction.git
    ```
2. Navigate to the project directory:
    ```sh
    cd real-estate-price-prediction
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Prepare your dataset and place it in the project directory.
2. Run the data preparation script:
    ```sh
    python data_preparation.py
    ```
3. Train the model:
    ```sh
    python train_model.py
    ```
4. Evaluate the model:
    ```sh
    python evaluate_model.py
    ```

---

Feel free to add more GIFs and emojis where you see fit! You can use online tools like Giphy to find relevant GIFs and GitHub’s emoji cheat sheet for more emojis.
