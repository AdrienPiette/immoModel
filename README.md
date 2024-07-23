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
   
   ![Data Preparation](https://media.giphy.com/media/3o6MboRJmXSeL5sK7W/giphy.gif)

2. **Model Training**
   - 🧪 Split the dataset into training and test sets (80% train, 20% test).
   - 📈 Trained a CatBoostRegressor model on the training set.
   - 🔍 Performed Grid Search to optimize hyperparameters.

   ![Model Training](https://media.giphy.com/media/3o6Zt7Uzh8a6p15i2U/giphy.gif)

3. **Model Evaluation**
   - 🧮 Evaluated the model using various performance metrics.
   - 🎯 Fine-tuned the model based on evaluation results.

   ![Model Evaluation](https://media.giphy.com/media/3o7btWoYp65N84s8Na/giphy.gif)

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
- 🔢 **Iterations:** XXXX
- 🔢 **Learning Rate:** X.XX
- 🔢 **Depth:** XX
- 🔢 **L2 Leaf Reg:** X.X

## Results
The model demonstrated high performance with low error metrics:
- 📏 **MAE:** X.XX
- 📏 **MSE:** X.XX
- 📏 **RMSE:** X.XX
- 📈 **R^2 Score:** X.XX

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

## Contributors
- Your Name - [@yourusername](https://github.com/yourusername) ✨

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to add more GIFs and emojis where you see fit! You can use online tools like Giphy to find relevant GIFs and GitHub’s emoji cheat sheet for more emojis.
