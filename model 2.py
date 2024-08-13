import catboost as cb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load your dataset
df = pd.read_csv('dataset.csv')
y = df['price']
X = df.drop(columns=['price'])

# Columns to encode
columns_to_encode = ['district', 'subtypeofproperty', 'peb', 'province', 'region',
                    'swimmingpool', 'terrace', 'kitchen', 'garden', 'typeofproperty']

# Initialize OneHotEncoder
one = OneHotEncoder(sparse_output=False, drop='first')

# Fit and transform the encoder on the features to be one-hot encoded
encoded_data = one.fit_transform(X[columns_to_encode])
encoded_df = pd.DataFrame(encoded_data, columns=one.get_feature_names_out(columns_to_encode))

# Save the OneHotEncoder to a pickle file
with open('onehotencoder.pkl', 'wb') as f: 
    pickle.dump(one, f)

# Concatenate the encoded features with the rest of the dataset
X_encoded = pd.concat([X.drop(columns=columns_to_encode), encoded_df], axis=1)

# Check class distribution
print("Class distribution in target variable:")
print(y.value_counts())

# Handle rare classes
min_class_count = 2  # Minimum number of instances required per class for stratification
class_counts = y.value_counts()
rare_classes = class_counts[class_counts < min_class_count].index
X_filtered = X_encoded[~y.isin(rare_classes)]
y_filtered = y[~y.isin(rare_classes)]

# Split the dataset into training and testing sets
# Use stratify only if classes are sufficiently represented
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered)

# Define the CatBoost model
model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, loss_function='MultiClass')

# Train the model
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

# Evaluate the model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Save the model
with open('catboost_model.cbm', 'wb') as f:
    pickle.dump(model, f)
  