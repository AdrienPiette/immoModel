import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

# Function to load the model
def load_model():
    model = CatBoostRegressor()
    model.load_model('catboost_model.cbm')
    return model

# Function to preprocess user input
def preprocess_input(user_input):
    categorical_features = ['furnished', 'district', 'subtypeofproperty', 'typeofsale', 'peb', 'province', 'region']
    user_input_df = pd.DataFrame([user_input])
    dummies = pd.get_dummies(user_input_df[categorical_features])
    user_input_df = pd.concat([user_input_df, dummies], axis=1)
    user_input_df = user_input_df.drop(categorical_features, axis=1)
    return user_input_df

# Main function to run the Streamlit app
def main():
    st.title("Real Estate Price Prediction")

    # Predefined options for each feature
    furnished_options = ["Yes", "No"]
    district_options = ["District1", "District2", "District3"]  # Add all possible districts
    subtypeofproperty_options = ["Apartment", "House", "Villa"]  # Add all possible subtypes
    typeofsale_options = ["Sale", "Rent"]  # Add all possible types of sale
    peb_options = ["A", "B", "C", "D", "E", "F", "G"]  # Add all possible PEB ratings
    province_options = ["Province1", "Province2", "Province3"]  # Add all possible provinces
    region_options = ["Region1", "Region2", "Region3"]  # Add all possible regions

    # User input fields
    furnished = st.selectbox("Furnished", furnished_options)
    district = st.selectbox("District", district_options)
    subtypeofproperty = st.selectbox("Subtype of Property", subtypeofproperty_options)
    typeofsale = st.selectbox("Type of Sale", typeofsale_options)
    peb = st.selectbox("PEB", peb_options)
    province = st.selectbox("Province", province_options)
    region = st.selectbox("Region", region_options)

    # Collect user input into a dictionary
    user_input = {
        'furnished': furnished,
        'district': district,
        'subtypeofproperty': subtypeofproperty,
        'typeofsale': typeofsale,
        'peb': peb,
        'province': province,
        'region': region
    }

    # Preprocess the user input
    user_input_df = preprocess_input(user_input)

    # Load the model
    model = load_model()

    # Predict button
    if st.button("Predict"):
        prediction = model.predict(user_input_df)
        st.write(f"Predicted Price: {prediction[0]}")

if __name__ == '__main__':
    main()