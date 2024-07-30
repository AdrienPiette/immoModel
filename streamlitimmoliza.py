import streamlit as st
import pandas as pd

from catboost import CatBoostRegressor
import numpy as np


model = open('catboost_model', 'rb')
df = pd.read_csv('final_dataset.csv')
df_drop = df.drop('price', axis=1)



def main ():

    st.title("House Price Prediction")  
    st.sidebar.title("House Price Prediction")
    st.markdown("Welcome to House Price Prediction")
    st.sidebar.markdown("Welcome to House Price Prediction")

    # variables for the model
    user_input = {}
    user_input['bedrooms'] = st.sidebar.slider('Bedrooms', 1, 5, 1)
    user_input['bathrooms'] = st.sidebar.slider('Bathrooms', 1, 5, 1)  
    user_input['sqft_living'] = st.sidebar.slider('Square Feet Living', 0, 100, 290)    
    









    user_input_df = pd.DataFrame([user_input])
    
    if st.button("Predict"):
        model = CatBoostRegressor()
        model.load_model('catboost_model')
        prediction = model.predict(user_input_df)
        st.write(f"Predicted Price: {prediction}")

if __name__ == '__main__':  
    main()


