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


    for columns in df.columns:
        if df[columns].dtype == 'int' or df[columns].dtype == 'float':
            user_input[columns] = st.number_input(f"Enter {columns}", value=0)
        else:
            user_input[columns] = st.selectbox(f"Select {columns}", df[columns].unique())
    
    user_input_df = pd.DataFrame([user_input])
    st.write(user_input_df)
    

    if st.button("Predict"):
        model = CatBoostRegressor()
        model.load_model('catboost_model')
        prediction = model.predict(user_input_df)
        st.write(f"Predicted Price: {prediction}")

if __name__ == '__main__':  
    main()


