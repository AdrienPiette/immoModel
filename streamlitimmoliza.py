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
    categorical_features = ['furnished', 'district', 'subtypeofproperty','peb', 'province', 'region',
                            'stateofbuilding', 'swimmingpool', 'terrace','kitchen', 'garden']
    user_input_df = pd.DataFrame([user_input])
    dummies = pd.get_dummies(user_input_df[categorical_features])
    user_input_df = pd.concat([user_input_df, dummies], axis=1)
    user_input_df = user_input_df.drop(categorical_features, axis=1)
    return user_input_df

# Main function to run the Streamlit app
def main():
    st.title("Real Estate Price Prediction")

    # Predefined options for each feature
    bathroom_options = ["1", "2", "3", "4", "5", "6"]
    bedroom_options = ["1", "2", "3", "4", "5", "6"]
    construction_year = st.number_input("Construction Year")
    country_options = st.selectbox("Country", country_options)
    district_options = st.selectbox("District", district_options)
    fire_place_options = st.selectbox("Fire Place", fire_place_options)
    flooding_zone_options = st.selectbox("Flooding Zone", flooding_zone_options)
    furnished_options = st.selectbox("Furnished", furnished_options)
    garden_options = st.selectbox("Garden", garden_options)
    kitchen_options = st.selectbox("Kitchen", kitchen_options)
    living_area = st.number_input("Living Area")
    locality_options = st.selectbox("Locality", locality_options)
    monthly_charges = st.number_input("Monthly Charges")
    number_of_facades = st.selectbox("Number of Facades", number_of_facades)
    peb_options = st.selectbox("PEB", peb_options)
    postal_code = st.number_input("Postal Code")
    property_id = st.number_input("Property ID")
    province_options = st.selectbox("Province", province_options)
    region_options = st.selectbox("Region", region_options)
    room_count_options = st.selectbox("Room Count", room_count_options)
    showercount_options = st.selectbox("Shower Count", showercount_options)
    stateofbuilding_options = st.selectbox("State of Building", stateofbuilding_options)
    subtypeofproperty_options = st.selectbox("Subtype of Property", subtypeofproperty_options)
    surfaceofplot = st.number_input("Surface of Plot")
    swimmingpool_options = st.selectbox("Swimming Pool", swimmingpool_options)
    terrace_options = st.selectbox("Terrace", terrace_options)
    toilet_options = st.selectbox("Toilet", toilet_options)
    typeofproperty_options = st.selectbox("Type of Property", typeofproperty_options)
    typeofsale_options = st.selectbox("Type of Sale", typeofsale_options)


    # Collect user input into a dictionary
    user_input = {
        'bathroomcount': bathroom_options,
        'bedroomrcount': bedroom_options,
        'constructionyear':construction_year,
        'country':country_options,
        'district': district_options,
        'fireplace': fire_place_options,
        'floodingzone': flooding_zone_options,
        'furnished': furnished_options,
        'garden': garden_options,
        'kitchen': kitchen_options,
        'livingarea': living_area,
        'locality': locality_options,
        'monthlycharges': monthly_charges,
        'numberoffacades': number_of_facades,
        'peb': peb_options,
        'postalcode': postal_code,
        'propertyid': property_id,
        'province': province_options,
        'region': region_options,
        'roomcount': room_count_options,
        'showercount': showercount_options,
        'stateofbuilding': stateofbuilding_options,
        'subtypeofproperty': subtypeofproperty_options,
        'surfaceofplot': surfaceofplot,
        'swimmingpool': swimmingpool_options,
        'terrace': terrace_options,
        'toiletcount': toilet_options,
        'typeofproperty': typeofproperty_options,
        'typeofsale': typeofsale_options
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

    

