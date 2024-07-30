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
    furnished_options = ["Yes", "No"]

    district_options = ["district_Aalst", "district_Antwerp","district_Arlon" , "district_Ath",
                        "district_Bastogne","district_Brugge","district_Brussels","district_Charleroi",
                        "district_Dendermonde","district_Diksmuide","district_Dinant","district_Eeklo",
                        "district_Gent","district_Halle-Vilvoorde","district_Hasselt","district_Huy",
                        "district_Ieper" ,"district_Kortrijk","district_Leuven","district_Liège",
                        "district_Maaseik","district_Marche-en-Famenne","district_Mechelen","district_Mons",
                        "district_Mouscron","district_Namur","district_Neufchâteau","district_Nivelles",
                        "district_Oostend","district_Oudenaarde","district_Philippeville","district_Sint-Niklaas",
                        "district_Roeselare","district_Soignies","district_Thuin","district_Tielt",
                        "district_Tongeren","district_Tournai","district_Turnhout","district_Verviers",
                        "district_Veurne","district_Virton","district_Waremme"]  

    subtypeofproperty_options = ["subtypeofproperty_apartment", "subtypeofproperty_apartement_block","subtypeofproperty_bungalow","subtypeofproperty_castle",
                                 "subtypeofproperty_chalet","subtypeofproperty_country_cottage","subtypeofproperty_duplex","subtypeofproperty_exeptional_property",
                                 "subtypeofproperty_farmhouse","subtypeofproperty_flat_studio","subtypeofproperty_ground_floor","subtypeofproperty_house",
                                 "subtypeofproperty_kot","subtypeofproperty_loft","subtypeofproperty_mansion","subtypeofproperty_manor_house",
                                 "subtypeofproperty_mixed_use_building","subtypeofproperty_other_property","subtypeofproperty_penthouse","subtypeofproperty_service_flat",
                                 "subtypeofproperty_pavilion","subtypeofproperty", "subtypeofproperty_town_house","subtypeofproperty_triplex","subtypeofproperty_villa",]  

    peb_options = ["peb_A", "peb_A+", "peb_A++", 
                   "peb_A_A+", "peb_B", "peb_B_A", 
                   "peb_C","peb_D", "peb_E","peb_E_D","peb_F","peb_F_C",
                   "peb_F_D","peb_F_E","peb_G"] 
    
    province_options = ["province_Antwerp", "province_Brussels", "province_East Flanders",
                        "province_Flemish Brabant","province_Hainaut", "province_Limburg",
                        "province_Liège","province_Luxembourg","province_Namur",
                        "province_Walloon Brabant", "province_West Flanders" ] 
    
    region_options = ["region_Brussels", "region_Flanders", "region_Wallonie"] 
    bathroom_options = ["1", "2", "3", "4", "5", "6"]
    bedroom_options = ["1", "2", "3", "4", "5", "6"]
    garden_options = ["Yes", "No"]
    kitchen_options = ["Yes", "No"]
    living_area = []
    number_of_facades = ["1", "2", "3", "4"]
    showercount_options = ["1", "2", "3"]
    stateofbuilding_options = ["stateofbuilding_as_new", "stateofbuilding_good", "stateofbuilding_just_renovated",
                               "stateofbuilding_to_be_done_up", "stateofbuilding_to_restore"]
    surfaceofplot = []
    swimmingpool_options = ["Yes", "No"]
    terrace_options = ["Yes", "No"]
    toilet_options = ["1", "2", "3", "4", "5", "6"]
    

    # User input fields
    furnished = st.selectbox("Furnished", furnished_options)
    district = st.selectbox("District", district_options)
    subtypeofproperty = st.selectbox("Subtype of Property", subtypeofproperty_options)
    peb = st.selectbox("PEB", peb_options)
    province = st.selectbox("Province", province_options)
    region = st.selectbox("Region", region_options)
    bathroom = st.selectbox("Bathroom", bathroom_options)
    bedroom = st.selectbox("Bedroom", bedroom_options)
    garden = st.selectbox("Garden", garden_options)
    kitchen = st.selectbox("Kitchen", kitchen_options)
    living_area = st.number_input("Living Area")
    number_of_facades = st.selectbox("Number of Facades", number_of_facades)
    showercount = st.selectbox("Shower Count", showercount_options) 
    stateofbuilding = st.selectbox("State of Building", stateofbuilding_options)
    surfaceofplot = st.number_input("Surface of Plot")
    swimmingpool = st.selectbox("Swimming Pool", swimmingpool_options)
    terrace = st.selectbox("Terrace", terrace_options)
    toilet = st.selectbox("Toilet", toilet_options)


    # Collect user input into a dictionary
    user_input = {
        'furnished': furnished,
        'district': district,
        'subtypeofproperty': subtypeofproperty,
        'peb': peb,
        'province': province,
        'region': region,
        'bathroom': bathroom,
        'bedroom': bedroom,
        'garden': garden,
        'kitchen': kitchen,
        'living_area': living_area,
        'number_of_facades': number_of_facades,
        'showercount': showercount,
        'stateofbuilding': stateofbuilding,
        'surfaceofplot': surfaceofplot,
        'swimmingpool': swimmingpool,
        'terrace': terrace,
        'toilet': toilet

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
    