import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
import pickle

df = pd.read_csv('cleaned_dataset.csv')
# Function to load the model
def load_model():
    model = CatBoostRegressor()
    model.load_model('catboost_model.cbm')
    return model


with open('onehotencoder.plk', 'rb') as f:
    one = pickle.load(f)

columns_to_encode = ['district', 'floodingzone', 'subtypeofproperty', 'peb', 'province', 'region',
                     'stateofbuilding', 'swimmingpool', 'terrace', 'kitchen', 'garden']

one_encoding = one.transform(df[columns_to_encode])
one_encoding_df = pd.DataFrame(one_encoding, columns=one.get_feature_names_out(columns_to_encode))
df_final= pd.concat([df.drop(columns=columns_to_encode), one_encoding_df], axis=1)





# Function to preprocess user input
def preprocess_input(user_input):
    categorical_features = ['district','fireplace','floodingzone','subtypeofproperty','peb', 'province', 'region',
                            'garden','kitchen','stateofbuilding','swimmingpool','terrace','typeofproperty']

    user_input_df = pd.DataFrame([user_input])
    dummies = pd.get_dummies(user_input_df[categorical_features])
    user_input_df = pd.concat([user_input_df, dummies], axis=1)
    user_input_df = user_input_df.drop(categorical_features, axis=1)
    return user_input_df

# Main function to run the Streamlit app
def main():
    st.title("Real Estate Price Prediction")

    st.sidebar.header("Property Details")

    # Predefined options for each feature
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

    fireplace_options = ["Yes", "No"]
    floodingzone_options = ["Yes", "No"]
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
                        "province_Walloon Brabant", "province_West Flanders"]

    region_options = ["region_Brussels", "region_Flanders", "region_Wallonie"]
    garden_options = ["Yes", "No"]
    kitchen_options = ["New", "Installed", "Semi-equipped", "USA Hyper-equipped"]
    number_of_facades_options = ["1", "2", "3", "4"]
    bathroom_options = ["1", "2", "3", "4", "5", "6"]
    bedroom_options = ["1", "2", "3", "4", "5", "6"]
    showercount_options = ["1", "2", "3","4","5","6"]
    stateofbuilding_options = ["As new", "Good", "Just renovated",
                               "To be done up", "To restore"]
    swimmingpool_options = ["Yes", "No"]
    terrace_options = ["Yes", "No"]
    toilet_options = ["1", "2", "3", "4", "5", "6"]
    typeofproperty_options = ["Apartment", "House"]

    # User input fields
    with st.sidebar:
        bathroom = st.selectbox("Bathroom", bathroom_options)
        bedroom = st.selectbox("Bedroom", bedroom_options)
        construction_year = st.number_input("Construction Year", min_value=1800, max_value=2024, value=2020)
        district = st.selectbox("District", district_options)
        fireplace = st.selectbox("Fireplace", fireplace_options)
        floodingzone = st.selectbox("Flooding Zone", floodingzone_options)
        garden = st.selectbox("Garden", garden_options)
        kitchen = st.selectbox("Kitchen", kitchen_options)
        living_area = st.number_input("Living Area (m²)", min_value=10, max_value=1000, value=100)
        monthlycharges = st.number_input("Monthly Charges (€)", min_value=0, max_value=5000, value=100)
        number_of_facades = st.selectbox("Number of Facades", number_of_facades_options)
        peb = st.selectbox("PEB", peb_options)
        province = st.selectbox("Province", province_options)
        region = st.selectbox("Region", region_options)
        roomcount = st.number_input("Room Count", min_value=1, max_value=20, value=5)
        showercount = st.selectbox("Shower Count", showercount_options)
        stateofbuilding = st.selectbox("State of Building", stateofbuilding_options)
        subtypeofproperty = st.selectbox("Subtype of Property", subtypeofproperty_options)
        surfaceofplot = st.number_input("Surface of Plot (m²)", min_value=10, max_value=10000, value=200)
        swimmingpool = st.selectbox("Swimming Pool", swimmingpool_options)
        terrace = st.selectbox("Terrace", terrace_options)
        toilet = st.selectbox("Toilet", toilet_options)
        typeofproperty = st.selectbox("Type of Property", typeofproperty_options)

    user_input = {
        'bathroomcount': bathroom,
        'bedroomrcount': bedroom,
        'constructionyear': construction_year,
        'district': district,
        'fireplace': fireplace,
        'floodingzone': floodingzone,
        'garden': garden,
        'kitchen': kitchen,
        'livingarea': living_area,
        'numberoffacades': number_of_facades,
        'monthlycharges': monthlycharges,
        'peb': peb,
        'province': province,
        'region': region,
        'roomcount': roomcount,
        'showercount': showercount,
        'stateofbuilding': stateofbuilding,
        'subtypeofproperty': subtypeofproperty,
        'surfaceofplot': surfaceofplot,
        'swimmingpool': swimmingpool,
        'terrace': terrace,
        'toiletcount': toilet,
        'typeofproperty': typeofproperty
    }

    user_input_df = preprocess_input(user_input)

    model = load_model()

    if st.button("Predict"):
        prediction = model.predict(user_input_df)
        st.subheader(f"Predicted Price: €{prediction[0]:,.2f}")

if __name__ == '__main__':
    main()