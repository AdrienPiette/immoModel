import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
import pickle
from sklearn.preprocessing import OneHotEncoder

# Function to load the model
def load_model():
    model = CatBoostRegressor()
    model.load_model('catboost_model.cbm')
    return model

# Load the one-hot encoder
with open('onehotencoder.pkl', 'rb') as f:
    one: OneHotEncoder = pickle.load(f)

# Load the cleaned dataset
df = pd.read_csv('cleaned_dataset.csv')

# One-hot encode the categorical features
columns_to_encode = ['district', 'subtypeofproperty', 'peb', 'province', 'region',
                     'stateofbuilding', 'swimmingpool', 'terrace', 'kitchen', 'garden', 'typeofproperty']
one_encoding = one.transform(df[columns_to_encode])
one_encoding_df = pd.DataFrame(one_encoding, columns=one.get_feature_names_out(columns_to_encode))
df_final = pd.concat([df.drop(columns=columns_to_encode), one_encoding_df], axis=1)

# Function to preprocess user input
def preprocess_input(user_input):
    categorical_features = ['district', 'subtypeofproperty', 'peb', 'province', 'region',
                            'stateofbuilding', 'swimmingpool', 'terrace', 'kitchen', 'garden', 'typeofproperty']
    user_input_df = pd.DataFrame([user_input])
    one_encoding = one.transform(user_input_df[categorical_features])
    one_encoding_df = pd.DataFrame(one_encoding, columns=one.get_feature_names_out(categorical_features))
    user_input_df = pd.concat([user_input_df.drop(columns=categorical_features), one_encoding_df], axis=1)
    return user_input_df

# Main function to run the Streamlit app
def main():
    st.title("🏡 Real Estate Price Prediction")
    st.markdown("### Predict the price of your property with our advanced CatBoost model")
    st.markdown("#### 📝 Provide the details of the property in the sidebar to get started! Click the 'Predict' button to get the estimated price.")
    st.markdown('<center><img src="https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExeXVzb2dxNHk2cXBtZmk0ZjYxbTh6ZGVha29la2liMmQ4eHJlZmYyYSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/1AHZBEKJx5Mf57NQqb/giphy-downsized-large.gif"></center>', unsafe_allow_html=True)

    st.sidebar.header("Property Details")
    st.sidebar.markdown("Provide the details of the property for prediction:")

    # Predefined options for each feature
    district_options = ["Aalst", "Antwerp", "Arlon", "Ath", "Bastogne", "Brugge", "Brussels", "Charleroi",
                        "Dendermonde", "Diksmuide", "Dinant", "Eeklo", "Gent", "Halle-Vilvoorde", "Hasselt", "Huy",
                        "Ieper", "Kortrijk", "Leuven", "Liège", "Maaseik", "Marche-en-Famenne", "Mechelen", "Mons",
                        "Mouscron", "Namur", "Neufchâteau", "Nivelles", "Oostend", "Oudenaarde", "Philippeville", "Sint-Niklaas",
                        "Roeselare", "Soignies", "Thuin", "Tielt", "Tongeren", "Tournai", "Turnhout", "Verviers",
                        "Veurne", "Virton", "Waremme"]

    fireplace_options = ["Yes", "No"]

    subtypeofproperty_options = ["apartment", "apartement_block", "bungalow", "castle", "chalet", "country_cottage", "duplex", "exceptional_property",
                                 "farmhouse", "flat_studio", "ground_floor", "house", "kot", "loft", "mansion", "manor_house",
                                 "mixed_use_building", "other_property", "penthouse", "service_flat", "pavilion", "town_house", "triplex", "villa"]

    peb_options = ["A", "A+", "A++", "A_A+", "B", "B_A", "C", "D", "E", "E_D", "F", "F_C", "F_D", "F_E", "G"]

    province_options = ["Antwerp", "Brussels", "East Flanders", "Flemish Brabant", "Hainaut", "Limburg", "Liège", "Luxembourg", "Namur",
                        "Walloon Brabant", "West Flanders"]

    region_options = ["Brussels", "Flanders", "Wallonie"]
    garden_options = ["Yes", "No"]
    kitchen_options = ["New", "Installed", "Semi-equipped", "USA Hyper-equipped"]
    number_of_facades_options = ["1", "2", "3", "4"]
    bathroom_options = ["1", "2", "3", "4", "5", "6"]
    bedroom_options = ["1", "2", "3", "4", "5", "6"]
    showercount_options = ["1", "2", "3", "4", "5", "6"]
    stateofbuilding_options = ["0", "1", "2", "3", "4", "5"]
    swimmingpool_options = ["Yes", "No"]
    terrace_options = ["Yes", "No"]
    toilet_options = ["1", "2", "3", "4", "5", "6"]
    typeofproperty_options = ["Apartment", "House"]

    # User input fields
    with st.sidebar:
        st.markdown("### 📑 General Information")
        typeofproperty = st.selectbox("Type of Property", typeofproperty_options)
        subtypeofproperty = st.selectbox("Subtype of Property", subtypeofproperty_options)
        province = st.selectbox("Province", province_options)
        region = st.selectbox("Region", region_options)
        district = st.selectbox("District", district_options)

        st.markdown("### 📐 Property Details")
        construction_year = st.number_input("Construction Year", min_value=1800, max_value=2024, value=2020)
        living_area = st.number_input("Living Area (m²)", min_value=10, max_value=1000, value=100)
        surfaceofplot = st.number_input("Surface of Plot (m²)", min_value=10, max_value=10000, value=200)
        roomcount = st.number_input("Room Count", min_value=1, max_value=20, value=5)
        number_of_facades = st.selectbox("Number of Facades", number_of_facades_options)

        st.markdown("### 🛋️ Amenities")
        kitchen = st.selectbox("Kitchen", kitchen_options)
        bathroom = st.selectbox("Bathroom", bathroom_options)
        bedroom = st.selectbox("Bedroom", bedroom_options)
        showercount = st.selectbox("Shower Count", showercount_options)
        toilet = st.selectbox("Toilet", toilet_options)
        garden = st.selectbox("Garden", garden_options)
        swimmingpool = st.selectbox("Swimming Pool", swimmingpool_options)
        terrace = st.selectbox("Terrace", terrace_options)
        fireplace = st.selectbox("Fireplace", fireplace_options)
        

        st.markdown("### 📜 Additional Information")
        peb = st.selectbox("PEB", peb_options)
        stateofbuilding = st.selectbox("State of Building", stateofbuilding_options)

    # Collect user input
    user_input = {
        'bathroomcount': bathroom,
        'bedroomcount': bedroom,
        'constructionyear': construction_year,
        'livingarea': living_area,
        'numberoffacades': number_of_facades,
        'roomcount': roomcount,
        'showercount': showercount,
        'surfaceofplot': surfaceofplot,
        'toiletcount': toilet,
        'district': district,
        'subtypeofproperty': subtypeofproperty,
        'peb': peb,
        'province': province,
        'region': region,
        'fireplace': fireplace,
        'stateofbuilding': stateofbuilding,
        'swimmingpool': swimmingpool,
        'terrace': terrace,
        'kitchen': kitchen,
        'garden': garden,
        'typeofproperty': typeofproperty,
    }

    # Preprocess the user input
    user_input_df = preprocess_input(user_input)
    
    model = load_model()

    if st.button("Predict"):
        try:
            prediction = model.predict(user_input_df)
            st.subheader(f"🏷️ Predicted Price: **€{prediction[0]:,.2f}**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
