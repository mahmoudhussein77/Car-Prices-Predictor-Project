#importing needed modules
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import pickle
from sklearn.preprocessing import LabelEncoder

# loading model
with open("car_prices_predictor.pk1", "rb") as f:
    model = pickle.load(f)

st.title("Car Prices Predictor By: Mahmoud Hussein")

# Reading Data
data = pd.read_csv("D:\Car Prices Predictor Final\car.csv")
warnings.filterwarnings('ignore',category=SyntaxWarning)

# Filter Models By Car Brand
brand = set(data["Make"])
all_model = [[] for _ in data['Make']]
for n,i in enumerate(brand):
    c = 0
    for n2,i2 in enumerate(data['Make']):
        if i == i2 :
            all_model[n].append(data['Model'][n2])

unique_model = set(map(tuple, all_model))
filtered_model = list(filter(bool, all_model))
result = {v: k for v, k in zip(brand, filtered_model)}
for key in result:
    result[key] = list(set(result[key]))


#input options
color_options = list(set(data["Color"]))

color_options.sort()

brand_options = list(set(data['Make'])) # Make = Brand

brand_options.sort()


# input car features
car_brand_classes = st.selectbox("Brand",options=brand_options)

# to show a specific models for each car brand
model_options = result[car_brand_classes]

choose = ['Yes','No']

car_model_classes = st.selectbox("Model",options=model_options)

year = st.number_input("Production Year (1963:2025)")
if year > 2025 or year < 1963 :
    st.write("please type a right car production year!")

color_classes = st.selectbox("Color",options=color_options)

# input as kilometers then convert to miles to match data
miles = st.number_input("Kilometers")*0.621371
if miles < 0 :
    st.write("please type right kilometers number!")

air_conditioner_classes = st.selectbox("Air Conditioner",options=choose)

power_steering_classes = st.selectbox("Power Steering",options=choose)

remote_control_classes = st.selectbox("Remote Control",options=choose)

transmission_classes = st.selectbox("Transmission",options=["Automatic", "Manual"])

# labeling inputs
automatic_classes = 1 if transmission_classes == "Automatic" else 0
air_conditioner = 1 if air_conditioner_classes == "Yes" else 0
power_steering = 1 if power_steering_classes == "Yes" else 0
remote_control = 1 if remote_control_classes == "Yes" else 0

# label encoding for alphapetic inputs
le_color = LabelEncoder()
le_color.fit(color_options)
car_color = le_color.transform([color_classes])[0]

le_brand = LabelEncoder()
le_brand.fit(brand_options)
car_brand = le_brand.transform([car_brand_classes])[0]

le_model = LabelEncoder()
le_model.fit(model_options)
car_model = le_model.transform([car_model_classes])[0]

# after clicking the button
if st.button("Predict Car Price"):
    
    input_data = np.array([[
    car_model, car_brand, car_color, automatic_classes, air_conditioner, power_steering, remote_control, year, miles
    ]])

    prediction = model.predict(input_data)
    result = int(prediction[0])
    formatted_price = f"{prediction[0]:,.0f}"
    st.markdown(
    f"{car_brand_classes} {car_model_classes} {int(year)} - Price is:   "
    f"<span style='color: green; font-weight: bold; font-size: 24px;'>  {formatted_price} EGP</span>",
    unsafe_allow_html=True)
    