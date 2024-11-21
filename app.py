import streamlit as st
import pickle
import numpy as np
import plotly.express as px

# Corrected background image URL (or use a local file if needed)
background_image_url = "https://drive.google.com/uc?export=view&id=1xSO1hH6-r2ZFzt1lsUo1w5JLoQMzU1w5"

# Set the background image and custom styling using CSS
st.markdown(
    f"""
    <style>
    body {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #ffffff; /* Adjust text color for readability */
    }}
    .main-title {{
        text-align: left;
        margin-left: 5%;
        font-size: 2.5rem;
        font-weight: bold;
        margin-top: 0;
    }}
    .block-container {{
        max-width: 70%; /* Adjust content width */
        margin-left: 5%; /* Adjust horizontal alignment */
        margin-right: auto;
    }}
    .footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: transparent;
        text-align: center;
        padding: 10px 0;
        color: #ffffff;
        font-size: 12px;
    }}
    .highlight {{
        font-size: 1.8rem;
        font-weight: bold;
        color: #FFD700; /* Gold color */
        background-color: #333333; /* Dark background for contrast */
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        margin-top: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the model and encoders
model = pickle.load(open('xgboost_model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# Placeholder for sample data
# Replace this with your actual DataFrame
import pandas as pd
df_final = pd.DataFrame({
    'Make': ['Toyota', 'Honda', 'BMW', 'Tesla', 'Ford'],
    'Price_usd': [25000, 22000, 45000, 60000, 30000]
})

# Title and instructions
st.markdown("<h1 class='main-title'>🚗 Car Price Prediction App</h1>", unsafe_allow_html=True)
st.write("### Welcome to the Car Price Prediction App!")
st.info("""
This app predicts the price of a car based on its features. 
Use the sidebar to input details such as transmission type, drivetrain, and more.
Ensure all fields are filled before clicking 'Predict Price.'
""")

# Feature descriptions
with st.expander("ℹ️ Feature Descriptions"):
    st.write("""
    - **Dealer Name**: Name of the car dealership.
    - **Transmission Type**: Type of transmission (e.g., Automatic, Manual).
    - **Drivetrain**: Power distribution system (e.g., Front-Wheel Drive, All-Wheel Drive).
    - **Fuel Type**: Type of fuel used (e.g., Gasoline, Diesel, Electric).
    - **Make**: Manufacturer of the car.
    - **Model**: Specific model of the car.
    - **Age**: Age of the car in years (e.g., -1 for future models).
    - **Mileage**: Number of miles driven.
    """)

# Sidebar inputs
st.sidebar.header("Car Details")
dealer_name = st.sidebar.selectbox(
    "Select Dealer Name",
    label_encoders['Dealer Name'].classes_,
    help="Choose the name of the dealer from the available list."
)
transmission_type = st.sidebar.selectbox(
    "Select Transmission Type",
    label_encoders['Transmission Type'].classes_,
    help="The type of transmission (e.g., Automatic, Manual)."
)
drivetrain = st.sidebar.selectbox(
    "Select Drivetrain",
    label_encoders['Drivetrain'].classes_,
    help="The power distribution system of the car (e.g., Front-Wheel Drive)."
)
fuel_type = st.sidebar.selectbox(
    "Select Fuel Type",
    label_encoders['Fuel Type'].classes_,
    help="The type of fuel used by the car (e.g., Gasoline, Diesel)."
)
make = st.sidebar.selectbox(
    "Select Make",
    label_encoders['Make'].classes_,
    help="The manufacturer of the car."
)
model_name = st.sidebar.selectbox(
    "Select Model",
    label_encoders['Model'].classes_,
    help="The specific model of the car."
)
county = st.sidebar.selectbox(
    "Select County",
    label_encoders['County'].classes_,
    help="The county where the car is located."
)
state = st.sidebar.selectbox(
    "Select State",
    label_encoders['State'].classes_,
    help="The state where the car is located."
)
age = st.sidebar.number_input(
    "Enter Car Age (years)",
    min_value=-1, max_value=69, value=5,
    help="Enter the age of the car in years (-1 for future models)."
)
mileage = st.sidebar.number_input(
    "Enter Mileage (in miles)",
    min_value=0, max_value=300000, value=50000,
    help="Enter the total miles the car has been driven."
)

# Convert inputs using label encoders
dealer_name_encoded = label_encoders['Dealer Name'].transform([dealer_name])[0]
transmission_type_encoded = label_encoders['Transmission Type'].transform([transmission_type])[0]
drivetrain_encoded = label_encoders['Drivetrain'].transform([drivetrain])[0]
fuel_type_encoded = label_encoders['Fuel Type'].transform([fuel_type])[0]
make_encoded = label_encoders['Make'].transform([make])[0]
model_name_encoded = label_encoders['Model'].transform([model_name])[0]
county_encoded = label_encoders['County'].transform([county])[0]
state_encoded = label_encoders['State'].transform([state])[0]

# Prepare input array
input_data = np.array([
    dealer_name_encoded, transmission_type_encoded, drivetrain_encoded,
    fuel_type_encoded, make_encoded, model_name_encoded, age,
    mileage, county_encoded, state_encoded
]).reshape(1, -1)

# Predict and display the highlighted price
try:
    if st.sidebar.button("Predict Price"):
        predicted_price = model.predict(input_data)[0]
        st.markdown(
            f"<div class='highlight'>Predicted Car Price: ${predicted_price:,.2f}</div>",
            unsafe_allow_html=True
        )
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")

# Add the plot below the predicted price
median_price_per_make = df_final.groupby('Make')['Price_usd'].median().reset_index()
median_price_per_make = median_price_per_make.sort_values(by='Price_usd', ascending=False)

# Creating the bar plot
fig_median_price_make = px.bar(
    median_price_per_make,
    x='Make',
    y='Price_usd',
    title='Median Car Price by Make',
    color='Price_usd',
    color_continuous_scale=px.colors.sequential.Viridis,
    labels={'Price_usd': 'Median Price (USD)'}
)

# Updating layout
fig_median_price_make.update_layout(
    xaxis_title='Car Make',
    yaxis_title='Median Price (USD)',
    title_font_size=20,
    title_font_color='blue'
)

# Display the plot
st.plotly_chart(fig_median_price_make, use_container_width=True)

# Footer
st.markdown(
    "<div class='footer'>Created by Kaushik Manjunatha</div>",
    unsafe_allow_html=True
)
