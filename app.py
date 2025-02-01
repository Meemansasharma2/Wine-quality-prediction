import streamlit as st
import joblib
import numpy as np

# Load the pre-trained models and scaler
try:
    wine_quality_model = joblib.load('wine_quality_model.pkl')
    wine_type_model = joblib.load('wine_type_model.pkl')
    scaler = joblib.load('scaler.pkl')
    st.write("Models and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading models or scaler: {e}")

# Set up the title and description
st.title('Wine Quality & Type Prediction')
st.write('Enter the values for the following features to predict the wine type (Red/White) and quality (Bad/Average/Good).')

# Get user input for the features with example and range
volatile_acidity = st.number_input('Volatile Acidity (example: 0.5, range: 0.0-1.0)', step=0.1, min_value=0.0, max_value=1.0)
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide (example: 30, range: 0-100)', step=1, min_value=0, max_value=100)
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide (example: 100, range: 0-300)', step=1, min_value=0, max_value=300)
sulphates = st.number_input('Sulphates (example: 0.6, range: 0.0-1.0)', step=0.1, min_value=0.0, max_value=1.0)
alcohol = st.number_input('Alcohol (example: 12.5, range: 8.0-14.0)', step=0.1, min_value=8.0, max_value=14.0)
residual_sugar = st.number_input('Residual Sugar (example: 2.0, range: 0.0-20.0)', step=0.1, min_value=0.0, max_value=20.0)

# When the user clicks "Predict"
if st.button('Predict'):
    try:
        # Prepare the input data for prediction
        input_data = np.array([[volatile_acidity, free_sulfur_dioxide, total_sulfur_dioxide, sulphates, alcohol, residual_sugar]])
        input_data_scaled = scaler.transform(input_data)

        # Predict wine type (Red/White)
        wine_type_prediction = wine_type_model.predict(input_data_scaled)[0]
        wine_type = "Red" if wine_type_prediction == 0 else "White"

        # Predict wine quality (Bad, Average, Good)
        wine_quality_prediction = wine_quality_model.predict(input_data_scaled)[0]
        if wine_quality_prediction == 0:
            wine_quality = "Bad"
        elif wine_quality_prediction == 1:
            wine_quality = "Average"
        else:
            wine_quality = "Good"

        # Display the results
        st.write(f"**Wine Type:** {wine_type}")
        st.write(f"**Wine Quality:** {wine_quality}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
