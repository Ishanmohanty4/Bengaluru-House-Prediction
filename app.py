import streamlit as st
import pickle
import json
import numpy as np

# Load model and columns
with open('bangalore_house_price_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']
    locations = data_columns[3:]  # Skip total_sqft, bath, bhk

st.title("Bengaluru House Price Estimator")

sqft = st.number_input("Total Square Feet", min_value=300, max_value=20000, value=1200)
size = st.selectbox("Size", [1, 2, 3, 4, 5])
bath = st.selectbox("Bathrooms", [1, 2, 3, 4, 5])
location = st.selectbox("Location", locations)

if st.button("Estimate Price"):
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = size
    
    if location in data_columns:
        loc_idx = data_columns.index(location)
        x[loc_idx] = 1
        
    pred_log = model.predict([x])[0]
    estimated_price = np.expm1(pred_log)
    st.success(f"Estimated Price: ₹{estimated_price:.2f} Lakhs")