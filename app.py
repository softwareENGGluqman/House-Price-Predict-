import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Page Title
st.title("🏡 Maharashtra House Price Predictor")
st.write("Random Forest Algorithm ka use karke apne ghar ki keemat janiye.")

# Data Load aur Model Train karne ka function
@st.cache_resource
def train_model():
    # CSV file read karna
    df = pd.read_csv('maharashtra_houses.csv')
    
    # Location (text) ko numbers mein convert karna ML ke liye
    le = LabelEncoder()
    df['location_encoded'] = le.fit_transform(df['location'])
    
    # Features (X) aur Target (y) set karna
    X = df[['location_encoded', 'bhk']]
    y = df['price_in_lakhs']
    
    # Random Forest Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, le, df['location'].unique()

# Model load karna
model, le, locations_list = train_model()

# User Input Section
st.header("Apni Requirement Batiye:")

selected_location = st.selectbox("Kaunsi city/district mein ghar chahiye?", locations_list)
bhk_count = st.number_input("Kitne BHK ka ghar chahiye?", min_value=1, max_value=5, value=2)

# Prediction Logic
if st.button("Price Predict Karein"):
    # User ki di hui location ko wapas number mein convert karna
    loc_encoded = le.transform([selected_location])[0]
    
    # Model se prediction lena
    prediction = model.predict([[loc_encoded, bhk_count]])
    
    # Result dikhana
    st.success(f"🎉 Estimated Price: ₹ {prediction[0]:.2f} Lakhs")
