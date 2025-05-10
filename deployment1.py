#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv(r"C:\Users\Jhansi\OneDrive\Desktop\EXCELR\ALZHEIMERS PROJECT\alzheimers_disease_data.csv")
# Drop confidential column if exists
data = data.drop(columns=["DoctorInCharge"], errors='ignore')

# Handling missing values
data.fillna(data.median(), inplace=True)

# Splitting features and target variable
X = data.drop(columns=["Diagnosis"])
y = data["Diagnosis"]

# Standardizing numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
pickle.dump(model, open("alzheimers_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Streamlit App
st.title("Alzheimer's Disease Prediction")
st.write("Enter patient details to predict Alzheimer's Disease")

# Define binary (0 or 1) features
binary_features = ["Gender", "Smoking", "FamilyHistory", "Alzheimers", "CardiovascularDisease", 
                   "Diabetes", "Depression", "HeadInjury", "Hypertension", "Confusion", "Disorientation", 
                   "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness"]

# Define decimal features
decimal_features = ["BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality", 
                    "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", 
                    "MMSE", "FunctionalAssessment"]

# User inputs displayed in separate columns
cols = st.columns(3)
features = {}
feature_list = list(X.columns)

for idx, col in enumerate(feature_list):
    with cols[idx % 3]:  # Arrange inputs in three columns
        if col in binary_features:
            features[col] = st.radio(f"{col}", [0, 1], index=int(data[col].median()))
        elif col in decimal_features:
            features[col] = st.number_input(f"{col}", value=float(data[col].median()), key=col, format="%.2f", step=0.01)
        else:
            features[col] = st.number_input(f"{col}", value=int(data[col].median()), key=col, format="%d", step=1)

# Prediction
if st.button("Predict"):
    user_data = np.array([list(features.values())]).reshape(1, -1)
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)[0]
    
    result = "Positive for Alzheimer's" if prediction == 1 else "Negative for Alzheimer's"
    st.write(f"Prediction: {result}")

