🧠 Alzheimer's Disease Prediction
This project is designed to predict Alzheimer’s Disease using a machine learning model trained on clinical and lifestyle data. The model assists in early diagnosis by classifying whether a person is likely positive or negative for Alzheimer's.
📋 Table of Contents
•	Overview
•	Dataset
•	Model
•	Installation
•	Usage
•	Features
•	Team

📌 Overview
This ML application uses a Random Forest Classifier to identify potential Alzheimer’s patients. The model is trained on a dataset containing both medical and lifestyle-related features. A Streamlit interface is used for easy input and real-time prediction.
Goal: Develop an accurate and efficient method for identifying individuals at risk of Alzheimer’s using diverse data inputs including clinical symptoms and lifestyle factors.

🧾 Dataset
•	Source: Collected manually from various public repositories and refined
•	File: alzheimers_disease_data.csv
•	Features: Includes Gender, Smoking, MMSE, Depression, BMI, SleepQuality, Cholesterol levels, and more
•	Target: Diagnosis (0 = Negative, 1 = Positive)

🧠 Model
•	Algorithm: Random Forest Classifier
•	Libraries: scikit-learn, pandas, numpy
•	Performance: High accuracy achieved using standardized inputs
•	Additional Components:
o	scaler.pkl: StandardScaler used for feature scaling
o	alzheimers_model.pkl: Trained classifier model

⚙️ Installation
git clone https://github.com/YourUsername/Alzheimers-Disease-Prediction.git
cd Alzheimers-Disease-Prediction
pip install -r requirements.txt

Sample requirements.txt
nginx
CopyEdit
streamlit
scikit-learn
pandas
numpy

▶️ Usage
streamlit run deployment1.py
Then enter patient data in the web interface and click "Predict" to receive the result.
💡 Features
•	Binary inputs for symptoms and conditions (e.g., Diabetes, Confusion)
•	Continuous inputs for lifestyle factors (e.g., Sleep Quality, Alcohol Consumption)
•	Prediction output: "Positive for Alzheimer's" or "Negative for Alzheimer's"
•	User-friendly interface via Streamlit

