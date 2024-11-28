import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the StandardScaler
scaler = StandardScaler()

# Set up the title and description of the app
st.title("Heart Disease Prediction")
st.write("""
    This app predicts the likelihood of heart disease based on user input.
    Please fill out the information below and click on 'Predict'.
""")

# Function to get user input from the Streamlit interface
def get_user_input():
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    chest_pain = st.selectbox("Chest Pain Type (0 = ASYSTOLE, 1 = Atrial Tachycardia, 2 = NAP, 3 = TA)", [0, 1, 2, 3])
    resting_bp = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=50, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol (in mg/dL)", min_value=100, max_value=500, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar (0 = <120 mg/dL, 1 = >120 mg/dL)", [0, 1])
    resting_ecg = st.selectbox("Resting ECG (0 = Normal, 1 = ST, 2 = LVH)", [0, 1, 2])
    max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
    oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    st_slope = st.selectbox("ST Slope (0 = Up, 1 = Flat, 2 = Down)", [0, 1, 2])

    # Collect all inputs into a numpy array
    user_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

    return user_data

# Get user input
user_data = get_user_input()

# Apply the same scaling as was used during training
user_data_scaled = scaler.fit_transform(user_data)  # Use the same scaler if saved
print(user_data_scaled)
# Predict and display the result
if st.button("Predict"):
    prediction = model.predict(user_data_scaled)
    if prediction == 1:
        st.write("The model predicts that the patient **has a heart disease**.")
    else:
        st.write("The model predicts that the patient **does not have a heart disease**.")
        st.write(user_data_scaled)
