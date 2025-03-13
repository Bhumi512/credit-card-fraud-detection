import streamlit as st
import pandas as pd
import pickle  # Load the trained model

# Load pre-trained model
model = pickle.load(open("fraud_model.pkl", "rb"))

# App title
st.title("💳 Credit Card Fraud Detection")

# Sidebar inputs
st.sidebar.header("Enter Transaction Details")

def user_input():
    amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.01, max_value=10000.0, value=100.0)
    oldbalanceOrg = st.sidebar.number_input("Original Balance Before Transaction", min_value=0.0, value=1000.0)
    newbalanceOrig = st.sidebar.number_input("New Balance After Transaction", min_value=0.0, value=900.0)

    # Convert to DataFrame
    data = {
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig
    }
    return pd.DataFrame(data, index=[0])

df = user_input()

# Show user inputs
st.subheader("Transaction Data 📊")
st.write(df)

# Predict button
if st.button("Detect Fraud"):
    prediction = model.predict(df)
    result = "🚨 Fraudulent Transaction" if prediction[0] == 1 else "✅ Legitimate Transaction"
    st.subheader("Prediction Result")
    st.write(result)
