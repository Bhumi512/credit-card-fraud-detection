import streamlit as st
import pandas as pd
import pickle  

# Load the trained model
with open("fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app title
st.title("💳 Credit Card Fraud Detection App")

# Sidebar for user inputs
st.sidebar.header("Enter Transaction Details")

def user_input():
    step = st.sidebar.number_input("Step (Transaction Time in Hours)", min_value=1, max_value=10000, value=1)
    amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.01, max_value=10000.0, value=100.0)
    oldbalanceOrg = st.sidebar.number_input("Original Balance Before Transaction", min_value=0.0, value=1000.0)
    newbalanceOrig = st.sidebar.number_input("New Balance After Transaction", min_value=0.0, value=900.0)
    oldbalanceDest = st.sidebar.number_input("Receiver's Balance Before Transaction", min_value=0.0, value=5000.0)
    newbalanceDest = st.sidebar.number_input("Receiver's Balance After Transaction", min_value=0.0, value=5100.0)

    type_options = ["CASH_OUT", "PAYMENT", "TRANSFER", "CASH_IN", "DEBIT"]
    transaction_type = st.sidebar.selectbox("Transaction Type", type_options)

    type_encoded = [1 if t == transaction_type else 0 for t in type_options]

    data = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "CASH_OUT": type_encoded[0],
        "PAYMENT": type_encoded[1],
        "TRANSFER": type_encoded[2],
        "CASH_IN": type_encoded[3],
        "DEBIT": type_encoded[4]
    }
    
    return pd.DataFrame(data, index=[0])

df = user_input()

st.subheader("📊 Transaction Details")
st.write(df)

if st.button("Detect Fraud 🚨"):
    prediction = model.predict(df)
    result = "🚨 Fraudulent Transaction Detected!" if prediction[0] == 1 else "✅ Transaction is Legitimate."
    
    st.subheader("🔍 Prediction Result")
    st.write(result)
    
    if prediction[0] == 1:
        st.error("⚠️ This transaction seems suspicious! Please investigate.")
    else:
        st.success("✅ This transaction looks safe.")
