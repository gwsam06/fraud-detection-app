import streamlit as st
import pandas as pd
import pickle

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Fraud Detection Prediction App",
    layout="centered"
)

# Dark theme styling
st.markdown("""
    <style>
    body { background-color: #0E1117; color: white; }
    </style>
""", unsafe_allow_html=True)

# ==============================
# TITLE
# ==============================
st.title("💳 Fraud Detection Prediction App")
st.write("Enter transaction details to predict whether it is Fraud or Not Fraud")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("final_fraud_model.pkl", "rb"))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is not None:
    st.success("✅ Model loaded successfully")
else:
    st.error("❌ Model not found. Ensure file is in folder.")

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ System Information")
st.sidebar.write("Model: Fraud Detection ML Model")
st.sidebar.write("Type: Classification")

# ==============================
# INPUT SECTION
# ==============================
st.subheader("📥 Input Transaction Details")

transaction_type = st.selectbox(
    "Transaction Type",
    ["Transfer", "Cash Out", "Payment", "Debit"]
)

amount = st.number_input("Amount", min_value=0.0)

oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)

oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

# ==============================
# VALIDATION
# ==============================
if amount <= 0:
    st.warning("⚠️ Amount must be greater than 0")

# ==============================
# FEATURE ENGINEERING
# ==============================
type_transfer = 1 if transaction_type == "Transfer" else 0
type_cashout = 1 if transaction_type == "Cash Out" else 0

balance_diff = oldbalanceOrg - newbalanceOrig
suspicious_flag = 1 if balance_diff > amount else 0

# ==============================
# PREDICTION (ONLY HERE)
# ==============================
if st.button("Predict"):

    if model is None:
        st.error("Model is not loaded.")
    elif amount <= 0:
        st.error("Enter a valid transaction amount.")
    else:
        input_data = pd.DataFrame([{
            'TransactionID': 1,
            'Amount': amount,
            'CustomerID': 0,
            'MerchantID': 0,
            'AnomalyScore': balance_diff,
            'Age': 30,
            'AccountBalance': oldbalanceOrg,
            'SuspiciousFlag': suspicious_flag,
            'Category_Online': type_transfer,
            'Category_Other': 0,
            'Category_Retail': 0,
            'Category_Travel': type_cashout
        }])

        # Prediction
        prediction = model.predict(input_data)[0]

        # Probability (safe fallback)
        try:
            probability = model.predict_proba(input_data)[0][1]
        except:
            probability = None

        # ==============================
        # OUTPUT
        # ==============================
        st.subheader("🔍 Prediction Result")

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
            st.write("This transaction shows suspicious financial behaviour.")
        else:
            st.success("✅ Legitimate Transaction")
            st.write("This transaction appears normal.")

        # Show probability
        if probability is not None:
            st.write(f"Fraud Probability: {probability:.2%}")