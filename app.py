import streamlit as st
import pandas as pd
import joblib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Fraud Detection Prediction App",
    layout="wide"
)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("final_fraud_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ==============================
# LAYOUT (LEFT = RESULT, RIGHT = INPUT)
# ==============================
left_col, right_col = st.columns([1, 2])

# ==============================
# LEFT COLUMN (SYSTEM + RESULT)
# ==============================
with left_col:
    st.sidebar.header("⚙️ System Information")
    st.sidebar.write("Model: Financial Fraud Detection ML Model")
    st.sidebar.write("Type: Classification")

    st.subheader("📊 Prediction Result")

    result_placeholder = st.empty()
    prob_placeholder = st.empty()

    if model is not None:
        st.success("✅ Model loaded successfully")
    else:
        st.error("❌ Model not found")

    # ==============================
    # FRAUD SENSITIVITY CONTROL
    # ==============================
    st.markdown("### ⚙️ Fraud Sensitivity Control")

    threshold = st.slider(
        "Set Fraud Detection Threshold (%)",
        min_value=1,
        max_value=50,
        value=5
    ) / 100

    st.caption(f"Current Fraud Alert Threshold: {threshold:.0%}")
    st.caption("Lower threshold = more sensitive (more fraud alerts)")

# ==============================
# RIGHT COLUMN (INPUT FORM ONLY)
# ==============================
with right_col:
    st.title("💳 Fraud Detection Prediction App")
    st.write("Enter transaction details to predict fraud")

    st.subheader("📥 Input Transaction Details")

    # ==============================
    # INPUT FIELDS
    # ==============================
    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox(
            "Transaction Type",
            ["Transfer", "Cash Out", "Payment", "Debit"]
        )
        amount = st.number_input("Amount", min_value=0.0)
        oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)

    with col2:
        newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)
        oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
        newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

    if amount <= 0:
        st.warning("⚠️ Amount must be greater than 0")

    # ==============================
    # FEATURE ENGINEERING (ALIGNED)
    # ==============================
    type_transfer = 1 if transaction_type == "Transfer" else 0
    type_cashout = 1 if transaction_type == "Cash Out" else 0

    balance_diff = oldbalanceOrg - amount
    suspicious_flag = 1 if balance_diff > amount else 0

    # ==============================
    # PREDICTION (OUTPUT GOES LEFT)
    # ==============================
    if st.button("Predict"):

        if model is None:
            result_placeholder.error("Model not loaded.")
        elif amount <= 0:
            result_placeholder.error("Enter valid amount.")
        else:
            input_data = pd.DataFrame([{
                "Amount": amount,
                "AccountBalance": oldbalanceOrg,
                "AnomalyScore": 0,
                "balance_diff": balance_diff,
                "suspicious_flag": suspicious_flag,
                "type_transfer": type_transfer,
                "type_cashout": type_cashout
            }])

            prediction = model.predict(input_data)[0]

            try:
                probability = model.predict_proba(input_data)[0][1]
            except:
                probability = None

            # ==============================
            # DISPLAY RESULT ON LEFT PANEL (SMART THRESHOLD)
            # ==============================

            if probability is not None:

                if probability >= threshold:
                    result_placeholder.error("🚨 Fraudulent Transaction Detected!")
                else:
                    result_placeholder.success("✅ Legitimate Transaction")

                prob_placeholder.metric("Fraud Probability", f"{probability:.2%}")

            else:
                result_placeholder.warning("Prediction made but probability unavailable.")