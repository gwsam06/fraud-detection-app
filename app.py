import streamlit as st
import numpy as np
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Fraud Detection AI System", layout="wide")

# ===============================
# GLOBAL STYLE (READABILITY FIX)
# ===============================
st.markdown("""
<style>
/* Improve caption visibility */
.css-1cpxqw2, .css-1v0mbdj {
    color: #CCCCCC !important;
    font-size: 14px !important;
}

/* Footer styling */
.footer-text {
    text-align: center;
    font-size: 13px;
    color: #BBBBBB;
    line-height: 1.6;
}

/* Disclaimer styling */
.disclaimer {
    font-size: 13px;
    color: #AAAAAA;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("final_fraud_model.pkl")

model = load_model()

# ===============================
# SAFE INITIALIZATION
# ===============================
prediction = None
probability = None

# ===============================
# SIDEBAR (CLEANED)
# ===============================
# ===============================
# SIDEBAR (CLEAN + FIXED)
# ===============================
with st.sidebar:

    st.markdown("## 🏢 System Information")

    st.success("Model loaded successfully")

    st.markdown("### 👤 Developer Identity")
    st.markdown("""
    **Name:** Gameli Samuel Wordui  
    **Project:** Thrive Africa ML and AI Initiative GP 15  
    **System:** Financial Fraud Detection ML  
    **Version:** v2.1  

    🌍 *Built under the Thrive Africa Project – advancing AI solutions for financial security and economic resilience across Africa.*
    """)

    st.markdown("## 🎯 Fraud Sensitivity Control")

    threshold = st.slider("Set Detection Threshold (%)", 1, 100, 5)

    st.caption(f"Current Threshold: {threshold}%")
    st.caption("Lower = more sensitive detection")
    
# ===============================
# MAIN HEADER
# ===============================
st.title("💳 Fraud Detection AI System")
st.markdown("<span style='color:#A0AEC0;'>Version 2.1 | Enterprise Fraud Risk Engine</span>", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 2])

# ===============================
# TOP LAYOUT (RESULT LEFT, INPUT RIGHT)
# ===============================
left_col, right_col = st.columns([1, 2])

# ===============================
# INPUT + BUTTON (RIGHT PANEL)
# ===============================

with right_col:

    st.markdown("## 📥 Input Transaction Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        transaction_type = st.selectbox(
            "Transaction Type",
            ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"]
        )
        amount = st.number_input("Amount", min_value=0.0)

    with col2:
        oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
        newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)

    with col3:
        oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
        newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

    # ✅ BUTTON FIXED INSIDE RIGHT PANEL
    predict_clicked = st.button("🔍 Predict Fraud", key="predict_main")
    
# ===============================
# MODEL EXECUTION (SAFE)
# ===============================

if predict_clicked:

    if amount <= 0:
        st.warning("Amount must be greater than 0")

    else:
        balance_diff = oldbalanceOrg - newbalanceOrig
        suspicious_flag = int(balance_diff > amount)

        type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0
        type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
        type_PAYMENT = 1 if transaction_type == "PAYMENT" else 0

        input_data = np.array([[ 
            amount,
            oldbalanceOrg,
            balance_diff,
            suspicious_flag,
            type_TRANSFER,
            type_CASH_OUT,
            type_PAYMENT
        ]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

# ===============================
# RESULT PANEL (LEFT)
# ===============================
with left_col:
    if probability is not None:

        st.markdown("## 📊 Prediction Result")

        if probability >= threshold / 100:
            st.markdown(f"""
            <div style='padding:12px; border-radius:8px; background:#5A1E1E; color:#FFB3B3; font-weight:600;'>
            🚨 FRAUD DETECTED ({probability:.2%})
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='padding:12px; border-radius:8px; background:#1E5A1E; color:#B3FFB3; font-weight:600;'>
            ✅ SAFE TRANSACTION ({probability:.2%})
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🔎 Risk Analysis")
        st.write(f"Fraud Probability: {probability:.2%}")

        # ✅ RISK LEVEL
        if probability >= threshold / 100:
            st.markdown("""
            <div style='padding:12px; border-radius:8px; background:#5A1E1E; color:#FFB3B3; font-weight:600;'>
            🔴 High Risk Transaction
            </div>
            """, unsafe_allow_html=True)

        elif probability >= (threshold / 100) * 0.5:
            st.markdown("""
            <div style='padding:12px; border-radius:8px; background:#5A4A1E; color:#FFE699; font-weight:600;'>
            🟠 Moderate Risk Transaction
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='padding:12px; border-radius:8px; background:#1E5A1E; color:#B3FFB3; font-weight:600;'>
            🟢 Low Risk Transaction
            </div>
            """, unsafe_allow_html=True)

        st.progress(float(probability))

        # ✅ CORRECT PLACEMENT (INSIDE RESULT BLOCK)
        with st.expander("🧠 How the Model Works"):
            st.info("""
            This system uses a machine learning model to estimate the probability of fraud.

            - The model outputs a fraud probability (%)
            - A threshold is used to classify:
                - Above threshold → Fraud
                - Below threshold → Safe

            Lower threshold = More sensitive detection  
            Higher threshold = Fewer false alarms
            """)
    
# ===============================
# FOOTER (FINAL POLISHED)
# ===============================
st.markdown("""
<div style='text-align:center; font-size:13px; color:#888; line-height:1.6;'>

© 2026 Gameli Samuel Wordui | Version 2.1  
Thrive Africa Project Group 15 – Financial Fraud Detection AI System  

⚠️ <b>Disclaimer:</b><br>
This AI system is designed for educational, research, and decision-support purposes only.<br>
Predictions are probabilistic and should not be used as the sole basis for financial, legal, or operational decisions.<br>
Users are advised to apply professional judgement, institutional policies, and regulatory compliance standards.

</div>
""", unsafe_allow_html=True)