# ============================================
# 🚀 TRAIN FRAUD DETECTION MODEL (FINAL CLEAN)
# Author: Gameli Samuel Wordui
# Version: v3.0 (Production Aligned)
# ============================================

import pandas as pd
from xgboost import XGBClassifier
import joblib

# ============================================
# 📥 LOAD PROCESSED DATA
# ============================================
df = pd.read_csv("processed_fraud_data.csv")

# ============================================
# ✅ REQUIRED FEATURES (STRICT ORDER)
# ============================================
X = df[
    [
        "Amount",
        "AccountBalance",
        "AnomalyScore",
        "balance_diff",
        "suspicious_flag",
        "type_TRANSFER",
        "type_CASH_OUT",
        "type_PAYMENT",
        "type_DEBIT"
    ]
]

TARGET = "FraudIndicator"

# ============================================
# 🔍 VALIDATION CHECK (CRITICAL)
# ============================================
missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]

if missing_cols:
    raise ValueError(f"❌ Missing columns in dataset: {missing_cols}")

# ============================================
# 📊 PREPARE DATA
# ============================================
X = df[FEATURES]
y = df[TARGET]

# ============================================
# 🤖 TRAIN MODEL
# ============================================
model = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

model.fit(X, y)

# ============================================
# 💾 SAVE MODEL
# ============================================
joblib.dump(model, "final_fraud_model.pkl")

print("✅ Model trained and saved successfully (FULLY ALIGNED)")