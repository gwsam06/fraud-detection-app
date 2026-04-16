# ============================================
# 🚀 PREPROCESS FRAUD DATA (FINAL CLEAN BUILD)
# Author: Gameli Samuel Wordui
# Version: v3.1 (FULLY FIXED & ALIGNED)
# ============================================

import pandas as pd

# ============================================
# 📥 LOAD DATASET
# ============================================
df = pd.read_csv("final_fraud_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("Original shape:", df.shape)

# ============================================
# 🧹 REMOVE DUPLICATES
# ============================================
if "TransactionID" in df.columns:
    df = df.drop_duplicates(subset=["TransactionID"])

print("After removing duplicates:", df.shape)

# ============================================
# 🔍 VALIDATION CHECK (CORRECTED)
# ============================================
required_columns = [
    "Amount",            # ✅ use this (NOT TransactionAmount)
    "AccountBalance",
    "AnomalyScore",
    "FraudIndicator",
    "Category"           # ✅ correct column name
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"❌ Missing required columns: {missing}")

# =========================
# FEATURE ENGINEERING (FULL TYPE SUPPORT)
# =========================

df["type_TRANSFER"] = (df["Category"].str.upper() == "TRANSFER").astype(int)
df["type_CASH_OUT"] = (df["Category"].str.upper() == "CASH_OUT").astype(int)
df["type_PAYMENT"] = (df["Category"].str.upper() == "PAYMENT").astype(int)
df["type_DEBIT"] = (df["Category"].str.upper() == "DEBIT").astype(int)

df["balance_diff"] = df["AccountBalance"] - df["TransactionAmount"]

df["suspicious_flag"] = (df["balance_diff"] > df["TransactionAmount"]).astype(int)

# ============================================
# 📊 FINAL FEATURE SET (STRICT ORDER)
# ============================================
final_df = df[
    [
        "Amount",
        "AccountBalance",
        "AnomalyScore",
        "balance_diff",
        "suspicious_flag",
        "type_transfer",
        "type_cashout",
        "FraudIndicator"
    ]
]

# ============================================
# 💾 SAVE CLEAN DATA
# ============================================
final_df.to_csv("processed_fraud_data.csv", index=False)

print("✅ Preprocessing completed successfully")
print(final_df.head())