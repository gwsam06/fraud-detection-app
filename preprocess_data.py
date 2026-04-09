import pandas as pd

# Load dataset
df = pd.read_csv("final_fraud_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("Original shape:", df.shape)

# =========================
# REMOVE DUPLICATES
# =========================

# Remove duplicate TransactionID
df = df.drop_duplicates(subset=["TransactionID"])

print("After removing duplicates:", df.shape)

# =========================
# FEATURE ENGINEERING (MATCH APP)
# =========================

# Transaction type encoding
df["type_transfer"] = (df["Category"] == "Transfer").astype(int)
df["type_cashout"] = (df["Category"] == "Cash Out").astype(int)

# Balance difference
df["balance_diff"] = df["AccountBalance"] - df["TransactionAmount"]

# Suspicious flag (same logic as app)
df["suspicious_flag"] = (df["balance_diff"] > df["TransactionAmount"]).astype(int)

# =========================
# FINAL FEATURES (STRICT ALIGNMENT)
# =========================

final_df = df[
    [
        "TransactionAmount",
        "AccountBalance",
        "AnomalyScore",
        "balance_diff",
        "suspicious_flag",
        "type_transfer",
        "type_cashout",
        "FraudIndicator"
    ]
]

# Rename for consistency with app
final_df = final_df.rename(columns={
    "TransactionAmount": "Amount"
})

# =========================
# SAVE CLEAN DATASET
# =========================

final_df.to_csv("processed_fraud_data.csv", index=False)

print("✅ FINAL CLEAN DATASET CREATED")
print(final_df.head())