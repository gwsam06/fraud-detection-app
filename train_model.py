import pandas as pd
from xgboost import XGBClassifier
import joblib

# Load CLEAN dataset (already engineered)
df = pd.read_csv("processed_fraud_data.csv")

# ===== FEATURES (MUST MATCH APP EXACTLY) =====
X = df[
    [
        "Amount",
        "AccountBalance",
        "AnomalyScore",
        "balance_diff",
        "suspicious_flag",
        "type_transfer",
        "type_cashout"
    ]
]

# Target
y = df["FraudIndicator"]

# ===== TRAIN MODEL =====
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# ===== SAVE MODEL =====
joblib.dump(model, "final_fraud_model.pkl")

print("✅ Model retrained successfully (APP-ALIGNED)")