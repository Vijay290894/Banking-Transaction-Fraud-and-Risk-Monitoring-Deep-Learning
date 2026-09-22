from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json

# ------------------------------------------------
# Load model and preprocessing objects
# ------------------------------------------------

model = tf.keras.models.load_model(
    "fraud_detection_model.keras"
)

preprocessor = joblib.load(
    "preprocessor.pkl"
)

with open(
    "threshold.json",
    "r"
) as file:
    
    threshold_data = json.load(file)

THRESHOLD = threshold_data["threshold"]


# ------------------------------------------------
# Feature definitions
# ------------------------------------------------

NUMERIC_FEATURES = [
    "transaction_hour",
    "transaction_dow",
    "is_weekend",
    "is_holiday",
    "month",
    "quarter",
    "amount",
    "old_balance",
    "new_balance",
    "balance_change",
    "amount_balance_ratio",
    "is_international",
    "is_round_amount",
    "trusted_device_count",
    "new_device_flag",
    "distance_from_home_km",
    "credit_limit",
    "credit_utilization",
    "credit_score",
    "account_age_days",
    "transaction_velocity_1h",
    "high_value_transaction",
    "risk_indicator"
]

CATEGORICAL_FEATURES = [
    "customer_segment",
    "account_type",
    "currency",
    "mcc_category",
    "merchant_sector",
    "payment_rail",
    "transaction_status",
    "device_type",
    "country"
]

FEATURES = (
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)


# ------------------------------------------------
# FastAPI
# ------------------------------------------------

app = FastAPI(
    title="Banking Fraud Detection API",
    version="1.0"
)


# ------------------------------------------------
# Input Schema
# ------------------------------------------------

class TransactionInput(BaseModel):

    transaction_hour: int
    transaction_dow: int
    is_weekend: int
    is_holiday: int
    month: int
    quarter: int

    amount: float
    old_balance: float
    new_balance: float
    balance_change: float
    amount_balance_ratio: float

    is_international: int
    is_round_amount: int
    trusted_device_count: int
    new_device_flag: int

    distance_from_home_km: float
    credit_limit: float
    credit_utilization: float
    credit_score: float
    account_age_days: int

    transaction_velocity_1h: int
    high_value_transaction: int
    risk_indicator: int

    customer_segment: str
    account_type: str
    currency: str
    mcc_category: str
    merchant_sector: str
    payment_rail: str
    transaction_status: str
    device_type: str
    country: str


# ------------------------------------------------
# Health Endpoint
# ------------------------------------------------

@app.get("/api/v1/health")
def health():

    return {
        "status": "healthy",
        "model": "banking-fraud-dnn",
        "threshold": THRESHOLD
    }


# ------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------

@app.post("/api/v1/predict")
def predict(
    transaction: TransactionInput
):

    input_data = pd.DataFrame(
        [transaction.model_dump()]
    )

    input_data = input_data[
        FEATURES
    ]

    processed_data = (
        preprocessor.transform(
            input_data
        )
    )

    probability = float(
        model.predict(
            processed_data,
            verbose=0
        )[0][0]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    risk_score = probability * 100

    if risk_score < 30:
        risk_level = "Low"

    elif risk_score < 60:
        risk_level = "Medium"

    elif risk_score < 80:
        risk_level = "High"

    else:
        risk_level = "Critical"

    return {

        "prediction": prediction,

        "prediction_label":
            "Fraud"
            if prediction == 1
            else "Legitimate",

        "fraud_probability":
            round(probability, 4),

        "risk_score":
            round(risk_score, 2),

        "risk_level":
            risk_level,

        "threshold_used":
            THRESHOLD
    }