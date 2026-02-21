
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import joblib
import pandas as pd
import json
import yaml
from pathlib import Path
import uvicorn

# Load config
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"

try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    # Fallback config
    config = {
        "app_name": "Retail Churn API",
        "version": "1.0.0",
        "model_path": "models/churn_rf_calibrated.pkl",
        "fallback_model_path": "models/churn_rf.pkl",
        "features_path": "models/features.json",
        "retention_policy": {"high_risk_threshold": 0.7, "medium_risk_threshold": 0.5, "low_risk_threshold": 0.3}
    }

# Load features
FEATURES_PATH = BASE_DIR / config["features_path"]
try:
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)
except FileNotFoundError:
    # Fallback features from notebook analysis
    features = ['recency_days', 'frequency_invoices', 'monetary', 'avg_order_value', 'avg_items_per_invoice', 'active_months']

# Load Model
MODEL_PATH = BASE_DIR / config["model_path"]
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / config["fallback_model_path"]

print(f"Loading model from {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = FastAPI(title=config["app_name"], version=config["version"])

# Define Input Schema dynamically
fields = {feat: (float, ...) for feat in features}
CustomerData = create_model("CustomerData", **fields)

class PredictionOutput(BaseModel):
    churn_probability: float
    churn_risk_level: str
    recommended_action: str

@app.get("/")
def home():
    return {"message": "Retail Churn Prediction API. Go to /docs for testing."}

@app.get("/health")
def health_check():
    status = "ok" if model is not None else "error"
    return {"status": status, "model": str(MODEL_PATH)}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Ensure column order matches training
        # Missing columns filled with 0 (should be handled by Pydantic validation though only present fields are passed)
        for col in features:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[features]
        
        # Predict probability
        try:
            # CalibratedClassifierCV and most classifiers expose predict_proba
            prob = model.predict_proba(input_data)[0, 1]
        except AttributeError:
             prob = float(model.predict(input_data)[0])
        except IndexError:
             # Some models might return only 1 prob if binary? No, usually [p0, p1]
             prob = float(model.predict(input_data)[0])

        # Logic for risk level and action
        high = config["retention_policy"]["high_risk_threshold"]
        medium = config["retention_policy"]["medium_risk_threshold"]
        low = config["retention_policy"]["low_risk_threshold"]
        
        if prob >= high:
            risk = "High"
            action = "Immediate Retention Campaign"
        elif prob >= medium:
            risk = "Medium"
            action = "Priority Retention Offer"
        elif prob >= low:
            risk = "Low-Medium"
            action = "Monitor / Standard Campaign"
        else:
            risk = "Low"
            action = "No Action Required"
            
        return {
            "churn_probability": round(prob, 4),
            "churn_risk_level": risk,
            "recommended_action": action
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
