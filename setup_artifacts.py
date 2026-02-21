
import pandas as pd
import json
import yaml
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
APP_DIR = BASE_DIR / "app"

MODELS_DIR.mkdir(exist_ok=True, parents=True)
APP_DIR.mkdir(exist_ok=True, parents=True)

print("Loading training data to identify features...")
try:
    # Mimic notebook logic
    df = pd.read_csv(DATA_DIR / "customer_features_train.csv")
    # Identify numeric columns used in training
    # columns in notebook: customer_id, recency_days, frequency_invoices, monetary, avg_order_value, avg_items_per_invoice, active_months, last_purchase
    # drops: customer_id, last_purchase
    
    cols_to_drop = ["customer_id", "last_purchase"]
    features = [c for c in df.columns if c not in cols_to_drop and pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"Identified {len(features)} features: {features}")
    
    # Save feature list
    feature_path = MODELS_DIR / "features.json"
    with open(feature_path, "w") as f:
        json.dump(features, f, indent=4)
    print(f"Saved feature list to {feature_path}")

    # Create config.yaml
    config = {
        "app_name": "Retail Churn Prediction API",
        "version": "1.0.0",
        "model_path": "models/churn_rf_calibrated.pkl",
        "fallback_model_path": "models/churn_rf.pkl",
        "features_path": "models/features.json",
        "retention_policy": {
            "high_risk_threshold": 0.7,
            "medium_risk_threshold": 0.5,
            "low_risk_threshold": 0.3
        }
    }
    
    config_path = BASE_DIR / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config to {config_path}")

    # Create requirements.txt
    requirements = """
fastapi>=0.68.0
uvicorn>=0.15.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.1.0
pyyaml>=5.4.1
pydantic>=1.8.0
shap>=0.40.0
streamlit>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
    """.strip()
    
    req_path = BASE_DIR / "requirements.txt"
    with open(req_path, "w") as f:
        f.write(requirements)
    print(f"Saved requirements to {req_path}")

except Exception as e:
    print(f"Error: {e}")
