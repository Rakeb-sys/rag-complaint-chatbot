# src/train_model.py
import os
import joblib
import pandas as pd
from xgboost import XGBClassifier

def train_and_save(
    data_path="data/processed/filtered_complaints.csv",
    model_output_path="models/xgb_model.pkl"
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    df = pd.read_csv(data_path)
    
    # Extract numerical features excluding identifiers/targets
    X = df.drop(columns=["target", "complaint_id"], errors="ignore").select_dtypes(include=["number"])
    
    if "target" in df.columns:
        y = df["target"]
    else:
        y = (X.iloc[:, 0] > X.iloc[:, 0].median()).astype(int)

    model = XGBClassifier(n_estimators=30, max_depth=3, random_state=42)
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model successfully saved to {model_output_path}")

if __name__ == "__main__":
    train_and_save()