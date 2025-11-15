#!/usr/bin/env python
# coding: utf-8

"""
Flask prediction service for Diabetes (BRFSS 2015) -- mirrors FinalModelpredict.py style.

- Loads a pickle produced by model-train-diabetes.py (default: model_diabetes.bin).
- Exposes a POST /diabetespredict/ endpoint that accepts a JSON body with BRFSS features, e.g.:
  {
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 31.2, "Smoker": 0, "Stroke": 0,
    "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3,
    "MentHlth": 2, "PhysHlth": 5, "DiffWalk": 0, "Sex": 1, "Age": 9, "Education": 5, "Income": 4
  }
- Returns: {"diabetes_prediction": <float>, "diabetes": <bool>} with a default threshold of 0.5.
"""

import os
import pickle
from typing import Dict, Any, Tuple

from flask import Flask, jsonify, request

DEFAULT_MODEL_FILE = os.environ.get("MODEL_FILE", "model_diabetes.bin")
DEFAULT_THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

# Load model
with open(DEFAULT_MODEL_FILE, "rb") as f_in:
    data = pickle.load(f_in)

# Support both (dv, model) and (dv, model, meta)
if isinstance(data, tuple) and len(data) == 2:
    dv, model = data
    meta = {"backend": "xgboost"}  # assume original format
elif isinstance(data, tuple) and len(data) == 3:
    dv, model, meta = data
else:
    raise RuntimeError("Modelo inválido. Esperava (dv, model) ou (dv, model, meta).")

BACKEND = meta.get("backend", "xgboost")
CATEGORICAL = meta.get("categorical")
NUMERICAL = meta.get("numerical")

# Fallback feature whitelist if meta is missing
DEFAULT_FEATURES = [
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack",
    "PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost",
    "GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"
]

FEATURES = (list(CATEGORICAL or []) + list(NUMERICAL or [])) or DEFAULT_FEATURES

# Try to import xgboost for Booster inference
try:
    import xgboost as xgb  # type: ignore
    XGB_OK = True
except Exception:
    XGB_OK = False

app = Flask("diabetes-prediction-service")

def prepare_features(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Select only known features; missing ones are left missing (DV handles unseen as zeros)."""
    return {k: payload.get(k) for k in FEATURES if k in payload}

def predict_proba_one(row: Dict[str, Any]) -> float:
    X = dv.transform([row])
    if BACKEND == "xgboost" and XGB_OK:
        dmat = xgb.DMatrix(X)
        pred = model.predict(dmat)  # returns array([p])
        return float(pred[0])
    elif hasattr(model, "predict_proba"):
        pred = model.predict_proba(X)[:, 1]
        return float(pred[0])
    elif hasattr(model, "predict"):
        # Some boosters expose only predict with proba
        pred = model.predict(X)
        # If this looks like probability-providing estimator
        val = float(pred[0])
        if 0.0 <= val <= 1.0:
            return val
        # otherwise map class 0/1 to proba-like
        return 1.0 if val >= 1.0 else 0.0
    else:
        raise RuntimeError("Modelo não suporta predict nem predict_proba.")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "backend": BACKEND, "n_features": len(FEATURES)}), 200

@app.route("/diabetespredict/", methods=["POST"])
def diabetes_predict():
    person = request.get_json()
    if not isinstance(person, dict):
        return jsonify({"error": "JSON inválido"}), 400

    features = prepare_features(person)
    if not features:
        return jsonify({"error": "Nenhuma feature reconhecida no payload", "expected_features": FEATURES}), 400

    y_pred = predict_proba_one(features)
    diabetes = bool(y_pred >= DEFAULT_THRESHOLD)

    result = {
        "diabetes_prediction": float(y_pred),
        "diabetes": diabetes,
        "threshold": DEFAULT_THRESHOLD,
        "backend": BACKEND
    }
    return jsonify(result)

if __name__ == "__main__":
    # host 0.0.0.0 útil para docker; debug opcional via FLASK_DEBUG=1
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
