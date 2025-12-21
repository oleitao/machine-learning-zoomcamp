import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from xg_futebol.data_prep import FEATURE_COLUMNS, shot_event_to_features


DEFAULT_MODEL_FILE = "models/xg_logreg_sigmoid.joblib"
MODEL_FILE = os.getenv("XG_MODEL_FILE", DEFAULT_MODEL_FILE)

app = Flask("xg_futebol")


def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}. "
            "Train a model with train.py and set XG_MODEL_FILE."
        )
    return joblib.load(MODEL_FILE)


model = load_model()


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive a JSON with a shot event (StatsBomb format or simplified)
    and return the goal probability.
    """
    event = request.get_json()
    if event is None:
        return jsonify({"error": "Invalid JSON"}), 400

    features = shot_event_to_features(event)

    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    prob = float(model.predict_proba(X)[0, 1])

    result = {
        "prob_goal": prob,
        "features": features,
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
