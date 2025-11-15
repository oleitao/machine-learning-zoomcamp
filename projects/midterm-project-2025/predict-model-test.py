#!/usr/bin/env python
# coding: utf-8


"""
Client test script for Diabetes prediction API (mirrors FinalModelpredicttest.py style).

- Sends a sample patient's BRFSS 2015 health indicators as JSON.
- Expects a local Flask endpoint by default: http://127.0.0.1:5000/diabetespredict/
- Prints the raw JSON response and a friendly message.

Response handling is robust:
- If server returns {"diabetes": true/false}, we use that.
- Else if it returns {"prediction": <float> or int} or {"probability": <float>},
  we interpret >= 0.5 as positive.
"""

import argparse
import json
import sys
import requests

DEFAULT_URL = "http://127.0.0.1:5000/diabetespredict/"

parser = argparse.ArgumentParser()
parser.add_argument("--url", default=DEFAULT_URL, help=f"Endpoint URL (default: {DEFAULT_URL})")
parser.add_argument("--person-id", default="person-abc123", help="Identifier for logging (default: person-abc123)")
args = parser.parse_args()

url = args.url
person_id = args.person_id

# BRFSS 2015 columns (features): excluding Diabetes_012 (target)
# Types: most are 0/1 flags or small integer categories; BMI/MentHlth/PhysHlth are numeric.
sample_person = {
    "HighBP": 1,
    "HighChol": 1,
    "CholCheck": 1,
    "BMI": 31.2,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 3,        # 1=excellent ... 5=poor
    "MentHlth": 2,       # days of poor mental health (0-30)
    "PhysHlth": 5,       # days of poor physical health (0-30)
    "DiffWalk": 0,
    "Sex": 1,            # 0=female, 1=male per dataset
    "Age": 9,            # categorical age bin per dataset (e.g., 9≈55-59)
    "Education": 5,      # 1..6
    "Income": 4          # 1..8
}

try:
    response = requests.post(url, json=sample_person, timeout=30)
    response.raise_for_status()
    payload = response.json()
except requests.exceptions.RequestException as e:
    print(f"[ERROR] Failed to call {url}: {e}", file=sys.stderr)
    sys.exit(1)
except ValueError:
    print("[ERROR] Response is not valid JSON:", response.text, file=sys.stderr)
    sys.exit(1)

print("Raw response JSON:")
print(json.dumps(payload, indent=2, ensure_ascii=False))

# Interpret response
label = None
if isinstance(payload, dict):
    if "diabetes" in payload:
        val = payload["diabetes"]
        if isinstance(val, bool):
            label = val
        else:
            # try numeric
            try:
                label = float(val) >= 0.5
            except Exception:
                label = None
    elif "probability" in payload:
        try:
            label = float(payload["probability"]) >= 0.5
        except Exception:
            label = None
    elif "prediction" in payload:
        try:
            # could be 0/1 or probability
            pred = float(payload["prediction"])
            label = (pred >= 0.5) if pred <= 1.0 else (pred >= 1.0)
        except Exception:
            label = None

if label is None:
    print(f"Couldn't infer a boolean prediction from server response for {person_id}.", file=sys.stderr)
    sys.exit(2)

if label:
    print(f"person is predicted DIABETIC {person_id}")
else:
    print(f"person is predicted NON-DIABETIC {person_id}")
