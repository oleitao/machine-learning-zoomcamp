#!/usr/bin/env python
# coding: utf-8

# ## Model for Predicting Diabetes (BRFSS 2015) - Mirror of Stroke Pipeline (no CLI)

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# Try XGBoost; fallback to sklearn if unavailable
XGB_OK = True
try:
    import xgboost as xgb
except Exception as e:
    XGB_OK = False
    XGB_IMPORT_ERROR = e
    from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

# ----------------------
# Config (no CLI)
# ----------------------
DATA_DIR = Path("data")
DATA_GLOB = "diabetes_*.csv"  # expects diabetes_012_health_indicators_BRFSS2015.csv inside data/
RANDOM_STATE = 1
N_SPLITS = 5
OUTPUT_FILE = "model_diabetes.bin"

# ----------------------
# Load
# ----------------------
files = sorted(DATA_DIR.glob(DATA_GLOB))
if not files:
    raise FileNotFoundError(f"Not found: {DATA_DIR / DATA_GLOB}")

df = pd.concat((pd.read_csv(p).assign(_source=p.name) for p in files), ignore_index=True)

# ----------------------
# Target & features
# ----------------------
# Diabetes_012: 0 = no; 1 = prediabetes; 2 = diabetes
# Binary target to mirror stroke example: {1,2} => 1 ; 0 => 0
if "Diabetes_012" not in df.columns:
    raise KeyError("Column 'Diabetes_012' not found.")
df["target"] = (df["Diabetes_012"] > 0).astype(int)

# Numerical and categorical
numerical = ["BMI", "MentHlth", "PhysHlth"]
for c in numerical:
    if c not in df.columns:
        raise KeyError(f"Expected numeric column not found: {c}")

feature_cols = [c for c in df.columns if c not in {"Diabetes_012", "target", "_source"}]
categorical = [c for c in feature_cols if c not in numerical]

# ----------------------
# Split 60/20/20
# ----------------------
df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["target"]
)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=RANDOM_STATE, stratify=df_full_train["target"]
)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train["target"].values
y_val = df_val["target"].values
y_test = df_test["target"].values

df_train = df_train.drop(columns=["target"])
df_val = df_val.drop(columns=["target"])
df_test_no_target = df_test.drop(columns=["target"])

# ----------------------
# Vectorization helpers
# ----------------------
def make_dicts(df_slice: pd.DataFrame):
    cols = [c for c in categorical + numerical if c in df_slice.columns]
    return df_slice[cols].to_dict(orient="records")

# ----------------------
# K-Fold validation on df_full_train
# ----------------------
print("Running K-Fold validation...")
skf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
scores = []

for fold, (tr_idx, va_idx) in enumerate(
    skf.split(df_full_train.drop(columns=["target"]), df_full_train["target"]), start=1
):
    df_ktrain = df_full_train.iloc[tr_idx].reset_index(drop=True)
    df_kval = df_full_train.iloc[va_idx].reset_index(drop=True)

    y_ktrain = df_ktrain["target"].values
    y_kval = df_kval["target"].values

    dv_k = DictVectorizer(sparse=False)
    X_ktrain = dv_k.fit_transform(make_dicts(df_ktrain.drop(columns=["target"])))
    X_kval = dv_k.transform(make_dicts(df_kval.drop(columns=["target"])))

    if XGB_OK:
        dtrain = xgb.DMatrix(X_ktrain, label=y_ktrain)  # let xgb infer feature names
        model = xgb.train(
            {
                "eta": 0.1,
                "max_depth": 3,
                "min_child_weight": 20,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "nthread": 0,
            },
            dtrain,
            num_boost_round=200,
        )
        dval = xgb.DMatrix(X_kval)
        y_pred = model.predict(dval)
    else:
        model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1)
        model.fit(X_ktrain, y_ktrain)
        y_pred = model.predict_proba(X_kval)[:, 1]

    auc = roc_auc_score(y_kval, y_pred)
    print(f"  Fold {fold}: AUC={auc:.4f}")
    scores.append(auc)

print(f"K-Fold AUC: mean={np.mean(scores):.4f} ± {np.std(scores):.4f}")

# ----------------------
# Train final model on full_train and evaluate on test
# ----------------------
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(make_dicts(df_full_train.drop(columns=["target"])))
y_full_train = df_full_train["target"].values

if XGB_OK:
    dfull = xgb.DMatrix(X_full_train, label=y_full_train)
    final_model = xgb.train(
        {
            "eta": 0.1,
            "max_depth": 3,
            "min_child_weight": 20,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "nthread": 0,
        },
        dfull,
        num_boost_round=200,
    )
    X_test = dv.transform(make_dicts(df_test_no_target))
    dtest = xgb.DMatrix(X_test)
    y_test_pred = final_model.predict(dtest)
else:
    final_model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1)
    final_model.fit(X_full_train, y_full_train)
    X_test = dv.transform(make_dicts(df_test_no_target))
    y_test_pred = final_model.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_test_pred)
print(f"Hold-out Test AUC: {test_auc:.4f}")

# ----------------------
# Save
# ----------------------
meta = {"backend": "xgboost" if XGB_OK else "sklearn_hgb", "categorical": categorical, "numerical": numerical}
with open(OUTPUT_FILE, "wb") as f_out:
    pickle.dump((dv, final_model, meta), f_out, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Model saved to {OUTPUT_FILE}")
if not XGB_OK:
    print("Warning: XGBoost unavailable; used HistGradientBoostingClassifier.")
    print("Import error:", repr(XGB_IMPORT_ERROR))
