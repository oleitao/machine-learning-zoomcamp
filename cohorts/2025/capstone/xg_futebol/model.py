from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_prep import FEATURE_COLUMNS, TARGET_COLUMN, load_shots_dataframe


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "distance",
        "angle",
        "minute",
        "second",
        "under_pressure",
        "first_time",
    ]

    categorical_features = [
        "play_pattern",
        "body_part",
        "shot_type",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def build_base_models() -> Dict[str, object]:
    models: Dict[str, object] = {}

    models["logreg"] = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )

    models["random_forest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    return models


def evaluate_probabilities(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    metrics = {
        "logloss": float(log_loss(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }
    return metrics


def train_model(
    events_root: Path,
    model_name: str,
    calibration: str = "none",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, float], Path]:
    """
    Train an xG model and return metrics and the path to the saved model file.
    """
    df = load_shots_dataframe(events_root)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor()

    base_models = build_base_models()
    if model_name not in base_models:
        raise ValueError(f"Unknown model: {model_name}")

    base_model = base_models[model_name]

    clf = base_model
    calibration = calibration.lower()
    calibration_suffix = ""

    if calibration in ("sigmoid", "isotonic"):
        clf = CalibratedClassifierCV(
            base_estimator=base_model,
            cv=5,
            method=calibration,
        )
        calibration_suffix = f"_{calibration}"
    elif calibration not in ("", "none"):
        raise ValueError(f"Unknown calibration method: {calibration}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_probabilities(y_test, y_proba)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_filename = f"xg_{model_name}{calibration_suffix}.joblib"
    model_path = models_dir / model_filename

    joblib.dump(pipeline, model_path)

    return metrics, model_path
