"""Training utilities for pose reranking models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from docking.ml_reranker import POSE_FEATURES


@dataclass
class PoseTrainingResult:
    model: object
    metrics: Dict[str, float | int | str]


def train_pose_reranker_frame(
    frame: pd.DataFrame,
    *,
    mode: str = "classification",
    model_type: str = "random_forest",
    target: str = "acceptable",
    acceptable_dockq: float = 0.23,
    trees: int = 300,
    min_samples_leaf: int = 2,
    seed: int = 42,
) -> PoseTrainingResult:
    """Train a pose reranker from a DataFrame of pose features and labels."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            mean_absolute_error,
            r2_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Training requires scikit-learn. Install project requirements first."
        ) from exc

    mode = mode.lower()
    model_type = model_type.lower()
    if mode not in {"classification", "regression"}:
        raise ValueError("mode must be classification or regression")
    if model_type not in {"random_forest", "mlp"}:
        raise ValueError("model_type must be random_forest or mlp")

    missing = [name for name in POSE_FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    if target not in frame.columns and not (
        mode == "classification" and target == "acceptable" and "dockq" in frame.columns
    ):
        raise ValueError(f"Missing target column: {target}")

    frame = frame.copy()
    if mode == "classification" and target == "acceptable" and target not in frame.columns:
        frame[target] = (frame["dockq"].astype(float) >= acceptable_dockq).astype(int)

    frame = frame.dropna(subset=[target])
    if len(frame) < 10:
        raise ValueError("Need at least 10 labeled rows to train a reranker")

    x = frame[POSE_FEATURES]
    y = frame[target].astype(int if mode == "classification" else float)
    test_size = min(0.25, max(0.1, 5 / len(frame)))
    class_counts = y.value_counts() if mode == "classification" else None
    stratify = (
        y
        if mode == "classification"
        and y.nunique() > 1
        and class_counts is not None
        and int(class_counts.min()) >= 2
        else None
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    if model_type == "random_forest":
        estimator = (
            RandomForestClassifier(
                n_estimators=trees,
                random_state=seed,
                min_samples_leaf=min_samples_leaf,
                class_weight="balanced",
                n_jobs=-1,
            )
            if mode == "classification"
            else RandomForestRegressor(
                n_estimators=trees,
                random_state=seed,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-1,
            )
        )
    else:
        estimator = (
            MLPClassifier(
                hidden_layer_sizes=(96, 48, 24),
                activation="relu",
                alpha=1e-4,
                max_iter=500,
                early_stopping=True,
                random_state=seed,
            )
            if mode == "classification"
            else MLPRegressor(
                hidden_layer_sizes=(96, 48, 24),
                activation="relu",
                alpha=1e-4,
                max_iter=500,
                early_stopping=True,
                random_state=seed,
            )
        )

    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )
    model.fit(x_train, y_train)

    if mode == "classification":
        predictions = model.predict(x_test)
        probabilities = _positive_probabilities(model, x_test)
        metrics = {
            "rows": int(len(frame)),
            "mode": mode,
            "model_type": model_type,
            "target": target,
            "positive_rows": int(y.sum()),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "average_precision": float(average_precision_score(y_test, probabilities))
            if y_test.nunique() > 1
            else 0.0,
            "roc_auc": float(roc_auc_score(y_test, probabilities))
            if y_test.nunique() > 1
            else 0.0,
        }
    else:
        predictions = model.predict(x_test)
        metrics = {
            "rows": int(len(frame)),
            "mode": mode,
            "model_type": model_type,
            "target": target,
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)) if len(y_test) > 1 else 0.0,
        }
    return PoseTrainingResult(model=model, metrics=metrics)


def _positive_probabilities(model, x_test) -> object:
    if not hasattr(model, "predict_proba"):
        return model.predict(x_test)
    probability_matrix = model.predict_proba(x_test)
    if probability_matrix.ndim == 2 and probability_matrix.shape[1] > 1:
        return probability_matrix[:, 1]
    return probability_matrix[:, 0]
