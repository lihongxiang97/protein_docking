#!/usr/bin/env python3
"""Train a lightweight ML reranker for docking poses from labeled feature rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from docking.ml_reranker import POSE_FEATURES


def train(args: argparse.Namespace) -> None:
    try:
        import joblib
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "Training requires scikit-learn and joblib. Install project requirements first."
        ) from exc

    frame = pd.read_csv(args.input)
    missing = [name for name in POSE_FEATURES if name not in frame.columns]
    if missing:
        raise SystemExit(f"Missing feature columns: {', '.join(missing)}")
    if args.target not in frame.columns:
        raise SystemExit(f"Missing target column: {args.target}")

    frame = frame.dropna(subset=[args.target])
    if len(frame) < 10:
        raise SystemExit("Need at least 10 labeled rows to train a reranker")

    x = frame[POSE_FEATURES]
    y = frame[args.target].astype(float)
    test_size = min(0.25, max(0.1, 5 / len(frame)))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=args.seed
    )
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=args.trees,
                    random_state=args.seed,
                    min_samples_leaf=args.min_samples_leaf,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "rows": int(len(frame)),
        "target": args.target,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)) if len(y_test) > 1 else 0.0,
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    metrics_out = model_out.with_suffix(".metrics.json")
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved reranker to {model_out}")
    print(f"Saved metrics to {metrics_out}")
    print(metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with pose features and target")
    parser.add_argument("--model-out", required=True, help="Output .joblib model path")
    parser.add_argument("--target", default="dockq", help="Regression target column")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(func=train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
