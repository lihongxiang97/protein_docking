#!/usr/bin/env python3
"""Train a lightweight ML reranker for docking poses from labeled feature rows.

The intended DB5.5/decoy workflow is to train a classifier on pose features
with an ``acceptable`` label, then use its class-1 probability as the pose
reranking bonus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from docking.model_training import train_pose_reranker_frame


def train(args: argparse.Namespace) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise SystemExit(
            "Training requires joblib. Install project requirements first."
        ) from exc

    frames = [pd.read_csv(path) for path in args.input]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    try:
        result = train_pose_reranker_frame(
            frame,
            mode=args.mode,
            model_type=args.model_type,
            target=args.target,
            acceptable_dockq=args.acceptable_dockq,
            trees=args.trees,
            min_samples_leaf=args.min_samples_leaf,
            seed=args.seed,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, model_out)
    metrics_out = model_out.with_suffix(".metrics.json")
    metrics_out.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    print(f"Saved reranker to {model_out}")
    print(f"Saved metrics to {metrics_out}")
    print(result.metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more CSV files with pose features and target labels",
    )
    parser.add_argument("--model-out", required=True, help="Output .joblib model path")
    parser.add_argument(
        "--mode",
        choices=("classification", "regression"),
        default="classification",
        help="Train acceptable-pose classifier or continuous-quality regressor",
    )
    parser.add_argument(
        "--model-type",
        choices=("random_forest", "mlp"),
        default="random_forest",
        help="Training algorithm: random_forest or MLP neural network",
    )
    parser.add_argument("--target", default="acceptable", help="Target column")
    parser.add_argument(
        "--acceptable-dockq",
        type=float,
        default=0.23,
        help="DockQ threshold used to derive acceptable labels when target is absent",
    )
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
