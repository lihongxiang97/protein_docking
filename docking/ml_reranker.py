"""Optional machine-learning reranker for docking poses."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from docking.scoring import DockingPose

logger = logging.getLogger(__name__)

POSE_FEATURES = [
    "raw_total_score",
    "hydrophobic_score",
    "electrostatic_score",
    "contact_score",
    "interface_area_score",
    "clash_penalty",
    "hbond_count",
    "contact_residues",
    "interface_area_angstrom2",
    "restraint_score",
    "restraint_violations",
    "prior_score",
    "search_score",
    "cluster_size",
]


def pose_feature_dict(pose: DockingPose) -> Dict[str, float]:
    scores = pose.scores
    return {
        "raw_total_score": float(scores.total),
        "hydrophobic_score": float(scores.hydrophobic),
        "electrostatic_score": float(scores.electrostatic),
        "contact_score": float(scores.contacts),
        "interface_area_score": float(scores.interface_area),
        "clash_penalty": float(scores.clash_penalty),
        "hbond_count": float(scores.hbonds),
        "contact_residues": float(scores.contact_residues),
        "interface_area_angstrom2": float(scores.raw_interface_area),
        "restraint_score": float(scores.restraint_score),
        "restraint_violations": float(scores.restraint_violations),
        "prior_score": float(scores.prior_score),
        "search_score": float(pose.search_score),
        "cluster_size": float(pose.cluster_size),
    }


def pose_feature_matrix(poses: Iterable[DockingPose]) -> np.ndarray:
    rows = []
    for pose in poses:
        features = pose_feature_dict(pose)
        rows.append([features[name] for name in POSE_FEATURES])
    return np.asarray(rows, dtype=float)


class PoseReranker:
    """Load a trained scikit-learn/joblib model and rerank poses."""

    def __init__(self, model, weight: float = 1.0):
        self.model = model
        self.weight = float(weight)

    def predict(self, poses: List[DockingPose]) -> np.ndarray:
        if not poses:
            return np.zeros(0, dtype=float)
        return np.asarray(self.model.predict(pose_feature_matrix(poses)), dtype=float)

    def rank_poses(self, poses: List[DockingPose]) -> List[DockingPose]:
        predictions = self.predict(poses)
        for pose, prediction in zip(poses, predictions):
            pose.scores.total += self.weight * float(prediction)
        ranked = sorted(poses, key=lambda pose: pose.scores.total, reverse=True)
        for index, pose in enumerate(ranked, 1):
            pose.rank = index
        return ranked


def load_pose_reranker(path: Optional[str], weight: float = 1.0) -> Optional[PoseReranker]:
    if not path:
        return None
    model_path = Path(path)
    if not model_path.exists():
        logger.warning("Pose reranker model not found: %s", model_path)
        return None
    try:
        import joblib

        return PoseReranker(joblib.load(model_path), weight=weight)
    except Exception as exc:
        logger.warning("Failed to load pose reranker %s: %s", model_path, exc)
        return None
