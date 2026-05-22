"""
PPI 互作可能性预测：基于特征 + Random Forest 分类器。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from docking.interface import InterfaceAnalyzer, InterfaceResult
from docking.scoring import ScoreComponents
from docking.structure import HYDROPHOBIC_AA, ProteinStructure

logger = logging.getLogger(__name__)


@dataclass
class PPIPrediction:
    """PPI 预测结果。"""
    interacts: bool
    probability: float
    confidence: float
    features: Dict[str, float]
    explanation: str = ""


class PPIPredictor:
    """
    蛋白互作可能性预测器。

    特征:
    - interface_area
    - contact_residue_count
    - hydrophobic_ratio
    - electrostatic_complementarity
    - total_docking_score
    - hbond_count
    - graph_density (接触网络)
    - mean_contact_distance

    模型: Random Forest (可扩展 XGBoost/SVM)
    """

    FEATURE_NAMES = [
        "interface_area",
        "contact_residues",
        "hydrophobic_ratio",
        "electrostatic_score",
        "docking_score",
        "hbond_count",
        "clash_penalty",
        "n_interface_residues",
        "mean_contact_distance",
        "contact_density",
    ]

    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        ppi_cfg = self.config.get("ppi_prediction", {})
        self.interaction_threshold = ppi_cfg.get("interaction_threshold", 0.5)
        self.min_interface_area = ppi_cfg.get("min_interface_area", 400.0)
        self.min_contact_residues = ppi_cfg.get("min_contact_residues", 8)
        self.model_type = ppi_cfg.get("model_type", "random_forest")
        self.interface_analyzer = InterfaceAnalyzer()
        self.model: Optional[RandomForestClassifier] = None
        self.scaler = StandardScaler()
        self._is_fitted = False

    def _load_config(self, config_path: Optional[Path]) -> dict:
        path = config_path or Path(__file__).parent.parent / "config.yaml"
        path = Path(path)
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return {}

    def extract_features(
        self,
        scores: ScoreComponents,
        interface: InterfaceResult,
        receptor: ProteinStructure,
        ligand: ProteinStructure,
    ) -> np.ndarray:
        """提取 PPI 特征向量。"""
        n_rec = len(receptor.get_residue_list())
        n_lig = len(ligand.get_residue_list())
        hydro_if = sum(
            1 for r in interface.receptor_interface + interface.ligand_interface
            if r.resname in HYDROPHOBIC_AA
        )
        n_iface = max(interface.n_interface_residues, 1)
        hydro_ratio = hydro_if / n_iface

        distances = [p[2] for p in interface.contact_pairs]
        mean_dist = np.mean(distances) if distances else 10.0

        features = {
            "interface_area": scores.raw_interface_area,
            "contact_residues": scores.contact_residues,
            "hydrophobic_ratio": hydro_ratio,
            "electrostatic_score": scores.electrostatic,
            "docking_score": scores.total,
            "hbond_count": scores.hbonds,
            "clash_penalty": scores.clash_penalty,
            "n_interface_residues": interface.n_interface_residues,
            "mean_contact_distance": mean_dist,
            "contact_density": scores.contact_residues / max(n_rec + n_lig, 1),
        }
        return np.array([features[name] for name in self.FEATURE_NAMES])

    def predict(
        self,
        scores: ScoreComponents,
        interface: InterfaceResult,
        receptor: ProteinStructure,
        ligand: ProteinStructure,
    ) -> PPIPrediction:
        """预测是否存在互作。"""
        features_vec = self.extract_features(scores, interface, receptor, ligand)
        features_dict = dict(zip(self.FEATURE_NAMES, features_vec))

        if self._is_fitted and self.model is not None:
            X = self.scaler.transform(features_vec.reshape(1, -1))
            prob = self.model.predict_proba(X)[0, 1]
            interacts = prob >= self.interaction_threshold
            confidence = abs(prob - 0.5) * 2
        else:
            # 基于规则的回退模型
            prob, interacts, confidence = self._rule_based_predict(
                scores, interface, features_dict
            )

        explanation = self._generate_explanation(features_dict, prob, interacts)
        return PPIPrediction(
            interacts=interacts,
            probability=float(prob),
            confidence=float(confidence),
            features=features_dict,
            explanation=explanation,
        )

    def _rule_based_predict(
        self,
        scores: ScoreComponents,
        interface: InterfaceResult,
        features: Dict[str, float],
    ) -> Tuple[float, bool, float]:
        """
        规则评分模型 (无训练数据时使用)。
        综合界面面积、接触数、对接分、碰撞惩罚。
        """
        score = 0.0
        if features["interface_area"] > self.min_interface_area:
            score += 0.25
        if features["contact_residues"] >= self.min_contact_residues:
            score += 0.25
        if features["docking_score"] > 30:
            score += 0.20
        if features["hbond_count"] >= 3:
            score += 0.15
        if features["hydrophobic_ratio"] > 0.2:
            score += 0.10
        if features["clash_penalty"] < 10:
            score += 0.05

        prob = min(max(score, 0.05), 0.95)
        interacts = prob >= self.interaction_threshold
        confidence = abs(prob - 0.5) * 2
        return prob, interacts, confidence

    def _generate_explanation(
        self, features: Dict[str, float], prob: float, interacts: bool
    ) -> str:
        status = "可能存在互作" if interacts else "互作可能性较低"
        return (
            f"{status} (概率={prob:.2%})。"
            f"界面面积={features['interface_area']:.1f}Å², "
            f"接触残基={features['contact_residues']:.0f}, "
            f"对接评分={features['docking_score']:.1f}, "
            f"氢键={features['hbond_count']:.0f}。"
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
    ) -> None:
        """训练 Random Forest 分类器。"""
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        )
        self.model.fit(X_scaled, y)
        self._is_fitted = True
        logger.info("PPI 模型训练完成: %d 样本", len(y))

    def save_model(self, path: Path) -> None:
        import joblib
        path = Path(path)
        joblib.dump({"model": self.model, "scaler": self.scaler, "fitted": self._is_fitted}, path)

    def load_model(self, path: Path) -> None:
        import joblib
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._is_fitted = data["fitted"]
