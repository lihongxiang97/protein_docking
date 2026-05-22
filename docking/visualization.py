"""
结果可视化：2D 图表与 3D 结构 (py3Dmol)。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from docking.interface import InterfaceResult
from docking.scoring import DockingPose, ScoreComponents

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """对接与评估结果可视化。"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")

    def plot_docking_scores(self, poses: List[DockingPose], filename: str = "docking_scores.png") -> Path:
        """对接评分排名图。"""
        if not poses:
            return self.output_dir / filename

        ranks = [p.rank for p in poses]
        totals = [p.scores.total for p in poses]
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(poses)))
        ax.barh(ranks, totals, color=colors)
        ax.set_xlabel("Docking Score")
        ax.set_ylabel("Rank")
        ax.set_title("Top Docking Poses by Score")
        ax.invert_yaxis()
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_score_components(self, pose: DockingPose, filename: str = "score_components.png") -> Path:
        """评分组成堆叠图。"""
        s = pose.scores
        components = {
            "Hydrophobic": s.hydrophobic * 0.25,
            "Electrostatic": s.electrostatic * 0.20,
            "Contacts": s.contacts * 0.25,
            "Interface Area": s.interface_area * 0.20,
            "Clash Penalty": -s.clash_penalty * 0.10,
        }
        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(components.keys())
        values = list(components.values())
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
        ax.bar(names, values, color=colors)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Weighted Score Contribution")
        ax.set_title(f"Score Components (Total={s.total:.1f})")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_interface_distribution(
        self, interface: InterfaceResult, filename: str = "interface_distribution.png"
    ) -> Path:
        """界面残基分布图。"""
        rec_types = [r.contact_type for r in interface.receptor_interface]
        lig_types = [r.contact_type for r in interface.ligand_interface]
        all_types = rec_types + lig_types

        if not all_types:
            return self.output_dir / filename

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, types, title in zip(
            axes,
            [rec_types, lig_types],
            ["Receptor Interface", "Ligand Interface"],
        ):
            if types:
                from collections import Counter
                counts = Counter(types)
                ax.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%")
            ax.set_title(title)
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_contact_map(
        self, interface: InterfaceResult, filename: str = "contact_map.png"
    ) -> Path:
        """接触热图。"""
        if interface.contact_map is None or interface.contact_map.size == 0:
            return self.output_dir / filename

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(interface.contact_map, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Contact strength"})
        ax.set_xlabel("Ligand Residue Index")
        ax.set_ylabel("Receptor Residue Index")
        ax.set_title("Residue Contact Map")
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_distance_distribution(
        self, interface: InterfaceResult, filename: str = "distance_distribution.png"
    ) -> Path:
        """接触距离分布。"""
        distances = [p[2] for p in interface.contact_pairs]
        if not distances:
            return self.output_dir / filename

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(distances, bins=20, color="#3498db", edgecolor="white", alpha=0.8)
        ax.axvline(x=5.0, color="red", linestyle="--", label="Contact cutoff (5Å)")
        ax.set_xlabel("Distance (Å)")
        ax.set_ylabel("Count")
        ax.set_title("Interface Contact Distance Distribution")
        ax.legend()
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_roc_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, filename: str = "roc_curve.png"
    ) -> Path:
        """ROC 曲线。"""
        from sklearn.metrics import auc, roc_curve

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="#2ecc71", lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - PPI Prediction")
        ax.legend(loc="lower right")
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_pr_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, filename: str = "pr_curve.png"
    ) -> Path:
        """Precision-Recall 曲线。"""
        from sklearn.metrics import average_precision_score, precision_recall_curve

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color="#9b59b6", lw=2, label=f"AP = {ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, filename: str = "confusion_matrix.png"
    ) -> Path:
        """混淆矩阵。"""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Interaction", "Interaction"],
                    yticklabels=["No Interaction", "Interaction"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_benchmark_accuracy(
        self, metrics: Dict[str, float], filename: str = "benchmark_accuracy.png"
    ) -> Path:
        """Benchmark 准确率柱状图。"""
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(metrics.keys())
        values = list(metrics.values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        bars = ax.bar(names, values, color=colors)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Benchmark Performance Metrics")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontsize=9)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_rmsd_distribution(
        self, rmsd_values: List[float], filename: str = "rmsd_distribution.png"
    ) -> Path:
        """RMSD 分布图。"""
        if not rmsd_values:
            return self.output_dir / filename
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(rmsd_values, bins=15, color="#e67e22", edgecolor="white")
        ax.axvline(x=10.0, color="red", linestyle="--", label="Acceptable threshold (10Å)")
        ax.set_xlabel("RMSD (Å)")
        ax.set_ylabel("Count")
        ax.set_title("Docking RMSD Distribution")
        ax.legend()
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def generate_3d_view(
        self,
        pdb_path: Path,
        interface_residues: Optional[List[str]] = None,
        filename: str = "structure_3d.html",
    ) -> Optional[Path]:
        """生成交互式 3D 视图 (py3Dmol)。"""
        try:
            import py3Dmol
        except ImportError:
            logger.warning("py3Dmol 未安装，跳过 3D 可视化")
            return None

        pdb_path = Path(pdb_path)
        with open(pdb_path) as f:
            pdb_data = f.read()

        view = py3Dmol.view(width=800, height=600)
        view.addModel(pdb_data, "pdb")
        view.setStyle({"cartoon": {"color": "spectrum"}})

        if interface_residues:
            for res in interface_residues:
                # res format: TYR45
                resname = "".join(c for c in res if c.isalpha())[:3]
                resnum = "".join(c for c in res if c.isdigit())
                if resnum:
                    view.addStyle(
                        {"resi": resnum, "resn": resname},
                        {"stick": {"colorscheme": "yellowCarbon", "radius": 0.3}},
                    )

        view.zoomTo()
        html = view._make_html()
        path = self.output_dir / filename
        with open(path, "w") as f:
            f.write(html)
        return path

    def generate_all_plots(
        self,
        poses: List[DockingPose],
        interface: InterfaceResult,
        best_pose: Optional[DockingPose] = None,
    ) -> List[Path]:
        """生成所有对接结果图表。"""
        paths = []
        paths.append(self.plot_docking_scores(poses))
        if best_pose:
            paths.append(self.plot_score_components(best_pose))
        paths.append(self.plot_interface_distribution(interface))
        paths.append(self.plot_contact_map(interface))
        paths.append(self.plot_distance_distribution(interface))
        return [p for p in paths if p.exists()]
