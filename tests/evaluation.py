"""
性能评估与自动报告生成。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from docking.visualization import ResultVisualizer
from tests.benchmark_test import BenchmarkResult

logger = logging.getLogger(__name__)


class EvaluationReport:
    """评估报告生成器。"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = ResultVisualizer(self.output_dir / "plots")

    def compute_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """计算分类指标。"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }
        if len(np.unique(y_true)) > 1:
            metrics["auroc"] = roc_auc_score(y_true, y_prob)
            metrics["auprc"] = average_precision_score(y_true, y_prob)
        else:
            metrics["auroc"] = 0.0
            metrics["auprc"] = 0.0
        return metrics

    def compute_docking_metrics(self, results: List[BenchmarkResult]) -> Dict[str, float]:
        """计算对接结构指标。"""
        rmsd_vals = [r.rmsd for r in results if r.rmsd is not None and r.label == 1]
        irmsd_vals = [r.irmsd for r in results if r.irmsd is not None and r.label == 1]
        fnat_vals = [r.fnat for r in results if r.fnat is not None and r.label == 1]
        dockq_vals = [r.dockq for r in results if r.dockq is not None and r.label == 1]

        metrics = {}
        if rmsd_vals:
            metrics["mean_lrmsd"] = float(np.mean(rmsd_vals))
            metrics["median_lrmsd"] = float(np.median(rmsd_vals))
            metrics["mean_rmsd"] = float(np.mean(rmsd_vals))
            metrics["median_rmsd"] = float(np.median(rmsd_vals))
            metrics["success_rate_10A"] = float(np.mean([r < 10 for r in rmsd_vals]))
        if irmsd_vals:
            metrics["mean_irmsd"] = float(np.mean(irmsd_vals))
            metrics["success_rate_irmsd_4A"] = float(np.mean([r <= 4 for r in irmsd_vals]))
        if fnat_vals:
            metrics["mean_fnat"] = float(np.mean(fnat_vals))
        if dockq_vals:
            metrics["mean_dockq"] = float(np.mean(dockq_vals))
            metrics["acceptable_dockq_rate"] = float(np.mean([q >= 0.23 for q in dockq_vals]))
        return metrics

    def compute_threshold_diagnostic(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Find the benchmark threshold with the best MCC as a diagnostic only."""
        thresholds = np.unique(np.concatenate([y_prob, y_prob - 1e-9, y_prob + 1e-9, [0.5]]))
        best: Dict[str, float] = {}
        best_key = (-np.inf, -np.inf, -np.inf)
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            metrics = self.compute_classification_metrics(y_true, y_pred, y_prob)
            key = (metrics["mcc"], metrics["accuracy"], metrics["f1_score"])
            if key > best_key:
                best_key = key
                best = {"threshold": float(threshold), **metrics}
        return best

    def generate_full_report(self, results: List[BenchmarkResult]) -> Path:
        """生成完整评估报告。"""
        if not results:
            logger.warning("无 benchmark 结果")
            return self.output_dir / "evaluation_report.md"

        y_true = np.array([r.label for r in results])
        y_pred = np.array([r.predicted_label for r in results])
        y_prob = np.array([r.probability for r in results])

        cls_metrics = self.compute_classification_metrics(y_true, y_pred, y_prob)
        threshold_diagnostic = self.compute_threshold_diagnostic(y_true, y_prob)
        dock_metrics = self.compute_docking_metrics(results)

        # 生成图表
        self.visualizer.plot_roc_curve(y_true, y_prob)
        self.visualizer.plot_pr_curve(y_true, y_prob)
        self.visualizer.plot_confusion_matrix(y_true, y_pred)
        self.visualizer.plot_benchmark_accuracy(cls_metrics)

        rmsd_vals = [r.rmsd for r in results if r.rmsd is not None]
        if rmsd_vals:
            self.visualizer.plot_rmsd_distribution(rmsd_vals)

        scores = [r.docking_score for r in results]
        if scores:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(scores, bins=15, color="#3498db", edgecolor="white")
            ax.set_xlabel("Docking Score")
            ax.set_title("Docking Score Distribution")
            plt.tight_layout()
            fig.savefig(self.output_dir / "plots" / "score_distribution.png", dpi=150)
            plt.close(fig)

        # Markdown 报告
        report_path = self.output_dir / "evaluation_report.md"
        self._write_markdown_report(
            report_path,
            results,
            cls_metrics,
            dock_metrics,
            threshold_diagnostic,
        )
        return report_path

    def _write_markdown_report(
        self,
        path: Path,
        results: List[BenchmarkResult],
        cls_metrics: Dict[str, float],
        dock_metrics: Dict[str, float],
        threshold_diagnostic: Optional[Dict[str, float]] = None,
    ) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("# PPI Docking Benchmark Evaluation Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total test cases**: {len(results)}\n")
            f.write(f"- Positive (interacting): {sum(1 for r in results if r.label == 1)}\n")
            f.write(f"- Negative (non-interacting): {sum(1 for r in results if r.label == 0)}\n\n")

            f.write("## 1. Software Overview\n\n")
            f.write("PPI Docking is a Python-based protein-protein interaction prediction ")
            f.write("and rigid-body docking software with self-implemented algorithms.\n\n")

            f.write("## 2. Method Pipeline\n\n")
            f.write("1. Structure preprocessing (water removal, hydrogen addition)\n")
            f.write("2. SASA-based surface residue identification\n")
            f.write("3. Grid search + Monte Carlo rigid-body docking\n")
            f.write("4. Multi-component scoring function\n")
            f.write("5. Interface residue prediction\n")
            f.write("6. Random Forest / rule-based PPI classification\n\n")

            f.write("## 3. Classification Performance\n\n")
            f.write("| Metric | Value |\n|--------|-------|\n")
            for name, val in cls_metrics.items():
                f.write(f"| {name.upper()} | {val:.4f} |\n")
            f.write("\n")
            unique_predictions = {r.predicted_label for r in results}
            if len(unique_predictions) == 1 and len(results) > 1:
                f.write(
                    "**Warning**: all benchmark cases received the same class prediction. "
                    "This usually means the operating threshold or rule-based PPI "
                    "calibration is saturated.\n\n"
                )
            if threshold_diagnostic:
                f.write("### Threshold Diagnostic\n\n")
                f.write(
                    "The table below is computed on this benchmark split and is intended "
                    "for calibration diagnosis, not as an independent test result.\n\n"
                )
                f.write("| Metric | Value |\n|--------|-------|\n")
                for name, val in threshold_diagnostic.items():
                    f.write(f"| {name.upper()} | {val:.4f} |\n")
                f.write("\n")

            f.write("## 4. Docking Structure Metrics\n\n")
            if dock_metrics:
                f.write("| Metric | Value |\n|--------|-------|\n")
                for name, val in dock_metrics.items():
                    f.write(f"| {name} | {val:.4f} |\n")
            else:
                f.write("No docking metrics available.\n")
            f.write("\n")

            f.write("## 5. Algorithm Complexity\n\n")
            f.write("- **SASA computation**: O(N log N) using KD-tree\n")
            f.write("- **Coarse docking**: O(R x T x N) where R=rotations, T=translations, N=atoms\n")
            f.write("- **Monte Carlo refinement**: O(M x N), M=iterations\n")
            f.write("- **Scoring**: O(N log N) per pose\n")
            f.write("- **Space**: O(N x P) for P poses\n\n")

            f.write("## 6. Figures\n\n")
            f.write("- `plots/roc_curve.png` - ROC curve\n")
            f.write("- `plots/pr_curve.png` - Precision-Recall curve\n")
            f.write("- `plots/confusion_matrix.png` - Confusion matrix\n")
            f.write("- `plots/benchmark_accuracy.png` - Metric bar chart\n")
            f.write("- `plots/rmsd_distribution.png` - RMSD distribution\n\n")

            f.write("## 7. Limitations\n\n")
            f.write("- Rigid-body docking without induced-fit\n")
            f.write("- Simplified SASA and hydrogen bond detection\n")
            f.write("- Benchmark may use synthetic structures when PDB download fails\n")
            f.write("- CAPRI/DockQ metrics depend on correct native chain mapping\n\n")

            f.write("## 8. Future Optimization\n\n")
            f.write("- Flexible docking with side-chain sampling\n")
            f.write("- FFT-based global search\n")
            f.write("- Graph neural network for interface prediction\n")
            f.write("- GPU acceleration for spatial search\n")
            f.write("- Integration with AlphaFold2 predicted structures\n\n")

            f.write("## 9. Detailed Results\n\n")
            f.write("| PDB ID | Label | Predicted | Prob | Score | LRMSD | iRMSD | FNAT | DockQ | CAPRI |\n")
            f.write("|--------|-------|-----------|------|-------|-------|-------|------|-------|-------|\n")
            for r in results:
                rmsd_str = f"{r.rmsd:.2f}" if r.rmsd is not None else "N/A"
                irmsd_str = f"{r.irmsd:.2f}" if r.irmsd is not None else "N/A"
                fnat_str = f"{r.fnat:.3f}" if r.fnat is not None else "N/A"
                dockq_str = f"{r.dockq:.3f}" if r.dockq is not None else "N/A"
                capri = r.capri_class or "N/A"
                f.write(
                    f"| {r.pdb_id} | {r.label} | {r.predicted_label} | "
                    f"{r.probability:.3f} | {r.docking_score:.1f} | {rmsd_str} | "
                    f"{irmsd_str} | {fnat_str} | {dockq_str} | {capri} |\n"
                )

        logger.info("评估报告已生成: %s", path)
