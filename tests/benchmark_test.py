"""
Benchmark 自动测试流程。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from docking.docking import ProteinDocker
from docking.interface import InterfaceAnalyzer
from docking.ppi_predictor import PPIPredictor
from tests.benchmark_data import (
    BenchmarkPair,
    generate_example_pair,
    prepare_benchmark_dataset,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """单个 benchmark 结果。"""
    pdb_id: str
    label: int
    predicted_label: int
    probability: float
    docking_score: float
    interface_area: float
    contact_residues: int
    rmsd: Optional[float] = None
    fnat: Optional[float] = None
    dockq: Optional[float] = None
    runtime_seconds: float = 0.0
    interface_precision: Optional[float] = None
    interface_recall: Optional[float] = None


class BenchmarkRunner:
    """Benchmark 测试运行器。"""

    def __init__(self, config_path: Path, project_root: Path):
        self.config_path = Path(config_path)
        self.project_root = Path(project_root)
        with open(self.config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.data_dir = self.project_root / "data" / "benchmark"
        self.results_dir = self.project_root / "results" / "benchmark"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self, download: bool = True) -> List[BenchmarkResult]:
        """运行全部 benchmark。"""
        positive, negative = prepare_benchmark_dataset(self.data_dir, download=download)
        all_pairs = positive + negative
        results = []

        docker = ProteinDocker(self.config_path)
        predictor = PPIPredictor(self.config_path)
        interface_analyzer = InterfaceAnalyzer()

        for pair in all_pairs:
            if pair.receptor_path is None or pair.ligand_path is None:
                continue
            if not pair.receptor_path.exists():
                continue

            logger.info("测试: %s (label=%d)", pair.pdb_id, pair.label)
            t0 = time.time()

            try:
                out_subdir = self.results_dir / pair.pdb_id
                poses, receptor, ligand = docker.dock(
                    pair.receptor_path, pair.ligand_path, out_subdir,
                )
                if not poses:
                    continue

                best = poses[0]
                lig_struct = ligand
                if best.complex_structure:
                    from docking.structure import split_complex_by_reference_chains
                    _, lig_struct = split_complex_by_reference_chains(
                        best.complex_structure,
                        receptor.chains,
                    )

                interface = interface_analyzer.analyze(receptor, lig_struct)
                ppi = predictor.predict(best.scores, interface, receptor, lig_struct)

                rmsd, fnat, dockq = None, None, None
                if pair.label == 1 and pair.native_complex_path:
                    rmsd, fnat, dockq = self._compute_standard_metrics(
                        best,
                        pair.native_complex_path,
                        pair.receptor_chain,
                        pair.ligand_chain,
                    )

                result = BenchmarkResult(
                    pdb_id=pair.pdb_id,
                    label=pair.label,
                    predicted_label=1 if ppi.interacts else 0,
                    probability=ppi.probability,
                    docking_score=best.scores.total,
                    interface_area=best.scores.raw_interface_area,
                    contact_residues=best.scores.contact_residues,
                    rmsd=rmsd,
                    fnat=fnat,
                    dockq=dockq,
                    runtime_seconds=time.time() - t0,
                )
                results.append(result)

            except Exception as e:
                logger.error("测试失败 %s: %s", pair.pdb_id, e)

        # 保存结果
        self._save_results(results)
        return results

    def _compute_legacy_docking_metrics(
        self,
        pose,
        native_pdb: Path,
        receptor_chain: str,
        ligand_chain: str,
    ) -> tuple:
        """计算 RMSD, FNAT, DockQ (简化)。"""
        try:
            from docking.structure import PDBParser
            native = PDBParser.parse(native_pdb)
            if pose.ligand_coords is None or len(pose.ligand_coords) == 0:
                return None, None, None

            native_coords = native.coords
            docked_coords = pose.ligand_coords
            n = min(len(native_coords), len(docked_coords))
            if n < 10:
                return None, None, None

            # 对齐后 RMSD (简化：直接比较前 n 个 CA)
            diff = native_coords[:n] - docked_coords[:n]
            rmsd = float(np.sqrt((diff ** 2).sum(axis=1).mean()))

            # FNAT: 原生接触在预测中保持的比例 (简化)
            fnat = max(0.0, min(1.0, 1.0 - rmsd / 20.0))

            # DockQ 简化公式
            dockq = 0.0
            if rmsd < 10:
                dockq += 0.3
            if fnat > 0.3:
                dockq += 0.4
            if pose.scores.contact_residues > 5:
                dockq += 0.3

            return rmsd, fnat, dockq
        except Exception:
            return None, None, None

    def _compute_standard_metrics(
        self,
        pose,
        native_pdb: Path,
        receptor_chain: str,
        ligand_chain: str,
    ) -> tuple:
        """Compute receptor-aligned LRMSD, native-contact fraction, and DockQ."""
        try:
            from docking.metrics import evaluate_complex
            from docking.structure import PDBParser

            if pose.complex_structure is None:
                return None, None, None
            quality = evaluate_complex(
                PDBParser.parse(native_pdb),
                pose.complex_structure,
                {receptor_chain},
                {ligand_chain},
            )
            return quality.lrmsd, quality.fnat, quality.dockq
        except Exception as exc:
            logger.warning("Docking metric calculation failed for %s: %s", native_pdb, exc)
            return None, None, None

    def _save_results(self, results: List[BenchmarkResult]) -> None:
        import pandas as pd
        rows = [
            {
                "pdb_id": r.pdb_id,
                "label": r.label,
                "predicted": r.predicted_label,
                "probability": r.probability,
                "docking_score": r.docking_score,
                "interface_area": r.interface_area,
                "contact_residues": r.contact_residues,
                "rmsd": r.rmsd,
                "fnat": r.fnat,
                "dockq": r.dockq,
                "runtime_s": r.runtime_seconds,
            }
            for r in results
        ]
        df = pd.DataFrame(rows)
        df.to_csv(self.results_dir / "benchmark_results.csv", index=False)
        with open(self.results_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
