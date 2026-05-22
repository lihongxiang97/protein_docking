"""
对接评分函数：Score = w1*H + w2*E + w3*C + w4*A - w5*P
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial import cKDTree

from docking.geometry import compute_interface_area, count_clashes, count_contacts
from docking.structure import (
    HYDROPHOBIC_AA,
    NEGATIVE_AA,
    POSITIVE_AA,
    ProteinStructure,
)


@dataclass
class ScoreComponents:
    """评分组成。"""
    total: float = 0.0
    hydrophobic: float = 0.0
    electrostatic: float = 0.0
    contacts: float = 0.0
    interface_area: float = 0.0
    clash_penalty: float = 0.0
    hbonds: int = 0
    contact_residues: int = 0
    raw_interface_area: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "total_score": self.total,
            "hydrophobic_score": self.hydrophobic,
            "electrostatic_score": self.electrostatic,
            "contact_score": self.contacts,
            "interface_area_score": self.interface_area,
            "clash_penalty": self.clash_penalty,
            "hbond_count": self.hbonds,
            "contact_residues": self.contact_residues,
            "interface_area_angstrom2": self.raw_interface_area,
        }


@dataclass
class DockingPose:
    """对接构象。"""
    rank: int = 0
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scores: ScoreComponents = field(default_factory=ScoreComponents)
    ligand_coords: Optional[np.ndarray] = None
    complex_structure: Optional[ProteinStructure] = None


class DockingScorer:
    """
    自主设计的蛋白-蛋白对接评分函数。

    Score = w_H * H + w_E * E + w_C * C + w_A * A - w_P * P

    H: 疏水互补 (疏水残基在界面 5Å 内配对)
    E: 静电互补 (异性电荷吸引)
    C: 接触残基数量 (归一化)
    A: 界面面积
    P: 原子碰撞惩罚
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        sc = self.config.get("scoring", {})
        w = sc.get("weights", {})
        self.w_h = w.get("hydrophobic", 0.25)
        self.w_e = w.get("electrostatic", 0.20)
        self.w_c = w.get("contacts", 0.25)
        self.w_a = w.get("interface_area", 0.20)
        self.w_p = w.get("clash_penalty", 0.10)
        self.contact_cutoff = sc.get("contact_cutoff", 5.0)
        self.hbond_distance = sc.get("hbond_distance", 3.5)
        self.clash_cutoff = 2.0

    def _load_config(self, config_path: Optional[str]) -> dict:
        path = Path(config_path) if config_path else Path(__file__).parent.parent / "config.yaml"
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return {}

    def score_complex(
        self,
        receptor: ProteinStructure,
        ligand: ProteinStructure,
    ) -> ScoreComponents:
        """对给定复合物构象评分。"""
        rec_coords = receptor.coords
        lig_coords = ligand.coords
        rec_radii = np.array([a.vdw_radius() for a in receptor.atoms])
        lig_radii = np.array([a.vdw_radius() for a in ligand.atoms])

        comp = ScoreComponents()

        # 碰撞惩罚 P
        clashes = count_clashes(rec_coords, rec_radii, lig_coords, lig_radii, self.clash_cutoff)
        comp.clash_penalty = min(clashes * 0.5, 50.0)

        # 接触 C
        n_contacts = count_contacts(rec_coords, lig_coords, self.contact_cutoff)
        comp.contacts = min(n_contacts / 100.0, 1.0) * 100

        # 界面面积 A
        area = compute_interface_area(rec_coords, lig_coords, self.contact_cutoff)
        comp.raw_interface_area = area
        comp.interface_area = min(area / 500.0, 1.0) * 100

        # 残基接触
        comp.contact_residues = self._count_contact_residues(receptor, ligand)

        # 疏水 H
        comp.hydrophobic = self._score_hydrophobic(receptor, ligand)

        # 静电 E
        comp.electrostatic = self._score_electrostatic(receptor, ligand)

        # 氢键
        comp.hbonds = self._count_hbonds(receptor, ligand)

        # 总分
        comp.total = (
            self.w_h * comp.hydrophobic
            + self.w_e * comp.electrostatic
            + self.w_c * comp.contacts
            + self.w_a * comp.interface_area
            - self.w_p * comp.clash_penalty
            + comp.hbonds * 2.0  # 氢键奖励
        )

        return comp

    def _count_contact_residues(
        self, receptor: ProteinStructure, ligand: ProteinStructure
    ) -> int:
        rec_ca = []
        rec_keys = []
        for res in receptor.get_residue_list():
            ca = res.ca_coord
            if ca is not None:
                rec_ca.append(ca)
                rec_keys.append(res.key)

        lig_ca = []
        for res in ligand.get_residue_list():
            ca = res.ca_coord
            if ca is not None:
                lig_ca.append(ca)

        if not rec_ca or not lig_ca:
            return 0

        tree = cKDTree(np.array(lig_ca))
        count = 0
        for ca in rec_ca:
            if tree.query(ca)[0] < self.contact_cutoff:
                count += 1
        return count

    def _score_hydrophobic(
        self, receptor: ProteinStructure, ligand: ProteinStructure
    ) -> float:
        rec_hydro = [
            r.ca_coord for r in receptor.get_residue_list()
            if r.resname in HYDROPHOBIC_AA and r.ca_coord is not None
        ]
        lig_hydro = [
            r.ca_coord for r in ligand.get_residue_list()
            if r.resname in HYDROPHOBIC_AA and r.ca_coord is not None
        ]
        if not rec_hydro or not lig_hydro:
            return 0.0

        tree = cKDTree(np.array(lig_hydro))
        pairs = 0
        for ca in rec_hydro:
            if tree.query(ca)[0] < self.contact_cutoff:
                pairs += 1
        return min(pairs / max(len(rec_hydro), 1) * 100, 100)

    def _score_electrostatic(
        self, receptor: ProteinStructure, ligand: ProteinStructure
    ) -> float:
        """异性电荷吸引得分。"""
        rec_charged = [
            (r.ca_coord, r.charge) for r in receptor.get_residue_list()
            if r.ca_coord is not None and r.charge != 0
        ]
        lig_charged = [
            (r.ca_coord, r.charge) for r in ligand.get_residue_list()
            if r.ca_coord is not None and r.charge != 0
        ]
        if not rec_charged or not lig_charged:
            return 0.0

        lig_coords = np.array([c[0] for c in lig_charged])
        lig_charges = np.array([c[1] for c in lig_charged])
        tree = cKDTree(lig_coords)

        score = 0.0
        for coord, charge in rec_charged:
            dist, idx = tree.query(coord)
            if dist < self.contact_cutoff:
                # 异性电荷互补
                product = charge * lig_charges[idx]
                if product < 0:
                    score += abs(product) * (1 - dist / self.contact_cutoff)
        return min(score * 20, 100)

    def _count_hbonds(
        self, receptor: ProteinStructure, ligand: ProteinStructure
    ) -> int:
        """简化氢键检测：N/O 原子对距离 < hbond_distance。"""
        rec_polar = [
            a.coords for a in receptor.atoms
            if a.name.strip()[0] in ("N", "O")
        ]
        lig_polar = [
            a.coords for a in ligand.atoms
            if a.name.strip()[0] in ("N", "O")
        ]
        if not rec_polar or not lig_polar:
            return 0

        tree = cKDTree(np.array(lig_polar))
        hbonds = 0
        for coord in rec_polar:
            dist, _ = tree.query(coord)
            if dist < self.hbond_distance:
                hbonds += 1
        return hbonds

    def rank_poses(self, poses: List[DockingPose]) -> List[DockingPose]:
        """按总分排序。"""
        sorted_poses = sorted(poses, key=lambda p: p.scores.total, reverse=True)
        for i, pose in enumerate(sorted_poses):
            pose.rank = i + 1
        return sorted_poses
