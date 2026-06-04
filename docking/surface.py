"""
表面残基识别：SASA (rolling sphere 近似)、疏水/电荷区域。
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
from docking.spatial import cKDTree

from docking.structure import (
    HYDROPHOBIC_AA,
    NEGATIVE_AA,
    POSITIVE_AA,
    RESIDUE_ATOMS,
    ProteinStructure,
    Residue,
    VDW_RADIUS,
)


class SurfaceAnalyzer:
    """
    溶剂可及表面积 (SASA) 与表面残基分析。

    算法：基于邻居密度的 rolling sphere 近似。
    - 对每个残基，统计 probe_radius 球内非自身原子的邻居密度
    - 低密度 → 暴露于溶剂 → 高 SASA
    时间复杂度: O(N log N) (KD-tree)
    空间复杂度: O(N)
    """

    def __init__(
        self,
        probe_radius: float = 1.4,
        sasa_threshold: float = 25.0,
        neighbor_radius: float = 8.0,
        neighbor_density_threshold: float = 0.15,
    ):
        self.probe_radius = probe_radius
        self.sasa_threshold = sasa_threshold
        self.neighbor_radius = neighbor_radius
        self.neighbor_density_threshold = neighbor_density_threshold

    def compute_sasa(self, structure: ProteinStructure) -> Dict[Tuple, float]:
        """计算每个残基的 SASA (Å²)。"""
        coords = structure.coords
        if len(coords) == 0:
            return {}

        radii = np.array([a.vdw_radius() for a in structure.atoms])
        tree = cKDTree(coords)
        atom_sasa = np.zeros(len(coords))

        # 每个原子的最大可能 SASA ≈ 4π r²
        for i, (coord, r) in enumerate(zip(coords, radii)):
            max_sasa = 4 * np.pi * (r + self.probe_radius) ** 2
            search_r = r + self.probe_radius + self.neighbor_radius
            neighbors = tree.query_ball_point(coord, search_r)
            neighbors = [j for j in neighbors if j != i]

            if not neighbors:
                atom_sasa[i] = max_sasa
                continue

            # 邻居遮蔽：距离越近遮蔽越多
            neighbor_coords = coords[neighbors]
            neighbor_radii = radii[neighbors]
            dists = np.linalg.norm(neighbor_coords - coord, axis=1)
            sum_overlap = 0.0
            for d, nr in zip(dists, neighbor_radii):
                contact_dist = r + nr + self.probe_radius
                if d < contact_dist:
                    overlap = (contact_dist - d) / contact_dist
                    sum_overlap += overlap ** 2
            exposure = max(0.0, 1.0 - sum_overlap / max(len(neighbors), 1))
            atom_sasa[i] = max_sasa * exposure

        # 聚合到残基
        residue_sasa: Dict[Tuple, float] = {}
        atom_indices: Dict[Tuple, List[int]] = {}
        for index, atom in enumerate(structure.atoms):
            atom_indices.setdefault(atom.residue_key, []).append(index)
        for res in structure.get_residue_list():
            indices = atom_indices.get(res.key, [])
            res_sasa = float(atom_sasa[indices].sum()) if indices else 0.0
            residue_sasa[res.key] = res_sasa
            res.sasa = res_sasa

        return residue_sasa

    def identify_surface_residues(self, structure: ProteinStructure) -> List[Residue]:
        """识别表面残基。"""
        self.compute_sasa(structure)
        surface = []
        for res in structure.get_residue_list():
            res.is_surface = res.sasa >= self.sasa_threshold
            if res.is_surface:
                surface.append(res)
        return surface

    def neighbor_density_surface(
        self, structure: ProteinStructure
    ) -> Set[Tuple]:
        """
        基于邻居密度的表面残基识别 (补充 SASA)。
        残基 CA 周围 neighbor_radius 内 CA 数量低于阈值 → 表面。
        """
        ca_coords = []
        ca_keys = []
        for res in structure.get_residue_list():
            ca = res.ca_coord
            if ca is not None:
                ca_coords.append(ca)
                ca_keys.append(res.key)

        if not ca_coords:
            return set()

        ca_coords = np.array(ca_coords)
        tree = cKDTree(ca_coords)
        surface_keys = set()
        n = len(ca_coords)
        volume = (4 / 3) * np.pi * self.neighbor_radius ** 3
        max_density = n / volume if volume > 0 else 0

        for i, key in enumerate(ca_keys):
            count = len(tree.query_ball_point(ca_coords[i], self.neighbor_radius)) - 1
            density = count / volume if volume > 0 else 0
            if density < max_density * self.neighbor_density_threshold:
                surface_keys.add(key)

        return surface_keys

    def hydrophobic_patches(self, structure: ProteinStructure) -> List[Residue]:
        """疏水表面斑块。"""
        surface = self.identify_surface_residues(structure)
        return [r for r in surface if r.resname in HYDROPHOBIC_AA]

    def charged_patches(self, structure: ProteinStructure) -> Dict[str, List[Residue]]:
        """带电表面区域。"""
        surface = self.identify_surface_residues(structure)
        return {
            "positive": [r for r in surface if r.resname in POSITIVE_AA],
            "negative": [r for r in surface if r.resname in NEGATIVE_AA],
        }

    def assign_residue_charges(self, structure: ProteinStructure) -> None:
        """分配残基电荷 (简化)。"""
        charge_map = {
            "ARG": 1.0, "LYS": 1.0, "HIS": 0.5,
            "ASP": -1.0, "GLU": -1.0,
        }
        for res in structure.get_residue_list():
            res.charge = charge_map.get(res.resname, 0.0)
