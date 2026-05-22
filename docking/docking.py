"""
蛋白-蛋白分子对接：网格搜索 + Monte Carlo 采样。
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

from docking.geometry import (
    apply_transform,
    count_clashes,
    grid_rotations,
    random_rotation_matrix,
)
from docking.scoring import DockingPose, DockingScorer
from docking.structure import Atom, ProteinStructure, Residue, merge_structures
from docking.preprocess import StructurePreprocessor

logger = logging.getLogger(__name__)


class ProteinDocker:
    """
    自主实现的蛋白-蛋白刚性对接。

    算法流程:
    1. 预处理受体与配体
    2. 粗搜索: 均匀旋转 × 平移网格
    3. 精修: Monte Carlo 局部优化
    4. 评分排序，输出 Top-N 构象

    时间复杂度: O(R * T * N_atoms) 粗搜索 + O(MC_iter * N_atoms) 精修
    空间复杂度: O(N_poses * N_atoms)
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        dock = self.config.get("docking", {})
        self.coarse_rotations = dock.get("coarse_rotations", 12)
        self.coarse_translations = dock.get("coarse_translations", 5)
        self.translation_step = dock.get("translation_step", 5.0)
        self.mc_iterations = dock.get("mc_iterations", 200)
        self.mc_temperature = dock.get("mc_temperature", 2.0)
        self.contact_distance = dock.get("contact_distance", 5.0)
        self.clash_distance = dock.get("clash_distance", 2.0)
        self.top_n = dock.get("top_n_poses", 10)
        self.use_grid = dock.get("grid_search", True)
        self.use_mc = dock.get("monte_carlo", True)

        self.preprocessor = StructurePreprocessor(config_path)
        self.scorer = DockingScorer(str(config_path) if config_path else None)
        self.rng = np.random.default_rng(42)

    def _load_config(self, config_path: Optional[Path]) -> dict:
        path = config_path or Path(__file__).parent.parent / "config.yaml"
        path = Path(path)
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return {}

    def dock(
        self,
        receptor_path: Path,
        ligand_path: Path,
        output_dir: Optional[Path] = None,
        receptor_chains: Optional[List[str]] = None,
        ligand_chains: Optional[List[str]] = None,
    ) -> Tuple[List[DockingPose], ProteinStructure, ProteinStructure]:
        """
        执行完整对接流程。

        Returns:
            (排序后的 poses, 预处理后的 receptor, 预处理后的 ligand)
        """
        receptor_path = Path(receptor_path)
        ligand_path = Path(ligand_path)

        receptor = self.preprocessor.preprocess(receptor_path, chain_ids=receptor_chains)
        ligand = self.preprocessor.preprocess(ligand_path, chain_ids=ligand_chains)

        logger.info("开始对接: receptor=%d atoms, ligand=%d atoms",
                    len(receptor.atoms), len(ligand.atoms))

        # 将配体移至受体附近
        rec_center = receptor.center
        lig_center = ligand.center
        initial_offset = rec_center - lig_center + np.array([20.0, 0.0, 0.0])

        ligand_coords = ligand.coords.copy()
        ligand_center = lig_center + initial_offset
        ligand_coords = ligand_coords - lig_center + ligand_center

        radii_rec = np.array([a.vdw_radius() for a in receptor.atoms])
        radii_lig = np.array([a.vdw_radius() for a in ligand.atoms])

        candidate_poses: List[DockingPose] = []

        # 粗搜索
        if self.use_grid:
            rotations = grid_rotations(self.coarse_rotations)
            translations = self._generate_translations(rec_center)
            for rot in rotations:
                for trans in translations:
                    transformed = apply_transform(
                        ligand_coords, ligand_center, rot, trans
                    )
                    clashes = count_clashes(
                        receptor.coords, radii_rec,
                        transformed, radii_lig, self.clash_distance,
                    )
                    if clashes > len(ligand.atoms) * 0.3:
                        continue
                    pose = self._evaluate_pose(
                        receptor, ligand, rot, trans + initial_offset,
                        ligand_center, ligand_coords,
                    )
                    candidate_poses.append(pose)

        # Monte Carlo 精修
        if self.use_mc and candidate_poses:
            best = max(candidate_poses, key=lambda p: p.scores.total)
            mc_poses = self._monte_carlo_refine(
                receptor, ligand, best, ligand_coords, ligand_center,
                initial_offset, radii_rec, radii_lig,
            )
            candidate_poses.extend(mc_poses)
        elif self.use_mc:
            # 无粗搜索结果时独立 MC
            rot = np.eye(3)
            trans = initial_offset
            best = self._evaluate_pose(
                receptor, ligand, rot, trans, ligand_center, ligand_coords
            )
            mc_poses = self._monte_carlo_refine(
                receptor, ligand, best, ligand_coords, ligand_center,
                initial_offset, radii_rec, radii_lig,
            )
            candidate_poses.extend(mc_poses)

        # 去重并排序
        unique_poses = self._deduplicate_poses(candidate_poses)
        ranked = self.scorer.rank_poses(unique_poses)[: self.top_n]

        # 保存结果
        if output_dir:
            self._save_results(ranked, receptor, ligand, Path(output_dir))

        return ranked, receptor, ligand

    def _generate_translations(self, center: np.ndarray) -> List[np.ndarray]:
        """生成平移采样点。"""
        translations = [np.zeros(3)]
        step = self.translation_step
        n = self.coarse_translations
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                for k in range(-n, n + 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    translations.append(np.array([i, j, k]) * step)
        return translations[: self.coarse_translations ** 3 + 1]

    def _evaluate_pose(
        self,
        receptor: ProteinStructure,
        ligand: ProteinStructure,
        rotation: np.ndarray,
        translation: np.ndarray,
        ligand_center: np.ndarray,
        ligand_coords: np.ndarray,
    ) -> DockingPose:
        """评估单个构象。"""
        transformed_coords = apply_transform(
            ligand_coords, ligand_center, rotation, translation
        )
        ligand_copy = self._copy_structure_with_coords(ligand, transformed_coords)
        scores = self.scorer.score_complex(receptor, ligand_copy)
        complex_struct = merge_structures(receptor, ligand_copy)
        return DockingPose(
            rotation=rotation.copy(),
            translation=translation.copy(),
            scores=scores,
            ligand_coords=transformed_coords,
            complex_structure=complex_struct,
        )

    def _copy_structure_with_coords(
        self, structure: ProteinStructure, coords: np.ndarray
    ) -> ProteinStructure:
        new_struct = ProteinStructure(name=structure.name + "_docked")
        for i, atom in enumerate(structure.atoms):
            new_atom = Atom(
                serial=atom.serial, name=atom.name, alt_loc=atom.alt_loc,
                resname=atom.resname, chain_id=atom.chain_id,
                resseq=atom.resseq, icode=atom.icode,
                x=coords[i][0], y=coords[i][1], z=coords[i][2],
                occupancy=atom.occupancy, bfactor=atom.bfactor,
                element=atom.element, record=atom.record,
            )
            new_struct.atoms.append(new_atom)
            key = new_atom.residue_key
            if key not in new_struct.residues:
                new_struct.residues[key] = Residue(
                    chain_id=new_atom.chain_id,
                    resseq=new_atom.resseq,
                    resname=new_atom.resname,
                    icode=new_atom.icode,
                )
            new_struct.residues[key].atoms.append(new_atom)
        new_struct.chains = structure.chains.copy()
        return new_struct

    def _monte_carlo_refine(
        self,
        receptor: ProteinStructure,
        ligand: ProteinStructure,
        start_pose: DockingPose,
        ligand_coords: np.ndarray,
        ligand_center: np.ndarray,
        initial_offset: np.ndarray,
        radii_rec: np.ndarray,
        radii_lig: np.ndarray,
    ) -> List[DockingPose]:
        """Monte Carlo 局部优化。"""
        best_pose = start_pose
        best_score = start_pose.scores.total
        current_rot = start_pose.rotation.copy()
        current_trans = start_pose.translation.copy()
        poses = []

        for _ in range(self.mc_iterations):
            # 小扰动
            delta_rot = random_rotation_matrix(self.rng)
            small_angle = self.rng.normal(0, 0.1, 3)
            from docking.geometry import euler_to_matrix
            perturb = euler_to_matrix(*small_angle)
            new_rot = perturb @ current_rot
            new_trans = current_trans + self.rng.normal(0, 1.0, 3)

            transformed = apply_transform(ligand_coords, ligand_center, new_rot, new_trans)
            clashes = count_clashes(
                receptor.coords, radii_rec, transformed, radii_lig, self.clash_distance
            )
            if clashes > len(ligand.atoms) * 0.4:
                continue

            pose = self._evaluate_pose(
                receptor, ligand, new_rot, new_trans,
                ligand_center, ligand_coords,
            )
            delta_e = pose.scores.total - best_score
            if delta_e > 0 or self.rng.random() < np.exp(delta_e / self.mc_temperature):
                current_rot = new_rot
                current_trans = new_trans
                if pose.scores.total > best_score:
                    best_score = pose.scores.total
                    best_pose = pose
                poses.append(pose)

        return poses

    def _deduplicate_poses(
        self, poses: List[DockingPose], score_threshold: float = 1.0
    ) -> List[DockingPose]:
        """去除评分相近的重复构象。"""
        if not poses:
            return []
        unique = []
        for pose in sorted(poses, key=lambda p: p.scores.total, reverse=True):
            is_dup = False
            for u in unique:
                if abs(pose.scores.total - u.scores.total) < score_threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(pose)
        return unique

    def _save_results(
        self,
        poses: List[DockingPose],
        receptor: ProteinStructure,
        ligand: ProteinStructure,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for pose in poses:
            if pose.complex_structure:
                path = output_dir / f"docked_complex_{pose.rank}.pdb"
                pose.complex_structure.write_pdb(path)
