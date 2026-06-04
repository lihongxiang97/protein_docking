"""Multi-stage protein-protein docking pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from docking.config import load_config
from docking.fft_search import FFTDockingSearch
from docking.geometry import apply_transform, count_clashes, euler_to_matrix, uniform_rotations
from docking.preprocess import StructurePreprocessor
from docking.scoring import DockingPose, DockingScorer
from docking.structure import (
    Atom,
    ProteinStructure,
    Residue,
    ensure_unique_chain_ids,
    merge_structures,
)

logger = logging.getLogger(__name__)


class ProteinDocker:
    """Global rigid-body search, atom-level rescoring, clustering, and refinement."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config = load_config(config_path)
        dock = self.config.get("docking", {})
        self.coarse_rotations = int(dock.get("coarse_rotations", 24))
        self.coarse_translations = int(dock.get("coarse_translations", 5))
        self.translation_step = float(dock.get("translation_step", 5.0))
        self.mc_iterations = int(dock.get("mc_iterations", 200))
        self.mc_temperature = float(dock.get("mc_temperature", 2.0))
        self.contact_distance = float(dock.get("contact_distance", 5.0))
        self.clash_distance = float(dock.get("clash_distance", 2.0))
        self.top_n = int(dock.get("top_n_poses", 10))
        self.use_grid = bool(dock.get("grid_search", True))
        self.use_mc = bool(dock.get("monte_carlo", True))
        self.search_method = str(dock.get("search_method", "fft")).lower()
        self.global_candidate_limit = int(dock.get("global_candidate_limit", 120))
        self.refine_seeds = int(dock.get("refine_seeds", 4))
        self.cluster_rmsd = float(dock.get("cluster_rmsd", 4.0))
        self.max_clash_fraction = float(dock.get("max_clash_fraction", 0.4))
        self.include_input_pose = bool(dock.get("include_input_pose", True))
        self.input_pose_bonus = float(dock.get("input_pose_bonus", 10.0))
        self.input_prior_decay_rmsd = float(dock.get("input_prior_decay_rmsd", 2.0))
        self.fft_rescore_weight = float(dock.get("fft_rescore_weight", 5.0))

        self.preprocessor = StructurePreprocessor(config_path)
        self.scorer = DockingScorer(str(config_path) if config_path else None)
        self.fft_search = FFTDockingSearch(self.config)
        self.rng = np.random.default_rng(int(dock.get("random_seed", 42)))

    def dock(
        self,
        receptor_path: Path,
        ligand_path: Path,
        output_dir: Optional[Path] = None,
        receptor_chains: Optional[List[str]] = None,
        ligand_chains: Optional[List[str]] = None,
    ) -> Tuple[List[DockingPose], ProteinStructure, ProteinStructure]:
        receptor_path = Path(receptor_path)
        ligand_path = Path(ligand_path)
        if not receptor_path.exists():
            raise FileNotFoundError(f"Receptor PDB file does not exist: {receptor_path}")
        if not ligand_path.exists():
            raise FileNotFoundError(f"Ligand PDB file does not exist: {ligand_path}")

        receptor = self.preprocessor.preprocess(receptor_path, chain_ids=receptor_chains)
        ligand = self.preprocessor.preprocess(ligand_path, chain_ids=ligand_chains)
        ligand = ensure_unique_chain_ids(ligand, receptor.chains, name=ligand.name)
        if not receptor.atoms:
            raise ValueError(f"Receptor structure has no usable protein atoms: {receptor_path}")
        if not ligand.atoms:
            raise ValueError(f"Ligand structure has no usable protein atoms: {ligand_path}")

        logger.info(
            "Starting %s docking: receptor=%d atoms, ligand=%d atoms",
            self.search_method,
            len(receptor.atoms),
            len(ligand.atoms),
        )
        ligand_coords = ligand.coords.copy()
        ligand_center = ligand.center
        base_translation = receptor.center - ligand_center
        receptor_radii = np.asarray([atom.vdw_radius() for atom in receptor.atoms])
        ligand_radii = np.asarray([atom.vdw_radius() for atom in ligand.atoms])
        rotations = uniform_rotations(self.coarse_rotations)

        candidates = self._global_search(
            receptor,
            ligand,
            rotations,
            ligand_coords,
            ligand_center,
            base_translation,
            receptor_radii,
            ligand_radii,
        )
        if not candidates:
            logger.warning("Global search yielded no valid candidates; using recovery seed")
            recovery = self._evaluate_pose(
                receptor,
                ligand,
                np.eye(3),
                base_translation,
                ligand_center,
                ligand_coords,
            )
            recovery.provenance = "recovery_seed"
            candidates = [recovery]

        if self.use_mc and self.mc_iterations > 0:
            seeds = sorted(candidates, key=lambda pose: pose.scores.total, reverse=True)[
                : max(1, self.refine_seeds)
            ]
            iterations = max(1, self.mc_iterations // len(seeds))
            for seed in seeds:
                candidates.extend(
                    self._monte_carlo_refine(
                        receptor,
                        ligand,
                        seed,
                        ligand_coords,
                        ligand_center,
                        receptor_radii,
                        ligand_radii,
                        iterations,
                    )
                )

        unique = self._deduplicate_poses(candidates)
        clustered = self._cluster_poses(unique)
        ranked = self.scorer.rank_poses(clustered)[: self.top_n]
        for pose in ranked:
            docked_ligand = self._copy_structure_with_coords(ligand, pose.ligand_coords)
            pose.complex_structure = merge_structures(receptor, docked_ligand)

        if output_dir:
            self._save_results(ranked, Path(output_dir))
        return ranked, receptor, ligand

    def _global_search(
        self,
        receptor: ProteinStructure,
        ligand: ProteinStructure,
        rotations: List[np.ndarray],
        ligand_coords: np.ndarray,
        ligand_center: np.ndarray,
        base_translation: np.ndarray,
        receptor_radii: np.ndarray,
        ligand_radii: np.ndarray,
    ) -> List[DockingPose]:
        if not self.use_grid:
            return []
        candidates: List[DockingPose] = []
        if self.search_method == "fft":
            fft_candidates = self.fft_search.search(receptor, ligand, rotations)
            transformations = [
                (item.rotation, item.translation, item.score, "fft_global")
                for item in fft_candidates[: self.global_candidate_limit]
            ]
        else:
            translations = self._generate_translations(receptor.coords, ligand.coords)
            transformations = [
                (rotation, base_translation + translation, 0.0, "directional_global")
                for rotation in rotations
                for translation in translations
            ][: self.global_candidate_limit]

        if self.include_input_pose:
            transformations.insert(0, (np.eye(3), np.zeros(3), 0.0, "input_pose"))

        for rotation, translation, search_score, provenance in transformations:
            transformed = apply_transform(ligand_coords, ligand_center, rotation, translation)
            clashes = count_clashes(
                receptor.coords,
                receptor_radii,
                transformed,
                ligand_radii,
                self.clash_distance,
            )
            if clashes > len(ligand.atoms) * self.max_clash_fraction:
                continue
            pose = self._evaluate_pose(
                receptor, ligand, rotation, translation, ligand_center, ligand_coords
            )
            pose.search_score = float(search_score)
            pose.provenance = provenance
            if provenance == "input_pose":
                pose.scores.prior_score = self.input_pose_bonus
            elif provenance == "fft_global":
                pose.scores.prior_score = max(0.0, float(search_score)) * self.fft_rescore_weight
            pose.scores.total += pose.scores.prior_score
            candidates.append(pose)
        logger.info("Global search retained %d atom-level candidates", len(candidates))
        return candidates

    def _generate_translations(
        self, receptor_coords: np.ndarray, ligand_coords: np.ndarray
    ) -> List[np.ndarray]:
        rec_center = receptor_coords.mean(axis=0)
        lig_center = ligand_coords.mean(axis=0)
        rec_radius = float(np.max(np.linalg.norm(receptor_coords - rec_center, axis=1)))
        lig_radius = float(np.max(np.linalg.norm(ligand_coords - lig_center, axis=1)))
        contact_radius = max(
            self.translation_step, rec_radius + lig_radius - self.contact_distance
        )
        direction_count = max(6, self.coarse_translations * 2)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        translations = []
        for index in range(direction_count):
            y = 1.0 - 2.0 * (index + 0.5) / direction_count
            radial = np.sqrt(max(0.0, 1.0 - y * y))
            theta = golden_angle * index
            direction = np.array([np.cos(theta) * radial, y, np.sin(theta) * radial])
            for offset in (-self.translation_step, 0.0, self.translation_step):
                translations.append(
                    direction * max(self.translation_step, contact_radius + offset)
                )
        return translations

    def _evaluate_pose(
        self,
        receptor: ProteinStructure,
        ligand: ProteinStructure,
        rotation: np.ndarray,
        translation: np.ndarray,
        ligand_center: np.ndarray,
        ligand_coords: np.ndarray,
    ) -> DockingPose:
        transformed = apply_transform(ligand_coords, ligand_center, rotation, translation)
        ligand_copy = self._copy_structure_with_coords(ligand, transformed)
        return DockingPose(
            rotation=rotation.copy(),
            translation=translation.copy(),
            scores=self.scorer.score_complex(receptor, ligand_copy),
            ligand_coords=transformed,
        )

    def _copy_structure_with_coords(
        self, structure: ProteinStructure, coords: np.ndarray
    ) -> ProteinStructure:
        if len(coords) != len(structure.atoms):
            raise ValueError("Coordinate count does not match structure atom count")
        copied = ProteinStructure(name=structure.name + "_docked")
        for atom, coord in zip(structure.atoms, coords):
            new_atom = Atom(
                serial=atom.serial,
                name=atom.name,
                alt_loc=atom.alt_loc,
                resname=atom.resname,
                chain_id=atom.chain_id,
                resseq=atom.resseq,
                icode=atom.icode,
                x=float(coord[0]),
                y=float(coord[1]),
                z=float(coord[2]),
                occupancy=atom.occupancy,
                bfactor=atom.bfactor,
                element=atom.element,
                record=atom.record,
            )
            copied.atoms.append(new_atom)
            if new_atom.residue_key not in copied.residues:
                copied.residues[new_atom.residue_key] = Residue(
                    chain_id=new_atom.chain_id,
                    resseq=new_atom.resseq,
                    resname=new_atom.resname,
                    icode=new_atom.icode,
                )
            copied.residues[new_atom.residue_key].atoms.append(new_atom)
            copied.chains.add(new_atom.chain_id)
        for key, residue in structure.residues.items():
            if key in copied.residues:
                copied.residues[key].sasa = residue.sasa
                copied.residues[key].is_surface = residue.is_surface
                copied.residues[key].charge = residue.charge
        return copied

    def _monte_carlo_refine(
        self,
        receptor: ProteinStructure,
        ligand: ProteinStructure,
        start_pose: DockingPose,
        ligand_coords: np.ndarray,
        ligand_center: np.ndarray,
        receptor_radii: np.ndarray,
        ligand_radii: np.ndarray,
        iterations: int,
    ) -> List[DockingPose]:
        current_score = start_pose.scores.total
        current_rotation = start_pose.rotation.copy()
        current_translation = start_pose.translation.copy()
        accepted: List[DockingPose] = []
        for _ in range(iterations):
            perturbation = euler_to_matrix(*self.rng.normal(0.0, 0.08, 3))
            rotation = perturbation @ current_rotation
            translation = current_translation + self.rng.normal(0.0, 0.75, 3)
            transformed = apply_transform(ligand_coords, ligand_center, rotation, translation)
            clashes = count_clashes(
                receptor.coords,
                receptor_radii,
                transformed,
                ligand_radii,
                self.clash_distance,
            )
            if clashes > len(ligand.atoms) * self.max_clash_fraction:
                continue
            pose = self._evaluate_pose(
                receptor, ligand, rotation, translation, ligand_center, ligand_coords
            )
            pose.provenance = f"{start_pose.provenance}+mc"
            pose.search_score = start_pose.search_score
            if start_pose.provenance.startswith("input_pose"):
                displacement = self._pose_rmsd(pose, start_pose)
                pose.scores.prior_score = start_pose.scores.prior_score * np.exp(
                    -displacement / self.input_prior_decay_rmsd
                )
            else:
                pose.scores.prior_score = start_pose.scores.prior_score
            pose.scores.total += pose.scores.prior_score
            delta = pose.scores.total - current_score
            probability = np.exp(np.clip(delta / self.mc_temperature, -700.0, 0.0))
            if delta > 0 or self.rng.random() < probability:
                current_rotation = rotation
                current_translation = translation
                current_score = pose.scores.total
                accepted.append(pose)
        return accepted

    def _deduplicate_poses(
        self, poses: List[DockingPose], score_threshold: float = 1.0
    ) -> List[DockingPose]:
        unique: List[DockingPose] = []
        for pose in sorted(poses, key=lambda item: item.scores.total, reverse=True):
            if pose.ligand_coords is None:
                continue
            if not any(self._pose_rmsd(pose, kept) < score_threshold for kept in unique):
                unique.append(pose)
            if len(unique) >= max(self.top_n * 20, self.top_n):
                break
        return unique

    def _cluster_poses(self, poses: List[DockingPose]) -> List[DockingPose]:
        clusters: List[List[DockingPose]] = []
        for pose in sorted(poses, key=lambda item: item.scores.total, reverse=True):
            for cluster in clusters:
                if self._pose_rmsd(pose, cluster[0]) < self.cluster_rmsd:
                    cluster.append(pose)
                    break
            else:
                clusters.append([pose])
        representatives = []
        for cluster in clusters:
            representative = cluster[0]
            representative.cluster_size = len(cluster)
            representatives.append(representative)
        return representatives

    @staticmethod
    def _pose_rmsd(a: DockingPose, b: DockingPose) -> float:
        if a.ligand_coords is None or b.ligand_coords is None:
            return float("inf")
        count = min(len(a.ligand_coords), len(b.ligand_coords))
        stride = max(1, count // 64)
        delta = a.ligand_coords[:count:stride] - b.ligand_coords[:count:stride]
        return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))

    @staticmethod
    def _save_results(poses: List[DockingPose], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for pose in poses:
            if pose.complex_structure:
                pose.complex_structure.write_pdb(output_dir / f"docked_complex_{pose.rank}.pdb")
