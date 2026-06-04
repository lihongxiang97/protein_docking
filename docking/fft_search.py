"""FFT-based global rigid-body docking search.

The implementation follows the public high-level idea used by FFT docking
programs: encode receptor/ligand properties on grids, evaluate every
translation with correlations, and retain a small number of translations per
rotation for detailed atom-level rescoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from docking.structure import HYDROPHOBIC_AA, ProteinStructure


@dataclass
class FFTCandidate:
    rotation: np.ndarray
    translation: np.ndarray
    score: float
    shape_score: float
    electrostatic_score: float
    hydrophobic_score: float


class FFTDockingSearch:
    """Generate global rigid-body candidates using 3D FFT correlations."""

    def __init__(self, config: dict):
        cfg = config.get("fft_search", {})
        self.grid_spacing = float(cfg.get("grid_spacing", 2.0))
        self.padding = float(cfg.get("padding", 8.0))
        self.max_grid_size = int(cfg.get("max_grid_size", 128))
        self.top_translations = int(cfg.get("top_translations_per_rotation", 6))
        self.min_translation_separation = float(cfg.get("min_translation_separation", 4.0))
        self.shape_weight = float(cfg.get("shape_weight", 1.0))
        self.electrostatic_weight = float(cfg.get("electrostatic_weight", 0.25))
        self.hydrophobic_weight = float(cfg.get("hydrophobic_weight", 0.20))
        self.core_penalty = float(cfg.get("core_penalty", 6.0))
        self.shell_width = float(cfg.get("shell_width", 3.0))

    def search(
        self,
        receptor: ProteinStructure,
        ligand: ProteinStructure,
        rotations: Iterable[np.ndarray],
    ) -> List[FFTCandidate]:
        rec_center = receptor.center
        lig_center = ligand.center
        rec_coords = self._heavy_atom_coords(receptor) - rec_center
        lig_coords = self._heavy_atom_coords(ligand) - lig_center
        shape, spacing = self._grid_geometry(rec_coords, lig_coords)

        rec_shape = self._shape_grid(rec_coords, shape, spacing, receptor_side=True)
        rec_charge = self._property_grid(
            *self._residue_property_points(receptor, rec_center, "charge"),
            shape,
            spacing,
        )
        rec_hydro = self._property_grid(
            *self._residue_property_points(receptor, rec_center, "hydrophobic"),
            shape,
            spacing,
        )
        rec_shape_fft = np.fft.fftn(rec_shape)
        rec_charge_fft = np.fft.fftn(rec_charge)
        rec_hydro_fft = np.fft.fftn(rec_hydro)
        base_translation = rec_center - lig_center

        lig_charge_points, lig_charge_values = self._residue_property_points(
            ligand, lig_center, "charge"
        )
        lig_hydro_points, lig_hydro_values = self._residue_property_points(
            ligand, lig_center, "hydrophobic"
        )
        candidates: List[FFTCandidate] = []

        for rotation in rotations:
            rotated_atoms = (rotation @ lig_coords.T).T
            lig_shape = self._shape_grid(rotated_atoms, shape, spacing, receptor_side=False)
            lig_charge = self._property_grid(
                (rotation @ lig_charge_points.T).T if len(lig_charge_points) else lig_charge_points,
                lig_charge_values,
                shape,
                spacing,
            )
            lig_hydro = self._property_grid(
                (rotation @ lig_hydro_points.T).T if len(lig_hydro_points) else lig_hydro_points,
                lig_hydro_values,
                shape,
                spacing,
            )

            shape_corr = self._correlate(rec_shape_fft, lig_shape)
            charge_corr = -self._correlate(rec_charge_fft, lig_charge)
            hydro_corr = self._correlate(rec_hydro_fft, lig_hydro)
            shape_corr /= max(len(lig_coords), 1)
            charge_corr /= max(len(lig_charge_points), 1)
            hydro_corr /= max(len(lig_hydro_points), 1)
            total = (
                self.shape_weight * shape_corr
                + self.electrostatic_weight * charge_corr
                + self.hydrophobic_weight * hydro_corr
            )

            for index in self._select_peaks(total, spacing):
                shift = (np.asarray(index, dtype=float) - np.asarray(shape) // 2) * spacing
                candidates.append(
                    FFTCandidate(
                        rotation=rotation.copy(),
                        translation=base_translation + shift,
                        score=float(total[index]),
                        shape_score=float(shape_corr[index]),
                        electrostatic_score=float(charge_corr[index]),
                        hydrophobic_score=float(hydro_corr[index]),
                    )
                )

        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def _grid_geometry(
        self, receptor_coords: np.ndarray, ligand_coords: np.ndarray
    ) -> Tuple[Tuple[int, int, int], float]:
        rec_span = np.ptp(receptor_coords, axis=0)
        lig_span = np.ptp(ligand_coords, axis=0)
        required_span = rec_span + lig_span + 2.0 * self.padding
        spacing = max(self.grid_spacing, float(np.max(required_span)) / (self.max_grid_size - 4))
        required = np.maximum(np.ceil(required_span / spacing).astype(int) + 2, 16)
        shape = tuple(min(self._next_power_of_two(int(size)), self.max_grid_size) for size in required)
        return shape, spacing

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 << max(4, (value - 1).bit_length())

    def _shape_grid(
        self,
        coords: np.ndarray,
        shape: Tuple[int, int, int],
        spacing: float,
        receptor_side: bool,
    ) -> np.ndarray:
        grid = np.zeros(shape, dtype=np.float32)
        indices = self._indices(coords, shape, spacing)
        valid = self._valid_indices(indices, shape)
        indices = indices[valid]
        if not len(indices):
            return grid
        if not receptor_side:
            grid[tuple(indices.T)] = 1.0
            return grid

        shell_voxels = max(1, int(np.ceil(self.shell_width / spacing)))
        offsets = self._sphere_offsets(shell_voxels)
        for offset in offsets:
            shell = indices + offset
            shell = shell[self._valid_indices(shell, shape)]
            grid[tuple(shell.T)] = 1.0
        grid[tuple(indices.T)] = -self.core_penalty
        return grid

    def _property_grid(
        self,
        points: np.ndarray,
        values: np.ndarray,
        shape: Tuple[int, int, int],
        spacing: float,
    ) -> np.ndarray:
        grid = np.zeros(shape, dtype=np.float32)
        if not len(points):
            return grid
        indices = self._indices(points, shape, spacing)
        valid = self._valid_indices(indices, shape)
        for index, value in zip(indices[valid], values[valid]):
            grid[tuple(index)] += float(value)
        return grid

    @staticmethod
    def _indices(
        coords: np.ndarray, shape: Tuple[int, int, int], spacing: float
    ) -> np.ndarray:
        center = np.asarray(shape, dtype=float) // 2
        return np.rint(coords / spacing + center).astype(int)

    @staticmethod
    def _valid_indices(indices: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
        if not len(indices):
            return np.zeros(0, dtype=bool)
        return np.all((indices >= 0) & (indices < np.asarray(shape)), axis=1)

    @staticmethod
    def _sphere_offsets(radius: int) -> np.ndarray:
        values = range(-radius, radius + 1)
        return np.asarray(
            [
                (x, y, z)
                for x in values
                for y in values
                for z in values
                if x * x + y * y + z * z <= radius * radius
            ],
            dtype=int,
        )

    @staticmethod
    def _correlate(receptor_fft: np.ndarray, ligand_grid: np.ndarray) -> np.ndarray:
        correlation = np.fft.ifftn(receptor_fft * np.conj(np.fft.fftn(ligand_grid))).real
        return np.fft.fftshift(correlation)

    def _select_peaks(
        self, scores: np.ndarray, spacing: float
    ) -> List[Tuple[int, int, int]]:
        flat_count = min(scores.size, max(self.top_translations * 30, self.top_translations))
        indices = np.argpartition(scores.ravel(), -flat_count)[-flat_count:]
        indices = indices[np.argsort(scores.ravel()[indices])[::-1]]
        selected: List[np.ndarray] = []
        min_voxel_distance = self.min_translation_separation / spacing
        for flat_index in indices:
            index = np.asarray(np.unravel_index(int(flat_index), scores.shape))
            if all(np.linalg.norm(index - kept) >= min_voxel_distance for kept in selected):
                selected.append(index)
            if len(selected) >= self.top_translations:
                break
        return [tuple(int(value) for value in index) for index in selected]

    @staticmethod
    def _residue_property_points(
        structure: ProteinStructure,
        center: np.ndarray,
        property_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        points = []
        values = []
        for residue in structure.get_residue_list():
            point = residue.ca_coord
            if point is None:
                continue
            if property_name == "charge":
                value = residue.charge
            elif property_name == "hydrophobic":
                value = 1.0 if residue.resname in HYDROPHOBIC_AA else 0.0
            else:
                raise ValueError(f"Unsupported grid property: {property_name}")
            if value:
                points.append(point - center)
                values.append(value)
        return np.asarray(points, dtype=float).reshape((-1, 3)), np.asarray(values, dtype=float)

    @staticmethod
    def _heavy_atom_coords(structure: ProteinStructure) -> np.ndarray:
        coords = [
            atom.coords
            for atom in structure.atoms
            if (atom.element or atom.name[:1]).upper() != "H"
        ]
        return np.asarray(coords, dtype=float)
