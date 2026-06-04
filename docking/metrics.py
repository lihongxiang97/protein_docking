"""Receptor-aligned docking quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np

from docking.structure import STANDARD_AA, ProteinStructure, Residue


@dataclass
class DockingQuality:
    lrmsd: Optional[float]
    irmsd: Optional[float]
    fnat: Optional[float]
    dockq: Optional[float]


def evaluate_complex(
    native: ProteinStructure,
    docked: ProteinStructure,
    receptor_chains: Iterable[str],
    native_ligand_chains: Iterable[str],
    contact_cutoff: float = 8.0,
) -> DockingQuality:
    receptor_chains = set(receptor_chains)
    native_ligand_chains = set(native_ligand_chains)
    docked_ligand_chains = set(docked.chains) - receptor_chains
    native_receptor = _ca_map(native, receptor_chains)
    docked_receptor = _ca_map(docked, receptor_chains)
    receptor_keys = sorted(set(native_receptor) & set(docked_receptor))
    if len(receptor_keys) < 3:
        return DockingQuality(None, None, None, None)
    rotation, translation = kabsch_transform(
        np.asarray([docked_receptor[key] for key in receptor_keys]),
        np.asarray([native_receptor[key] for key in receptor_keys]),
    )
    native_ligand = _ca_map(native, native_ligand_chains)
    docked_ligand = _ca_map(docked, docked_ligand_chains)
    ligand_keys = sorted(set(native_ligand) & set(docked_ligand))
    if len(ligand_keys) < 3:
        return DockingQuality(None, None, None, None)
    native_ligand_coords = np.asarray([native_ligand[key] for key in ligand_keys])
    aligned_ligand_coords = apply_rigid_transform(
        np.asarray([docked_ligand[key] for key in ligand_keys]), rotation, translation
    )
    lrmsd = rmsd(aligned_ligand_coords, native_ligand_coords)
    native_contacts = residue_contacts(
        native, receptor_chains, native_ligand_chains, contact_cutoff
    )
    docked_contacts = residue_contacts(
        docked, receptor_chains, docked_ligand_chains, contact_cutoff, rotation, translation
    )
    fnat = (
        len(native_contacts & docked_contacts) / len(native_contacts)
        if native_contacts
        else None
    )
    interface_keys = [
        key for key in ligand_keys if key in {ligand_key for _, ligand_key in native_contacts}
    ]
    irmsd = None
    if interface_keys:
        irmsd = rmsd(
            apply_rigid_transform(
                np.asarray([docked_ligand[key] for key in interface_keys]),
                rotation,
                translation,
            ),
            np.asarray([native_ligand[key] for key in interface_keys]),
        )
    dockq = None
    if fnat is not None and irmsd is not None:
        dockq = (
            fnat
            + 1.0 / (1.0 + (irmsd / 1.5) ** 2)
            + 1.0 / (1.0 + (lrmsd / 8.5) ** 2)
        ) / 3.0
    return DockingQuality(float(lrmsd), irmsd, fnat, dockq)


def kabsch_transform(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("Kabsch inputs must have matching shape (N, 3)")
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ mobile_center
    return rotation, translation


def apply_rigid_transform(
    coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return (rotation @ coords.T).T + translation


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def residue_contacts(
    structure: ProteinStructure,
    receptor_chains: Set[str],
    ligand_chains: Set[str],
    cutoff: float,
    rotation: Optional[np.ndarray] = None,
    translation: Optional[np.ndarray] = None,
) -> Set[Tuple[Tuple, Tuple]]:
    receptor = [
        r
        for r in structure.get_residue_list()
        if r.chain_id in receptor_chains and r.resname in STANDARD_AA
    ]
    ligand = [
        r
        for r in structure.get_residue_list()
        if r.chain_id in ligand_chains and r.resname in STANDARD_AA
    ]
    contacts = set()
    for rec_residue in receptor:
        rec_coord = _transformed_ca(rec_residue, rotation, translation)
        if rec_coord is None:
            continue
        for lig_residue in ligand:
            lig_coord = _transformed_ca(lig_residue, rotation, translation)
            if lig_coord is not None and np.linalg.norm(rec_coord - lig_coord) <= cutoff:
                contacts.add((_residue_identity(rec_residue), _residue_identity(lig_residue)))
    return contacts


def _transformed_ca(
    residue: Residue,
    rotation: Optional[np.ndarray],
    translation: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    coord = residue.ca_coord
    if coord is not None and rotation is not None and translation is not None:
        coord = rotation @ coord + translation
    return coord


def _ca_map(structure: ProteinStructure, chains: Set[str]) -> Dict[Tuple, np.ndarray]:
    return {
        _residue_identity(residue): residue.ca_coord
        for residue in structure.get_residue_list()
        if residue.chain_id in chains
        and residue.resname in STANDARD_AA
        and residue.ca_coord is not None
    }


def _residue_identity(residue: Residue) -> Tuple[int, str, str]:
    return residue.resseq, residue.icode, residue.resname
