"""CAPRI/DockQ-style protein docking quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from docking.spatial import cKDTree
from docking.structure import Atom, STANDARD_AA, ProteinStructure, Residue


BACKBONE_ATOMS = ("N", "CA", "C", "O")
DOCKQ_LRMSD_SCALE = 8.5
DOCKQ_IRMSD_SCALE = 1.5


@dataclass
class DockingQuality:
    lrmsd: Optional[float]
    irmsd: Optional[float]
    fnat: Optional[float]
    dockq: Optional[float]
    native_contacts: int = 0
    model_contacts: int = 0
    shared_contacts: int = 0
    capri_class: str = "not_available"


def evaluate_complex(
    native: ProteinStructure,
    docked: ProteinStructure,
    receptor_chains: Iterable[str],
    native_ligand_chains: Iterable[str],
    contact_cutoff: float = 5.0,
    interface_cutoff: float = 10.0,
) -> DockingQuality:
    """Evaluate a docked complex with receptor-aligned CAPRI/DockQ metrics.

    FNAT uses native residue contacts with any heavy-atom pair within
    ``contact_cutoff``. Interface residues for iRMSD are defined from native
    heavy-atom contacts within ``interface_cutoff``.
    """
    receptor_chains = set(receptor_chains)
    native_ligand_chains = set(native_ligand_chains)
    docked_ligand_chains = set(docked.chains) - receptor_chains

    receptor_mobile, receptor_target = _common_atom_arrays(
        docked,
        native,
        receptor_chains,
        receptor_chains,
        BACKBONE_ATOMS,
    )
    if len(receptor_mobile) < 3:
        return DockingQuality(None, None, None, None)
    rotation, translation = kabsch_transform(
        receptor_mobile,
        receptor_target,
    )

    lrmsd = _aligned_role_rmsd(
        docked,
        native,
        docked_ligand_chains,
        native_ligand_chains,
        BACKBONE_ATOMS,
        rotation,
        translation,
    )

    native_contacts = residue_contacts(
        native, receptor_chains, native_ligand_chains, contact_cutoff
    )
    docked_contacts = residue_contacts(
        docked, receptor_chains, docked_ligand_chains, contact_cutoff
    )
    shared_contacts = native_contacts & docked_contacts
    fnat = (
        len(shared_contacts) / len(native_contacts)
        if native_contacts
        else None
    )

    interface_contacts = residue_contacts(
        native, receptor_chains, native_ligand_chains, interface_cutoff
    )
    irmsd = interface_rmsd(
        native=native,
        docked=docked,
        receptor_chains=receptor_chains,
        native_ligand_chains=native_ligand_chains,
        docked_ligand_chains=docked_ligand_chains,
        native_interface_contacts=interface_contacts,
    )

    dockq = dockq_score(fnat, lrmsd, irmsd)
    capri_class = classify_capri(fnat, lrmsd, irmsd)
    return DockingQuality(
        lrmsd=float(lrmsd) if lrmsd is not None else None,
        irmsd=float(irmsd) if irmsd is not None else None,
        fnat=float(fnat) if fnat is not None else None,
        dockq=float(dockq) if dockq is not None else None,
        native_contacts=len(native_contacts),
        model_contacts=len(docked_contacts),
        shared_contacts=len(shared_contacts),
        capri_class=capri_class,
    )


def interface_rmsd(
    native: ProteinStructure,
    docked: ProteinStructure,
    receptor_chains: Set[str],
    native_ligand_chains: Set[str],
    docked_ligand_chains: Set[str],
    native_interface_contacts: Set[Tuple[Tuple[int, str, str], Tuple[int, str, str]]],
) -> Optional[float]:
    """Return backbone iRMSD after optimal interface-atom superposition."""
    if not native_interface_contacts:
        return None
    receptor_interface = {contact[0] for contact in native_interface_contacts}
    ligand_interface = {contact[1] for contact in native_interface_contacts}
    mobile_parts: List[np.ndarray] = []
    target_parts: List[np.ndarray] = []
    for docked_structure, native_structure, docked_chains, native_chains, residues in [
        (docked, native, receptor_chains, receptor_chains, receptor_interface),
        (docked, native, docked_ligand_chains, native_ligand_chains, ligand_interface),
    ]:
        mobile, target = _common_atom_arrays(
            docked_structure,
            native_structure,
            docked_chains,
            native_chains,
            BACKBONE_ATOMS,
            allowed_residues=residues,
        )
        if len(mobile):
            mobile_parts.append(mobile)
            target_parts.append(target)

    if not mobile_parts:
        return None
    mobile = np.vstack(mobile_parts)
    target = np.vstack(target_parts)
    if len(mobile) < 3:
        return None
    rotation, translation = kabsch_transform(mobile, target)
    return rmsd(apply_rigid_transform(mobile, rotation, translation), target)


def dockq_score(
    fnat: Optional[float],
    lrmsd: Optional[float],
    irmsd: Optional[float],
) -> Optional[float]:
    if fnat is None or lrmsd is None or irmsd is None:
        return None
    return float(
        (
            fnat
            + 1.0 / (1.0 + (irmsd / DOCKQ_IRMSD_SCALE) ** 2)
            + 1.0 / (1.0 + (lrmsd / DOCKQ_LRMSD_SCALE) ** 2)
        )
        / 3.0
    )


def classify_capri(
    fnat: Optional[float],
    lrmsd: Optional[float],
    irmsd: Optional[float],
) -> str:
    """Classify a model using common CAPRI quality thresholds."""
    if fnat is None or lrmsd is None or irmsd is None:
        return "not_available"
    if fnat >= 0.5 and (lrmsd <= 1.0 or irmsd <= 1.0):
        return "high"
    if fnat >= 0.3 and (lrmsd <= 5.0 or irmsd <= 2.0):
        return "medium"
    if fnat >= 0.1 and (lrmsd <= 10.0 or irmsd <= 4.0):
        return "acceptable"
    return "incorrect"


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
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) == 0:
        raise ValueError("RMSD inputs must have matching non-empty shape (N, 3)")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def residue_contacts(
    structure: ProteinStructure,
    receptor_chains: Set[str],
    ligand_chains: Set[str],
    cutoff: float,
    rotation: Optional[np.ndarray] = None,
    translation: Optional[np.ndarray] = None,
) -> Set[Tuple[Tuple, Tuple]]:
    """Return residue-residue contacts using heavy-atom distances."""
    receptor = _heavy_residue_atoms(structure, receptor_chains, rotation, translation)
    ligand = _heavy_residue_atoms(structure, ligand_chains, rotation, translation)
    if not receptor or not ligand:
        return set()
    ligand_coords = np.asarray([coord for _, coord in ligand], dtype=float)
    ligand_ids = [residue_id for residue_id, _ in ligand]
    ligand_tree = cKDTree(ligand_coords)
    contacts = set()
    for receptor_id, receptor_coord in receptor:
        neighbor_indices = ligand_tree.query_ball_point(receptor_coord, cutoff)
        for index in neighbor_indices:
            contacts.add((receptor_id, ligand_ids[index]))
    return contacts


def _aligned_role_rmsd(
    docked: ProteinStructure,
    native: ProteinStructure,
    docked_chains: Set[str],
    native_chains: Set[str],
    atom_names: Sequence[str],
    rotation: Optional[np.ndarray],
    translation: Optional[np.ndarray],
) -> Optional[float]:
    mobile, target = _common_atom_arrays(
        docked,
        native,
        docked_chains,
        native_chains,
        atom_names,
    )
    if len(mobile) == 0:
        return None
    return rmsd(apply_rigid_transform(mobile, rotation, translation), target)


def _common_atom_arrays(
    mobile_structure: ProteinStructure,
    target_structure: ProteinStructure,
    mobile_chains: Set[str],
    target_chains: Set[str],
    atom_names: Sequence[str],
    allowed_residues: Optional[Set[Tuple[int, str, str]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    mobile = _atom_map(mobile_structure, mobile_chains, atom_names, allowed_residues)
    target = _atom_map(target_structure, target_chains, atom_names, allowed_residues)
    keys = sorted(set(mobile) & set(target))
    if not keys:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return (
        np.asarray([mobile[key] for key in keys], dtype=float),
        np.asarray([target[key] for key in keys], dtype=float),
    )


def _atom_map(
    structure: ProteinStructure,
    chains: Set[str],
    atom_names: Sequence[str],
    allowed_residues: Optional[Set[Tuple[int, str, str]]] = None,
) -> Dict[Tuple[Tuple[int, str, str], str], np.ndarray]:
    allowed_atoms = {name.upper() for name in atom_names}
    coords: Dict[Tuple[Tuple[int, str, str], str], np.ndarray] = {}
    for residue in structure.get_residue_list():
        if residue.chain_id not in chains or residue.resname not in STANDARD_AA:
            continue
        residue_id = _residue_identity(residue)
        if allowed_residues is not None and residue_id not in allowed_residues:
            continue
        for atom in residue.atoms:
            atom_name = atom.name.strip().upper()
            if atom_name not in allowed_atoms:
                continue
            key = (residue_id, atom_name)
            if key not in coords and _is_finite_coord(atom.coords):
                coords[key] = atom.coords
    return coords


def _heavy_residue_atoms(
    structure: ProteinStructure,
    chains: Set[str],
    rotation: Optional[np.ndarray] = None,
    translation: Optional[np.ndarray] = None,
) -> List[Tuple[Tuple[int, str, str], np.ndarray]]:
    atoms: List[Tuple[Tuple[int, str, str], np.ndarray]] = []
    for residue in structure.get_residue_list():
        if residue.chain_id not in chains or residue.resname not in STANDARD_AA:
            continue
        residue_id = _residue_identity(residue)
        for atom in residue.atoms:
            if not _is_heavy_atom(atom):
                continue
            coord = atom.coords
            if not _is_finite_coord(coord):
                continue
            if rotation is not None and translation is not None:
                coord = rotation @ coord + translation
            atoms.append((residue_id, coord))
    return atoms


def _is_heavy_atom(atom: Atom) -> bool:
    element = (atom.element or "").strip().upper()
    if element:
        return element != "H"
    atom_name = atom.name.strip().upper()
    return not atom_name.startswith("H") and not atom_name[:1].isdigit()


def _is_finite_coord(coord: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(coord)))


def _residue_identity(residue: Residue) -> Tuple[int, str, str]:
    return residue.resseq, residue.icode.strip(), residue.resname
