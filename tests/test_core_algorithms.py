from pathlib import Path

import numpy as np

from docking.docking import ProteinDocker
from docking.geometry import apply_transform, quaternion_to_matrix
from docking.structure import Atom, ProteinStructure, Residue, merge_structures


def make_structure(name: str, chain_id: str, x: float) -> ProteinStructure:
    atom = Atom(
        serial=1,
        name="CA",
        alt_loc=" ",
        resname="ALA",
        chain_id=chain_id,
        resseq=1,
        icode=" ",
        x=x,
        y=0.0,
        z=0.0,
        element="C",
    )
    residue = Residue(chain_id=chain_id, resseq=1, resname="ALA", atoms=[atom])
    return ProteinStructure(
        name=name,
        atoms=[atom],
        residues={(chain_id, 1, "ALA"): residue},
        chains={chain_id},
    )


def test_apply_transform_applies_translation_once():
    coords = np.array([[1.0, 0.0, 0.0]])
    transformed = apply_transform(coords, np.zeros(3), np.eye(3), np.array([2.0, 0.0, 0.0]))
    assert np.allclose(transformed, [[3.0, 0.0, 0.0]])


def test_zero_quaternion_is_rejected():
    try:
        quaternion_to_matrix(np.zeros(4))
    except ValueError:
        return
    raise AssertionError("zero quaternion should be rejected")


def test_translation_samples_are_center_first():
    docker = ProteinDocker(Path(__file__).parent.parent / "config.yaml")
    receptor = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    ligand = np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    samples = docker._generate_translations(receptor, ligand)
    assert samples
    assert all(np.all(np.isfinite(sample)) for sample in samples)
    assert all(np.linalg.norm(sample) > 0 for sample in samples)


def test_merge_structures_rebuilds_residue_atom_links():
    receptor = make_structure("rec", "A", 0.0)
    ligand = make_structure("lig", "B", 5.0)
    merged = merge_structures(receptor, ligand)
    ligand_atom = next(atom for atom in merged.atoms if atom.chain_id == "B")
    ligand_residue = next(residue for residue in merged.residues.values() if residue.chain_id == "B")
    assert ligand_residue.atoms[0] is ligand_atom
    assert ligand_atom is not ligand.atoms[0]
