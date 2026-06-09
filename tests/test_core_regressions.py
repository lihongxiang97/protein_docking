import tempfile
import unittest
from pathlib import Path

import numpy as np

from docking.config import load_config
from docking.docking import ProteinDocker
from docking.fft_search import FFTDockingSearch
from docking.interface import InterfaceResult
from docking.metrics import evaluate_complex, residue_contacts
from docking.ppi_predictor import PPIPredictor
from docking.scoring import DockingScorer, ScoreComponents
from docking.spatial import cKDTree
from docking.structure import Atom, ProteinStructure, Residue, merge_structures
from docking.structure import ensure_unique_chain_ids
from docking.surface import SurfaceAnalyzer


def make_structure(name, chain, offset=0.0):
    structure = ProteinStructure(name=name)
    for index in range(4):
        atom = Atom(
            serial=index + 1,
            name="CA",
            alt_loc=" ",
            resname="ALA",
            chain_id=chain,
            resseq=index + 1,
            icode=" ",
            x=offset + index * 3.8,
            y=0.0,
            z=0.0,
            element="C",
        )
        structure.atoms.append(atom)
        structure.residues[atom.residue_key] = Residue(
            chain_id=chain,
            resseq=index + 1,
            resname="ALA",
            icode=" ",
            atoms=[atom],
        )
        structure.chains.add(chain)
    return structure


def make_backbone_structure(name, chain, offset=(0.0, 0.0, 0.0)):
    structure = ProteinStructure(name=name)
    origin = np.asarray(offset, dtype=float)
    serial = 1
    for index in range(3):
        base = origin + np.array([index * 3.8, 0.0, 0.0])
        residue = Residue(chain_id=chain, resseq=index + 1, resname="ALA", icode=" ")
        for atom_name, delta, element in [
            ("N", (-1.2, 0.0, 0.0), "N"),
            ("CA", (0.0, 0.0, 0.0), "C"),
            ("C", (1.3, 0.0, 0.0), "C"),
            ("O", (1.8, 0.5, 0.0), "O"),
            ("CB", (0.0, 1.5, 0.0), "C"),
        ]:
            coord = base + np.asarray(delta, dtype=float)
            atom = Atom(
                serial=serial,
                name=atom_name,
                alt_loc=" ",
                resname="ALA",
                chain_id=chain,
                resseq=index + 1,
                icode=" ",
                x=float(coord[0]),
                y=float(coord[1]),
                z=float(coord[2]),
                element=element,
            )
            residue.atoms.append(atom)
            structure.atoms.append(atom)
            serial += 1
        structure.residues[residue.key] = residue
        structure.chains.add(chain)
    return structure


class CoreRegressionTests(unittest.TestCase):
    def test_residue_key_preserves_insertion_code(self):
        atom_a = Atom(1, "CA", " ", "ALA", "A", 10, "A", 0, 0, 0)
        atom_b = Atom(2, "CA", " ", "ALA", "A", 10, "B", 0, 0, 0)
        self.assertNotEqual(atom_a.residue_key, atom_b.residue_key)

    def test_chain_remapping_preserves_residue_annotations(self):
        structure = make_structure("ligand", "A")
        residue = structure.get_residue_list()[0]
        residue.charge = -1.0
        residue.sasa = 42.0
        residue.is_surface = True
        remapped = ensure_unique_chain_ids(structure, {"A"})
        copied = remapped.get_residue_list()[0]
        self.assertEqual(copied.charge, -1.0)
        self.assertEqual(copied.sasa, 42.0)
        self.assertTrue(copied.is_surface)

    def test_sasa_aggregation_uses_atom_keys_not_residue_sort_order(self):
        structure = ProteinStructure(name="out_of_order")
        residue_b = Residue("B", 1, "ALA", " ")
        residue_a = Residue("A", 1, "ALA", " ")
        atom_b = Atom(1, "CA", " ", "ALA", "B", 1, " ", 0, 0, 0, element="C")
        atom_a = Atom(2, "CA", " ", "ALA", "A", 1, " ", 100, 0, 0, element="C")
        residue_b.atoms.append(atom_b)
        residue_a.atoms.append(atom_a)
        structure.atoms = [atom_b, atom_a]
        structure.residues = {residue_b.key: residue_b, residue_a.key: residue_a}
        structure.chains = {"A", "B"}
        values = SurfaceAnalyzer().compute_sasa(structure)
        self.assertEqual(set(values), {residue_a.key, residue_b.key})
        self.assertGreater(values[residue_a.key], 0)
        self.assertGreater(values[residue_b.key], 0)

    def test_spatial_index_api(self):
        tree = cKDTree(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
        self.assertEqual(tree.query_ball_point(np.array([0.0, 0.0, 0.0]), 1.0), [0])
        distance, index = tree.query(np.array([4.0, 0.0, 0.0]))
        self.assertEqual(int(index), 1)
        self.assertAlmostEqual(float(distance), 1.0)

    def test_invalid_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("docking:\n  coarse_rotations: 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_docker_generates_ranked_pose_for_small_structures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receptor = make_structure("receptor", "A")
            ligand = make_structure("ligand", "B")
            receptor_path = root / "receptor.pdb"
            ligand_path = root / "ligand.pdb"
            receptor.write_pdb(receptor_path)
            ligand.write_pdb(ligand_path)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
docking:
  coarse_rotations: 2
  coarse_translations: 2
  translation_step: 2.0
  mc_iterations: 2
  mc_temperature: 2.0
  contact_distance: 5.0
  clash_distance: 2.0
  top_n_poses: 2
  grid_search: true
  monte_carlo: true
preprocessing:
  add_hydrogens: false
scoring: {}
ppi_prediction:
  interaction_threshold: 0.5
""",
                encoding="utf-8",
            )
            poses, _, _ = ProteinDocker(config_path).dock(
                receptor_path, ligand_path, root / "results"
            )
            self.assertTrue(poses)
            self.assertEqual(poses[0].rank, 1)
            self.assertIsNotNone(poses[0].complex_structure)

    def test_fft_correlation_reports_required_translation(self):
        receptor = np.zeros((9, 9, 9), dtype=float)
        ligand = np.zeros((9, 9, 9), dtype=float)
        receptor[6, 4, 4] = 1.0
        ligand[4, 4, 4] = 1.0
        correlation = FFTDockingSearch._correlate(np.fft.fftn(receptor), ligand)
        peak = np.asarray(np.unravel_index(np.argmax(correlation), correlation.shape))
        shift = peak - np.asarray(correlation.shape) // 2
        self.assertTrue(np.array_equal(shift, [2, 0, 0]))

    def test_ambiguous_restraint_score_rewards_close_active_residues(self):
        receptor = make_structure("receptor", "A", offset=0.0)
        ligand_close = make_structure("ligand", "B", offset=4.0)
        ligand_far = make_structure("ligand", "B", offset=40.0)
        scorer = DockingScorer()
        scorer.restraints = {
            "enabled": True,
            "receptor_active": ["A:1"],
            "ligand_active": ["B:1"],
            "target_distance": 6.0,
            "upper_distance": 10.0,
        }
        close_score, close_violations = scorer._score_restraints(receptor, ligand_close)
        far_score, far_violations = scorer._score_restraints(receptor, ligand_far)
        self.assertGreater(close_score, far_score)
        self.assertEqual(close_violations, 0)
        self.assertGreater(far_violations, 0)

    def test_identical_complex_has_perfect_docking_metrics(self):
        receptor = make_structure("receptor", "A", offset=0.0)
        ligand = make_structure("ligand", "B", offset=4.0)
        complex_structure = merge_structures(receptor, ligand)
        quality = evaluate_complex(complex_structure, complex_structure, {"A"}, {"B"})
        self.assertAlmostEqual(quality.lrmsd, 0.0)
        self.assertAlmostEqual(quality.irmsd, 0.0)
        self.assertAlmostEqual(quality.fnat, 1.0)
        self.assertAlmostEqual(quality.dockq, 1.0)
        self.assertEqual(quality.capri_class, "high")

    def test_native_contacts_use_heavy_atoms_not_only_ca_atoms(self):
        receptor = ProteinStructure(name="receptor")
        rec_residue = Residue(chain_id="A", resseq=1, resname="ALA", icode=" ")
        for serial, atom_name, x in [(1, "CA", 0.0), (2, "CB", 0.0)]:
            atom = Atom(serial, atom_name, " ", "ALA", "A", 1, " ", x, 0.0, 0.0, element="C")
            rec_residue.atoms.append(atom)
            receptor.atoms.append(atom)
        receptor.residues[rec_residue.key] = rec_residue
        receptor.chains.add("A")

        ligand = ProteinStructure(name="ligand")
        lig_residue = Residue(chain_id="B", resseq=1, resname="ALA", icode=" ")
        for serial, atom_name, x in [(3, "CA", 12.0), (4, "CB", 4.5)]:
            atom = Atom(serial, atom_name, " ", "ALA", "B", 1, " ", x, 0.0, 0.0, element="C")
            lig_residue.atoms.append(atom)
            ligand.atoms.append(atom)
        ligand.residues[lig_residue.key] = lig_residue
        ligand.chains.add("B")

        contacts = residue_contacts(merge_structures(receptor, ligand), {"A"}, {"B"}, 5.0)
        self.assertEqual(len(contacts), 1)

    def test_capri_metrics_penalize_shifted_ligand_pose(self):
        receptor = make_backbone_structure("receptor", "A", offset=(0.0, 0.0, 0.0))
        native_ligand = make_backbone_structure("ligand", "B", offset=(0.0, 4.0, 0.0))
        shifted_ligand = make_backbone_structure("ligand", "B", offset=(0.0, 16.0, 0.0))
        native_complex = merge_structures(receptor, native_ligand)
        shifted_complex = merge_structures(receptor, shifted_ligand)

        quality = evaluate_complex(native_complex, shifted_complex, {"A"}, {"B"})

        self.assertGreater(quality.lrmsd, 8.0)
        self.assertEqual(quality.shared_contacts, 0)
        self.assertAlmostEqual(quality.fnat, 0.0)
        self.assertLess(quality.dockq, 0.35)
        self.assertEqual(quality.capri_class, "incorrect")

    def test_ppi_rule_model_penalizes_overpacked_interfaces(self):
        predictor = PPIPredictor()
        features = {
            "interface_area": 2500.0,
            "contact_residues": 45.0,
            "hydrophobic_ratio": 0.3,
            "electrostatic_score": 0.0,
            "docking_score": 55.0,
            "hbond_count": 12.0,
            "clash_penalty": 8.0,
            "n_interface_residues": 45.0,
            "mean_contact_distance": 6.0,
            "contact_density": 0.12,
        }
        low_density_prob, _, _ = predictor._rule_based_predict(
            ScoreComponents(), InterfaceResult(), features
        )
        features["contact_density"] = 0.55
        high_density_prob, _, _ = predictor._rule_based_predict(
            ScoreComponents(), InterfaceResult(), features
        )

        self.assertLess(high_density_prob, low_density_prob)

    def test_ppi_rule_model_keeps_experimental_complex_features_positive(self):
        predictor = PPIPredictor()
        features = {
            "interface_area": 1215.0,
            "contact_residues": 18.0,
            "hydrophobic_ratio": 0.1667,
            "electrostatic_score": 0.0,
            "docking_score": 51.0,
            "hbond_count": 12.0,
            "clash_penalty": 0.0,
            "n_interface_residues": 18.0,
            "mean_contact_distance": 6.67,
            "contact_density": 0.097,
        }

        probability, interacts, _ = predictor._rule_based_predict(
            ScoreComponents(), InterfaceResult(), features
        )

        self.assertTrue(interacts)
        self.assertGreater(probability, 0.65)


if __name__ == "__main__":
    unittest.main()
