import tempfile
import unittest
from pathlib import Path

import numpy as np

from docking.ml_reranker import POSE_FEATURES, PoseReranker, pose_feature_dict
from docking.scoring import DockingPose, ScoreComponents
from scripts.collect_reliable_ppi_benchmarks import (
    parse_db55_cases,
    split_bound_complex,
)
from tests.benchmark_test import BenchmarkResult
from tests.evaluation import EvaluationReport


class ConstantModel:
    def predict(self, matrix):
        return np.asarray(matrix[:, POSE_FEATURES.index("search_score")], dtype=float)


class BenchmarkSourceTests(unittest.TestCase):
    def test_db55_parser_extracts_complex_chain_mapping(self):
        html = """
        <tr><td><a>1AHW_AB:C</a></td><td>Rigid-body</td><td>AA</td></tr>
        <tr><td><a>2OOB_AB:CD *</a></td><td>Medium</td><td>EI</td></tr>
        """
        cases = parse_db55_cases(html)

        self.assertEqual([case.complex_id for case in cases], ["1AHW_AB:C", "2OOB_AB:CD"])
        self.assertEqual(cases[0].pdb_id, "1AHW")
        self.assertEqual(cases[0].receptor_chains, "AB")
        self.assertEqual(cases[0].ligand_chains, "C")

    def test_split_bound_complex_writes_receptor_and_ligand_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdb_path = root / "toy.pdb"
            pdb_path.write_text(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
                "ATOM      2  CA  ALA B   1       1.000   0.000   0.000  1.00  0.00           C\n"
                "ATOM      3  CA  ALA C   1       4.000   0.000   0.000  1.00  0.00           C\n"
                "END\n",
                encoding="utf-8",
            )

            receptor_path, ligand_path = split_bound_complex(
                pdb_path,
                receptor_chains="AB",
                ligand_chains="C",
                output_dir=root / "pairs",
                complex_id="TOY_AB:C",
            )

            self.assertIn(" A   1", receptor_path.read_text(encoding="utf-8"))
            self.assertIn(" B   1", receptor_path.read_text(encoding="utf-8"))
            self.assertIn(" C   1", ligand_path.read_text(encoding="utf-8"))

    def test_pose_reranker_can_reorder_poses(self):
        low = DockingPose(scores=ScoreComponents(total=100.0), search_score=0.1)
        high = DockingPose(scores=ScoreComponents(total=1.0), search_score=200.0)
        reranker = PoseReranker(ConstantModel(), weight=1.0)

        ranked = reranker.rank_poses([low, high])

        self.assertIs(ranked[0], high)
        self.assertEqual(ranked[0].rank, 1)
        self.assertIn("raw_total_score", pose_feature_dict(high))

    def test_evaluation_report_is_ascii_safe_and_has_threshold_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            report = EvaluationReport(Path(tmp) / "evaluation")
            results = [
                BenchmarkResult("pos", 1, 1, 0.8, 10.0, 100.0, 8),
                BenchmarkResult("neg", 0, 1, 0.7, 9.0, 90.0, 7),
            ]
            y_true = np.asarray([result.label for result in results])
            y_pred = np.asarray([result.predicted_label for result in results])
            y_prob = np.asarray([result.probability for result in results])

            report._write_markdown_report(
                path,
                results,
                report.compute_classification_metrics(y_true, y_pred, y_prob),
                {},
                report.compute_threshold_diagnostic(y_true, y_prob),
            )
            text = path.read_text(encoding="utf-8")

            self.assertIn("Threshold Diagnostic", text)
            self.assertIn("O(R x T x N)", text)
            self.assertTrue(all(ord(char) < 128 for char in text))


if __name__ == "__main__":
    unittest.main()
