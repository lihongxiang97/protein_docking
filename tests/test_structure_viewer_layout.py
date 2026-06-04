import unittest

from docking.interface import InterfaceResult
from web.structure_viewer import build_structure_viewer_html, build_viewer_payload


class StructureViewerLayoutTests(unittest.TestCase):
    def test_contact_table_scroll_is_bounded_by_analysis_panel(self):
        html = build_structure_viewer_html(
            {
                "pdbText": "",
                "poseTitle": "test",
                "metrics": [],
                "receptorChains": ["A"],
                "ligandChains": ["B"],
                "receptorChainSelection": ":A",
                "ligandChainSelection": ":B",
                "interface": {
                    "receptorSelection": "",
                    "ligandSelection": "",
                    "combinedSelection": "",
                    "contactPairsTotal": 0,
                    "interfaceResiduesTotal": 0,
                    "receptorResidues": 0,
                    "ligandResidues": 0,
                },
                "contacts": [],
            }
        )
        self.assertIn(".analysis-pane {", html)
        self.assertIn("height: 100%;", html)
        self.assertIn("overflow: hidden;", html)
        self.assertIn(".panel:has(.contacts-table-wrap)", html)
        self.assertIn("overflow-y: auto;", html)
        self.assertIn("overscroll-behavior: contain;", html)
        self.assertIn("isolation: isolate;", html)
        self.assertIn("padding-top: 0;", html)
        self.assertIn("border-collapse: separate;", html)
        self.assertIn("thead {", html)
        self.assertIn("z-index: 10;", html)
        self.assertIn("background: var(--bg-panel-strong);", html)

    def test_viewer_payload_uses_requested_language(self):
        payload = build_viewer_payload(
            pdb_content="",
            interface_result=InterfaceResult(),
            receptor_chains=["A"],
            ligand_chains=["B"],
            pose_title="Rank 1 Docked Complex",
            language="en",
        )

        self.assertEqual(payload["language"], "en")
        self.assertEqual(payload["labels"]["shortest_contacts"], "Shortest Contacts")
        self.assertEqual(payload["labels"]["receptor"], "Receptor")

        html = build_structure_viewer_html(payload)
        self.assertIn('"shortest_contacts": "Shortest Contacts"', html)
        self.assertIn('id="contact-receptor-heading"', html)


if __name__ == "__main__":
    unittest.main()
