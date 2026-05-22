"""
Protein-Protein Interaction (PPI) Docking Package.

自主实现的蛋白-蛋白分子对接与互作预测系统。
"""

__version__ = "1.0.0"
__author__ = "PPI Docking Team"

from docking.preprocess import StructurePreprocessor
from docking.docking import ProteinDocker
from docking.scoring import DockingScorer
from docking.interface import InterfaceAnalyzer
from docking.ppi_predictor import PPIPredictor
from docking.visualization import ResultVisualizer

__all__ = [
    "StructurePreprocessor",
    "ProteinDocker",
    "DockingScorer",
    "InterfaceAnalyzer",
    "PPIPredictor",
    "ResultVisualizer",
]
