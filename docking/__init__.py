"""
Protein-Protein Interaction (PPI) Docking Package.

自主实现的蛋白-蛋白分子对接与互作预测系统。
"""

__version__ = "1.0.0"
__author__ = "PPI Docking Team"

__all__ = [
    "StructurePreprocessor",
    "ProteinDocker",
    "DockingScorer",
    "InterfaceAnalyzer",
    "PPIPredictor",
    "PoseReranker",
    "ResultVisualizer",
]

_LAZY_IMPORTS = {
    "StructurePreprocessor": ("docking.preprocess", "StructurePreprocessor"),
    "ProteinDocker": ("docking.docking", "ProteinDocker"),
    "DockingScorer": ("docking.scoring", "DockingScorer"),
    "InterfaceAnalyzer": ("docking.interface", "InterfaceAnalyzer"),
    "PPIPredictor": ("docking.ppi_predictor", "PPIPredictor"),
    "PoseReranker": ("docking.ml_reranker", "PoseReranker"),
    "ResultVisualizer": ("docking.visualization", "ResultVisualizer"),
}


def __getattr__(name):
    """Load public components only when requested."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
