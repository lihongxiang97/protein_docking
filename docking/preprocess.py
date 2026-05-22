"""
结构预处理模块。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import yaml

from docking.structure import (
    STANDARD_AA,
    Atom,
    PDBParser,
    ProteinStructure,
    Residue,
)
from docking.surface import SurfaceAnalyzer

logger = logging.getLogger(__name__)

WATER_NAMES = {"HOH", "WAT", "H2O", "TIP", "TIP3", "SOL"}
ION_NAMES = {"NA", "CL", "K", "MG", "CA", "ZN", "FE", "MN", "CU", "CO", "NI"}
COMMON_LIGANDS = {"ATP", "ADP", "GTP", "GDP", "NAD", "FAD", "HEME", "NDP"}


@dataclass
class ValidationReport:
    """结构验证报告。"""
    pdb_path: str
    n_atoms: int = 0
    n_residues: int = 0
    chains: List[str] = field(default_factory=list)
    missing_atoms: List[str] = field(default_factory=list)
    non_standard_residues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_valid: bool = True
    protein_type: str = "unknown"


class StructurePreprocessor:
    """蛋白质结构预处理器。"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        prep = self.config.get("preprocessing", {})
        surf = self.config.get("surface", {})
        self.remove_water = prep.get("remove_water", True)
        self.remove_hetatm = prep.get("remove_hetatm", True)
        self.remove_ions = prep.get("remove_ions", True)
        self.add_hydrogens = prep.get("add_hydrogens", True)
        self.standardize_names = prep.get("standardize_names", True)
        self.surface_analyzer = SurfaceAnalyzer(
            probe_radius=prep.get("probe_radius", 1.4),
            sasa_threshold=surf.get("sasa_threshold", 25.0),
            neighbor_radius=surf.get("neighbor_radius", 8.0),
            neighbor_density_threshold=surf.get("neighbor_density_threshold", 0.15),
        )

    def _load_config(self, config_path: Optional[Path]) -> dict:
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        default = Path(__file__).parent.parent / "config.yaml"
        if default.exists():
            with open(default) as f:
                return yaml.safe_load(f)
        return {}

    def validate_structure(self, pdb_path: Path) -> ValidationReport:
        """检查结构完整性。"""
        pdb_path = Path(pdb_path)
        structure = PDBParser.parse(pdb_path)
        report = ValidationReport(
            pdb_path=str(pdb_path),
            n_atoms=len(structure.atoms),
            n_residues=len(structure.residues),
            chains=sorted(structure.chains),
        )

        if report.n_atoms == 0:
            report.is_valid = False
            report.warnings.append("文件中无 ATOM 记录")
            return report

        for res in structure.get_residue_list():
            if res.resname not in STANDARD_AA:
                report.non_standard_residues.append(
                    f"{res.chain_id}:{res.resseq}:{res.resname}"
                )
            has_ca = any(a.name == "CA" for a in res.atoms)
            if not has_ca and res.resname in STANDARD_AA:
                report.missing_atoms.append(f"{res.chain_id}:{res.resseq}:CA")

        if report.non_standard_residues:
            report.warnings.append(
                f"发现 {len(report.non_standard_residues)} 个非标准残基"
            )

        n_aa = sum(1 for r in structure.get_residue_list() if r.resname in STANDARD_AA)
        if n_aa > 500:
            report.protein_type = "large"
        elif n_aa > 100:
            report.protein_type = "medium"
        else:
            report.protein_type = "small"

        return report

    def preprocess(
        self,
        pdb_path: Path,
        output_path: Optional[Path] = None,
        chain_ids: Optional[List[str]] = None,
        keep_hetatm: bool = False,
    ) -> ProteinStructure:
        """
        完整预处理流程。

        Returns:
            处理后的 ProteinStructure
        """
        pdb_path = Path(pdb_path)
        structure = PDBParser.parse(pdb_path)

        if chain_ids:
            structure = self._select_chains(structure, chain_ids)

        if self.remove_water:
            structure = self._remove_by_resname(structure, WATER_NAMES)

        if self.remove_ions:
            structure = self._remove_by_resname(structure, ION_NAMES)

        if self.remove_hetatm and not keep_hetatm:
            structure = self._remove_hetatm(structure)

        if self.standardize_names:
            structure = self._standardize_atom_names(structure)

        if self.add_hydrogens:
            structure = self._add_hydrogens(structure)

        # 表面分析
        self.surface_analyzer.identify_surface_residues(structure)
        self.surface_analyzer.assign_residue_charges(structure)

        if output_path:
            structure.write_pdb(Path(output_path))

        logger.info(
            "预处理完成: %s, %d 原子, %d 残基, %d 表面残基",
            pdb_path.name,
            len(structure.atoms),
            len(structure.residues),
            sum(1 for r in structure.get_residue_list() if r.is_surface),
        )
        return structure

    def _select_chains(
        self, structure: ProteinStructure, chain_ids: List[str]
    ) -> ProteinStructure:
        allowed = set(chain_ids)
        new_atoms = [a for a in structure.atoms if a.chain_id in allowed]
        new_struct = ProteinStructure(name=structure.name, atoms=new_atoms)
        for atom in new_atoms:
            key = atom.residue_key
            if key not in new_struct.residues:
                new_struct.residues[key] = Residue(
                    chain_id=atom.chain_id,
                    resseq=atom.resseq,
                    resname=atom.resname,
                    icode=atom.icode,
                )
            new_struct.residues[key].atoms.append(atom)
        new_struct.chains = allowed & structure.chains
        return new_struct

    def _remove_by_resname(
        self, structure: ProteinStructure, names: Set[str]
    ) -> ProteinStructure:
        new_atoms = [a for a in structure.atoms if a.resname not in names]
        return self._rebuild_structure(structure, new_atoms)

    def _remove_hetatm(self, structure: ProteinStructure) -> ProteinStructure:
        new_atoms = [
            a for a in structure.atoms
            if a.record == "ATOM" and a.resname in STANDARD_AA
        ]
        return self._rebuild_structure(structure, new_atoms)

    def _rebuild_structure(
        self, structure: ProteinStructure, atoms: List[Atom]
    ) -> ProteinStructure:
        new_struct = ProteinStructure(name=structure.name, atoms=atoms)
        for atom in atoms:
            key = atom.residue_key
            if key not in new_struct.residues:
                new_struct.residues[key] = Residue(
                    chain_id=atom.chain_id,
                    resseq=atom.resseq,
                    resname=atom.resname,
                    icode=atom.icode,
                )
            new_struct.residues[key].atoms.append(atom)
        new_struct.chains = {a.chain_id for a in atoms}
        return new_struct

    def _standardize_atom_names(self, structure: ProteinStructure) -> ProteinStructure:
        """标准化原子命名 (去除数字前缀等)。"""
        for atom in structure.atoms:
            name = atom.name.strip()
            if len(name) == 4 and name[0].isdigit():
                name = name[1:]
            atom.name = name[:4].ljust(4) if len(name) <= 4 else name[:4]
            if not atom.element:
                atom.element = name[0]
        return structure

    def _add_hydrogens(self, structure: ProteinStructure) -> ProteinStructure:
        """
        简化氢原子添加：在 N、O 等极性原子附近添加 H。
        完整质子化需 pKa 计算，此处为几何近似。
        """
        new_atoms = list(structure.atoms)
        offset = max((a.serial for a in new_atoms), default=0)

        for res in structure.get_residue_list():
            for atom in res.atoms:
                if atom.name.strip() in ("N", "O", "OG", "OH", "NZ", "NE"):
                    h_coord = atom.coords + np.array([0.0, 0.0, 1.0])
                    offset += 1
                    h_atom = Atom(
                        serial=offset,
                        name=" H  ",
                        alt_loc=" ",
                        resname=atom.resname,
                        chain_id=atom.chain_id,
                        resseq=atom.resseq,
                        icode=atom.icode,
                        x=h_coord[0], y=h_coord[1], z=h_coord[2],
                        element="H",
                        record="ATOM",
                    )
                    new_atoms.append(h_atom)

        return self._rebuild_structure(structure, new_atoms)

    def get_surface_residues(self, structure: ProteinStructure) -> List[Residue]:
        """获取表面残基列表。"""
        return [r for r in structure.get_residue_list() if r.is_surface]
