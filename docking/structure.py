"""
蛋白质结构数据模型与 PDB 解析。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# 标准氨基酸
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

# 疏水残基
HYDROPHOBIC_AA = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}

# 带电残基 (pH 7 近似)
POSITIVE_AA = {"ARG", "LYS", "HIS"}
NEGATIVE_AA = {"ASP", "GLU"}

# 原子范德华半径 (Å)
VDW_RADIUS = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "DEFAULT": 1.70,
}

# 残基标准原子 (用于 SASA)
RESIDUE_ATOMS = {
    "GLY": ["N", "CA", "C", "O"],
    "ALA": ["N", "CA", "C", "O", "CB"],
    "VAL": ["N", "CA", "C", "O", "CB"],
    "LEU": ["N", "CA", "C", "O", "CB"],
    "ILE": ["N", "CA", "C", "O", "CB"],
    "MET": ["N", "CA", "C", "O", "CB"],
    "PHE": ["N", "CA", "C", "O", "CB"],
    "TRP": ["N", "CA", "C", "O", "CB"],
    "PRO": ["N", "CA", "C", "O", "CB"],
    "SER": ["N", "CA", "C", "O", "CB"],
    "THR": ["N", "CA", "C", "O", "CB"],
    "CYS": ["N", "CA", "C", "O", "CB"],
    "TYR": ["N", "CA", "C", "O", "CB"],
    "ASN": ["N", "CA", "C", "O", "CB"],
    "GLN": ["N", "CA", "C", "O", "CB"],
    "ASP": ["N", "CA", "C", "O", "CB"],
    "GLU": ["N", "CA", "C", "O", "CB"],
    "LYS": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB"],
    "HIS": ["N", "CA", "C", "O", "CB"],
}


@dataclass
class Atom:
    """单个原子。"""
    serial: int
    name: str
    alt_loc: str
    resname: str
    chain_id: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    bfactor: float = 0.0
    element: str = ""
    record: str = "ATOM"

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @coords.setter
    def coords(self, value: np.ndarray) -> None:
        self.x, self.y, self.z = float(value[0]), float(value[1]), float(value[2])

    @property
    def residue_key(self) -> Tuple[str, int, str]:
        return (self.chain_id, self.resseq, self.resname)

    def vdw_radius(self) -> float:
        elem = self.element.upper() if self.element else self.name[0]
        return VDW_RADIUS.get(elem, VDW_RADIUS["DEFAULT"])


@dataclass
class Residue:
    """残基。"""
    chain_id: str
    resseq: int
    resname: str
    icode: str = ""
    atoms: List[Atom] = field(default_factory=list)
    sasa: float = 0.0
    is_surface: bool = False
    charge: float = 0.0

    @property
    def key(self) -> Tuple[str, int, str]:
        return (self.chain_id, self.resseq, self.resname)

    @property
    def ca_coord(self) -> Optional[np.ndarray]:
        for atom in self.atoms:
            if atom.name == "CA":
                return atom.coords
        if self.atoms:
            return self.atoms[0].coords
        return None

    @property
    def center_of_mass(self) -> np.ndarray:
        if not self.atoms:
            return np.zeros(3)
        coords = np.array([a.coords for a in self.atoms])
        return coords.mean(axis=0)


@dataclass
class ProteinStructure:
    """蛋白质结构。"""
    name: str
    atoms: List[Atom] = field(default_factory=list)
    residues: Dict[Tuple[str, int, str], Residue] = field(default_factory=dict)
    chains: Set[str] = field(default_factory=set)

    @property
    def coords(self) -> np.ndarray:
        if not self.atoms:
            return np.zeros((0, 3))
        return np.array([a.coords for a in self.atoms])

    @property
    def center(self) -> np.ndarray:
        c = self.coords
        if len(c) == 0:
            return np.zeros(3)
        return c.mean(axis=0)

    def get_residue_list(self) -> List[Residue]:
        return sorted(self.residues.values(), key=lambda r: (r.chain_id, r.resseq))

    def transform(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        """应用刚体变换。"""
        for atom in self.atoms:
            atom.coords = rotation @ atom.coords + translation

    def to_pdb_lines(self) -> List[str]:
        """导出 PDB 格式行。"""
        lines = []
        for i, atom in enumerate(self.atoms, 1):
            line = (
                f"{atom.record:<6}{i:5d} {atom.name:>4}{atom.alt_loc:1}"
                f"{atom.resname:>3} {atom.chain_id:1}{atom.resseq:4d}{atom.icode:1}   "
                f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
                f"{atom.occupancy:6.2f}{atom.bfactor:6.2f}          "
                f"{atom.element:>2}\n"
            )
            lines.append(line)
        lines.append("END\n")
        return lines

    def write_pdb(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.writelines(self.to_pdb_lines())


class PDBParser:
    """PDB 文件解析器。"""

    ATOM_RE = re.compile(
        r"^(ATOM|HETATM)\s+(\d+)\s+(\S+)\s+(\S?)\s+(\S{3})\s+(\S)\s*"
        r"(-?\d+)\s*(\S?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
        r"\s+(-?\d+\.\d+)?\s+(-?\d+\.\d+)?"
    )

    @classmethod
    def parse(cls, pdb_path: Path, structure_name: str = "") -> ProteinStructure:
        pdb_path = Path(pdb_path)
        name = structure_name or pdb_path.stem
        structure = ProteinStructure(name=name)
        residue_map: Dict[Tuple, Residue] = {}

        with open(pdb_path) as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                atom = cls._parse_atom_line(line)
                if atom is None:
                    continue
                structure.atoms.append(atom)
                structure.chains.add(atom.chain_id)
                key = atom.residue_key
                if key not in residue_map:
                    residue_map[key] = Residue(
                        chain_id=atom.chain_id,
                        resseq=atom.resseq,
                        resname=atom.resname,
                        icode=atom.icode,
                    )
                residue_map[key].atoms.append(atom)

        structure.residues = residue_map
        return structure

    @classmethod
    def _parse_atom_line(cls, line: str) -> Optional[Atom]:
        try:
            record = line[0:6].strip()
            serial = int(line[6:11])
            name = line[12:16].strip()
            alt_loc = line[16:17].strip() or " "
            resname = line[17:20].strip()
            chain_id = line[21:22].strip() or "A"
            resseq = int(line[22:26])
            icode = line[26:27].strip() or " "
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            occ = float(line[54:60]) if len(line) > 54 else 1.0
            bfac = float(line[60:66]) if len(line) > 60 else 0.0
            element = line[76:78].strip() if len(line) > 76 else name[0]
            return Atom(
                serial=serial, name=name, alt_loc=alt_loc, resname=resname,
                chain_id=chain_id, resseq=resseq, icode=icode,
                x=x, y=y, z=z, occupancy=occ, bfactor=bfac,
                element=element, record=record,
            )
        except (ValueError, IndexError):
            return None


def merge_structures(receptor: ProteinStructure, ligand: ProteinStructure) -> ProteinStructure:
    """合并两个结构为复合物。"""
    merged = ProteinStructure(name=f"{receptor.name}_{ligand.name}")
    offset = len(receptor.atoms)
    for atom in receptor.atoms:
        merged.atoms.append(atom)
    for atom in ligand.atoms:
        new_atom = Atom(
            serial=atom.serial + offset,
            name=atom.name, alt_loc=atom.alt_loc, resname=atom.resname,
            chain_id=atom.chain_id, resseq=atom.resseq, icode=atom.icode,
            x=atom.x, y=atom.y, z=atom.z,
            occupancy=atom.occupancy, bfactor=atom.bfactor,
            element=atom.element, record=atom.record,
        )
        merged.atoms.append(new_atom)
    merged.residues = {**receptor.residues, **ligand.residues}
    merged.chains = receptor.chains | ligand.chains
    return merged
