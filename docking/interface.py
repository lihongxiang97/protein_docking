"""
蛋白互作位点 (interface residues) 预测模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from docking.spatial import cKDTree

from docking.structure import HYDROPHOBIC_AA, ProteinStructure, Residue


@dataclass
class InterfaceResidue:
    """界面残基。"""
    chain_id: str
    resseq: int
    resname: str
    partner_chain: str = ""
    min_distance: float = 0.0
    contact_type: str = ""  # hydrophobic, electrostatic, hbond, van_der_waals
    partner_residues: List[str] = field(default_factory=list)


@dataclass
class InterfaceResult:
    """界面分析结果。"""
    receptor_interface: List[InterfaceResidue] = field(default_factory=list)
    ligand_interface: List[InterfaceResidue] = field(default_factory=list)
    contact_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    contact_map: Optional[np.ndarray] = None
    n_interface_residues: int = 0


class InterfaceAnalyzer:
    """互作界面残基识别与分析。"""

    def __init__(self, contact_cutoff: float = 8.0, hbond_cutoff: float = 3.5):
        self.contact_cutoff = contact_cutoff
        self.hbond_cutoff = hbond_cutoff

    def analyze(
        self,
        receptor: ProteinStructure,
        ligand: ProteinStructure,
        receptor_chains: Optional[Set[str]] = None,
        ligand_chains: Optional[Set[str]] = None,
    ) -> InterfaceResult:
        """识别界面残基与接触类型。"""
        result = InterfaceResult()
        rec_chains = receptor_chains or receptor.chains
        lig_chains = ligand_chains or ligand.chains

        rec_residues = [
            r for r in receptor.get_residue_list() if r.chain_id in rec_chains
        ]
        lig_residues = [
            r for r in ligand.get_residue_list() if r.chain_id in lig_chains
        ]

        lig_ca = []
        lig_info = []
        for res in lig_residues:
            ca = res.ca_coord
            if ca is not None:
                lig_ca.append(ca)
                lig_info.append(res)

        if not lig_ca:
            return result

        lig_tree = cKDTree(np.array(lig_ca))

        for res in rec_residues:
            ca = res.ca_coord
            if ca is None:
                continue
            neighbor_indices = lig_tree.query_ball_point(ca, self.contact_cutoff)
            if neighbor_indices:
                partners = [(lig_info[idx], float(np.linalg.norm(ca - lig_ca[idx]))) for idx in neighbor_indices]
                partners.sort(key=lambda item: item[1])
                partner, dist = partners[0]
                contact_type = self._classify_contact(res, partner, dist)
                iface = InterfaceResidue(
                    chain_id=res.chain_id,
                    resseq=res.resseq,
                    resname=res.resname,
                    partner_chain=partner.chain_id,
                    min_distance=dist,
                    contact_type=contact_type,
                    partner_residues=[
                        f"{p.resname}{p.resseq}" for p, _ in partners
                    ],
                )
                result.receptor_interface.append(iface)
                for partner_res, partner_dist in partners:
                    result.contact_pairs.append((
                        f"{res.chain_id}:{res.resname}{res.resseq}",
                        f"{partner_res.chain_id}:{partner_res.resname}{partner_res.resseq}",
                        partner_dist,
                    ))

        # 配体侧界面
        rec_ca = []
        rec_info = []
        for res in rec_residues:
            ca = res.ca_coord
            if ca is not None:
                rec_ca.append(ca)
                rec_info.append(res)

        if rec_ca:
            rec_tree = cKDTree(np.array(rec_ca))
            for res in lig_residues:
                ca = res.ca_coord
                if ca is None:
                    continue
                neighbor_indices = rec_tree.query_ball_point(ca, self.contact_cutoff)
                if neighbor_indices:
                    partners = [(rec_info[idx], float(np.linalg.norm(ca - rec_ca[idx]))) for idx in neighbor_indices]
                    partners.sort(key=lambda item: item[1])
                    partner, dist = partners[0]
                    contact_type = self._classify_contact(res, partner, dist)
                    result.ligand_interface.append(InterfaceResidue(
                        chain_id=res.chain_id,
                        resseq=res.resseq,
                        resname=res.resname,
                        partner_chain=partner.chain_id,
                        min_distance=dist,
                        contact_type=contact_type,
                        partner_residues=[f"{p.resname}{p.resseq}" for p, _ in partners],
                    ))

        result.n_interface_residues = (
            len(result.receptor_interface) + len(result.ligand_interface)
        )
        result.contact_map = self._build_contact_map(rec_residues, lig_residues)
        return result

    def _classify_contact(
        self, res_a: Residue, res_b: Residue, distance: float
    ) -> str:
        """分类接触类型。"""
        if distance < self.hbond_cutoff:
            polar = {"SER", "THR", "ASN", "GLN", "TYR", "ARG", "LYS", "HIS", "ASP", "GLU"}
            if res_a.resname in polar and res_b.resname in polar:
                return "hbond"
        if res_a.resname in HYDROPHOBIC_AA and res_b.resname in HYDROPHOBIC_AA:
            return "hydrophobic"
        charges = {"ARG", "LYS", "HIS", "ASP", "GLU"}
        if res_a.resname in charges or res_b.resname in charges:
            return "electrostatic"
        return "van_der_waals"

    def _build_contact_map(
        self,
        rec_residues: List[Residue],
        lig_residues: List[Residue],
    ) -> np.ndarray:
        """构建残基接触矩阵。"""
        n_rec = len(rec_residues)
        n_lig = len(lig_residues)
        cmap = np.zeros((n_rec, n_lig))

        lig_ca = []
        lig_col_index = []  # lig_ca 索引 → lig_residues 列索引
        for j, res in enumerate(lig_residues):
            ca = res.ca_coord
            if ca is not None:
                lig_ca.append(ca)
                lig_col_index.append(j)

        if not lig_ca:
            return cmap

        tree = cKDTree(np.array(lig_ca))
        for i, res in enumerate(rec_residues):
            ca = res.ca_coord
            if ca is None:
                continue
            k = min(len(lig_ca), 10)
            dists, indices = tree.query(ca, k=k)
            if np.isscalar(dists):
                dists, indices = [dists], [indices]
            for j_idx, dist in zip(indices, dists):
                if dist < self.contact_cutoff:
                    col = lig_col_index[j_idx]
                    cmap[i, col] = 1 - dist / self.contact_cutoff

        return cmap

    def save_interface_report(
        self, result: InterfaceResult, output_path: Path
    ) -> None:
        """保存界面残基文本报告。"""
        output_path = Path(output_path)
        lines = ["# Protein-Protein Interface Residues\n"]

        lines.append("## Receptor (Chain A) Interface Residues:\n")
        for iface in result.receptor_interface:
            lines.append(f"{iface.resname}{iface.resseq} ({iface.contact_type}, d={iface.min_distance:.2f}Å)\n")

        lines.append("\n## Ligand (Chain B) Interface Residues:\n")
        for iface in result.ligand_interface:
            lines.append(f"{iface.resname}{iface.resseq} ({iface.contact_type}, d={iface.min_distance:.2f}Å)\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def to_dataframe(self, result: InterfaceResult) -> pd.DataFrame:
        """转为 DataFrame。"""
        rows = []
        for iface in result.receptor_interface:
            rows.append({
                "role": "receptor",
                "chain": iface.chain_id,
                "residue": f"{iface.resname}{iface.resseq}",
                "contact_type": iface.contact_type,
                "min_distance": iface.min_distance,
                "partners": ",".join(iface.partner_residues),
            })
        for iface in result.ligand_interface:
            rows.append({
                "role": "ligand",
                "chain": iface.chain_id,
                "residue": f"{iface.resname}{iface.resseq}",
                "contact_type": iface.contact_type,
                "min_distance": iface.min_distance,
                "partners": ",".join(iface.partner_residues),
            })
        return pd.DataFrame(rows)

    def build_interaction_network(
        self, result: InterfaceResult
    ) -> "nx.Graph":
        """构建残基互作网络。"""
        import networkx as nx
        G = nx.Graph()
        for a, b, dist in result.contact_pairs:
            G.add_edge(a, b, distance=dist, weight=1.0 / (dist + 0.1))
        return G
