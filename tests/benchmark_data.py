"""
Benchmark 数据集管理：下载 PDB 复合物与生成示例结构。
"""

from __future__ import annotations

import json
import logging
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 已知互作复合物 (PDB ID, 受体链, 配体链)
POSITIVE_BENCHMARK = [
    ("1AY7", "A", "B"),   # Barnase-Barstar
    ("1BUH", "A", "B"),   # CDK2-Cyclin
    ("1CGI", "E", "I"),   # Eglin-Chemotrypsin
    ("1DQJ", "A", "B"),   # RhoGAP-RhoA
    ("1EAW", "A", "B"),   # Integrin
    ("1F34", "A", "B"),   # Ferredoxin
    ("1FQJ", "A", "B"),   # RalGDS-Ral
    ("1FQ1", "A", "B"),   # UbcH7-CHIP
    ("1GC1", "C", "Q"),   # GCN4-p16
    ("1GP2", "A", "B"),   # HRAS-RalGDS
    ("1H1D", "A", "B"),   # TCR-pMHC
    ("1I2M", "A", "B"),   # Complement
    ("1JTD", "A", "B"),   # Cytokine receptor
    ("1KAC", "A", "B"),   # Alpha-amylase inhibitor
    ("1M9C", "A", "B"),   # Ubiquitin complex
]

# 非互作/分离链作为负样本 (同一 PDB 的不同链组合，或小型单链蛋白对)
NEGATIVE_BENCHMARK = [
    ("1CRN", "A", "A"),   # Crambin 自身 (小蛋白)
    ("1UBQ", "A", "A"),   # Ubiquitin
    ("1PGB", "A", "A"),   # Protein G B1
    ("1VII", "A", "A"),   # Villin headpiece
    ("2VLX", "A", "A"),   # Lysozyme
    ("1BBA", "A", "A"),   # Rubredoxin
    ("1ENH", "A", "A"),   # Engrailed homeodomain
    ("1FKB", "A", "A"),   # FKBP
]


@dataclass
class BenchmarkPair:
    """Benchmark 蛋白对。"""
    pdb_id: str
    receptor_chain: str
    ligand_chain: str
    label: int  # 1=互作, 0=非互作
    receptor_path: Optional[Path] = None
    ligand_path: Optional[Path] = None
    native_complex_path: Optional[Path] = None


@dataclass(frozen=True)
class ExampleStructure:
    """内置示例结构定义。"""
    example_id: str
    pdb_id: str
    receptor_chain: str
    ligand_chain: str
    display_name: str
    receptor_name: str
    ligand_name: str
    source_url: str
    description: str


EXAMPLE_CASES = [
    ExampleStructure(
        example_id="1AY7_rnase_sa_barstar",
        pdb_id="1AY7",
        receptor_chain="A",
        ligand_chain="B",
        display_name="RNase Sa - Barstar",
        receptor_name="Ribonuclease Sa",
        ligand_name="Barstar",
        source_url="https://www.rcsb.org/structure/1AY7",
        description="RCSB PDB 已知互作复合物，适合作为小型蛋白-蛋白对接示例。",
    ),
    ExampleStructure(
        example_id="1BUH_cdk2_ckshs1",
        pdb_id="1BUH",
        receptor_chain="A",
        ligand_chain="B",
        display_name="CDK2 - CksHs1",
        receptor_name="Cyclin-dependent kinase 2",
        ligand_name="CksHs1",
        source_url="https://www.rcsb.org/structure/1BUH",
        description="RCSB PDB 细胞周期调控相关复合物，界面相对紧凑。",
    ),
    ExampleStructure(
        example_id="1OW3_rhoa_rhogap",
        pdb_id="1OW3",
        receptor_chain="A",
        ligand_chain="B",
        display_name="RhoA - RhoGAP",
        receptor_name="Rho GTPase-activating protein",
        ligand_name="Transforming protein RhoA",
        source_url="https://www.rcsb.org/structure/1OW3",
        description="RCSB PDB 信号转导复合物，适合演示不同形状蛋白的结合界面。",
    ),
]

DEFAULT_EXAMPLE_ID = EXAMPLE_CASES[0].example_id


def download_pdb(pdb_id: str, output_dir: Path) -> Path:
    """从 RCSB PDB 下载结构。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = pdb_id.upper()
    out_path = output_dir / f"{pdb_id}.pdb"

    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, out_path)
        logger.info("已下载: %s", pdb_id)
        return out_path
    except Exception as e:
        logger.warning("下载失败 %s: %s，将生成合成结构", pdb_id, e)
        return generate_synthetic_pdb(pdb_id, out_path)


def download_pdb_with_fallback(
    pdb_id: str,
    output_dir: Path,
    fallback_dirs: Optional[List[Path]] = None,
) -> Path:
    """从 RCSB 下载结构，失败时优先复用本地缓存。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = pdb_id.upper()
    out_path = output_dir / f"{pdb_id}.pdb"

    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, out_path)
        logger.info("已从 RCSB 下载示例结构: %s", pdb_id)
        return out_path
    except Exception as exc:
        for fallback_dir in fallback_dirs or []:
            candidate = Path(fallback_dir) / f"{pdb_id}.pdb"
            if candidate.exists() and candidate.stat().st_size > 1000:
                shutil.copyfile(candidate, out_path)
                logger.warning("下载失败 %s: %s，已改用本地缓存 %s", pdb_id, exc, candidate)
                return out_path
        logger.warning("下载失败 %s: %s，将生成合成结构", pdb_id, exc)
        return generate_synthetic_pdb(pdb_id, out_path)


def extract_chain(pdb_path: Path, chain_id: str, output_path: Path) -> Path:
    """从 PDB 提取单链。"""
    output_path = Path(output_path)
    lines = []
    with open(pdb_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("ATOM") and len(line) > 21:
                if line[21] == chain_id:
                    lines.append(line)
    if not lines:
        # 尝试提取第一条链
        with open(pdb_path, encoding="utf-8", errors="replace") as f:
            seen_chains = set()
            for line in f:
                if line.startswith("ATOM"):
                    ch = line[21]
                    if ch not in seen_chains:
                        seen_chains.add(ch)
                    if ch == chain_id or (not lines and ch == list(seen_chains)[0]):
                        lines.append(line)
    lines.append("END\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return output_path


def generate_synthetic_pdb(pdb_id: str, output_path: Path, n_residues: int = 30) -> Path:
    """
    生成合成 α-螺旋蛋白结构 (用于离线测试)。
    沿螺旋轴放置 CA 原子。
    """
    import math
    output_path = Path(output_path)
    lines = []
    atom_id = 1
    aa_list = ["ALA", "VAL", "LEU", "ILE", "GLY", "SER", "THR", "ASP", "GLU", "LYS"]

    for i in range(n_residues):
        resname = aa_list[i % len(aa_list)]
        resseq = i + 1
        # α-螺旋参数
        t = i * 100 * math.pi / 180
        x = 5.0 * math.cos(t)
        y = 5.0 * math.sin(t)
        z = i * 1.5
        for atom_name, offset in [("N", (0, 0, -1.5)), ("CA", (0, 0, 0)), ("C", (0, 0, 1.5)), ("O", (1.2, 0, 1.5))]:
            ax, ay, az = x + offset[0], y + offset[1], z + offset[2]
            line = (
                f"ATOM  {atom_id:5d} {atom_name:>4} {resname:>3} A{resseq:4d}    "
                f"{ax:8.3f}{ay:8.3f}{az:8.3f}  1.00  0.00           "
                f"{atom_name[0]:>2}\n"
            )
            lines.append(line)
            atom_id += 1
    lines.append("END\n")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return output_path


def prepare_benchmark_dataset(
    data_dir: Path,
    download: bool = True,
) -> Tuple[List[BenchmarkPair], List[BenchmarkPair]]:
    """准备正负样本 benchmark 数据集。"""
    data_dir = Path(data_dir)
    positive_dir = data_dir / "positive"
    negative_dir = data_dir / "negative"
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    positive_pairs = []
    for pdb_id, rec_chain, lig_chain in POSITIVE_BENCHMARK:
        pair = BenchmarkPair(
            pdb_id=pdb_id, receptor_chain=rec_chain,
            ligand_chain=lig_chain, label=1,
        )
        if download:
            pdb_path = download_pdb(pdb_id, data_dir / "raw")
            rec_path = positive_dir / f"{pdb_id}_receptor_{rec_chain}.pdb"
            lig_path = positive_dir / f"{pdb_id}_ligand_{lig_chain}.pdb"
            extract_chain(pdb_path, rec_chain, rec_path)
            extract_chain(pdb_path, lig_chain, lig_path)
            pair.receptor_path = rec_path
            pair.ligand_path = lig_path
            pair.native_complex_path = pdb_path
        positive_pairs.append(pair)

    negative_pairs = []
    for i, (pdb_id, rec_chain, lig_chain) in enumerate(NEGATIVE_BENCHMARK):
        pair = BenchmarkPair(
            pdb_id=f"{pdb_id}_{i}", receptor_chain=rec_chain,
            ligand_chain=lig_chain, label=0,
        )
        if download:
            pdb_path = download_pdb(pdb_id, data_dir / "raw")
            rec_path = negative_dir / f"{pdb_id}_receptor.pdb"
            lig_path = negative_dir / f"{pdb_id}_ligand_shifted.pdb"
            extract_chain(pdb_path, rec_chain, rec_path)
            # 负样本：同一结构复制但空间分离
            extract_chain(pdb_path, rec_chain, lig_path)
            _shift_pdb(lig_path, shift=(50.0, 50.0, 50.0))
            pair.receptor_path = rec_path
            pair.ligand_path = lig_path
        negative_pairs.append(pair)

    # 保存元数据
    meta = {
        "positive": [{"pdb_id": p.pdb_id, "label": p.label} for p in positive_pairs],
        "negative": [{"pdb_id": p.pdb_id, "label": p.label} for p in negative_pairs],
    }
    with open(data_dir / "benchmark_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return positive_pairs, negative_pairs


def _shift_pdb(pdb_path: Path, shift: Tuple[float, float, float]) -> None:
    """平移 PDB 坐标 (制造非互作负样本)。"""
    lines = []
    with open(pdb_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("ATOM"):
                x = float(line[30:38]) + shift[0]
                y = float(line[38:46]) + shift[1]
                z = float(line[46:54]) + shift[2]
                line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
            lines.append(line)
    with open(pdb_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _write_example_manifest(
    output_dir: Path,
    example_rows: List[Dict[str, str]],
) -> Path:
    manifest = {
        "source": "RCSB Protein Data Bank",
        "default_example_id": DEFAULT_EXAMPLE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "examples": example_rows,
    }
    manifest_path = Path(output_dir) / "example_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path


def _write_example_readme(output_dir: Path, example_rows: List[Dict[str, str]]) -> Path:
    lines = [
        "# Example PDB Structures",
        "",
        "以下示例来自 RCSB Protein Data Bank，可直接作为 Web 端和命令行示例输入。",
        "",
        "| Example ID | PDB ID | Receptor | Ligand | Files | Source |",
        "|---|---|---|---|---|---|",
    ]
    for row in example_rows:
        files = f"`{row['receptor_file']}` / `{row['ligand_file']}`"
        source = f"[RCSB {row['pdb_id']}]({row['source_url']})"
        lines.append(
            f"| {row['example_id']} | {row['pdb_id']} | {row['receptor_name']} (chain {row['receptor_chain']}) | "
            f"{row['ligand_name']} (chain {row['ligand_chain']}) | {files} | {source} |"
        )
    lines.extend(
        [
            "",
            f"默认快捷输入文件为 `{DEFAULT_EXAMPLE_ID}` 对应的 `receptor.pdb` 与 `ligand.pdb`。",
            "",
        ]
    )
    readme_path = Path(output_dir) / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return readme_path


def prepare_example_dataset(output_dir: Path) -> List[Dict[str, str]]:
    """准备真实互作示例数据集，并写出 manifest。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    project_root = Path(__file__).resolve().parent.parent
    fallback_dirs = [project_root / "data" / "benchmark" / "raw"]

    example_rows: List[Dict[str, str]] = []
    for example in EXAMPLE_CASES:
        pdb_path = download_pdb_with_fallback(example.pdb_id, raw_dir, fallback_dirs=fallback_dirs)
        receptor_file = f"{example.example_id}_receptor_{example.receptor_chain}.pdb"
        ligand_file = f"{example.example_id}_ligand_{example.ligand_chain}.pdb"
        receptor_path = output_dir / receptor_file
        ligand_path = output_dir / ligand_file
        extract_chain(pdb_path, example.receptor_chain, receptor_path)
        extract_chain(pdb_path, example.ligand_chain, ligand_path)
        example_rows.append(
            {
                "example_id": example.example_id,
                "pdb_id": example.pdb_id,
                "display_name": example.display_name,
                "receptor_name": example.receptor_name,
                "ligand_name": example.ligand_name,
                "receptor_chain": example.receptor_chain,
                "ligand_chain": example.ligand_chain,
                "receptor_file": receptor_file,
                "ligand_file": ligand_file,
                "description": example.description,
                "source_url": example.source_url,
            }
        )

    default_row = next(row for row in example_rows if row["example_id"] == DEFAULT_EXAMPLE_ID)
    shutil.copyfile(output_dir / default_row["receptor_file"], output_dir / "receptor.pdb")
    shutil.copyfile(output_dir / default_row["ligand_file"], output_dir / "ligand.pdb")
    _write_example_manifest(output_dir, example_rows)
    _write_example_readme(output_dir, example_rows)
    return example_rows


def list_example_pairs(output_dir: Path) -> List[Dict[str, str]]:
    """读取示例对清单；若不存在则自动生成。"""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "example_manifest.json"
    if not manifest_path.exists():
        return prepare_example_dataset(output_dir)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest.get("examples", [])


def generate_example_pair(
    output_dir: Path,
    example_id: Optional[str] = None,
) -> Tuple[Path, Path]:
    """返回指定示例对；默认返回内置默认示例。"""
    output_dir = Path(output_dir)
    examples = list_example_pairs(output_dir)
    wanted_id = example_id or DEFAULT_EXAMPLE_ID
    for row in examples:
        if row["example_id"] == wanted_id:
            receptor_path = output_dir / row["receptor_file"]
            ligand_path = output_dir / row["ligand_file"]
            if receptor_path.exists() and ligand_path.exists():
                return receptor_path, ligand_path
            prepare_example_dataset(output_dir)
            return output_dir / row["receptor_file"], output_dir / row["ligand_file"]

    if examples:
        first = examples[0]
        return output_dir / first["receptor_file"], output_dir / first["ligand_file"]

    prepare_example_dataset(output_dir)
    return output_dir / "receptor.pdb", output_dir / "ligand.pdb"
