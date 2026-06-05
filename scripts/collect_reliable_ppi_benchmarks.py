#!/usr/bin/env python3
"""Collect reliable experimental protein complex benchmark manifests.

The first supported source is the official Protein-Protein Docking Benchmark 5.5
table hosted by the Weng lab. The script writes a reproducible manifest and can
optionally download bound PDB files from RCSB and split the listed receptor and
ligand chains into pair files.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DB55_URL = "https://zlab.wenglab.org/benchmark/benchmark5.5.html"
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


@dataclass(frozen=True)
class BenchmarkCase:
    source: str
    complex_id: str
    pdb_id: str
    receptor_chains: str
    ligand_chains: str
    category: str = ""
    source_url: str = DB55_URL
    bound_pdb_path: str = ""
    receptor_path: str = ""
    ligand_path: str = ""


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_db55_cases(page_text: str) -> List[BenchmarkCase]:
    """Parse DB5.5 bound complex identifiers such as ``1AHW_AB:C``.

    The official page is not guaranteed to remain a strict HTML table, so this
    parser works on normalized visible text and extracts only identifiers that
    include the receptor:ligand chain split.
    """
    visible_text = re.sub(r"<[^>]+>", " ", page_text)
    visible_text = html.unescape(re.sub(r"\s+", " ", visible_text))
    pattern = re.compile(
        r"\b(?P<pdb>[0-9][A-Za-z0-9]{3})_"
        r"(?P<receptor>[A-Za-z0-9]+):(?P<ligand>[A-Za-z0-9]+)"
        r"(?:\s+\*)?\s+(?P<category>AA|AS|EI|ER|ES|OR|OX|MI|DI|RB|MD|D)?\b"
    )
    cases: List[BenchmarkCase] = []
    seen = set()
    for match in pattern.finditer(visible_text):
        pdb_id = match.group("pdb").upper()
        receptor = match.group("receptor")
        ligand = match.group("ligand")
        complex_id = f"{pdb_id}_{receptor}:{ligand}"
        if complex_id in seen:
            continue
        seen.add(complex_id)
        cases.append(
            BenchmarkCase(
                source="db5.5",
                complex_id=complex_id,
                pdb_id=pdb_id,
                receptor_chains=receptor,
                ligand_chains=ligand,
                category=match.group("category") or "",
            )
        )
    return cases


def write_manifest(cases: Sequence[BenchmarkCase], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(case) for case in cases]
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": "Protein-Protein Docking Benchmark 5.5",
                "source_url": DB55_URL,
                "case_count": len(rows),
                "cases": rows,
            },
            handle,
            indent=2,
        )
    with open(output_dir / "manifest.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def download_pdb(pdb_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{pdb_id.upper()}.pdb"
    if path.exists() and path.stat().st_size > 1000:
        return path
    url = RCSB_PDB_URL.format(pdb_id=pdb_id.upper())
    urllib.request.urlretrieve(url, path)
    return path


def split_bound_complex(
    pdb_path: Path,
    receptor_chains: str,
    ligand_chains: str,
    output_dir: Path,
    complex_id: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_set = set(receptor_chains)
    ligand_set = set(ligand_chains)
    receptor_lines: List[str] = []
    ligand_lines: List[str] = []
    with open(pdb_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 22:
                continue
            chain_id = line[21].strip()
            if chain_id in receptor_set:
                receptor_lines.append(line)
            elif chain_id in ligand_set:
                ligand_lines.append(line)
    if not receptor_lines or not ligand_lines:
        raise ValueError(
            f"Could not split {pdb_path.name} with chains "
            f"{receptor_chains}:{ligand_chains}"
        )
    safe_id = complex_id.replace(":", "_")
    receptor_path = output_dir / f"{safe_id}_receptor.pdb"
    ligand_path = output_dir / f"{safe_id}_ligand.pdb"
    receptor_path.write_text("".join(receptor_lines) + "END\n", encoding="utf-8")
    ligand_path.write_text("".join(ligand_lines) + "END\n", encoding="utf-8")
    return receptor_path, ligand_path


def collect_db55(args: argparse.Namespace) -> None:
    text = fetch_text(args.url)
    cases = parse_db55_cases(text)
    if args.limit:
        cases = cases[: args.limit]

    output_dir = Path(args.out)
    enriched: List[BenchmarkCase] = []
    for case in cases:
        bound_path = ""
        receptor_path = ""
        ligand_path = ""
        if args.download_pdb:
            pdb_path = download_pdb(case.pdb_id, output_dir / "pdb")
            bound_path = str(pdb_path)
            if args.split_bound:
                rec_path, lig_path = split_bound_complex(
                    pdb_path,
                    case.receptor_chains,
                    case.ligand_chains,
                    output_dir / "pairs",
                    case.complex_id,
                )
                receptor_path = str(rec_path)
                ligand_path = str(lig_path)
        enriched.append(
            BenchmarkCase(
                **{
                    **asdict(case),
                    "bound_pdb_path": bound_path,
                    "receptor_path": receptor_path,
                    "ligand_path": ligand_path,
                }
            )
        )

    write_manifest(enriched, output_dir)
    print(f"Wrote {len(enriched)} DB5.5 cases to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    db55 = subparsers.add_parser("db55", help="Collect the Weng lab DB5.5 benchmark table")
    db55.add_argument("--url", default=DB55_URL)
    db55.add_argument("--out", default="data/reliable_ppi/db55")
    db55.add_argument("--limit", type=int, default=0, help="Optional case limit for smoke tests")
    db55.add_argument("--download-pdb", action="store_true", help="Download bound PDB files from RCSB")
    db55.add_argument("--split-bound", action="store_true", help="Split bound PDB into receptor/ligand chains")
    db55.set_defaults(func=collect_db55)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
