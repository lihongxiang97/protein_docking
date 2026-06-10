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
import hashlib
import html
import json
import re
import sys
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DB55_URL = "https://zlab.wenglab.org/benchmark/benchmark5.5.html"
DB55_ARCHIVE_URL = "https://zlab.wenglab.org/benchmark/benchmark5.5.tgz"
DB55_TABLE_URL = "https://zlab.wenglab.org/benchmark/Table_BM5.5.xlsx"
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


def write_manifest(
    cases: Sequence[BenchmarkCase],
    output_dir: Path,
    metadata: Optional[dict] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(case) for case in cases]
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": "Protein-Protein Docking Benchmark 5.5",
                "source_url": DB55_URL,
                "case_count": len(rows),
                **(metadata or {}),
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


def download_file(url: str, output_path: Path, force: bool = False) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 1000 and not force:
        return output_path
    urllib.request.urlretrieve(url, output_path)
    return output_path


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            target = (output_dir / member.name).resolve()
            if output_root not in target.parents and target != output_root:
                raise ValueError(f"Unsafe tar member path: {member.name}")
        archive.extractall(output_dir, members=members)


def pdb_chain_order(path: Path) -> List[str]:
    chains: List[str] = []
    seen = set()
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 22:
                continue
            chain_id = line[21]
            if chain_id not in seen:
                seen.add(chain_id)
                chains.append(chain_id)
    return chains


def write_chain_mapped_pdb(source_path: Path, target_chains: str, output_path: Path) -> Path:
    """Write a copy whose chain IDs match the DB5.5 native complex chain split."""
    source_chains = pdb_chain_order(source_path)
    target_chain_list = list(target_chains)
    if {chain.strip() for chain in source_chains} == set(target_chain_list):
        return source_path
    if not source_chains or len(source_chains) != len(target_chain_list):
        return source_path
    chain_map = dict(zip(source_chains, target_chain_list))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    with open(source_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM", "ANISOU", "TER")) and len(line) > 21:
                chain_id = line[21]
                if chain_id in chain_map:
                    line = f"{line[:21]}{chain_map[chain_id]}{line[22:]}"
            lines.append(line)
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path


def archive_case_paths(
    extracted_dir: Path,
    output_dir: Path,
    case: BenchmarkCase,
) -> tuple[str, str, str]:
    """Return unbound receptor/ligand and merged bound-native paths from DB5.5 archive."""
    structures_dir = Path(extracted_dir) / "benchmark5.5" / "structures"
    pdb_id = case.pdb_id.upper()
    receptor_unbound = structures_dir / f"{pdb_id}_r_u.pdb"
    ligand_unbound = structures_dir / f"{pdb_id}_l_u.pdb"
    receptor_bound = structures_dir / f"{pdb_id}_r_b.pdb"
    ligand_bound = structures_dir / f"{pdb_id}_l_b.pdb"
    required = [receptor_unbound, ligand_unbound, receptor_bound, ligand_bound]
    if not all(path.exists() for path in required):
        return "", "", ""

    mapped_dir = output_dir / "unbound_mapped"
    safe_case_id = case.complex_id.replace(":", "_")
    mapped_receptor = write_chain_mapped_pdb(
        receptor_unbound,
        case.receptor_chains,
        mapped_dir / f"{safe_case_id}_r_u_mapped.pdb",
    )
    mapped_ligand = write_chain_mapped_pdb(
        ligand_unbound,
        case.ligand_chains,
        mapped_dir / f"{safe_case_id}_l_u_mapped.pdb",
    )

    native_dir = output_dir / "native_bound"
    native_dir.mkdir(parents=True, exist_ok=True)
    native_path = native_dir / f"{pdb_id}_native_bound.pdb"
    if not native_path.exists():
        lines: List[str] = []
        for source_path in [receptor_bound, ligand_bound]:
            with open(source_path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith(("ATOM", "HETATM", "TER")):
                        lines.append(line)
        lines.append("END\n")
        native_path.write_text("".join(lines), encoding="utf-8")
    return str(mapped_receptor), str(mapped_ligand), str(native_path)


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
    if args.input_manifest:
        cases = read_manifest_cases(Path(args.input_manifest))
    else:
        text = fetch_text(args.url)
        cases = parse_db55_cases(text)
    if args.limit:
        cases = cases[: args.limit]

    output_dir = Path(args.out)
    metadata = {
        "archive_url": args.archive_url,
        "table_url": DB55_TABLE_URL,
        "archive_path": "",
        "archive_md5": "",
        "table_path": "",
        "extracted_dir": "",
    }
    if args.download_table:
        table_path = download_file(DB55_TABLE_URL, output_dir / "Table_BM5.5.xlsx", force=args.force)
        metadata["table_path"] = str(table_path)
    if args.download_archive:
        archive_path = download_file(
            args.archive_url,
            output_dir / "benchmark5.5.tgz",
            force=args.force,
        )
        checksum = md5_file(archive_path)
        if args.archive_md5 and checksum.lower() != args.archive_md5.lower():
            raise ValueError(
                f"DB5.5 archive checksum mismatch: expected {args.archive_md5}, got {checksum}"
            )
        metadata["archive_path"] = str(archive_path)
        metadata["archive_md5"] = checksum
        if args.extract_archive:
            extracted_dir = output_dir / "archive"
            safe_extract_tar(archive_path, extracted_dir)
            metadata["extracted_dir"] = str(extracted_dir)

    enriched: List[BenchmarkCase] = []
    for case in cases:
        bound_path = ""
        receptor_path = ""
        ligand_path = ""
        if metadata["extracted_dir"]:
            archive_rec, archive_lig, archive_native = archive_case_paths(
                Path(metadata["extracted_dir"]),
                output_dir,
                case,
            )
            receptor_path = archive_rec
            ligand_path = archive_lig
            bound_path = archive_native
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

    write_manifest(enriched, output_dir, metadata=metadata)
    print(f"Wrote {len(enriched)} DB5.5 cases to {output_dir}")


def read_manifest_cases(manifest_path: Path) -> List[BenchmarkCase]:
    with open(manifest_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_cases = payload.get("cases", payload if isinstance(payload, list) else [])
    cases: List[BenchmarkCase] = []
    for row in raw_cases:
        case = BenchmarkCase(
            source=str(row.get("source", "db5.5")),
            complex_id=str(row["complex_id"]),
            pdb_id=str(row.get("pdb_id") or str(row["complex_id"])[:4]),
            receptor_chains=str(row.get("receptor_chains", "")),
            ligand_chains=str(row.get("ligand_chains", "")),
            category=str(row.get("category", "")),
            source_url=str(row.get("source_url", DB55_URL)),
        )
        cases.append(case)
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    db55 = subparsers.add_parser("db55", help="Collect the Weng lab DB5.5 benchmark table")
    db55.add_argument("--url", default=DB55_URL)
    db55.add_argument("--input-manifest", help="Reuse an existing manifest instead of fetching DB5.5 HTML")
    db55.add_argument("--out", default="data/reliable_ppi/db55")
    db55.add_argument("--limit", type=int, default=0, help="Optional case limit for smoke tests")
    db55.add_argument("--download-archive", action="store_true", help="Download official benchmark5.5.tgz")
    db55.add_argument("--extract-archive", action="store_true", help="Safely extract the official archive")
    db55.add_argument("--archive-url", default=DB55_ARCHIVE_URL)
    db55.add_argument("--archive-md5", default="", help="Optional expected MD5 checksum")
    db55.add_argument("--download-table", action="store_true", help="Download official Table_BM5.5.xlsx")
    db55.add_argument("--force", action="store_true", help="Re-download existing files")
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
