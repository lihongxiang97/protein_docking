#!/usr/bin/env python3
"""生成示例 PDB 文件。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.benchmark_data import generate_example_pair

if __name__ == "__main__":
    out = Path(__file__).parent.parent / "data" / "example_pdb"
    rec, lig = generate_example_pair(out)
    print(f"Generated: {rec}, {lig}")
