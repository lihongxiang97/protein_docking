#!/usr/bin/env python3
"""
PPI Docking - 蛋白质-蛋白质相互作用预测与分子对接软件

用法:
    python main.py --receptor receptor.pdb --ligand ligand.pdb --output results/
    python main.py --benchmark
    python main.py --web
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from docking.docking import ProteinDocker
from docking.config import load_config as load_validated_config
from docking.interface import InterfaceAnalyzer
from docking.ppi_predictor import PPIPredictor
from docking.preprocess import StructurePreprocessor
from docking.structure import split_complex_by_reference_chains

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ppi_docking")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_config(config_path: Path) -> dict:
    return load_validated_config(config_path)


def run_docking(args: argparse.Namespace) -> int:
    """执行对接分析。"""
    config_path = Path(args.config) if args.config else PROJECT_ROOT / "config.yaml"
    try:
        config = load_config(config_path)
    except Exception as exc:
        logger.error("Failed to load config %s: %s", config_path, exc)
        return 1
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    receptor_path = Path(args.receptor)
    ligand_path = Path(args.ligand)

    if not receptor_path.exists():
        logger.error("受体 PDB 不存在: %s", receptor_path)
        return 1
    if not ligand_path.exists():
        logger.error("配体 PDB 不存在: %s", ligand_path)
        return 1

    # 结构验证
    preprocessor = StructurePreprocessor(config_path)
    rec_report = preprocessor.validate_structure(receptor_path)
    lig_report = preprocessor.validate_structure(ligand_path)

    logger.info("受体: %d 原子, 链 %s, 类型 %s",
                rec_report.n_atoms, rec_report.chains, rec_report.protein_type)
    logger.info("配体: %d 原子, 链 %s, 类型 %s",
                lig_report.n_atoms, lig_report.chains, lig_report.protein_type)

    # 对接
    try:
        docker = ProteinDocker(config_path)
        docker.top_n = args.top_n
        if args.blind:
            docker.include_input_pose = False
            docker.input_pose_bonus = 0.0
        if args.rotations is not None:
            docker.coarse_rotations = args.rotations
        if args.mc_iterations is not None:
            docker.mc_iterations = args.mc_iterations
        if args.receptor_active or args.ligand_active:
            if not args.receptor_active or not args.ligand_active:
                raise ValueError(
                    "--receptor-active and --ligand-active must be provided together"
                )
            docker.scorer.restraints = {
                "enabled": True,
                "receptor_active": _split_csv(args.receptor_active),
                "ligand_active": _split_csv(args.ligand_active),
                "target_distance": args.restraint_target,
                "upper_distance": args.restraint_upper,
            }
        poses, receptor, ligand = docker.dock(
            receptor_path, ligand_path, output_dir,
            receptor_chains=[c.strip() for c in args.receptor_chains.split(",") if c.strip()] if args.receptor_chains else None,
            ligand_chains=[c.strip() for c in args.ligand_chains.split(",") if c.strip()] if args.ligand_chains else None,
        )
    except Exception as exc:
        logger.exception("Docking failed: %s", exc)
        return 1

    if not poses:
        logger.error("未找到有效对接构象")
        return 1

    best_pose = poses[0]
    logger.info("最佳构象: Rank=%d, Score=%.2f", best_pose.rank, best_pose.scores.total)

    # 界面分析
    interface_analyzer = InterfaceAnalyzer()
    ligand_docked = best_pose.complex_structure
    if ligand_docked:
        _, lig_struct = split_complex_by_reference_chains(ligand_docked, receptor.chains)
    else:
        lig_struct = ligand

    interface = interface_analyzer.analyze(receptor, lig_struct)

    # PPI 预测
    predictor = PPIPredictor(config_path)
    ppi_result = predictor.predict(
        best_pose.scores, interface, receptor, lig_struct
    )

    logger.info("PPI 预测: %s", ppi_result.explanation)

    # 保存结果
    _save_results(output_dir, poses, interface, ppi_result, interface_analyzer)

    # 可视化
    if not args.no_plots:
        from docking.visualization import ResultVisualizer

        visualizer = ResultVisualizer(output_dir / "plots")
        visualizer.generate_all_plots(poses, interface, best_pose)
        if best_pose.complex_structure:
            iface_labels = [
                f"{r.resname}{r.resseq}" for r in interface.receptor_interface[:20]
            ]
            pdb_file = output_dir / f"docked_complex_{best_pose.rank}.pdb"
            if pdb_file.exists():
                visualizer.generate_3d_view(pdb_file, iface_labels)

    # 生成报告
    _generate_summary_report(output_dir, poses, interface, ppi_result, rec_report, lig_report)

    logger.info("分析完成，结果保存在: %s", output_dir)
    return 0


def _save_results(output_dir, poses, interface, ppi_result, interface_analyzer):
    """保存 CSV 和文本结果。"""
    # 对接评分
    score_rows = []
    for pose in poses:
        row = {
            "rank": pose.rank,
            "search_score": pose.search_score,
            "cluster_size": pose.cluster_size,
            "provenance": pose.provenance,
            **pose.scores.to_dict(),
        }
        score_rows.append(row)
    pd.DataFrame(score_rows).to_csv(output_dir / "docking_scores.csv", index=False)

    # 界面残基
    df = interface_analyzer.to_dataframe(interface)
    df.to_csv(output_dir / "interface_residues.csv", index=False)
    interface_analyzer.save_interface_report(interface, output_dir / "interface_residues.txt")

    # 互作摘要
    with open(output_dir / "interaction_summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("PPI Docking - Interaction Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"PPI Prediction: {'YES' if ppi_result.interacts else 'NO'}\n")
        f.write(f"Probability: {ppi_result.probability:.4f}\n")
        f.write(f"Confidence: {ppi_result.confidence:.4f}\n")
        f.write(f"\n{ppi_result.explanation}\n\n")
        f.write("Top Docking Scores:\n")
        for pose in poses[:5]:
            f.write(f"  Rank {pose.rank}: {pose.scores.total:.2f}\n")
        f.write(f"\nInterface Residues: {interface.n_interface_residues}\n")
        f.write("\nFeatures:\n")
        for k, v in ppi_result.features.items():
            f.write(f"  {k}: {v:.4f}\n")


def _generate_summary_report(output_dir, poses, interface, ppi_result, rec_report, lig_report):
    """生成 Markdown 报告。"""
    report_path = output_dir / "docking_report.md"
    best = poses[0] if poses else None
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# PPI Docking Analysis Report\n\n")
        f.write("## 1. Input Structures\n\n")
        f.write(f"- **Receptor**: {rec_report.pdb_path} ({rec_report.n_residues} residues)\n")
        f.write(f"- **Ligand**: {lig_report.pdb_path} ({lig_report.n_residues} residues)\n\n")
        f.write("## 2. Docking Results\n\n")
        if best:
            f.write(f"- **Best Score**: {best.scores.total:.2f}\n")
            f.write(f"- **Interface Area**: {best.scores.raw_interface_area:.1f} Å²\n")
            f.write(f"- **Contact Residues**: {best.scores.contact_residues}\n")
            f.write(f"- **H-bonds**: {best.scores.hbonds}\n\n")
        f.write("## 3. PPI Prediction\n\n")
        f.write(f"- **Interacts**: {'Yes' if ppi_result.interacts else 'No'}\n")
        f.write(f"- **Probability**: {ppi_result.probability:.2%}\n")
        f.write(f"- {ppi_result.explanation}\n\n")
        f.write("## 4. Interface Residues\n\n")
        f.write(f"Total interface residues: {interface.n_interface_residues}\n\n")
        f.write("## 5. Output Files\n\n")
        f.write("| File | Description |\n|------|-------------|\n")
        f.write("| docking_scores.csv | All pose scores |\n")
        f.write("| interface_residues.csv | Interface residue list |\n")
        f.write("| docked_complex_*.pdb | Docked structures |\n")
        f.write("| plots/ | Visualization figures |\n\n")
        f.write("## 6. Limitations\n\n")
        f.write("- Simplified rigid-body docking without side-chain flexibility\n")
        f.write("- SASA computed via neighbor-density approximation\n")
        f.write("- Rule-based PPI model when no training data available\n\n")
        f.write("## 7. Future Improvements\n\n")
        f.write("- Flexible backbone/side-chain refinement\n")
        f.write("- FFT-based coarse docking\n")
        f.write("- GNN-based interface prediction\n")
        f.write("- GPU-accelerated spatial search\n")


def run_benchmark(args: argparse.Namespace) -> int:
    """运行 benchmark 测试。"""
    from tests.benchmark_test import BenchmarkRunner
    from tests.evaluation import EvaluationReport

    config_path = Path(args.config) if args.config else PROJECT_ROOT / "config.yaml"
    output_dir = Path(args.output) / "benchmark"
    runner = BenchmarkRunner(config_path, PROJECT_ROOT)
    results = runner.run_all(download=not args.no_download)

    evaluator = EvaluationReport(output_dir / "evaluation")
    report_path = evaluator.generate_full_report(results)
    logger.info("Benchmark 完成，报告: %s", report_path)
    return 0


def run_web(args: argparse.Namespace) -> int:
    """启动 Streamlit Web 界面。"""
    import subprocess
    app_path = PROJECT_ROOT / "web" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(args.port)]
    logger.info("启动 Web 界面: http://localhost:%s", args.port)
    subprocess.run(cmd)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="PPI Docking - 蛋白质-蛋白质相互作用预测软件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--receptor", "-r", type=str, help="受体 PDB 文件路径")
    parser.add_argument("--ligand", "-l", type=str, help="配体 PDB 文件路径")
    parser.add_argument("--output", "-o", type=str, default="results", help="输出目录")
    parser.add_argument("--config", "-c", type=str, help="配置文件路径")
    parser.add_argument("--receptor-chains", type=str, help="受体链 ID (逗号分隔)")
    parser.add_argument("--ligand-chains", type=str, help="配体链 ID (逗号分隔)")
    parser.add_argument("--top-n", type=int, default=10, help="输出 Top N 构象")
    parser.add_argument("--blind", action="store_true", help="禁用输入相对构象先验，执行纯盲搜")
    parser.add_argument("--rotations", type=int, help="覆盖全局旋转采样数量")
    parser.add_argument("--mc-iterations", type=int, help="覆盖局部精修迭代数")
    parser.add_argument("--receptor-active", type=str, help="受体活性残基，如 A:12,A:45")
    parser.add_argument("--ligand-active", type=str, help="配体活性残基，如 B:8,B:19")
    parser.add_argument("--restraint-target", type=float, default=6.0)
    parser.add_argument("--restraint-upper", type=float, default=10.0)
    parser.add_argument("--no-plots", action="store_true", help="跳过图表生成")
    parser.add_argument("--benchmark", action="store_true", help="运行 benchmark 测试")
    parser.add_argument("--no-download", action="store_true", help="不下载 benchmark 数据")
    parser.add_argument("--web", action="store_true", help="启动 Web 界面")
    parser.add_argument("--port", type=int, default=8501, help="Web 端口")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.benchmark:
        return run_benchmark(args)
    if args.web:
        return run_web(args)
    if not args.receptor or not args.ligand:
        parser.print_help()
        print("\n错误: 请提供 --receptor 和 --ligand，或使用 --benchmark / --web")
        return 1

    return run_docking(args)


if __name__ == "__main__":
    sys.exit(main())
