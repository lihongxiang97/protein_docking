"""
PPI Docking Streamlit Web 界面
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from docking.docking import ProteinDocker
from docking.interface import InterfaceAnalyzer
from docking.ppi_predictor import PPIPredictor
from docking.preprocess import StructurePreprocessor
from docking.visualization import ResultVisualizer
from tests.benchmark_data import generate_example_pair

st.set_page_config(
    page_title="PPI Docking",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 PPI Docking - 蛋白质互作预测")
st.markdown("自主实现的蛋白-蛋白分子对接与互作位点预测系统")

tab_dock, tab_bench, tab_about = st.tabs(["分子对接", "Benchmark", "关于"])

with tab_dock:
    col1, col2 = st.columns(2)
    with col1:
        receptor_file = st.file_uploader("受体 PDB 文件", type=["pdb"], key="receptor")
    with col2:
        ligand_file = st.file_uploader("配体 PDB 文件", type=["pdb"], key="ligand")

    use_example = st.checkbox("使用内置示例结构")
    top_n = st.slider("输出 Top N 构象", 1, 10, 5)
    run_btn = st.button("开始对接分析", type="primary")

    if run_btn:
        output_dir = PROJECT_ROOT / "results" / "web_run"
        output_dir.mkdir(parents=True, exist_ok=True)

        if use_example:
            rec_path, lig_path = generate_example_pair(PROJECT_ROOT / "data" / "example_pdb")
            st.info(f"使用示例结构: {rec_path.name}, {lig_path.name}")
        elif receptor_file and ligand_file:
            rec_path = output_dir / "receptor.pdb"
            lig_path = output_dir / "ligand.pdb"
            rec_path.write_bytes(receptor_file.read())
            lig_path.write_bytes(ligand_file.read())
        else:
            st.error("请上传 PDB 文件或勾选使用示例结构")
            st.stop()

        config_path = PROJECT_ROOT / "config.yaml"
        with st.spinner("正在执行分子对接..."):
            docker = ProteinDocker(config_path)
            poses, receptor, ligand = docker.dock(rec_path, lig_path, output_dir)

        if not poses:
            st.error("对接失败，未找到有效构象")
            st.stop()

        best = poses[0]
        interface_analyzer = InterfaceAnalyzer()
        interface = interface_analyzer.analyze(receptor, ligand)
        predictor = PPIPredictor(config_path)
        ppi = predictor.predict(best.scores, interface, receptor, ligand)

        st.success("分析完成！")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("对接评分", f"{best.scores.total:.1f}")
        m2.metric("互作概率", f"{ppi.probability:.1%}")
        m3.metric("界面残基", interface.n_interface_residues)
        m4.metric("界面面积", f"{best.scores.raw_interface_area:.0f} Å²")

        st.subheader("PPI 预测")
        if ppi.interacts:
            st.success(f"✅ 预测存在互作 — {ppi.explanation}")
        else:
            st.warning(f"⚠️ 预测互作可能性较低 — {ppi.explanation}")

        st.subheader("对接评分排名")
        score_df = pd.DataFrame([
            {"Rank": p.rank, "Total": p.scores.total,
             "Hydrophobic": p.scores.hydrophobic,
             "Electrostatic": p.scores.electrostatic,
             "Contacts": p.scores.contact_residues,
             "Clash": p.scores.clash_penalty}
            for p in poses[:top_n]
        ])
        st.dataframe(score_df, use_container_width=True)

        st.subheader("界面残基")
        iface_df = interface_analyzer.to_dataframe(interface)
        st.dataframe(iface_df, use_container_width=True)

        # 可视化
        viz = ResultVisualizer(output_dir / "plots")
        viz.generate_all_plots(poses[:top_n], interface, best)
        plot_files = list((output_dir / "plots").glob("*.png"))
        if plot_files:
            cols = st.columns(min(3, len(plot_files)))
            for i, pf in enumerate(plot_files[:6]):
                cols[i % 3].image(str(pf), caption=pf.name)

        pdb_out = output_dir / f"docked_complex_{best.rank}.pdb"
        if pdb_out.exists():
            st.download_button(
                "下载最佳对接结构",
                pdb_out.read_text(),
                file_name="docked_complex.pdb",
            )

with tab_bench:
    st.markdown("### Benchmark 测试")
    st.markdown("自动测试已知互作/非互作蛋白对，生成评估报告。")
    if st.button("运行 Benchmark"):
        from tests.benchmark_test import BenchmarkRunner
        from tests.evaluation import EvaluationReport

        with st.spinner("运行 benchmark (可能需要几分钟)..."):
            runner = BenchmarkRunner(PROJECT_ROOT / "config.yaml", PROJECT_ROOT)
            results = runner.run_all(download=True)
            evaluator = EvaluationReport(PROJECT_ROOT / "results" / "benchmark" / "evaluation")
            report = evaluator.generate_full_report(results)

        st.success(f"完成 {len(results)} 个测试")
        if results:
            df = pd.read_csv(PROJECT_ROOT / "results" / "benchmark" / "benchmark_results.csv")
            st.dataframe(df)
        report_path = PROJECT_ROOT / "results" / "benchmark" / "evaluation" / "evaluation_report.md"
        if report_path.exists():
            st.markdown(report_path.read_text())

with tab_about:
    st.markdown("""
    ## 算法原理

    ### 评分函数
    $$Score = w_H \\cdot H + w_E \\cdot E + w_C \\cdot C + w_A \\cdot A - w_P \\cdot P$$

    ### 对接流程
    1. 结构预处理 → 2. 表面残基识别 → 3. 网格搜索 → 4. Monte Carlo 精修 → 5. 评分排序

    ### 复杂度
    - SASA: O(N log N)
    - 对接: O(R × T × N + M × N)

    详见项目 README.md
    """)
