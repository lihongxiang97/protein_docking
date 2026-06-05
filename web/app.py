"""
PPI Docking Streamlit Web 界面
支持 ECharts 图表和 NGL 对接结构可视化
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from docking.docking import ProteinDocker
from docking.interface import InterfaceAnalyzer
from docking.ppi_predictor import PPIPredictor
from docking.structure import split_complex_by_reference_chains
from docking.visualization import ResultVisualizer
from tests.benchmark_data import generate_example_pair, list_example_pairs
from web.structure_viewer import build_viewer_payload, render_structure_viewer

st.set_page_config(
    page_title="PPI Docking",
    page_icon="🧬",
    layout="wide",
)

# ECharts CDN URL
ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"

TEXT = {
    "zh": {
        "language_help": "切换为英文",
        "app_title": "🧬 PPI Docking - 蛋白质互作预测",
        "app_subtitle": "自主实现的蛋白-蛋白分子对接与互作位点预测系统",
        "tab_dock": "分子对接",
        "tab_bench": "Benchmark",
        "tab_about": "关于",
        "receptor_file": "受体 PDB 文件",
        "ligand_file": "配体 PDB 文件",
        "use_example": "使用内置真实示例结构",
        "example_complex": "示例复合物",
        "example_source": "来源: RCSB PDB {pdb_id} · {description} · {source_url}",
        "example_description": "RCSB PDB 已知互作复合物。",
        "top_n": "输出 Top N 构象",
        "start_docking": "开始对接分析",
        "using_example": "使用示例结构: {display_name} ({receptor}, {ligand})",
        "input_error": "请上传 PDB 文件或勾选使用示例结构",
        "docking_spinner": "正在执行分子对接...",
        "docking_failed": "对接失败，未找到有效构象",
        "analysis_complete": "分析完成！",
        "docking_score": "对接评分",
        "interaction_probability": "互作概率",
        "interface_residues": "界面残基",
        "interface_area": "界面面积",
        "ppi_prediction": "PPI 预测",
        "ppi_positive": "预测存在互作",
        "ppi_negative": "预测互作可能性较低",
        "ppi_detail": "概率={probability:.2%}；界面面积={area:.1f} Å²；接触残基={contacts:.0f}；对接评分={score:.1f}；氢键={hbonds:.0f}。",
        "score_ranking": "对接评分排名",
        "interface_residue_table": "界面残基",
        "score_components": "评分组成分析",
        "interface_distribution": "界面残基类型分布",
        "distance_distribution": "接触距离分布",
        "contact_map": "残基接触热图",
        "structure_viewer": "3D 对接结构可视化",
        "select_pose": "选择对接构象",
        "pose_option": "Rank {rank}（评分: {score:.1f}）",
        "pose_title": "Rank {rank} 对接复合物",
        "pose_score": "构象评分",
        "viewer_caption": "受体以蓝色显示，配体以橙色显示；界面 sticks、接触连线和辅因子显示可在视图内部切换。",
        "selected_interface": "选中构象的界面残基",
        "download_pdb": "⬇️ 下载对接结构 (PDB)",
        "download_pdb_help": "下载 PDB 格式的对接复合物结构",
        "download_html": "⬇️ 下载 3D 视图 (HTML)",
        "download_html_help": "下载当前 NGL 交互视图 HTML 文件",
        "benchmark_title": "### Benchmark 测试",
        "benchmark_description": "自动测试已知互作/非互作蛋白对，生成评估报告。",
        "run_benchmark": "运行 Benchmark",
        "benchmark_spinner": "运行 benchmark（可能需要几分钟）...",
        "benchmark_complete": "完成 {count} 个测试",
        "about": """
## 算法原理

### 评分函数
$$Score = w_H \\cdot H + w_E \\cdot E + w_C \\cdot C + w_A \\cdot A - w_P \\cdot P$$

### 对接流程
1. 结构预处理 → 2. 表面残基识别 → 3. FFT 全局搜索 → 4. Monte Carlo 精修 → 5. 聚类与评分排序

### 复杂度
- SASA: O(N log N)
- 对接搜索: O(R × G log G)

详见项目 README.md
""",
        "col_rank": "排名",
        "col_total": "总分",
        "col_hydrophobic": "疏水",
        "col_electrostatic": "静电",
        "col_contacts": "接触残基",
        "col_clash": "冲突惩罚",
        "col_role": "角色",
        "col_chain": "链",
        "col_residue": "残基",
        "col_contact_type": "接触类型",
        "col_min_distance": "最短距离",
        "col_partners": "互作残基",
    },
    "en": {
        "language_help": "Switch to Chinese",
        "app_title": "🧬 PPI Docking - Protein Interaction Prediction",
        "app_subtitle": "An independent protein-protein docking and interaction-site prediction system",
        "tab_dock": "Docking",
        "tab_bench": "Benchmark",
        "tab_about": "About",
        "receptor_file": "Receptor PDB File",
        "ligand_file": "Ligand PDB File",
        "use_example": "Use a Built-in Experimental Complex",
        "example_complex": "Example Complex",
        "example_source": "Source: RCSB PDB {pdb_id} · {description} · {source_url}",
        "example_description": "Known interacting complex from RCSB PDB.",
        "top_n": "Number of Top Poses",
        "start_docking": "Start Docking Analysis",
        "using_example": "Using example: {display_name} ({receptor}, {ligand})",
        "input_error": "Upload receptor and ligand PDB files or select the built-in example",
        "docking_spinner": "Running protein-protein docking...",
        "docking_failed": "Docking failed: no valid poses were found",
        "analysis_complete": "Analysis complete.",
        "docking_score": "Docking Score",
        "interaction_probability": "Interaction Probability",
        "interface_residues": "Interface Residues",
        "interface_area": "Interface Area",
        "ppi_prediction": "PPI Prediction",
        "ppi_positive": "Interaction predicted",
        "ppi_negative": "Low interaction probability",
        "ppi_detail": "Probability={probability:.2%}; interface area={area:.1f} Å²; contact residues={contacts:.0f}; docking score={score:.1f}; hydrogen bonds={hbonds:.0f}.",
        "score_ranking": "Docking Score Ranking",
        "interface_residue_table": "Interface Residues",
        "score_components": "Score Component Analysis",
        "interface_distribution": "Interface Residue Type Distribution",
        "distance_distribution": "Contact Distance Distribution",
        "contact_map": "Residue Contact Map",
        "structure_viewer": "3D Docked Structure",
        "select_pose": "Select Docking Pose",
        "pose_option": "Rank {rank} (Score: {score:.1f})",
        "pose_title": "Rank {rank} Docked Complex",
        "pose_score": "Pose Score",
        "viewer_caption": "The receptor is blue and the ligand is orange. Interface sticks, contact lines, and cofactors can be toggled inside the viewer.",
        "selected_interface": "Interface Residues of Selected Pose",
        "download_pdb": "⬇️ Download Docked Structure (PDB)",
        "download_pdb_help": "Download the docked complex in PDB format",
        "download_html": "⬇️ Download 3D View (HTML)",
        "download_html_help": "Download the current interactive NGL view",
        "benchmark_title": "### Benchmark",
        "benchmark_description": "Evaluate known interacting and non-interacting protein pairs and generate a report.",
        "run_benchmark": "Run Benchmark",
        "benchmark_spinner": "Running benchmark (this may take a few minutes)...",
        "benchmark_complete": "Completed {count} tests",
        "about": """
## Method

### Scoring Function
$$Score = w_H \\cdot H + w_E \\cdot E + w_C \\cdot C + w_A \\cdot A - w_P \\cdot P$$

### Docking Pipeline
1. Structure preprocessing → 2. Surface residue detection → 3. Global FFT search → 4. Monte Carlo refinement → 5. Clustering and ranking

### Complexity
- SASA: O(N log N)
- Docking search: O(R × G log G)

See the project README.md for details.
""",
        "col_rank": "Rank",
        "col_total": "Total",
        "col_hydrophobic": "Hydrophobic",
        "col_electrostatic": "Electrostatic",
        "col_contacts": "Contact Residues",
        "col_clash": "Clash Penalty",
        "col_role": "Role",
        "col_chain": "Chain",
        "col_residue": "Residue",
        "col_contact_type": "Contact Type",
        "col_min_distance": "Minimum Distance",
        "col_partners": "Partners",
    },
}

ECHARTS_ZH = {
    "Top Docking Poses by Score": "Top 对接构象评分",
    "Docking Score": "对接评分",
    "Rank": "排名",
    "Score": "评分",
    "Hydrophobic": "疏水",
    "Electrostatic": "静电",
    "Contacts": "接触",
    "Interface Area": "界面面积",
    "Clash Penalty": "冲突惩罚",
    "Weighted Contribution": "加权贡献",
    "Interface Contact Distance Distribution": "界面接触距离分布",
    "Distance Distribution": "距离分布",
    "Contact Cutoff": "接触阈值",
    "Distance (Å)": "距离 (Å)",
    "Count": "数量",
    "Interface Residue Type Distribution": "界面残基类型分布",
    "Receptor Interface": "受体界面",
    "Ligand Interface": "配体界面",
    "Residue Contact Map": "残基接触热图",
    "Ligand Residue Index": "配体残基索引",
    "Receptor Residue Index": "受体残基索引",
    "Contact Strength": "接触强度",
    "Receptor:{b}<br/>Ligand:{c}<br/>Strength:{d}": "受体:{b}<br/>配体:{c}<br/>强度:{d}",
    "hbond": "氢键",
    "hydrophobic": "疏水",
    "electrostatic": "静电",
    "van_der_waals": "范德华",
    "contact": "接触",
}


def tr(key, **kwargs):
    """Return a localized UI string."""
    language = st.session_state.get("language", "zh")
    template = TEXT.get(language, TEXT["zh"]).get(key, TEXT["zh"].get(key, key))
    return template.format(**kwargs) if kwargs else template


def localize_interface_dataframe(dataframe):
    """Localize interface table headings while keeping scientific values intact."""
    return dataframe.rename(
        columns={
            "role": tr("col_role"),
            "chain": tr("col_chain"),
            "residue": tr("col_residue"),
            "contact_type": tr("col_contact_type"),
            "min_distance": tr("col_min_distance"),
            "partners": tr("col_partners"),
        }
    )


def localized_ppi_detail(ppi):
    features = ppi.features
    return tr(
        "ppi_detail",
        probability=ppi.probability,
        area=features.get("interface_area", 0.0),
        contacts=features.get("contact_residues", 0.0),
        score=features.get("docking_score", 0.0),
        hbonds=features.get("hbond_count", 0.0),
    )


def localize_echarts_option(value):
    """Translate ECharts display strings without changing chart structure."""
    if st.session_state.get("language", "zh") != "zh":
        return value
    if isinstance(value, dict):
        return {key: localize_echarts_option(item) for key, item in value.items()}
    if isinstance(value, list):
        return [localize_echarts_option(item) for item in value]
    if isinstance(value, str):
        if value.startswith("Score Components (Total = "):
            return value.replace("Score Components (Total = ", "评分组成（总分 = ", 1).replace(")", "）", 1)
        return ECHARTS_ZH.get(value, value)
    return value


def render_echarts(option_json, width="100%", height="500px"):
    """渲染 ECharts 图表"""
    try:
        option_json = json.dumps(
            localize_echarts_option(json.loads(option_json)),
            ensure_ascii=False,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="%s"></script>
        <style>
            body { margin: 0; padding: 0; overflow: hidden; }
            #chart { width: %s; height: %s; }
        </style>
    </head>
    <body>
        <div id="chart"></div>
        <script type="text/javascript">
            var chart = echarts.init(document.getElementById('chart'));
            var option = %s;
            chart.setOption(option);
            window.addEventListener('resize', function() {
                chart.resize();
            });
        </script>
    </body>
    </html>
    """ % (ECHARTS_CDN, width, height, option_json)
    return html(html_code, scrolling=False, height=int(height.replace('px','')))


def get_docked_ligand_structure(receptor, ligand, pose):
    """从构象中恢复配体结构。"""
    if pose.complex_structure:
        _, docked_ligand = split_complex_by_reference_chains(pose.complex_structure, receptor.chains)
        if docked_ligand.atoms:
            return docked_ligand
    return ligand


def analyze_pose_details(poses, receptor, ligand, predictor):
    """为每个构象计算界面与 PPI 结果。"""
    details = {}
    interface_analyzer = InterfaceAnalyzer()
    for pose in poses:
        docked_ligand = get_docked_ligand_structure(receptor, ligand, pose)
        pose_interface = interface_analyzer.analyze(receptor, docked_ligand)
        pose_ppi = predictor.predict(pose.scores, pose_interface, receptor, docked_ligand)
        details[pose.rank] = {
            "interface": pose_interface,
            "ppi": pose_ppi,
            "ligand_chains": sorted(docked_ligand.chains),
        }
    return details


def ensure_state_defaults():
    defaults = {
        "analysis_done": False,
        "poses": [],
        "interface": None,
        "ppi": None,
        "best": None,
        "viz": None,
        "output_dir": None,
        "selected_pose_rank": 1,
        "receptor_structure": None,
        "ligand_structure": None,
        "pose_details": {},
        "language": "zh",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

ensure_state_defaults()

title_col, language_col = st.columns([10, 1])
with language_col:
    switch_label = "EN" if st.session_state.language == "zh" else "中文"
    if st.button(switch_label, key="language_toggle", help=tr("language_help"), use_container_width=True):
        st.session_state.language = "en" if st.session_state.language == "zh" else "zh"
        st.rerun()
with title_col:
    st.title(tr("app_title"))

st.markdown(tr("app_subtitle"))

tab_dock, tab_bench, tab_about = st.tabs([tr("tab_dock"), tr("tab_bench"), tr("tab_about")])

with tab_dock:
    col1, col2 = st.columns(2)
    with col1:
        receptor_file = st.file_uploader(tr("receptor_file"), type=["pdb"], key="receptor")
    with col2:
        ligand_file = st.file_uploader(tr("ligand_file"), type=["pdb"], key="ligand")

    use_example = st.checkbox(tr("use_example"))
    example_pairs = list_example_pairs(PROJECT_ROOT / "data" / "example_pdb")
    selected_example_id = None
    selected_example = None
    if use_example and example_pairs:
        example_options = {row["example_id"]: row for row in example_pairs}
        selected_example_id = st.selectbox(
            tr("example_complex"),
            options=list(example_options.keys()),
            format_func=lambda item: (
                f"{example_options[item]['display_name']} "
                f"({example_options[item]['pdb_id']} {example_options[item]['receptor_chain']}/{example_options[item]['ligand_chain']})"
            ),
        )
        selected_example = example_options[selected_example_id]
        st.caption(
            tr(
                "example_source",
                pdb_id=selected_example["pdb_id"],
                description=(
                    selected_example["description"]
                    if st.session_state.language == "zh"
                    else tr("example_description")
                ),
                source_url=selected_example["source_url"],
            )
        )

    top_n = st.slider(tr("top_n"), 1, 10, 5)
    run_btn = st.button(tr("start_docking"), type="primary")

    if run_btn:
        output_dir = PROJECT_ROOT / "results" / "web_run"
        output_dir.mkdir(parents=True, exist_ok=True)

        if use_example:
            rec_path, lig_path = generate_example_pair(
                PROJECT_ROOT / "data" / "example_pdb",
                example_id=selected_example_id,
            )
            if selected_example:
                st.info(
                    tr(
                        "using_example",
                        display_name=selected_example["display_name"],
                        receptor=rec_path.name,
                        ligand=lig_path.name,
                    )
                )
        elif receptor_file and ligand_file:
            rec_path = output_dir / "receptor.pdb"
            lig_path = output_dir / "ligand.pdb"
            rec_path.write_bytes(receptor_file.read())
            lig_path.write_bytes(ligand_file.read())
        else:
            st.error(tr("input_error"))
            st.stop()

        config_path = PROJECT_ROOT / "config.yaml"
        with st.spinner(tr("docking_spinner")):
            docker = ProteinDocker(config_path)
            docker.top_n = top_n
            poses, receptor, ligand = docker.dock(rec_path, lig_path, output_dir)

        if not poses:
            st.error(tr("docking_failed"))
            st.stop()

        best = poses[0]
        predictor = PPIPredictor(config_path)
        pose_details = analyze_pose_details(poses, receptor, ligand, predictor)
        interface = pose_details[best.rank]["interface"]
        ppi = pose_details[best.rank]["ppi"]

        # 保存到 session_state
        st.session_state.analysis_done = True
        st.session_state.poses = poses
        st.session_state.interface = interface
        st.session_state.ppi = ppi
        st.session_state.best = best
        st.session_state.viz = ResultVisualizer(output_dir / "plots")
        st.session_state.output_dir = output_dir
        st.session_state.selected_pose_rank = poses[0].rank
        st.session_state.receptor_structure = receptor
        st.session_state.ligand_structure = ligand
        st.session_state.pose_details = pose_details

    # 显示分析结果
    if st.session_state.analysis_done:
        poses = st.session_state.poses
        interface = st.session_state.interface
        ppi = st.session_state.ppi
        best = st.session_state.best
        viz = st.session_state.viz
        output_dir = st.session_state.output_dir
        receptor = st.session_state.receptor_structure
        ligand = st.session_state.ligand_structure
        pose_details = st.session_state.pose_details

        st.success(tr("analysis_complete"))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(tr("docking_score"), f"{best.scores.total:.1f}")
        m2.metric(tr("interaction_probability"), f"{ppi.probability:.1%}")
        m3.metric(tr("interface_residues"), interface.n_interface_residues)
        m4.metric(tr("interface_area"), f"{best.scores.raw_interface_area:.0f} Å²")

        st.subheader(tr("ppi_prediction"))
        if ppi.interacts:
            st.success(f"{tr('ppi_positive')} — {localized_ppi_detail(ppi)}")
        else:
            st.warning(f"{tr('ppi_negative')} — {localized_ppi_detail(ppi)}")

        st.subheader(tr("score_ranking"))
        score_df = pd.DataFrame([
            {tr("col_rank"): p.rank, tr("col_total"): p.scores.total,
             tr("col_hydrophobic"): p.scores.hydrophobic,
             tr("col_electrostatic"): p.scores.electrostatic,
             tr("col_contacts"): p.scores.contact_residues,
             tr("col_clash"): p.scores.clash_penalty}
            for p in poses[:top_n]
        ])
        st.dataframe(score_df, use_container_width=True)

        st.subheader(tr("interface_residue_table"))
        interface_analyzer = InterfaceAnalyzer()
        iface_df = localize_interface_dataframe(interface_analyzer.to_dataframe(interface))
        st.dataframe(iface_df, use_container_width=True)

        st.markdown("---")
        st.subheader(f"📊 {tr('score_ranking')}")
        echarts_option = viz.get_docking_scores_echarts(poses[:top_n])
        if echarts_option != "{}":
            render_echarts(echarts_option, height="500px")

        st.markdown("---")
        st.subheader(f"📈 {tr('score_components')}")
        echarts_option = viz.get_score_components_echarts(best)
        if echarts_option != "{}":
            render_echarts(echarts_option, height="500px")

        st.markdown("---")
        st.subheader(f"🔬 {tr('interface_distribution')}")
        echarts_option = viz.get_interface_distribution_echarts(interface)
        if echarts_option != "{}":
            render_echarts(echarts_option, height="550px")

        st.markdown("---")
        st.subheader(f"📉 {tr('distance_distribution')}")
        echarts_option = viz.get_distance_distribution_echarts(interface)
        if echarts_option != "{}":
            render_echarts(echarts_option, height="500px")

        st.markdown("---")
        st.subheader(f"🔥 {tr('contact_map')}")
        echarts_option = viz.get_contact_map_echarts(interface)
        if echarts_option != "{}":
            render_echarts(echarts_option, height="600px")

        # 3D 结构可视化
        st.markdown("---")
        st.subheader(f"🧪 {tr('structure_viewer')}")

        # 构象选择
        pose_options = {
            p.rank: tr("pose_option", rank=p.rank, score=p.scores.total)
            for p in poses
        }

        selected_rank = st.selectbox(
            tr("select_pose"),
            options=list(pose_options.keys()),
            format_func=lambda x: pose_options[x],
            index=list(pose_options.keys()).index(st.session_state.selected_pose_rank),
            key="pose_selector"
        )

        st.session_state.selected_pose_rank = selected_rank

        # 找到选中的构象
        selected_pose = next((p for p in poses if p.rank == selected_rank), poses[0])
        selected_detail = pose_details.get(selected_rank, {})
        selected_interface = selected_detail.get("interface", interface)
        selected_ppi = selected_detail.get("ppi", ppi)
        selected_ligand_chains = selected_detail.get("ligand_chains", sorted(ligand.chains))
        pdb_out = output_dir / f"docked_complex_{selected_rank}.pdb"

        if pdb_out.exists():
            pdb_content = pdb_out.read_text(encoding="utf-8", errors="replace")
            viewer_metrics = [
                {"label": tr("docking_score"), "value": f"{selected_pose.scores.total:.1f}"},
                {"label": tr("interaction_probability"), "value": f"{selected_ppi.probability:.1%}"},
                {"label": tr("interface_residues"), "value": str(selected_pose.scores.contact_residues)},
                {"label": tr("interface_area"), "value": f"{selected_pose.scores.raw_interface_area:.0f} Å²"},
            ]
            viewer_payload = build_viewer_payload(
                pdb_content=pdb_content,
                interface_result=selected_interface,
                receptor_chains=sorted(receptor.chains),
                ligand_chains=selected_ligand_chains,
                pose_title=tr("pose_title", rank=selected_rank),
                metrics=viewer_metrics,
                language=st.session_state.language,
            )
            viewer_html = render_structure_viewer(viewer_payload, height_px=760)

            # 显示选中构象的评分信息
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(tr("pose_score"), f"{selected_pose.scores.total:.1f}")
            with col2:
                st.metric(tr("interaction_probability"), f"{selected_ppi.probability:.1%}")
            with col3:
                st.metric(tr("interface_residues"), selected_interface.n_interface_residues)
            with col4:
                st.metric(tr("interface_area"), f"{selected_pose.scores.raw_interface_area:.0f} Å²")

            st.caption(tr("viewer_caption"))

            st.subheader(tr("selected_interface"))
            st.dataframe(
                localize_interface_dataframe(InterfaceAnalyzer().to_dataframe(selected_interface)),
                use_container_width=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    tr("download_pdb"),
                    pdb_content,
                    file_name=f"docked_complex_rank{selected_rank}.pdb",
                    help=tr("download_pdb_help"),
                )
            with col2:
                st.download_button(
                    tr("download_html"),
                    viewer_html,
                    file_name=f"structure_3d_rank{selected_rank}.html",
                    help=tr("download_html_help"),
                )

with tab_bench:
    st.markdown(tr("benchmark_title"))
    st.markdown(tr("benchmark_description"))
    if st.button(tr("run_benchmark")):
        from tests.benchmark_test import BenchmarkRunner
        from tests.evaluation import EvaluationReport

        with st.spinner(tr("benchmark_spinner")):
            runner = BenchmarkRunner(PROJECT_ROOT / "config.yaml", PROJECT_ROOT)
            results = runner.run_all(download=True)
            evaluator = EvaluationReport(PROJECT_ROOT / "results" / "benchmark" / "evaluation")
            report = evaluator.generate_full_report(results)

        st.success(tr("benchmark_complete", count=len(results)))
        if results:
            df = pd.read_csv(PROJECT_ROOT / "results" / "benchmark" / "benchmark_results.csv")
            st.dataframe(df)
        report_path = PROJECT_ROOT / "results" / "benchmark" / "evaluation" / "evaluation_report.md"
        if report_path.exists():
            st.markdown(report_path.read_text(encoding="utf-8", errors="replace"))

with tab_about:
    st.markdown(tr("about"))
