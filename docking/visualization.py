"""
结果可视化：2D 图表与 3D 结构 (py3Dmol)。
学术风格配色方案参考 Nature/Science 期刊风格。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from docking.interface import InterfaceResult
from docking.scoring import DockingPose, ScoreComponents

logger = logging.getLogger(__name__)

# 学术风格配色方案 - 参考 Nature/Science 期刊
ACADEMIC_COLORS = {
    'primary': '#1a1a2e',      # 深蓝灰 - 主标题
    'secondary': '#16213e',    # 深蓝色 - 副标题
    'accent': '#0f3460',       # 亮蓝 - 强调色
    'highlight': '#e94560',    # 珊瑚红 - 高亮
    'success': '#27ae60',      # 绿色 - 成功/正面
    'warning': '#f39c12',      # 橙色 - 警告
    'info': '#3498db',         # 蓝色 - 信息
    'purple': '#9b59b6',       # 紫色
    'cyan': '#1abc9c',         # 青色
    'gray': '#7f8c8d',         # 灰色
    'light_gray': '#bdc3c7',   # 浅灰
    
    # 渐变配色
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2',
    
    # 分类配色（类似 Nature 期刊）
    'category': [
        '#E64B35',  # 红色
        '#4DBBD5',  # 青色
        '#00A087',  # 绿色
        '#3C5488',  # 深蓝
        '#F39B7F',  # 浅红
        '#8491B4',  # 浅紫蓝
        '#91D1C2',  # 浅青绿
        '#DC0000',  # 深红
        '#7E6148',  # 棕色
    ]
}

# 学术风格参数配置
ACADEMIC_STYLE = {
    'font_family': 'DejaVu Sans',
    'font_size': 10,
    'title_font_size': 12,
    'label_font_size': 10,
    'legend_font_size': 9,
    'line_width': 1.2,
    'marker_size': 6,
    'dpi': 300,
    'figure_width': 8,
    'figure_height': 5,
}


class ResultVisualizer:
    """对接与评估结果可视化 - 学术风格。"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_academic_style()

    def _setup_academic_style(self):
        """配置学术风格的 matplotlib 参数。"""
        plt.rcParams.update({
            'font.family': ACADEMIC_STYLE['font_family'],
            'font.sans-serif': [ACADEMIC_STYLE['font_family']],
            'font.size': ACADEMIC_STYLE['font_size'],
            'axes.titlesize': ACADEMIC_STYLE['title_font_size'],
            'axes.labelsize': ACADEMIC_STYLE['label_font_size'],
            'legend.fontsize': ACADEMIC_STYLE['legend_font_size'],
            'xtick.labelsize': ACADEMIC_STYLE['font_size'] - 1,
            'ytick.labelsize': ACADEMIC_STYLE['font_size'] - 1,
            'lines.linewidth': ACADEMIC_STYLE['line_width'],
            'figure.dpi': ACADEMIC_STYLE['dpi'],
            'savefig.dpi': ACADEMIC_STYLE['dpi'],
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'grid.alpha': 0.3,
        })
        sns.set_style({
            'axes.edgecolor': ACADEMIC_COLORS['gray'],
            'axes.facecolor': 'white',
            'grid.color': ACADEMIC_COLORS['light_gray'],
        })
    
    def _save_figure(self, fig, path):
        """保存图片，使用统一的配置。"""
        fig.savefig(
            path, 
            dpi=ACADEMIC_STYLE['dpi'],
            format='png',
            bbox_inches='tight',
            pad_inches=0.1
        )
        plt.close(fig)
        return path

    def plot_docking_scores(self, poses: List[DockingPose], filename: str = "docking_scores.png") -> Path:
        """对接评分排名图 - 学术风格。"""
        if not poses:
            return self.output_dir / filename

        ranks = [p.rank for p in poses]
        totals = [p.scores.total for p in poses]
        
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # 学术风格渐变配色
        colors = [ACADEMIC_COLORS['category'][i % len(ACADEMIC_COLORS['category'])] 
                  for i in range(len(poses))]
        
        bars = ax.barh(ranks, totals, color=colors, edgecolor='white', linewidth=0.5)
        
        # 最佳构象高亮
        if poses:
            bars[0].set_edgecolor(ACADEMIC_COLORS['highlight'])
            bars[0].set_linewidth(1.5)
        
        ax.set_xlabel('Docking Score', color=ACADEMIC_COLORS['primary'])
        ax.set_ylabel('Rank', color=ACADEMIC_COLORS['primary'])
        ax.set_title('Top Docking Poses by Score', color=ACADEMIC_COLORS['primary'], pad=12)
        ax.invert_yaxis()
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}', ha='left', va='center', 
                    fontsize=ACADEMIC_STYLE['font_size']-1, color=ACADEMIC_COLORS['gray'])
        
        ax.grid(axis='x', linestyle=':', alpha=0.3)
        plt.tight_layout()
        path = self.output_dir / filename
        return self._save_figure(fig, path)

    def plot_score_components(self, pose: DockingPose, filename: str = "score_components.png") -> Path:
        """评分组成堆叠图 - 学术风格。"""
        s = pose.scores
        components = {
            "Hydrophobic": s.hydrophobic * 0.25,
            "Electrostatic": s.electrostatic * 0.20,
            "Contacts": s.contacts * 0.25,
            "Interface Area": s.interface_area * 0.20,
            "Clash Penalty": -s.clash_penalty * 0.10,
        }
        
        fig, ax = plt.subplots(figsize=(7, 4))
        names = list(components.keys())
        values = list(components.values())
        
        # 学术配色方案
        colors = [ACADEMIC_COLORS['category'][i] for i in range(len(names))]
        
        bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=0.8)
        
        ax.axhline(y=0, color=ACADEMIC_COLORS['gray'], linestyle='-', linewidth=0.8)
        ax.set_ylabel('Weighted Score Contribution', color=ACADEMIC_COLORS['primary'])
        ax.set_title(f'Score Components (Total = {s.total:.1f})', 
                     color=ACADEMIC_COLORS['primary'], pad=12)
        plt.xticks(rotation=30, ha='right', color=ACADEMIC_COLORS['secondary'])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            pos_y = height + (0.3 if height > 0 else -0.8)
            ax.text(bar.get_x() + bar.get_width()/2, pos_y,
                    f'{height:.2f}', ha='center', va='bottom',
                    fontsize=ACADEMIC_STYLE['font_size']-1, color=ACADEMIC_COLORS['gray'])
        
        ax.grid(axis='y', linestyle=':', alpha=0.3)
        plt.tight_layout()
        path = self.output_dir / filename
        return self._save_figure(fig, path)

    def plot_interface_distribution(
        self, interface: InterfaceResult, filename: str = "interface_distribution.png"
    ) -> Path:
        """界面残基分布图 - 学术风格。"""
        rec_types = [r.contact_type for r in interface.receptor_interface]
        lig_types = [r.contact_type for r in interface.ligand_interface]
        all_types = rec_types + lig_types

        if not all_types:
            return self.output_dir / filename

        from collections import Counter
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, types, title in zip(
            axes,
            [rec_types, lig_types],
            ["Receptor Interface", "Ligand Interface"],
        ):
            if types:
                counts = Counter(types)
                wedges, texts, autotexts = ax.pie(
                    counts.values(), 
                    labels=counts.keys(), 
                    autopct='%1.1f%%',
                    colors=ACADEMIC_COLORS['category'][:len(counts)],
                    textprops={'fontsize': ACADEMIC_STYLE['font_size']-1},
                    startangle=90
                )
                ax.set_aspect('equal')
            ax.set_title(title, color=ACADEMIC_COLORS['primary'], pad=10)
        
        plt.tight_layout()
        path = self.output_dir / filename
        return self._save_figure(fig, path)

    def plot_contact_map(
        self, interface: InterfaceResult, filename: str = "contact_map.png"
    ) -> Path:
        """接触热图 - 学术风格。"""
        if interface.contact_map is None or interface.contact_map.size == 0:
            return self.output_dir / filename

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 使用更专业的配色方案
        ax = sns.heatmap(
            interface.contact_map, 
            cmap='Blues', 
            ax=ax, 
            cbar_kws={
                'label': 'Contact strength',
                'shrink': 0.8
            },
            linewidths=0.5,
            linecolor='white'
        )
        
        ax.set_xlabel('Ligand Residue Index', color=ACADEMIC_COLORS['primary'])
        ax.set_ylabel('Receptor Residue Index', color=ACADEMIC_COLORS['primary'])
        ax.set_title('Residue Contact Map', color=ACADEMIC_COLORS['primary'], pad=12)
        
        plt.tight_layout()
        path = self.output_dir / filename
        return self._save_figure(fig, path)

    def plot_distance_distribution(
        self, interface: InterfaceResult, filename: str = "distance_distribution.png"
    ) -> Path:
        """接触距离分布 - 学术风格。"""
        distances = [p[2] for p in interface.contact_pairs]
        if not distances:
            return self.output_dir / filename

        fig, ax = plt.subplots(figsize=(7, 4))
        
        # 学术风格直方图
        n, bins, patches = ax.hist(
            distances, 
            bins=20, 
            color=ACADEMIC_COLORS['info'], 
            edgecolor='white', 
            alpha=0.7,
            linewidth=0.5
        )
        
        # 阈值线
        ax.axvline(
            x=5.0, 
            color=ACADEMIC_COLORS['highlight'], 
            linestyle='--', 
            linewidth=1.2,
            label=r'Contact cutoff (5$\AA$)'
        )
        
        ax.set_xlabel(r'Distance ($\AA$)', color=ACADEMIC_COLORS['primary'])
        ax.set_ylabel('Count', color=ACADEMIC_COLORS['primary'])
        ax.set_title('Interface Contact Distance Distribution', 
                     color=ACADEMIC_COLORS['primary'], pad=12)
        ax.legend(fontsize=ACADEMIC_STYLE['legend_font_size'])
        
        ax.grid(axis='y', linestyle=':', alpha=0.3)
        plt.tight_layout()
        path = self.output_dir / filename
        return self._save_figure(fig, path)

    def plot_roc_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, filename: str = "roc_curve.png"
    ) -> Path:
        """ROC 曲线 - 学术风格。"""
        from sklearn.metrics import auc, roc_curve

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        
        # 学术风格 ROC 曲线
        ax.plot(
            fpr, tpr, 
            color=ACADEMIC_COLORS['success'], 
            lw=2, 
            label=f'AUC = {roc_auc:.3f}',
            marker='o',
            markerfacecolor='white',
            markeredgecolor=ACADEMIC_COLORS['success'],
            markersize=4,
            alpha=0.9
        )
        ax.plot([0, 1], [0, 1], color=ACADEMIC_COLORS['gray'], linestyle='--', lw=1)
        
        ax.set_xlabel('False Positive Rate', color=ACADEMIC_COLORS['primary'])
        ax.set_ylabel('True Positive Rate', color=ACADEMIC_COLORS['primary'])
        ax.set_title('ROC Curve - PPI Prediction', color=ACADEMIC_COLORS['primary'], pad=12)
        ax.legend(loc='lower right', fontsize=ACADEMIC_STYLE['legend_font_size'])
        
        ax.grid(linestyle=':', alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        
        plt.tight_layout()
        path = self.output_dir / filename
        return self._save_figure(fig, path)

    def plot_pr_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, filename: str = "pr_curve.png"
    ) -> Path:
        """Precision-Recall 曲线 - 学术风格。"""
        from sklearn.metrics import average_precision_score, precision_recall_curve

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(6, 5))
        
        ax.plot(
            recall, precision, 
            color=ACADEMIC_COLORS['purple'], 
            lw=2, 
            label=f'AP = {ap:.3f}',
            marker='s',
            markerfacecolor='white',
            markeredgecolor=ACADEMIC_COLORS['purple'],
            markersize=4,
            alpha=0.9
        )
        
        ax.set_xlabel('Recall', color=ACADEMIC_COLORS['primary'])
        ax.set_ylabel('Precision', color=ACADEMIC_COLORS['primary'])
        ax.set_title('Precision-Recall Curve', color=ACADEMIC_COLORS['primary'], pad=12)
        ax.legend(fontsize=ACADEMIC_STYLE['legend_font_size'])
        
        ax.grid(linestyle=':', alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        
        plt.tight_layout()
        path = self.output_dir / filename
        return self._save_figure(fig, path)

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, filename: str = "confusion_matrix.png"
    ) -> Path:
        """混淆矩阵 - 学术风格。"""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        
        ax = sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            ax=ax,
            xticklabels=['No Interaction', 'Interaction'],
            yticklabels=['No Interaction', 'Interaction'],
            cbar=False,
            annot_kws={'fontsize': ACADEMIC_STYLE['font_size']}
        )
        
        ax.set_xlabel('Predicted', color=ACADEMIC_COLORS['primary'])
        ax.set_ylabel('Actual', color=ACADEMIC_COLORS['primary'])
        ax.set_title('Confusion Matrix', color=ACADEMIC_COLORS['primary'], pad=12)
        
        plt.tight_layout()
        path = self.output_dir / filename
        return self._save_figure(fig, path)

    def plot_benchmark_accuracy(
        self, metrics: Dict[str, float], filename: str = "benchmark_accuracy.png"
    ) -> Path:
        """Benchmark 准确率柱状图 - 学术风格。"""
        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(metrics.keys())
        values = list(metrics.values())
        
        colors = [ACADEMIC_COLORS['category'][i % len(ACADEMIC_COLORS['category'])] 
                  for i in range(len(names))]
        
        bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score', color=ACADEMIC_COLORS['primary'])
        ax.set_title('Benchmark Performance Metrics', color=ACADEMIC_COLORS['primary'], pad=12)
        
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, 
                bar.get_height() + 0.02,
                f'{val:.3f}', 
                ha='center', 
                fontsize=ACADEMIC_STYLE['font_size']-1,
                color=ACADEMIC_COLORS['gray']
            )
        
        plt.xticks(rotation=45, ha='right', color=ACADEMIC_COLORS['secondary'])
        ax.grid(axis='y', linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / filename
        return self._save_figure(fig, path)

    def plot_rmsd_distribution(
        self, rmsd_values: List[float], filename: str = "rmsd_distribution.png"
    ) -> Path:
        """RMSD 分布图 - 学术风格。"""
        if not rmsd_values:
            return self.output_dir / filename
        
        fig, ax = plt.subplots(figsize=(7, 4))
        
        n, bins, patches = ax.hist(
            rmsd_values, 
            bins=15, 
            color=ACADEMIC_COLORS['warning'], 
            edgecolor='white',
            alpha=0.7,
            linewidth=0.5
        )
        
        ax.axvline(
            x=10.0, 
            color=ACADEMIC_COLORS['highlight'], 
            linestyle='--', 
            linewidth=1.2,
            label=r'Acceptable threshold (10$\AA$)'
        )
        
        ax.set_xlabel(r'RMSD ($\AA$)', color=ACADEMIC_COLORS['primary'])
        ax.set_ylabel('Count', color=ACADEMIC_COLORS['primary'])
        ax.set_title('Docking RMSD Distribution', color=ACADEMIC_COLORS['primary'], pad=12)
        ax.legend(fontsize=ACADEMIC_STYLE['legend_font_size'])
        
        ax.grid(axis='y', linestyle=':', alpha=0.3)
        plt.tight_layout()
        path = self.output_dir / filename
        return self._save_figure(fig, path)

    def generate_3d_view(
        self,
        pdb_path: Path,
        interface_residues: Optional[List[str]] = None,
        filename: str = "structure_3d.html",
    ) -> Optional[Path]:
        """生成交互式 3D 视图 (py3Dmol)。"""
        try:
            import py3Dmol
        except ImportError:
            logger.warning("py3Dmol 未安装，跳过 3D 可视化")
            return None

        pdb_path = Path(pdb_path)
        with open(pdb_path, encoding="utf-8", errors="replace") as f:
            pdb_data = f.read()

        view = py3Dmol.view(width=800, height=600)
        view.addModel(pdb_data, "pdb")
        view.setStyle({"cartoon": {"color": "spectrum"}})

        if interface_residues:
            for res in interface_residues:
                # res format: TYR45
                resname = "".join(c for c in res if c.isalpha())[:3]
                resnum = "".join(c for c in res if c.isdigit())
                if resnum:
                    view.addStyle(
                        {"resi": resnum, "resn": resname},
                        {"stick": {"colorscheme": "yellowCarbon", "radius": 0.3}},
                    )

        view.zoomTo()
        html = view._make_html()
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    def generate_all_plots(
        self,
        poses: List[DockingPose],
        interface: InterfaceResult,
        best_pose: Optional[DockingPose] = None,
    ) -> List[Path]:
        """生成所有对接结果图表。"""
        paths = []
        paths.append(self.plot_docking_scores(poses))
        if best_pose:
            paths.append(self.plot_score_components(best_pose))
        paths.append(self.plot_interface_distribution(interface))
        paths.append(self.plot_contact_map(interface))
        paths.append(self.plot_distance_distribution(interface))
        return [p for p in paths if p.exists()]

    # ============ ECharts 交互式图表生成 ============
    
    def get_docking_scores_echarts(self, poses: List[DockingPose]) -> str:
        """生成对接评分的 ECharts JSON 配置。"""
        if not poses:
            return json.dumps({})
        
        ranks = [p.rank for p in poses]
        totals = [p.scores.total for p in poses]
        
        option = {
            "title": {
                "text": "Top Docking Poses by Score",
                "left": "center",
                "textStyle": {"color": ACADEMIC_COLORS['primary'], "fontSize": 14}
            },
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow"},
                "formatter": "{b}: {c}"
            },
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "xAxis": {
                "type": "value",
                "name": "Docking Score",
                "nameTextStyle": {"color": ACADEMIC_COLORS['gray']},
                "axisLabel": {"color": ACADEMIC_COLORS['secondary']}
            },
            "yAxis": {
                "type": "category",
                "data": [str(r) for r in reversed(ranks)],
                "name": "Rank",
                "nameTextStyle": {"color": ACADEMIC_COLORS['gray']},
                "axisLabel": {"color": ACADEMIC_COLORS['secondary']}
            },
            "series": [{
                "name": "Score",
                "type": "bar",
                "data": list(reversed(totals)),
                "itemStyle": {
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 1, "y2": 0,
                        "colorStops": [
                            {"offset": 0, "color": ACADEMIC_COLORS['gradient_start']},
                            {"offset": 1, "color": ACADEMIC_COLORS['gradient_end']}
                        ]
                    },
                    "borderRadius": [0, 4, 4, 0]
                },
                "emphasis": {
                    "itemStyle": {"color": ACADEMIC_COLORS['highlight']}
                },
                "label": {
                    "show": True,
                    "position": "right",
                    "formatter": "{c}",
                    "color": ACADEMIC_COLORS['gray']
                }
            }]
        }
        return json.dumps(option)

    def get_score_components_echarts(self, pose: DockingPose) -> str:
        """生成评分组成的 ECharts JSON 配置。"""
        s = pose.scores
        components = {
            "Hydrophobic": s.hydrophobic * 0.25,
            "Electrostatic": s.electrostatic * 0.20,
            "Contacts": s.contacts * 0.25,
            "Interface Area": s.interface_area * 0.20,
            "Clash Penalty": -s.clash_penalty * 0.10,
        }
        
        option = {
            "title": {
                "text": f"Score Components (Total = {s.total:.1f})",
                "left": "center",
                "textStyle": {"color": ACADEMIC_COLORS['primary'], "fontSize": 14}
            },
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow"},
                "formatter": "{b}: {c}"
            },
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "xAxis": {
                "type": "category",
                "data": list(components.keys()),
                "axisLabel": {"rotate": 30, "color": ACADEMIC_COLORS['secondary']},
                "axisTick": {"alignWithLabel": True}
            },
            "yAxis": {
                "type": "value",
                "name": "Weighted Contribution",
                "nameTextStyle": {"color": ACADEMIC_COLORS['gray']},
                "axisLabel": {"color": ACADEMIC_COLORS['secondary']}
            },
            "series": [{
                "type": "bar",
                "data": list(components.values()),
                "itemStyle": {
                    "color": ACADEMIC_COLORS['category'][:len(components)],
                    "borderRadius": [4, 4, 0, 0]
                },
                "emphasis": {
                    "itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0,0,0,0.3)"}
                },
                "label": {
                    "show": True,
                    "position": "top",
                    "formatter": "{c}",
                    "color": ACADEMIC_COLORS['gray']
                }
            }]
        }
        return json.dumps(option)

    def get_distance_distribution_echarts(self, interface: InterfaceResult) -> str:
        """生成接触距离分布的 ECharts JSON 配置。"""
        distances = [p[2] for p in interface.contact_pairs]
        if not distances:
            return json.dumps({})
        
        # 计算直方图数据
        hist, bins = np.histogram(distances, bins=20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        option = {
            "title": {
                "text": "Interface Contact Distance Distribution",
                "left": "center",
                "textStyle": {"color": ACADEMIC_COLORS['primary'], "fontSize": 14}
            },
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "cross"},
                "formatter": "{b}Å: {c}"
            },
            "legend": {
                "data": ["Distance Distribution", "Contact Cutoff"],
                "top": 30,
                "textStyle": {"color": ACADEMIC_COLORS['secondary']}
            },
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "xAxis": {
                "type": "value",
                "name": "Distance (Å)",
                "nameTextStyle": {"color": ACADEMIC_COLORS['gray']},
                "axisLabel": {"color": ACADEMIC_COLORS['secondary']}
            },
            "yAxis": {
                "type": "value",
                "name": "Count",
                "nameTextStyle": {"color": ACADEMIC_COLORS['gray']},
                "axisLabel": {"color": ACADEMIC_COLORS['secondary']}
            },
            "series": [
                {
                    "name": "Distance Distribution",
                    "type": "bar",
                    "data": [{"value": float(h), "name": float(c)} for h, c in zip(hist, bin_centers)],
                    "itemStyle": {
                        "color": ACADEMIC_COLORS['info'],
                        "borderRadius": [2, 2, 0, 0]
                    },
                    "emphasis": {"itemStyle": {"color": ACADEMIC_COLORS['accent']}}
                },
                {
                    "name": "Contact Cutoff",
                    "type": "line",
                    "data": [[5.0, 0], [5.0, max(hist) * 1.1]],
                    "symbol": "none",
                    "lineStyle": {
                        "color": ACADEMIC_COLORS['highlight'],
                        "type": "dashed",
                        "width": 2
                    },
                    "markLine": {
                        "silent": True,
                        "data": [{"xAxis": 5.0}]
                    }
                }
            ]
        }
        return json.dumps(option)

    def get_interface_distribution_echarts(self, interface: InterfaceResult) -> str:
        """生成界面残基分布的 ECharts JSON 配置。"""
        from collections import Counter
        
        rec_types = [r.contact_type for r in interface.receptor_interface]
        lig_types = [r.contact_type for r in interface.ligand_interface]
        
        rec_counts = Counter(rec_types)
        lig_counts = Counter(lig_types)
        
        option = {
            "title": {
                "text": "Interface Residue Type Distribution",
                "left": "center",
                "textStyle": {"color": ACADEMIC_COLORS['primary'], "fontSize": 14}
            },
            "tooltip": {
                "trigger": "item",
                "formatter": "{b}: {c} ({d}%)"
            },
            "legend": {
                "bottom": 10,
                "textStyle": {"color": ACADEMIC_COLORS['secondary']}
            },
            "series": [
                {
                    "name": "Receptor Interface",
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "center": ["25%", "50%"],
                    "avoidLabelOverlap": False,
                    "itemStyle": {
                        "borderRadius": 4,
                        "borderColor": "#fff",
                        "borderWidth": 2
                    },
                    "label": {"show": True, "color": ACADEMIC_COLORS['secondary']},
                    "data": [{"value": v, "name": k} for k, v in rec_counts.items()]
                },
                {
                    "name": "Ligand Interface",
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "center": ["75%", "50%"],
                    "avoidLabelOverlap": False,
                    "itemStyle": {
                        "borderRadius": 4,
                        "borderColor": "#fff",
                        "borderWidth": 2
                    },
                    "label": {"show": True, "color": ACADEMIC_COLORS['secondary']},
                    "data": [{"value": v, "name": k} for k, v in lig_counts.items()]
                }
            ],
            "color": ACADEMIC_COLORS['category']
        }
        return json.dumps(option)

    def get_contact_map_echarts(self, interface: InterfaceResult) -> str:
        """生成接触热图的 ECharts JSON 配置。"""
        if interface.contact_map is None or interface.contact_map.size == 0:
            return json.dumps({})
        
        data = interface.contact_map.tolist()
        n_rows, n_cols = interface.contact_map.shape
        
        option = {
            "title": {
                "text": "Residue Contact Map",
                "left": "center",
                "textStyle": {"color": ACADEMIC_COLORS['primary'], "fontSize": 14}
            },
            "tooltip": {
                "position": "top",
                "formatter": "Receptor:{b}<br/>Ligand:{c}<br/>Strength:{d}"
            },
            "grid": {"left": "10%", "right": "10%", "bottom": "15%", "top": "15%"},
            "xAxis": {
                "type": "category",
                "data": [str(i) for i in range(n_cols)],
                "name": "Ligand Residue Index",
                "nameTextStyle": {"color": ACADEMIC_COLORS['gray']},
                "axisLabel": {"color": ACADEMIC_COLORS['secondary'], "interval": 5},
                "splitArea": {"show": True}
            },
            "yAxis": {
                "type": "category",
                "data": [str(i) for i in range(n_rows)],
                "name": "Receptor Residue Index",
                "nameTextStyle": {"color": ACADEMIC_COLORS['gray']},
                "axisLabel": {"color": ACADEMIC_COLORS['secondary'], "interval": 5},
                "splitArea": {"show": True}
            },
            "visualMap": {
                "min": 0,
                "max": np.max(interface.contact_map),
                "calculable": True,
                "orient": "horizontal",
                "left": "center",
                "bottom": "0%",
                "textStyle": {"color": ACADEMIC_COLORS['secondary']},
                "inRange": {"color": ['#e8f4f8', '#9fd5e8', '#3498db', '#2980b9']}
            },
            "series": [{
                "name": "Contact Strength",
                "type": "heatmap",
                "data": [
                    [j, i, float(data[i][j])]
                    for i in range(n_rows)
                    for j in range(n_cols)
                ],
                "label": {"show": False},
                "emphasis": {
                    "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}
                }
            }]
        }
        return json.dumps(option)
