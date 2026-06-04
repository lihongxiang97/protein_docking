# PPI Docking — 蛋白质-蛋白质相互作用预测与分子对接软件

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

自主实现的 Python 蛋白-蛋白分子对接（Protein-Protein Docking）与互作位点预测系统。核心算法完全基于 Python 编写，不依赖 HDOCK、ZDOCK、ClusPro 等外部对接程序。

---

## 1. 软件简介

PPI Docking 能够：

- 输入两个蛋白质 PDB 三维结构；
- 自动完成结构预处理（去水、加氢、表面残基识别）；
- 执行刚性蛋白-蛋白分子对接（网格搜索 + Monte Carlo）；
- 预测蛋白间是否存在潜在互作；
- 识别并输出界面残基（interface residues）；
- 生成对接评分、可视化图表与 Markdown 报告；
- 支持 Benchmark 自动测试与性能评估。

---

## 2. 安装方式

### 环境要求

- Python ≥ 3.10
- pip 或 conda

### 安装依赖

```bash
cd protein_docking_project
pip install -r requirements.txt
```

### Conda 环境（推荐）

```bash
conda create -n ppi_docking python=3.11 -y
conda activate ppi_docking
pip install -r requirements.txt
```

### Docker

```bash
docker build -t ppi-docking .
docker run -p 8501:8501 ppi-docking
# 访问 http://localhost:8501
```

---

## 3. 快速开始

### 生成示例结构

```bash
python scripts/generate_examples.py
```

该命令会在 `data/example_pdb/` 下准备来自 **RCSB PDB** 的真实已知互作复合物示例，并生成：

- `example_manifest.json`：示例清单与来源链接
- `receptor.pdb` / `ligand.pdb`：默认示例对的快捷入口
- `*_receptor_*.pdb` / `*_ligand_*.pdb`：按复合物拆分后的单链输入文件

### 命令行对接

```bash
python main.py \
    --receptor data/example_pdb/receptor.pdb \
    --ligand data/example_pdb/ligand.pdb \
    --output results/
```

### Web 界面

```bash
python main.py --web
# 或
streamlit run web/app.py
```

Web 端支持直接选择内置真实示例，并使用 NGL 展示完整对接复合物、界面残基 sticks、接触连线与半透明表面。

### Benchmark 测试

```bash
python main.py --benchmark --output results/
```

---

## 4. 输入输出说明

### 输入

| 参数 | 说明 |
|------|------|
| `--receptor` | 受体 PDB 文件 |
| `--ligand` | 配体 PDB 文件 |
| `--output` | 输出目录（默认 `results/`） |
| `--config` | 配置文件（默认 `config.yaml`） |
| `--receptor-chains` | 受体链 ID，逗号分隔 |
| `--ligand-chains` | 配体链 ID，逗号分隔 |

### 输出

```
results/
├── docked_complex_1.pdb      # 最佳对接复合物
├── docked_complex_2.pdb      # 其他候选构象
├── docking_scores.csv        # 各构象评分
├── interface_residues.csv    # 界面残基列表
├── interface_residues.txt    # 文本格式界面残基
├── interaction_summary.txt   # 互作预测摘要
├── docking_report.md         # 分析报告
└── plots/
    ├── docking_scores.png
    ├── score_components.png
    ├── interface_distribution.png
    ├── contact_map.png
    └── structure_3d.html     # 3D 交互视图
```

---

## 5. 算法原理

### 5.1 表面残基识别 (SASA)

采用基于 **KD-tree 邻居密度** 的 rolling sphere 近似：

1. 对每个原子，在探针半径 (1.4 Å) 范围内统计邻居遮蔽；
2. 暴露度 → 原子 SASA → 残基 SASA 聚合；
3. SASA ≥ 25 Å² 的残基标记为表面残基。

**复杂度**: O(N log N) 时间, O(N) 空间

### 5.2 分子对接

**粗搜索**: 斐波那契球面均匀旋转 × 平移网格  
**精修**: Metropolis Monte Carlo 局部优化

刚体变换：

$$\mathbf{x}' = \mathbf{R}(\mathbf{x} - \mathbf{c}) + \mathbf{c} + \mathbf{t}$$

旋转矩阵由四元数或 ZYZ 欧拉角生成。

**复杂度**: O(R × T × N) + O(M × N)，R=旋转数, T=平移数, M=MC 迭代, N=原子数

### 5.3 评分函数

$$Score = w_H \cdot H + w_E \cdot E + w_C \cdot C + w_A \cdot A - w_P \cdot P + 2 \cdot N_{HB}$$

| 项 | 含义 | 权重 |
|----|------|------|
| H | 疏水残基界面配对 | 0.25 |
| E | 静电互补 (异性电荷) | 0.20 |
| C | 原子接触数 | 0.25 |
| A | 界面面积 | 0.20 |
| P | 碰撞惩罚 | 0.10 |

### 5.4 PPI 预测

**特征** (10 维): 界面面积、接触残基数、疏水比例、静电得分、对接分、氢键数、碰撞惩罚、界面残基数、平均接触距离、接触密度

**模型**: Random Forest（有训练数据时）或基于规则的评分模型（默认）

### 5.5 界面残基识别

CA 原子距离 < 5 Å 的残基对定义为界面残基，并按氢键/疏水/静电/van der Waals 分类。

---

## 6. 项目结构

```
protein_docking_project/
├── data/
│   ├── example_pdb/          # 示例 PDB
│   └── benchmark/            # Benchmark 数据
├── docking/
│   ├── structure.py          # PDB 解析与结构模型
│   ├── preprocess.py         # 结构预处理
│   ├── surface.py            # SASA 与表面分析
│   ├── geometry.py           # 刚体变换与碰撞检测
│   ├── docking.py            # 对接算法
│   ├── scoring.py            # 评分函数
│   ├── interface.py          # 界面残基预测
│   ├── ppi_predictor.py      # PPI 分类器
│   └── visualization.py      # 可视化
├── tests/
│   ├── benchmark_data.py     # 数据集管理
│   ├── benchmark_test.py     # 自动测试
│   └── evaluation.py         # 评估报告
├── web/
│   └── app.py                # Streamlit 界面
├── scripts/
│   └── generate_examples.py
├── main.py                   # 主入口
├── config.yaml               # 配置
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 7. Benchmark 测试说明

### 数据集

- **正样本**: 15 对已知互作复合物 (PDB: 1AY7, 1BUH, 1CGI, ...)
- **负样本**: 8 对空间分离的非互作蛋白对

数据自动从 RCSB PDB 下载；离线时生成合成螺旋结构。

### 评估指标

| 类别 | 指标 |
|------|------|
| 分类 | Accuracy, Precision, Recall, F1, MCC, AUROC, AUPRC |
| 对接 | RMSD, FNAT, DockQ |
| 位点 | Interface precision/recall |

### 运行

```bash
python main.py --benchmark
```

报告输出至 `results/benchmark/evaluation/evaluation_report.md`

---

## 8. 常见问题

**Q: 对接速度很慢？**  
A: 可在 `config.yaml` 中减少 `coarse_rotations`、`mc_iterations`。

**Q: PDB 下载失败？**  
A: 软件会自动生成合成结构用于离线测试。检查网络或使用 `--no-download`。

**Q: 如何提高准确度？**  
A: 提供高质量实验结构；指定正确链 ID；增大采样数；使用训练数据微调 Random Forest。

**Q: 是否支持多链复合物？**  
A: 支持，通过 `--receptor-chains` 和 `--ligand-chains` 指定。

---

## 9. 算法局限性与优化方向

### 局限性

- 刚性对接，不考虑侧链柔性及诱导契合
- SASA 为近似算法，精度低于 MSMS/Shrake-Rupley
- 氢键检测基于距离阈值，未考虑角度
- Benchmark RMSD 采用简化对齐

### 优化方向

- FFT 加速全局搜索
- 侧链旋转采样
- 图神经网络 (GNN) 界面预测
- GPU/CUDA 并行空间搜索
- AlphaFold2 预测结构对接
- Transformer 蛋白表征

---

## 10. 引用与许可

本软件仅供科研与学习使用。算法思想参考了蛋白对接领域的经典方法（如 FFT docking、CAPRI 评估标准），但所有代码均为自主实现。

MIT License
