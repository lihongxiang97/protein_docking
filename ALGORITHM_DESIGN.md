# 混合蛋白-蛋白对接算法设计

## 设计目标

本项目现在采用独立实现的分阶段混合对接框架。设计参考 HDOCK、HADDOCK 和
MEGADOCK 的公开方法思想，但不复制它们的源代码或专有参数。

核心目标：

1. 使用 FFT 扩大全局刚体搜索覆盖范围。
2. 将自由对接、输入相对构象先验和实验/预测界面约束统一重排。
3. 通过多种子局部精修和构象聚类输出多样化结果。
4. 保留可解释的评分分量、候选来源和聚类规模。

## 与参考方法的对应关系

### MEGADOCK 风格全局搜索

`docking/fft_search.py` 将受体和配体编码为三维网格，对每个配体旋转使用 FFT
相关计算同时评估全部平移。

当前网格分量：

- 形状互补：受体表面壳层奖励与核心碰撞惩罚。
- 静电互补：带电残基网格的反相关。
- 疏水互补：疏水残基网格相关。

FFT 阶段只负责生成候选，候选随后必须通过原子级碰撞检查与详细评分。

### HADDOCK 风格信息驱动对接

配置或命令行可以提供受体与配体活性残基。每个活性残基只要求接近另一侧任一
活性残基，形成简化的模糊相互作用约束。

约束评分输出：

- `restraint_score`
- `restraint_violations`

约束参与最终详细重排，但当前尚未直接参与 FFT 网格相关。

### HDOCK 风格混合候选

默认混合模式会同时考虑：

- FFT 自由对接候选。
- 输入结构原有相对构象候选。

输入相对构象通过 `input_pose_bonus` 获得可配置先验。若要进行严格盲对接，应使用
`--blind` 禁用该先验。

## 当前流水线

1. PDB 解析、链选择、清理、表面和电荷赋值。
2. 低差异旋转采样。
3. FFT 形状/静电/疏水全局平移搜索。
4. 原子级碰撞过滤与多分量详细重排。
5. 多个高分种子的 Monte Carlo 局部刚体精修。
6. 配体 RMSD 去重与聚类。
7. 输出每个聚类的最高分代表构象。

## 使用模式

混合模式，适合输入可能保留有意义相对位置的场景：

```powershell
python main.py -r receptor.pdb -l ligand.pdb -o results/hybrid
```

严格盲对接：

```powershell
python main.py -r receptor.pdb -l ligand.pdb -o results/blind --blind --rotations 360
```

使用活性残基约束：

```powershell
python main.py -r receptor.pdb -l ligand.pdb -o results/restrained `
  --blind `
  --receptor-active A:12,A:45 `
  --ligand-active B:8,B:19 `
  --restraint-target 6 `
  --restraint-upper 10
```

## 关键配置

- `docking.coarse_rotations`: 全局旋转数量。盲搜精度与耗时的主要控制项。
- `docking.include_input_pose`: 是否加入输入相对构象先验。
- `docking.input_pose_bonus`: 输入相对构象先验强度。
- `docking.global_candidate_limit`: 进入原子级重排的候选数量。
- `docking.refine_seeds`: 进入局部精修的独立种子数量。
- `docking.cluster_rmsd`: 聚类 RMSD 阈值。
- `fft_search.grid_spacing`: FFT 网格间距。
- `fft_search.top_translations_per_rotation`: 每个旋转保留的平移峰数量。

## 当前验证结果

在内置 1AY7 示例上：

- 旧搜索 Top-1 配体 RMSD 约为 40 Å。
- 完整无绘图运行约 13 秒到 30 秒，取决于环境和旋转数量。

混合模式默认保留可信输入相对构象，并将其与自由搜索候选共同重排。若输入来自已知
复合物，保留该构象属于先验保持，不能作为对接准确率报告。

严格盲搜诊断（48 个旋转，无输入先验）：

- FFT 候选集中生成了 2.62 Å 的近天然候选。
- 当前手工详细评分没有将该候选排入 Top-5。
- 无局部精修时，Top-2 中最佳配体 RMSD 为 13.35 Å。

因此当前主要瓶颈已经从候选采样转移到重排评分。下一阶段应使用标准 benchmark
训练或标定统计势，而不是仅针对单个复合物继续手工调权重。

## 已补充的标准评估

`docking/metrics.py` 已实现 CAPRI/DockQ 风格结构评估：

- FNAT：使用受体-配体跨链重原子 5 Å 内的原生残基接触保留比例。
- LRMSD：受体骨架对齐后的配体骨架 RMSD。
- iRMSD：原生界面残基骨架原子最优叠合 RMSD，界面由重原子 10 Å 接触定义。
- DockQ：使用 FNAT、iRMSD 和 LRMSD 的标准缩放公式。
- CAPRI class：输出 high / medium / acceptable / incorrect / not_available 分级。

Benchmark 输出继续保留原 `rmsd` 字段作为 LRMSD 兼容列，并新增 `irmsd`、
`dockq` 与 `capri_class`。

## 可靠数据与机器学习重排

`BENCHMARK_DATASETS.md` 给出了推荐数据源分层：

- DB5.5：优先用于最终 docking 验证。
- DOCKGROUND decoys：适合训练和评估 pose-ranking 模型。
- DIPS-Plus：适合深度学习界面/contact 预测预训练。
- SKEMPI 2.0：适合结合能和突变敏感评分标定。
- IntAct/BioGRID：适合 PPI 分类候选，但需要结构映射后才能用于 docking。

新增 `scripts/collect_reliable_ppi_benchmarks.py` 用于解析 DB5.5 官方表格、写出
manifest，并可下载官方 `benchmark5.5.tgz` cleaned-up 结构归档、官方 Excel 表格，
以及按需从 RCSB PDB 下载和拆分 bound 受体/配体链。

新增 pose reranker 训练流水线：

1. `scripts/build_pose_training_set.py` 接收 DB5.5 或用户 CSV/JSON manifest。
2. 对每个 case 执行 docking，收集 Top-N pose 特征。
3. 用 `docking.metrics.evaluate_complex` 计算 LRMSD、iRMSD、FNAT、DockQ 和 CAPRI class。
4. 可用 `--include-native-pose` 为每个 case 加入 bound-native 正样本锚点，避免
   严格盲对接 decoy 全为负样本。
5. 写出 `acceptable` 标签，作为分类器监督信号。
6. `scripts/train_pose_reranker.py` 训练 Random Forest 或 MLP 神经网络。

用户自定义 manifest CSV 格式：

```csv
case_id,receptor_path,ligand_path,native_complex_path,receptor_chains,ligand_chains
case_001,data/user/case_001_rec.pdb,data/user/case_001_lig.pdb,data/user/case_001_native.pdb,A,B
```

默认 ML 重排器配置：

```yaml
docking:
  reranker_model: models/default_pose_reranker.joblib
  reranker_weight: 40.0
```

若模型文件不存在，软件自动退回经验评分；一旦训练并保存
`models/default_pose_reranker.joblib`，命令行和 Web 端都会默认使用该模型。训练数据
应划分 train/validation/test，并保留一部分 DB5.5 或 DOCKGROUND case 作为最终盲测，
避免把最终测试集泄漏到训练中。

## 尚未完成

- 基于模板库的序列搜索和复合物模板建模。
- 半柔性界面侧链/主链精修。
- 显式溶剂或力场能量最小化。
- 训练得到的统计势或原子对势。
- GPU FFT、旋转并行和大规模筛选。

在这些能力完成前，结果应被视为候选构象生成和排序，而不是实验级结合结构证明。

## 方法参考

- HDOCK hybrid strategy:
  https://academic.oup.com/nar/article/45/W1/W365/3829194
- HADDOCK ambiguous interaction restraints:
  https://pubs.acs.org/doi/10.1021/ja026939x
- HADDOCK restraints manual:
  https://www.bonvinlab.org/haddock3-user-manual/intro_restraints.html
- MEGADOCK 4.0 FFT docking:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4221127/
- DockQ protein-protein docking quality measure:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4999177/
- CAPRI-Q / DockQ metric definitions:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11458157/
