# Training-Data Provenance Governs the Cognitive Theories Encoded by Neural Models of Risky Choice

训练数据来源决定风险选择神经模型所编码的认知理论

[English README](README.md)

同名论文的代码与结果。

## 这是什么

为预测人类风险选择而训练的神经网络，正越来越多地被解读为决策过程的
*被发现的理论*。本仓库实现了一套实验框架，把每一个训练好的网络都当作一名
**被试**：对来自每一个（数据集 × 架构 × 随机种子）单元的每个模型施以同一套
诊断探针电池，从而把每个模型归约为其所编码理论的四种读出量——

1. 一个**行为指纹**（原始探针预测向量，1,742 个探针），
2. 五个可解释的**行为效应指数**（罕见事件、确定性、反射、损失厌恶 κ、
   EV 一致性），
3. 六个拟合到探针行为上的 **CPT 理论坐标**（α、β、λ、γ⁺、γ⁻、τ），
4. 一个**归因读出量**——赌局特征上的精确分组 Shapley 值，它完全不假设任何
   函数形式（`shapcode/`）。

随后，一个双因素分解追问：每个读出量在多大程度上由训练数据的**数据来源**
决定，又在多大程度上由**模型类别**决定。

## 核心结果，以及这些数字该怎么读

**真正重要的比较对象是人为基线（artifact floor），而不是原始 ω²。**

每一种读出量自身都会制造出一些表观效应。架构直接扰动预测曲面，因此即便
*真实*的生成理论被字面地固定不变，探针电池仍会报告出很大的架构效应。所以
每个因素都要与在生成主体已知的数据臂上、针对该因素测得的基线相比较
（`06_analyze.py`、`results/tables/artifact_corrected.csv`、
`artifact_bootstrap.csv`）。这两条基线是在**不同的**套件上测得的，并不对称：
**架构**基线是 `sufficiency` 数据臂上的架构 ω²（那里每个数据臂共享同一个真实
主体），而**数据来源**基线是受控 2⁴ 套件中的问题池 ω²（`omega2_problems`）。
因此数据来源基线具体而言是一个问题池替换效应；同一析因设计还报告了一个
数据臂规模成分（`omega2_n_problems`，中位数 0.209），而该基线并不包含它。

| | 超过其基线（点估计） | 自助法 95% 区间严格高于基线 |
|---|---|---|
| **数据来源** | **11 / 11** 个读出量 | **10 / 11** |
| **架构** | 2 / 11 个读出量 | **0 / 11** |

这就是论文的主张，也正是图 2 所展示的内容。

那个免于函数形式假设的归因读出量，在其最强的特征组上指向同一方向——在两个
选项之间的 EV 差上，数据来源 ω² = 0.881，对应基线为 0.019——但它是一次比上述
11/11 **更弱的**复制，引用时也应当如此说明：在其 11 个特征组中，数据来源在 5 个
组上超过基线，架构在 3 个组上超过基线
（`results/tables/attribution_decomposition.csv`）。

**未经校正的分解指向相反的方向，本仓库对此直言不讳。**同样这 11 个读出量上的
原始偏 ω²（`results/tables/h1_variance_decomposition.csv`，表 3）：

| 读出量族 | 筛选 | n | ω²_dataset 中位数 | ω²_arch 中位数 | 数据集胜出 |
|---|---|---|---|---|---|
| CPT 坐标（6） | `theory_identifiable`，剔除 LINEAR | 255 | 0.363 | **0.470** | 3 / 6 |
| 行为指数（5） | `identifiable` | 309 | 0.231 | **0.497** | 2 / 5 |
| 全部 11 个合并 | — | — | 0.360 | **0.487** | 5 / 11 |

多元指纹与未校正的观点一致：PERMANOVA 架构 *F* = 27.2，数据集 *F* = 17.8
（两者 *p* = 0.0005）；按架构的轮廓系数为 0.136，按数据集为 0.058。

所以，原始数字在 11 个读出量中的 6 个上偏向架构，而**这一架构信号几乎全部是
基线**——即便每个数据臂共享同样的真实坐标，它也依然存在。数据来源才是在校正
之后仍然存活的效应。任何引用本仓库时只报原始 ω² 而不给出基线的做法，引用的都是
错误的数字；`results/summary.md` 把两者并排打印出来，并在每次运行时由
`06_analyze.py` 重新生成。

## 仓库结构

```
code/                   main pipeline
  src/                  package (paths.py is the single source of truth for paths)
  scripts/              00-10, the pipeline stages
  tests/                CPT numpy/torch agreement, lottery expansion
shapcode/               attribution read-out (own README + axiom test)
data/external/          redistributable CPC18 aggregates + ATTRIBUTION.md
data/processed/         generated arm caches (.npz) — not in git, ~1 min to rebuild
results/                summary.md, analysis tables and per-run records
                        (the 2,240 raw run directories themselves are not in git)
paper_results/          manuscript deliverables: 6 main figures, 8 supplementary, 20 tables
```

以下内容不在 git 中，原因是它们要么可再生成、要么不归我们再分发（见
`.gitignore`）：`results/runs/` 是 2,240 个逐次运行的 `.npz` 目录（约 32 MB，
可由 `01_run_grid.py` 完整重建——但由其派生出的表格*确实*已提交）；
`data/processed/` 约一分钟即可重建；第三方原始数据必须按其自身许可协议获取。

## 环境要求

Python ≥ 3.9，仅需 CPU（Intel 或 Apple Silicon）。这些模型都很小；`--device
mps` 可用，但通常 CPU 更快。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

可选：`pip install pysr` 以获得质量更高的符号回归后端（首次使用时会下载
Julia）。没有它时则使用 `gplearn`。

## 数据

**已随仓库提供**（`data/external/`，其数据来源与许可见 `ATTRIBUTION.md`）：
CPC18 估计集——210 个问题，附完全展开的结果分布——以及官方的汇总 B-rates，
还有本项目从公开的 CPC18 原始试次中导出的总体 / 分实验室比率（复现已发表数值，
误差 < 1e-9）。

**choices13k 未随仓库提供**（CC BY-NC-SA 4.0）。请让阶段 00 指向你本地的副本：

```bash
python3 code/scripts/00_prepare_data.py --c13k-dir /path/to/choices13k-main
```

**完整的 CPC18 原始发布数据也未随仓库提供**（Zenodo 记录 2571510）。它仅在
GameID 211–270 时需要——即构成留出迁移目标 `cpc18_comp_desc` /
`cpc18_comp_exp` 的 48 个非歧义竞赛问题，这些问题不出现在任何其他公开文件中。
没有它，流水线照常运行，那两列仅用于评估的列会直接从迁移矩阵中缺席：

```bash
export CPC18_RAW="/path/to/all CPC18 raw data.csv"
```

`data/external/ATTRIBUTION.md` 还记录了为什么 CPC15（Zenodo 321652）*没有*被
作为一个独立来源加入：它是 CPC18 原始数据 Sets 1–5 的换名版本，已在试次层面
验证为逐字节相同。

## 复现

```bash
C13K_DIR=/path/to/choices13k-main bash run_all.sh            # full pipeline, ~2.5-3.5 h
C13K_DIR=/path/to/choices13k-main QUICK=1 bash run_all.sh    # ~30 min smoke version
```

`run_all.sh` 使用 `python3`；可用 `PYTHON=/path/to/python` 覆盖。
`results/runs/` 按训练单元缓存，因此中断的运行可以续跑，只补上缺失的单元格。

下面的耗时仅供参考，是在一台 Apple 芯片 CPU 上测得的；仓库中没有提交任何计时
产物，所以请把它们当作数量级来看待。

| stage | script | what it does | approx. time |
|---|---|---|---|
| 00 | `00_prepare_data.py` | 构建全部数据臂 + 各数据臂的噪声下限 | ~1 min |
| — | `code/tests/*`, `shapcode/test_shapley.py` | CPT numpy/torch 一致性、赌局展开、Shapley 公理 | seconds |
| 01 | `01_run_grid.py` × 9 suites | 训练全部 2,240 个单元；可续跑 | ~1–1.5 h |
| 02 | `02_probe_models.py` | 探针电池 → 行为指数 | ~1 min |
| 03 | `03_fit_theories.py` | 每个模型的 CPT 坐标（L-BFGS，8 个初值）+ 可辨识性 | ~3.5 s/model |
| 04 | `04_transfer_matrix.py` | 迁移矩阵 + 完备度归一化 | seconds |
| 05 | `05_symbolic_distill.py` | 对 CE 曲面的符号蒸馏 | ~22 s/model |
| 06 | `06_analyze.py` | ω² 分解、PERMANOVA、人为基线、自助法、对照 | ~2 min |
| 07–08 | `07_figures.py`, `08_paper_outputs.py` | 工作图、稿件表格 | ~30 s |
| — | `shapcode/run_decomposition.py`, `run_figures.py` | 归因读出量及其图 | minutes |
| 10 | `10_main_figures.py` | 组装 `paper_results/` | seconds |

阶段 09（`09_lr_calibration.py`）有意不包含在默认运行中：经过标定的学习率已固定
写入 `code/src/models.ARCH_LR`，报告的结果正是使用这些学习率。仅在更改架构或
数据臂之后才需要重新运行它。

## 网格

**九个套件，合计 2,240 个训练单元。**每个套件都把它的数据臂与全部 8 种架构
相交叉；随机种子数因套件而异。主网格是 4 个数据来源臂 × 8 种架构 × 10 个种子
= 320 个单元，其余八个套件（`synthetic` 160、`experience_gradient` 360、
`paired` 80、`population` 80、`size_control` 80、`reweight_control` 40、
`controlled` 640、`sufficiency` 480）提供各项对照、真值数据臂以及人为基线。
这些套件定义在 `code/scripts/01_run_grid.py` 中；它们所取用的 57 个数据臂由
`code/src/data.py` 构建，并汇总在 `paper_results/tables/TABLES.md` 的表 1 中。

参数量跨越五个数量级，所以“架构并不决定理论”并不是一个只在小模型之间检验过的
主张：

| architecture | parameters | inductive bias |
|---|---:|---|
| `CPT_NET` | 6 | 可微分 CPT；理论即模型 |
| `LINEAR` | 56 | 在展平特征上的线性模型 |
| `MLP_S` | 7,809 | 在展平特征上的 MLP |
| `SET` | 16,897 | DeepSets，对结果槽位置换不变 |
| `TRANSFORMER` | 75,649 | 跨结果槽位的自注意力 |
| `MLP_L` | 211,969 | 更深的 MLP |
| `RESMLP` | 2,134,529 | 宽残差 MLP，作为规模对照 |
| `GBT` | — | 梯度提升树，非神经网络基线 |

**学习率按架构分别标定**（`09_lr_calibration.py` →
`code/src/models.ARCH_LR`），且使用与所报告训练相同的 600 轮预算。这是一个
公平性前提条件，而非调参细节：在单一共享学习率下，SET 与 MLP_L 都远离各自的
最优点，而 RESMLP 在该共享学习率下坍缩为一个常数（在 3e-3 处验证损失为 4.212，
而在其选定的 3e-4 处为 0.419；`results/tables/lr_calibration.csv`）。标定必须
使用最终的训练预算，因为学习率与早停是耦合的——在短预算下选出的学习率，并不是
在完整预算下最优的那个。

## 可辨识性筛选（引用任何 CPT 坐标前请先读这一节）

在 182 个问题的 CPC18 数据臂上，某些架构在整个探针电池上的表现近乎恒定
（`probe_sd` 低至 0.0003）。此时六个 CPT 坐标完全不受约束，优化器只是把参数推到
其箱型边界上。把这样的“坐标”送进方差分解，分解的是噪声。因此阶段 03 为每次运行
记录两个判据：

- **`identifiable`**——`probe_sd ≥ 0.02`，即该网络对电池至少有所反应。用于
  **行为指数**，这些指数直接由预测计算得到，不涉及任何 CPT 拟合。主网格保留
  **309 / 320**。
- **`theory_identifiable`**——在上述条件之外，*并且*没有参数被钉在箱型边界上。
  用于 **CPT 坐标**。主网格保留 **276 / 320**。

当某个单元格中存活的运行少于 3 次时，阶段 06 会整列丢弃该架构，并记录在
`h1_variance_decomposition.csv` 的 `dropped_archs` 列中。按此规则，当前的
CPT 坐标分析丢弃了 **LINEAR**（在 `cpc18_desc` 上存活 1/10，在 `cpc18_exp` 上
存活 0/10）。这会从架构一侧移走方差，稿件中已作说明。

## 参数可恢复性

`build_sufficiency_arms()` 检验了每一项基于 CPT 的分析所依赖的假设：在生成主体
为已知真值的条件下，只改变问题数量（60 → 1000）和每个问题的应答人数
（8 → 128），并测量恢复误差（20 个数据臂 × 8 种架构 × 3 个种子）。六个坐标上的
平均相对误差始终处于 0.14–0.32 区间，且**在任何应答人数下都没有任何一个扫描到的
设计达到 10 % 的误差**（`results/tables/sufficiency_error.csv`、
`sufficiency_threshold.csv`）。τ 和 β 最差：τ 与价值尺度只能作为一个乘积被识别，
而 β 在损失一侧与 λ 共线。更多数据也修不好这两者中的任何一个。

这正是为什么 CPT 坐标只被当作四个读出量族之一来报告，也是为什么行为指数——由
预测直接计算、无需拟合——以及归因读出量才是承载论证的部分。

## 受控析因实验：数据来源的哪一个成分在起作用

把 choices13k 与 CPC18 做观察性比较，会同时混淆问题池、每问题应答人数、数据臂
规模与实验协议，而且真实来源只有两个相互独立的。`build_controlled_arms()` 针对
一个已知的生成主体运行了一个 2⁴ 完全析因设计，所有因素交叉且不混淆
（16 个合成数据臂 × 8 种架构 × 5 个种子 = 640 个单元；
`results/tables/controlled_factorial.csv`）：

| factor | levels | median ω² |
|---|---|---:|
| architecture | 8 architectures | 0.768 |
| **generating agent** | Prelec γ = 0.65 vs γ = 1.0 | **0.467** |
| number of problems | 182 vs 1500 | 0.209 |
| problem pool | c13k vs CPC18 distribution | 0.062 |
| respondents per problem | 16 vs 100 | 0.005 |

在“数据来源”的四个成分中，**只有应答者行为上的真实差异具有很大的效应**。问题池
（0.062）和汇总噪声（0.005）基本上什么都没做。因此，当在 choices13k 上训练的
模型与在 CPC18 上训练的模型编码出不同的理论时，原因并不是问题或噪声不同——这与
观察性的子抽样 / 重加权对照独立地收敛到同一结论。这里的架构份额（0.768）同样是
那个未经校正的量，而基线分析的存在正是为了校正它。

支持性证据：在 `rare_event_index` 上，主体因素的 ω² 达到 0.859——探针把设计上
注入的 γ 差异读了回来——而两个问题池给出近乎平行的曲线，因此问题池并不调节主体
效应。

## 描述—经验梯度：在两项独立设计上的复制

经验被从一个二元标签扩展为一个有序因素。CPC18 的五个区块是同一批应答者在同样的
182 个非歧义问题上、处于五个反馈阶段的数据（问题内设计）；choices13k 则把每个
问题恰好指派到一个区块，从而给出一个问题间对照，每个水平约有 2,450 个问题。这
两项设计不共享任何问题、应答者或国家。趋势为架构内的 Spearman ρ 再取平均；
*p* 来自在架构内置换区块标签（`results/tables/experience_trend.csv`）：

| read-out | c13k ρ | CPC18 ρ | note |
|---|---|---|---|
| `rare_event_index` | −0.258 (p = .002) | −0.376 (p < .001) | 同号，CPC18 上 **8/8 种架构** |
| `gamma_neg` | +0.321 (p < .001) | +0.536 (p < .001) | 同号，两者均显著 |
| `reflection_index` | −0.178 (p = .025) | −0.190 (p = .010) | 同号，两者均显著 |
| `gamma_pos` | −0.184 (p = .024) | +0.463 (p < .001) | **反号**，两者均显著 |

11 个读出量中有 5 个在符号上一致。`gamma_neg` 在两个来源中都具有最大的 ρ，但
最*一致*的却是罕见事件指数随累积经验的单调下降：它是唯一一个在 CPC18 上
**全部 8 种**架构中趋势同号的读出量。这是关于描述—经验差距如何*形成*的直接证据，
并且在两项毫无共享之处的设计上得到了复制。`gamma_pos` 的反转是一处真实的
不一致，并且被如实报告为不一致。

## 其他结果

- **H3，描述—经验符号翻转**在 1,266 个匹配问题上于 **8 种架构中的 5 种**成立
  （`results/tables/h3_de_flip.csv`）。
- **归因读出量在不假设任何函数形式的情况下复制了核心结果。**在 EV 差特征组上，
  数据来源达到 ω² = 0.881，对应的人为基线为 0.019
  （`results/tables/attribution_decomposition.csv`）。
- **H4，迁移**具有强烈的不对称性（`results/tables/transfer.csv`，完备度中位数）：
  在同一数据来源内部，choices13k 训练的模型为 0.84，CPC18 训练的为 0.40；跨数据
  来源则分别为 0.67 和 0.29。在 48 个真正未见过的竞赛问题上，它坍缩至 0.47 与
  −0.03。这种不对称性与 CPC18 只有 182 个问题相混淆，必须与 `size_control`
  数据臂一起解读。
- **规模不是解释。**把 choices13k 子抽样到 CPC18 的 182 个问题并不能复现 CPC18
  的坐标，把其特征向 CPC18 重加权也不能：三个对照数据臂落在 λ = 1.18 / 1.22 /
  0.98 与 τ = 5.28 / 6.57 / 7.84（`results/tables/controls_summary.csv`，
  表 9）。可与 `results/tables/theory_coords.csv` 中逐次运行的 CPC18 坐标对比。
- **一把衡量群体效应的尺子**：同样的问题、同样的协议、不同的实验室——τ 在
  Technion 与 Rehovot 之间相差约 30 %（`results/tables/population_arm.csv`）。

## 结果 → 稿件

以下全部内容都已提交，因此读者无需重跑任何东西即可核对稿件：

| file | what it supports |
|---|---|
| `results/summary.md` | 行文中的每一个核心数字，由阶段 06 重新生成 |
| `results/tables/h1_variance_decomposition.csv`, `h1_multivariate.json` | H1：ω²、PERMANOVA、轮廓系数 |
| `results/tables/artifact_corrected.csv`, `artifact_bootstrap.csv` | 人为基线与自助法区间——图 2 |
| `results/tables/identifiability.csv`, `h1_threshold_sensitivity.csv` | 筛选保留率与阈值敏感性 |
| `results/tables/h2_synthetic_recovery.csv` | H2：恢复偏差，同一主体在两种问题分布上 |
| `results/tables/h3_de_flip.csv` | H3：描述 / 经验符号翻转 |
| `results/tables/transfer.csv` | H4：迁移矩阵 |
| `results/tables/controlled_factorial.csv` | 2⁴ 受控实验 |
| `results/tables/experience_trend.csv` | 经验梯度 |
| `results/tables/sufficiency_error.csv`, `sufficiency_threshold.csv` | 参数可恢复性 |
| `results/tables/attribution_*.csv` | Shapley 归因读出量 |
| `paper_results/` | 投稿时的 6 幅主图、8 幅补充图与 20 个表 |

逐次运行的记录表也已提交（`results/tables/theory_coords.csv`、
`indices.csv`、`probe_level_omega2.csv`、`sr_expressions.csv` 以及
`results/runs_*.csv`，约 3.5 MB）：每个训练好的模型一行，携带其拟合得到的 CPT
坐标、行为指数、蒸馏出的表达式与各项指标。它们是阶段 06 的输入，因此整套分析
无需重新训练网格即可重跑与核对。

## 图与表

`paper_results/` 由 `code/scripts/10_main_figures.py` 组装，并在每次运行时被
整体覆盖——请勿手工编辑。图的编号存放在该脚本的 `MAIN_FIGS` / `SUPP_FIGS`
列表中；把某幅图在正文与补充材料之间挪动，只需编辑这两个列表，别的都不用改。
每幅图都以 600 dpi PNG 以及嵌入字体的矢量 PDF 形式发布
（`pdf.fonttype=42`）。`paper_results/README.md` 与
`paper_results/tables/TABLES.md` 载有带图注的完整清单。

## 许可、数据条款与引用

**代码**采用 MIT 许可（见 `LICENSE`），它覆盖本仓库中的一切，
**但不包括** `data/external/` 下的第三方数据。

**数据不在该许可的覆盖范围内。**choices13k 为 CC BY-NC-SA 4.0——不得商用、
需相同方式共享——而 CPC18 材料受其作者的条款约束。
`data/external/ATTRIBUTION.md` 记录了每个数据集的来源、许可与必需的引用；
用户有责任遵守这些条款。

如果你使用本仓库，请引用论文（见 `CITATION.cff`）以及数据集：

- choices13k —— Peterson, Bourgin, Agrawal, Reichman & Griffiths (2021),
  *Science* **372**:1209–1214.
- CPC18 —— Plonsky, Erev et al., CPC18 white paper; Erev, Ert, Plonsky, Cohen &
  Cohen (2017), *Psychological Review*.
