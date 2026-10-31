# FNQS universality project

目标是让条件波函数覆盖多个 microscopic Hamiltonians，再从波函数提取普适数据及 coupling-space geometry。当前主路线改为每个尺寸分别训练、在该尺寸内共享耦合；对独立尺寸的物理测量做有限尺寸标度。跨尺寸共享保留为对照。分类标签不是训练目标。

已完成 Phase 0 框架、精确参考、优化修复及 **v18 冻结方案后的独立确认**：三个新冷 seeds 在固定 L16/L24/L32 完整重复 legacy width16、energy-only 方案；全部九个端点封存后测原保留的40个磁场。L16 的 χ 最坏误差 **8.03%／9.88%／6.85%** 均通过10%门槛；L24 为 **12.56%／17.54%／15.81%**，L32 为 **29.87%／22.98%／22.08%**，后两者包含2SE并全部失败。九个模型的全部 C(r) 和能量通过。自由 pooled 拟合 hc=.9816、ν=1.234、η=.170，窗口／拟合形式不稳定，峰位也存在种子偏差；**Phase 0 科学 gate=false**。原始数据、独立重放与实际成本见 [progression_v18](results/progression_v18.json)，[响应曲线](results/phase0_frozen_confirmation_v18/confirmation_profiles.png) 和 [指数／峰位图](results/phase0_frozen_confirmation_v18/scaling_and_peak_stability.png) 可直接检查。完整记录见 [RESULTS.md](RESULTS.md)，文献对应见 [LITERATURE.md](LITERATURE.md)，路线见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)。

Phase 1 的固定 L12 A/B/C 小预算训练和 Phase 2 的 L6 三折工程 canary 已执行；本轮 [独立复核](results/phase_checks_v18d/receipt.json) 通过物理量、来源、祖先持出和实际预算检查。Phase 1 仍缺较大尺寸 B/C 独立参考、响应精度和冻结确认；Phase 2 尚未完成科学 transfer 或含预训练成本的精度优势验证。

下一轮固定L32的预算／步长对照已按[新协议](results/phase0_l32_budget_rate_v19r/protocol.json)启动：三个父模型，lr=2/4/8，连续轨迹的新增100／400步端点；特征、width16和energy-only保持不变。上轮交互中断留下的partial保留，新目录完整重跑，并用后台supervisor记录退出状态。目前尚无完整v19结果；基线与所有新端点将使用相同审计预算，已打开的v18字段只作development。

按用户“相差一个量级内可考虑推进”的新规则，[门槛距离](results/gate_distance_v10/assessment.json) 支持上述探索性工作。原精度门槛与旧封存结果保持不变，v11–v14 没有使用保留的 confirmation fields；较大尺寸 B/C 独立参考与 class assignment 仍待完成。

当前 FNQS **169项检查通过**；最新全仓2927 passed／6 failed／2 skipped，失败项独立复查4 passed／2 failed，持续失败仍为既有外部 artifact 来源与 seed 冲突。[验证记录](results/test_validation_v18.json)、[完整日志](results/full_suite_v18.log) 与 [独立复查](results/failure_recheck_v18.log) 保留实际失败，全仓不是全绿。

## 最新辨因与已执行修复（v12–v14）

固定 L12、width8、no_explicit_size，三个配对冷 seeds=113/127/139。v12 的 2×2 使用五个训练点、窗口中心±.04／±.15 和 150／600 单点 SR 更新；两个预算是同轨迹端点，成本不重复相加。主比较始终为同四个中心 development fields，更宽区域另报。窄窗 150 步参数逐项复现 v11，最大差为 0。

v13 将同一宽窗的 600 单点 updates 改为 120 次五点 ensemble SR；每 task 64 samples，共 600 tasks／38400 Born draws，SR 超参数和模型不变。三 seeds 的 χ 改善均超过预声明 5 个百分点，并同时降低全部 C(r) 的最大误差，因此启动 v14 的 A/B/C 冷复验。更新次数、每次批量噪声和 SR 条件化随组织方式一起改变，该对照没有唯一隔离其中某个机制。

共享 A/B/C 每次联合全部 15 个 fields，各族独立模型每次联合其五个 fields，均 120 updates、每族 600 tasks。总 H／Born 成本匹配，总参数量与独立模型更新次数不匹配，不能据此声称 foundation 计算优势。该方案复用已有、每个 Hamiltonian 分别中心化的 ensemble SR；没有改写旧训练或数学约定。

复现时使用新目录：

```bash
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_coupling_factorial prepare --output /tmp/fnqs-factorial
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_coupling_factorial run --output /tmp/fnqs-factorial
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_joint_field_probe prepare --parent /tmp/fnqs-factorial --output /tmp/fnqs-joint
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_joint_field_probe run --output /tmp/fnqs-joint
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_joint_field_probe prepare-families --parent /tmp/fnqs-joint --output /tmp/fnqs-families
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_joint_field_probe run --output /tmp/fnqs-families
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.review_coupling_remedies --factorial /tmp/fnqs-factorial --probe /tmp/fnqs-joint --families /tmp/fnqs-families
```

若 joint 对照不满足三 seed 条件，`prepare-families` 会拒绝启动。检查点、原始穷举数据、独立参考、source snapshots 与实际成本分别保留。L12 的实测收益见 [结果图](results/joint_family_solution_v14/remedy_comparison.png)；随后执行的 L16 配对没有复现三个 seeds 一致的收益，须分别保留两项结果。

## 固定 L16 配对及优化修复（v15–v17）

三个独立预训练父模型 seed=71/83/97。v15 比较 1050 逐场 updates 与 50 次全 21 场联合 updates，曝光顺序及 total H／Born 成本相同。v16 使用等价的可逆参数坐标，初始物理波函数及 Γ 导数不变；50 步是同预算坐标对照，200 步另含更多优化预算。v17 从各自的 v16 step200 父检查点及其终端 PRNG 分叉，对照学习率 .5／4，每支 100 次联合更新，其余设置一致。每批全部端点先 seal，再穷举 65536 配置、全部 C(r) 和固定 θ 的 χ，使用独立 JW 与稀疏 Pauli 重放。原 40 点 confirmation fields 保留，不作开发选择。

v15–v17 保持 1474 nominal 参数，未加宽或增加响应损失。学习率对照支持优化收敛会影响 χ，不证明容量已足够覆盖其他尺寸或更宽窗口。三个 warm 初态来自独立预训练 seeds，但它们参与了方法选择。v18 已另用三个冷初始化完成冻结确认，复现了 L16 达标，同时暴露 L24/L32 响应失败；单尺寸 grid peak 不作为热力学 hc。

复现时使用新目录，依次运行：

```bash
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_l16_pair prepare --output /tmp/fnqs-l16-pair
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_l16_pair run --output /tmp/fnqs-l16-pair
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_coordinate_repair prepare --parent /tmp/fnqs-l16-pair --output /tmp/fnqs-l16-coordinates
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_coordinate_repair run --output /tmp/fnqs-l16-coordinates
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_step_pair prepare --parent /tmp/fnqs-l16-coordinates --output /tmp/fnqs-l16-steps
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_step_pair run --output /tmp/fnqs-l16-steps
```

后续修复的 prepare 会在上一批三个 seeds 已全通过时拒绝启动。条件扩展使用新协议，不增加旧协议预算或覆盖旧检查点。ED/JW observable labels 没有作为监督目标进入训练。

## 冻结确认与稳定性核验（v18）

复现时使用新目录，先声明全部方案及响应补充，再训练和测量：

```bash
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_frozen_confirmation prepare --output /tmp/fnqs-confirmation
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.check_frozen_responses prepare --output /tmp/fnqs-confirmation
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_frozen_confirmation run --output /tmp/fnqs-confirmation
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.check_frozen_responses run --output /tmp/fnqs-confirmation
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.review_frozen_confirmation --output /tmp/fnqs-confirmation
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.analyze_frozen_confirmation --output /tmp/fnqs-confirmation
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.plot_confirmation_stability --output /tmp/fnqs-confirmation
```

每端点完整1300次更新；总训练11700 updates、5932800 Born draws，独立测量另计。χ 导数固定 θ，分别报告 MC 误差、初始化差异和拟合形式敏感性。自由 η 在自由拟合的 hc 处提取，不用已知 hc=1 替换失败拟合。200次 paired-seed／条件 MC bootstrap 只有三个初始化，不能覆盖共同系统误差。三个尺寸不足以完成删去最小 L 后的三尺寸检验。v18 保留字段已打开，后续方法开发若使用这些数据，应另声明未见过的确认字段与初始化。

## 快速运行

在仓库根目录执行，现有 `nqs` conda 环境已具备依赖：

```bash
conda run -n nqs python -m pytest fnqs/tests
conda run -n nqs python -m fnqs reference --config fnqs/configs/phase0_reference.json --output fnqs/results/phase0_reference
conda run -n nqs python -m fnqs train --config fnqs/configs/phase0_smoke.json --output /tmp/fnqs-smoke
conda run -n nqs python -m fnqs recon --config fnqs/configs/phase1_recon.json --output /tmp/fnqs-recon
```

全仓 artifact 检查会比较运行环境的字面值，因此全仓回归使用 `JAX_PLATFORM_NAME=cpu conda run -n nqs python -m pytest`。

新环境可从根目录安装 `python -m pip install -e '.[fnqs,test]'`。核心神经模型与 Adam 不依赖 Optax；SciPy 用于 FNQS 精确参考、稀疏 ED 和拟合，Matplotlib 用于导出图。

VMC 管线短测与正式研究配置分别为：

```bash
JAX_PLATFORM_NAME=cpu python -m fnqs train --config fnqs/configs/phase0_vmc_canary.json --output /tmp/fnqs-vmc-canary
JAX_PLATFORM_NAME=cpu python -m fnqs train --config fnqs/configs/phase0_vmc.json --output /tmp/fnqs-vmc
```

正式配置有 3 seeds、5 个训练尺寸、一个未见尺寸 (L=40)、held-out fields 和独立测量。它是预算起点，不是已验证能够收敛的配置；其较大计算未在初次开发中执行。已存在的训练 checkpoint 不会被覆盖。继续同一实验时加 `--resume`：每次执行增加 `config.steps` 次更新，恢复参数、Adam moments、更新编号和 PRNG。数值源代码或训练签名改变会拒绝续训；更改审计样本数不改变训练轨迹。

## 模块边界

| 模块 | 内容 |
| --- | --- |
| `basis.py` | Hilbert labels、尺寸验证和确定性的 bit 枚举 |
| `hamiltonians.py` | Pauli super-Hamiltonian、唯一 connectivity、有限 ED |
| `exact.py` | 偶数 PBC TFIM Jordan–Wigner 能量、QGT、关联、Rényi 熵 |
| `models.py` | 纯 JAX、共享多尺寸参数、coupling-conditioned normalized GRU |
| `training.py` | 精确能量／独立采样 VMC，Adam／ensemble SR |
| `observables.py`, `entropy.py` | QGT covariance、关联、精确 S2、独立 replica SWAP |
| `scaling.py` | 自由 (h_c,\nu,\eta,c) 拟合、有限尺寸修正、验收门槛 |
| `subspace.py` | fidelity 选择 snapshots 和固定算符 Ritz 投影的小系统 control |
| `experiments.py`, `artifacts.py` | 运行／独立审计、JSON/CSV、原子 NPZ checkpoints、来源 hash |
| `initialization.py` | 显式参数 warm start；重置优化器／PRNG，保留父 checkpoint 与成本 |
| `controls.py` | 固定尺寸／共享尺寸、Adam／SR、稀疏／密集磁场的预声明对照 |
| `sectors.py` | N≤14 的独立 Z2 奇偶扇区、谱隙与 projected-response fidelity susceptibility |
| `staging.py` | 从物理 couplings 检查整 family 持出；区分参考／协议开发与科学 gate |
| `diagnostics/response.py` | 固定参数 JVP／FD、投影物理响应残差、HF 缺陷、独立谱扰动参考 |
| `diagnostics/finite_chain.py` | 明确限制 L≤16 的 full-space Pauli action 与穷举 C(r) 审计 |
| `diagnostics/confirmation.py`, `frozen_stability.py` | 独立采样误差、χ/C 联合 covariance、自由关联指数与局部峰位检查 |
| `variants/size_features.py` | 原特征、去显式 L、完全去长度特征的 normalized GRU；独立 API，须冷初始化 |
| `variants/coupling_units.py`, `pilots/coordinate_sr.py` | 可逆 GRU 参数坐标、物理 Γ 导数及 energy-only ensemble SR |
| `pilots/engine.py`, `measurements.py`, `lineage.py` | 使用独立特征 API 的固定 L SR、物理响应损失、审计、传递祖先与成本去重 |

## 输入导数与尺寸路线（v9）

[135 点诊断](results/coupling_response_diagnostics_v9/diagnosis.json) 区分 h 输入的 AD/FD 一致性、物理 response equation 和 HF 缺陷；[图](results/coupling_response_diagnostics_v9/response_diagnosis.png) 展示三 seed 范围。阶段判断见 [progression_v9.json](results/progression_v9.json)。复查原始保存行，不更新参数或增加采样：

```bash
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.verify_response_diagnostics
```

尺寸特征模型采用独立 `SizeFeatureConfig` 和 `fnqs.variants.size_features` API；当前 `fnqs train` 不支持该配置，传给旧模型会明确失败，避免静默忽略 policy。v9 时尚未训练该模型；v11–v14 已通过独立 pilot 运行器完成固定 L12 的 no_explicit_size 冷训练。三种 feature policies 的公平配对比较仍未完成；不要求单一模型同时覆盖不同 L。

## 响应精度改进对照

```bash
JAX_PLATFORM_NAME=cpu conda run -n nqs python -m fnqs.controls --output /tmp/fnqs-controls --steps 1200
```

六组对照均使用宽度 16、相同初始化 seed 和每步五个 Hamiltonian 的主梯度预算。固定 L=8 的五磁场点／21 磁场点分别比较 Adam 与 SR；共享 L=4,6,8 在密集磁场上再比较两者。密集 ensemble 每步不放回均匀抽取五个 task，PRNG 及工作计数随 checkpoint 恢复；没有把每一步的计算直接扩大到全部磁场。

SR 使用每个 Hamiltonian 分别中心化的参数 score、带对角预条件的 CG、参数更新范数限制和精确能量回溯。CG 的实际 relative residual 必须 ≤`max(1e-7, 10*sr_tolerance)`，否则报错；回溯产生的额外 Hamiltonian 查询单独记录。因此这是相同主梯度预算下的方法对照，不能宣称相同总成本的 foundation 优势。运行日志记录梯度范数、方差、求解 residual、有效步长和累计计算量。

`protocol.json` 在训练前声明 development 和独立 confirmation 磁场网格。开发网格可以指导方法选择；后续确认必须使用保留网格与新 seeds。小尺寸 observable 达标仍不解锁 universality gate。

本轮已运行六组 1200 步对照、共享 SR 的追加 1200 步、50 步完整 ensemble 精修，以及三 seed 的 200 步研究尺寸 VMC pilot。保留了各阶段失败记录与 [响应变化图](results/response_progress_v3.png)。完整 ensemble 精修同时降低了 SR shift，因此其改善不能单独归因于 task batching。

显式 warm start 使用配置中的 `warm_start_checkpoint`，只导入模型参数；`--resume` 继续恢复原优化器和随机数，并要求所有数值源代码 hash 一致。warm start 验证模型、basis 与 Hamiltonian 语义，记录父参数 hash、训练成本和新随机轨迹；从同一父参数生成多个 seeds 不是独立预训练初始化。以下是原始 refinement 预设；实际配对批次使用单独封存的新确认网格：

```bash
JAX_PLATFORM_NAME=cpu conda run -n nqs python -m fnqs train --config fnqs/configs/phase0_vmc_sr_warm_refinement.json --output /tmp/fnqs-warm-refinement
JAX_PLATFORM_NAME=cpu conda run -n nqs python -m fnqs train --config fnqs/configs/phase0_vmc_sr_scratch_refinement.json --output /tmp/fnqs-scratch-refinement
```

第二组采用相同后续训练预算的随机初始化。原预设文件保持不变；实际执行配置以 [v6 配对协议](results/phase0_pair_v6/protocol.json) 为准，v5 中断结果另存。warm-start 比较须加入已记录的预训练成本。

`results/runtime_v3/` 保留早期实验的数值源码快照及 manifest。需要严格续训旧 v3 checkpoint 时，从该目录运行 Python 并使用配置与输出的绝对路径。v4 包含 unsigned-bit local-energy 修复和新初始化接口；旧结果没有改写来源 hash。

这是一条独立的 Z2 Ising 研究路线。TFIM 不具有 SU(2)，所以没有将它错误投影到原项目的 singlet multiplicity space；没有更改 `su2_nqs` 的数学约定、模型或已有研究结果。根 `pyproject.toml` 仅新增 FNQS 可选依赖、包发现和测试入口。

## 固定约定

\[
H=-J_1\sum_i Z_i Z_{i+1}-J_2\sum_i Z_i Z_{i+2}-J_x\sum_i X_i X_{i+1}-h\sum_i X_i.
\]

算符是 **Pauli** (X,Z)，PBC，(J_1=1) 设置能量尺度，TFIM 临界场是 (h=1)。bit 0 对应 (Z=-1)，bit 1 对应 (Z=+1)，site 0 是 MSB。所有求和包含全部 (L) 个起点；(L=4) 的次近邻项会按此定义重复无向 bond。

耦合顺序固定为 `(j1,j2,jx,h)`。浮点数仅用于耦合和物理量；没有半整数 Hilbert labels。新参考限制偶数 (L\ge4)，TFIM 参考要求 (J_1,h>0)；精确 ED 限 (L\le14)，dense Hamiltonian／精确训练限 (L\le10)。

GRU 是可审计的第一版 baseline，**不是论文 ViT 的原样复现**。它使用位置比例、`log(L)`、`1/L` 等连续特征而非每个尺寸独立 embedding。自回归的两个 bit actions 为 `(0,1)`。先计算归一化 (p_{AR})，再定义

\[
p(b)=\frac{p_{AR}(b)+p_{AR}(1-b)}2,\qquad \log\psi=\frac12\log p+i\,0.
\]

这样精确保留 Z2；采样是 AR 独立生成加一个独立全局 flip coin。该网络没有强制平移对称，也没有 phase head，因此训练只允许 (h>0,J_x\ge0) 的 stoquastic cuts。signed couplings 仍可用 ED 检查。

QGT 使用固定训练参数下的输入导数：(O_i=\partial_{\gamma_i}\log\psi\)，(Q_{ij}=\langle O_i^*O_j\rangle-\langle O_i^*\rangle\langle O_j\rangle)。记录总量 (chi_F=\operatorname{Re}Q_{hh})，并另给每格点量。没有把参数空间 SR metric 和 coupling QGT 混用。

Rényi-2 使用自然对数；PBC 拟合系数为 (c/4)，von Neumann 才是 (c/3)。SWAP 函数保留 raw mean 与 SE，未自动接入大尺寸审计；默认大尺寸报告暂不提取 (c)。

## 实验与验收

训练最小化等权 Hamiltonian ensemble 的 **每格点能量**，使不同 (L) 权重可比较。SR 在每个 Hamiltonian 内分别中心化 score 和能量，再组合；不能在混合 ensemble 中只减一次 global mean。配置记录每个 task 样本数及总预算，比较 scratch/transfer 时要固定后者。

训练与评估的 field 集合必须互斥，未知配置字段拒绝执行。审计在训练结束后固定参数；VMC 使用独立 folded PRNG draws，关联给出独立样本 SE，(chi_F) 给出 bootstrap SE。checkpoint 使用 `allow_pickle=False`，JSON 拒绝 NaN/Inf。

参考拟合包含自由 (h_c,\nu)、多项式 scaling curve 和 (1/L) correction；提供 degree／去掉最小尺寸的敏感性结果。图中实线移除拟合 correction，淡虚线保留原始 scaled data。拟合 residual **不是**统计误差；不得把输出数字直接当作置信区间。

Phase 0 科学 gate 要求 (h_c,\nu,\eta) 通过计划阈值、至少 3 个研究尺寸与 3 seeds，并且每个 seed 的独立 observable audit 通过。可观测量工程阈值暂定：energy relative error ≤0.002、(chi_F) relative error ≤0.1、最大关联 absolute error ≤0.03。门槛不自动放宽；即使 small-size energy 很好，也不能解锁 universality claim。

`phase1_recon` 是有限尺寸筛选，输出磁化平方、远距关联、S2 和 finite-difference fidelity peaks/brackets；它没有为 Family B/C 假定已知 (h_c) 或 Ising 指数。Hamiltonian tokens 目前仅为 metadata，实际模型输入仍是固定四维耦合向量，不代表 unseen operator content 的泛化。

## 当前状态与下一步（2026-10-02）

v5 配对实验在 scratch seed 29 第 225 步因真实 SR residual 超限而停止，旧 checkpoint 和 [故障系统](results/sr_failure_diagnostic_v5/failure.json) 完整保留。v6 加入至多三次 matrix-free 真实残差重求解，原停止阈值保持 1e-6；与独立 SVD 的方向最大差异 2.56e-9，容差 4e-9。每个完成 seed 独立保存审计，partial artifact 不能打开科学 gate。

[v6 配对批次](results/phase0_pair_v6/protocol.json) 已全部完成：各 3 seeds、1000 updates、6 tasks ×128 samples/update、8192 independent audit draws/point，采用新的 40 点 confirmation 网格。六个 checkpoint 的 hash／预算／测量网格 [核验通过](results/phase0_checkpoint_validation_v6.json)。两组仅后续预算相同，warm parent 的 15150 主梯度查询、30300 实际查询及 3361248 个 exact basis-state evaluations 须另计。原始测量汇总见 [progression_v6.json](results/progression_v6.json)，[诊断图](results/phase0_pair_v6/observables.png) 保留三个 seeds。

旧 collapse 的 nu-dependent residual normalization 会压低含噪拟合的 nu。v7 改为固定物理 chi SE 权重，独立真实 nu=0.9 的含噪控制最大偏差为 0.001688（容差 0.005）。[保存测量的重分析](results/phase0_reanalysis_v7/reanalysis.json) 没有新增训练或 draws，原始 v6 测量和 checkpoint 未改写；修正拟合仍不稳定，科学 gate 为 false。最终 [阶段汇总](results/progression_v7.json) 分开记录测量与提取源码。

v7 的 **83 项 FNQS 检查通过**。隔离副本全仓回归为 2719 passed、5 failed、2 skipped；失败项独立复查后 4 passed、1 failed，持续失败为未修改的历史 SR artifact 来源不匹配。合入源码与已运行全仓回归的副本逐字相同，验证见 [validation_v7.json](results/validation_v7.json)。下一项 Phase 0 开发工作是固定尺寸／共享尺寸对照，再分离追加优化与增大宽度的影响，详见研究计划。

Phase 1 已完成更精细的 [奇偶扇区参考](results/phase1_parity_reference_v5/reference.json) 和 [图](results/phase1_parity_reference_v5/reference.png)：L=6,8,10,12,14，磁场步长 0.005，1230 个 grid points 加 120 个 finite-difference response audit points。该结果是独立有限尺寸参考，不是 thermodynamic hc／class assignment。复现命令：

```bash
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.refine_phase1_references --input fnqs/results/phase1_recon_refined_v2/reconnaissance.json --output /tmp/fnqs-parity-reference
```

Phase 2 已准备 [三折协议](results/phase2_protocol_v5/protocol.json) 与物理 coupling 的 holdout 检查，包括 frozen zero-shot、1/2/4/8 个 coupling points、scratch 和 pointwise NQS。协议禁止使用已见过 held family 的任何参数祖先；持出 A 时不能沿用 TFIM A checkpoint，A+B+C 的 Phase 1 checkpoint 也不能直接用于任一整 family 持出。完整祖先检查和成本调度仍须落实到未来训练运行器。Phase 0／1 的 reproduction 和较大尺寸 B/C 独立参考未通过前，不运行 Phase 2 transfer 或解释其优势。

v5 复用训练／审计 kernel 的编译，参数和 couplings 为动态输入。旧数值版本分别保存在 `results/runtime_v4/`、`results/runtime_v5/` 和 [runtime_v6](results/runtime_v6/manifest.json)；当前版本封存在 [runtime_v7](results/runtime_v7/manifest.json)。v6 checkpoint 不能在 v7 代码下 strict resume，须使用原冻结版本；参数 warm start 保留祖先成本并重置优化器和 PRNG。关联接口也修复了 uint8 的 Z 值溢出，已有 AR 样本为 int32，原实验不受影响。

## 固定尺寸、优化预算与宽度对照（v8）

用户选择三个新冷初始化 seeds=71,83,97。[实验协议](results/size_capacity_controls_v8/protocol.json) 和每个 seed 的字段调度在训练前封存；**15 条训练轨迹、21 个审计端点现已全部完成并核验**。每个 L=16,24,32 每步恰好两个不同训练字段，各模型在前 1000 步看到相同的逐尺寸字段序列。固定尺寸宽度 16 分别训练 1000 步；共享宽度 16／32 分别保存同一轨迹的 1000、2000 步端点，审计不消耗训练 PRNG。

1000 步时固定三模型合计与各共享模型均为 6000 Hamiltonian 查询／768000 training Born draws。参数量分别为 fixed 合计 4422、shared16=1474、shared32=4482；宽度 32 比 fixed 合计多 60 参数。查询预算不等于线性代数或 wall-time 成本。宽度的影响也可能包含优化条件变化，不能直接称为纯表达能力效应。

[分析协议](results/size_capacity_controls_v8/analysis_protocol.json) 在首个审计前封存：主要比较 development fields 上最大 chi error+2SE，保留三 seed 配对差值、逐尺寸精度和收敛诊断。20 个新 development fields 与训练／历史 held grids 分离，另保留 40 个未测量 confirmation fields。此批不拟合指数、不打开 universality gate。数值库仍为原 v7；新增运行器的独立 hash 和库快照封存在 [runtime_v8](results/runtime_v8/manifest.json)。

新增 focused 检查 7 项、汇总／成本去重检查 4 项均通过，最终 FNQS 全套 **94 passed**：[fnqs_validation_v8_complete.log](results/fnqs_validation_v8_complete.log)。实际宽度的 [数值核验](results/size_capacity_numerical_validation_v8.json) 包含 L=4,6 归一化／Z2 和固定参数输入 QGT；AD／finite-difference 最大差异 3.31e-8（tolerance=5e-8），新旧审计最大差异 0（tolerance=1e-12）。

[完整比较](results/size_capacity_controls_v8/comparison.json)、[结果与配对表](RESULTS.md#固定尺寸预算与容量对照v8已完成) 和 [诊断图](results/size_capacity_controls_v8/comparison.png)：共享宽度 16 的收益依赖 seed；翻倍预算降低六条共享轨迹的 χ RMS 误差，但最坏点改善不稳定；宽度 32 在两个预算、全部三个 seeds 的 χ 最坏点／RMS 都劣于宽度 16。能量误差很小并未保证 coupling response 准确。所有端点组仍不满足 χ≤10%，Phase 0 科学 gate 未打开，正式 Phase 1／2 训练未启动。容量变化同时影响优化条件，本批不证明模型表达能力的上限。

实际训练合计 90000 Hamiltonian 查询、11520000 Born draws，独立审计 15114240 draws；共享轨迹的两个累计训练预算只计末端一次。额外 SR residual solves=6229，资源 profile 的 60 Hamiltonian／7680 draws 另计。源码、调度、检查点、审计和图的核验见 [validation_v8.json](results/validation_v8.json)。全仓回归 2754 passed／5 failed／2 skipped；独立复查仍有一项原项目历史 artifact 来源失败，详情见结果文档。

复现到新目录（原封存目录不覆盖；声明后不能改运行器／数值源码）：

```bash
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.size_capacity_controls --output /tmp/fnqs-size-controls --declare --seeds 71 83 97
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.analyze_size_capacity --directory /tmp/fnqs-size-controls --declare
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.size_capacity_controls --output /tmp/fnqs-size-controls --seeds 71 83 97
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.analyze_size_capacity --directory /tmp/fnqs-size-controls
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.plot_size_capacity --directory /tmp/fnqs-size-controls
```
