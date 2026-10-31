# 开发与实验记录（2026-10-01 起）

**v18 已完成冻结后的三新 seed、固定 L16/L24/L32 独立确认：L16 通过，L24/L32 的 χ 失败，自由指数及峰位可靠性不足；Phase 0 科学 reproduction gate 未通过。Phase 1/2 的独立物理量、来源、祖先和成本核验通过，科学阶段仍未完成。** 最新实测、稳定性及推进条件见文末与 [progression_v18](results/progression_v18.json)。下方各版本保留当时的结果与阶段判断，仍无 universality transfer、Transformer 或 RG discovery 结论。

所有 `*_v2` / `phase1_recon_refined_v2` 结果均记录实际配置、Python/NumPy/JAX 版本、CPU backend 和初次开发时的数值源代码 SHA256。本轮优化器改进没有重新封存旧结果；新旧实验各自保留实际版本。初次调试与依赖整理前的输出保留，以下初次开发记录引用其最终版本。正式三 seed、L=16–64 的 2000 步 VMC 配置未运行。

## Phase 0：独立精确校准

数据：[reference.json](results/phase0_reference_v2/reference.json)、[CSV](results/phase0_reference_v2/reference.csv)、[图](results/phase0_reference_v2/reference.png)。

偶数 PBC Pauli TFIM，L=16,24,32,48,64，每个尺寸 41 个 critical-window fields，使用 exact Jordan–Wigner oracle。(h_c,\nu) 自由拟合，Chebyshev scaling curve 加 (1/L) correction。

| L | fidelity peak h*(L) | eta（chord fit） | c（S2 slope） |
| --- | --- | --- | --- |
| 16 | 0.9778393 | 0.248304 | 0.524497 |
| 24 | 0.9899764 | 0.248925 | 0.516777 |
| 32 | 0.9943101 | 0.249204 | 0.512723 |
| 48 | 0.9974473 | 0.249654 | 0.508562 |
| 64 | 0.9985571 | 0.249808 | 0.506448 |

自由 collapse：**hc=1.0001983，nu=0.9993871**，relative RMS=1.4166e-4。改变 polynomial degree（4/6）和去掉最小尺寸后，hc 在 1.0001056–1.0001984，nu 在 0.9993865–0.9996051。该区间是拟合敏感性，不是统计置信区间。

按 TFIM peak 的 (1/L^2) 漂移 extrapolate 得 hc≈0.9998746；仅拟合 critical susceptibility 高度得到有效 nu≈0.983010，展示有限尺寸 correction 的作用。图中淡虚线为原始 scaled curves，实线移除拟合 (1/L) 项；不隐藏 correction。

独立 ED audit 覆盖 L=4,6,8,10，h=0.7,1.0,1.3，`float64/complex128`。绝对验收 tolerance=4e-11；最大观测差异：

| 项目 | 最大差异 |
| --- | --- |
| E0：JW vs ED | 1.42e-14 |
| C(r)：determinant vs computational-state expectation | 1.55e-15 |
| S2：Majorana spectrum vs Schmidt SVD | 2.58e-15 |
| ED eigenvector residual | 7.57e-14 |

另外以谱扰动 QGT 检查 fidelity susceptibility，并测试 (chi_F(1,L)=L(L-1)/32)；不是用同一求和重复构造“独立”参考。这些是 **精确解校准**，不是 FNQS 的测量成绩。

## Phase 0：共享神经态 smoke

数据：[neural.json](results/phase0_smoke_v2/neural.json)、[checkpoint](results/phase0_smoke_v2/seed_17.npz)。固定宽度 8 GRU，L=4,6,8，训练 fields=0.8,1.0,1.2，held fields=0.9,1.1；seed=17，300 次 Adam updates，learning rate=0.003，精确 ensemble energy/site。约 17.37s CPU wall（包含编译及审计）。

最大 exhaustive normalization error=4.44e-16，达到 1e-10 的约定。训练后 AD QGT 与 fixed-theta fidelity finite difference 的最大 absolute difference=1.68e-7（dh=1e-4，非 infinitesimal）。

held-field 平均 relative energy error=0.001714，最大=0.003075；全部点最大=0.004666。可是 held-field (chi_F) 最大 relative error=0.7083，全部点最大=0.8573，关联最大 absolute error=0.08497。

**科学 gate=false。** 小尺寸 energy 改善没有建立 critical derivative 的精度；未拟合或填入 (h_c,\nu,\eta)。这也说明当前框架能识别“energy 尚可但 universality observables 不合格”的情况。没有放宽门槛或通过 exact oracle 填补神经结果。

## Phase 0：研究尺寸 VMC canary

数据：[neural.json](results/phase0_vmc_canary_v2/neural.json)、[checkpoint](results/phase0_vmc_canary_v2/seed_17.npz)。L=16,24,32；训练 fields=0.98,1.0,1.02，held fields=0.99,1.01，seed=17，宽度 4，3 次 Adam updates，8 independent samples/task/update，9 tasks 即 72 samples/update；每点独立 audit=128 draws。约 10.00s CPU wall。

完成共享尺寸训练、全 Hamiltonian local energy、input QGT、关联、独立样本 SE 与 chi bootstrap SE 的端到端运行。该预算只验证管线，**没有收敛或 critical scaling 结论**；critical grid 分辨率不够，代码返回缺失 exponent，科学 gate=false。

## Phase 0：固定预算的响应精度 pilot

数据：[neural.json](results/phase0_response_pilot_v2/neural.json)、[checkpoint](results/phase0_response_pilot_v2/seed_17.npz)、[配置](configs/phase0_response_pilot.json)。L=4,6,8，宽度 16，5 个 training fields=0.8,0.9,1.0,1.1,1.2；4 个 held fields=0.85,0.95,1.05,1.15。seed=17，600 Adam updates，约 32.57s CPU wall，完成预先设定预算后没有追加训练。

held-field 平均 relative energy error 降到 **2.56e-4**，最大=5.54e-4；全部点最大=9.78e-4。可是 held-field chi 最大 relative error 仍为 **0.5587**，全部点=0.6913；关联最大 absolute error=0.03356。norm error=2.22e-16，AD/fidelity discrepancy 最大=1.36e-7。

科学 gate 仍为 false。这个 pilot 表明在当前 shared energy-only training 中，更小的 energy error 没有自动建立 coupling response accuracy。网络宽度、训练网格和更新预算同时改变，不能将差异归因于其中单一因素；后续需要相同架构、相同总预算的 optimizer/field-density controls。仍不提供小尺寸 critical exponents。

## Phase 1：ED reconnaissance

数据：[扩展扫描 JSON](results/phase1_recon_refined_v2/reconnaissance.json)、[图](results/phase1_recon_refined_v2/reconnaissance.png)。初始 h=0.7–1.8 扫描保留在 [phase1_recon_v1](results/phase1_recon_v1/reconnaissance.json)。发现 C 的 peak 在边缘后，将 h 扩展到 0.2–1.8，步长 0.05；6 cuts ×3 sizes ×33 fields，共 594 个 ED 点，约 7.55s CPU wall。

| cut | L=6 peak | L=8 peak | L=10 peak |
| --- | --- | --- | --- |
| A，J2=Jx=0 | 0.85 | 0.90 | 0.95 |
| B，J2=0.1 | 0.95 | 1.05 | 1.10 |
| B，J2=0.2 | 1.10 | 1.20 | 1.25 |
| B，J2=0.3 | 1.20 | 1.30 | 1.35 |
| C，Jx=0.1 | 0.70 | 0.75 | 0.80 |
| C，Jx=0.2 | 0.55 | 0.60 | 0.65 |

全部 peak 均在扩展 grid 内部，bracket 是相邻 ±0.05 fields；最大 ED residual=2.61e-13。χ 使用邻近 normalized ground states 的 central finite-difference fidelity estimator，带有限 dh 误差。该表不是 thermodynamic hc，也未确证这些 cuts 全都属于 Ising class。保留 m²、C(L/2)、S2 与所有 energy/residual，后续先 refine brackets 和补大尺寸 independent references。

## 初次开发验证

新增 **47 tests 全通过**，包括 Pauli signs、Z2、ED/JW、谱 QGT、correct Renyi coefficients、normalized shared-L GRU、24,000-draw sampling frequency、input AD、full-H local energy、exact vs covariance gradient、ensemble SR、exact/VMC checkpoint resumes，以及 SWAP vs independent Schmidt purity。另用 synthetic nu=0.9 数据验证 collapse fitter 没有硬编码 Ising exponent。

按根 AGENTS 规定顺序，原项目路径／掩码、耦合基底、F-move、N≤10 Hamiltonian、AR normalization／sampling 的 **39 tests 全通过**。最终 focused log 见 [focused_validation.log](results/focused_validation.log)。

全仓首轮：[full_suite.log](results/full_suite.log)，**2614 passed、23 failed、2 skipped，22m29s**。其中 18 个 artifact/provenance 测试要求显式 `JAX_PLATFORM_NAME=cpu`，首轮环境未设置该变量；另 4 个 protocol 测试在累计进程中触发 registered RSS cap。使用 CPU 环境、独立进程对全部失败项复查：[cpu_failure_recheck.log](results/cpu_failure_recheck.log)，**22 passed、1 failed，4.37s**。因此不能宣布全仓全绿。

剩余失败是 `tests/test_fixed_sz_spatial_vit_sr_canary_artifact.py::test_spatial_vit_sr_canary_is_complete_sealed_and_matrix_free`：历史 artifact 中 `su2_nqs/vmc/exact_state_sr.py` 的 hash 与该文件当前 hash 不一致。此任务未编辑该源码、相关测试或封存结果。此前单独诊断日志保留在 [existing_failures.log](results/existing_failures.log)。

全仓运行开始后补充的 FNQS 检查均已在最终 [fnqs_validation.log](results/fnqs_validation.log) 的 47 tests 中复验通过。验证汇总见 [validation.json](results/validation.json)。

下一步优先解决共享 neural state 的 (chi_F)／关联精度，再执行正式多 seed／多 L Phase 0。Phase 1 的共享训练和 Phase 2 transfer 仍需该科学 gate；不能由上述 ED tables 或 smoke 结果直接解锁。

## 响应精度改进：六组对照

协议和结果：[protocol.json](results/phase0_controls_v3/protocol.json)、[summary.json](results/phase0_controls_v3/summary.json)、[controls.png](results/phase0_controls_v3/controls.png)。六组均为 1474 参数、宽度 16 GRU、seed=17 相同初始化，1200 updates ×5 Hamiltonians/update，即 6000 主梯度 Hamiltonian 查询。固定尺寸 L=8；共享尺寸 L=4,6,8。稀疏训练为 h=0.8,0.9,1.0,1.1,1.2，密集训练为 0.8–1.2、步长 0.02；development 审计为 0.81–1.19、步长 0.02，与训练互斥。

| 对照 | held 最大能量相对误差 | held 最大 chi 相对误差 | held 最大关联绝对误差 |
| --- | ---: | ---: | ---: |
| 固定 L，稀疏场，Adam | 6.09e-5 | 25.64% | 0.006525 |
| 固定 L，稀疏场，SR | 5.63e-7 | 3.09% | 0.000582 |
| 固定 L，密集场，Adam | 2.78e-4 | 27.46% | 0.007467 |
| 固定 L，密集场，SR | 1.06e-6 | 4.51% | 0.000519 |
| 共享 L，密集场，Adam | 5.91e-4 | 75.56% | 0.024267 |
| 共享 L，密集场，SR | 3.52e-6 | 10.55% | 0.002447 |

当前设置下 SR 明显改善固定 L 的 coupling response；仅加密磁场没有解决 Adam 的响应误差。单 seed、固定超参数的对照不能推广为一般 optimizer 优劣结论。共享尺寸 SR 较困难，仍超出 chi 的 10% 门槛。

SR 对每个 Hamiltonian 分别中心化 score 和 local energy，再等权合并；使用 matrix-free、对角预条件 CG（tolerance=1e-8，maxiter=2000）、shift=1e-4、更新范数上限 0.5 和精确能量回溯。Adam learning rate=0.003，SR=0.5。回溯的候选均首轮接受，但每步仍增加一次能量查询：Adam 实际查询 6000，SR 12000。固定 L 的 basis-state evaluations 分别为 1,536,000／3,072,000；共享 L 为 670,320／1,340,640。CPU wall times 含并行运行争用，不能用它们作严格速度比较，也没有等总成本的 foundation 优势结论。

## 共享响应精修：小尺寸校准通过

共享 SR 再续训 1200 步，使用保留 confirmation 网格 0.805–1.195、步长 0.01：[neural.json](results/phase0_shared_sr_refinement_v3/neural.json)。2400 步后最大 held chi 误差仍为 **10.09%**，最大能量相对误差 4.38e-6、关联绝对误差 0.002156，observable 检查仍为 false。旧失败结果完整保留。

随后将上述审计作为开发证据，在训练前另行声明 confirmation 网格 0.8075–1.1975、步长 0.01；从 2400 步参数启动独立优化阶段，重置优化器，进行 **50 步全部 63 个 Hamiltonians 的精确 SR**，shift=1e-5、参数更新范数上限 0.25。新增主梯度查询 3150，实际查询 6300，父阶段累计成本单独记录：[protocol.json](results/phase0_full_ensemble_refinement_v3/protocol.json)、[neural.json](results/phase0_full_ensemble_refinement_v3/neural.json)、[checkpoint](results/phase0_full_ensemble_refinement_v3/seed_17.npz)。

结果：最大 held chi 相对误差 **7.770%**、能量相对误差 **1.791e-6**、关联绝对误差 **0.002248**；训练点对应值为 7.838%、2.105e-6、0.002361。最大归一化误差 4.44e-16，AD 与 fixed-theta finite-difference fidelity 最大差异 1.88e-7（dh=1e-4）。完整 ensemble 梯度范数由 1.11e-4 降至最后一步的 1.30e-6。**单 seed 小尺寸 observable checks=true，Phase 0 科学 gate=false**，因为尚未建立研究尺寸、多 seed 和自由指数的 reproduction。

该精修同时改变 task batch 和 SR 正则化，不能将进步单独归因于某一项。确认网格与此前磁场点、训练点均分开；仍需要新初始化 seeds 评估方法稳定性。各阶段的 chi 曲线见 [response_progress_v3.png](results/response_progress_v3.png)。

## 研究尺寸三 seed VMC：仍未通过

预算预先声明于 [phase0_vmc_sr_pilot.json](configs/phase0_vmc_sr_pilot.json)，数据见 [neural.json](results/phase0_vmc_sr_pilot_v3/neural.json)、[独立响应／关联审计图](results/phase0_vmc_sr_pilot_v3/observables.png)。L=16,24,32、独立初始化 seeds=17,29,43，固定宽度 16；训练 h=0.95–1.05、步长 0.005，held 为相邻中点。每 seed 200 updates、每步 3 Hamiltonians ×64 Born draws；600 主梯度／实际 Hamiltonian 查询、38,400 训练 draws。每点独立 audit=4096 draws，三个 seed 共约 1124.45s CPU wall。

| seed | held 最大能量相对误差 | held 最大 chi 相对误差 | held 最大关联绝对误差 | 自由拟合 hc | nu | eta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 7.44e-4 | 73.61% | 0.14285 | 0.99222 | 0.7（下界） | 0.31942 |
| 29 | 1.90e-3 | 84.43% | 0.27595 | 1.03001 | 0.7（下界） | 0.45298 |
| 43 | 1.67e-3 | 78.72% | 0.25121 | 1.01263 | 0.7（下界） | 0.43745 |

拟合数字仅用于失败诊断：三个 nu 均触及约束下界，关联／响应与独立参考不符，不能解释为可信 universality 数据。即使 collapse relative RMS 约 0.0011–0.0017、平均 hc 数值接近 1，也不能代替物理精度证据。末步梯度范数仍约 0.012–0.027、最大 task 方差/site 约 1.3e-3–1.6e-3；这个 200 步 pilot 尚未收敛，不证明 GRU 无法表示目标态。科学 gate=false，未启动 Phase 1 shared training。

## 当前实现、来源与下一轮

SR 新增 CG／dense 求解、真实 residual 检查、任务 minibatch、更新范数上限、精确回溯和累计工作量；按尺寸批处理 exact gradients，避免 score Jacobian 的不必要样本二次内存。多 Hamiltonian 的局部中心化以独立 dense Pauli／reverse-Jacobian reference 检查；有限样本测试还能区分错误的 pooled global centering。

主体实验的 v3 数值源码封存在 [runtime_v3/manifest.json](results/runtime_v3/manifest.json) 及同目录包快照，可严格恢复其 checkpoint。当前 v4 修复了外部 uint8 位串下 local energy 的 Z 值算术溢出；独立 N=4,6,8,10 dense-H 测试覆盖此接口。本轮 v3 训练和审计的样本都是有符号整数，故已有结果不受影响。旧数值来源 hash 没有改写。

v4 提供显式 `warm_start_checkpoint`：只导入参数、重置优化器／随机数，核对 model/basis/Hamiltonian 语义并保留父 checkpoint SHA、阶段与成本。三步 [canary](results/phase0_warm_start_canary_v4/neural.json) 完成集成检查；它不是精度证据。下一轮 [warm refinement 配置](configs/phase0_vmc_sr_warm_refinement.json) 与同后续预算的 [scratch 配置](configs/phase0_vmc_sr_scratch_refinement.json) 已准备（1000 步、6 tasks ×128 samples/update、8192 audit draws/point），**均尚未运行**。同一父参数下三个 seeds 是随机优化副本，不能作为独立预训练初始化，也不能忽略父阶段成本。

下一步先比较研究尺寸 warm-start／scratch 的收敛与独立 chi/C 精度，依据梯度、方差和样本预算调整 SR，再考虑模型容量。Phase 0 科学 gate 继续约束 Phase 1 shared training。

## 改进后的验证

v4 focused 数学／初始化检查 **18 passed**：[fnqs_sr_focused_validation_v4.log](results/fnqs_sr_focused_validation_v4.log)。FNQS 完整测试 **58 passed，25.70s**：[fnqs_improvement_validation_v4.log](results/fnqs_improvement_validation_v4.log)。独立 SR 方向误差：dense 最大 1.55e-15、CG 最大 6.69e-11，对照 tolerance=2e-10；混合尺寸和有限样本中心化也有独立 witness，详见 [sr_numerical_validation.json](results/sr_numerical_validation.json)。unsigned local energy 与独立 N≤10 dense Pauli 参考最大差异 **1.42e-14**，tolerance=4e-11。当前来源、版本核验与验收状态汇总于 [improvement_validation.json](results/improvement_validation.json)。

v3 全仓回归 **2641 passed、5 failed、2 skipped，27m01s**：[full_suite_improvement.log](results/full_suite_improvement.log)。独立 CPU 进程复查失败项 **4 passed、1 failed**：[full_suite_improvement_failure_recheck.log](results/full_suite_improvement_failure_recheck.log)。

最终 v4 全仓回归 **2667 passed、5 failed、2 skipped，21m56s**：[full_suite_improvement_v4.log](results/full_suite_improvement_v4.log)，包括 FNQS 的全部 58 项。独立进程复查 **4 passed、1 failed，1.48s**：[full_suite_improvement_failure_recheck_v4.log](results/full_suite_improvement_failure_recheck_v4.log)。4 项为累计进程触及原项目 registered RSS caps；持续失败仍是上文历史 SR artifact 的 `exact_state_sr.py` 来源 hash 不匹配。未修改相关原项目源码、测试或封存结果，不能宣布全仓全绿。

## Phase 1：奇偶扇区精细参考（v5）

数据：[预声明协议](results/phase1_parity_reference_v5/protocol.json)、[完整参考](results/phase1_parity_reference_v5/reference.json)、[峰漂移与谱隙图](results/phase1_parity_reference_v5/reference.png)。6 个 cuts × L=6,8,10,12,14；依据先前粗扫描构造固定局部窗口，dh=0.005，共 **1230 个网格点**。另用两个有限差分步长检查每个峰处的响应，共 **120 次额外参考求解**；约 11.13s CPU wall。全部 30 个峰在窗口内部，以下括号宽度为相邻 ±0.005 磁场点。

| cut | L=6 peak | L=8 peak | L=10 peak | L=12 peak | L=14 peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.855 | 0.915 | 0.945 | 0.960 | 0.970 |
| B，J2=0.1 | 0.975 | 1.050 | 1.090 | 1.115 | 1.125 |
| B，J2=0.2 | 1.085 | 1.185 | 1.230 | 1.260 | 1.275 |
| B，J2=0.3 | 1.195 | 1.310 | 1.365 | 1.400 | 1.420 |
| C，Jx=0.1 | 0.695 | 0.755 | 0.780 | 0.800 | 0.810 |
| C，Jx=0.2 | 0.540 | 0.600 | 0.630 | 0.645 | 0.655 |

χ 通过固定规范的投影线性响应方程计算，保留同奇偶和异奇偶谱隙、关联、m²、半链 Renyi-2 与求解 residual。最大谱 residual=2.10e-11（guard=1e-9），响应相对 residual=9.99e-12（guard=1e-8），归一化误差=4.77e-15。A 的 JW 能量和 χ 最大差异分别为 4.97e-14、1.39e-12；峰处 dh=0.0025 的有限差分 χ 相对差异最大 8.64e-5，包含有限 dh 截断误差。

独立完整 Pauli／谱和的差异记录于 [sector_numerical_validation_v5.json](results/sector_numerical_validation_v5.json)：扇区 Hamiltonian 最大绝对误差 3.55e-15（tolerance=2e-14），χ 最大差异 1.34e-14（tolerance=5e-10）。该参考只提供有限尺寸峰位，**没有 thermodynamic hc、critical exponent 或 universality-class 确证**。L≥16 的 B/C independent references 仍待开发。

## Phase 2：留出协议（v5）

[protocol.json](results/phase2_protocol_v5/protocol.json) 声明 A+B→C、A+C→B、B+C→A 三个 whole-family folds，以及 zero-shot、1/2/4/8 个独立 coupling points 的适应、同架构 scratch 和 pointwise 对照。few-shot 数量指整个 held family 的总点数；确认字段与适应候选网格互斥。split 检查直接读取 Hamiltonian 耦合，拒绝伪装的 family metadata 和 model-selection 泄漏。

该文件是**工程协议，transfer_experiments_started=false**；宽场网格与成本上限仍需 Phase 1 校准后形成可执行调度。正式运行器和全部 checkpoint 祖先的逐文件校验尚未完成。所有参数祖先都必须排除 held family：全家族 Phase 1 checkpoint 不能用于这些 folds，已有 Phase 0 A checkpoint 不能用于 held A。协议使用旧失败 pilot 声明当时 readiness；新配对结果及当前 readiness 单独汇总到 [progression_v5.json](results/progression_v5.json)，保留原输入 hash。

## 研究尺寸配对验证（v5，未完成）

[预声明协议](results/phase0_pair_v5/protocol.json) 在运行前封存 warm/scratch 两个实际配置。L=16,24,32；seeds=17,29,43；每 seed 1000 updates ×6 tasks ×128 draws；SR shift=1e-4、CG tolerance=1e-7、maxiter=5000、更新范数上限=0.25、learning rate=0.5。训练磁场仍为 0.95–1.05、步长 0.005，使用新的 40 点 confirmation 网格 0.95125–1.04875、步长 0.0025；独立 audit=8192 draws/point。

每 arm/seed 后续成本为 6000 主梯度／实际 Hamiltonian 查询、768,000 训练 Born draws、1,499,136 独立 audit draws。warm 的共同父链另有 15,150 主梯度、30,300 实际查询和 3,361,248 精确 basis-state evaluations，跨随机优化副本只计一次；因此是等后续成本对照，不是等总成本对照。同一父参数下三个 warm seeds 不能充当三个独立预训练初始化。

数值源码已封存在 [runtime_v5/manifest.json](results/runtime_v5/manifest.json)；v4 历史版本另存于 [runtime_v4/manifest.json](results/runtime_v4/manifest.json)。新 JIT kernels 在不改变 per-task 中心化、目标函数和预算的前提下复用训练／审计编译；unsigned 关联的独立测试也已补齐。本轮 focused 检查 **19 passed**，新增 sector/split 检查 **17 passed**；FNQS 全套 **76 passed，27.04s**：[fnqs_validation_v5.log](results/fnqs_validation_v5.log)。

两组 seed 17 完成 1000 步并保存 checkpoint；scratch seed 29 的第 225 步因真实 SR residual=1.22657e-6 超过原停止阈值 1e-6 而中止，未接受该方向。warm 随后暂停。旧输出保留于 [status.json](results/phase0_pair_v5/status.json)，批次没有完整 audit 汇总，不能形成配对科学结论。重放相同 cold 初始化和 PRNG 精确重现该故障；[捕获系统](results/sr_failure_diagnostic_v5/failure.json) 的额外诊断消耗 1350 Hamiltonian 查询、172800 Born draws，单独计数。

v5 全仓回归 **2697 passed、5 failed、2 skipped，27m34s**：[full_suite_v5.log](results/full_suite_v5.log)。独立 CPU 进程复查 **4 passed、1 failed，1.68s**：[failure_recheck_v5.log](results/failure_recheck_v5.log)。4 项累计进程 registered RSS cap 在独立进程中通过；持续失败仍为历史 SR artifact 的 `exact_state_sr.py` 来源 hash 不匹配。未修改该文件、相关测试或重封存 artifact，不能宣布全仓全绿。验证汇总：[validation_v5.json](results/validation_v5.json)。

## SR 数值修复与配对重跑（v6，已完成）

故障系统形状为 768×1474，regularized operator 条件数上界约 3.94e6。独立 float64 SVD 解的真实相对 residual=3.51e-14，旧 CG residual=1.23e-6。v6 保留原初解；只有其真实 residual 超过原 guard 时，使用同一 score rows／force 重求解残差方程，至多三次，仍未通过就停止。没有新增 Hamiltonian 查询或 Born draws，没有放松阈值；新增线性求解次数写入累计 work。

[独立数值核验](results/sr_refinement_validation_v6.json)：一次修复后真实 relative residual=1.23e-9，与 SVD 方向的最大绝对差异=2.56e-9（tolerance=4e-9）。新增检查覆盖修复方向、成功初解不变、修复失败仍停止。focused **22 passed，23.91s**；FNQS 全套 **79 passed，24.80s**：[focused_validation_v6.log](results/focused_validation_v6.log)、[fnqs_validation_v6.log](results/fnqs_validation_v6.log)。每个完成 seed 现在独立写入 `neural.partial.json`；partial artifact 不能打开科学 gate。

[新配对协议](results/phase0_pair_v6/protocol.json) 保留 1000 updates／seed、6×128 training draws／update 和 8192 audit draws／point，用新的 40 点确认网格 0.950625–1.048125、步长 0.0025；先运行 seed 29，再运行 17、43。各组重新初始化；旧 source 不满足 strict resume，因此没有将 v5 checkpoint 伪装为 v6 续训。训练源码封存于 [runtime_v6/manifest.json](results/runtime_v6/manifest.json)。两个批次均已完成；Phase 1 shared training 与 Phase 2 transfer 仍未启动。

完整测量：[warm](results/phase0_pair_v6/warm/neural.json)、[scratch](results/phase0_pair_v6/scratch/neural.json)、[原始汇总](results/progression_v6.json)、[诊断图](results/phase0_pair_v6/observables.png)。下表为 held fields 上逐点最大误差；chi 加 2SE 是相对误差加两倍 bootstrap SE／reference，关联加 2SE 是各距离的绝对误差加两倍 SE 后取最大值。该余量是工程审计规则，不是同时置信区间。

| 初始化 | seed | 最大能量相对误差 | 最大 chi 相对误差 | chi 加 2SE | 最大关联绝对误差 | 关联加 2SE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| warm | 29 | 3.95e-5 | 23.57% | 25.50% | 0.02465 | 0.03298 |
| warm | 17 | 3.60e-5 | 21.77% | 25.78% | 0.02204 | 0.03088 |
| warm | 43 | 3.09e-5 | 23.75% | 26.90% | 0.02757 | 0.03668 |
| scratch | 29 | 6.42e-5 | 26.45% | 28.90% | 0.02214 | 0.03083 |
| scratch | 17 | 4.69e-5 | 44.20% | 46.90% | 0.02563 | 0.03359 |
| scratch | 43 | 5.30e-5 | 36.56% | 39.41% | 0.02982 | 0.03786 |

六个轨迹均未通过 chi≤10% 和包含采样余量的关联≤0.03 门槛。warm 后续精度较高只是本批描述：三副本共享同一个预训练 parent，不能替代独立 pretraining seeds，且总成本不同。每组主梯度／实际 Hamiltonian 查询均为 18000，训练 Born draws=2304000，审计 Born draws=4497408；warm 另计共同祖先的 15150 主梯度、30300 实际查询和 3361248 个 exact basis-state evaluations，祖先在三副本之间只计一次。额外残差重求解 warm=1 次、scratch=589 次，未增加 Hamiltonian／Born 查询。并行 CPU wall time 不作严格速度优势比较。

[六个 checkpoint 核验](results/phase0_checkpoint_validation_v6.json) 验证了参数有限、源码 hash、更新数、所有测量点和声明预算。已封存的失败批次及其独立 replay 诊断成本另存，不混入本批科学结果。

v6 全仓回归已完成：**2706 passed、5 failed、2 skipped，1750.72s**；失败项独立复查为 **4 passed、1 failed，1.94s**。持续失败仍是未修改的历史 `exact_state_sr.py` artifact 来源不匹配。汇总见 [validation_v6.json](results/validation_v6.json)。

## 含噪 scaling 拟合的统计修复（v7，已合入）

旧 collapse 的残差先除以 `L^(2/nu)`，再以固定的 `mean(chi/L²)` 归一化。自由改变 nu 会同时缩小噪声的残差幅度，使拟合偏向较小 nu；无噪声 synthetic 检查不能暴露这个问题。因此旧神经结果触及 nu=0.7 不能解释为物理指数。

[独立含噪控制](results/scaling_noise_probe_v6.json) 使用真实 nu=0.9、hc=1.07、五个尺寸、1% Gaussian noise 和 seeds=17,29,43。旧拟合得到 0.869716、0.700000、0.864728；在固定物理 measurement-error 单位下消去自由 scaling-curve coefficients，得到 0.901151、0.900403、0.898312，最大 nu 偏差 0.001688（检查 tolerance=0.005）。该控制不是 neural reproduction。

修复副本在 [runtime_v7_candidate](results/runtime_v7_candidate/manifest.json) 隔离验证，**83 tests passed，30.92s**，完成全仓回归后在两个 v6 训练进程结束时合入。当前源码与已验证副本逐字相同，封存在 [runtime_v7](results/runtime_v7/manifest.json)，合入后 FNQS 再验 **83 passed，24.45s**。[生产拟合器数值核验](results/scaling_numerical_validation_v7.json) 的最大 nu 偏差=0.001688（tolerance=0.005），最大 hc 偏差=0.000128（tolerance=0.0003）。

预先声明的 [保存测量重分析协议](results/phase0_reanalysis_v7/protocol.json) 使用相同 primary window 和已有 window／degree／correction 敏感性网格，显式标记 post-hoc statistical repair，不增加训练、draws、checkpoint selection 或放宽门槛。原始测量、checkpoint 和 v6 来源 hash 保留，修正提取结果另存于 [reanalysis.json](results/phase0_reanalysis_v7/reanalysis.json)。旧目标下的 [敏感性检查](results/scaling_stability_v6/neural.json) 只作历史诊断。

修正后的 primary fit 中 warm 三 seeds、scratch 29／43 仍触及 nu=1.3 上界；scratch 17 的 nu=0.82293、hc=0.96796，改变 held-field window／degree／correction 后 nu 跨 0.80206–1.3。eta 在 0.24344–0.26374，但不能由单项指数接近参考值替代其他门槛。较低 collapse residual 不保证逐点 chi 准确，也不是指数置信区间。两组修正后的科学 gate 均为 false，不解释这些 nu 为 universality class。

v7 隔离副本全仓回归：**2719 passed、5 failed、2 skipped，1518.19s**；失败项独立复查：**4 passed、1 failed，1.51s**。持续失败仍是未修改的历史 SR artifact 来源 hash 不匹配；没有放宽测试或重封存旧结果。首个隔离入口缺少 workspace import path 的 collection 错误及修正后的完整日志分别保留。验证见 [validation_v7.json](results/validation_v7.json)。

## 本轮阶段判断

[最终阶段汇总](results/progression_v7.json) 使用完整冷初始化三 seed 与修正提取方法：**Phase 0 科学 gate 未通过**。Phase 1 的 L≤14 独立 references 和 Phase 2 三折协议已经完成，允许继续较大尺寸参考与运行器准备；尚未完成 B/C 的大尺寸 class verification、Phase 1 neural reproduction 或 Phase 2 ancestor／exact-cost runner，正式 shared training／transfer 均未启动。下一批先做固定尺寸／共享尺寸对照，并分离追加优化与增大宽度的效果，见 [研究计划](RESEARCH_PLAN.md)。

## 固定尺寸、预算与容量对照（v8，已完成）

用户选择三个新 seeds=71,83,97。[训练协议](results/size_capacity_controls_v8/protocol.json)、三份字段调度和 [数值库／运行器快照](results/runtime_v8/manifest.json) 在正式训练前封存；[分析协议](results/size_capacity_controls_v8/analysis_protocol.json) 在首个审计前封存。根目录 18 个 FNQS 数值文件保持 v7 不变，新增调度／审计运行器单独记录 SHA。2026-10-03，**15 条轨迹、21 个审计端点全部完成且通过核验，失败轨迹为 0**：[完整比较](results/size_capacity_controls_v8/comparison.json)。核验包含来源、调度、签名、参数有限性、逐步 SR guard、全部字段、预算、共享轨迹前缀及末端 accepted／audited checkpoint 一致性。运行中的 partial 文件保留，完整结论以本比较为准。

每个 seed 的三个 fixed-width16 模型各自训练 L=16,24,32、1000 步；shared-width16 和 shared-width32 各训练 2000 步并保留 1000／2000 步审计。各尺寸每步两个不同字段，同 seed 所有模型在前 1000 步看到完全相同的逐尺寸字段序列。训练 21 fields=0.95:0.005:1.05；新的 development 20 fields=0.952:0.005:1.047；40 个 reserved confirmation fields 此批不查询。每个 train／development point 独立 audit=8192 Born draws。

| 端点组 | 参数量 | 主梯度／实际 Hamiltonian 查询 | training Born draws | audit Born draws |
| --- | ---: | ---: | ---: | ---: |
| fixed-width16 三尺寸合计，1000 步 | 4422 | 6000 | 768000 | 1007616 |
| shared-width16，1000 步 | 1474 | 6000 | 768000 | 1007616 |
| shared-width32，1000 步 | 4482 | 6000 | 768000 | 1007616 |
| shared-width16，2000 步 | 1474 | 12000 | 1536000 | 1007616 |
| shared-width32，2000 步 | 4482 | 12000 | 1536000 | 1007616 |

表中 training cost 是每条轨迹的累计成本，**不把同一共享轨迹的 1000 与 2000 步相加**。每 seed 合计 30000 Hamiltonian／3840000 training draws，五组独立端点合计 5038080 audit draws；三 seeds 的实际总量核验为 90000 Hamiltonian、11520000 training draws、15114240 audit draws，21000 次 accepted optimizer updates。额外 SR residual solves 共 6229 次（fixed=967、shared16=458、shared32=4804），未增加 Hamiltonian／Born draws。资源 profiling 另有 60 Hamiltonian／7680 draws，不含准确性审计。宽度 32 的 10 步 profile 约 16.20s、peak RSS 772161536 bytes，不能外推完整训练耗时。

主要指标为 development 全点最大 chi relative error+2SE，同时保留 C、E、逐尺寸误差、RMS chi、三 seed 配对差值和宽度×预算交互。误差加 2SE 是已有描述性工程余量，不是同时置信区间。fixed 的汇总连接三个模型的所有点、合计参数／成本，并取真正全点最大误差。共享宽度 32 比 fixed 合计多 60 参数（1.36%）；共享／独立优化和 CG 条件也不同，预算与宽度控制不自动证明纯容量因果关系。CG 额外 residual solves 单独记录，未记录总 CG iterations，不宣称相同线性计算成本或速度优势。

focused **7 passed，5.98s**、汇总／去重 checks **4 passed，6.27s**，最终 FNQS **94 passed，58.34s**。实际 width16／32 的 L=4,6 穷举归一化与 Z2 最大可见误差为 0（归一化 tolerance=1e-12，Z2=1e-14）；固定参数 coupling QGT 的 AD／有限差分最大差异 **3.31e-8**（tolerance=5e-8），新旧同样本审计差异 **0**（tolerance=1e-12）。见 [size_capacity_numerical_validation_v8.json](results/size_capacity_numerical_validation_v8.json)。本批不拟合指数，40 个 confirmation fields 未测量。

全部新增检查之后的全仓回归为 **2754 passed、5 failed、2 skipped，2343.37s**：[full_suite_v8_final.log](results/full_suite_v8_final.log)。失败项在独立 CPU 进程中复查为 **4 passed、1 failed，3.01s**：[failure_recheck_v8_final.log](results/failure_recheck_v8_final.log)。持续失败仍是未修改的历史 `exact_state_sr.py` artifact 来源不匹配；四项累计 RSS cap 在独立进程中通过。没有重封存或弱化断言，全仓尚非全绿。首轮在最后四项汇总检查加入前收集，为 2736 passed／5 failed／2 skipped，另存原日志；期间原项目新增测试也进入了最终回归。运行中测试快照保留于 [validation_v8_tests_complete.json](results/validation_v8_tests_complete.json)，最终训练与验证汇总另存 [validation_v8.json](results/validation_v8.json)。

### 结果：共享、预算与宽度

下表为 development 全尺寸／全字段的最大 χ relative error+2SE，以及不加 SE 的 χ RMS relative error。列按 seeds=71,83,97 排列；固定组在三个模型的全部测量行上取最大值。阈值为最大 χ error+2SE≤10%。

| 端点组 | χ 加 2SE：71 | 83 | 97 | χ RMS：71 | 83 | 97 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-width16，1000 步／尺寸 | 24.61% | 32.69% | 24.12% | 13.58% | 12.33% | 13.07% |
| shared-width16，1000 步 | 39.82% | 24.61% | 35.38% | 16.69% | 8.73% | 15.27% |
| shared-width16，2000 步 | 35.46% | 24.68% | 28.89% | 13.87% | 7.50% | 10.17% |
| shared-width32，1000 步 | 49.02% | 36.76% | 68.18% | 21.09% | 14.76% | 25.82% |
| shared-width32，2000 步 | 51.27% | 25.02% | 57.33% | 17.67% | 10.20% | 19.39% |

配对误差减少量，单位为百分点；**正值表示第二组更好**。保留每个 seed，不用三个 seeds 建立渐近置信区间或显著性结论。

| 对照：第一组 → 第二组 | 71 | 83 | 97 |
| --- | ---: | ---: | ---: |
| fixed16 → shared16，同预算 | −15.22 | +8.07 | −11.26 |
| fixed16 → shared32，同预算、近似总参数匹配 | −24.41 | −4.08 | −44.05 |
| shared16：1000 → 2000 步 | +4.36 | −0.07 | +6.49 |
| shared32：1000 → 2000 步 | −2.25 | +11.74 | +10.85 |
| shared16 → shared32，1000 步 | −9.19 | −12.15 | −32.79 |
| shared16 → shared32，2000 步 | −15.81 | −0.34 | −28.44 |

共享训练在两个 seeds 上更差、一个更好，不能稳定归因于共享尺寸；固定模型自身也未通过 χ 精度。预算翻倍改善六条共享轨迹的 RMS、末 100 步平均梯度／方差，但最坏点不是一致改善（seed 83 的 narrow 变化很小）。宽度 32 在两个预算、全部三个 seeds 下的 χ 最坏点及 RMS 均高于宽度 16；增宽没有解决响应误差。宽度收益随预算的变化为 −6.62、+11.81、+4.36 个百分点，方向也依赖 seed。梯度／方差下降不能证明已完全收敛，本批不能排除更充分优化或不同条件化下的宽模型收益，也不证明表达能力上限。

所有端点组的最大 energy relative error+2SE≤8.50e-5，远小于 0.002；所有 fixed 及 2000 步 shared 组的 correlation error+2SE≤0.03，部分 1000 步共享组仍失败。所有组的 χ 原始最大相对误差也超过 20%，故没有仅因采样余量造成的临界失败。接受更新的最大真实 SR relative residual=9.996e-7，保持原 1e-6 guard。与能量／方差的改善并存的响应形状偏差，是下一项开发需要诊断的对象。

[全组／逐尺寸比较图](results/size_capacity_controls_v8/comparison.png)、[χ/reference 磁场曲线](results/size_capacity_controls_v8/response_profiles.png)、[训练诊断](results/size_capacity_controls_v8/convergence.png) 已逐图检查；图来源与输入 hash 见 [figures.json](results/size_capacity_controls_v8/figures.json)。磁场曲线普遍在窗口右端过高，宽模型尤其明显；在 h≈1 附近接近参考不能替代全窗口审计。

**Phase 0 科学 gate 继续为 false**：[progression_v8.json](results/progression_v8.json)。当前优先小尺寸 coupling tangent／物理响应诊断，逐项控制新的目标函数、采样或条件化变更；不能只增加宽度或依据低能量推进 universality 结论。方法选择后的确认仍需要未使用字段和新声明的独立 seeds。Phase 1 的独立参考和 Phase 2 协议准备可继续，正式 shared training／transfer 未启动。

## 固定参数输入导数与物理响应诊断（v9，已完成）

按用户要求优先诊断 coupling derivatives，并考虑取消跨尺寸训练。[新协议](results/coupling_response_diagnostics_v9/protocol.json) 和 [源码快照](results/runtime_v9/manifest.json) 在新测量前封存，原数值库 18 个文件、v8 检查点和审计全部保持不变。五组端点、seeds=71,83,97、L=16,24,32、已有 development h=0.952/1.002/1.047 共 **135 个点全部完成**：[diagnosis.json](results/coupling_response_diagnostics_v9/diagnosis.json)。每点独立 1024 Born draws，两个 FD 步长复用同一批行；参数 θ 不更新，不对优化器求导。

诊断定义：s=∂h log ψ，τ=(s−⟨s⟩)ψ，E=⟨H⟩，V=∂hH=−ΣX，Q=I−|ψ⟩⟨ψ|。测量 r=Q[(H−E)τ+Vψ]，ρ=‖r‖/‖QVψ‖；χ=‖τ‖² 为 total。另保留状态残差 √Var(H) 和 Hellmann–Feynman 缺陷 dE/dh−⟨V⟩=2Re⟨(s−⟨s⟩)* (E_local−E)⟩。每个 bootstrap replica 都重新计算 E、均值及 Q，而非固定原样本的中心。ρ 不是 inexact neural state 的 tangent-error bound；没有谱隙／状态误差控制，不将它换算成 χ 误差。

**AD 与有限差分一致，但物理响应仍有缺陷。** 中心差分 δ=0.001/0.0005 的 centered-score relative RMS 最大差异分别为 **2.6751e-5／6.6875e-6**；全部 135 点在步长减半后，误差比为 0.249995–0.250007。对应 χ 的 AD/FD 最大相对差为 3.9179e-5／9.7950e-6。该结果支持输入导数实现正确，不证明 neural manifold 是精确 ground-state manifold。

物理响应 ρ 跨点为 **0.04561–0.36437**，逐点 iid-row bootstrap SE 范围 0.00197–0.01713。下表给每组每 seed 的九点平均 ρ（单位 %），并列出跨三 seeds 的点上最大值；只是 development 描述，不是新的科学 gate 或三个 seeds 的置信区间。

| 端点组 | 平均 ρ：71 | 83 | 97 | 所有点最大 ρ |
| --- | ---: | ---: | ---: | ---: |
| fixed16，1000 步／尺寸 | 22.37% | 21.62% | 20.07% | 36.44% |
| shared16，1000 步 | 15.21% | 15.89% | 16.67% | 27.13% |
| shared16，2000 步 | 13.87% | 12.53% | 12.29% | 25.10% |
| shared32，1000 步 | 15.21% | 13.54% | 15.23% | 24.78% |
| shared32，2000 步 | 13.74% | 11.68% | 11.19% | 22.94% |

三个 development 点都未使用 confirmation fields。χ 准确性仍引用 v8 的独立 8192-draw 审计：本窗口右端 h=1.047 的全部 45 个端点都偏高，χ/reference=1.069–1.648。新 1024-draw χ 只作为 paired diagnostic，不能替换原审计或重开 gate。ρ 在窗口中部较小、两端较大；固定 L 模型也有明显残差，因此尺寸共享不是已经确证的唯一原因。更宽模型的 ρ 和状态残差通常较低，但 χ 更差；物理残差对不同 excitation gaps 的权重与 tangent norm 不同，不能把 ρ 的减少直接解释为 χ 改善，也不直接把它当唯一训练目标。每格点 HF 缺陷估计跨点 −0.00213 至 0.00410，并保留 SE；小 HF 缺陷也不能保证正确 χ。

[独立精确校准](results/coupling_response_diagnostics_v9/calibration.json) 用 L=4,6,8 的完整谱扰动。局部 Hψ、H(ψs)、Vψ 与 dense actions 最大差异 **2.49e-14**（tolerance=2e-11）；ρ 与精确预期最大差 **7.11e-15**（tolerance=2e-11），FD-score RMS 最大差 **3.03e-10**（tolerance=2e-9）。独立 Kronecker Pauli 测试另覆盖四个 Hamiltonian 项至 L=10、unsigned bits 和 HF 能量导数。构造在 h=1.02 的 exact-state witness：保持该点 ψ 和 E 完全正确，把 tangent 缩放 1.25；AD/FD 仍一致，而 χ=1.5625χ_exact、ρ=0.25。scale=0 的 frozen state 则 χ=0、ρ=1。能量正确与导数正确是不同要求。

本批新增 **138240 diagnostic Born draws、270 次 H 的 state/tangent actions、135 次 V actions、0 optimizer updates**；不重复计入 v8 训练成本。声明的资源 profile 另计 128 draws、2 H actions、1 V action，含编译约 0.482s，只做资源/有限性检查；小系统谱校准无 Born draws。逐点原始行与 bits 在 pickle-free NPZ，JSON 绑定输入、protocol 和行 SHA；可严格续跑而不重复已完成行。所有 40 个 reserved confirmation fields 继续未测量，不拟合指数。

[响应诊断图](results/coupling_response_diagnostics_v9/response_diagnosis.png) 和 [状态／HF 缺陷图](results/coupling_response_diagnostics_v9/state_and_energy_derivative.png) 已检查；阴影为三 seed 的 min–max，不是置信区间。输入和绘图源码 hash 见 [figures.json](results/coupling_response_diagnostics_v9/figures.json)。

**尺寸路线调整：每个 L 独立、耦合共享。** 原 L 是输入信息，不是优化参数；FNQS 共享 Hamiltonian 耦合族不必同时共享尺寸。`variants/size_features.py` 已实现 legacy、去 log L／1/L 的 no_explicit_size、完全去长度提示的 no_length。后两者 nominal 参数量仍为 1474，但 width16 active 参数分别为 1346 和 1298，不能声称有效容量完全匹配。no_explicit_size 仍通过归一化几何感知 L；no_length 则强制跨 L prefix consistency。PBC TFIM h=1 的 L=4 与 L=6 前四位 marginal 的精确最大差为 **0.01581137**，揭示该 no_length AR 无法同时精确表示这两种分布。此限制仅针对该模型；没有显式 L scalar 的其他架构仍可能通过序列结构感知长度。

新模型三种 policy 均通过 L=4,6,8 穷举 normalization/Z2 与独立 sampling-frequency checks；legacy 与原模型的值、采样、coupling/parameter gradients 逐元素一致。**新特征模型尚未训练**，原检查点也没有套用新 policy。正式固定 L 冷对照仍需独立运行器与预算声明；研究计划已改为先确保每个 L 的响应，再做 FSS。Phase 0 科学 gate 继续 false，Phase 1/2 正式训练／transfer 未启动。

[可复核数值误差](results/response_numerical_validation_v9.json)：一般四项 Hamiltonian／混合耦合方向的 L=4,6,8,10 independent Kronecker action 最大差 **8.88e-15**（tolerance=2e-11）；三种 policy 的归一化最大差 **4.44e-16**（tolerance=1e-12），Z2 log-amplitude 最大差 **0**（tolerance=1e-14）。这些是有限 Hilbert 数学验证，不是额外训练或物理审计。

这里的 active count 仅扣除 size policy 强制置零的输入行，按一般四维耦合架构计数。A-cut 训练中固定为零的 j2/jx 还会使对应权重没有更新，固定 L 的常量特征也可能线性相关；该表不等同于特定训练网格的有效秩／表达能力。135 点物理响应诊断针对 h 方向，不声称已验证未训练的其他耦合方向。

[保存行独立复查](results/coupling_response_diagnostics_v9/replay_validation.json) 再次验证了原 21 个端点、135 个新点的 labels／counts、全部 moments／bootstrap 和原审计 reference，重算最大差为 **0**，不增加 Born／Hamiltonian 查询。绘图／验证器及 focused tests 另存于 [analysis_manifest.json](results/runtime_v9/analysis_manifest.json)，不改变诊断运行器的既有封存。

新增 focused **26 passed，13.84s**；FNQS 全部 **120 passed，48.71s**。全仓回归 **2826 passed、6 failed、2 skipped，1289.04s**：[full_suite_v9.log](results/full_suite_v9.log)。失败六项独立 CPU 复查为 **4 passed、2 failed，1.37s**：[failure_recheck_v9_final.log](results/failure_recheck_v9_final.log)。四项 cumulative RSS guard 在独立进程中通过；两个持续失败均在未修改的 FNQS 之外：历史 `exact_state_sr.py` artifact SHA 不匹配，以及 matched-distance／finite-schur-reanchor 两份配置复用数字 seeds=2026100411/2026100412 的 freshness 检查失败。后者为本轮全仓相对 v8 新出现的失败，不归称为本 FNQS 改动引起或既有 v8 失败。没有修改这些源码、配置、测试或封存结果。最终来源／数值／成本汇总见 [validation_v9.json](results/validation_v9.json)，全仓尚非全绿。

## 门槛距离与探索性推进（v10，保存数据评估）

用户提出相差一个量级内可以考虑推进。[距离核验](results/gate_distance_v10/assessment.json) 只读取原 v8/v9 封存结果，不增加 Born draws、Hamiltonian queries 或 optimizer updates。比值定义为现有 development error+2SE／**原门槛**；此描述性工程余量不是同时置信区间，原 scientific gate 保持 false。

| 指标 | 原门槛 | 五组×三 seeds 的结果范围 | 结果／门槛 |
| --- | ---: | ---: | ---: |
| χ 最大相对误差+2SE | 10% | 24.12%–68.18% | 2.41–6.82 |
| C(r) 最大绝对误差+2SE | 0.03 | 0.02254–0.03414 | 0.75–1.14 |
| E 最大相对误差+2SE | 0.2% | 0.00328%–0.00849% | 0.016–0.042 |

主路线 fixed-width16 的三个 seeds：χ=24.61%／32.69%／24.12%，为门槛的 2.46／3.27／2.41 倍；仍需降低当前误差约 59%–69% 才能达到原 10% 条件。三个 fixed 组的 C margin=0.02568–0.02971 已通过；能量远低于门槛。χ RMS=12.33%–13.58%，但原门槛针对全窗口最坏点，不将 RMS 接近门槛改写为已通过。最新批次的可靠 hc／nu／eta、拟合稳定性和方法选择后的独立确认仍待完成，不能为整个 scientific gate 给出可信的单一“距离”。v9 的物理响应残差不是 χ 误差，也未用来计算此距离。

按用户的新推进条件，允许 **Phase 1 固定尺寸、小预算 neural pilot** 与 Phase 0 响应修复并行；先在独立参考下比较 A／弱 B／弱 C 的状态、响应和条件化，运行前声明预算与 controls。Phase 2 同步准备 ancestry／成本／holdout runner，pilot 核验后可做小 canary。该判断是研究推进选择，不是放宽原精度门槛或已完成 neural universality／transfer；正式阶段结论仍须相应证据。具体后续范围已写入 [研究计划](RESEARCH_PLAN.md)。

## 固定 L 的 Phase 1、响应修复与 Phase 2 工程运行器（v11，2026-10-04）

[预声明协议](results/fixed_size_pilots_v11/protocol.json) 的 **18 条轨迹全部完成**；[配对 review](results/fixed_size_pilots_v11/review.json) 核验源码、协议、optimizer/history、父检查点、实际 task budgets 和审计标签。原 root 数值库、v8/v9 seals 与科学门槛保持不变，两个科学开发方向分别执行；没有追加失败后预算。代码与两个运行器的逐文件快照分别在两个新协议目录的 `runtime/`。

**Phase 1 固定 L=12、width8、no_explicit_size、冷 seeds=113/127/139。** 比较 A/B(J2=.1)/C(Jx=.1) 共用 θ 与每族独立、族内共享 h 的 controls。五个训练点围绕 v5 有限尺寸 χ peak，另四个 development 点和四个 reserved 点；后者未测量。每 seed 每族均 150 tasks×64 draws，shared 150 updates 对 family controls 合计 450 updates。共享 546 nominal／482 size-mask active 参数；三个独立模型合计 1638／1446。总容量及更新次数不匹配，因此只称为相同 Hamiltonian/Born 预算对照，不能单独归因于容量或优化收敛。

各族四个 development 点上最大 χ relative error（%，完整穷举，无 MC SE）：

| seed | A shared / separate | B shared / separate | C shared / separate |
| --- | ---: | ---: | ---: |
| 113 | 98.08 / 98.45 | 96.94 / 94.84 | 98.41 / 97.37 |
| 127 | 96.38 / 97.38 | 94.52 / 99.18 | 96.51 / 97.69 |
| 139 | 97.86 / 98.00 | 97.98 / 98.85 | 97.30 / 98.12 |

所有 development 点的 χ/reference=0.00821–0.05739，显著低估；最大 energy/site absolute error 的组范围 4.86e-4–9.42e-4，state infidelity 跨点仅 0.000232–0.007306，但 half-ring correlation 的组最大绝对误差为 0.0384–0.0718。物理响应 ρ 跨点 0.762–1.147。**窄耦合窗小预算能量训练学到了状态，但 manifold tangent 基本仍平坦**；共享／独立均有此问题，不能据此声称 A/B/C 共享成功、失败由共享造成或其 universality 已确认。较宽耦合覆盖和更充分曝光应作为下一批的新协议变量。本轮未提取 hc/ν/η。

**Phase 0 固定 L=16、legacy width16、v8 三个独立父模型，各分叉两个 50-step branches。** 每步两个训练 fields×64 draws，SR shift=.001、learning rate=.5、最大更新 norm=.05。仅能量对照与 energy+.02ρ² 保持父权重／PRNG／调度一致。ρ² 损失不用 JW/ED χ 标签；其参数梯度包括 Born measure 导数，finite-batch self-normalized estimator 有偏，不能把它当 certified χ error bound。两臂都计算 response diagnostic，logical H actions 相同，微分 FLOPs 不相同。

同五个已有 development h=.952/.977/1.002/1.027/1.047，每点独立 4096 Born draws，χ 给 paired-row bootstrap SE：

| seed | parent χ error+2SE | energy-only | +response | energy-only ρ | +response ρ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 71 | 22.79% | 23.69% | 22.22% | 21.85% | 22.87% |
| 83 | 29.34% | 29.69% | 22.71% | 19.03% | 14.62% |
| 97 | 15.64% | 15.22% | 17.65% | 21.66% | 16.41% |

ρ 栏为五点最大值。响应约束相对 energy-only 的 χ 收益为 +1.46/+6.98/−2.43 个百分点；只有 seed83 同时改善两者。seed97 的 ρ 下降但 χ 变差，seed71 的 χ 略降但 ρ 上升。约束臂 energy/site absolute error 的组最大值为 0.001157/0.000289/0.000229，energy-only 为 5.5e-5/3.6e-5/2.5e-5，需继续控制状态精度 tradeoff；这些是 per-site absolute errors，不能直接与旧相对能量门槛混用。**修复候选未稳定通过 10% χ 门槛**，不默认选择 λ=.02。该五点 L16 审计不是原三尺寸／完整网格的重新验收，也不将 parent 这轮测量替代 v8 原审计。

[对照图](results/fixed_size_pilots_v11/comparison.png) 已检查，左图实线圆点为 shared、同色虚线叉号为 separate，颜色表示 seed；中／右图分别保留三种子。没有把三 seeds min/max 当渐近置信区间。[逐磁场修复曲线](results/fixed_size_pilots_v11/repair_profiles.png) 保留 1 bootstrap SE：seed83 的右端 χ/reference 从 1.261 降至 1.098，但左端从 0.836 降至 0.806；seed97 也把左端从 0.923 压到 0.857。因此当前损失降低部分残差时，仍可能将本已低估的 χ 进一步压低；应检查整条响应曲线而非仅挑右端改善。该绘图只读保存行，无新增物理查询。

**Phase 2 运行器完成三折 L=6 工程 canary**：[协议](results/transfer_runner_canary_v11/protocol.json)、[实际结果／成本](results/transfer_runner_canary_v11/summary.json)。seen-only cold pretraining、frozen zero-shot、1/2/4/8-point adaptation、各 k 配对 scratch、四个 direct pointwise controls 共 39 checkpoints 全部封存并核验。持出依据物理 couplings，每个 pretraining 参数祖先的 train／selection points 都排除整族；拒绝不完整祖先、源代码／特征／尺寸／checkpoint hash 变化及 task/history 不一致。每折所有端点先 seal，再读取 held reference，适配点与 held audit 点互斥。pointwise 直接在四个 audit Hamiltonians 训练，是 supervised per-H control，无持出字段声明。共享祖先按 checkpoint 内容去重；每臂 inclusive cost 同时保留。一个 seed、小尺寸、极短预算只验证工程逻辑，**完整科学 transfer campaign 未启动**。

实际成本：Phase 1／0 新训练合计 **6600 logical H actions、3300 V actions、211200 Born draws、2100 updates**。三个 v8 参数父模型去重后另计 H=6000、Born=768000、updates=3000；inclusive 总计 H=12600、Born=979200、updates=5100。这里不摊销历史方法选择／非参数祖先的研究成本，也不把 logical action count 当 FLOPs。科学 pilot 审计另计 **184320 Born draws、663552 exhaustive neural basis rows、414 H actions、207 V actions**。Phase 2 canary 训练为 H=480、V=240、Born=7680、updates=126，审计为 7680 exhaustive rows、H=240、V=120；无 audit Born draws。独立参考生成与数学校准分开记录，不冒充神经训练。

新增 **19 focused passed／10.56s**，FNQS **139 passed／53.35s**；稠密响应损失及含 Born 导数的参数梯度、arbitrary-force SR、Pauli 四项至 L10、归一化／Z2、物理持出／祖先 tampering／成本去重都有独立检查。[数值误差](results/pilot_numerical_validation_v11.json)：Pauli action 最大差 3.55e-15（容差 2e-11）、response objective 差 9.65e-13（2e-9）、独立 h/θ 有限差分 directional gradient 差 1.10e-8（2e-7）、SR 对稠密解最大差 3.40e-9（1e-6）、normalization 最大差 2.22e-16（1e-12）、Z2 差 0（1e-14）。嵌套有限差分使用四阶 h stencil 控制消去误差，未放宽原精确检查。保存行与 bootstrap 的重算最大差为 **0**（容差 1e-12），不增加 Born/H queries。原 v9 的 135 点／21 原端点再次重算差为 0。

全仓回归 **2867 passed、7 failed、2 skipped，1335.18s**：[完整日志](results/full_suite_v11.log)。七项失败独立复查为 **5 passed、2 failed，1.34s**：[复查日志](results/failure_recheck_v11.log)。四项 cumulative RSS guards 在全仓进程 peak RSS=8.804GiB 时失败，独立进程通过；新增 Schur capacity artifact 检查在全仓时缺失 `results.json`，复查时通过，没有由本任务创建或修改该外部结果。两个持续失败与 v9 相同：未修改的历史 spatial-ViT SR artifact 来源 SHA 不匹配，以及 matched-distance／finite-schur-reanchor 配置的 numeric seeds freshness 冲突。全仓不是全绿，**没有 FNQS 失败**；未重封存历史 artifact 或放宽断言。最终来源／预算／状态见 [validation_v11.json](results/validation_v11.json) 和 [progression_v11.json](results/progression_v11.json)。

复现到新目录，数值源码或协议不一致会拒绝执行；未完成轨迹保留，不自动延长或覆盖：

```bash
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_fixed_size_pilots prepare --output /tmp/fnqs-pilots
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_transfer_pilot prepare --output /tmp/fnqs-transfer-canary
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_fixed_size_pilots run --output /tmp/fnqs-pilots --track phase1
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_fixed_size_pilots run --output /tmp/fnqs-pilots --track phase0
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.run_transfer_pilot run --output /tmp/fnqs-transfer-canary
JAX_PLATFORM_NAME=cpu python -m fnqs.scripts.analyze_fixed_size_pilots --directory /tmp/fnqs-pilots --transfer /tmp/fnqs-transfer-canary
```

## 耦合响应辨因与已执行方案（v12–v14，2026-10-04）

按用户授权先完成辨因，再以预声明条件推进修复。全程固定 L12、width8、no_explicit_size、冷 seeds=113/127/139，energy-only，无 ED/JW χ 或 state 标签进入优化。每个阶段训练前保存实际 schedule、budget 和 runtime snapshots，全部端点封存后才审计。旧 v11 及 root 数值源码未改变，原 χ≤10%、全部 C(r) 绝对误差≤.03、能量相对误差≤.002 的阈值不变。

**第一步：覆盖范围 ×优化预算。** [v12 协议](results/coupling_factorial_v12/protocol.json) 保持五个训练点，窗口中心±.04／±.15，单点 SR 150／600 updates，每 task 64 Born draws。center=.96 是 L12 有限尺寸 χ peak，不能视为 hc。主比较使用两个训练区间内相同四点 center±.03、±.01；另十四点检验宽区域，窄窗下属于外推，单独报告。原四 reserved fields 保持未测量。150/600 是同一轨迹的两端点，成本只计 600 步一次。

中心 development 的 χ 最大相对误差（%，穷举，无 MC SE）：

| seed | 窄窗150 | 窄窗600 | 宽窗150 | 宽窗600 |
| --- | ---: | ---: | ---: | ---: |
| 113 | 98.45 | 97.17 | 98.73 | 98.82 |
| 127 | 97.38 | 96.52 | 98.16 | 98.66 |
| 139 | 98.00 | 98.79 | 97.54 | 98.80 |

[v12 review](results/coupling_factorial_v12/review.json) 的预声明判据为至少一种另一因素设置下，三个 paired seeds 均改善≥5个百分点。覆盖与预算均不满足，不能默认选宽窗或追加预算。窄窗150逐参数重现 v11 A controls，三 seeds 最大差均为 **0**，PRNG/work 相同：[重放核验](results/coupling_factorial_v12/narrow150_replay_validation.json)。这是声明范围内的负结果，不证明更大预算永远无效，也不证明容量已到极限。

**第二步：相同总工作量的更新组织对照。** [v13 协议](results/joint_field_probe_v13/protocol.json) 复用宽窗600的单点 controls，以相同五训练 fields、相同 task 顺序和每 task 64 draws，改为 120 updates×5 Hamiltonians/update。双方每 seed 均 600 tasks、1200 H actions、600 V actions、38400 Born draws；模型、shift=.001、learning rate=.5、max-step=.1 不变。每个 Hamiltonian 内独立中心化，再合并 SR metric/force。终点 PRNG 完全相同；参数变化使抽到的 Born 位串不同，不能称为共用相同配置样本。

| seed | 逐场600 χ | 联合120 χ | 联合全部C(r)最大误差 | 联合能量相对误差 | 联合ρ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 113 | 98.82% | 20.60% | .01859 | .0001940 | .2073 |
| 127 | 98.66% | 24.99% | .01036 | .0001247 | .1996 |
| 139 | 98.80% | 17.70% | .00586 | .0001845 | .1742 |

三个 seeds 分别降低 χ 误差 **78.23、73.66、81.11 个百分点**，全部 C(r) 亦改善，满足 [v13 review](results/joint_field_probe_v13/review.json) 的预声明启动条件。结果支持在当前模型／预算下修复更新组织。每 update 批量从64增至320、参数更新从600减至120，batch noise、metric 条件化与更新次数一起变化，不能唯一判定主导机制。矩阵求逆与 ensemble 平均不交换是本项目的解释，不能据此宣称单点 SR 数学错误。

[随场响应曲线](results/joint_field_probe_v13/response_profiles.png) 直接读封存的18个development点，没有新增查询：逐场χ几乎平坦且近零，联合更新恢复数量级，但seed127/139的峰仍向较大h偏移，右侧ρ也上升。曲线说明幅值恢复后仍有形状误差，不能由中心误差下降就宣布critical peak正确；连线及灰色中心区域仅作视觉引导，不是额外测量。

**第三步：条件推进 A/B/C 方案。** [v14 协议](results/joint_family_solution_v14/protocol.json) 声明共享 ABC 及各族独立、族内共享 h 的12条冷轨迹；三个模型不共享参数。窗口均 center±.15，各族五 fields、120联合 updates。共享每步15 tasks，独立每步5 tasks；每族总600 tasks、38400 Born draws，等总 H／Born work。共享 nominal546／active482参数，三个独立模型合计1638／1446；updates120对aggregate360，仍不能单独分离容量／更新组织或提出 foundation 成本优势。

各族中心四点最大 χ 误差（%，shared / separate）：

| seed | A | B(J2=.1) | C(Jx=.1) |
| --- | ---: | ---: | ---: |
| 113 | 30.52 / 20.60 | 8.22 / 14.22 | 20.78 / 15.86 |
| 127 | 23.79 / 24.99 | 5.44 / 8.23 | 16.83 / 29.59 |
| 139 | 29.16 / 17.70 | 6.44 / 15.31 | 24.95 / 10.69 |

[v14 review](results/joint_family_solution_v14/review.json) 中，18个 model-family中心组的全部 C(r) 最大误差≤**.0239261**，能量相对误差≤**.000223788**。共享 B 的中心 χ 三 seeds均<10%，但十四点宽区域为 **13.72%、25.12%、34.33%**；部分 A/C 宽区域关联也超过 .03。中心达标不等于整窗口、研究尺寸或 universality 通过。相对 v11 的 ABC 总体改善同时涉及窗口、预算和组织方式；直接隔离更新组织的证据来自 v13 的 matched-work A 对照。

v14 的 A controls 实际重新训练，全部费用计入；参数与终点 PRNG逐项复现v13，最大差 **0**。[对照图](results/joint_family_solution_v14/remedy_comparison.png) 左图为2×2，中图为matched-work，右图实线圆点为ABC、虚线叉号为separate，同色为同seed，10%虚线只标原observable阈值。未用三 seeds范围构造置信区间，未提取hc/ν/η/c。

**原始数据与独立核验。** 共21条冷轨迹、27个封存端点、594个 development points，每点穷举4096位串，保存ψ、h-score、Hψ、H(ψs)、Vψ与两步长FD原始行。独立奇偶扇区 CSR action 重算，与训练 local-energy 路径比较；全部距离的平移平均 C(r) 从原始位串另算；Born-weighted χ／响应／HF moments重放最大差0。三个阶段的 [核验1](results/coupling_factorial_v12/replay_validation.json)、[核验2](results/joint_field_probe_v13/replay_validation.json)、[核验3](results/joint_family_solution_v14/replay_validation.json) 给出：H/V action最大差 **1.07e-14**（容差2e-11），C(r)重算最大差 **2.00e-15**（1e-12），归一化误差 **6.66e-16**（1e-10），AD/FD centered-score最大relative RMS **3.74e-6**（guard5e-4，dh=.001/.0005）。输入导数实现误差不足以解释先前近99%的χ低估。

**实际成本，阶段端点不重复计费：**

| 阶段 | updates | tasks | H actions | V actions | training Born draws |
| --- | ---: | ---: | ---: | ---: | ---: |
| v12窗口／预算 | 3600 | 3600 | 7200 | 3600 | 230400 |
| v13联合更新 | 360 | 1800 | 3600 | 1800 | 115200 |
| v14家族复验 | 1440 | 10800 | 21600 | 10800 | 691200 |
| 本轮合计 | 5400 | 16200 | 32400 | 16200 | 1036800 |

神经审计另计 **2433024 exhaustive rows、H1188、V594、Born0**；独立action核验另计 **H1188、V594、Born0**。独立ED参考按stage分别18／18／54点（总90个stage内独立参考，各stage之间有重复点），不纳入神经预算。source copies分别位于三个protocol目录的`runtime/`，分析与新增测试源码在 [analysis manifest](results/runtime_v14_analysis/manifest.json)；[progression_v14.json](results/progression_v14.json) 绑定全部protocol/review/核验SHA、参数重现和费用。本轮所有字段都作development，四reserved points及旧研究尺寸confirmation未使用；参数祖先均为空的冷初始化，没有Phase2持出结论。

[执行环境记录](results/execution_environment_v12_v14.json) 绑定三份协议，记录同一个显式`nqs`解释器的Python／NumPy／SciPy／JAX版本、CPU及float64状态；该补充记录在实验后生成，未伪称训练前已封存版本信息，本轮没有安装或升级环境。

新增辨因判据、split／seal与独立dense Pauli/χ/全部C(r)检查 **8 passed，1.42s**：[focused log](results/factorial_focused_validation_v12.log)。三阶段真实SR relative residual最大 **9.99999e-8**，均在原guard内，额外残差重求解次数均为0；checkpoint实际工作量与schedule聚合完全一致。

最终全仓回归 **2883 passed、6 failed、2 skipped，1614.85s**：[完整日志](results/full_suite_v12.log)。FNQS全部 **147 tests通过**，没有FNQS失败；[收集日志](results/fnqs_collection_v12.log)核对该数量，不将collection冒充单独测试运行。六项失败独立进程复查为 **4 passed、2 failed，1.61s**：[复查日志](results/failure_recheck_v12.log)。四项累计进程RSS guard在peak RSS=8.84663GiB时触限，独立进程通过。两个持续失败与v11相同：旧spatial-ViT SR artifact的`exact_state_sr.py` hash不匹配，以及`spin_finite_schur_reanchor.json`的numeric seeds freshness冲突。未修改这些外部源码、配置、测试或重新封存历史结果，全仓不是全绿。最终数值／来源／成本／测试状态见 [validation_v12_v14.json](results/validation_v12_v14.json)。

**科学gate仍false。** 已将有实证支持的修复推进到A/B/C复验；下一项是固定L16、保持特征／损失／shift不变的paired更新组织验证，分辨噪声与条件化，再冻结方法，用新冷seeds与reserved fields独立确认。尚未启动Phase2科学transfer或研究尺寸指数拟合。

## 固定 L16 配对验证与后续响应修复（v15–v17，2026-10-05）

用户要求优先解决阻碍Phase0的响应精度，并执行保持特征、宽度和损失不变的L16配对。当前最大缺口是神经态的coupling tangent／χ精度；此前AD/FD与精确算符检查通过，低能量误差没有保证导数准确。尚无证据将缺口唯一归因于模型容量，后续允许扩大容量，但先执行可区分优化问题的对照。

[v15协议](results/phase0_joint_pair_v15/protocol.json) 保持legacy width16、1474参数、energy-only、SR shift=.001、learning rate=.5、最大更新norm=.05。三个独立v8父模型seed71/83/97，各自分叉为1050逐场updates与50个全21场joint updates；原21个train fields与顺序相同，每场总曝光50次、每task64 Born draws。每臂1050 tasks、2100 H actions、1050 V actions、67200 draws；两臂参数更新数不同。全部六端点先封存，再在原20点development网格穷举65536配置，保留40个confirmation fields未使用。

| seed | 逐场χ最大误差 | 联合χ最大误差 | 联合全部C(r)最大误差 | 联合能量相对误差 |
| --- | ---: | ---: | ---: | ---: |
| 71 | 20.84% | 22.54% | .005161 | 7.90e-6 |
| 83 | 24.79% | 23.96% | .003897 | 5.56e-6 |
| 97 | 13.42% | 14.99% | .002933 | 5.58e-6 |

[v15 review](results/phase0_joint_pair_v15/review.json)：联合更新仅一个seed改善χ，三seed一致收益判据为false，全部χ仍超过10%。C(r)与能量改善不能替代响应精度；L12的冷训练收益没有自动推广到L16 warm branches。本轮实际新训练H12600、V6300、Born403200、updates3300。三个v8参数父模型去重后的inclusive为H18600、Born1171200、updates6300；旧父链另有170次残差修复，新增分支均0次。

独立的 `diagnostics/finite_chain.py` 将本次审计明确限制到L≤16，原 `basis.all_bits`／ED guard仍为L≤14。新增7项检查与原pilot math合计 **18 passed／17.22s**：[focused log](results/l16_pair_focused_v15_final.log)，初版接口接错返回值的失败日志也保留。N≤10与独立dense Pauli比较；[L16独立product-state校准](results/l16_numerical_validation_v15.json) 用解析单点Pauli／Bernoulli-QGT检验能量、χ及全部C(r)，最大差2.13e-14，容差2e-11。它不是神经训练成绩。120个保存审计点的 [独立CSR重放](results/phase0_joint_pair_v15/replay_validation.json) 最大action差1.07e-14、物理moment差7.46e-14，容差均2e-11。首次神经审计另计H240、V120、7864320 basis rows；独立重放另计H240、V120；两者Born0，product-state校准另计H2、V1。

**后续候选：等价参数坐标中的SR。** [输入Gram诊断](results/feature_units_gram_v16.json) 的常数＋h特征条件数为4365.64；用x=(h-j1)/.05后为2.73。这是两维输入Gram，不能当完整神经SR条件数或χ误差界。`variants/coupling_units.py` 与 `pilots/coordinate_sr.py` 在优化器内做可逆仿射参数变换，回调、检查点及审计都转换回原physical GRU API。初始ψ及全部physical coupling derivatives严格相同；物理特征信息、width、表达能力与损失不变。δI damping及Euclidean clipping的有效作用随坐标改变，是本次修复变量。

新增等价性、四个physical Γ导数、dense Pauli force与identity-chart整段训练／PRNG复现检查 **9 passed／15.98s**：[focused log](results/coordinate_sr_focused_v16.log)。[v16预声明协议](results/phase0_coordinate_repair_v16/protocol.json) 从同三个原父模型出发，保留21场、M64、SR数值超参数、loss与PRNG。50步对原joint50是相同task／H／Born／updates的坐标对照；200步为同轨迹预声明较长优化，尚无raw-coordinate200配对，不能唯一将该端点收益归因于坐标。三个轨迹全部封存后再审计，没有χ/reference labels进入训练。

v16已完成三个累积轨迹、六个端点。最坏χ误差（50步／200步）为seed71 **21.46%／16.53%**、83 **20.38%／14.96%**、97 **10.52%／5.67%**；50步改善方向均相同，但seed71的1.09个百分点未满足预声明的≥2个百分点条件，故一致改善判据仍false。200步只有seed97通过全部observable门槛。全部端点C(r)最大误差≤.004522、能量相对误差≤6.65e-6；下降的响应残差不是χ误差界。实际新训练仅对200步终端计费：updates600、H25200、V12600、Born806400，50步中间端点不重复相加。训练标签与reserved fields均未使用。

**继续检验优化速度。** [v17协议](results/phase0_step_pair_v17/protocol.json) 从每个v16 step200的同一父权重与终端PRNG分叉，学习率.5对4，每支100次全21场联合更新。特征、width16、energy-only、坐标、每场samples64、SR shift=.001、max_step=.05及solve tolerance全部相同；每臂2100 tasks、H4200、V2100、Born134400。先封存唯一100步端点、再穷举审计，不以中间χ挑检查点。初始配对／参数签名／成本已检查：[preflight](results/phase0_step_pair_v17/preflight.json)；复跑核心数学检查16 passed／14.43s：[log](results/step_pair_focused_v17.log)。

最终每seed的20点development最坏误差：

| seed | 学习率.5 χ | 学习率4 χ | 快分支全部C(r) | 快分支能量相对误差 |
| --- | ---: | ---: | ---: | ---: |
| 71 | 14.15% | **6.42%** | .002384 | 7.93e-6 |
| 83 | 13.48% | **9.33%** | .000970 | 2.93e-6 |
| 97 | 4.79% | **3.56%** | .000778 | 4.16e-6 |

[v17 review](results/phase0_step_pair_v17/review.json)：学习率4三个seeds的χ≤10%、所有C(r)≤.03、Erelative≤.002，预声明的local observable条件通过；慢分支只有seed97通过。终端PRNG全部配对相同。两个学习率仅是名义步长变化，快分支部分更新受相同norm cap限制；不声称每步都恰好是八倍更新。这支持当前固定L16的优化速度影响coupling响应，不能唯一解释全部旧缺口或证明其他尺寸容量充足。物理GRU保持1474 nominal参数，本轮没有增宽／加χ监督／加响应损失。

[配对响应图](results/phase0_step_pair_v17/comparison.png) 展示全部20点及parent200。这里的field grid峰位置仍有差异：快分支为.987／.982／.957，JW网格峰为.977；seed97的χ点值虽更准确，峰位置未同步改善。χ最大误差达标不是hc准确的保证；这些离散单尺寸峰也不是热力学临界场。没有从本批提取或确认hc／ν／η。

**最终核验。** 三批共18个审计端点、360点、23592960穷举rows。独立CSR重放最大H/V action差 **1.42e-14**，physical moment差 **7.46e-14**（容差均2e-11），全部C(r)重放差 **7.77e-16**（1e-12），归一化差 **6.66e-16**（1e-10），两步长AD/FD centered-score relative RMS最大 **4.50e-6**（guard5e-4）。v16初始chart检查物理h-score最大差 **1.95e-14**（1e-11）。全部实际SR residual最大≤1e-7，新增残差修正0次。三份独立核验分别在 [v15](results/phase0_joint_pair_v15/replay_validation.json)、[v16](results/phase0_coordinate_repair_v16/replay_validation.json)、[v17](results/phase0_step_pair_v17/replay_validation.json)。

实际新训练成本；v16只计每条200步终端，不另加50步：

| 批次 | updates | tasks | H actions | V actions | Born draws |
| --- | ---: | ---: | ---: | ---: | ---: |
| v15更新组织配对 | 3300 | 6300 | 12600 | 6300 | 403200 |
| v16等价坐标 | 600 | 12600 | 25200 | 12600 | 806400 |
| v17学习率配对 | 600 | 12600 | 25200 | 12600 | 806400 |
| 新训练合计 | **4500** | **31500** | **63000** | **31500** | **2016000** |

三个参数父模型与本轮终端去重后的inclusive为18个checkpoints、updates7500、H69000、Born2784000；旧父模型另有170次SR残差修正，新增为0。这里是参数祖先与本轮费用，不是整个项目方法开发的累计成本。首次穷举审计另计H720、V360、Born0；独立重放另计H720、V360、Born0；product-state校准另计H2、V1、Born0。初始等价性检查另有4608 model rows、H0／Born0。声明前封存的数值source/runtime/env分别在各protocol，分析／新增数学测试copies见 [analysis manifest](results/runtime_v17_analysis/manifest.json)，[progression_v17](results/progression_v17.json) 绑定所有审计、来源、工作量和阶段状态。

全仓回归 **2910 passed、6 failed、2 skipped，2367.42s**：[完整日志](results/full_suite_v15_v16.log)。FNQS全部 **163 tests通过**，[collection](results/fnqs_collection_v15_v17.log)仅核对该数目，不伪称单独运行了另一批163项。六项失败在新进程复查为 **4 passed、2 failed，2.17s**：[复查](results/failure_recheck_v15_v17.log)。四项累计RSS guard在8.86496GiB的full-process peak触限，独立通过；持续失败仍是既有spatial-ViT SR source hash和numeric-seed freshness冲突。未修改外部源代码、配置或重封存artifact，全仓不是全绿。完整记录见 [validation_v15_v17](results/validation_v15_v17.json)。

**固定L16 development的observable门槛已恢复，原Phase0科学gate仍false。** 本轮三个warm初态来自独立预训练seeds，却已参与方法选择；不能代替冻结方案后的独立初始化／reserved确认。当前没有独立方法确认、L24/L32响应修复或自由指数核验，没有新Phase1训练或Phase2科学transfer。后续先冻结这个energy-only、固定尺寸候选，再执行新确认与研究尺寸校准；不因本轮达标自动扩大旧协议预算。

## 冻结后独立确认与各阶段复核（v18，2026-10-05）

[确认协议](results/phase0_frozen_confirmation_v18/protocol.json) 在任何新训练及reserved参考测量前固定方案：fixed L16/L24/L32，legacy width16，energy-only；冷初始化seeds=2026100511/2026100513/2026100517，在三个尺寸间配对初始化，但各L独立优化。每个端点完整重复1000步原始预训练、200步affine-coordinate联合SR和100步learning_rate=4的联合SR。每端点1300 updates、8300 tasks、H14600、V6300、Born659200；九个端点预定总成本updates11700、H131400、V56700、Born5932800。任何reserved结果不会用于追加训练或选择预算。

所有九个最终端点先seal，再打开原保留的40个磁场点。L16穷举65536 configurations；L24/L32每点65536独立Born draws测物理h方向χ和全部C(r)，前4096行测能量、前2048行做两个FD检查；验收为误差+2SE，原阈值χ10%、C(r).03、Erelative.002。h=1另作为训练字段calibration，明确不计入持出。自由hc/ν collapse、自由η、窗口/拟合形式、seed bootstrap和峰位二次拟合/网格稀疏检查均在看确认数据前声明。只有三个尺寸，删除最小L后不足三尺寸，不能声称完成该项尺寸稳定性核验。[响应补充协议](results/phase0_frozen_confirmation_v18/response_supplement/protocol.json) 另预声明L24/L32三个磁场点的投影响应残差、HF偏差和200次paired bootstrap；这项诊断不替代χ精度。

Phase 1/2 已完成read-only独立复核：[Phase 1](results/phase_checks_v18d/phase1.json)、[Phase 2](results/phase_checks_v18d/phase2.json)。Phase 1的12个原封存端点、324个完整Born配置测量点和方法选择输入均匹配；独立稀疏扇区H/V作用最大差异1.07e-14（tolerance2e-11），保存moments重算差异0。原训练成本updates1440、H21600、V10800、Born691200不重复计作新训练；本次replay另外H648、V324、Born0。B中心χ通过仍不能抵消宽窗口13.72%/25.12%/34.33%的误差，A/C响应继续超限。固定L12、未测reserved及L≤14参考均不足以支持普适类结论。

Phase 2的三折持出、所有参数祖先、39个检查点、12组few-shot/scratch的实际任务/预算/终端PRNG，以及pointwise直接训练字段均通过核验。120个审计点用独立完整Pauli张量积H和谱求和重算，χ参考最大差异5.77e-15（tolerance5e-10），moments最大差异2.66e-15（tolerance2e-11）。原inclusive训练成本updates126、H480、V240、Born7680匹配；本次审计另计H240、V120、Born0和120次独立dense参考求解。few-shot适配与scratch各256 draws，但前者另含256 draws的预训练；完整成本不能仅报适配预算。L6、单seed和极短预算仍只支持engineering_complete，不能声称科学transfer或foundation成本优势。旧输出未重封存；核验器调试失败日志也保留。

### Phase 0 独立确认实测

九个端点和所有预声明测量已完成，[pipeline receipt](results/phase0_frozen_confirmation_v18/pipeline_complete.json) 与 [完整来源／成本／阶段判断](results/progression_v18.json) 绑定实际输出。三条初始化的参数hash互异，在各尺寸间配对；没有从v17开发模型继承参数。下面是原40点reserved区域的最坏χ相对误差，L24/L32使用误差+2SE；阈值仍为10%。

| 新 seed | 固定 L16（穷举） | 固定 L24（MC） | 固定 L32（MC） |
| --- | ---: | ---: | ---: |
| 2026100511 | **8.03%** | 12.56% | 29.87% |
| 2026100513 | **9.88%** | 17.54% | 22.98% |
| 2026100517 | **6.85%** | 15.81% | 22.08% |

**L16冻结后独立确认通过，L24/L32全部失败。** 九个模型的全部C(r)与能量均通过：最大的C绝对误差+2SE为.0115835（门槛.03），Erelative+2SE为1.00746e-4（门槛.002）。L32最坏χ点在seed2026100511、h=1.0484375，原始误差28.85%，计入2SE后29.87%；2SE仅贡献1.02个百分点。因此主要偏差不能靠增加审计样本消除。该点在预声明主标度窗口外，但仍在原确认区域内，不能事后缩小精度门槛的适用区域。[完整χ曲线](results/phase0_frozen_confirmation_v18/confirmation_profiles.png) 显示幅值及形状偏差。

18个L24/L32响应补充点的投影物理方程相对残差ρ为.09895–.25263，保存HF缺陷、paired bootstrap SE和两个物理h差分步长；这是诊断，不能当χ误差界。369点[独立重放](results/phase0_frozen_confirmation_v18/independent_replay.json) 通过：H/V作用最大差异2.77e-13（tolerance2e-11）、moments7.46e-14（2e-11）、完整C(r)3.93e-14（1e-12）、归一化6.66e-16（1e-10）、centered-score AD/FD relative RMS最大7.29e-6（5e-4），χ/C covariance差异1.69e-21（1e-10）。L16使用独立稀疏Pauli算符；L24/L32重算全部保存估计量，并独立重放每点32行local energy，没有声称穷举较大Hilbert空间。没有发现需要用修改AD或算符约定解释χ偏差的证据。

### 自由指数与峰位稳定性

预声明主拟合使用 |(h−1)L|≤.8 的字段选择、degree6曲线及1/L修正；选择窗口的中心不固定拟合hc。η在自由拟合hc处插值C(r)，以PBC chord、r_min=4和每个L独立幅值自由拟合。以下均为失败确认数据的描述性拟合，不能作为可靠的universality估计。

| seed／pooled | 自由 hc | 自由 ν | 自由 η |
| --- | ---: | ---: | ---: |
| 2026100511 | 1.04844（触及边界） | 1.18364 | .58851 |
| 2026100513 | .97741 | 1.30000（触及边界） | .15735 |
| 2026100517 | .96305 | 1.12231 | .11672 |
| pooled | .98163 | 1.23424 | .16965 |

pooled的窗口／degree／修正项敏感性spread为Δhc=.06161、Δν=.20836；关联起始距离／修正项spread为Δη=.03976。原容差分别为.02/.05/.03，窗口与拟合形式稳定性失败；pooled主拟合也只通过hc值门槛，ν及η失败。200次paired-seed和条件MC bootstrap保留在[完整拟合](results/phase0_frozen_confirmation_v18/scaling_stability.json)，只有三个独立初始化且不包含共同系统偏差，不能把其区间当作充分校准的统计置信区间。三个尺寸也不足以完成删除最小L后的三尺寸复验；本轮未执行L48或尺寸bootstrap。

峰位与同尺寸连续JW峰比较，另检查局部5/7/9点窗口、半网格抽取和条件MC波动。不能把有限尺寸χ峰当成热力学hc。

| L | 独立连续 JW 峰 | 三个模型峰位范围 | 最大绝对峰位偏差 |
| --- | ---: | ---: | ---: |
| 16 | .9778393 | .9602983–.9848286 | .0175410 |
| 24 | .9899764 | .9974770–1.0034375 | .0134611 |
| 32 | .9943101 | .9934375–1.0150897 | .0207796 |

L16局部窗口／抽取spread最多3.52e-5，但不同初始化的峰位跨度.02453，说明算法局部稳定不等于物理峰准确。L24相同检查spread最多.00924；L32最多.00425。三个MC主峰的二次拟合顶点未通过局部守卫，明确回退为grid peak；没有将其当高精度连续峰。细节见[峰位参考对照](results/phase0_frozen_confirmation_v18/peak_reference_check.json) 和[指数／峰位图](results/phase0_frozen_confirmation_v18/scaling_and_peak_stability.png)，图中峰位error bars表示分析形式spread，不是统计SE。

### 成本、验证与后续边界

实际新训练共27个去重检查点、11700 updates、74700 tasks、H131400、V56700、Born5932800；实际SR残差修正求解647次，最大true relative residual=9.97537e-7，满足1e-6守卫。并非所有求解都达到请求的1e-7，未把请求容差写成实际精度。下表把测量、独立核验与资源预检分开；H/V是逻辑算符作用计数，不能替代FLOPs或wall time。

| 本轮额外工作 | H actions | V actions | Born draws |
| --- | ---: | ---: | ---: |
| 369点首次确认／calibration | 492 | 123 | 16121856 |
| 18点响应补充 | 36 | 18 | 147456 |
| 369点独立重放 | 492 | 123 | 0 |
| Phase 1 read-only复核 | 648 | 324 | 0 |
| Phase 2 read-only复核 | 240 | 120 | 0 |
| 审计资源预检 | 2 | 0 | 65536 |

首次确认含360个held点及9个h=1训练字段calibration；L16累计8060928个穷举行，较大尺寸能量1007616行、FD503808行。独立重放再次使用8060928个L16穷举行及7872个独立local-energy行，无新增Born draws。Phase 2另有120次实际dense参考求解。旧Phase 1/2训练费保留在核验JSON，未当本轮新训练重复计算。

预声明的两个新seed worker复用原数值代码、完整三尺寸随机字段调度及PRNG，冷训练与参考测量分离。原顺序runner完成seed11后遇到另一个worker已启动的轨迹，按部分轨迹守卫退出；全部27个节点完成后只核验既有端点并生成九模型seal，没有重启或重复更新。原退出日志保留。核验跳过端点时重写的timing.json只表示skip耗时，实际训练用原日志fast100的elapsed markers另记于progression；并行worker、测试和编译混合，不能据此作公平wall-time比较。

新增数学／自由指数focused检查 **10 passed，2.29s**，随后FNQS全套 **169 passed，82.98s**。全仓 **2927 passed、6 failed、2 skipped，1701.76s**；六项独立重查 **4 passed、2 failed，2.55s**。四项累计RSS守卫独立通过，持续失败是既有spatial-ViT SR source hash与spin_finite_schur_reanchor seed freshness冲突。未改写外部artifact或放宽测试；全仓不是全绿。[测试记录](results/test_validation_v18.json) 绑定实际日志与新增测试源代码。

**本轮要求的确认、响应、自由指数／峰位检查及Phase 1/2复核已完成；Phase 0科学gate=false，Phase 1/2科学gate也未解锁。** 下一步优先固定L32，在新development协议中分离优化预算／步长与容量因素，保留energy-only／特征／宽度的首轮控制。v18 reserved已被打开，若用于诊断或方法选择，后续确认必须另声明未见字段与新初始化。L24/L32响应达标后再做至少四尺寸的自由标度稳定性，Phase 1同步补B/C独立参考，Phase 2保留现有工程运行器等待科学campaign。

## 固定 L32 的预算／步长辨因（v19，运行中）

保持v18的legacy特征、width16、energy-only、affine坐标、21个训练字段、每task64 draws及SR正则化／clipping不变。三个L32完整父端点各自分叉到lr=2/4/8，连续训练400次新增更新，第100次仅保存快照，不重启优化坐标。100／400是同轨迹预算端点；新增训练总量为3600 updates、75600 tasks、H151200、V75600、Born4838400。报告实际step norm与clipping比例，避免将被clipping抵消的名义学习率差异误认为有效步长对照。

全部18个端点封存后，对三个基线和18个端点使用相同32768 draws／field审计；v18已打开的40点在此明确为development。另预声明三个字段的8192-draw物理响应残差／HF核验。候选必须以同一个学习率在400步端点让三个seeds分别通过原χ、全部C(r)和能量门槛；没有事后逐seed挑选或自动续训。即使候选通过，也仍需新的冻结后确认。

初版分段训练的前置测试发现重复参数坐标往返会产生约2.33e-10的参数轨迹差异；运行器改为连续400步快照方案，没有放宽2e-12测试精度。修正后focused **18 passed，9.72s**。上轮交互中断停止了尚未完成的实验与全仓回归，留下1／50／1步快照，没有任何完整对照端点或新审计结果；原日志、快照和[中断记录](results/phase0_l32_budget_rate_v19/interruption_receipt.json)保留，不把partial suite当作通过。

重跑使用[独立新目录协议](results/phase0_l32_budget_rate_v19r/protocol.json)，数值方案及原父模型不变，未复用有损恢复的partial优化坐标。已保存的中断训练至少52 updates／69888 Born draws，最后未落盘工作未知，不能伪称这是精确总成本。新的supervisor将任务与交互会话分离，并保存进程退出状态；后台按“完整端点→测量→独立重放／比较”执行。目前尚无v19的科学结果。

另做了标为post-hoc的[精确解可辨识性检查](results/phase0_l32_budget_rate_v19r/three_size_exact_fit_diagnostic.json)：在完全相同的三个尺寸、字段与主拟合形式上，JW数据给出hc=1.000461、ν=.999121、η=.251430；窗口／形式spread分别.011406／.012379／.002439。这支持拟合管线在精确数据上可以工作，不能用精确解替代神经数据或改写v18 gate。
