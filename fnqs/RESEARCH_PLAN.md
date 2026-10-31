# 研究计划与 go/no-go

研究链条：criticality → universal scaling → cross-Hamiltonian transfer → RG geometry。遵循你提供的 12–18 月计划，但把每阶段的代码完成、参考校准、科学结果三个层面分开。

## Phase 0：TFIM calibration

对象：一个共享 (psi_\theta(b\mid h,L))，PBC Pauli TFIM。研究尺寸 L=16,24,32,48,64；L=40 可作为未见尺寸。保留小尺寸 ED control。

已具备：精确 scalable oracle、normalized coupling/L GRU、独立采样、完整 local energy、精确与 VMC Adam、ensemble SR、输入 QGT、correlation/S2/SWAP、拟合与 checkpoints。原 SU(2) 约定按 AGENTS 要求顺序检查通过，新增数学组件各有独立检验。

已执行 `python -m fnqs.controls` 的六组小尺寸响应对照：固定 L=8 与共享 L=4,6,8；固定宽度 16；Adam 与 ensemble SR；五点与 21 点磁场覆盖。每组 1200 updates、每步五个主梯度 Hamiltonians，统一 seed=17。均匀 task minibatch 保持 declared ensemble energy/site 目标；额外 SR 回溯查询、精确 Hilbert 枚举量和 wall time 分开计数。共享尺寸改变每点枚举量，该对照不能直接支持 foundation 计算优势。

先在 development 网格比较方法，再用预先保留的 confirmation fields 和 seeds=29,43 等进行新训练确认，不用同一 held audit 反复挑最佳 checkpoint。训练方差、梯度范数、实际线性求解 residual 与能量曲线共同用于判断是否还需优化；observable 审计不能由能量平台替代。

本轮共享 SR 在 1200／2400 步后的最大 held chi 误差为 10.55%／10.09%，没有因接近 10% 就放宽门槛。随后声明新 confirmation fields，进行 50 步完整 63-task ensemble、较低 shift 的精修，误差降至 7.77%，小尺寸 observable 校准通过。这只验证 seed=17 的有限尺寸响应，仍须独立初始化确认方法稳定性；批量与正则化同时改变，需拆开比较才能作因果归因。

研究尺寸单步耗时／内存已校准，随后执行 L=16,24,32、seeds=17,29,43 的 200 步 SR pilot：21 个训练 fields、每步 3 tasks ×64 independent samples，审计每点 4096 draws。三个 seed 的最大 held chi 误差为 73.61%–84.43%，全部 nu 拟合触及下界，科学 gate 未通过。当前优先解决研究尺寸波函数和 coupling response 的精度，而非解释这些指数。

`phase0_vmc_sr_warm_refinement.json` 和 `phase0_vmc_sr_scratch_refinement.json` 是原始 refinement 预设：每步 6 tasks ×128 samples、1000 updates/seed、审计 8192 draws/point。2026-10-02 的 v5 配对批次因 scratch seed 29 的真实 CG residual 超限而停止，没有完整科学结果。v6 用独立 SVD 验证残差重求解，不增加 Hamiltonian／Born 查询或放松 guard，并以新确认字段重开相同预算；实际配置封存于 `results/phase0_pair_v6/protocol.json`。原预设文件保持不变。warm 的三 seeds 是同一预训练参数的随机优化副本，不能替代独立预训练 seeds 或等总成本 scratch control；父阶段参数、样本和 Hamiltonian 工作量须合并报告。先比较收敛诊断和独立 observable，再决定增大宽度、样本或更新数。

研究尺寸配置最终扩展至 L=16–64，保留 L=40 未见尺寸；临界窗口按 O(1/L) 缩放，每次预算或数值代码变化使用新输出目录。现有 `phase0_vmc.json` 是预算初值，尚未执行，也不保证可达到目标。

必须在独立 reference 下同时检查 energy、variance、C(r)、(chi_F)、fixed-theta AD vs finite difference；不能由低 energy 宣布成功。至少自由恢复 (|h_c-1|\le0.02)、(|\nu-1|\le0.05)、(|\eta-.25|\le.03)。扩大 field grid、改变 critical window、排除最小 L，检查 fit stability；最终用独立 draws 与 seed/size bootstrap 给置信区间。当前 collapse residual 只作描述，不作置信区间。(c) 可暂后移，S2 的系数始终用 c/4。

若 observable accuracy 不通过，先继续优化／增大可验证表达能力；避免把 auxiliary loss 训练过的 observable 当独立 evidence。通过后再比较 papers' ViT baseline，保持相同 couplings、尺寸、样本预算和审计协议。

2026-10-02 的含噪独立控制识别了旧 collapse residual 的 nu-dependent noise suppression。新的统计提取在固定物理 chi SE 下消去 scaling curve，不固定 Ising 指数，原测量、阈值和训练代码封存另存。旧拟合触及 nu 下界不是物理证据；修复后仍须先通过逐点 chi／C(r) 审计，再解释指数。当前三尺寸允许 window／degree／correction 敏感性检查，但不足以执行去掉最小尺寸后的三尺寸拟合或可靠的 size bootstrap。

若本批完整预算仍不能通过响应精度，下一批 development 对照先分离两个问题：同架构的固定 L 训练与共享 L 训练对比，定位尺寸共享的代价；共享模型较长优化与增大宽度的对照，定位收敛与表达能力。分别记录总 Hamiltonian／Born／线性求解成本，不能把额外计算的改善归因于架构。当前确认网格此后只作开发证据；任何方法选择后的最终确认必须重新声明磁场与独立 seeds，不追加预算后反复使用同一确认结果挑 checkpoint。

v8 已完成此对照，使用三个新冷 seeds=71,83,97，L=16,24,32。三个 fixed-width16 独立模型各 1000 步；shared-width16 与 shared-width32 各保存 1000／2000 步同轨迹端点。每尺寸每步两个字段，前 1000 步逐尺寸字段曝光完全一致；固定组合计／共享组各为 6000 Hamiltonian 查询和 768000 training draws。fixed 参数合计 4422、shared16=1474、shared32=4482；后者近似匹配总容量，仍保留 60 参数差异及独立优化／共享优化差异。预声明新的 20 点 development 与未测量的 40 点 confirmation，协议见 [size_capacity_controls_v8](results/size_capacity_controls_v8/protocol.json)，全部 15 条轨迹／21 个审计端点已通过来源与预算核验。

2026-10-03 的 [完整结果](results/size_capacity_controls_v8/comparison.json) 显示：共享宽度 16 相对固定模型的 χ 最坏点变化方向随 seed 改变；翻倍预算降低所有共享模型的 RMS 误差，但最坏点改善不稳定。增宽在两个预算、全部三个 seeds 下都未改善 χ，且增加 SR 残差重求解。最好的端点组 χ error+2SE 仍为 24.12%，科学 gate 未通过。这是声明预算下的行为比较，不证明任一模型已达表达能力上限或优化已完全收敛。

下一轮优先在小尺寸独立精确参考下诊断条件波函数的 coupling tangent 和响应曲线形状，分别检查波函数精度、输入导数与物理响应；若改变目标函数、采样量或 SR 条件化，逐项控制并重新封存协议，不与宽度变化同时修改。任何辅助响应目标用过的 observable 不再作为独立确认依据。仅在新的 development 对照达到精度后，使用保留的 confirmation fields 与另外声明的独立初始化确认，再扩大尺寸。当前不自动追加第三种预算或启动 Phase 1 shared training／Phase 2 transfer。

2026-10-03，按用户新指示，v9 已完成固定参数耦合输入／物理响应诊断：[135 点结果](results/coupling_response_diagnostics_v9/diagnosis.json)。AD／FD 最大 centered-score relative RMS 差异为 2.68e-5，差分步长减半后全部点约缩至四分之一；物理 projected-response relative residual 为 0.0456–0.3644。当前优先处理学到的 Hamiltonian manifold 响应，未见应通过修改 AD 实现解决的证据。该残差不是 χ 误差的界，不能用它替代独立 χ 审计。

**当前主路线采用固定 L 的 coupling-shared FNQS**：分别训练 L=16,24,32，各自覆盖同一耦合族，再汇合独立物理量进行 FSS。L 是已知系统几何，原先也是输入特征而非优化参数；不要求同一个 θ 跨 L。Phase 1/2 的 family sharing／holdout 也可分别在每个 L 执行，之后比较指数及成本。跨 L zero-shot／共享权重成为附加对照，不是进入下一阶段的前提。

`variants/size_features.py` 已加入三种 policy，保留独立的原模型语义：`legacy`；`no_explicit_size` 去掉 log L、1/L，仍含 site/L、remaining/L 和累计磁化/L；`no_length` 改用固定 geometry_scale 的位置／累计磁化，去掉所有总长度提示。后者会强制不同长度前缀的 Born 分布一致；独立 PBC ED 的 L=4/6 前缀概率最大差为 0.01581，故不能要求这种 AR 同时精确表示两者。这一限制针对该 recurrent AR，并非所有未显式输入 L 的架构。

policy 的 active 参数计数只扣除尺寸 mask 的不活跃行，不包括 A-cut 的 j2/jx=0 或固定 L 常量特征的秩退化；实际优化条件化仍需分别检查。当前大尺寸响应结果仅针对 h，不能用来声称未训练 interaction directions 的导数准确。

下一批尺寸特征实验应 **从随机初始化重新训练**，先在固定 L 比较 legacy 与 no_explicit_size；只有在保持耦合网格、SR、样本、更新预算及配对初始化一致时才解释特征变化。width16 nominal 参数均为 1474，但 structurally active 参数分别为 1474/1346/1298，须记录有效容量变化；不称为纯尺寸因果控制。不要把现有检查点直接套入新 policy 当作已训练对照。新 policy 尚未接入原 `fnqs train` CLI，配置刻意不兼容旧训练器以避免被静默忽略；正式冷实验前需开发使用该 API 的独立运行器并封存完整预算。

响应精度改善应作为另一个独立对照：先在有谱参考的小系统检查 tangent 的方向与低能分量，再试更充分的固定 L 耦合训练或逐项声明的响应约束。未经验证，不直接把 unweighted response residual 作为唯一训练目标；v9 中更宽模型的物理残差较低，但 v8 χ 反而较差，说明二者不可互换。响应约束用过的 χ／coupling point 只能作 development，确认继续使用保留字段和新声明的独立 seeds。

## Phase 1：同一 class 的多 microscopic cuts

**2026-10-03，用户调整阶段推进规则。** 用户表示“若相差在一个量级内，可以考虑推进到后续步骤”。[门槛距离核验](results/gate_distance_v10/assessment.json) 以原 error+2SE／原精度门槛比较：全部 development 端点 χ 为 2.41–6.82 倍门槛；固定尺寸路线为 2.41–3.27 倍，关联与能量也满足“不超过原门槛十倍”的条件。因此允许与 Phase 0 修复并行准备并执行 **固定 L、小预算的 Phase 1 neural pilot**，先使用可独立校准的 A／弱 B／弱 C 参考，声明冷初始化 seeds、点位、优化／采样预算及 pointwise control 后运行。该用户指示取代本计划此前对探索性 Phase 1 训练的全面等待要求；原科学验收门槛仍保留，旧结果和 `phase0_scientific_gate=false` 不变。

该核验仅覆盖可量化的 observable 精度。最新批次没有可靠的自由 hc／nu／eta 及拟合稳定性确认，保留 confirmation fields 仍未使用，故不将“相差不到十倍”称为完整 Phase 0 通过。Phase 1 pilot 以耦合共享、不同 cuts 的状态／响应精度和训练条件化为目标；大尺寸 class assignment／universality 或 foundation 优势仍需独立参考及正式验收。Phase 2 可同步开发 ancestry／exact-cost runner，在 Phase 1 pilot 核验后安排小 canary；不由本次距离评估自动启动完整 transfer campaign。

共同 super-Hamiltonian 的 A=(J2=Jx=0)、B=(J2>0,Jx=0)、C=(J2=0,Jx>0)。已具备唯一 connectivity、独立 Pauli 构造测试和 ED 勘探 CLI；先扫描 B 的 0.1/0.2/0.3 与 C 的 0.1/0.2。

有限尺寸 susceptibility peak 只是 critical bracket，不能把它当 (h_c)。必须 refine grid、检查 parity、gap、序参量和有限尺寸漂移；再接入更大 L 的独立 DMRG／可适用 QMC references。若发现 multicritical/floating/first-order 情形，移除相应 candidate 并记录依据。

共享 A+B+C 训练以后，分别拟合 (h_c^{(m)},\nu_m,\eta_m)。跨 family collapse 需要非普适 scaling-field 和 amplitude metric factors，例如 (x=a_m(h-h_c^{(m)})L^{1/\nu})，scaled observable 再除 b_m；仅用 ((h-h_c)/h_c) 不保证不同 microscopic curves 自动相同。几何 QGT 还要明确坐标 transformation (g'=J^TgJ)。不得预置相同指数后把 collapse 称为 universality 恢复。

Phase 0 科学 gate 未通过时，继续完善独立 references；按上述用户新规则，可做固定尺寸的小预算 neural pilot。大规模正式训练与 critical-exponent／universality 解释仍需相应参考和验收证据。

2026-10-02 已把 parity ED／projected-response 参考扩展到 L=14，步长 0.005，所有 finite-size peaks 均被 bracket。增加 same-parity／opposite-parity gaps、完整 C(r)、S2 和缩小 dh 的 fidelity audit。下一项参考工作是独立的 L≥16 DMRG／可适用 QMC 及 candidate class verification；N≤14 tables 不能替代它们。Phase 0 的新 warm／scratch 对照已完成，但冷初始化最大 held chi 误差仍为 26.45%–44.20%，修正含噪拟合后科学 gate 仍未通过，不打开 shared-training gate。

## Phase 2：leave-one-family-out transfer

三组 reciprocal experiments：(A+B)→C、(A+C)→B、(B+C)→A。训练 split 在任何 pretraining／model selection 中排除整组 held family；仅共享 local Hilbert space 或显式 operator metadata 不算见过该 family 的数据。

比较 frozen zero-shot、1/2/4/8-field few-shot、同架构随机初始化 scratch，以及 pointwise NQS。对齐 total optimization updates、Hamiltonian queries、total Born samples 与参数量；记录预训练成本及适用摊销场景。指标包含 ΔE、variance、hc、nu、eta、collapse error 和达到同一 observable 门槛的成本。没有这些 controls 就不提出 foundation advantage。

固定四维 coupling input 只是 baseline。真正 unseen operator-content 测试前，实现可变 operator/range/coefficient token encoder；保留固定向量 control，区分新 interaction type 和已知 interaction 的新 coupling。

2026-10-02 已封存三折 Phase 2 协议和 coupling-based holdout validator。初始宽网格用于工程 split 验证，正式 critical-window 与 exact-cost update 调度仍需 Phase 1 校准。禁止在整 family 持出中复用 A+B+C 权重；持出 A 时，祖先链也不能包含 Phase 0 A。当前只有协议／split 层，不是已完成的 transfer runner；完整 checkpoint ancestry 校验、独立 confirmation 和聚合 pointwise 成本是正式执行前的要求。

## 2026-10-04：固定尺寸并行开发的已执行范围（v11）

用户授权后，Phase 1 小预算实验与 Phase 0 修复分别运行；协议、完整任务调度、冷 seeds、父检查点和源代码均在训练前封存。[v11 review](results/fixed_size_pilots_v11/review.json) 记录实际结果，原科学门槛不变。

Phase 1 固定 L=12、width=8、no_explicit_size，冷 seeds=113/127/139。选择 A、B(J2=.1)、C(Jx=.1)，使用已封存有限尺寸 χ peak 定位窗口，不能将这些中心视为热力学 hc。每族五训练点、四 development 点及四未测 reserved 点。每 seed 比较一个 A+B+C 共享模型和三个各族独立、族内 h 共享的模型；各族均为 150 task evaluations、64 draws/task，整体 H／Born 预算相同。共享模型 546 参数，独立组合计 1638 参数；更新数 150 对合计 450，故不能称为单独隔离容量或优化条件的对照。它们是 family-wise controls，尚非 Phase 1 正式多预算／容量验证。

Phase 0 固定 L=16，从 v8 三个独立父模型 seed=71/83/97 分叉，每分支重置 SR/PRNG，同 50 updates、2 tasks/update、64 draws/task，比较仅能量与能量+.02ρ²。保持 legacy 特征，避免同时改变输入特征。ρ 是投影响应／source norm；参数梯度包括冻结 proposal 的 self-normalized Born importance weights 导数，其有限 batch plug-in 有偏。没有 ED/JW 的 χ 标签进入优化。两臂都计算相同响应诊断；H actions／draws 相同不代表微分 FLOPs 或 wall time 相同。当前三种子收益不一致，不默认采用该 λ 或作为唯一损失。

Phase 2 新运行器完成 three-fold L=6、一个 seed 的工程 canary，包括 frozen zero-shot、1/2/4/8 点适配、每个 k 的同架构 scratch 和四个 pointwise models。每优化臂总八个 tasks、256 draws，预算在训练前封存。逐点模型总参数量和累计 work 相加；共享祖先按内容去重，同时报告每个部署端点的 inclusive cost。每折全部端点先 seal，再读 held reference；所有 pretraining 参数祖先的训练／selection points 必须排除整族。A+B+C Phase 1 权重不允许作为任何一折的 seen-only 父模型。该 canary 验证运行器而非科学迁移，未来大尺寸 campaign 需另行封存研究配置和独立参考。

下一批科学实验优先恢复固定 L 的 coupling tangent：Phase 1 窄窗能量下降但 χ 几乎不响应，需在新协议中对照更宽 coupling 覆盖／更充分曝光；Phase 0 的响应约束须比较权重与状态误差的 tradeoff，并检查低能 tangent 分量。仍不从单一 ρ 推断 χ 精度。不追加 v11 的既定预算；开发选择后使用保留字段和新独立 seeds 确认，再扩展 L 和指数提取。

## 2026-10-04：辨因结果与联合更新修复（v12–v14）

按用户“先辨因、找到问题再推进方案”的指示，固定 L12、width8、no_explicit_size、三个配对冷 seeds=113/127/139，先执行窗口±.04／±.15 ×150／600 单点 SR updates 的 2×2。两种预算是同一轨迹的封存端点；主比较使用同四个中心 development fields，另报十四个宽区域 fields。扩大覆盖与增加更新都没有三个 seeds 一致的改善，不能把失败归因于在该范围内单纯曝光不足。

随后另行封存有限的更新组织对照：同宽窗口、每点 64 draws、同顺序的 600 task evaluations，比较 600 单点 updates 与 120 五点联合 ensemble SR updates。后者 χ 中心最坏误差为 20.60%／24.99%／17.70%，前者约 98.7%–98.8%；完整 C(r) 与 energy 亦改善。按预声明条件，已执行 A/B/C 共享及三个各族模型的联合更新复验，共 12 条新冷轨迹，各族 600 tasks。共享 B 中心 χ=8.22%／5.44%／6.44%，A/C 仍在约 16.8%–30.5%，宽区域仍有响应／关联超限。[完整结果与成本](results/progression_v14.json) 保留全部 seeds 和次要区域。

这支持优先采用每次联合多个 couplings 的 SR 作为当前修复候选；不证明 GRU 容量足够，也没有唯一分离每次噪声、metric 条件化与更新数。共享／独立总 H 与 Born 预算相同，参数总量及 aggregate updates 不同，仍无 foundation 优势结论。全程 energy-only、没有 χ 标签训练，辅助响应损失的 λ=.02 仍未默认采用。

接下来先声明固定 L16 的配对 development 验证，保持 legacy 特征、父权重／PRNG、SR shift 与损失一致，按 total H／Born 工作量比较逐场与多场联合更新。不要同时切换尺寸特征、增宽或加响应损失。若需要进一步分辨 batch noise 与几何，增加同更新数／同 Born 总量的单场大 batch control，并明确其 H-action 数不同；所有新预算和字段都在训练前封存。方法在 development 上稳定后，另用新冷 seeds 与保留字段做独立确认，再扩展固定尺寸 L16/24/32 并提取自由指数。L12 结果不自动解锁正式 Phase 0 gate 或 Phase 2 科学 transfer；既有持出／祖先／成本运行器继续可用。

## 2026-10-05：固定 L16 配对与响应修复（v15–v17）

2026-10-05，用户授权优先解决Phase0响应缺口，并允许容量更大的后续方案。v15已完成固定L16配对，保留legacy width16与energy-only；1050逐场对50全21场联合更新的total H／Born预算一致。三个seed的χ最坏误差为逐场20.84%／24.79%／13.42%，联合22.54%／23.96%／14.99%，没有稳定收益，原10%门槛未通过。当前不把L12方法选择直接当L16解法。

v16继续保持物理GRU、width、loss和初始物理波函数不变，仅检验优化器内的可逆仿射参数坐标。先用50步作同预算对照，再按提前声明的200步端点检验进一步收敛；较长端点没有raw-coordinate200控制，归因须保留此限制。若响应仍不能达标，先用同一200步父模型、同一终端PRNG检验学习率0.5与4的配对，每支100次联合更新；保持特征、width16、energy-only、samples64、SR shift=.001及max_step=.05。各支只有一个预声明审计端点，不挑选中间χ最优检查点。所有原confirmation fields继续保留，科学验收条件不变。

若更充分的优化仍不能恢复响应，再执行独立声明的容量修复，不以额外参数或低能量本身作为成功证据。固定L16实验只解决该尺寸的状态／tangent精度；三尺寸、自由指数和独立确认仍需另行完成。

v17现已完成：从各自v16 step200父模型分叉的100步同预算对照，学习率.5的χ最坏误差为14.15%／13.48%／4.79%，学习率4为6.42%／9.33%／3.56%；后者三个seeds的全部C(r)及能量也通过原阈值。实际保持1474 nominal参数、legacy特征、width16和energy-only。先将可逆坐标＋更充分联合优化列为L16候选，不需要在本次development内加宽。[完整来源／成本／验证](results/progression_v17.json) 已封存。该候选还需方法冻结后的独立初始化和40点reserved确认；随后在固定L24/L32重新校准并审计χ+2SE、全部C(r)和E，再汇合多尺寸数据做自由hc／ν／η与敏感性检查。不要把单尺寸χ达标或离散peak当作完整Phase0科学通过。

## 2026-10-05：冻结后确认与阶段复核（v18）

本轮按 [冻结确认协议](results/phase0_frozen_confirmation_v18/protocol.json) 完整重复原候选：三个新冷seeds=2026100511/2026100513/2026100517，每个seed分别训练L16/L24/L32；legacy width16和energy-only保持不变。每端点1000步原预训练、200步等价坐标联合SR、100步lr=4联合SR；更新、H/V和Born总量在运行前声明。按seed并行调度的补充协议保留原三尺寸随机字段调度和逐步PRNG，数值方案没有变更。九个最终端点seal后才打开保留的40点确认字段。

L16以穷举数据核验，L24/L32以独立Born采样测χ与全部C(r)，使用误差+2SE和原阈值。另预声明三点物理响应方程残差/HF诊断，保留独立算符重放。自由hc/ν/η、窗宽、修正项、关联距离起点、paired-seed/MC bootstrap，以及峰位局部窗口和网格稀疏检查均预先固定。h=1仅为训练字段calibration。三个尺寸不足以做删除最小L后的三尺寸fit；目前仅报告该限制，不扩大既定确认预算，不据pooled均值隐去seed失败。

v18实测已完成：[完整复核](results/progression_v18.json)。L16三个新seed的χ最坏误差8.03%／9.88%／6.85%均通过；L24为12.56%／17.54%／15.81%，L32为29.87%／22.98%／22.08%，计入2SE后全部失败。九模型全部C(r)及能量通过，独立算符及AD/FD核验通过。pooled自由hc=.9816、ν=1.234、η=.170，窗口／拟合形式不稳定；峰位仍有初始化与物理偏差。原Phase0 gate=false，不能以L16确认或pooled hc接近1解锁科学阶段。

下一轮首先在固定L32的新development协议下做配对收敛实验：先保持特征、width16和energy-only，单独比较优化预算与步长；若充分优化后响应仍失败，再用匹配预算的容量对照。当前能量和C(r)通过而χ失败，没有证明容量是唯一原因。v18已打开的reserved字段可作已见诊断，但不得再次充当方法选择后的独立确认；候选冻结后另外声明未见字段与新初始化。响应通过后再扩展第四个尺寸，进行删最小L、自由指数、峰位与拟合形式核验，不在失败确认轨迹上追加预算。

Phase 1/2的 [独立复核](results/phase_checks_v18d/receipt.json) 已完成，旧artifact未重封存。Phase 1的324点及12个端点来源/成本通过，但宽窗口χ、A/C精度、独立确认和B/C较大尺寸class references仍缺。Phase 2的39个检查点、三折整族祖先持出、12组few-shot/scratch任务/预算/PRNG与120点独立dense谱和均通过；L6单seed只是engineering canary。后续成本比较同时计入预训练，pointwise仍作为直接训练审计字段的监督控制。科学Phase 1/2的推进依据新确认与独立参考，而不是工程核验状态。

## Phase 3：普适指纹与 adversarial classes

Ising 对照 XXZ 的 c=1 gapless region；BKT 暂作后期 stress test，因为它不服从普通 power-law correlation length。再考虑 spin-1 Blume–Capel tricritical benchmark，专门检验相同 Z2 symmetry 的不同 IR fixed point，并显式控制 local dimension 改变。

网络仍输出波函数而非 class label。指纹 (U=(\nu,\eta,c,p_\chi,p_m,\ldots)) 由独立测量获取。不得把 symmetry classification 等同于 universality inference。

## Phase 4：QGT 与 RG geometry

先用 (J2,h) 二维空间建立独立 critical line (h_c(J2))。比较 metric eigenvectors 与 boundary normal/tangent，分析 eigenvalue 的 singular 与 analytic extensive background。

metric eigenvectors 随 coupling coordinate rescaling 改变；比较前固定物理单位、坐标 metric 与规范，报告坐标敏感性。沿切线 susceptibility 也可能有 O(L) 非普适背景，不要求它趋零。主 eigenvalue 的 (L^{2/\nu}) leading term 要与 subleading corrections 分开。

QGT 最大方向接近法向本身不是完整 RG reconstruction；还需 finite-size stability、coordinate controls、independent critical-line uncertainty 和跨 family 验证。通过后可尝试 R3 的 NFS 动态 scaling 作为额外 physics check，不把它混入当前 static reproduction。
