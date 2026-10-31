# 文献方法与实现对应

初次核读日期：2026-10-01；2026-10-02 再次核对三篇原文及 arXiv submission history，R2 仍为 v1、R3 仍为 v2。以下区分论文的方法、当前借鉴部分和尚未复现内容；预印本固定版本，后续更新应记录版本差异。

## R1：共享 Hamiltonian manifold 与 coupling QGT

[Foundation neural-networks quantum states as a unified Ansatz for multiple Hamiltonians, Nature Communications 16 (2025)](https://www.nature.com/articles/s41467-025-62098-x)。

核读 Results 的 ensemble variational objective、coupling-conditioned ViT、TFIM fidelity scaling，以及 Methods 的 multi-system SR 和 generalized fidelity susceptibility。它用共享参数联合优化 Hamiltonian ensemble，通过输入耦合的 log-amplitude 导数 covariance 计算量子度量。

当前映射：`models.py` 的显式 coupling inputs；`training.py` 的 ensemble energy 与分别中心化的 SR；`observables.py` 的 Born-weighted QGT；`scaling.py` 的自由临界拟合。与论文不同：采用 GRU，并显式 conditioning on L、按 energy/site 平衡尺寸。尚未复现论文 Transformer、N=100 或其蒙特卡洛精度；没有加载 pretrained weights。[论文公开架构入口](https://huggingface.co/nqs-models)留待后续公平 baseline。

2026-10-03 再核对 R1 Results 的定义、固定 N=100 TFIM 示例及 Methods 输入导数：FNQS 核心是共享 θ 覆盖耦合族；L 不必作为可微耦合输入。当前把固定 L、耦合共享作为主路线，跨 L 共享作附加实验。这是本项目的阶段选择，不声称原文的不同 N collapse 使用或没有使用同一组 θ。v9 的 AD/FD 与 projected-response equation 分开检验；后者沿用本项目独立谱扰动推导，不归称为 R1 的训练损失。

Fig. 2 的五点训练展示 energy／magnetization；fidelity susceptibility 与 N=40,80,100 collapse 则使用 [0.85,1.15] 内 6000 个 coupling points。不能把当前五点 GRU 的响应问题解释为该论文方案已经失败。本轮新增固定计算预算的稀疏／密集 coupling 对照及实际 ensemble SR 实验；本地 CG 和精确回溯是本项目的数值实现选择，并非声称论文使用同一求解器。

论文正文的 TFIM (S^{x,z}) 符号与临界比值需要谨慎核对。此项目直接固定 Pauli Hamiltonian，并用独立 ED 校验所有系数，不依赖符号推断。

v5／v6 对照固定每步总训练 draws=768，并从 63 个尺寸／磁场 tasks 中抽取 6 个；目标仍是等权 task 的 energy/site 平均，每个 task 内独立中心化。R1 Fig. 2 的固定总 sample budget 思路是比较依据；随机 task minibatch、混合尺寸和本轮具体 SR 超参数是本项目选择，尚未复现 R1 的 ViT 结果。

v6 的真实残差重求解和 v7 的固定物理 measurement-error collapse 权重是本项目数值修复，分别用独立 SVD 和非 Ising 指数的含噪控制检验；不归称为三篇论文的实现。已保存的 v6 波函数测量与 v7 后处理源码分开封存。当前独立 chi／correlation 审计未通过，不能用低能量误差或较低拟合 residual 声称已复现论文的 universality 结果。

2026-10-04 再核对 [R1 的 theoretical framework 与 Methods multi-system SR](https://www.nature.com/articles/s41467-025-62098-x)：共享 θ 的联合变分目标与多个系统合并的 SR 是当前更新组织复验的方法依据。v12 的窗口／预算对照没有稳定恢复 χ，随后 v13 在相同总 task、H／Born 预算下比较逐场 SR 和五场同时 SR，三个冷 seeds 均显著改善。v14 已将联合更新用于 A/B/C 共享及族内独立模型。这些是本项目 GRU 和小尺寸上的实测，不是论文 ViT 精度或 universality 的复现。

本项目的代数解释是：分别求解各系统的正则化 SR 方向后再轮流更新，一般不等于先合并各系统中心化的 metric／force 再求解；矩阵求逆与平均不交换。这是本地推论。v13 同时改变每 update 的 batch size、条件化和总更新次数，不能从 χ 改善唯一判定是哪项主导，也不能声称所有单系统 SR 都有错误。所有 ED/JW 标签均只用于封存后的审计，未作为联合更新的监督目标。

2026-10-05，v15 已将这一候选用于固定 L16 的同父模型／同 H 和 Born 成本配对，保持 legacy width16 与 energy-only。χ 改善没有在三个 seeds 中复现，因此不把 L12 结果推广为一般结论。v16 的可逆参数坐标修复是本项目独立的优化条件化实验：原物理 GRU 函数族、初态及 physical Γ 导数不变，而 δI 正则化和 Euclidean clipping 的作用依赖参数坐标；不能声称是完全坐标不变的 natural-gradient 实现，也不归称为 R1 的具体训练技巧。v17 对同一 v16 step200 父模型再比较学习率 .5／4，其他设置与新增预算匹配，检验剩余收敛速度。没有用 χ 标签训练或将 reference collapse 记作 neural reproduction。

v17 学习率4的三条warm分支已在L16全部20点development通过原χ／C(r)／E门槛，χ最大误差6.42%／9.33%／3.56%。这是本项目GRU的优化修复结果，未复现R1 ViT或universality scaling；没有以三个已参与方法选择的seeds替代独立确认。尤其单尺寸离散peak位置仍有偏差，后续自由hc与指数必须另外验证。

v18将选定方案冻结后，用三个新初始化分别训练固定L16/L24/L32，所有端点封存后才测保留字段。L16确认通过，L24/L32响应失败，自由hc/ν/η及峰位检查也没有建立可靠标度结果。该实验继续借鉴R1的ensemble能量目标和固定θ耦合QGT；本项目的GRU、可逆参数坐标、held-field独立确认、χ/C联合影响函数covariance、三seed/条件MC bootstrap及局部峰位守卫均是本地设计，不归称为论文原实现。自由η在自由拟合hc处提取，未用已知Ising指数或hc替换失败的神经数据。[完整结果](results/progression_v18.json) 区分估计器实现核验、神经物理误差和拟合不稳定性；未因精确参考或低能量误差声称复现R1的universality结果。

v19的可行性判断须限定到具体实现：目前重复失败削弱了“小GRU、21个离散耦合点、energy-only就足以稳定提取大尺寸输入响应”的假设。Phase 2科学campaign尚未执行，不能把工程canary的不准确结果记成独立的迁移失败。再次核对[R1作者版v3的Fig.2与架构](https://arxiv.org/html/2502.09488v3)：χ示例使用ViT、6000个训练couplings以及N=40/80/100的collapse，与本项目存在实质差异。该论文为总体FNQS方法提供已有正面证据，但不保证当前GRU或新的跨族universality目标可行。v19固定L32的预算／步长实验运行中，不能在结果出来前声称已找到收敛或容量原因。

## R2：coupling optimization 和 geometry

[Designing Strongly Correlated Quantum Phases of Matter with Foundation Neural-Network Quantum States, arXiv:2609.40209v1](https://arxiv.org/html/2609.40209v1)，2026-09-30。

核读 Methods II.1，尤其式 (1)、(6)、(7)：局部 ensemble ground-state training 后，固定神经参数，对目标物理量的 couplings 求导；coupling QGT 可作为 natural-gradient preconditioner。论文指出该 metric 是 generalized fidelity susceptibility，并采用 MARCH 来处理优化。

当前映射：明确 `coupling_scores` 与 parameter scores 的不同用途，按固定 theta 测量 geometry，保留全部四维 covariance。尚未实现 inverse design、MARCH 或 coupling natural-gradient optimization。Phase IV 中 QGT eigenvector 与 critical-line normal 的比较是本项目拟议实验，不能写成该论文已经证明的 RG 结论。

## R3：fidelity-selected subspace 与可复用算符

[Kibble–Zurek Dynamics in Two-dimensional Frustrated Systems with a Neural Foundation-state Subspace Method, arXiv:2609.05556v2](https://arxiv.org/html/2609.05556v2)，2026-09-18 修订。

核读 NFS construction、式 (4) 的固定状态线性空间、overlap/projected Hamiltonian，以及 End Matter 的 fidelity selection。它以共享 FNQS 生成 ground-state snapshots，沿 coupling path 用 infidelity 选择少量状态，预先投影各 Hamiltonian 项，跨 ramp time 复用矩阵。

当前映射：`subspace.py` 的 ordered snapshots、两端点保留、rank-filtered overlap whitening 和 Ritz control，经小系统精确验证。没有用此 control 替代神经态；尚未实现实时动力学、Kibble–Zurek collapse 或论文 2D 算例。可在静态 gate 通过后作为独立动态扩展。

## 独立校准依据

[Damski, Fidelity susceptibility of the quantum Ising model in the transverse field: The exact solution (2013)](https://arxiv.org/abs/1212.1528)：有限偶数 PBC 的动量求和及 critical identity (chi_F=L(L-1)/32)，并提醒 finite-size peak 不恰在 (h=1)。`exact.py` 使用求和以避免 closed form 在 (h=1) 的消去；检验用独立谱扰动与 ED fidelity。

[Pfeuty, The one-dimensional Ising model with a transverse field (1970)](https://physics-legacy.pbsci.ucsc.edu/~sriram/Courses_All/Physics-220-2011/pfeuty_Ising_Transverse_Field.pdf)：自由费米子参考和关联 determinant。实现针对本文 Pauli normalization 推导，并通过 ED 校验 energy、全部 separation 的 correlations 和 S2。

[Calabrese–Cardy, Entanglement Entropy and Quantum Field Theory (2004)](https://arxiv.org/abs/hep-th/0405152)：CFT interval entropy。PBC Rényi-n slope 是 (c(1+1/n)/6)；本项目严格区分 n=2 与 von Neumann。

后续在 B/C 上进行 independent reference 时，应另补连续临界线的 DMRG／有限尺寸文献。这里的弱耦合参数是 reconnaissance candidates，不是已由文献担保的 universality assignments。

2026-10-04 的 v11 沿用 R1 的 shared-theta／coupling-conditioned ensemble variational objective 和 coupling QGT 分工，开发固定 L 的 A/B/C 小预算对照。公开论文入口另有 [PMC 镜像](https://pmc.ncbi.nlm.nih.gov/articles/PMC12325958/)；本轮再次核对出版页面的 theoretical framework。所用 normalized GRU、尺寸特征 mask、projected-response auxiliary loss 和祖先成本 DAG 是本项目实现，不归称为 R1/R2/R3 的原方法。尤其 ρ² 辅助损失及冻结 Born proposal 的参数导数是本项目独立推导，并用稠密 Hamiltonian／有限差分校准。未以论文的泛化结论替代本地 χ／correlation 审计，也未从单尺寸结果推断 universality。

2026-10-02 的有限尺寸参考采用本项目独立推导的 parity sector 和 projected ground-state response 方程，经过完整 Pauli 谱求和及 JW 对照验证；不将它归称为 R1/R2/R3 的具体算法。SciPy CG 的接口、SPD 条件和 residual 定义核对了 [官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html)。B/C 的 class assignment 仍需匹配其准确算符／系数的文献与较大尺寸参考，不能套用 longitudinal-field 或 frustrated ANNNI 的结果。
