# ClebschTree NQS

当前工作仅由两份正文档管理：

- [SU2NQS_all.md](SU2NQS_all.md)：项目问题、架构、物理协议、研究顺序、文章与汇报大纲。用户手工决策栏保持独立。
- [RESULTS.md](RESULTS.md)：已核验的数值、原始证据路径、质量限制与续跑回执；不是实时队列监控。

绘图直接使用仓库根目录的 [三角格 notebook](../../TRIANGULAR_J1J2_RESULTS.ipynb) 和 [方格 notebook](../../SQUARE_J1J2_RESULTS.ipynb)。数据保持内联数组，便于手改；主图统一标记 `SU(2)`，文字注明实际选用的树。探索性结果不进入精选图。

目录职责：`workflows/` 为实验驱动，`configs/` 为执行协议，`remote/` 为集群封装，`references/` 为文献数值／来源，`results/` 为精选证据、当前续跑血缘及仍被通用测试使用的兼容数据。数学与模型实现位于仓库 `su2_nqs/`。

旧计划、报告、探索 notebook、图像和废弃实验的快照测试，已移出活跃目录，见 [可恢复归档](../../archive/README.md)。旧配置／驱动保留为源代码，并不意味着对应实验仍在开展；恢复旧实验时须先按归档清单恢复输入，不能把旧日志当作新计划。

HubbardCG 独立管理于 [../hubbard_cg/](../hubbard_cg/README.md)，不并入当前 spin-only 主线。
