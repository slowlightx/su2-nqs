# FNQS universality 工作约定

本目录是用户授权的 Z2 Ising universality methodology 子项目。TFIM 不具有 SU(2)，不要将它套入根项目的 singlet fusion Hilbert space。根目录工程与精确验证原则继续适用；现有 `su2_nqs` 和其他项目的封存 artifacts 不属于此目录的修改范围。

- 先读 `README.md`、`RESEARCH_PLAN.md`、`RESULTS.md` 和 `LITERATURE.md`。
- Hamiltonian 使用 Pauli X/Z、PBC，bit 0=-1、1=+1，site 0 是 MSB；固定 coupling ordering `(j1,j2,jx,h)`。不要将物理耦合的浮点数误用为角动量 Hilbert labels。
- 依赖方向：`basis` → operators/references 和 pure-JAX models → training/measurements → experiments。模型不得导入 Hamiltonian connectivity 或 ED。
- Reference chi 是 total coupling QGT；凡给每格点量，必须显式除 L。coupling derivative 在固定 theta 下计算；不要与 parameter-space SR 混用。
- Multi-system SR 必须在每个 Hamiltonian 内分别中心化，然后按 declared ensemble weights 合并。
- 当前模型是 positive-amplitude Z2 probability-mixture GRU，支持 stoquastic cuts，不能宣称是原论文 ViT 或通用 sign/phase model。
- PBC Renyi-2 的 CFT coefficient 是 c/4；von Neumann 是 c/3。SWAP replicas 必须独立，保留 mean/SE，不能把 log(mean) 的估计当作无偏量。
- 每项新的数学功能都加入独立 finite-system reference test，保留 float64/complex128 tolerances。先 focused tests，再全仓回归；旧 artifacts 的 provenance 失败不得通过重封存或放宽断言解决。
- 除非有新的科学证据，不改变已声明验收门槛。参考解 collapse 不算 neural reproduction；小尺寸训练或 canary 不算 universality transfer。
- 扩大计算前记录 sizes、fields、seeds、optimizer、updates 和 total sample budget；独立保留 train/held-family/held-field/held-size splits。代码使用 CLI config validation 拒绝 field leakage。
- 训练输出保留 checkpoints、Adam/PRNG、版本与 source hashes。已有输出用新版本目录；续训要求一致代码与训练签名，不覆盖失败结果。
- Phase 0 科学 gate 未通过前，优先解决 neural chi/correlation accuracy；Phase 1 可以做独立 ED/DMRG 勘探，但不解释 shared-training critical exponents 或声称 foundation 优势。

常用命令（仓库根目录）：

```bash
conda run -n nqs python -m pytest fnqs/tests
conda run -n nqs python -m fnqs reference --config fnqs/configs/phase0_reference.json --output /tmp/fnqs-reference
conda run -n nqs python -m fnqs train --config fnqs/configs/phase0_smoke.json --output /tmp/fnqs-smoke
```
