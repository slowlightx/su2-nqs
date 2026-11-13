# Core-library engineering conventions

## Mission and labels

Represent non-Abelian SU(2) structure exactly. Group theory fixes magnetic
indices; learned models represent fusion-path amplitudes or multiplicity
tensors. Establish exact small-system algebra before using it in VMC.

- All angular momenta and magnetic projections are doubled integers:
  `q = 2j`, `m = 2m_j`. Never store floating half-integer Hilbert labels.
- A left-associated path has length N. `path[k]` is doubled spin after sites
  0..k; it starts at1, changes by±1, never goes negative and ends at target_q.
- Autoregressive actions are ordered `(-1,+1)`. Apply exact reachability masks
  before softmax, never repair invalid samples afterwards.
- Computational bit0 means m=-1/2, bit1 means m=+1/2; site0 is the most
  significant bit.
- Use the Condon–Shortley convention in `su2/clebsch_gordan.py`.
- `f_move(q1,q2,q3,q_total)` gives `F[q23,q12]=<right|left>`; a pair operator
  in the left basis is `F.T @ D_pair @ F`.
- Connectivity means `H[new_path,old_path]`; combine repeated paths from bonds.
- Autoregressive `log(psi)=0.5*log(p)+1j*phase`; exact conditional
  normalization must imply exhaustive state normalization.

## Dependency direction

`su2` supplies labels/CG/6j/trees/recoupling; `operators`, `lattices` and
`benchmarks` supply connectivity and exact references; `models` contain pure
JAX models without Hamiltonian logic; `vmc` supplies generic optimization,
sampling, checkpoints and SR. Generate symbolic tables offline only; never
call SymPy inside JAX runtime. Preserve one consistent local-F-move convention.

## Required validation order

1. Path counts/masks, including singlets N=2,4,6,8,10 ->1,2,5,14,42.
2. Coupled-basis orthonormality and `S^2=S(S+1)`.
3. `F.T @ F=I` and explicit computational-basis recoupling.
4. `H_fusion=U.conj().T @ H_sz @ U` through N<=10.
5. Exhaustive autoregressive normalization to about1e-10 or better, then
   sampling-frequency checks.
6. Only then VMC or optimization interpretation.

Never weaken exact references to accommodate training. Use float64/complex128
for reference calculations. Every mathematical/sign change needs an
independent exact test. Read relevant modules/tests before editing, preserve
user changes, validate API labels, document public functions and keep basis
order deterministic. Run focused tests before the full bundled suite and
report tolerances and largest discrepancies. Core algebra must not require
Optax; optional integrations remain optional.

Research configurations, job scripts, notebooks, result artifacts and private
credentials are outside this core repository. Never publish them by widening
the source allowlist or by copying an entire research workspace.

## Git publication and staging (applies to every agent: Codex, Claude, ...)

The published repositories (GitHub `origin` and the Kim mirror `kim`) hold
ONLY the core library: `.gitignore`, `README.md`, `pyproject.toml`,
`AGENTS.md`, `su2_nqs/`, and a small `tests/` set of self-contained exact
tests of `su2_nqs` code. Never commit research tests, results, logs, configs,
notebooks, checkpoints, workflows, explorations, job scripts or other
experiment material (`projects/`, `fnqs/`, `peps-vmc/`, `archive/`, `data/`,
environment files, research `tests/test_*`).

The local workspace deliberately keeps hundreds of UNTRACKED research files
(including about 500 research tests under `tests/`). Keep them on disk; do
not delete them and do not track them.

- Never use `git add -A`, `git add .`, `git add tests/`, `git add -u` on
  directories you did not change, or `git commit -a`. Stage explicit file
  paths only, then check `git diff --cached --stat` before committing.
- Publish a new test file only if it tests core `su2_nqs` code, imports
  nothing from `projects/` or other research packages, and needs no external
  artifact or checkpoint. Run it in a clean checkout of the published tree.
- The local `.gitignore` may differ from the committed allowlist; do not rely
  on it to block research files, and never commit a widened allowlist.
- Keep `kim` and `origin` on the same branch commit. Push fast-forward only;
  never force-push or rewrite published history without explicit user approval.
