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
