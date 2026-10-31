# AGENTS.md

## Mission

Build neural quantum states directly in non-Abelian SU(2) multiplicity
spaces. Group theory fixes magnetic-index structure; learned models represent
only the remaining fusion-path amplitudes and correlations. Establish every
group-theory component with exact small-system checks before using it in VMC.

## Non-negotiable mathematical conventions

- Represent every angular momentum and projection by doubled integers:
  `q = 2j`, `m = 2m_j`. Never store floating-point half-integers as Hilbert
  labels.
- A left-associated path has length `N`; `path[k]` is the doubled total spin
  after sites `0..k`. It starts at `path[0] == 1`, changes by exactly `+/-1`,
  never becomes negative, and ends at `target_q`.
- Fusion autoregressive actions are ordered `(-1, +1)`. Apply the exact
  reachability mask before every softmax; do not repair invalid samples after
  generation.
- In the computational basis, bit `0` means `m=-1/2`, bit `1` means
  `m=+1/2`, and site zero is the most significant bit.
- Use the Condon-Shortley CG convention implemented in
  `su2/clebsch_gordan.py`.
- `f_move(q1,q2,q3,q_total)` returns `F[q23,q12] = <right|left>`: rows are
  right-associated channels and columns are left-associated channels. Thus a
  pair operator in the left basis is `F.T @ D_pair @ F`.
- Connectivity follows `H[new_path, old_path]`. Combine repeated paths from
  different bonds before returning them.
- Autoregressive amplitudes are `log(psi) = 0.5*log(p) + 1j*phase`; exact
  conditional normalization must imply exhaustive wavefunction normalization.

## Architecture and dependency direction

Keep the layers narrow and one-way:

1. `su2_nqs/su2`: quantum numbers, paths, CG/6j data, trees, recoupling.
2. `su2_nqs/operators`, `lattices`, and `benchmarks`: Hamiltonian
   connectivity and independent exact references.
3. `su2_nqs/models`: pure-JAX normalized autoregressive models; no Hamiltonian
   logic.
4. `su2_nqs/vmc`: generic sampling, local energy, gradients, optimization,
   checkpoints, and SR linear algebra.

Do not call SymPy or perform symbolic work inside JAX runtime. Generate or
validate symbolic tables offline, cache plain numeric data, and load arrays at
runtime. The current CG and Racah implementations use finite numerical
factorial sums and may be checked against SymPy in tests or offline scripts.

The left-associated chain is the scalable 1D baseline. Generic binary fusion
trees and computational-basis overlaps are exact small-system references.
Geometry-aware 2D trees should reuse local F-moves rather than introduce a
second convention.

## Required validation order

For changes to symmetry or operators, run these checks in order:

1. Path counts and masks, including singlet counts `N=2,4,6,8,10 ->
   1,2,5,14,42`.
2. Coupled-basis orthonormality and `S^2 = S(S+1)`.
3. `F.T @ F = I` and agreement with explicit computational-basis recoupling.
4. `H_fusion = U.conj().T @ H_sz @ U` through `N <= 10`.
5. Exhaustive autoregressive normalization to roughly `1e-10` or better,
   followed by sampling-frequency checks.
6. Only then run VMC or interpret optimization results.

A neural model that appears to converge is not evidence that CG phases,
6j orientation, or operator connectivity are correct. Never weaken the exact
reference tests to accommodate training output.

## Engineering workflow for agents

- Read this file and the relevant module/tests before editing. Preserve user
  changes outside the requested scope.
- For Kim GPU submissions, let Slurm choose any compatible compute node;
  do not set a node list or require a particular hostname. Request the GPU
  type/memory needed by the calculation. Validate saved-state numerical replay
  when moving between compatible runtimes, and preserve running allocations.
- After result/status checks, update the corresponding root model notebooks
  when new verified independent audits are available, including matching
  endpoint provenance, uncertainties, work accounting, and regenerated plots.
  Preserve inline editable arrays, user plot styling, and omitted series.
  In plots, label binary and joint-ternary ClebschTree results uniformly as
  `SU(2)`. At each size/coupling within the same training protocol, display the
  lower independently audited energy mean; take its uncertainty, variance,
  and cost from that same state, state the chosen tree in text, and retain
  both raw results. This is best-result reporting, not evidence of a
  statistically established architecture win. Never promote training means,
  failed/zero-update runs, or invent extrapolations; existing physical readouts
  keep their original parent-state identity. This does not authorize launching
  background monitoring or changing remote jobs.
- Keep public functions typed and documented in terms of doubled labels.
  Validate invalid quantum numbers at API boundaries.
- Prefer small pure functions and deterministic basis orderings. Cache
  group-theory scalars where useful, but do not hide conventions in generated
  blobs.
- Add an independent exact test with every mathematical feature or sign
  change. Use `complex128/float64` for reference calculations.
- Keep optional training integrations optional. Core math tests must not need
  Optax; `train_adam` uses Optax when installed and its local equivalent
  otherwise.
- Do not begin Transformer or large 2D work while an earlier exact invariant
  is failing. The intended research sequence is chain fusion GRU, fair `Sz=0`
  comparison, generic/geometry-aware trees, then frustrated 2D models.
- Run the focused test first, then the full suite. Report numerical tolerances
  and the largest observed discrepancy when handing work off.

## Commands

```bash
python -m pip install -e '.[train,test]'
pytest
```

For a minimal environment, JAX, NumPy, and pytest are enough; Optax has a
built-in Adam fallback for functional testing. Force CPU explicitly when a
machine has a partially configured accelerator:

```bash
JAX_PLATFORM_NAME=cpu pytest
```
