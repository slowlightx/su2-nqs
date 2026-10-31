# Direct fusion-path legacy boundary

Effective date: 2026-09-16

## Status

The direct, left-associated fusion-path ansatz is **frozen as a production
method**. It is not an active route for square, triangular, or Hubbard
calculations, and it must not be reintroduced into production runners,
parameter searches, paper figures, or the active notebook.

This decision does not invalidate the SU(2) algebra developed during that
work. It separates an exact coordinate system from a scalable two-dimensional
wavefunction architecture.

## Why it was frozen

A left-associated path is naturally compatible with a one-dimensional chain.
In two dimensions it imposes an arbitrary one-dimensional ordering: bonds that
are local on the lattice generally cross long portions of the path, while
translations and point-group operations mix many fusion coordinates through
recoupling. Exact SU(2) is retained, but spatial locality and lattice quantum
numbers are not organized economically.

The repository evidence did not establish a compensating benefit:

- on small square and triangular graphs, physical-input Clebsch trees with a
  faithful lattice-irrep projector were substantially more accurate than the
  direct-path route at the useful capacities;
- local multi-route and group-closed DAG extensions did not robustly improve
  energy, variance, and correlations over the projected single-tree control;
- Hubbard showed a low-capacity CG advantage but a capacity crossover, not a
  universal direct-path compression law; and
- no direct-path campaign established lower end-to-end work, GPU-hours, or a
  better accuracy-versus-work scaling law beyond exact enumeration.

These are architecture and scaling conclusions, not failures of the exact
Clebsch--Gordan conventions. They also do not imply that every possible fusion
network is ineffective; they rule out continuing the direct path and its
incremental variants as this repository's production strategy.

## Exact material that remains authoritative

The following content is retained and tested as reusable mathematical
infrastructure:

- doubled-integer angular-momentum labels and reachability rules;
- the Condon--Shortley Clebsch--Gordan convention;
- generic binary fusion-tree construction and computational-basis overlaps;
- Racah/F-move recoupling with the documented matrix orientation;
- exact-small coupled-basis transformations and operator identities such as
  `H_fusion = U.conj().T @ H_sz @ U`; and
- path enumeration as an exact-small oracle for dimensions, spectra, matrix
  elements, and regression tests.

In particular, `su2_nqs/su2/fusion_tree.py` is generic SU(2) infrastructure;
it is not the frozen direct-path model. The historical left-comb/path modules
may be imported by exact reference tests only. The validation order in
`AGENTS.md` remains mandatory.

## Historical evidence policy

Old plans, scripts, simulations, reports, figures, and sealed manifests are
historical evidence. They may be inspected to understand a decision or replay
an exact-small invariant, but they are not active dependencies and are not
inputs to new paper-facing figures. Archived results retain their original
source hashes and must never be silently re-signed after a refactor.

The negative results worth retaining are the geometry mismatch, the absence
of a broad Pareto win, the failure of local route closure to contain the
projected single-tree orbit coherently, and the lack of a demonstrated
end-to-end compute advantage. Re-running large collections of failed
hyperparameter searches is not part of the active program.

## Production rule

All new calculations use physical spin/Fock inputs, explicit internal-symmetry
controls, and the same lattice projector on the compared backbones. The sole
active project is `projects/clebsch_tree_nqs/`. A future architecture may use
the retained SU(2) algebra, but it must enter through a new, preregistered
physical-input comparison and may not reactivate or tune the direct-path
ansatz.
