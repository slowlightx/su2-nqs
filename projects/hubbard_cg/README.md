# HubbardCG: deferred independent project

Effective date: 2026-09-16

HubbardCG is now a separate, downstream research project. It does not share
the spin project's promotion decision and will not consume large-scale compute
until the strict local-Schur/support-ledger spin route demonstrates a matched
energy--work advantage.

## Current evidence

The retained exact-small baseline uses physical Fock tokens and compares the
same binary tree with fixed `(Ne,Sz)` versus exact `(Ne,S)`, followed by the
same signed fermionic lattice projector. On the periodic `2x4`, `U/t=4`
ladder, the exact-spin `G16` arm has median `dE/N=7.9133e-5` versus
`3.6596e-4` for the Abelian control, and uses 1226 versus 2074 parameters.
This is a small ladder calibration, not a bulk-two-dimensional result or a
wall-time win.

The immutable historical artifact remains at
`../clebsch_tree_nqs/results/hubbard_validation_v1/results.json`. Its old
workflow and configuration remain compatibility-frozen so that the artifact
is not relocated or re-signed. New Hubbard-specific workflows, configs,
references, and results belong in this directory.

## Limited active work

1. Add the missing fixed-`Ne` charge-only control on `L=4,6,8` and `2x4`.
2. Keep exact reconstruction, fermionic-sign, `(Ne,S)` purity, and signed
   space-group tests mandatory.
3. Compare energy, variance, double occupancy, kinetic/potential energy, spin
   and charge structure factors on small systems only.
4. Use stored or published reference numbers; do not repeat large ED.
5. Treat sampled `3x4` as a future readiness test and `4x4` as remote work,
   only after the spin project passes its architecture and GPU Pareto gates.

The singlet `3Hz` estimator is spin-Hamiltonian-specific and does not apply to
Hubbard hopping or onsite interactions.

## Literature anchors

- LeBlanc et al., *Solutions of the Two-Dimensional Hubbard Model*,
  arXiv:1505.02290: multi-method energies and double occupancies.
- Qin, Shi, and Zhang, arXiv:1605.09421: finite-size and thermodynamic AFQMC
  energy, double occupancy, effective hopping, magnetization, and momentum
  distributions through `16x16`; half filling is sign-problem-free.
- Exact 1D Bethe-ansatz values remain the preferred large-`L` chain reference.

Every imported number must record geometry, boundary conditions, filling,
`t,t'`, `U`, normalization, raw/extrapolated status, and uncertainty before it
can enter a comparison plot.
