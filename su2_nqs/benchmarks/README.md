# Benchmark and exact-reference layer

This package contains reusable exact-small references and physical diagnostics;
it is not an experiment menu. Production campaigns are owned by
`projects/clebsch_tree_nqs/` and may import only the subset reached by their
transitive source manifests.

## Active ClebschTree dependencies

- `regional_su2_readout.py`: complete bounded local Schur measurement and
  symmetry-labelled multiplicity spectra for invariant regional RDMs.
- `tensor_network_readout.py`: optional TeNPy graph/MPS interoperability,
  explicit down/up and requested-site ordering, and spin weights of general
  (including symmetry-broken) physical RDMs. TeNPy is not a core dependency.
- `tensor_network_observables.py`: independent spin/dimer closure and joint
  regional spin/cyclic projector probabilities, plus finite-MPS S² and K/M
  structure factors. Probabilities do not require invariant rho; interpreting
  a labelled entanglement spectrum does. No state/RDM twirl or repair.
- `clebsch_tree_exact.py`: sparse COO action and physical permutation/Reynolds
  helpers.
- `spin_sector.py`: independent fixed-magnetization `S^2` reference.
- `spatial_vit_sampling.py`: physical-input factored-ViT patch-translation
  coset projector, log-domain exchange sampling and bounded-memory JAX
  full-H ratios. No ED/support enumeration and no claim of exact SU2.
- `hubbard_cg_exact.py`: independent Hubbard total-spin reference.
- `hubbard_fock.py`: fixed-charge Fock basis/reference utilities.
- `hubbard_space_group.py`: signed fermionic permutation and projection.

The active workflow also uses lattice and operator modules directly. Its exact
dependency list is generated into every result as `source_sha256`; it should
not be maintained by hand here.

## Exact algebra retained from the frozen route

`coupled_basis.py`, recoupling/F-move checks, path enumeration, and independent
computational-basis Hamiltonians remain authoritative small-system oracles.
They validate conventions; they are not scalable production wavefunctions.
Modules associated with direct fusion-path models or abandoned exploratory
routes may remain for regression/history but are blocked from production by
`su2_nqs.frontier.route_policy`.

## Rules

- Angular momenta and projections use doubled integers.
- Connectivity is `H[new_state, old_state]`.
- Basis ordering is deterministic and part of the interface.
- Computational/Fock expansion is exact-small validation only.
- No symbolic work runs inside JAX.
- A result must distinguish exact-support accuracy, sampled accuracy,
  end-to-end performance, and phase-physics claims.
- Shared physical observables belong in this package or the appropriate
  operator/lattice layer, not in one-off scripts.

Dependency direction remains
`su2 -> operators/lattices -> benchmarks`, while models never import benchmark
logic.
# Active physical readout calibration (2026-10-05)

`tensor_network_readout.py`/`tensor_network_observables.py` retain legal spin
and cyclic probabilities for broken-symmetry RDMs without inventing an
invariant entanglement spectrum. `readout_quality.py` compares same-region
RDM trace distance and p(q) total variation against EACH endpoint's declared
refinement history. Multiplicity spectra are compared only for invariant
inputs. Successive-chi changes are empirical scales, not certified bias bounds.
The shared `spin_readout_calibration` workflow/configuration covers existing
small states and six fixed-D5 ordered controls; no DMRG dependency, new ED or
training. Specific external PRL competing states are a late optional task.
The same runner now supports `--screen-candidates`: hash the labelled physical
Hamiltonian/boundary, align saved wavefunctions and compare six fixed candidate
families against EACH endpoint's bounded quality flow. `physical_state_similarity`
removes only overall norm/phase, not sector contamination. Neither a carrier
label nor high seed fidelity is a thermodynamic phase identification. Saved
S0/A1 columns avoid new ED; snapshots preserve previous calibration sources.

# Historical phase-aware flux/loop mechanism gate

`flux_loop_interference_frontier.py` combines exact spatially projected
pi/0-flux and paired-`Z2` carriers with a full-lattice `A1` dimer-resonance
action.  It compares the physical response with equal-rank, exact-`S=0,A1`
geometry scrambles and a dimension-matched Hamiltonian Krylov control.  The
triangular `N=12` sector is reported only as a saturation warning; causal
comparisons use triangular and square `N=16`.

`flux_loop_n36_preflight.py` is the deliberately bounded continuation.  It
samples fresh pi/0-flux chains, evaluates full-space-group paired carriers,
and computes only the `J1`, `J2`, and dimer actions needed to distinguish a
configuration-dependent resonance from the Hamiltonian tangent.  It does not
fit or report a variational energy.

`relative_flux_crossfit.py` is the fail-closed follow-up.  Development chains
alone set carrier normalization, a fixed spectral cutoff, and regression
coefficients.  Untouched audit chains test the frozen relative-flux modes and
the part of their dimer response outside carrier, one-Hamiltonian, and
`H^2|pi>` directions.  Independently row-scrambled responses distinguish a
structured but redundant response from uncorrelated sampled variation.  The
module reports state-space capacity only and cannot authorize GPU production.

# Neural-backflow/Pfaffian exact-small gate

`neural_backflow_pfaffian.py` builds a configuration-dependent,
number-projected singlet Pfaffian and projects it into the complete sealed
`S=0,Gamma-A1` basis on triangular `N=16`. Compact J1 and extended J1/J2
relation kernels are compared with site-scrambled controls and same-carrier,
equal-dimension Krylov spaces. The exact projector is an oracle for this
small-system falsifier and is not presented as a scalable evaluator.

`residual_neural_backflow.py` performs the mandatory carrier-preserving test:
the projected-VMC/two-Lanczos state is an exact additive channel, while the
orthogonal neural-Pfaffian residual may change amplitudes and nodes. This
distinguishes a poor direct carrier from a poor correction manifold.

`resonating_neural_pfaffian.py` removes the single-optimization-basin
explanation by solving the coherent Ritz problem in banks of independently
optimized direct and residual Pfaffian states. Physical, site-scrambled, and
same-rank Krylov banks are compared at identical exact support. These three
modules are a sealed negative gate; they are reusable controls, not an active
`N=36` production architecture.

# Phase-aware long-range loop-RVB stop gate

`long_range_loop_rvb.py` is the final exact-small carrier contingency after
the neural-Pfaffian rejection. It enumerates triangular `N=16` J1 coverings
and coverings with at most one J2 bond, reduces them to spatial orbits, and
represents every orbit directly in the sealed complete `S=0,Gamma-A1` basis.
Projected pi/0-flux and paired-Z2 kernels supply fixed covering signs. A
three-parameter loop-Jastrow acts on long-bond and connected four-/six-site
alternating-resonance counts, which cannot be factorized into independent
bond weights.

The module contains a single-PSG primary model, a preregistered two-PSG plus
loop-parity contingency, independent orbit-feature scrambles, and matched
five-/seven-dimensional Krylov controls. Its exact A1 basis is an N=16 oracle,
not a scalable evaluator. The sealed result rejects N36 promotion and closes
further variants of the same carrier family.
