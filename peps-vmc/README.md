# Finite SU(2)-PEPS VMC: isolated local validation

This directory contains a small, standalone PEPS implementation. It does not
modify the Spatial-ViT or ClebschTree models, checkpoints, or production jobs.
It reuses the repository's Clebsch–Gordan convention, triangular Hamiltonian
geometry, and optional generic VMC utilities. It is a **development test of a
looped tensor-network ansatz**, not a competitive-energy result or a new ViT.

The completed 6×6 local run, including actual costs and the unresolved
importance-concentration limitation, is documented in
[LOCAL_VALIDATION.md](LOCAL_VALIDATION.md).

## The actual first experiment

The physical problem is the spin-one-half triangular-lattice Hamiltonian

\[
H=\sum_{\langle ij\rangle}\mathbf S_i\cdot\mathbf S_j
 +0.125\sum_{\langle\!\langle ij\rangle\!\rangle}
 \mathbf S_i\cdot\mathbf S_j
\]

on the **6×6 periodic torus**. The full Hamiltonian has 108 nearest-neighbor
and 108 next-nearest-neighbor bonds; all **216 bonds** enter the action.

The initial *virtual* graph is a **four-leg square torus**, not a native
six-leg triangular PEPS. Hamiltonian bonds need not coincide with virtual
edges. This less expensive graph is a deliberate first contraction test and
can introduce a geometric bias. There is presently **no coherent lattice
\(A_1\) projection**, no guarantee of translation invariance for free-site
coefficients, and no imposed triangular point-group symmetry.

Each virtual edge carries

\[
\mathcal V=V_0\oplus V_{1/2},\qquad
D=1+2=3.
\]

There are **12 complete local reduced channels** for four virtual legs.
The free-site 6×6 model therefore has **36×12=432 real reduced parameters**;
none of these 12 channels is removed by a further multiplicity truncation.
The RVB-dominated initializer includes nonzero activation of the
three-occupied-leg channels and small site-dependent noise. This is not just
a product of fixed dimers. A one-site-shared option exists but is a different
parameter-sharing choice and must be identified in reports.

The local symmetry module also constructs the complete **76-channel,
six-virtual-leg** basis for the same D=3 space. This does **not** mean that the
first pilot contracts a native six-leg 6×6 graph: that contraction is guarded
against excessive intermediates. Supporting a local tensor and efficiently
contracting its full two-dimensional network are distinct achievements.

## Mathematical conventions and strict symmetry

- Every Hilbert label is a doubled integer: `q=2j`, `m=2m_j`.
- Physical bit `0` is down (`m=-1`); bit `1` is up (`m=+1`). Site zero is
  first / most significant in exhaustive computational-basis references.
- Virtual indices `0,1,2` are `(q,m)=(0,0),(1,-1),(1,+1)`.
- Fixed CG coefficients use the repository's Condon–Shortley convention.
- A local tensor `A[s,v0,...]` maps virtual states to the physical spin:
  `m_phys = sum(m_virtual)`. The learned parameters are only the coefficients
  of fixed equivariant intertwiners, not their magnetic-index structure.
- Local coupled vectors are orthonormal separately for each physical `m`.
  Consequently the Frobenius Gram matrix of full intertwiner maps is `2 I`,
  not `I`; this is an explicit normalization convention.

Virtual legs close through oriented singlet bonds,

\[
B_{(q,m),(q',n)}=
\delta_{q q'}\delta_{n,-m}
\frac{(-1)^{(q-m)/2}}{\sqrt{q+1}}.
\]

Thus `B[0,0]=1`, `B[1,2]=-1/sqrt(2)`, and
`B[2,1]=+1/sqrt(2)`. In the current even-by-even square virtual graph, edges
are oriented from the even to the odd sublattice and their metrics are
absorbed into the even-site tensors. Every metric is inserted exactly once.
Reversing one edge without a compensating half-integer parity operation can
change the physical state; it is not generally a harmless global sign.

A closed network of these intertwiners and singlet metrics is formally an
**exact total-spin singlet**. There is no tree root fixing the spin: the
guarantee follows from local equivariance and invariant bond closure.
Exact finite contraction preserves this construction up to floating-point
roundoff. Symmetry alone does not establish correct magnetic order, a QSL,
or adequate variational accuracy.

## Exact contraction and full Hamiltonian action

`contraction.py` builds a deterministic `opt_einsum` contraction plan and
executes it with NumPy or differentiable JAX. This is **exact single-layer
contraction**: no boundary-MPS truncation, CTM truncation, SVD cutoff, or
fitted environment is used in the pilot. A binary *contraction order* does
not turn a looped PEPS into a tree *wavefunction*: all original virtual loops
are still summed exactly.

Repeated Hamiltonian queries are expensive. `make_heisenberg_action` avoids
contracting 216 independently modified PEPS networks: it propagates the
unmodified tensor, needed single-spin boundary flips, and already inserted
Hamiltonian terms through the fixed contraction order. Each exchange term is
inserted once at the endpoints' lowest common contraction ancestor. Only
flips needed by bonds crossing the current contraction boundary are retained.
This is exact algebraic reuse, not a stale numerical cache or a Hamiltonian
approximation. `make_swap_query_plan` provides an alternative exact exchange
query DAG for checking such reuse.

Path FLOP and intermediate-size metadata are **estimates**, not measured
runtime or process/device peak memory. The executable pilot records actual
costs. Exact contraction remains expensive with increasing lattice size or
virtual dimension; this first implementation does not claim scalable PEPS
production at arbitrary size.

## Sampling and node-safe estimates

The pilot uses an exactly known **IID importance proposal**, not independent
Born samples from the PEPS and not an equilibrated PEPS Markov chain.
`proposals.py` freezes a finite library of triangular-neighbor dimer
coverings, mixes their product-singlet probabilities with uniform sampling
over the full zero-magnetization sector, then applies a half-identity /
half-Hamiltonian-exchange pushforward. Null exchanges are included in the
probability law. The uniform component gives strictly positive proposal
density everywhere in the target `M=0` sector; all terms of this finite
mixture are explicitly evaluated.

With proposal density \(Q(\sigma)>0\), moments use products such as

\[
\frac{|\psi|^2}{Q},\qquad
\frac{\psi^*H\psi}{Q},\qquad
\frac{|H\psi-E\psi|^2}{Q}.
\]

They do not divide by \(\psi\), discard target nodes, or clip local-energy
outliers. In particular, a configuration with \(\psi=0\) may still contribute
to the second moment through \(H\psi\ne0\). Finite updates are calibrated
from paired parent/candidate amplitudes rather than parameter norms alone.

Full support is **not** a guarantee of useful finite-sample coverage.
Norm/H² concentration, effective contributions, maximum sample fractions,
paired differences, and independent-group errors must accompany estimates.
Self-normalized finite-sample ratio bias and unseen-tail errors are not
certified by a small panel or by a jackknife error bar. A short pilot cannot
be presented as a converged variational benchmark.

## Interfaces

- `symmetry.py`: `local_intertwiner_basis`, `intertwiner_labels`,
  `virtual_bond_metric`, `reduced_tensor`, `rvb_coefficients`.
- `model.py`: `SquareGraphSU2PEPS`, parameter initialization and pure-JAX
  legal reduced-tensor decoding.
- `contraction.py`: exact scalar/batched amplitudes, exchange-query reuse,
  and complete Heisenberg action.
- `proposals.py`: frozen covering library and full-support IID bridge law.
- `diagnostics.py`: node-safe paired energy/variance estimates and finite
  Fubini–Study geometry.
- `pilot.py`: bounded train / independent-process replay / untouched-audit
  driver. Data products and measured results are written by the executable,
  not inserted into this README in advance.

## Local commands

Run from the repository root, using the existing local environment:

```bash
PYTHONPATH=peps-vmc JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=True \
  /Users/slowlight/miniconda3/envs/nqs/bin/python -m peps_vmc.pilot \
  --phase train \
  --config peps-vmc/configs/triangular_6x6_j2_0125_local.json \
  --output peps-vmc/results/triangular_6x6_j2_0125_local_20261029
```

Then use the same arguments in **new processes**, first replacing
`--phase train` with `--phase replay`, then with `--phase audit`.
Training writes registration and preflight diagnostics and attempts the
configured two full-reduced-parameter SR updates. Replay checks the final
saved transition and next RNG state. Audit uses a new 256-record paired IID
panel; training or calibration records must not be relabeled as terminal
audit samples. Failures or rejected/zero updates must remain visible.

For the isolated tests:

```bash
PYTHONPATH=peps-vmc JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=True \
  /Users/slowlight/miniconda3/envs/nqs/bin/python -m pytest -q peps-vmc/tests
```

The repository itself must remain importable to reuse its CG and Hamiltonian
code, as it is in the existing editable environment. This directory is not a
standalone copy of all NQS mathematical utilities.

## Validation and interpretation boundaries

Tests include local coupled-basis orthogonality, dense Lie-generator and
Casimir identities, an independent CG oracle for bond phases, exhaustive
small closed-loop physical singlets, direct virtual-index contraction,
complete dense-H action, exact finite proposal normalization/frequencies,
gradient checks, and preservation of contributions at target nodes.

The N=36 preflight also checks the actual amplitude Ward identity. For a
target configuration with 19 up spins, the 19 `M=0` precursor amplitudes
generated by lowering one up spin must sum to zero under total `S+`.
Require nonzero individual contributions and assess
`abs(sum(terms))/sum(abs(terms))`, with a float64 tolerance of `1e-11`.
An all-zero panel is not a meaningful symmetry check. This finite check
supports the local exact-construction proof; it is not an exhaustive N=36
measurement of every total-spin matrix element.

The initial acceptance question is narrow: can the looped, strictly SU(2)
ansatz be evaluated, differentiated, updated, checkpointed, restored, and
independently audited at a measured local cost? Increasing D, introducing
native triangular/PESS connectivity, enforcing full spatial projection,
complex parameters, production-scale statistics, and approximate scalable
boundary contraction are **subsequent developments**, not completed claims.
