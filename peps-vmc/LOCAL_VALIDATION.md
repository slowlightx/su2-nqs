# Local validation — triangular 6×6, J₂/J₁=0.125

Date: 2026-10-29. This is an engineering validation, **not a converged energy
benchmark or a QSL claim**. Existing CT/ViT source, checkpoints and remote jobs
were not modified.

## State and virtual space

The pilot is a closed, four-virtual-leg PEPS with
`V = V_0 ⊕ V_(1/2)`, hence `D=3`. All 12 local intertwiner channels are retained,
with 432 independent real reduced coefficients on 36 sites. Magnetic indices
are fixed by CG coefficients; oriented singlet metrics close every virtual
edge. This constructs a strict physical singlet without selecting a tree root.

The virtual graph is a square torus used as a variational ansatz for the **full
triangular Hamiltonian**: 108 J₁ and 108 J₂ bonds, all included. It is not yet a
native six-leg triangular PEPS and does not impose coherent spatial A₁.
Contraction is exact; no boundary-MPS/CTM truncation is involved.

## What was actually run

The predeclared [configuration](configs/triangular_6x6_j2_0125_local.json) uses
two full-parameter importance-amplitude SR updates, each with 64 IID fitting
records, followed by separate 64-record selection and veto panels for the
actual finite physical step. Both updates were accepted. Their veto-panel FS
angles were 0.0026653090 and 0.0056018672. The fit tangent radius was 0.005;
the finite-panel ceiling was 0.02. Finite-FS acceptance is not an energy test.

Parameters, SR counter/moment state and next RNG were saved atomically. A
**new process** reproduced the final training transition with maximum parameter
and optimizer differences exactly 0, identical next RNG, and relative difference
0 in two recomputed full-H products. The terminal state was then frozen before
opening the new audit seed. Source/configuration/proposal hashes are recorded.

The proposal is a frozen dimer-covering mixture plus a 25% uniform-M=0
component, followed by a half-identity/half-H-exchange pushforward. Its density
is exactly known and positive over all M=0 states. These are importance samples,
not PEPS Born samples. Moments use ψ and Hψ directly, retaining possible target
node contributions to H².

## Correctness and measured CPU cost

| Check | Actual result |
| --- | ---: |
| Isolated package tests | 63 passed |
| Independent N=36 full-H relative discrepancy | 1.13013×10⁻¹⁵ |
| Full native gradient vs finite difference, relative discrepancy | 5.57851×10⁻¹¹ |
| Nonzero three-occupied-leg gradient entries | 288 / 288 |
| Initial six nontrivial S⁺ cancellation tests, maximum relative residual | 2.36042×10⁻¹⁶ |
| Final six nontrivial S⁺ cancellation tests, maximum relative residual | 4.25939×10⁻¹⁶ |
| Warm exact scalar amplitude | 0.006208 s |
| Warm complete (ψ,Hψ), all 216 bonds | 0.375837 s / center |
| First full-native-gradient call, including compilation | 18.073901 s |
| Two update segments, including their panels | 29.123463 + 28.521987 s |
| Training including preflight | 81.166105 s |
| Untouched paired audit, 512 full-H centers | 188.580429 s |

The S⁺ checks use 19 M=0 precursor amplitudes for each chosen M=1
configuration. They test the actual evaluator, but are not a complete measured
bound on ⟨S²⟩. Separate exhaustive closed-network tests check the physical
singlet and exact algebra independently of this sampling protocol.

The full-H kernel propagates only required single-spin boundary flips and
internal H products along the contraction path, inserting each exchange once.
The contraction path's largest scalar intermediate is 531441 elements; the
largest planned H-boundary intermediate is 13286025 elements. These are
planning sizes, **not measured process peak memory**. The run had a 6 GiB
peak-RSS safety cap and a 1800 s total train/replay/audit wall cap.

Explicit panel accounting is 128 training full-H centers plus 512 audit
centers, with two further H centers recomputed during replay. Preflight direct
H checks and internal SR amplitude/derivative calls are additional work, not
included in those panel counts.

Runtime: macOS 26.6 arm64, Python 3.14.7, JAX/JAXlib 0.11.1 on CPU, NumPy 2.5.2,
SciPy 1.18.1, opt_einsum 3.4.0; float64 enabled. Existing ordered group-theory,
Hamiltonian and autoregressive reference checks passed (34 tests). Full-root
pytest collection still has the pre-existing duplicate `test_coordinate_sr.py`
module-name conflict; no tests were weakened or renamed to bypass it.

## Untouched audit: valid workflow, insufficient statistical efficiency

Both endpoints were evaluated on the same **new 256 IID records**, divided
into 16 independent groups. Errors below are one group-jackknife SE:

| Quantity | Initial state | Two-update endpoint |
| --- | ---: | ---: |
| E/N | −0.50419549 ± 0.00928376 | −0.50455746 ± 0.00955598 |
| Var(H)/N | 0.08039229 ± 0.03808233 | 0.08309594 ± 0.03920718 |
| Norm importance ESS | 9.0249 / 256 | 9.0867 / 256 |
| Maximum norm contribution | 15.25% | 15.41% |

Paired ΔE/N = **−0.00036197 ± 0.00033347** (only 1.09 SE);
paired ΔVar(H)/N = **+0.00270365 ± 0.00186714**. Thus neither a reliable
energy improvement nor non-worsening variance has been established. Fit-panel
norm ESS was even lower: 2.176/64 and 1.966/64. Finite-sample ratio bias and
unseen-tail errors are not certified by these error bars or by full support.
The tiny audit estimates must not be promoted into the main energy plots.

**Conclusion:** a strict-SU(2), genuinely looped PEPS can execute complete
triangular-H action, full-W SR and exact checkpoint continuation locally at
this size. The next bottleneck is the proposal's importance concentration,
not a failed CG constraint or an unimplemented Hamiltonian. Better target-aware
sampling should precede claims about convergence, larger D, native six-leg
production, or comparisons with CT/ViT/GCNN.

## Reproducible artifacts

All states, raw panels, provenance and numeric reports are in
[`results/triangular_6x6_j2_0125_local_20261029/`](results/triangular_6x6_j2_0125_local_20261029/):

- `registration.json`, `preflight.json`, `training.json`;
- `step_01/` and `step_02/`, including before/after checkpoints and raw data;
- `frozen_endpoint.pkl`, `replay.json`;
- `audit_started.json`, `untouched_audit.npz`, `audit.json`.

The execution commands and interpretation limits are in [README.md](README.md).

## 2026-10-29: bounded learned-proposal comparison

The two-update PEPS endpoint above remains frozen. New code is isolated in
`experiments/`; neither the core evaluator nor the frozen source manifest is
changed. This experiment asks about sampling efficiency, not a new state or
an energy improvement. Attention has 2754 parameters, the GRU control 2716;
both use 300 fixed training steps and the same teacher batches. Their causal
fixed-M=0 masks apply before softmax. Final proposals have exact computable
densities and mix 75% learned probability with 25% of the old full-support
bridge, preserving node-safe importance estimates.

The first teacher used 1536 new IID bridge configurations, but its target-norm
ESS was only 13.05. Fresh 512-record evaluation gave norm ESS 10.68 / 1.98 /
10.56 for bridge / attention / GRU. Attention overfit this poorly covered
teacher. The failure is retained, not replaced by a better checkpoint.

One predeclared remedy replaced the teacher with 16 exchange-Metropolis chains
(64 burn-in, 96 saved states per chain with stride 4) plus 10% fresh bridge
regularization, without changing the networks, learning rate, or step budget.
These teacher records are correlated: mean lag-one log-amplitude correlation
was .904 and chain drift remained. They are **not certified equilibrated Born
samples**. The final frozen proposals are still evaluated with new IID
importance draws, independently of teacher quality.

| Fresh evaluation, 512 records per arm | Norm ESS | Online ESS/s | ESS/s including teacher + fit |
| --- | ---: | ---: | ---: |
| Existing bridge | 7.5469 | 3.9960 | 3.9960 |
| Attention | 9.0021 | 2.6406 | .30257 |
| Matched GRU | 9.5753 | 3.9745 | .32654 |

**No attention-specific or end-to-end efficiency benefit is established.**
Improved teacher coverage repairs the initial severe regression, but the
learned proposals have not beaten the bridge. Norm ESS is not an energy,
gradient, or H² certificate: the first run's tiny full-H audit, 32 centers per
arm, had H² ESS only 1.60–2.40. No energy or variance comparison is promoted
from it. The remedy uses no additional full-H queries.

The two experiments took 76.52 CPU seconds in total, with 11792 additional
amplitude centers and 96 full-H centers; teacher generation, fitting and
evaluation are all counted. There were zero PEPS parameter updates. Nineteen
focused tests and all 82 isolated-package tests passed; exhaustive normalized
proposal discrepancies through N=10 were at most 3.33e−16. An independent
agent reconstructed saved proposal densities and ESS with maximum discrepancy
0 and checked the symmetric exchange teacher and defensive mixture. Root
pytest still has the pre-existing duplicate-module collection conflict.

Raw registrations, fixed seeds, teacher coverage diagnostics, densities,
timings, and the preserved first failure are in
[`results/attention_sampling_20261029/`](results/attention_sampling_20261029/);
the single remedy is in its `target_aware_teacher/` subdirectory. Reproduce via
the corresponding scripts in `experiments/` with `PYTHONPATH=peps-vmc` and x64
enabled. The next step is not a larger attention model: first improve target
coverage and compare actual observable-specific effective samples per total
second against the unchanged bridge and GRU controls.
