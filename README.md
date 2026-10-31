# su2-nqs

Neural quantum states, exact SU(2) tools, and symmetry-resolved physical readouts for frustrated spin-1/2 magnets. Angular momenta use doubled integer labels: `q = 2j`, `m = 2m_j`.

## Current research

Two independent mainlines are ordinary Spatial-ViT (high-quality states and physical readouts) and triangular finite-width ClebschTree (strict-SU(2) variational states). Square ClebschTree is a reproduction sideline; static ClebschTree+attention is isolated development, with no established benefit. Direct fusion-path research is frozen; its validated algebra remains available.

- [Project overview, methods, protocol and talk/article outline](projects/clebsch_tree_nqs/SU2NQS_all.md)
- [Selected results, limitations and continuation receipts](projects/clebsch_tree_nqs/RESULTS.md)
- [Triangular J1–J2 plots](TRIANGULAR_J1J2_RESULTS.ipynb)
- [Square J1–J2 plots](SQUARE_J1J2_RESULTS.ipynb)
- [Separate HubbardCG project](projects/hubbard_cg/README.md)
- [Frozen direct-fusion conclusions](DIRECT_FUSION_LEGACY.md)
- [Recoverable historical archive](archive/README.md)

The two model notebooks use directly editable arrays, preserve audited state identities, and distinguish raw finite-size energies from zero-variance and thermodynamic extrapolations. Missing extrapolations remain empty. Selected exports live in `figures/{triangular,square}_j1j2/`; superseded exploration notebooks and figures are archived.

## Code boundaries

`su2_nqs/su2/` provides charges, CG, paths, trees and recoupling; `operators/`, `lattices/`, `benchmarks/` provide Hamiltonian connectivity and exact references; `models/` contains neural models; `vmc/` provides generic sampling, optimization and checkpoint tools. Project-specific drivers/configuration/evidence live under `projects/clebsch_tree_nqs/`. The `fnqs/` package has its own tests and remains unchanged by the documentation cleanup.

Current source/configs and continuation parents are retained. An old workflow remaining in source is not authorization to rerun a retired campaign; archived inputs must be explicitly restored if needed. Physical reference wavefunctions under `data/` and remote storage are protected.

## Source versions and scientific artifacts

The private source repository is `slowlightx/su2-nqs`. Its initial Git import is
a snapshot of the current source, not reconstructed Git ancestry for older
experiments. Historical runs retain their original source/input SHA256 seals.
Use short-lived feature branches for code changes, and lock each new job to a
full commit SHA, configuration hash, environment versions and input-checkpoint
identity. Running jobs must not import from a checkout whose branch is changed.

Large wavefunctions, raw results, logs, historical conversations, environments
and archives are deliberately excluded from Git. Small deterministic numerical
fixtures are explicitly allowlisted. Result links and some integration tests
need the external artifact store; a source-only clone does not contain the
research checkpoints. Keep the full exact assertions when restoring those
assets. See the [source import and index audit](configs/git_import_20261031/README.md)
and [deployment/versioning plan](configs/CODE_VERSIONING_PLAN_20261030.md).

On Kim, a local-SSH-fed bare mirror and immutable code-only releases avoid
copying private keys or forwarding an authentication agent to the cluster.
Different jobs at the same commit can reuse the same source release while
writing separate external run directories. Existing sealed job directories
are not migrated or deleted underneath running allocations.

## Install and validate

```bash
python -m pip install -e '.[train,test]'
JAX_PLATFORM_NAME=cpu python -m pytest -q tests/test_fusion_paths.py tests/test_coupled_basis.py tests/test_recoupling.py tests/test_heisenberg.py tests/test_models.py
```

See [validation guide](tests/README.md) and [AGENTS.md](AGENTS.md) for conventions and validation order. Tests establish algebraic invariants before VMC; numerical convergence is not evidence that symmetry or operator signs are correct. Optional TeNPy/NetKet integrations are not required by core mathematics.
