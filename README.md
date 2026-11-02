# su2-nqs — core library

Core neural-quantum-state and SU(2) tensor tools for spin-1/2 systems. Quantum
numbers use doubled integers, `q = 2j` and `m = 2m_j`. Clebsch–Gordan structure
fixes magnetic indices; learned amplitudes or reduced tensors control the
remaining multiplicity degrees of freedom.

The library contains SU(2) algebra and recoupling, lattice/Hamiltonian tools,
independent small-system references, neural/tensor models, and generic VMC,
sampling, differentiation, checkpoint and stochastic-reconfiguration tools.
Optional integrations may require their own external data or dependencies;
the bundled core tests generate their own bounded synthetic inputs.

This repository intentionally does not publish research configurations,
cluster job scripts, analysis notebooks, figures, result tables, wavefunctions,
credentials, or experiment history. A core-source version is not sufficient
to reproduce an external production campaign without its separately retained
configuration, environment and input-state identities.

## Installation

```bash
python -m pip install -e '.[train,test]'
```

## Exact validation before VMC

```bash
JAX_PLATFORM_NAME=cpu python -m pytest -q \
  tests/test_fusion_paths.py tests/test_coupled_basis.py \
  tests/test_recoupling.py tests/test_heisenberg.py tests/test_models.py
JAX_PLATFORM_NAME=cpu python -m pytest -q
```

The second command runs the complete bundled **core** test suite, not the
research workspace's campaign-specific tests. Exact reference assertions are
retained unchanged. See [AGENTS.md](AGENTS.md) for the conventions and required
order of checks. JAX, NumPy and pytest suffice for core numerical execution;
Optax has an optional local Adam fallback, and symbolic/offline checks use
SymPy only outside JAX runtime.
