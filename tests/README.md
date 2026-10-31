# Validation guide

Core mathematical conventions and the required validation order are in [AGENTS.md](../AGENTS.md). Test code and mathematical tolerances were not changed by the 2026-10-20 documentation/data curation.

## Core exact invariants

```bash
JAX_PLATFORM_NAME=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_fusion_paths.py tests/test_coupled_basis.py \
  tests/test_recoupling.py tests/test_heisenberg.py tests/test_models.py
```

These cover doubled labels/counts/masks, coupled-basis orthogonality and spin, F-moves, Hamiltonian transforms, and autoregressive normalization. Keep independent reference calculations and original tolerances; never reseal an old numerical result merely to accommodate changed code.

## Current state production and notebook provenance

```bash
JAX_PLATFORM_NAME=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_remote_result_monitor.py tests/test_spatial_vit_cost.py \
  tests/test_triangle_clebsch_width_repair.py tests/test_triangle_clebsch_continuation.py \
  tests/test_triangle_n144_continue.py tests/test_square_paper_continue.py
```

Pure CG/Schur/Hamiltonian/sampling/checkpoint tests, active production tests and generic readout tests remain active. Optional TeNPy/NetKet tests skip when their dependencies are absent.

Selected notebook plot cells must contain actual embedded `image/png` output,
not only a `<Figure ...>` text representation or an exported file path.
`test_remote_result_monitor.py` checks this for both model notebooks. When
capturing output with `TerminalInteractiveShell`, explicitly enable the PNG
MIME formatter: its default text-only formatter can execute every plot without
storing a visible image in the notebook. Preserve user kernelspecs and plot code.

## Historical snapshot tests

With user approval, 60 retired `*_artifact.py`, `*_artifacts.py` and `*_artifact_independent.py` result-snapshot files were archived together with their old experiments. Their original assertions and inputs are recoverable, not rewritten to pass against a different state. Generic tests of the same mathematical components remain. Mixed test modules that also contain generic mathematical checks were kept, together with required numerical fixtures; this explains some older dates in active `results/`.

See [archive instructions](../archive/README.md) and `archive/project_curation_20261020/manifest.json` for the exact list. No global missing-data skip was added. Restoring a historical replay requires restoring its original inputs and, where sealed, its original source snapshot; a restored old snapshot may correctly reject newer live source hashes.

## Full suite

```bash
JAX_PLATFORM_NAME=cpu python -m pytest
```

The repository has previously hit an unrelated collection collision between `tests/test_coordinate_sr.py` and `fnqs/tests/test_coordinate_sr.py`. This curation does not alter package/module naming to hide that failure. Focused checks and any full-suite outcome must be reported separately; neither historical pass counts nor skipped snapshot replays certify current code.
