#!/usr/bin/env python3
"""A100 feasibility benchmark for an exactly projected adaptive N=36 carrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from su2_nqs.benchmarks.adaptive_projected_pair_carrier import (  # noqa: E402
    triangular_physical_pairing_bank,
)
from su2_nqs.benchmarks.projected_singlet_agp import (  # noqa: E402
    triangular_hnf_gamma_a1_group,
)
from su2_nqs.benchmarks.triangular_induced_parton import (  # noqa: E402
    projected_pi_zero_flux_a1_amplitudes_jax,
)
from su2_nqs.frontier.artifacts import atomic_json  # noqa: E402
from su2_nqs.models.projected_singlet_agp import (  # noqa: E402
    ProjectedSingletAGPConfig,
    projected_singlet_agp_amplitudes,
)


TRANSFER_PARAMETERS = np.asarray(
    [-0.37892179225984485, -0.24983280650264278, 0.4064207615603173],
    dtype=np.float64,
)


def _states(batch_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(batch_size):
        row = np.zeros(36, dtype=np.int8)
        row[rng.choice(36, size=18, replace=False)] = 1
        rows.append(row)
    return np.stack(rows)


def _timed(callable_):
    started = time.perf_counter()
    result = callable_()
    jax.tree.map(lambda value: value.block_until_ready(), result)
    return result, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(1, 8, 32))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("this benchmark requires the GPU backend")

    base, directions, labels = triangular_physical_pairing_bank(6)
    upper = np.triu_indices(36, 1)
    packed_base = jnp.asarray(base[upper], dtype=jnp.float64)
    packed_directions = jnp.asarray(
        np.stack([direction[upper] for direction in directions]),
        dtype=jnp.float64,
    )
    group = triangular_hnf_gamma_a1_group(((6, 0), (0, 6)))
    carrier = ProjectedSingletAGPConfig(
        n_sites=36,
        group_images=tuple(tuple(int(site) for site in image) for image in group),
    )

    def amplitudes(parameters, states):
        packed = packed_base + jnp.asarray(parameters) @ packed_directions
        return projected_singlet_agp_amplitudes(
            {"pairing_real": packed}, states, carrier
        ).amplitudes

    amplitude_kernel = jax.jit(amplitudes)
    gradient_kernel = jax.jit(
        jax.value_and_grad(
            lambda parameters, states: jnp.mean(
                jnp.square(jnp.abs(amplitudes(parameters, states)))
            )
        )
    )
    rows = []
    for index, batch_size in enumerate(args.batch_sizes):
        if batch_size < 1:
            raise ValueError("batch sizes must be positive")
        states = jnp.asarray(_states(batch_size, 20260921 + index))
        compiled, compile_seconds = _timed(
            lambda: amplitude_kernel(jnp.asarray(TRANSFER_PARAMETERS), states)
        )
        _warm, warm_seconds = _timed(
            lambda: amplitude_kernel(jnp.asarray(TRANSFER_PARAMETERS), states)
        )
        _gradient_compiled, gradient_compile_seconds = _timed(
            lambda: gradient_kernel(jnp.asarray(TRANSFER_PARAMETERS), states)
        )
        (_value, gradient), gradient_warm_seconds = _timed(
            lambda: gradient_kernel(jnp.asarray(TRANSFER_PARAMETERS), states)
        )
        rows.append(
            {
                "batch_size": batch_size,
                "compile_and_first_forward_seconds": compile_seconds,
                "warm_forward_seconds": warm_seconds,
                "warm_forward_states_per_second": batch_size / warm_seconds,
                "compile_and_first_gradient_seconds": gradient_compile_seconds,
                "warm_gradient_seconds": gradient_warm_seconds,
                "warm_gradient_states_per_second": batch_size / gradient_warm_seconds,
                "all_amplitudes_finite": bool(
                    np.all(np.isfinite(np.asarray(compiled)))
                ),
                "gradient": np.asarray(gradient).tolist(),
            }
        )

    audit_states = jnp.asarray(_states(16, 20261001))
    zero = np.asarray(amplitude_kernel(jnp.zeros(3), audit_states))
    induced = np.asarray(projected_pi_zero_flux_a1_amplitudes_jax(audit_states))
    scale = np.vdot(induced, zero) / np.vdot(induced, induced)
    relative_base_orbit_error = float(
        np.linalg.norm(zero - scale * induced) / np.linalg.norm(zero)
    )
    image = np.asarray(group[137], dtype=np.int32)
    original = np.asarray(
        amplitude_kernel(jnp.asarray(TRANSFER_PARAMETERS), audit_states)
    )
    transformed = np.asarray(
        amplitude_kernel(
            jnp.asarray(TRANSFER_PARAMETERS), audit_states[:, image]
        )
    )
    relative_a1_error = float(
        np.linalg.norm(original - transformed) / np.linalg.norm(original)
    )
    payload = {
        "schema_version": 1,
        "task": "n36_adaptive_pair_full_reynolds_jax_benchmark",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "group_order": len(group),
        "parameter_labels": list(labels),
        "transferred_n16_parameters": TRANSFER_PARAMETERS.tolist(),
        "rows": rows,
        "zero_deformation_relative_induced_orbit_error": relative_base_orbit_error,
        "deformed_gamma_a1_relative_error": relative_a1_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
