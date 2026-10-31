#!/usr/bin/env python3
"""Bounded A100 benchmark for the reusable N=36 scalar-word kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from su2_nqs.benchmarks.operator_vit_sampled import scalar_component_words  # noqa: E402
from su2_nqs.benchmarks.operator_word_jax import (  # noqa: E402
    build_triangular_a1_word_action_plan,
    evaluate_fixed_capacity_scalar_word_kernel,
    make_fixed_capacity_scalar_word_kernel,
)
from su2_nqs.benchmarks.triangular_induced_parton import (  # noqa: E402
    build_n36_pi_zero_flux_orbit,
    projected_pi_zero_flux_a1_amplitudes_jax,
)
from su2_nqs.frontier.artifacts import atomic_json  # noqa: E402
from su2_nqs.operators.triangular_a1_scalar import (  # noqa: E402
    TRIANGULAR_A1_SCALAR_LABELS,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--amplitude-batch-size", type=int, default=4096)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("this benchmark requires the GPU backend")

    labels = TRIANGULAR_A1_SCALAR_LABELS[:3]
    components = scalar_component_words(labels)
    words = tuple(
        dict.fromkeys(
            (
                *components,
                *((label, *word) for label in labels[:2] for word in components),
            )
        )
    )
    # A deterministic half-filled center; connectivity and amplitude evaluation
    # are the production N=36 operations, not a synthetic dense surrogate.
    center = np.asarray([[index % 2 for index in range(36)]], dtype=np.int8)
    plan, plan_ledger = build_triangular_a1_word_action_plan(
        center, ((6, 0), (0, 6)), words
    )
    batch = int(args.amplitude_batch_size)
    terminal_capacity = ((len(plan.terminal_configurations) + batch - 1) // batch) * batch
    orbit = build_n36_pi_zero_flux_orbit()

    def amplitude(states):
        return projected_pi_zero_flux_a1_amplitudes_jax(states, orbit=orbit)

    kernel = make_fixed_capacity_scalar_word_kernel(
        amplitude,
        n_sites=36,
        amplitude_batch_size=batch,
        terminal_capacity=terminal_capacity,
        entry_capacity=len(plan.entry_coefficients),
        output_count=len(words),
    )
    first, compile_ledger = evaluate_fixed_capacity_scalar_word_kernel(kernel, plan)
    warm_rows = []
    for _ in range(3):
        started = time.perf_counter()
        observed, ledger = evaluate_fixed_capacity_scalar_word_kernel(kernel, plan)
        wall = time.perf_counter() - started
        if not np.allclose(observed, first, atol=1.0e-10, rtol=1.0e-10):
            raise RuntimeError("warm reusable kernel changed its result")
        warm_rows.append({**ledger, "end_to_end_call_wall_seconds": wall})
    warm_seconds = [float(row["jax_kernel_wall_seconds"]) for row in warm_rows]
    payload = {
        "schema_version": 1,
        "task": "n36_scalar_word_jax_kernel_benchmark",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "labels": list(labels),
        "word_count": len(words),
        "plan": plan_ledger,
        "fixed_capacity": {
            "terminal_capacity": terminal_capacity,
            "entry_capacity": len(plan.entry_coefficients),
            "amplitude_batch_size": batch,
        },
        "compile_and_first_execution": compile_ledger,
        "warm_executions": warm_rows,
        "best_warm_kernel_seconds": min(warm_seconds),
        "best_warm_unique_amplitudes_per_second": (
            len(plan.terminal_configurations) / min(warm_seconds)
        ),
        "all_outputs_finite": bool(np.all(np.isfinite(first))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
