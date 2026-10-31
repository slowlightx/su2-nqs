"""Local non-ED runtime canary for the strict triangle-patch Clebsch tree.

The canary answers one narrow question: can the exact-SU(2), reduced-width
``6x6`` ansatz execute the kernels needed by sampled VMC on the local JAX
backend?  It deliberately applies no lattice projector and performs no
optimization.  The fixed-``M2=0`` configurations are screened only to avoid
zero centers in a numerical kernel probe; their local-energy average is not a
variational-energy estimator.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    pytree_sha256,
)
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import (
    rectangular_translation_group,
    triangular_hnf_translation_group,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector,
    spatial_projected_amplitude,
)
from su2_nqs.models.triangle_patch_clebsch_tree import (
    init_triangle_patch_clebsch_tree,
    triangle_patch_6x6_canary_plan,
    triangle_patch_6x6_plan,
    triangle_patch_amplitude,
    triangle_patch_amplitudes,
    triangle_patch_parameter_count,
)
from su2_nqs.vmc.physical_vmc import physical_hpsi_vmc_loss_with_aux
from su2_nqs.vmc.spin_physical import padded_spin_connectivity, spin_physical_hpsi


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
TASK = "triangle_patch_n36_runtime_canary"
CLAIM_BOUNDARY = (
    "Non-ED local runtime/gradient readiness only; screened configurations are "
    "not Born samples, and no optimized energy, convergence, efficiency, phase, "
    "QSL, scaling, or PRL advantage is claimed."
)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate(spec: dict[str, object]) -> None:
    """Fail closed around the registered ``6x6`` runtime-only experiment."""

    if spec.get("schema_version") != 1 or spec.get("domain") != "spin":
        raise ValueError("N36 canary requires schema_version=1 and domain='spin'")
    if spec.get("task") != TASK or spec.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("task or claim boundary does not match the N36 canary")
    model = spec.get("model")
    if not isinstance(model, dict):
        raise TypeError("model must be an object")
    if model.get("family") != "strict_triangle_patch_clebsch_tree":
        raise ValueError("the canary is restricted to the strict triangle-patch tree")
    if model.get("local_carrier") != "complete_three_spin_schur":
        raise ValueError("the complete three-spin Schur carrier must be retained")
    if model.get("feature_dim") != 2 or model.get("maximum_intermediate_q") != 10:
        raise ValueError("the runtime canary is preregistered at D=2 and qmax=10")
    if model.get("expected_real_parameter_count") != 1252:
        raise ValueError("the registered D=2, qmax=10 model must have 1252 parameters")
    scale = float(model.get("initialization_scale", np.nan))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("model.initialization_scale must be positive and finite")
    if model.get("lattice_projection") != "none":
        raise ValueError("the first N36 canary must not apply a lattice projector")

    cases = spec.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError("cases must contain one square and one triangular 6x6 case")
    expected = {
        "square_6x6_j2_0p5": ("square", 0.5),
        "triangular_6x6_j2_0p125": ("triangular", 0.125),
    }
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("every case must be an object")
        case_id = str(case.get("case_id", ""))
        if case_id not in expected or case_id in seen:
            raise ValueError("case ids must be the registered square/triangular pair")
        seen.add(case_id)
        geometry, ratio = expected[case_id]
        if (
            case.get("geometry") != geometry
            or case.get("rows") != 6
            or case.get("columns") != 6
            or case.get("periodic") is not True
            or case.get("support") != "fixed_m0"
            or case.get("ed") is not False
            or not math.isclose(
                float(case.get("j2_over_j1", np.nan)), ratio, rel_tol=0.0, abs_tol=1.0e-15
            )
        ):
            raise ValueError(f"case {case_id!r} is not the registered N36 torus")
        seed = case.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("every case seed must be an integer")
    if seen != set(expected):
        raise ValueError("both registered N36 cases are required")

    runtime = spec.get("runtime")
    if not isinstance(runtime, dict):
        raise TypeError("runtime must be an object")
    for name in (
        "candidate_pool_size",
        "forward_batch_size",
        "hpsi_batch_size",
        "warm_repeats",
        "cutoff_stress_pool_size",
    ):
        _positive_integer(runtime.get(name), f"runtime.{name}")
    pool = int(runtime["candidate_pool_size"])
    forward = int(runtime["forward_batch_size"])
    hpsi = int(runtime["hpsi_batch_size"])
    if not hpsi <= forward <= pool:
        raise ValueError("require hpsi_batch_size <= forward_batch_size <= pool")
    threshold = float(runtime.get("minimum_center_magnitude", np.nan))
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum_center_magnitude must be positive and finite")
    relative_threshold = float(
        runtime.get("minimum_relative_center_magnitude", np.nan)
    )
    if not np.isfinite(relative_threshold) or not 0.0 < relative_threshold <= 1.0:
        raise ValueError("minimum_relative_center_magnitude must lie in (0,1]")
    if runtime.get("compile_with_jit") is not True:
        raise ValueError("the runtime canary must compile its JAX kernels")
    if runtime.get("padded_width_policy") != "bond_count_plus_one":
        raise ValueError("the padded-width policy must be bond_count_plus_one")
    if runtime.get("perform_parameter_update") is not False:
        raise ValueError("the runtime canary must not update parameters")
    if runtime.get("translation_projected_hpsi_probe") is not True:
        raise ValueError("the canary must include the read-only translation-Hpsi probe")
    _positive_integer(
        runtime.get("translation_orbit_chunk_size"),
        "runtime.translation_orbit_chunk_size",
    )
    if runtime.get("translation_rematerialize_orbit") is not False:
        raise ValueError("the forward-only translation probe must not rematerialize")
    if runtime.get("cutoff_stress_q_values") != [4, 6, 8, 10]:
        raise ValueError("cutoff stress scan must register qmax=4,6,8,10")
    stress_seed = runtime.get("cutoff_stress_seed")
    if isinstance(stress_seed, bool) or not isinstance(stress_seed, int):
        raise TypeError("cutoff_stress_seed must be an integer")


def _peak_rss_bytes() -> int:
    """Normalize ``ru_maxrss`` to bytes on macOS and Linux."""

    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else 1024 * raw


def _fixed_m0_candidates(seed: int, count: int, n_sites: int = 36) -> np.ndarray:
    """Generate deterministic fixed-magnetization configurations without ED."""

    if n_sites % 2:
        raise ValueError("fixed-M2=0 candidates require an even site count")
    generator = np.random.default_rng(seed)
    base = np.concatenate(
        (
            np.zeros(n_sites // 2, dtype=np.int32),
            np.ones(n_sites // 2, dtype=np.int32),
        )
    )
    result = np.empty((count, n_sites), dtype=np.int32)
    for index in range(count):
        result[index] = generator.permutation(base)
    return result


def _block(value: object) -> object:
    return jax.block_until_ready(value)


def _time_call(function, *arguments) -> tuple[object, float]:
    started = time.perf_counter()
    result = function(*arguments)
    _block(result)
    return result, time.perf_counter() - started


def _tree_finite(tree: object) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _tree_l2_norm(tree: object) -> float:
    squares = [
        float(np.sum(np.abs(np.asarray(leaf, dtype=np.complex128)) ** 2))
        for leaf in jax.tree_util.tree_leaves(tree)
    ]
    return float(np.sqrt(np.sum(squares)))


def _bonds(geometry: str, ratio: float) -> tuple[tuple[int, int, float], ...]:
    if geometry == "square":
        return square_j1j2_bonds(6, 6, j1=1.0, j2=ratio, periodic=True)
    if geometry == "triangular":
        return triangular_j1j2_bonds(6, 6, j1=1.0, j2=ratio, periodic=True)
    raise ValueError(f"unknown geometry: {geometry!r}")


def _cutoff_stress_scan(
    model: dict[str, object], runtime: dict[str, object]
) -> dict[str, object]:
    """Measure exact-zero support loss before declaring a cutoff sampling-ready.

    Square and triangular plans have isomorphic abstract merge trees under a
    site relabeling, but their physical patches differ.  The full cutoff scan
    is run on square and ``qmax=10`` is independently confirmed on triangular.
    No Hamiltonian is evaluated.
    """

    seed = int(runtime["cutoff_stress_seed"])
    pool = _fixed_m0_candidates(
        seed + 1,
        int(runtime["cutoff_stress_pool_size"]),
        36,
    )
    rows = []
    for maximum_q in runtime["cutoff_stress_q_values"]:
        plan = triangle_patch_6x6_plan(
            "square",
            feature_dim=int(model["feature_dim"]),
            maximum_intermediate_q=int(maximum_q),
        )
        params = init_triangle_patch_clebsch_tree(
            jax.random.PRNGKey(seed),
            plan,
            scale=float(model["initialization_scale"]),
        )
        kernel = jax.jit(
            lambda candidate, states: triangle_patch_amplitudes(
                candidate, states, plan
            )
        )
        values, seconds = _time_call(kernel, params, jnp.asarray(pool))
        concrete = np.asarray(values)
        exact_zero_count = int(np.count_nonzero(concrete == 0.0))
        nonzero = np.abs(concrete[concrete != 0.0])
        rows.append(
            {
                "maximum_intermediate_q": int(maximum_q),
                "parameter_count_real_scalars": triangle_patch_parameter_count(
                    params
                ),
                "pool_size": len(pool),
                "exact_zero_count": exact_zero_count,
                "exact_zero_fraction": exact_zero_count / len(pool),
                "minimum_nonzero_magnitude": (
                    float(np.min(nonzero)) if len(nonzero) else None
                ),
                "maximum_magnitude": float(np.max(np.abs(concrete))),
                "all_values_finite": bool(np.all(np.isfinite(concrete))),
                "cold_compile_and_call_seconds": seconds,
                "sampling_support_ready_on_this_probe": exact_zero_count == 0,
            }
        )
    zero_counts = [int(row["exact_zero_count"]) for row in rows]
    if (
        zero_counts[0] <= 0
        or any(left < right for left, right in zip(zero_counts, zero_counts[1:]))
        or zero_counts[-1] != 0
        or not all(bool(row["all_values_finite"]) for row in rows)
    ):
        raise RuntimeError(
            "cutoff support stress gate failed: qmax=4 must expose hard zeros, "
            "zero counts must be nonincreasing, and qmax=10 must remove them"
        )
    triangular_plan = triangle_patch_6x6_plan(
        "triangular",
        feature_dim=int(model["feature_dim"]),
        maximum_intermediate_q=10,
    )
    triangular_params = init_triangle_patch_clebsch_tree(
        jax.random.PRNGKey(seed),
        triangular_plan,
        scale=float(model["initialization_scale"]),
    )
    triangular_kernel = jax.jit(
        lambda candidate, states: triangle_patch_amplitudes(
            candidate, states, triangular_plan
        )
    )
    triangular_values, triangular_seconds = _time_call(
        triangular_kernel, triangular_params, jnp.asarray(pool)
    )
    triangular_concrete = np.asarray(triangular_values)
    triangular_zero_count = int(np.count_nonzero(triangular_concrete == 0.0))
    if triangular_zero_count != 0 or not np.all(np.isfinite(triangular_concrete)):
        raise RuntimeError("triangular qmax=10 support confirmation failed")
    return {
        "cross_geometry_relation": (
            "Square and triangular registrations use different physical patch "
            "sites but isomorphic abstract merge trees under site relabeling."
        ),
        "sample_pool_sha256": array_sha256(pool),
        "rows": rows,
        "triangular_qmax10_confirmation": {
            "maximum_intermediate_q": 10,
            "parameter_count_real_scalars": triangle_patch_parameter_count(
                triangular_params
            ),
            "pool_size": len(pool),
            "exact_zero_count": triangular_zero_count,
            "all_values_finite": bool(np.all(np.isfinite(triangular_concrete))),
            "cold_compile_and_call_seconds": triangular_seconds,
        },
        "decision": (
            "qmax=4 is not sampling-ready because it has hard-zero centers; "
            "qmax=10 is the first registered primary canary cutoff with no hard "
            "zeros on this fixed-M2=0 stress pool."
        ),
    }


def _run_case(
    case: dict[str, object],
    model: dict[str, object],
    runtime: dict[str, object],
) -> dict[str, object]:
    """Profile forward/JVP/Hpsi/gradient kernels for one geometry."""

    geometry = str(case["geometry"])
    plan = triangle_patch_6x6_canary_plan(geometry)
    if (
        plan.feature_dim != int(model["feature_dim"])
        or plan.maximum_intermediate_q != int(model["maximum_intermediate_q"])
    ):
        raise RuntimeError("registered canary plan disagrees with model config")
    params = init_triangle_patch_clebsch_tree(
        jax.random.PRNGKey(int(case["seed"])),
        plan,
        scale=float(model["initialization_scale"]),
    )
    parameter_count = triangle_patch_parameter_count(params)
    if parameter_count != plan.parameter_count:
        raise RuntimeError("plan and parameter-tree scalar counts disagree")
    if parameter_count != int(model["expected_real_parameter_count"]):
        raise RuntimeError("runtime parameter count disagrees with the config")

    pool = _fixed_m0_candidates(
        int(case["seed"]) + 10_000,
        int(runtime["candidate_pool_size"]),
        plan.n_sites,
    )
    batched_forward = jax.jit(
        lambda candidate, states: triangle_patch_amplitudes(candidate, states, plan)
    )
    pool_values, screening_seconds = _time_call(
        batched_forward, params, jnp.asarray(pool)
    )
    pool_values_host = np.asarray(pool_values)
    order = np.argsort(-np.abs(pool_values_host), kind="stable")
    threshold = float(runtime["minimum_center_magnitude"])
    pool_maximum = float(np.max(np.abs(pool_values_host)))
    relative_threshold = float(runtime["minimum_relative_center_magnitude"])
    selection_floor = max(threshold, relative_threshold * pool_maximum)
    eligible = order[np.abs(pool_values_host[order]) >= selection_floor]
    if len(eligible) < int(runtime["forward_batch_size"]):
        raise RuntimeError(
            f"only {len(eligible)} nonzero runtime centers survived for {geometry}"
        )
    samples = pool[eligible[: int(runtime["forward_batch_size"])]]
    if not np.all(np.sum(samples, axis=1) == plan.n_sites // 2):
        raise RuntimeError("the runtime probe escaped fixed M2=0 support")
    sample_values = pool_values_host[eligible[: int(runtime["forward_batch_size"])]]

    # Compile the registered batch shape separately from the screening shape.
    forward_args = (params, jnp.asarray(samples))
    forward_cold, forward_cold_seconds = _time_call(batched_forward, *forward_args)
    forward_warm_seconds = []
    for _ in range(int(runtime["warm_repeats"])):
        _, seconds = _time_call(batched_forward, *forward_args)
        forward_warm_seconds.append(seconds)

    tangent_key = jax.random.PRNGKey(int(case["seed"]) + 20_000)
    tangent_keys = iter(
        jax.random.split(tangent_key, len(jax.tree_util.tree_leaves(params)))
    )
    tangent = jax.tree_util.tree_map(
        lambda leaf: jax.random.normal(next(tangent_keys), leaf.shape, dtype=leaf.dtype),
        params,
    )

    def jvp_kernel(candidate, direction, states):
        return jax.jvp(
            lambda theta: triangle_patch_amplitudes(theta, states, plan),
            (candidate,),
            (direction,),
        )

    compiled_jvp = jax.jit(jvp_kernel)
    jvp_cold, jvp_cold_seconds = _time_call(
        compiled_jvp, params, tangent, jnp.asarray(samples)
    )
    jvp_warm_seconds = []
    for _ in range(int(runtime["warm_repeats"])):
        _, seconds = _time_call(compiled_jvp, params, tangent, jnp.asarray(samples))
        jvp_warm_seconds.append(seconds)

    hpsi_samples = samples[: int(runtime["hpsi_batch_size"])]
    bonds = _bonds(geometry, float(case["j2_over_j1"]))
    expected_bond_count = 144 if geometry == "square" else 216
    if len(bonds) != expected_bond_count:
        raise RuntimeError(
            f"registered {geometry} N36 geometry expected {expected_bond_count} bonds, "
            f"found {len(bonds)}"
        )
    padded_width = len(bonds) + 1
    connectivity_started = time.perf_counter()
    connected, elements, valid, widths = padded_spin_connectivity(
        hpsi_samples, bonds, padded_width=padded_width
    )
    connectivity_seconds = time.perf_counter() - connectivity_started
    hpsi_arguments = tuple(
        jnp.asarray(value)
        for value in (hpsi_samples, connected, elements, valid)
    )

    def hpsi_kernel(candidate, states, connected_states, matrix_elements, mask):
        return spin_physical_hpsi(
            candidate,
            states,
            connected_states,
            matrix_elements,
            mask,
            plan,
            amplitude_fn=triangle_patch_amplitude,
        )

    compiled_hpsi = jax.jit(hpsi_kernel)
    hpsi_cold, hpsi_cold_seconds = _time_call(
        compiled_hpsi, params, *hpsi_arguments
    )
    hpsi_warm_seconds = []
    for _ in range(int(runtime["warm_repeats"])):
        _, seconds = _time_call(compiled_hpsi, params, *hpsi_arguments)
        hpsi_warm_seconds.append(seconds)

    if geometry == "square":
        translation_group = rectangular_translation_group(6, 6)
    else:
        translation_group = triangular_hnf_translation_group(((6, 0), (0, 6)))
    projector = PhysicalSpatialProjector(
        translation_group,
        orbit_chunk_size=int(runtime["translation_orbit_chunk_size"]),
        rematerialize_orbit=bool(runtime["translation_rematerialize_orbit"]),
    )

    def projected_amplitude(candidate, state, _config):
        return spatial_projected_amplitude(
            candidate,
            state,
            plan,
            projector,
            amplitude_fn=triangle_patch_amplitude,
        )

    def projected_hpsi_kernel(
        candidate, states, connected_states, matrix_elements, mask
    ):
        return spin_physical_hpsi(
            candidate,
            states,
            connected_states,
            matrix_elements,
            mask,
            plan,
            amplitude_fn=projected_amplitude,
        )

    compiled_projected_hpsi = jax.jit(projected_hpsi_kernel)
    projected_hpsi_cold, projected_hpsi_cold_seconds = _time_call(
        compiled_projected_hpsi, params, *hpsi_arguments
    )
    projected_hpsi_warm_seconds = []
    for _ in range(int(runtime["warm_repeats"])):
        _, seconds = _time_call(
            compiled_projected_hpsi, params, *hpsi_arguments
        )
        projected_hpsi_warm_seconds.append(seconds)

    def loss_kernel(candidate, states, connected_states, matrix_elements, mask):
        return physical_hpsi_vmc_loss_with_aux(
            candidate,
            hpsi_fn=lambda theta: hpsi_kernel(
                theta, states, connected_states, matrix_elements, mask
            ),
        )

    compiled_value_and_grad = jax.jit(
        jax.value_and_grad(loss_kernel, has_aux=True)
    )
    gradient_cold, gradient_cold_seconds = _time_call(
        compiled_value_and_grad, params, *hpsi_arguments
    )
    gradient_warm_seconds = []
    for _ in range(int(runtime["warm_repeats"])):
        _, seconds = _time_call(
            compiled_value_and_grad, params, *hpsi_arguments
        )
        gradient_warm_seconds.append(seconds)

    (loss, auxiliary), gradient = gradient_cold
    centers, actions = hpsi_cold
    tangent_values = jvp_cold[1]
    center_replay_error = float(
        np.max(
            np.abs(
                np.asarray(centers)
                - np.asarray(forward_cold)[: int(runtime["hpsi_batch_size"])]
            )
        )
    )
    tangent_output_norm = float(np.linalg.norm(np.asarray(tangent_values)))
    gradient_norm = _tree_l2_norm(gradient)
    finite = (
        _tree_finite(forward_cold)
        and _tree_finite(jvp_cold)
        and _tree_finite(hpsi_cold)
        and _tree_finite(projected_hpsi_cold)
        and _tree_finite(gradient_cold)
        and int(np.asarray(auxiliary["nonfinite_local_count"])) == 0
        and center_replay_error < 1.0e-13
        and tangent_output_norm > 0.0
        and gradient_norm > 0.0
    )
    if not finite:
        raise RuntimeError(f"the {geometry} N36 runtime kernel produced nonfinite data")

    return {
        "case_id": str(case["case_id"]),
        "geometry": geometry,
        "n_sites": plan.n_sites,
        "feature_dim": plan.feature_dim,
        "maximum_intermediate_q": plan.maximum_intermediate_q,
        "patch_count": len(plan.patches),
        "parameter_count_real_scalars": parameter_count,
        "initial_params_sha256": pytree_sha256(params),
        "candidate_pool_size": len(pool),
        "candidate_pool_sha256": array_sha256(pool),
        "eligible_center_count": int(len(eligible)),
        "selected_samples_sha256": array_sha256(samples),
        "center_selection_absolute_floor": selection_floor,
        "selected_center_minimum_to_maximum_ratio": float(
            np.min(np.abs(sample_values)) / np.max(np.abs(sample_values))
        ),
        "selected_center_magnitude_minimum": float(np.min(np.abs(sample_values))),
        "selected_center_magnitude_maximum": float(np.max(np.abs(sample_values))),
        "screening_forward_seconds": screening_seconds,
        "forward_batch_size": len(samples),
        "forward_cold_compile_and_call_seconds": forward_cold_seconds,
        "forward_warm_seconds": forward_warm_seconds,
        "jvp_cold_compile_and_call_seconds": jvp_cold_seconds,
        "jvp_warm_seconds": jvp_warm_seconds,
        "tangent_parameter_l2_norm": _tree_l2_norm(tangent),
        "tangent_output_l2_norm": tangent_output_norm,
        "bond_count": len(bonds),
        "padded_width": padded_width,
        "actual_width_minimum": int(np.min(widths)),
        "actual_width_maximum": int(np.max(widths)),
        "valid_slot_fraction": float(np.mean(valid)),
        "connectivity_seconds_host": connectivity_seconds,
        "hpsi_batch_size": len(hpsi_samples),
        "hpsi_cold_compile_and_call_seconds": hpsi_cold_seconds,
        "hpsi_warm_seconds": hpsi_warm_seconds,
        "translation_projected_hpsi_probe": {
            "group_order": projector.group_order,
            "orbit_chunk_size": projector.orbit_chunk_size,
            "rematerialize_orbit": projector.rematerialize_orbit,
            "cold_compile_and_call_seconds": projected_hpsi_cold_seconds,
            "warm_seconds": projected_hpsi_warm_seconds,
            "center_magnitude_minimum": float(
                np.min(np.abs(np.asarray(projected_hpsi_cold[0])))
            ),
            "center_magnitude_maximum": float(
                np.max(np.abs(np.asarray(projected_hpsi_cold[0])))
            ),
            "hpsi_l2_norm": float(
                np.linalg.norm(np.asarray(projected_hpsi_cold[1]))
            ),
            "all_values_finite": _tree_finite(projected_hpsi_cold),
            "gradient_profiled": False,
            "interpretation": (
                "Read-only exact k=0 translation-projected Hpsi feasibility; "
                "it is not a training or energy-convergence result."
            ),
        },
        "gradient_cold_compile_and_call_seconds": gradient_cold_seconds,
        "gradient_warm_seconds": gradient_warm_seconds,
        "sample_fixed_vmc_loss": float(np.asarray(loss)),
        "probe_local_energy_mean": float(np.asarray(auxiliary["energy"])),
        "probe_local_energy_variance": float(np.asarray(auxiliary["variance"])),
        "probe_energy_imaginary_abs": float(
            np.asarray(auxiliary["energy_imaginary_abs"])
        ),
        "minimum_hpsi_center_magnitude": float(
            np.asarray(auxiliary["minimum_center_magnitude"])
        ),
        "maximum_probe_local_energy_magnitude": float(
            np.asarray(auxiliary["maximum_local_energy_magnitude"])
        ),
        "gradient_l2_norm": gradient_norm,
        "forward_hpsi_center_replay_max_abs_error": center_replay_error,
        "center_l2_norm": float(np.linalg.norm(np.asarray(centers))),
        "hpsi_l2_norm": float(np.linalg.norm(np.asarray(actions))),
        "nonfinite_local_count": int(np.asarray(auxiliary["nonfinite_local_count"])),
        "all_profiled_values_finite": finite,
        "peak_rss_bytes_process_high_water_after_case": _peak_rss_bytes(),
        "lattice_projection_applied": False,
        "translation_projected_hpsi_probe_performed": True,
        "hilbert_space_enumerated": False,
        "hamiltonian_diagonalization_performed": False,
        "parameter_update_performed": False,
        "interpretation": (
            "Kernel-readiness measurement only. The screened configurations "
            "are not Born-distributed, so probe local-energy moments are not "
            "variational-energy estimates."
        ),
    }


def run_triangle_patch_n36_runtime_canary(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and seal the square/triangular ``6x6`` non-ED kernel canary."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    manifest = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    runtime_identity = numerical_runtime_identity()
    started = time.perf_counter()
    rss_before = _peak_rss_bytes()
    cutoff_stress_scan = _cutoff_stress_scan(
        dict(spec["model"]), dict(spec["runtime"])
    )
    reports = [
        _run_case(case, dict(spec["model"]), dict(spec["runtime"]))
        for case in spec["cases"]
    ]
    total_seconds = time.perf_counter() - started
    peak_rss = _peak_rss_bytes()
    if not all(bool(report["all_profiled_values_finite"]) for report in reports):
        raise RuntimeError("one or more N36 runtime cases failed the finite-data gate")

    now = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": now,
        "completed_utc": now,
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_manifest": manifest,
        "source_manifest_sha256": canonical_sha256(manifest),
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        "total_wall_seconds": total_seconds,
        "peak_rss_bytes_before_cases": rss_before,
        "peak_rss_bytes_after_cases": peak_rss,
        "peak_rss_gib_after_cases": peak_rss / (1024.0**3),
        "rss_interpretation": (
            "ru_maxrss process high-water proxy; the second case inherits the "
            "first case's peak and this is not isolated per-kernel memory."
        ),
        "cases": reports,
        "cutoff_stress_scan": cutoff_stress_scan,
        "hamiltonian_diagonalization_performed": False,
        "hilbert_space_enumerated": False,
        "training_performed": False,
        "primary_gradient_lattice_projection_applied": False,
        "translation_projected_hpsi_probe_performed": True,
        "promotion_decision": (
            "pass_n36_unprojected_gradient_and_translation_hpsi_runtime_gate_only"
        ),
        "next_gate": (
            "Run a very short translation-projected optimization canary, then "
            "compare early slope at matched base-amplitude work before any "
            "publication-scale training."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = [
    "CLAIM_BOUNDARY",
    "TASK",
    "run_triangle_patch_n36_runtime_canary",
]
