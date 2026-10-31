"""Sealed exact-small validation for the sampled full-spin structure factor.

This workflow is deliberately independent of any production-size VMC run. It
checks the physical-bit JAX action against the established fixed-``Sz``
operator, verifies the singlet ``q=0`` identity, and freezes both reciprocal
sum-rule normalizations before the observable is integrated into sampled
training.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    embedded_trial_manifest,
    pytree_sha256,
    repo_relative_path,
    trial_payload_sha256,
)
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.models.physical_tree import (
    PhysicalSpinTreeConfig,
    init_physical_spin_tree,
    physical_spin_tree_amplitude,
)
from su2_nqs.operators.sz_graph_sparse import heisenberg_sz_graph_sparse
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.sz_basis import enumerate_sz_configs
from su2_nqs.vmc.spin_observables import (
    prepare_spin_structure_factor,
    reciprocal_phase_weights,
    spin_structure_factor_hpsi,
    spin_structure_factor_local_values,
    spin_structure_factor_multi_hpsi,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]


def _validate_spec(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 2 or spec.get("domain") != "spin":
        raise ValueError("observable validation requires schema_version=2 and domain='spin'")
    if spec.get("task") != "exact_small_spin_observable_validation":
        raise ValueError("config task is not exact-small spin observable validation")
    normalization = spec.get("normalization")
    if not isinstance(normalization, dict) or normalization != {
        "default": "structure_factor_1_over_N",
        "m2": "explicit_1_over_N_squared",
    }:
        raise ValueError("normalization contract must register standard S(q) and explicit m2")
    complex_state = spec.get("complex_state")
    if not isinstance(complex_state, dict):
        raise TypeError("complex_state must be an object")
    n_sites = complex_state.get("n_sites")
    if n_sites != 8:
        raise ValueError("the complex-state action audit is fixed at N=8")
    coordinates = np.asarray(complex_state.get("coordinates"), dtype=np.float64)
    wavevector = np.asarray(complex_state.get("wavevector"), dtype=np.float64)
    if coordinates.shape != (8, 2) or wavevector.shape != (2,):
        raise ValueError("complex_state requires eight 2D coordinates and one 2D wavevector")
    if np.any(~np.isfinite(coordinates)) or np.any(~np.isfinite(wavevector)):
        raise ValueError("complex-state coordinates and wavevector must be finite")
    for section_name in ("complex_state", "singlet_q0", "sum_rule"):
        section = spec.get(section_name)
        if not isinstance(section, dict):
            raise TypeError(f"{section_name} must be an object")
        seed = section.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"{section_name}.seed must be an integer")
    singlet = spec["singlet_q0"]
    singlet_n = singlet.get("n_sites")
    if (
        isinstance(singlet_n, bool)
        or not isinstance(singlet_n, int)
        or singlet_n < 2
        or singlet_n > 8
        or singlet_n % 2
    ):
        raise ValueError("singlet_q0.n_sites must be an even integer in [2,8]")
    feature_dim = singlet.get("feature_dim")
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
        raise ValueError("singlet_q0.feature_dim must be a positive integer")
    sum_n = spec["sum_rule"].get("n_sites")
    if (
        isinstance(sum_n, bool)
        or not isinstance(sum_n, int)
        or sum_n < 2
        or sum_n > 8
        or sum_n % 2
    ):
        raise ValueError("sum_rule.n_sites must be an even integer in [2,8]")
    tolerances = spec.get("tolerances")
    if not isinstance(tolerances, dict):
        raise TypeError("tolerances must be an object")
    required = ("action_max_abs", "singlet_q0_max_abs", "sum_rule_abs")
    for name in required:
        value = float(tolerances.get(name, -1.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"tolerances.{name} must be positive and finite")


def _source_manifest() -> dict[str, str]:
    return transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )


def _bit_lookup_amplitude(values: jax.Array, bits: jax.Array) -> jax.Array:
    powers = 2 ** jnp.arange(bits.shape[0] - 1, -1, -1, dtype=jnp.int32)
    return values[jnp.sum(bits.astype(jnp.int32) * powers)]


def _random_complex_sector_state(
    n_sites: int, seed: int
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray, np.ndarray]:
    basis = tuple(enumerate_sz_configs(n_sites, n_sites // 2))
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))
    coefficients = np.asarray(coefficients / np.linalg.norm(coefficients), dtype=np.complex128)
    powers = 2 ** np.arange(n_sites - 1, -1, -1, dtype=np.int64)
    full_state = np.zeros(2**n_sites, dtype=np.complex128)
    for bits, value in zip(basis, coefficients, strict=True):
        full_state[int(np.dot(bits, powers))] = value
    return basis, coefficients, full_state


def _expectation(coefficients: np.ndarray, action: np.ndarray) -> complex:
    return complex(np.vdot(coefficients, action))


def _run_trial(spec: dict[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    complex_spec = spec["complex_state"]
    n_sites = int(complex_spec["n_sites"])
    basis, coefficients, full_state = _random_complex_sector_state(
        n_sites, int(complex_spec["seed"])
    )
    coordinates = tuple(tuple(map(float, point)) for point in complex_spec["coordinates"])
    wavevector = tuple(map(float, complex_spec["wavevector"]))
    weights = reciprocal_phase_weights(coordinates, wavevector)
    structure_spec = prepare_spin_structure_factor(weights)
    samples = jnp.asarray(basis, dtype=jnp.int32)

    @jax.jit
    def evaluated_action(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        return spin_structure_factor_hpsi(
            values,
            samples,
            structure_spec,
            amplitude_fn=_bit_lookup_amplitude,
        )

    centers_device, action_device = evaluated_action(jnp.asarray(full_state))
    jax.block_until_ready(action_device)
    centers = np.asarray(centers_device, dtype=np.complex128)
    action = np.asarray(action_device, dtype=np.complex128)
    exact_bonds = []
    for first in range(n_sites):
        for second in range(first + 1, n_sites):
            coefficient = 2.0 * float(
                np.real(weights[first] * np.conj(weights[second]))
            ) / n_sites
            exact_bonds.append((first, second, coefficient))
    exact_pair, exact_basis = heisenberg_sz_graph_sparse(
        n_sites, exact_bonds, n_up=n_sites // 2
    )
    if tuple(exact_basis) != basis:
        raise RuntimeError("fixed-Sz basis order changed in the exact observable reference")
    exact_onsite = float(0.75 * np.vdot(weights, weights).real / n_sites)
    exact_action = np.asarray(
        exact_pair @ coefficients + exact_onsite * coefficients,
        dtype=np.complex128,
    )
    center_discrepancy = float(np.max(np.abs(centers - coefficients)))
    action_discrepancy = float(np.max(np.abs(action - exact_action)))
    local = np.asarray(
        spin_structure_factor_local_values(
            jnp.asarray(full_state),
            samples,
            structure_spec,
            amplitude_fn=_bit_lookup_amplitude,
        ),
        dtype=np.complex128,
    )
    born_expectation = _expectation(coefficients, coefficients * local)
    exact_expectation = _expectation(coefficients, exact_action)
    expectation_discrepancy = float(abs(born_expectation - exact_expectation))

    singlet_spec = spec["singlet_q0"]
    singlet_n = int(singlet_spec["n_sites"])
    tree_config = PhysicalSpinTreeConfig(
        FusionTree.balanced(singlet_n),
        target_q=0,
        target_m2=0,
        feature_dim=int(singlet_spec["feature_dim"]),
        initialization_scale=float(singlet_spec["initialization_scale"]),
    )
    tree_params = init_physical_spin_tree(
        jax.random.PRNGKey(int(singlet_spec["seed"])), tree_config
    )
    singlet_samples = jnp.asarray(
        enumerate_sz_configs(singlet_n, singlet_n // 2), dtype=jnp.int32
    )
    q0_spec = prepare_spin_structure_factor(np.ones(singlet_n, dtype=np.complex128))

    @jax.jit
    def singlet_action(candidate: object) -> tuple[jax.Array, jax.Array]:
        return spin_structure_factor_hpsi(
            candidate,
            singlet_samples,
            q0_spec,
            amplitude_fn=lambda values, bits: physical_spin_tree_amplitude(
                values, bits, tree_config
            ),
        )

    singlet_centers_device, singlet_action_device = singlet_action(tree_params)
    jax.block_until_ready(singlet_action_device)
    singlet_centers = np.asarray(singlet_centers_device)
    singlet_q0_maximum = float(np.max(np.abs(np.asarray(singlet_action_device))))
    singlet_norm = float(np.vdot(singlet_centers, singlet_centers).real)

    sum_spec = spec["sum_rule"]
    sum_n = int(sum_spec["n_sites"])
    sum_basis, sum_coefficients, sum_full_state = _random_complex_sector_state(
        sum_n, int(sum_spec["seed"])
    )
    ring_coordinates = tuple((float(site),) for site in range(sum_n))
    standard_specs = []
    m2_specs = []
    for mode in range(sum_n):
        mode_weights = reciprocal_phase_weights(
            ring_coordinates, (2.0 * math.pi * mode / sum_n,)
        )
        standard = prepare_spin_structure_factor(mode_weights)
        m2 = prepare_spin_structure_factor(
            mode_weights, normalization=float(sum_n**2)
        )
        if standard.normalization_kind != "structure_factor" or m2.normalization_kind != "m2":
            raise RuntimeError("observable normalization labels are inconsistent")
        standard_specs.append(standard)
        m2_specs.append(m2)
    _, sum_actions_device = spin_structure_factor_multi_hpsi(
        jnp.asarray(sum_full_state),
        jnp.asarray(sum_basis),
        tuple(standard_specs + m2_specs),
        amplitude_fn=_bit_lookup_amplitude,
    )
    sum_actions = np.asarray(sum_actions_device)
    structure_values = [
        _expectation(sum_coefficients, sum_actions[:, mode])
        for mode in range(sum_n)
    ]
    m2_values = [
        _expectation(sum_coefficients, sum_actions[:, sum_n + mode])
        for mode in range(sum_n)
    ]
    structure_sum = complex(sum(structure_values))
    m2_sum = complex(sum(m2_values))
    structure_target = 0.75 * sum_n
    m2_target = 0.75
    structure_sum_error = float(abs(structure_sum - structure_target))
    m2_sum_error = float(abs(m2_sum - m2_target))

    tolerances = spec["tolerances"]
    gates = {
        "complex_action": action_discrepancy <= float(tolerances["action_max_abs"]),
        "singlet_q0": singlet_q0_maximum
        <= float(tolerances["singlet_q0_max_abs"]),
        "structure_factor_sum_rule": structure_sum_error
        <= float(tolerances["sum_rule_abs"]),
        "m2_sum_rule": m2_sum_error <= float(tolerances["sum_rule_abs"]),
    }
    if not all(gates.values()):
        raise RuntimeError(f"exact-small observable validation failed: {gates}")

    return {
        "arm_id": "full_su2_structure_factor",
        "seed": int(complex_spec["seed"]),
        "state_stage": "exact_small_validation",
        "normalization_contract": {
            "default_name": "structure_factor",
            "default_formula": "S(q)=N^-1 sum_ij exp(iq.(r_i-r_j)) S_i.S_j",
            "default_scale": n_sites,
            "m2_name": "m2",
            "m2_formula": "m2(q)=N^-2 sum_ij exp(iq.(r_i-r_j)) S_i.S_j",
            "m2_request": "normalization=N**2",
            "multi_q_pair_amplitudes_shared": True,
        },
        "state_hashes": {
            "n8_basis": array_sha256(np.asarray(basis, dtype=np.int32)),
            "n8_complex_coefficients": array_sha256(coefficients),
            "n8_full_state": array_sha256(full_state),
            "singlet_parameters": pytree_sha256(tree_params),
            "sum_rule_coefficients": array_sha256(sum_coefficients),
        },
        "metrics": {
            "n8_complex_state_action_max_abs_discrepancy": action_discrepancy,
            "n8_center_max_abs_discrepancy": center_discrepancy,
            "n8_born_expectation_abs_discrepancy": expectation_discrepancy,
            "n8_exact_expectation_real": float(exact_expectation.real),
            "n8_exact_expectation_imaginary_abs": float(abs(exact_expectation.imag)),
            "singlet_n_sites": singlet_n,
            "singlet_state_norm": singlet_norm,
            "singlet_q0_action_max_abs": singlet_q0_maximum,
            "sum_rule_n_sites": sum_n,
            "structure_factor_sum": float(structure_sum.real),
            "structure_factor_sum_imaginary_abs": float(abs(structure_sum.imag)),
            "structure_factor_sum_target": structure_target,
            "structure_factor_sum_abs_error": structure_sum_error,
            "m2_sum": float(m2_sum.real),
            "m2_sum_imaginary_abs": float(abs(m2_sum.imag)),
            "m2_sum_target": m2_target,
            "m2_sum_abs_error": m2_sum_error,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Exact-small normalization and operator-action validation only; "
            "not sampled-MCMC, optimization, scaling, or phase evidence."
        ),
    }


def run_spin_observable_validation(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run or reuse the sealed exact-small full-spin observable audit."""

    spec = json.loads(config_path.read_text())
    _validate_spec(spec)
    config_hash = canonical_sha256(spec)
    manifest = _source_manifest()
    manifest_hash = canonical_sha256(manifest)
    runtime_identity = numerical_runtime_identity()
    runtime_hash = canonical_sha256(runtime_identity)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        existing = json.loads(result_path.read_text())
        expected = (config_hash, manifest_hash, runtime_hash, "complete")
        observed = (
            existing.get("config_sha256"),
            existing.get("source_manifest_sha256"),
            existing.get("runtime_identity_sha256"),
            existing.get("status"),
        )
        if observed != expected:
            raise RuntimeError(f"refusing stale or incomplete observable result: {result_path}")
        embedded = embedded_trial_manifest(existing)
        if existing.get("trial_manifest_sha256") != canonical_sha256(embedded):
            raise RuntimeError(f"refusing observable result with broken trial seal: {result_path}")
        print(f"reused immutable complete observable validation: {result_path}", flush=True)
        return existing

    trial = _run_trial(spec)
    trial["trial_payload_sha256"] = trial_payload_sha256(trial)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "domain": "spin",
        "task": "exact_small_spin_observable_validation",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": repo_relative_path(config_path, root=ROOT),
        "config_sha256": config_hash,
        "config": spec,
        "source_sha256": manifest,
        "source_manifest_sha256": manifest_hash,
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_hash,
        "jax_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "route_boundary": {
            "historical_project_imports": False,
            "direct_fusion_path_used": False,
            "hamiltonian_diagonalization_performed": False,
            "exact_fixed_sz_observable_reference_used": True,
            "sampled_mcmc_tested": False,
            "production_size_claim": False,
        },
        "cases": [
            {
                "case_id": "full_spin_structure_factor_exact_small",
                "trials": [trial],
            }
        ],
    }
    trial_manifest = embedded_trial_manifest(payload)
    payload["trial_manifest"] = trial_manifest
    payload["trial_manifest_sha256"] = canonical_sha256(trial_manifest)
    atomic_json(result_path, payload)
    print(
        f"observable validation: action error="
        f"{trial['metrics']['n8_complex_state_action_max_abs_discrepancy']:.3e}",
        flush=True,
    )
    return payload


__all__ = ["run_spin_observable_validation"]
