"""Build provenance-preserving evidence for the sparse causal production gate.

Only measurements made with the sparse causal endpoint recurrence are assigned
to that model family.  In particular, the older connected-proposal audit is
kept under its original generic TreeGRU identity, while instantaneous tangent
and exploratory LSQR results are retained as explicitly non-gating diagnostics.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from su2_nqs.su2.reduced_vector import (
    VECTOR_DELTA_Q,
    advance_weighted_spin_vector,
    compile_weighted_spin_vector,
    contract_reduced_vector_scalar,
)

from .sparse_causal_recurrence import compile_sparse_causal_recurrence_plan
from .sparse_recurrence_audit import audit_sparse_causal_recurrence_plan


MODEL_FAMILY = "joint_endpoint_vector_scalar"
ENGINE = "joint_endpoint_vector_scalar_causal_recurrence"
PHYSICAL_CONTROL = "joint_endpoint_vector_scalar"

# This is the first preregistered, frozen-encoder physical/K=0 comparison.  A
# later hyperparameter scan must not silently replace it with the best row.
CANONICAL_PROJECTION_CONTRACT: dict[str, Any] = {
    "control": PHYSICAL_CONTROL,
    "parameter_match_family": "frozen_encoder_zero_head",
    "steps": 4,
    "damping": 1.0e-5,
    "cg_max_iterations": 100,
    "message_hidden_dim": 24,
    "projection_measure": "exact_exhaustive_base_born",
}

SPARSE_IMPLEMENTATION_FILES: tuple[str, ...] = (
    "su2_nqs/benchmarks/sparse_causal_recurrence.py",
    "su2_nqs/models/sparse_causal_tree.py",
    "su2_nqs/vmc/sampled_sr.py",
    "scripts/run_sparse_causal_recurrence_benchmark.py",
)


def sparse_implementation_fingerprint(root: Path) -> str:
    """Return the implementation fingerprint used by the v3 benchmark."""

    digest = hashlib.sha256()
    for relative in SPARSE_IMPLEMENTATION_FILES:
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _directional_graph(n_sites: int) -> tuple[tuple[int, int, float, int], ...]:
    return tuple((site + 1, site, 1.0, 0) for site in range(n_sites - 1)) + tuple(
        (site + 2, site, 0.37, 1) for site in range(n_sites - 2)
    )


def _event_for_old_count(plan: Any, old_count: int) -> tuple[int, int]:
    matches = np.argwhere(
        plan.event_valid & (plan.revealed_count_before == old_count)
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one recurrence event with {old_count} revealed sites"
        )
    return tuple(map(int, matches[0]))


def _singlet_cut_reachable(q: int, block_sites: int, hidden_sites: int) -> bool:
    return (
        0 <= q <= block_sites
        and (block_sites - q) % 2 == 0
        and q <= hidden_sites
        and (hidden_sites - q) % 2 == 0
    )


def _maximum_sparse_difference(first: Any, second: Any) -> float:
    left = first.entry_dict()
    right = second.entry_dict()
    return float(
        max(
            (
                abs(left.get(key, 0.0j) - right.get(key, 0.0j))
                for key in set(left) | set(right)
            ),
            default=0.0,
        )
    )


def _audit_numeric_error(audit: Mapping[str, Any]) -> float:
    validation = audit["validation"]
    continuous = (
        "source_graph_maximum_error",
        "maximum_endpoint_reconstruction_error",
        "maximum_source_coupling_error",
        "maximum_closing_coupling_error",
        "maximum_partition_sum_or_squared_error",
        "maximum_forbidden_qop2_coefficient",
    )
    discrete = (
        "event_metadata_mismatches",
        "endpoint_set_mismatches",
        "lifecycle_mismatches",
        "closing_metadata_mismatches",
        "maximum_partition_count_error",
        "q_transition_mask_mismatches",
        "final_hidden_bond_count",
        "final_crossing_bond_count",
    )
    error = max(float(validation[name]) for name in continuous)
    if any(int(validation[name]) != 0 for name in discrete):
        error = max(error, 1.0)
    if not audit.get("passed", False):
        error = max(error, 1.0)
    return error


def audit_sparse_causal_exact_tables(
    *,
    n_sites_values: Sequence[int] = (4, 6, 8, 10),
) -> dict[str, Any]:
    """Numerically audit graph partitions and local reduced SU(2) tables.

    The reduced-table check advances one deterministic complex weighted vector
    through every cut up to the largest requested system, compares it against a
    direct compiler, and independently contracts dying vectors with one spin.
    It is deliberately small-system and wavefunction-free.
    """

    sizes = tuple(int(value) for value in n_sites_values)
    if not sizes or any(value < 2 for value in sizes):
        raise ValueError("n_sites_values must contain integers >= 2")
    maximum_n = max(sizes)
    graph_rows: list[dict[str, Any]] = []
    maximum_graph_error = 0.0
    for n_sites in sizes:
        bonds = _directional_graph(n_sites)
        plan = compile_sparse_causal_recurrence_plan(n_sites, bonds)
        audit = audit_sparse_causal_recurrence_plan(plan, bonds)
        error = _audit_numeric_error(audit)
        maximum_graph_error = max(maximum_graph_error, error)
        graph_rows.append(
            {
                "n_sites": n_sites,
                "passed": bool(audit["passed"]),
                "maximum_error": error,
                "event_count": int(audit["event_count"]),
                "uses_hidden_suffix": bool(audit["uses_hidden_suffix"]),
                "enumerates_revealed_prefixes": bool(
                    audit["enumerates_revealed_prefixes"]
                ),
            }
        )

    plan = compile_sparse_causal_recurrence_plan(
        maximum_n, _directional_graph(maximum_n)
    )
    rng = np.random.default_rng(7182)
    weights = rng.normal(size=maximum_n) + 0.19j * rng.normal(size=maximum_n)
    delta_position = {delta: slot for slot, delta in enumerate(VECTOR_DELTA_Q)}
    table = compile_weighted_spin_vector(weights[:1])
    maximum_advance_error = 0.0
    for old_count in range(1, maximum_n):
        decision, micro = _event_for_old_count(plan, old_count)
        advanced = advance_weighted_spin_vector(
            table, weights[old_count], zero_tolerance=1.0e-13
        )
        old_entries = table.entry_dict()
        predicted: dict[tuple[Any, Any], complex] = {}
        for new_path in advanced.paths:
            prefix_new = new_path[:-1]
            for old_path in advanced.paths:
                prefix_old = old_path[:-1]
                q_before = prefix_old[-1]
                q_after = old_path[-1]
                if not plan.q_transition_valid[
                    decision, micro, q_before, q_after
                ]:
                    continue
                delta_old = prefix_new[-1] - prefix_old[-1]
                delta_new = new_path[-1] - old_path[-1]
                if delta_old not in delta_position or delta_new not in delta_position:
                    continue
                value = (
                    plan.old_block_transport[
                        decision,
                        micro,
                        q_before,
                        q_after,
                        delta_position[delta_new],
                        delta_position[delta_old],
                    ]
                    * old_entries.get((prefix_new, prefix_old), 0.0j)
                )
                if prefix_new == prefix_old:
                    value += (
                        plan.endpoint_source[
                            decision,
                            micro,
                            q_before,
                            q_after,
                            delta_position[delta_new],
                        ]
                        * weights[old_count]
                    )
                if abs(value) > 1.0e-13:
                    predicted[(new_path, old_path)] = value
        restricted_exact = {
            key: value
            for key, value in advanced.entry_dict().items()
            if plan.q_transition_valid[
                decision, micro, key[1][:-1][-1], key[1][-1]
            ]
            and _singlet_cut_reachable(
                key[0][:-1][-1], old_count, maximum_n - old_count
            )
            and _singlet_cut_reachable(
                key[0][-1], old_count + 1, maximum_n - old_count - 1
            )
            and (key[0][-1] - key[1][-1]) in delta_position
            and (key[0][:-1][-1] - key[1][:-1][-1]) in delta_position
        }
        maximum_advance_error = max(
            maximum_advance_error,
            float(
                max(
                    (
                        abs(
                            predicted.get(key, 0.0j)
                            - restricted_exact.get(key, 0.0j)
                        )
                        for key in set(predicted) | set(restricted_exact)
                    ),
                    default=0.0,
                )
            ),
        )
        table = advanced

    direct = compile_weighted_spin_vector(weights, zero_tolerance=1.0e-13)
    maximum_direct_error = _maximum_sparse_difference(table, direct)

    maximum_scalar_error = 0.0
    spin = compile_weighted_spin_vector([1.0])
    for old_count in range(1, min(6, maximum_n - 1) + 1):
        decision, micro = _event_for_old_count(plan, old_count)
        local_rng = np.random.default_rng(8800 + old_count)
        old_table = compile_weighted_spin_vector(
            local_rng.normal(size=old_count)
        )
        entries = old_table.entry_dict()
        for q_after in range(maximum_n + 1):
            matrix, basis = contract_reduced_vector_scalar(
                old_table, spin, q_after, total_m=q_after
            )
            positions = {state: index for index, state in enumerate(basis)}
            for (new_path, old_path), reduced in entries.items():
                q_before = old_path[-1]
                if not plan.q_transition_valid[
                    decision, micro, q_before, q_after
                ]:
                    continue
                delta_old = new_path[-1] - old_path[-1]
                if delta_old not in delta_position:
                    continue
                if not _singlet_cut_reachable(
                    new_path[-1], old_count, maximum_n - old_count
                ):
                    continue
                row = positions.get((new_path, (1,)))
                column = positions.get((old_path, (1,)))
                if row is None or column is None:
                    continue
                expected = (
                    plan.dying_scalar[
                        decision,
                        micro,
                        q_before,
                        q_after,
                        delta_position[delta_old],
                    ]
                    * reduced
                )
                maximum_scalar_error = max(
                    maximum_scalar_error,
                    float(abs(matrix[row, column] - expected)),
                )

    maximum_reduced_error = max(
        maximum_advance_error, maximum_direct_error, maximum_scalar_error
    )
    return {
        "n_sites_values": list(sizes),
        "maximum_tested_n": maximum_n,
        "uses_hidden_suffix": False,
        "graph_partition_rows": graph_rows,
        "maximum_su2_error": maximum_graph_error,
        "maximum_reduced_error": maximum_reduced_error,
        "maximum_recursive_advance_error": maximum_advance_error,
        "maximum_direct_compile_error": maximum_direct_error,
        "maximum_dying_scalar_error": maximum_scalar_error,
        "passed": maximum_graph_error <= 1.0e-12
        and maximum_reduced_error <= 1.0e-12,
    }


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _canonical_projection_rows(
    sparse: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = sparse.get("rows")
    if not isinstance(rows, list):
        raise ValueError("sparse causal artifact has no row list")
    selected = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and all(row.get(name) == value for name, value in contract.items())
        and row.get("benchmark_version") == sparse.get("benchmark_version")
        and row.get("implementation_fingerprint")
        == sparse.get("implementation_fingerprint")
    ]
    identities: set[tuple[float, int]] = set()
    output: list[dict[str, Any]] = []
    for row in sorted(
        selected, key=lambda item: (float(item["j2_over_j1"]), int(item["seed"]))
    ):
        identity = (float(row["j2_over_j1"]), int(row["seed"]))
        if identity in identities:
            raise ValueError(f"duplicate canonical projection row: {identity}")
        identities.add(identity)
        teacher = _require_mapping(row.get("teacher"), "row.teacher")
        base_energy = float(teacher["energy_before"])
        projected_energy = float(row["projected_energy"])
        output.append(
            {
                "model_family": MODEL_FAMILY,
                "engine": ENGINE,
                "trial_kind": "production_causal",
                "evidence_stage": "exact_full_support_teacher_projection_only",
                "j2_over_j1": identity[0],
                "seed": identity[1],
                "uses_hidden_suffix": False,
                "recovery_fraction": float(
                    row["ideal_energy_improvement_recovery"]
                ),
                "exact_energy_delta": projected_energy - base_energy,
                "paired_ucb95": None,
                "evaluation_independent": False,
                "connected_sampled": False,
                "held_out_paired_energy_evaluated": False,
                "production_readiness_claim": False,
                "projection_measure": row["projection_measure"],
                "implementation_fingerprint": row[
                    "implementation_fingerprint"
                ],
            }
        )
    if not output:
        raise ValueError("no rows match the canonical sparse projection contract")
    return output


def _summarize_tangent(tangent: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source_row in tangent.get("rows", []):
        groups = source_row.get("groups", [])
        joint = next(
            (group for group in groups if group.get("group") == "all"), None
        )
        if joint is None:
            raise ValueError("energy-tangent row has no all-head group")
        local = joint["local_energy_response"]
        finite = joint["finite_imaginary_time_log_target"]
        rows.append(
            {
                "j2_over_j1": float(source_row["j2_over_j1"]),
                "seed": int(source_row["seed"]),
                "unregularized_local_energy_projection_fraction": float(
                    local["projection_fraction"]
                ),
                "finite_log_target_projection_fraction": float(
                    finite["projection_fraction"]
                ),
                "tangent_condition_number": float(
                    local["tangent_condition_number"]
                ),
                "fisher_condition_number": float(local["fisher_condition_number"]),
                "damped_response_fraction": {
                    key: float(value["fraction_of_target_norm_in_damped_response"])
                    for key, value in local["damped_natural_gradient"].items()
                },
            }
        )
    return {
        "production_gate_eligible": False,
        "reason": "instantaneous exact tangent capacity is not finite sampled training",
        "scope": tangent.get("scope"),
        "implementation_fingerprint": tangent.get("implementation_fingerprint"),
        "input_fingerprint": tangent.get("input_fingerprint"),
        "rows": rows,
    }


def _summarize_lsqr(lsqr: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source_row in lsqr.get("rows", []):
        history = source_row.get("history", [])
        rows.append(
            {
                "j2_over_j1": float(source_row["j2_over_j1"]),
                "seed": int(source_row["seed"]),
                "control": source_row["control"],
                "damping": float(source_row["damping"]),
                "lsqr_max_iterations": int(source_row["lsqr_max_iterations"]),
                "recovery_fraction": float(
                    source_row["ideal_energy_improvement_recovery"]
                ),
                "exact_energy_delta": float(source_row["projected_energy"])
                - float(source_row["teacher"]["energy_before"]),
                "all_linear_solves_converged": bool(history)
                and all(step.get("lsqr_converged") is True for step in history),
            }
        )
    return {
        "production_gate_eligible": False,
        "reason": "exact full-support solver study has no connected sampled or held-out paired gate",
        "solver_contract": lsqr.get("solver_contract"),
        "implementation_fingerprint": lsqr.get("implementation_fingerprint"),
        "requested_run": lsqr.get("requested_run"),
        "rows": rows,
    }


def _triangular_exact_graph(triangular: Mapping[str, Any]) -> dict[str, Any]:
    geometry = _require_mapping(triangular.get("geometry"), "triangular.geometry")
    operators = _require_mapping(
        triangular.get("operators"), "triangular.operators"
    )
    graph_errors: list[float] = []
    for shell in ("j1", "j2"):
        shell_geometry = _require_mapping(geometry.get(shell), f"geometry.{shell}")
        graph_errors.append(
            abs(
                float(shell_geometry["bond_count"])
                - float(shell_geometry["expected_bond_count"])
            )
        )
        graph_errors.extend(
            float(value)
            for value in shell_geometry.get("translation_mismatches", {}).values()
        )
    if triangular.get("no_j1_j2_edge_aliasing") is not True:
        graph_errors.append(1.0)
    if triangular.get("basis_order_matches_left_fusion_paths") is not True:
        graph_errors.append(1.0)
    hermiticity = max(
        float(operators[shell]["maximum_hermiticity_residual"])
        for shell in ("j1", "j2")
    )
    return {
        "lattice_family": "triangular_j1j2",
        "boundary_condition": "pbc",
        "source_boundary_condition": triangular.get("boundary_condition"),
        "n_sites": int(triangular["n"]),
        "exact": True,
        "j1_bonds": int(geometry["j1"]["bond_count"]),
        "j2_bonds": int(geometry["j2"]["bond_count"]),
        "maximum_hermiticity_error": hermiticity,
        "maximum_graph_error": max(graph_errors, default=0.0),
        "scope": "independent exact graph/operator audit; no triangular neural-energy claim",
        "source": "sims/triangular_pbc16_pilot/results.json",
    }


def build_sparse_causal_production_evidence(
    *,
    sparse: Mapping[str, Any],
    tangent: Mapping[str, Any],
    lsqr: Mapping[str, Any],
    connected: Mapping[str, Any],
    triangular: Mapping[str, Any],
    triangular_sparse: Mapping[str, Any],
    exact_audit: Mapping[str, Any],
    current_implementation_fingerprint: str,
    projection_contract: Mapping[str, Any] = CANONICAL_PROJECTION_CONTRACT,
) -> dict[str, Any]:
    """Map current artifacts into a fail-closed production evidence document."""

    sparse = _require_mapping(sparse, "sparse")
    runtime = _require_mapping(sparse.get("runtime"), "sparse.runtime")
    artifact_fingerprint = sparse.get("implementation_fingerprint")
    if artifact_fingerprint != current_implementation_fingerprint:
        raise ValueError(
            "sparse runtime artifact does not match the current implementation"
        )
    if sparse.get("benchmark_version") != 3:
        raise ValueError("only sparse causal benchmark v3 is production-evidence eligible")
    if runtime.get("plan_has_path_count_dimension") is not False:
        raise ValueError("runtime plan must not contain a path-count dimension")
    if int(runtime.get("n_sites", 0)) < 16:
        raise ValueError("runtime evidence must be measured at N>=16")
    if exact_audit.get("passed") is not True:
        raise ValueError("small-system sparse exact audit did not pass")

    identity = {
        "model_family": MODEL_FAMILY,
        "engine": ENGINE,
        "uses_hidden_suffix": False,
    }
    combined = _require_mapping(
        connected.get("common_optima", {}).get("combined"),
        "connected.common_optima.combined",
    )
    triangular_compiler_passed = bool(triangular_sparse.get("passed"))
    return {
        "schema_version": 2,
        "evidence_policy": (
            "Only measurements made by the named implementation are assigned to "
            "the sparse causal family. Generic connected sampling and exact-only "
            "tangent/solver diagnostics remain non-production evidence."
        ),
        "production_readiness_claim": False,
        "exact_validation": {
            **identity,
            "maximum_su2_error": float(exact_audit["maximum_su2_error"]),
            "maximum_reduced_error": float(
                exact_audit["maximum_reduced_error"]
            ),
            "maximum_tested_n": int(exact_audit["maximum_tested_n"]),
            "audit": exact_audit,
            "implementation_fingerprint": artifact_fingerprint,
            "source": "deterministic current-code small-system audit",
        },
        "connected_residual_sampling": {
            "model_family": "generic_tree_gru",
            "engine": "left_fusion_gru_connected_proposal_audit",
            "n_sites": int(connected["system_size"]),
            "uses_hidden_suffix": False,
            "proposal_family": "hamiltonian_connected",
            "minimum_ess_fraction": float(combined["minimum_ess_fraction"]),
            "production_family_measured": False,
            "source": "sims/connected_residual_audit/results.json",
        },
        "runtime": {
            **identity,
            "n_sites": int(runtime["n_sites"]),
            "compiled": float(runtime["sampler_first_jit_seconds"]) > 0.0,
            "scalable": runtime["plan_has_path_count_dimension"] is False,
            "production_engine": True,
            "production_readiness_claim": False,
            "uses_dense_exhaustive_plan": False,
            "throughput_samples_per_second": float(
                runtime["sampler_throughput_per_second"]
            ),
            "throughput_kind": "compiled_autoregressive_sampler",
            "plan_payload_bytes": int(runtime["plan_payload_bytes"]),
            "dynamic_state_bytes_per_sample": int(
                runtime["dynamic_state_bytes_per_sample"]
            ),
            "implementation_fingerprint": artifact_fingerprint,
            "source": "sims/sparse_causal_recurrence/results.json",
        },
        "normalization": {
            **identity,
            "n_sites": int(runtime["n_sites"]),
            "exhaustive": True,
            "singlet_dimension": int(runtime["singlet_dimension"]),
            "maximum_error": float(runtime["zero_head_normalization_error"]),
            "initialization": "zero_head_exact_base_identity",
            "implementation_fingerprint": artifact_fingerprint,
            "source": "sims/sparse_causal_recurrence/results.json",
        },
        "trials": _canonical_projection_rows(sparse, projection_contract),
        "non_gate_diagnostics": {
            "energy_tangent_projection": _summarize_tangent(tangent),
            "matrix_free_lsqr": _summarize_lsqr(lsqr),
            "triangular_sparse_compiler": {
                "production_gate_eligible": False,
                "passed": triangular_compiler_passed,
                "scope": triangular_sparse.get("scope"),
                "ratios": triangular_sparse.get("ratios"),
                "reason": (
                    "wavefunction-free graph compiler audit is not triangular "
                    "neural, sampling, energy, or phase evidence"
                ),
                "source": "sims/triangular_sparse_recurrence_audit/results.json",
            },
        },
        "triangular_exact_graph": _triangular_exact_graph(triangular),
    }


__all__ = [
    "CANONICAL_PROJECTION_CONTRACT",
    "ENGINE",
    "MODEL_FAMILY",
    "SPARSE_IMPLEMENTATION_FILES",
    "audit_sparse_causal_exact_tables",
    "build_sparse_causal_production_evidence",
    "sparse_implementation_fingerprint",
]
