from copy import deepcopy

import pytest

from su2_nqs.benchmarks.production_evidence import (
    CANONICAL_PROJECTION_CONTRACT,
    audit_sparse_causal_exact_tables,
    build_sparse_causal_production_evidence,
)
from su2_nqs.benchmarks.production_gate import (
    DEFAULT_REQUIREMENTS,
    evaluate_production_gate,
)


def _sparse_fixture() -> dict:
    rows = []
    for ratio, base, projected, recovery in (
        (0.5, -7.2, -7.205, 0.16),
        (0.6, -7.0, -7.002, 0.15),
    ):
        rows.append(
            {
                **CANONICAL_PROJECTION_CONTRACT,
                "benchmark_version": 3,
                "implementation_fingerprint": "sparse-fingerprint",
                "j2_over_j1": ratio,
                "seed": 0,
                "projected_energy": projected,
                "ideal_energy_improvement_recovery": recovery,
                "teacher": {"energy_before": base},
            }
        )
        # A better exploratory row must not replace the canonical contract.
        rows.append(
            {
                **CANONICAL_PROJECTION_CONTRACT,
                "steps": 99,
                "benchmark_version": 3,
                "implementation_fingerprint": "sparse-fingerprint",
                "j2_over_j1": ratio,
                "seed": 0,
                "projected_energy": base - 1.0,
                "ideal_energy_improvement_recovery": 1.0,
                "teacher": {"energy_before": base},
            }
        )
    return {
        "benchmark_version": 3,
        "implementation_fingerprint": "sparse-fingerprint",
        "runtime": {
            "n_sites": 16,
            "singlet_dimension": 1430,
            "plan_has_path_count_dimension": False,
            "sampler_first_jit_seconds": 2.0,
            "sampler_throughput_per_second": 1234.0,
            "plan_payload_bytes": 1000,
            "dynamic_state_bytes_per_sample": 512,
            "zero_head_normalization_error": 2.0e-16,
        },
        "rows": rows,
    }


def _tangent_fixture() -> dict:
    return {
        "scope": {"production_gate_eligible": False},
        "implementation_fingerprint": "tangent-fingerprint",
        "input_fingerprint": "input-fingerprint",
        "rows": [
            {
                "j2_over_j1": 0.5,
                "seed": 0,
                "groups": [
                    {
                        "group": "all",
                        "local_energy_response": {
                            "projection_fraction": 0.93,
                            "tangent_condition_number": 1.0e9,
                            "fisher_condition_number": 1.0e18,
                            "damped_natural_gradient": {
                                "1e-05": {
                                    "fraction_of_target_norm_in_damped_response": 0.32
                                }
                            },
                        },
                        "finite_imaginary_time_log_target": {
                            "projection_fraction": 0.98
                        },
                    }
                ],
            }
        ],
    }


def _lsqr_fixture() -> dict:
    return {
        "solver_contract": {"name": "matrix_free_golub_kahan_lsqr"},
        "implementation_fingerprint": "lsqr-fingerprint",
        "requested_run": {"damping": 1.0e-5},
        "rows": [
            {
                "j2_over_j1": 0.5,
                "seed": 0,
                "control": "joint_endpoint_vector_scalar",
                "damping": 1.0e-5,
                "lsqr_max_iterations": 50,
                "ideal_energy_improvement_recovery": 0.21,
                "projected_energy": -7.21,
                "teacher": {"energy_before": -7.2},
                "history": [{"lsqr_converged": False}],
            }
        ],
    }


def _connected_fixture() -> dict:
    return {
        "system_size": 16,
        "common_optima": {"combined": {"minimum_ess_fraction": 0.22}},
    }


def _triangular_fixture() -> dict:
    shell_geometry = {
        "bond_count": 48,
        "expected_bond_count": 48,
        "translation_mismatches": {"1,0": 0, "0,1": 0},
    }
    return {
        "n": 16,
        "boundary_condition": "periodic",
        "no_j1_j2_edge_aliasing": True,
        "basis_order_matches_left_fusion_paths": True,
        "geometry": {"j1": shell_geometry, "j2": shell_geometry},
        "operators": {
            "j1": {"maximum_hermiticity_residual": 8.0e-16},
            "j2": {"maximum_hermiticity_residual": 9.0e-16},
        },
    }


def _exact_fixture() -> dict:
    return {
        "passed": True,
        "maximum_su2_error": 0.0,
        "maximum_reduced_error": 2.22e-16,
        "maximum_tested_n": 10,
    }


def _build() -> dict:
    return build_sparse_causal_production_evidence(
        sparse=_sparse_fixture(),
        tangent=_tangent_fixture(),
        lsqr=_lsqr_fixture(),
        connected=_connected_fixture(),
        triangular=_triangular_fixture(),
        triangular_sparse={
            "passed": True,
            "scope": {"wavefunction_free": True},
            "ratios": [0.2, 0.3],
        },
        exact_audit=_exact_fixture(),
        current_implementation_fingerprint="sparse-fingerprint",
    )


def test_exact_sparse_causal_audit_reproduces_reduced_tables():
    audit = audit_sparse_causal_exact_tables()
    assert audit["passed"]
    assert audit["n_sites_values"] == [4, 6, 8, 10]
    assert audit["maximum_su2_error"] == 0.0
    assert audit["maximum_reduced_error"] < 1.0e-14


def test_builder_promotes_only_owned_exact_runtime_and_normalization_evidence():
    evidence = _build()
    identity = (
        "joint_endpoint_vector_scalar",
        "joint_endpoint_vector_scalar_causal_recurrence",
    )
    for section in ("exact_validation", "runtime", "normalization"):
        assert (
            evidence[section]["model_family"],
            evidence[section]["engine"],
        ) == identity
    assert evidence["connected_residual_sampling"]["model_family"] == (
        "generic_tree_gru"
    )
    assert evidence["production_readiness_claim"] is False
    assert [row["recovery_fraction"] for row in evidence["trials"]] == [
        0.16,
        0.15,
    ]
    assert all(row["paired_ucb95"] is None for row in evidence["trials"])
    assert evidence["non_gate_diagnostics"]["energy_tangent_projection"][
        "production_gate_eligible"
    ] is False
    assert evidence["non_gate_diagnostics"]["matrix_free_lsqr"][
        "production_gate_eligible"
    ] is False


def test_refreshed_evidence_gate_passes_four_provenance_checks_but_fails_closed():
    evidence = _build()
    result = evaluate_production_gate(evidence, DEFAULT_REQUIREMENTS)
    passed = {row["name"] for row in result["checks"] if row["passed"]}
    failed = set(result["failed_checks"])
    assert passed == {
        "exact_su2_and_reduced",
        "scalable_n16_runtime",
        "n16_exhaustive_normalization",
        "triangular_exact_graph",
    }
    assert failed == {
        "connected_residual_ess",
        "strict_no_suffix",
        "per_seed_recovery",
        "no_exact_energy_loss",
        "independent_paired_ucb",
    }
    assert result["decision"] == "FAIL"
    recovery = next(
        row for row in result["checks"] if row["name"] == "per_seed_recovery"
    )
    assert recovery["observed"]["minimum"] == 0.15
    paired = next(
        row
        for row in result["checks"]
        if row["name"] == "independent_paired_ucb"
    )
    assert paired["observed"]["maximum_ucb95"] is None


def test_builder_is_deterministic_and_rejects_stale_runtime_identity():
    assert _build() == _build()
    sparse = _sparse_fixture()
    with pytest.raises(ValueError, match="does not match"):
        build_sparse_causal_production_evidence(
            sparse=sparse,
            tangent=_tangent_fixture(),
            lsqr=_lsqr_fixture(),
            connected=_connected_fixture(),
            triangular=_triangular_fixture(),
            triangular_sparse={"passed": True},
            exact_audit=_exact_fixture(),
            current_implementation_fingerprint="different",
        )
    failed_audit = deepcopy(_exact_fixture())
    failed_audit["passed"] = False
    with pytest.raises(ValueError, match="did not pass"):
        build_sparse_causal_production_evidence(
            sparse=sparse,
            tangent=_tangent_fixture(),
            lsqr=_lsqr_fixture(),
            connected=_connected_fixture(),
            triangular=_triangular_fixture(),
            triangular_sparse={"passed": True},
            exact_audit=failed_audit,
            current_implementation_fingerprint="sparse-fingerprint",
        )
