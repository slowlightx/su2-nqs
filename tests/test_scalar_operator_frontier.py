from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.scalar_operator_frontier import (
    triangular_pi_flux_scalar_operator_frontier,
)


def test_scalar_operator_frontier_is_nested_singlet_variational_hierarchy() -> None:
    result = triangular_pi_flux_scalar_operator_frontier(0.125)
    rows = {row["label"]: row for row in result["rows"]}
    bare = rows["bare_pi0"]
    one = rows["one_lanczos_H"]
    shell = rows["shell_resolved_J1_J2"]
    two = rows["two_lanczos_H"]
    quadratic = rows["shell_resolved_quadratic"]
    assert result["fixed_sz_basis_dimension"] == 12870
    assert result["maximum_scalar_carrier_splus_relative_norm"] < 2.0e-12
    assert one["energy_total"] <= bare["energy_total"] + 1.0e-12
    assert shell["energy_total"] <= one["energy_total"] + 1.0e-12
    assert two["energy_total"] <= one["energy_total"] + 1.0e-12
    assert quadratic["energy_total"] <= shell["energy_total"] + 1.0e-12
    assert result["exact_energy_total"] <= quadratic["energy_total"] + 1.0e-10
    assert result["carrier_variance_total"] > 0.0
    assert np.isclose(
        result["one_lanczos_linear_descent_slope_per_site"],
        2.0 * result["carrier_variance_per_site"],
    )
    assert "e2_scalar_bank" in rows
    assert "krylov2_plus_e2_scalar_bank" in rows


def test_shell_response_and_two_lanczos_are_same_dimension() -> None:
    result = triangular_pi_flux_scalar_operator_frontier(0.125)
    gate = result["same_dimension_gate"]
    assert gate["shell_dimension"] == 3
    assert gate["two_lanczos_dimension"] == 3


def test_e2_scalar_gate_is_dimension_matched_and_predeclared() -> None:
    result = triangular_pi_flux_scalar_operator_frontier(0.125)
    gate = result["e2_scalar_gate"]
    assert gate["e2_bank_dimension"] == gate["degree3_krylov_dimension"]
    assert gate["joint_dimension"] == gate["degree5_krylov_dimension"]
    assert isinstance(gate["e2_bank_beats_matched_krylov"], bool)
    assert isinstance(gate["e2_adds_after_matched_krylov"], bool)


def test_resonance_operator_gate_is_exact_singlet_and_dimension_matched() -> None:
    result = triangular_pi_flux_scalar_operator_frontier(0.125)
    gate = result["resonance_operator_gate"]
    assert result["maximum_scalar_carrier_splus_relative_norm"] < 3.0e-12
    assert gate["bank_dimension"] == gate["degree3_krylov_dimension"]
    assert gate["joint_dimension"] == gate["degree5_krylov_dimension"]
    assert gate["deep_joint_dimension"] == gate["degree8_krylov_dimension"]
    assert gate["carrier_labels"] == (
        "rhombus_ring_exchange",
        "rhombus_dimer_resonance",
        "six_site_loop_rearrangement",
    )
