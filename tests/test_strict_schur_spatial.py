from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.strict_schur_spatial import (
    four_plaquette_n16_smoke,
    strict_schur_coefficient_basis,
    strict_schur_spatial_oracle,
    strict_schur_spatial_oracle_suite,
)
from su2_nqs.frontier.route_policy import frozen_production_modules
from su2_nqs.models.strict_schur_spatial import (
    FourPlaquetteSchurPlan,
    SchurCell,
    StrictSchurSpatialPlan,
    four_plaquette_amplitude,
    four_plaquette_amplitude_from_spinors,
    four_plaquette_d4_plan,
    init_four_plaquette_reduced,
    init_static_schur_generator,
    init_strict_schur_reduced,
    square_plaquette_plan,
    static_schur_reduced_blocks,
    strict_schur_amplitude,
    strict_schur_amplitude_from_spinors,
    strict_schur_amplitudes,
    strict_schur_parameter_count,
    strict_schur_spatial_descriptors,
    two_plaquette_plan,
    two_triangle_plan,
)


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("factory", "expected_q", "expected_shapes", "expected_rank"),
    [
        (square_plaquette_plan, (0,), ((2,),), 2),
        (two_triangle_plan, (1, 3), ((2, 2), (1, 1)), 5),
        (two_plaquette_plan, (0, 2, 4), ((2, 2), (3, 3), (1, 1)), 14),
    ],
)
def test_complete_schur_plans_have_the_full_singlet_multiplicity(
    factory, expected_q, expected_shapes, expected_rank
) -> None:
    plan = factory()
    assert plan.active_q == expected_q
    assert plan.reduced_shapes == expected_shapes
    assert plan.reduced_parameter_count == expected_rank
    basis = strict_schur_coefficient_basis(plan)
    assert basis.shape == (1 << plan.n_sites, expected_rank)
    np.testing.assert_allclose(
        basis.T @ basis,
        np.eye(expected_rank),
        atol=3.0e-15,
        rtol=0.0,
    )


def test_two_cell_plan_exposes_a_disjoint_exact_cover_ledger() -> None:
    plan = two_plaquette_plan()
    merge = plan.ledger_merge
    assert merge is not None
    assert set(merge.left.sites).isdisjoint(merge.right.sites)
    assert set(merge.left.sites) | set(merge.right.sites) == set(
        merge.parent.sites
    )
    with pytest.raises(ValueError, match="cover physical sites"):
        StrictSchurSpatialPlan(
            (
                SchurCell((0, 1, 2, 3), "square_plaquette"),
                SchurCell((3, 4, 5, 6), "square_plaquette"),
            ),
            ((0.0, 0.0), (1.0, 0.0)),
        )


@pytest.mark.parametrize(
    "plan", (square_plaquette_plan(), two_triangle_plan(), two_plaquette_plan())
)
def test_bit_and_multilinear_spinor_evaluators_agree(plan) -> None:
    blocks = init_strict_schur_reduced(jax.random.PRNGKey(plan.n_sites), plan)
    rng = np.random.default_rng(100 + plan.n_sites)
    for _ in range(4):
        bits = rng.integers(0, 2, size=plan.n_sites, dtype=np.int32)
        spinors = np.eye(2, dtype=np.float64)[bits]
        bit_value = strict_schur_amplitude(blocks, bits, plan)
        spinor_value = strict_schur_amplitude_from_spinors(blocks, spinors, plan)
        np.testing.assert_allclose(bit_value, spinor_value, atol=2.0e-15, rtol=0.0)


def test_integer_spinors_are_promoted_before_schur_contraction() -> None:
    plan = square_plaquette_plan()
    blocks = init_strict_schur_reduced(jax.random.PRNGKey(19), plan)
    bits = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    integer_spinors = jnp.eye(2, dtype=jnp.int32)[bits]
    floating_spinors = integer_spinors.astype(jnp.float64)
    np.testing.assert_allclose(
        strict_schur_amplitude_from_spinors(blocks, integer_spinors, plan),
        strict_schur_amplitude_from_spinors(blocks, floating_spinors, plan),
        atol=1.0e-14,
        rtol=1.0e-14,
    )


@pytest.mark.parametrize(
    "spins",
    (
        jnp.asarray([0.0, 1.0, 0.0, 1.0]),
        jnp.asarray([False, True, False, True]),
    ),
)
def test_bit_amplitude_rejects_noninteger_or_boolean_dtype(spins) -> None:
    plan = square_plaquette_plan()
    blocks = init_strict_schur_reduced(jax.random.PRNGKey(23), plan)
    with pytest.raises(TypeError, match="integer dtype"):
        strict_schur_amplitude(blocks, spins, plan)
    with pytest.raises(TypeError, match="integer dtype"):
        strict_schur_amplitudes(blocks, spins[None, :], plan)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("target_q", False),
        ("target_q", 0.0),
        ("target_m2", False),
        ("target_m2", 0.0),
    ),
)
def test_plan_rejects_noninteger_doubled_target_labels(field, value) -> None:
    kwargs = {field: value}
    with pytest.raises(TypeError, match=field):
        StrictSchurSpatialPlan(
            (SchurCell((0, 1, 2, 3), "square_plaquette"),),
            ((0.0, 0.0),),
            **kwargs,
        )


def test_geometry_generator_is_static_and_exactly_nests_direct_blocks() -> None:
    plan = two_plaquette_plan()
    params = init_static_schur_generator(jax.random.PRNGKey(71), plan, hidden_dim=5)
    generated = static_schur_reduced_blocks(params, plan)
    for actual, expected in zip(generated, params["direct"], strict=True):
        np.testing.assert_array_equal(actual, expected)
    descriptors = strict_schur_spatial_descriptors(plan)
    assert descriptors.shape == (14, 7)
    assert not descriptors.flags.writeable
    # The nonlinear generator API has no spin/configuration argument.
    assert tuple(inspect.signature(static_schur_reduced_blocks).parameters) == (
        "params",
        "plan",
    )
    assert frozen_production_modules(
        ("su2_nqs.models.strict_schur_spatial",)
    ) == ()


@pytest.mark.parametrize("name", ("square_n4", "triangular_n6", "square_n8"))
def test_strict_schur_spatial_oracle_passes_all_permanent_gates(name: str) -> None:
    report = strict_schur_spatial_oracle(name)
    assert report["passed"]
    assert report["complete_singlet_rank_observed"] == report[
        "complete_singlet_rank_expected"
    ]
    assert report["coefficient_basis_gram_error"] < 3.0e-15
    assert report["coefficient_basis_spin_squared_residual_ratio"] < 2.0e-15
    assert report["physical_leg_multi_affinity_defect"] < 1.0e-14
    assert report["generator_nested_direct_max_abs_error"] == 0.0
    assert report["generator_tangent_spin_squared_residual_ratio"] < 1.0e-13
    assert report["geometry_residual_scale_tangent_norm"] > 1.0e-6


def test_complete_oracle_suite_promotes_only_to_the_n16_causal_gate() -> None:
    suite = strict_schur_spatial_oracle_suite()
    assert suite["passed"]
    assert set(suite["reports"]) == {"square_n4", "triangular_n6", "square_n8"}
    assert "Four complete plaquette carriers" in suite["next_gate"]


def test_four_plaquette_d4_plan_has_exact_619_parameter_ledger() -> None:
    plan = four_plaquette_d4_plan()
    assert isinstance(plan, FourPlaquetteSchurPlan)
    assert plan.intermediate_multiplicities == (
        (0, 14),
        (2, 28),
        (4, 20),
        (6, 7),
        (8, 1),
    )
    assert plan.intermediate_widths == ((0, 4), (2, 4), (4, 4), (6, 4), (8, 1))
    assert len(plan.level1_layout) == 19
    assert sum(int(np.prod(block.shape)) for block in plan.level1_layout) == 277
    assert sum(int(np.prod(shape)) for shape in plan.root_shapes) == 65
    assert plan.reduced_parameter_count == 619
    assert plan.internal_gauge_dimension == 130
    assert plan.generic_effective_tangent_dimension == 489
    assert set(plan.root_route.parent.sites) == set(range(16))


def test_four_plaquette_plan_canonicalizes_mutable_inputs_and_rejects_bool_pairs() -> None:
    reference = four_plaquette_d4_plan()
    cells = list(reference.cells)
    coordinates = [list(row) for row in reference.cell_coordinates]
    pairings = [[0, 1], [2, 3]]
    plan = FourPlaquetteSchurPlan(cells, coordinates, pairings)
    cells.pop()
    coordinates[0][0] = 99.0
    pairings[0][0] = 3
    assert plan.cells == reference.cells
    assert plan.cell_coordinates == reference.cell_coordinates
    assert plan.pairings == reference.pairings
    with pytest.raises(TypeError, match="pairing"):
        FourPlaquetteSchurPlan(
            reference.cells,
            reference.cell_coordinates,
            ((False, True), (2, 3)),
        )


def test_four_plaquette_bit_spinor_and_multiaffinity_contract() -> None:
    plan = four_plaquette_d4_plan()
    params = init_four_plaquette_reduced(jax.random.PRNGKey(101), plan)
    rng = np.random.default_rng(102)
    bits = rng.integers(0, 2, size=16, dtype=np.int32)
    one_hot = np.eye(2, dtype=np.float64)[bits]
    np.testing.assert_allclose(
        four_plaquette_amplitude(params, bits, plan),
        four_plaquette_amplitude_from_spinors(params, one_hot, plan),
        atol=2.0e-15,
        rtol=0.0,
    )
    spinors = rng.normal(size=(16, 2)) + 1.0j * rng.normal(size=(16, 2))
    first = rng.normal(size=2) + 1.0j * rng.normal(size=2)
    second = rng.normal(size=2) + 1.0j * rng.normal(size=2)
    alpha, beta = 0.31 - 0.17j, -0.23 + 0.29j
    for site in range(plan.n_sites):
        combined = spinors.copy()
        combined[site] = alpha * first + beta * second
        first_input = spinors.copy()
        first_input[site] = first
        second_input = spinors.copy()
        second_input[site] = second
        actual = four_plaquette_amplitude_from_spinors(params, combined, plan)
        expected = alpha * four_plaquette_amplitude_from_spinors(
            params, first_input, plan
        ) + beta * four_plaquette_amplitude_from_spinors(
            params, second_input, plan
        )
        np.testing.assert_allclose(actual, expected, atol=2.0e-12, rtol=2.0e-12)


def test_parameter_counter_counts_complex_entries_as_two_real_degrees() -> None:
    assert strict_schur_parameter_count(jnp.zeros((3,), dtype=jnp.float64)) == 3
    assert strict_schur_parameter_count(jnp.zeros((3,), dtype=jnp.complex128)) == 6


def test_four_plaquette_n16_forward_jvp_hpsi_smoke_passes() -> None:
    report = four_plaquette_n16_smoke()
    assert report["passed"]
    assert report["fixed_m0_basis_size"] == 12870
    assert report["raw_trainable_scalar_count"] == 619
    assert report["internal_gauge_nullity"] == 130
    assert report["unprojected_generic_effective_tangent_dimension"] == 489
    assert report["translation_invariant_singlet_codomain_dimension"] == 107
    assert report["translation_projected_random_tangent_rank_observed"] == 107
    assert report["state_spin_raising_relative_norm"] < 1.0e-11
    assert report["tangent_spin_raising_relative_norm"] < 1.0e-11
    assert report["hpsi_spin_raising_relative_norm"] < 1.0e-11
    assert report["translation_projected_spin_raising_relative_norm"] < 1.0e-11
    assert report["physical_leg_multi_affinity_defect"] < 5.0e-12
    assert report["state_jvp_central_difference_relative_error"] < 2.0e-8
    assert report["hpsi_jvp_central_difference_relative_error"] < 2.0e-8
    assert report["translation_hpsi_commutator_relative_error"] < 1.0e-11
    assert report["full_energy_vs_three_hz_abs_error"] < 1.0e-11
    assert report["hpsi_jvp_linearity_relative_error"] < 1.0e-11
    assert report["hamiltonian_diagonalization_performed"] is False
