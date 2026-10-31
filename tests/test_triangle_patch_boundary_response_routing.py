from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.su2_vit_architecture_oracle import (
    spin_half_su2_rotation,
)
from su2_nqs.benchmarks.exact_diagonalization import heisenberg_sz_matrix
from su2_nqs.benchmarks.two_cluster_virtual_processes import (
    two_cluster_open_partition,
    two_cluster_virtual_process_report,
)
from su2_nqs.frontier.triangle_patch_boundary_response_init import (
    init_triangle_patch_boundary_response_carriers,
    triangle_patch_boundary_response_reference,
    triangle_patch_boundary_response_routing,
)
from su2_nqs.frontier.triangle_patch_physical_init import (
    triangle_patch_cluster_reference,
)
from su2_nqs.frontier.triangle_patch_physical_multiplet_init import (
    _cluster_multiplet_basis,
    init_triangle_patch_cluster_multiplets,
)
from su2_nqs.lattices.spatial_symmetry import rectangular_translation_group
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector,
    spatial_projected_amplitudes,
)
from su2_nqs.models.triangle_patch_clebsch_tree import (
    triangle_patch_6x6_canary_plan,
    triangle_patch_amplitudes,
    triangle_patch_grid_plan,
    triangle_patch_pair_layout,
    triangle_patch_parameter_count,
    triangle_root_layout,
)
from su2_nqs.models.triangle_patch_interface_routing import (
    TrianglePatchInterfaceRouting,
    triangle_patch_interface_amplitude,
    triangle_patch_interface_amplitude_from_spinors,
    triangle_patch_interface_amplitudes,
)


jax.config.update("jax_enable_x64", True)


def _configurations(n_sites: int) -> np.ndarray:
    return np.asarray(
        [
            [(state >> (n_sites - site - 1)) & 1 for site in range(n_sites)]
            for state in range(1 << n_sites)
        ],
        dtype=np.int32,
    )


def _lower_spin_one_member(member: np.ndarray) -> np.ndarray:
    result = np.zeros_like(member)
    for state, amplitude in enumerate(member):
        if amplitude == 0.0:
            continue
        for site in range(6):
            mask = 1 << (5 - site)
            if state & mask:
                result[state ^ mask] += amplitude / np.sqrt(2.0)
    return result


def _triplet_members(geometry: str, coefficients: np.ndarray) -> tuple[np.ndarray, ...]:
    highest_basis, _ = _cluster_multiplet_basis(geometry, 2)
    plus = highest_basis @ coefficients
    zero = _lower_spin_one_member(plus)
    minus = _lower_spin_one_member(zero)
    for member in (plus, zero, minus):
        np.testing.assert_allclose(np.linalg.norm(member), 1.0, atol=4.0e-14)
    return plus, zero, minus


def _two_cluster_triplet_singlet(
    geometry: str, left_coefficients: np.ndarray, right_coefficients: np.ndarray
) -> np.ndarray:
    left_plus, left_zero, left_minus = _triplet_members(
        geometry, left_coefficients
    )
    right_plus, right_zero, right_minus = _triplet_members(
        geometry, right_coefficients
    )
    local_matrix = (
        np.outer(left_plus, right_minus)
        - np.outer(left_zero, right_zero)
        + np.outer(left_minus, right_plus)
    ) / np.sqrt(3.0)
    cluster_sites = ((0, 1, 2, 6, 7, 8), (3, 4, 5, 9, 10, 11))
    answer = np.empty(1 << 12, dtype=np.float64)
    for state in range(1 << 12):
        local = []
        for sites in cluster_sites:
            index = 0
            for site in sites:
                index = (index << 1) | ((state >> (11 - site)) & 1)
            local.append(index)
        answer[state] = local_matrix[local[0], local[1]]
    return answer


def _two_cluster_product(local: np.ndarray) -> np.ndarray:
    cluster_sites = ((0, 1, 2, 6, 7, 8), (3, 4, 5, 9, 10, 11))
    answer = np.empty(1 << 12, dtype=np.float64)
    for state in range(1 << 12):
        local_indices = []
        for sites in cluster_sites:
            index = 0
            for site in sites:
                index = (index << 1) | ((state >> (11 - site)) & 1)
            local_indices.append(index)
        answer[state] = local[local_indices[0]] * local[local_indices[1]]
    return answer


def _pure_routed_q2_params(params: dict, plan) -> dict:
    result = jax.tree_util.tree_map(jnp.zeros_like, params)
    patch = []
    for layout, source in zip(
        triangle_patch_pair_layout(plan), params["patch_pair"], strict=True
    ):
        patch.append(source if layout.q_parent == 2 else jnp.zeros_like(source))
    roots = []
    for layout, source in zip(triangle_root_layout(plan), params["root"], strict=True):
        block = jnp.zeros_like(source)
        if layout.q == 2:
            block = block.at[1, 1].set(1.0)
        roots.append(block)
    return {
        "patch_pair": tuple(patch),
        "reduced": result["reduced"],
        "root": tuple(roots),
        "patch_gates": tuple(jnp.ones_like(block) for block in params["patch_gates"]),
        "reduced_gates": result["reduced_gates"],
    }


@pytest.mark.parametrize(
    ("geometry", "ratio", "expected_overlap"),
    (("square", 0.5, 0.450594976027227), ("triangular", 0.125, 0.422347923216776)),
)
def test_dressed_left_right_carriers_solve_resolvent_and_are_not_one_mode(
    geometry: str, ratio: float, expected_overlap: float
) -> None:
    reference = triangle_patch_boundary_response_reference(geometry, ratio)
    np.testing.assert_allclose(
        reference.left_right_overlap, expected_overlap, atol=3.0e-13
    )
    assert reference.sylvester_residual < 2.0e-14
    assert reference.triplet_basis_orthonormality_error < 2.0e-14
    assert reference.carrier_basis_orthonormality_error < 2.0e-14
    assert reference.span_reconstruction_residual < 2.0e-14
    np.testing.assert_allclose(
        reference.carrier_basis.T @ reference.carrier_basis,
        np.eye(2),
        atol=2.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        reference.carrier_basis @ reference.left_coordinates,
        reference.left_coefficients,
        atol=2.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        reference.carrier_basis @ reference.right_coordinates,
        reference.right_coefficients,
        atol=2.0e-14,
        rtol=0.0,
    )
    assert abs(reference.left_right_overlap) < 0.5
    assert reference.correction_singular_values[1] > 0.2 * reference.correction_singular_values[0]
    audited = two_cluster_virtual_process_report(geometry, ratio)
    np.testing.assert_allclose(
        reference.correction_singular_values,
        audited.correction_singular_values,
        atol=3.0e-14,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("geometry", "ratio"), (("square", 0.5), ("triangular", 0.125))
)
def test_initializer_changes_only_q2_local_bank_and_adds_no_parameters(
    geometry: str, ratio: float
) -> None:
    plan = triangle_patch_grid_plan(
        geometry, 2, 6, feature_dim=2, maximum_intermediate_q=10
    )
    key = jax.random.PRNGKey(1701)
    original = init_triangle_patch_cluster_multiplets(
        key, plan, j2_over_j1=ratio, background_scale=0.05
    )
    response = init_triangle_patch_boundary_response_carriers(
        key, plan, j2_over_j1=ratio, background_scale=0.05
    )
    assert triangle_patch_parameter_count(response) == triangle_patch_parameter_count(
        original
    )
    for name in ("reduced", "root", "patch_gates", "reduced_gates"):
        for actual, expected in zip(response[name], original[name], strict=True):
            np.testing.assert_array_equal(actual, expected)
    q2_pieces = [[], []]
    for layout, actual, expected in zip(
        triangle_patch_pair_layout(plan),
        response["patch_pair"],
        original["patch_pair"],
        strict=True,
    ):
        if layout.q_parent != 2:
            np.testing.assert_array_equal(actual, expected)
        else:
            q2_pieces[0].append(np.asarray(actual)[0].reshape(-1))
            q2_pieces[1].append(np.asarray(actual)[1].reshape(-1))
    installed = np.column_stack(
        (np.concatenate(q2_pieces[0]), np.concatenate(q2_pieces[1]))
    )
    physical = triangle_patch_boundary_response_reference(geometry, ratio)
    np.testing.assert_allclose(installed, physical.carrier_basis, atol=0.0, rtol=0.0)
    routing = triangle_patch_boundary_response_routing(
        plan, j2_over_j1=ratio
    )
    assert routing.added_trainable_parameter_count == 0
    np.testing.assert_allclose(
        routing.left_rotation @ routing.left_rotation.T,
        np.eye(2),
        atol=2.0e-15,
    )


@pytest.mark.parametrize(
    ("geometry", "ratio"), (("square", 0.5), ("triangular", 0.125))
)
def test_n12_routed_q2_branch_is_exact_dressed_left_right_triplet_singlet(
    geometry: str, ratio: float
) -> None:
    plan = triangle_patch_grid_plan(
        geometry, 2, 6, feature_dim=2, maximum_intermediate_q=10
    )
    params = init_triangle_patch_boundary_response_carriers(
        jax.random.PRNGKey(1702), plan, j2_over_j1=ratio
    )
    routing = triangle_patch_boundary_response_routing(
        plan, j2_over_j1=ratio
    )
    pure = _pure_routed_q2_params(params, plan)
    configurations = _configurations(12)
    actual = np.asarray(
        triangle_patch_interface_amplitudes(
            pure, jnp.asarray(configurations), plan, routing
        )
    )
    reference = triangle_patch_boundary_response_reference(geometry, ratio)
    expected = _two_cluster_triplet_singlet(
        geometry, reference.left_coefficients, reference.right_coefficients
    )
    np.testing.assert_allclose(np.linalg.norm(actual), 1.0, atol=3.0e-13)
    np.testing.assert_allclose(actual, expected, atol=5.0e-14, rtol=0.0)

    # The routed tree realizes the same two-vector Ritz space as the exact
    # mechanism audit, not merely an abstract span in local coefficient space.
    partition = two_cluster_open_partition(geometry, ratio)
    hamiltonian, integer_basis = heisenberg_sz_matrix(
        12, partition.full_bonds, n_up=6
    )
    local_ground = triangle_patch_cluster_reference(
        geometry, ratio
    ).computational_state
    product_full = _two_cluster_product(local_ground)
    positions = np.asarray(integer_basis, dtype=np.int64)
    columns = np.column_stack((product_full[positions], actual[positions]))
    np.testing.assert_allclose(columns.T @ columns, np.eye(2), atol=4.0e-13)
    routed_energy = float(np.linalg.eigvalsh(columns.T @ hamiltonian @ columns)[0])
    report = two_cluster_virtual_process_report(geometry, ratio)
    registered = next(
        carrier
        for carrier in report.full_coupling_carriers
        if carrier.carrier == "resolvent_dressed_schmidt_k1"
    )
    np.testing.assert_allclose(routed_energy, registered.ground_energy, atol=4.0e-13)


def test_routed_forward_preserves_su2_multiaffinity_and_parameter_tangent() -> None:
    plan = triangle_patch_grid_plan(
        "triangular", 2, 6, feature_dim=2, maximum_intermediate_q=10
    )
    params = init_triangle_patch_boundary_response_carriers(
        jax.random.PRNGKey(1703), plan, j2_over_j1=0.125
    )
    routing = triangle_patch_boundary_response_routing(
        plan, j2_over_j1=0.125
    )
    rng = np.random.default_rng(1704)
    spinors = rng.normal(size=(12, 2)) + 1.0j * rng.normal(size=(12, 2))
    rotation = spin_half_su2_rotation(rng.normal(size=3), 0.81)
    tangent = jax.tree_util.tree_map(jnp.ones_like, params)
    original, original_tangent = jax.jvp(
        lambda candidate: triangle_patch_interface_amplitude_from_spinors(
            candidate, spinors, plan, routing
        ),
        (params,),
        (tangent,),
    )
    rotated, rotated_tangent = jax.jvp(
        lambda candidate: triangle_patch_interface_amplitude_from_spinors(
            candidate, spinors @ rotation.T, plan, routing
        ),
        (params,),
        (tangent,),
    )
    np.testing.assert_allclose(rotated, original, atol=8.0e-10, rtol=8.0e-10)
    np.testing.assert_allclose(
        rotated_tangent, original_tangent, atol=2.0e-8, rtol=2.0e-8
    )

    first = rng.normal(size=2) + 1.0j * rng.normal(size=2)
    second = rng.normal(size=2) + 1.0j * rng.normal(size=2)
    alpha, beta = 0.27 - 0.16j, -0.19 + 0.23j
    for site in (0, 5, 8, 11):
        combined = spinors.copy()
        combined[site] = alpha * first + beta * second
        first_input = spinors.copy()
        first_input[site] = first
        second_input = spinors.copy()
        second_input[site] = second
        actual = triangle_patch_interface_amplitude_from_spinors(
            params, combined, plan, routing
        )
        expected = alpha * triangle_patch_interface_amplitude_from_spinors(
            params, first_input, plan, routing
        ) + beta * triangle_patch_interface_amplitude_from_spinors(
            params, second_input, plan, routing
        )
        np.testing.assert_allclose(actual, expected, atol=2.0e-9, rtol=2.0e-9)


def test_external_translation_projection_closes_without_new_model_parameters() -> None:
    plan = triangle_patch_grid_plan(
        "square", 2, 6, feature_dim=2, maximum_intermediate_q=10
    )
    params = init_triangle_patch_boundary_response_carriers(
        jax.random.PRNGKey(1705), plan, j2_over_j1=0.5
    )
    routing = triangle_patch_boundary_response_routing(plan, j2_over_j1=0.5)
    group = rectangular_translation_group(2, 6)
    projector = PhysicalSpatialProjector(group, orbit_chunk_size=4)
    rng = np.random.default_rng(1706)
    states = []
    for _ in range(4):
        row = np.zeros(12, dtype=np.int32)
        row[rng.choice(12, 6, replace=False)] = 1
        states.append(row)
    states = np.asarray(states)
    image = np.asarray(group[5], dtype=np.int32)
    translated = states[:, image]
    amplitude_fn = lambda candidate, state, config: triangle_patch_interface_amplitude(
        candidate, state, config, routing
    )
    original = spatial_projected_amplitudes(
        params,
        jnp.asarray(states),
        plan,
        projector,
        amplitude_fn=amplitude_fn,
    )
    shifted = spatial_projected_amplitudes(
        params,
        jnp.asarray(translated),
        plan,
        projector,
        amplitude_fn=amplitude_fn,
    )
    assert np.linalg.norm(np.asarray(original)) > 1.0e-16
    np.testing.assert_allclose(shifted, original, atol=2.0e-13, rtol=0.0)


@pytest.mark.parametrize(
    ("geometry", "ratio"), (("square", 0.5), ("triangular", 0.125))
)
def test_n36_interface_route_forward_and_jvp_are_finite(
    geometry: str, ratio: float
) -> None:
    plan = triangle_patch_6x6_canary_plan(geometry)
    params = init_triangle_patch_boundary_response_carriers(
        jax.random.PRNGKey(1708), plan, j2_over_j1=ratio
    )
    routing = triangle_patch_boundary_response_routing(
        plan, j2_over_j1=ratio
    )
    assert triangle_patch_parameter_count(params) == plan.parameter_count == 1252
    rng = np.random.default_rng(1709)
    states = []
    for _ in range(3):
        row = np.zeros(36, dtype=np.int32)
        row[rng.choice(36, 18, replace=False)] = 1
        states.append(row)
    states = jnp.asarray(states)
    tangent = jax.tree_util.tree_map(jnp.ones_like, params)
    values, directional = jax.jvp(
        lambda candidate: triangle_patch_interface_amplitudes(
            candidate, states, plan, routing
        ),
        (params,),
        (tangent,),
    )
    assert values.shape == directional.shape == (3,)
    assert np.all(np.isfinite(np.asarray(values)))
    assert np.all(np.isfinite(np.asarray(directional)))
    assert np.linalg.norm(np.asarray(values)) > 1.0e-15
    assert np.linalg.norm(np.asarray(directional)) > 1.0e-15


def test_interface_routing_fails_closed_on_nonorthogonal_or_wrong_width() -> None:
    with pytest.raises(ValueError, match="orthogonal"):
        TrianglePatchInterfaceRouting(
            routed_q=2,
            support_size=6,
            left_rotation=np.ones((2, 2)),
            right_rotation=np.eye(2),
        )
    plan = triangle_patch_grid_plan(
        "square", 2, 6, feature_dim=2, maximum_intermediate_q=10
    )
    params = init_triangle_patch_boundary_response_carriers(
        jax.random.PRNGKey(1707), plan, j2_over_j1=0.5
    )
    bad = TrianglePatchInterfaceRouting(
        routed_q=2,
        support_size=6,
        left_rotation=np.eye(3),
        right_rotation=np.eye(3),
    )
    state = jnp.asarray([0, 1] * 6, dtype=jnp.int32)
    with pytest.raises(ValueError, match="feature width"):
        triangle_patch_interface_amplitude(params, state, plan, bad)


def test_identity_interface_frames_reproduce_the_established_forward_exactly() -> None:
    plan = triangle_patch_grid_plan(
        "square", 2, 6, feature_dim=2, maximum_intermediate_q=10
    )
    params = init_triangle_patch_boundary_response_carriers(
        jax.random.PRNGKey(1710), plan, j2_over_j1=0.5
    )
    identity = TrianglePatchInterfaceRouting(
        routed_q=2,
        support_size=6,
        left_rotation=np.eye(2),
        right_rotation=np.eye(2),
    )
    rng = np.random.default_rng(1711)
    states = []
    for _ in range(16):
        row = np.zeros(12, dtype=np.int32)
        row[rng.choice(12, 6, replace=False)] = 1
        states.append(row)
    states = jnp.asarray(states)
    expected = triangle_patch_amplitudes(params, states, plan)
    actual = triangle_patch_interface_amplitudes(params, states, plan, identity)
    np.testing.assert_allclose(actual, expected, atol=2.0e-13, rtol=0.0)
