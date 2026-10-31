from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.su2_vit_architecture_oracle import (
    spin_half_su2_rotation,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangle_patch_clebsch_tree import (
    TrianglePatchClebschTreePlan,
    TriangleSchurPatch,
    init_triangle_patch_clebsch_tree,
    triangle_patch_6x6_canary_plan,
    triangle_patch_6x6_plan,
    triangle_patch_amplitude,
    triangle_patch_amplitude_from_spinors,
    triangle_patch_amplitudes,
    triangle_patch_grid_plan,
    triangle_patch_parameter_count,
)


jax.config.update("jax_enable_x64", True)


def _all_configurations(n_sites: int) -> np.ndarray:
    return np.asarray(
        [
            [
                (state >> (n_sites - site - 1)) & 1
                for site in range(n_sites)
            ]
            for state in range(1 << n_sites)
        ],
        dtype=np.int32,
    )


def _random_tangent(key: jax.Array, tree: object) -> object:
    leaves, structure = jax.tree_util.tree_flatten(tree)
    tangent = [
        jax.random.normal(
            jax.random.fold_in(key, index), leaf.shape, dtype=leaf.dtype
        )
        for index, leaf in enumerate(leaves)
    ]
    return jax.tree_util.tree_unflatten(structure, tangent)


def _spin_raising_relative_norm(state: np.ndarray, configurations: np.ndarray) -> float:
    n_sites = configurations.shape[1]
    source_mask = configurations.sum(axis=1) == n_sites // 2
    source = configurations[source_mask]
    amplitudes = np.asarray(state)[source_mask]
    targets = configurations[configurations.sum(axis=1) == n_sites // 2 + 1]
    positions = {tuple(row): index for index, row in enumerate(targets)}
    raised = np.zeros(len(targets), dtype=np.complex128)
    for row, amplitude in zip(source, amplitudes, strict=True):
        for site in np.flatnonzero(row == 0):
            candidate = row.copy()
            candidate[site] = 1
            raised[positions[tuple(candidate)]] += amplitude
    norm = float(np.linalg.norm(amplitudes))
    if norm <= 1.0e-24:
        raise AssertionError("test state has zero numerical norm")
    return float(np.linalg.norm(raised) / norm)


def _unit_n6_parameter_sets(
    plan: TrianglePatchClebschTreePlan,
) -> tuple[dict[str, tuple[jax.Array, ...]], ...]:
    result = []
    count = sum(int(np.prod(shape)) for shape in plan.root_shapes)
    for selected in range(count):
        blocks = []
        offset = 0
        for shape in plan.root_shapes:
            size = int(np.prod(shape))
            flat = np.zeros(size, dtype=np.float64)
            if offset <= selected < offset + size:
                flat[selected - offset] = 1.0
            blocks.append(jnp.asarray(flat.reshape(shape)))
            offset += size
        result.append(
            {
                "patch_pair": (),
                "reduced": (),
                "root": tuple(blocks),
                "patch_gates": (),
                "reduced_gates": (),
            }
        )
    return tuple(result)


def test_n6_complete_triangle_carriers_span_all_five_singlets() -> None:
    plan = triangle_patch_grid_plan(
        "triangular", 2, 3, feature_dim=4, maximum_intermediate_q=4
    )
    assert plan.parameter_count == 5
    assert plan.patch_pair_shapes == ()
    assert plan.reduced_shapes == ()
    assert plan.root_shapes == ((2, 2), (1, 1))
    configurations = _all_configurations(6)
    columns = np.column_stack(
        [
            np.asarray(
                triangle_patch_amplitudes(params, configurations, plan)
            )
            for params in _unit_n6_parameter_sets(plan)
        ]
    )
    np.testing.assert_allclose(
        columns.conj().T @ columns,
        np.eye(5),
        atol=3.0e-15,
        rtol=0.0,
    )
    assert np.linalg.matrix_rank(columns, tol=1.0e-12) == 5


@pytest.mark.parametrize(
    ("rows", "columns", "expected_parameters"),
    ((2, 3, 5), (2, 6, 142)),
)
def test_n6_n12_exhaustive_states_and_parameter_tangents_are_singlets(
    rows: int, columns: int, expected_parameters: int
) -> None:
    plan = triangle_patch_grid_plan(
        "triangular",
        rows,
        columns,
        feature_dim=4,
        maximum_intermediate_q=4,
    )
    params = init_triangle_patch_clebsch_tree(
        jax.random.PRNGKey(plan.n_sites), plan, scale=0.7
    )
    tangent = _random_tangent(jax.random.PRNGKey(100 + plan.n_sites), params)
    configurations = jnp.asarray(_all_configurations(plan.n_sites))
    state, tangent_state = jax.jvp(
        lambda candidate: triangle_patch_amplitudes(
            candidate, configurations, plan
        ),
        (params,),
        (tangent,),
    )
    assert triangle_patch_parameter_count(params) == expected_parameters
    for vector in (np.asarray(state), np.asarray(tangent_state)):
        assert np.all(np.isfinite(vector))
        assert _spin_raising_relative_norm(
            vector, np.asarray(configurations)
        ) < 2.0e-12


def test_n12_rotation_invariance_and_physical_leg_multiaffinity() -> None:
    plan = triangle_patch_grid_plan(
        "square", 2, 6, feature_dim=3, maximum_intermediate_q=4
    )
    params = init_triangle_patch_clebsch_tree(
        jax.random.PRNGKey(71), plan, scale=0.6
    )
    rng = np.random.default_rng(72)
    spinors = rng.normal(size=(plan.n_sites, 2)) + 1.0j * rng.normal(
        size=(plan.n_sites, 2)
    )
    rotation = spin_half_su2_rotation(rng.normal(size=3), 0.73)
    original = triangle_patch_amplitude_from_spinors(params, spinors, plan)
    rotated = triangle_patch_amplitude_from_spinors(
        params, spinors @ rotation.T, plan
    )
    np.testing.assert_allclose(rotated, original, atol=3.0e-11, rtol=3.0e-11)

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
        actual = triangle_patch_amplitude_from_spinors(
            params, combined, plan
        )
        expected = alpha * triangle_patch_amplitude_from_spinors(
            params, first_input, plan
        ) + beta * triangle_patch_amplitude_from_spinors(
            params, second_input, plan
        )
        np.testing.assert_allclose(
            actual, expected, atol=5.0e-11, rtol=5.0e-11
        )


@pytest.mark.parametrize("geometry", ("square", "triangular"))
def test_n36_shared_tree_forward_and_jvp_smoke(geometry: str) -> None:
    plan = triangle_patch_6x6_plan(
        geometry, feature_dim=4, maximum_intermediate_q=4
    )
    assert len(plan.patches) == 12
    assert len(plan.routes) == 11
    assert plan.parameter_count == 1198
    assert len(plan.patch_pair_shapes) == 9
    assert len(plan.reduced_shapes) == 15
    assert plan.root_shapes == ((4, 4), (4, 4), (4, 4))
    for route in plan.routes:
        assert set(route.left.sites).isdisjoint(route.right.sites)
        assert set(route.left.sites) | set(route.right.sites) == set(
            route.parent.sites
        )

    params = init_triangle_patch_clebsch_tree(
        jax.random.PRNGKey(211), plan, scale=0.6
    )
    assert triangle_patch_parameter_count(params) == 1198
    tangent = _random_tangent(jax.random.PRNGKey(212), params)
    rng = np.random.default_rng(213)
    batch = []
    for _ in range(3):
        row = np.zeros(36, dtype=np.int32)
        row[rng.choice(36, size=18, replace=False)] = 1
        batch.append(row)
    configurations = jnp.asarray(batch)
    state, tangent_state = jax.jvp(
        lambda candidate: triangle_patch_amplitudes(
            candidate, configurations, plan
        ),
        (params,),
        (tangent,),
    )
    assert state.shape == tangent_state.shape == (3,)
    assert np.all(np.isfinite(np.asarray(state)))
    assert np.all(np.isfinite(np.asarray(tangent_state)))
    assert np.linalg.norm(np.asarray(state)) > 1.0e-16
    np.testing.assert_allclose(
        state[0],
        triangle_patch_amplitude(params, configurations[0], plan),
        atol=2.0e-13,
        rtol=2.0e-13,
    )


def test_n36_patch_supports_match_each_geometrys_physical_motif() -> None:
    """Square patches contain J1-J1-J2; triangular patches contain J1^3."""

    square = triangle_patch_6x6_plan(
        "square", feature_dim=2, maximum_intermediate_q=10
    )
    triangular = triangle_patch_6x6_plan(
        "triangular", feature_dim=2, maximum_intermediate_q=10
    )
    square_j1 = {
        (first, second)
        for first, second, _ in square_j1j2_bonds(
            6, 6, j1=1.0, j2=0.0, periodic=True
        )
    }
    square_j2 = {
        (first, second)
        for first, second, _ in square_j1j2_bonds(
            6, 6, j1=0.0, j2=1.0, periodic=True
        )
    }
    triangular_j1 = {
        (first, second)
        for first, second, _ in triangular_j1j2_bonds(
            6, 6, j1=1.0, j2=0.0, periodic=True
        )
    }

    for patch in square.patches:
        edges = {
            tuple(sorted((patch.sites[left], patch.sites[right])))
            for left, right in ((0, 1), (0, 2), (1, 2))
        }
        assert len(edges & square_j1) == 2
        assert len(edges & square_j2) == 1
    for patch in triangular.patches:
        edges = {
            tuple(sorted((patch.sites[left], patch.sites[right])))
            for left, right in ((0, 1), (0, 2), (1, 2))
        }
        assert edges <= triangular_j1


def test_d2_n36_q4_diagnostic_has_the_registered_284_parameter_layout() -> None:
    plan = triangle_patch_6x6_plan(
        "square", feature_dim=2, maximum_intermediate_q=4
    )
    params = init_triangle_patch_clebsch_tree(jax.random.PRNGKey(301), plan)
    assert plan.parameter_count == 284
    assert triangle_patch_parameter_count(params) == 284


@pytest.mark.parametrize("geometry", ("square", "triangular"))
def test_n36_generic_m0_support_distinguishes_q4_stress_from_q10_canary(
    geometry: str,
) -> None:
    """A hard q cutoff may create nodes even though SU(2) remains exact.

    Route-specific reduced gates remove accidental identical-subtree
    constraints, but cannot restore irreps deleted by ``qmax``.  The D2/q4
    arm is therefore an algebra/runtime diagnostic only.  The registered
    zero-support canary raises the cutoff to 10 before local-energy sampling.
    """

    rng = np.random.default_rng(401)
    batch = []
    for _ in range(2048):
        row = np.zeros(36, dtype=np.int32)
        row[rng.choice(36, size=18, replace=False)] = 1
        batch.append(row)
    configurations = jnp.asarray(batch)

    zero_counts = {}
    for maximum_q in (4, 10):
        plan = (
            triangle_patch_6x6_plan(
                geometry, feature_dim=2, maximum_intermediate_q=4
            )
            if maximum_q == 4
            else triangle_patch_6x6_canary_plan(geometry)
        )
        params = init_triangle_patch_clebsch_tree(
            jax.random.PRNGKey(402), plan, scale=0.6
        )
        amplitudes = np.asarray(
            triangle_patch_amplitudes(params, configurations, plan)
        )
        zero_counts[maximum_q] = int(np.count_nonzero(amplitudes == 0.0))
    assert zero_counts[4] > 0
    assert zero_counts[10] == 0
    assert triangle_patch_6x6_canary_plan(geometry).parameter_count == 1252


def test_plan_validation_fails_closed_on_bad_types_and_overlapping_patches() -> None:
    reference = triangle_patch_grid_plan(
        "square", 2, 3, feature_dim=2, maximum_intermediate_q=4
    )
    with pytest.raises(ValueError, match="positive integer"):
        triangle_patch_grid_plan("square", 2, 3, feature_dim=False)
    with pytest.raises(TypeError, match="maximum_intermediate_q"):
        triangle_patch_grid_plan(
            "square", 2, 3, feature_dim=2, maximum_intermediate_q=False
        )
    bad_patches = (
        TriangleSchurPatch((0, 1, 2)),
        TriangleSchurPatch((2, 4, 5)),
    )
    with pytest.raises(ValueError, match="cover physical sites"):
        TrianglePatchClebschTreePlan(
            geometry="square",
            rows=2,
            columns=3,
            patches=bad_patches,
            routes=reference.routes,
            feature_dim=2,
            maximum_intermediate_q=4,
        )
