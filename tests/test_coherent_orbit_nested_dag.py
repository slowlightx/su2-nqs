from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.lattices.fusion_catalog import (
    DisjointMergeRoute,
    FusionSupport,
    ReducedFusionCatalog,
)
from su2_nqs.lattices.geometry_tree import morton_order
from su2_nqs.lattices.reduced_fusion_catalogs import diamond_fusion_catalog
from su2_nqs.lattices.spatial_symmetry import (
    compose_site_permutations,
    rectangular_translation_group,
)
from su2_nqs.models.coherent_orbit_nested_dag import (
    SpinCoherentOrbitNestedDAGConfig,
    coherent_group_label_product,
    init_spin_coherent_orbit_nested_dag,
    spin_coherent_orbit_nested_dag_amplitude,
    spin_coherent_orbit_nested_dag_amplitudes,
    spin_coherent_orbit_nested_dag_components,
    spin_coherent_orbit_nested_dag_parameter_count,
)
from su2_nqs.models.persistent_route_pair_dag import (
    SpinPersistentRoutePairConfig,
    select_group_closed_persistent_pair_supports,
    spin_persistent_route_pair_dag_amplitude,
)
from su2_nqs.models.reduced_fusion_dag import (
    SpinReducedFusionDAGConfig,
    spin_reduced_fusion_dag_amplitude,
)
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector,
    spatial_projected_amplitudes,
)


jax.config.update("jax_enable_x64", True)


def _nested_n8_catalogs(
    *, resonance_group: bool = False
) -> tuple[
    ReducedFusionCatalog,
    ReducedFusionCatalog,
    FusionSupport,
    tuple[tuple[int, ...], ...],
]:
    """Return a tree contained in a two-diamond resonating seed."""

    leaves = tuple(FusionSupport((site,)) for site in range(8))
    p01 = FusionSupport((0, 1))
    p02 = FusionSupport((0, 2))
    left = FusionSupport((0, 1, 2))
    p34 = FusionSupport((3, 4))
    p35 = FusionSupport((3, 5))
    right = FusionSupport((3, 4, 5))
    joined = FusionSupport((0, 1, 2, 3, 4, 5))
    tail = FusionSupport((6, 7))
    root = FusionSupport(tuple(range(8)))
    r01 = DisjointMergeRoute(p01, leaves[0], leaves[1])
    r02 = DisjointMergeRoute(p02, leaves[0], leaves[2])
    rleft_1 = DisjointMergeRoute(left, p01, leaves[2])
    rleft_2 = DisjointMergeRoute(left, p02, leaves[1])
    r34 = DisjointMergeRoute(p34, leaves[3], leaves[4])
    r35 = DisjointMergeRoute(p35, leaves[3], leaves[5])
    rright_1 = DisjointMergeRoute(right, p34, leaves[5])
    rright_2 = DisjointMergeRoute(right, p35, leaves[4])
    rjoined = DisjointMergeRoute(joined, left, right)
    rtail = DisjointMergeRoute(tail, leaves[6], leaves[7])
    rroot = DisjointMergeRoute(root, joined, tail)
    identity = tuple(range(8))
    reflection = (0, 2, 1, 3, 5, 4, 6, 7)
    physical_group = (identity, reflection)

    tree_routes = (r01, rleft_1, r34, rright_1, rjoined, rtail, rroot)
    tree_supports = {
        *leaves,
        p01,
        left,
        p34,
        right,
        joined,
        tail,
        root,
    }
    tree = ReducedFusionCatalog(
        8, (identity,), tuple(tree_supports), tree_routes
    )
    seed_routes = (
        r01,
        r02,
        rleft_1,
        rleft_2,
        r34,
        r35,
        rright_1,
        rright_2,
        rjoined,
        rtail,
        rroot,
    )
    seed_supports = {
        *leaves,
        p01,
        p02,
        left,
        p34,
        p35,
        right,
        joined,
        tail,
        root,
    }
    seed = ReducedFusionCatalog(
        8,
        physical_group if resonance_group else (identity,),
        tuple(seed_supports),
        seed_routes,
    )
    return tree, seed, joined, physical_group


def _config(
    *, alpha_group_closed_seed: bool = False
) -> SpinCoherentOrbitNestedDAGConfig:
    tree, seed, joined, physical_group = _nested_n8_catalogs(
        resonance_group=alpha_group_closed_seed
    )
    baseline = SpinReducedFusionDAGConfig(
        tree,
        target_q=0,
        target_m2=0,
        feature_dim=2,
        route_mode="single_route",
    )
    resonance = SpinPersistentRoutePairConfig(
        seed,
        (joined,),
        target_q=0,
        target_m2=0,
        feature_dim=2,
        pair_mode=(
            "equivariant_full_rank" if alpha_group_closed_seed else "full_rank"
        ),
    )
    return SpinCoherentOrbitNestedDAGConfig(
        baseline,
        resonance,
        PhysicalSpatialProjector(physical_group),
    )


def _spin_squared_in_basis(basis: Sequence[tuple[int, ...]]) -> np.ndarray:
    n_sites = len(basis[0])
    matrix, integer_basis = spin_squared_matrix(
        n_sites, n_up=sum(basis[0])
    )
    positions = {value: index for index, value in enumerate(integer_basis)}
    order = [
        positions[
            sum(bit << (n_sites - site - 1) for site, bit in enumerate(row))
        ]
        for row in basis
    ]
    return matrix[np.ix_(order, order)]


def _translation_coherent_n12_config() -> SpinCoherentOrbitNestedDAGConfig:
    rows, columns = 3, 4
    order = morton_order(rows, columns)
    baseline = SpinReducedFusionDAGConfig(
        diamond_fusion_catalog(order, resonating=False),
        target_q=0,
        target_m2=0,
        feature_dim=1,
        route_mode="single_route",
    )
    # Crucially this remains the identity-group seed.  Translated copies are
    # represented by the explicit h carrier below, not closed independently
    # into every local support.
    seed = diamond_fusion_catalog(order, resonating=True)
    pair_supports = select_group_closed_persistent_pair_supports(seed)
    assert len(pair_supports) == 1
    resonance = SpinPersistentRoutePairConfig(
        seed,
        pair_supports,
        target_q=0,
        target_m2=0,
        feature_dim=1,
        pair_mode="equivariant_full_rank",
    )
    projector = PhysicalSpatialProjector(
        rectangular_translation_group(rows, columns)
    )
    return SpinCoherentOrbitNestedDAGConfig(baseline, resonance, projector)


def _projected_tree_and_dag(
    params: dict[str, object],
    basis: jax.Array,
    config: SpinCoherentOrbitNestedDAGConfig,
) -> tuple[jax.Array, jax.Array]:
    resonance = params["resonance"]
    tree_params = {
        "merges": resonance["merges"],
        "readout": resonance["readout"],
    }
    tree = spatial_projected_amplitudes(
        tree_params,
        basis,
        config.baseline,
        config.projector,
        amplitude_fn=spin_reduced_fusion_dag_amplitude,
    )
    dag = spatial_projected_amplitudes(
        resonance,
        basis,
        config.resonance,
        config.projector,
        amplitude_fn=spin_persistent_route_pair_dag_amplitude,
    )
    return tree, dag


def test_global_group_label_product_excludes_independent_cross_terms() -> None:
    local = jnp.asarray([[1.0, 2.0], [3.0, 5.0]], dtype=jnp.float64)
    coherent = coherent_group_label_product(local)
    independent_local_sums = jnp.prod(jnp.mean(local, axis=0))
    np.testing.assert_allclose(coherent, 8.5, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(independent_local_sums, 7.0, atol=0.0, rtol=0.0)
    assert not np.isclose(coherent, independent_local_sums)


def test_alpha_zero_is_exact_projected_tree_and_alpha_one_is_seed_dag() -> None:
    config = _config()
    params = init_spin_coherent_orbit_nested_dag(
        jax.random.PRNGKey(810), config, alpha=0.0
    )
    basis = jnp.asarray(fixed_m_spin_basis(8, 0))
    tree, dag = _projected_tree_and_dag(params, basis, config)
    at_zero = spin_coherent_orbit_nested_dag_amplitudes(params, basis, config)
    np.testing.assert_allclose(at_zero, tree, atol=2.0e-15, rtol=2.0e-15)

    at_one_params = {**params, "alpha": jnp.asarray(1.0, dtype=jnp.float64)}
    at_one = spin_coherent_orbit_nested_dag_amplitudes(
        at_one_params, basis, config
    )
    np.testing.assert_allclose(at_one, dag, atol=2.0e-15, rtol=2.0e-15)
    # The extension is non-vacuous: a generic full-rank local kernel is not
    # simply the anchored tree state.
    assert np.linalg.norm(np.asarray(dag - tree)) > 1.0e-12


def test_global_h_carrier_covariance_a1_and_exact_total_spin() -> None:
    config = _config()
    params = init_spin_coherent_orbit_nested_dag(
        jax.random.PRNGKey(811), config, alpha=0.37
    )
    basis_values = fixed_m_spin_basis(8, 0)
    basis = jnp.asarray(basis_values)
    vector = np.asarray(
        spin_coherent_orbit_nested_dag_amplitudes(params, basis, config)
    )
    assert np.linalg.norm(vector) > 1.0e-13
    s2 = _spin_squared_in_basis(basis_values)
    assert np.linalg.norm(s2 @ vector) / np.linalg.norm(vector) < 3.0e-12

    group = config.projector.site_permutations
    nonidentity = group[1]
    transformed_basis = basis[:, jnp.asarray(nonidentity)]
    transformed = np.asarray(
        spin_coherent_orbit_nested_dag_amplitudes(
            params, transformed_basis, config
        )
    )
    np.testing.assert_allclose(transformed, vector, atol=3.0e-14, rtol=3.0e-14)

    row = basis[9]
    tree_h, dag_h = spin_coherent_orbit_nested_dag_components(params, row, config)
    moved_tree_h, moved_dag_h = spin_coherent_orbit_nested_dag_components(
        params, row[jnp.asarray(nonidentity)], config
    )
    carrier_permutation = [
        group.index(compose_site_permutations(nonidentity, image))
        for image in group
    ]
    np.testing.assert_allclose(
        moved_tree_h,
        tree_h[jnp.asarray(carrier_permutation)],
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        moved_dag_h,
        dag_h[jnp.asarray(carrier_permutation)],
        atol=2.0e-14,
        rtol=2.0e-14,
    )


def test_nested_model_is_jittable_and_differentiable() -> None:
    config = _config()
    params = init_spin_coherent_orbit_nested_dag(
        jax.random.PRNGKey(812), config, alpha=0.23
    )
    flat, unravel = ravel_pytree(params)
    state = jnp.asarray(fixed_m_spin_basis(8, 0)[13])

    def amplitude(vector: jax.Array) -> jax.Array:
        return spin_coherent_orbit_nested_dag_amplitude(
            unravel(vector), state, config
        )

    compiled = jax.jit(amplitude)(flat)
    eager = amplitude(flat)
    np.testing.assert_allclose(compiled, eager, atol=2.0e-14, rtol=2.0e-14)
    tangent = jnp.linspace(-0.2, 0.3, flat.size, dtype=jnp.float64)
    _, jvp = jax.jvp(amplitude, (flat,), (tangent,))
    gradient = jax.grad(lambda vector: jnp.abs(amplitude(vector)) ** 2)(flat)
    assert np.all(np.isfinite(np.asarray(jvp)))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.linalg.norm(np.asarray(gradient)) > 1.0e-14
    assert spin_coherent_orbit_nested_dag_parameter_count(params) == flat.size


def test_n12_translation_seed_is_strictly_nested_statewise_and_tangentially() -> None:
    config = _translation_coherent_n12_config()
    params = init_spin_coherent_orbit_nested_dag(
        jax.random.PRNGKey(814), config, alpha=0.0
    )
    basis = jnp.asarray(fixed_m_spin_basis(12, 0))
    nested = spin_coherent_orbit_nested_dag_amplitudes(params, basis, config)
    resonance = params["resonance"]
    tree_params = {
        "merges": resonance["merges"],
        "readout": resonance["readout"],
    }
    projected_tree = spatial_projected_amplitudes(
        tree_params,
        basis,
        config.baseline,
        config.projector,
        amplitude_fn=spin_reduced_fusion_dag_amplitude,
    )
    np.testing.assert_allclose(
        nested, projected_tree, atol=3.0e-15, rtol=3.0e-15
    )

    # A unit alpha tangent is exactly the coherently projected local-DAG
    # correction.  This checks nesting at model/Jacobian level, rather than
    # only comparing two coincident initial state vectors.
    alpha_tangent = jax.tree_util.tree_map(jnp.zeros_like, params)
    alpha_tangent = {**alpha_tangent, "alpha": jnp.asarray(1.0)}
    subset = basis[:24]
    _, tangent = jax.jvp(
        lambda candidate: spin_coherent_orbit_nested_dag_amplitudes(
            candidate, subset, config
        ),
        (params,),
        (alpha_tangent,),
    )
    tree_h, dag_h = jax.vmap(
        lambda row: spin_coherent_orbit_nested_dag_components(params, row, config)
    )(subset)
    expected_tangent = jnp.mean(dag_h - tree_h, axis=1)
    np.testing.assert_allclose(
        tangent, expected_tangent, atol=3.0e-14, rtol=3.0e-14
    )
    assert np.linalg.norm(np.asarray(tangent)) > 1.0e-14

    # At alpha=0, changing route-pair weights alone leaves the exact tree
    # submanifold fixed.  The kernel becomes trainable immediately after the
    # alpha direction turns on.
    pair_tangent = jax.tree_util.tree_map(jnp.zeros_like, params)
    resonance_tangent = dict(pair_tangent["resonance"])
    resonance_tangent["pair_orbit_weights"] = jax.tree_util.tree_map(
        jnp.ones_like, resonance_tangent["pair_orbit_weights"]
    )
    pair_tangent = {**pair_tangent, "resonance": resonance_tangent}
    _, pair_derivative = jax.jvp(
        lambda candidate: spin_coherent_orbit_nested_dag_amplitudes(
            candidate, subset, config
        ),
        (params,),
        (pair_tangent,),
    )
    np.testing.assert_allclose(pair_derivative, 0.0, atol=2.0e-15, rtol=0.0)


def test_group_closed_internal_seed_is_rejected_fail_closed() -> None:
    with pytest.raises(ValueError, match="unclosed identity-group seed"):
        _config(alpha_group_closed_seed=True)


def test_parameter_and_shape_errors_fail_at_public_boundary() -> None:
    config = _config()
    params = init_spin_coherent_orbit_nested_dag(
        jax.random.PRNGKey(813), config
    )
    state = jnp.asarray(fixed_m_spin_basis(8, 0)[0])
    with pytest.raises(ValueError, match="parameter keys"):
        spin_coherent_orbit_nested_dag_amplitude(
            {"resonance": params["resonance"]}, state, config
        )
    with pytest.raises(ValueError, match="alpha must be a scalar"):
        spin_coherent_orbit_nested_dag_amplitude(
            {**params, "alpha": jnp.ones((1,))}, state, config
        )
    with pytest.raises(ValueError, match="expected spin shape"):
        spin_coherent_orbit_nested_dag_amplitude(params, state[:-1], config)
    with pytest.raises(ValueError, match="expected spin batch shape"):
        spin_coherent_orbit_nested_dag_amplitudes(params, state, config)
    with pytest.raises(ValueError, match="characters"):
        coherent_group_label_product(jnp.ones((2, 2)), jnp.ones((3,)))
