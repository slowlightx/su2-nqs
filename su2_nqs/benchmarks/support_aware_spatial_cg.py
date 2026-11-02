"""Exact-small architecture oracle for support-aware Spatial-CG states."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .clebsch_tree_exact import (
    fixed_m_spin_basis,
    physical_permutation_index_table,
    reynolds_project_state,
)
from .exact_diagonalization import spin_squared_matrix
from .su2_vit_architecture_oracle import spin_half_su2_rotation
from ..lattices.geometry_tree import morton_order
from ..lattices.j1j2 import square_j1j2_bonds
from ..lattices.spatial_routes import (
    bond_shell_route_features,
    rectangular_support_route_orders,
)
from ..lattices.spatial_symmetry import (
    rectangular_translation_group,
    triangular_hnf_translation_group,
)
from ..lattices.triangular_j1j2 import triangular_j1j2_bonds
from ..models.spin_clebsch_tree import SpinClebschTreeConfig
from ..models.support_aware_spatial_cg import (
    SupportAwareSpatialCGConfig,
    init_support_aware_spatial_cg,
    support_aware_spatial_cg_amplitudes,
    support_aware_spatial_cg_component_matrix,
    support_aware_spatial_cg_parameter_count,
)
from ..su2.fusion_tree import FusionTree


jax.config.update("jax_enable_x64", True)


def spatial_cg_route_spec(
    lattice: str,
    rows: int,
    columns: int,
    *,
    feature_dim: int,
    maximum_intermediate_q: int | None,
    initialization_scale: float,
) -> tuple[SupportAwareSpatialCGConfig, dict[str, object]]:
    """Construct the physical route bank and its shell-locality ledger."""

    n_sites = rows * columns
    tree = FusionTree.balanced(n_sites)
    routes = rectangular_support_route_orders(
        lattice, rows, columns, drop_duplicates=True
    )
    if lattice == "square":
        j1 = square_j1j2_bonds(
            rows, columns, j1=1.0, j2=0.0, periodic=True
        )
        j2 = square_j1j2_bonds(
            rows, columns, j1=0.0, j2=1.0, periodic=True
        )
    elif lattice == "triangular":
        j1 = triangular_j1j2_bonds(
            rows, columns, j1=1.0, j2=0.0, periodic=True
        )
        j2 = triangular_j1j2_bonds(
            rows, columns, j1=0.0, j2=1.0, periodic=True
        )
    else:
        raise ValueError("lattice must be 'square' or 'triangular'")
    features = bond_shell_route_features(
        tree,
        routes,
        {"J1": j1, "J2": j2},
    )
    reference_order = morton_order(rows, columns)
    matches = [
        index
        for index, order in enumerate(routes.values())
        if order == reference_order
    ]
    if len(matches) != 1:
        raise RuntimeError("route bank does not contain one unique Morton baseline")
    config = SupportAwareSpatialCGConfig(
        backbone=SpinClebschTreeConfig(
            tree,
            target_q=0,
            target_m2=0,
            feature_dim=feature_dim,
            maximum_intermediate_q=maximum_intermediate_q,
            initialization_scale=initialization_scale,
        ),
        route_names=tuple(routes),
        route_orders=tuple(routes.values()),
        route_features=features.standardized_localities,
        feature_names=features.labels,
        reference_route=matches[0],
    )
    ledger = {
        "route_names": list(config.route_names),
        "route_orders": [list(order) for order in config.route_orders],
        "reference_route": config.route_names[config.reference_route],
        "feature_names": list(config.feature_names),
        "raw_mean_tree_distances": [
            list(row) for row in features.raw_mean_tree_distances
        ],
        "standardized_localities": [
            list(row) for row in features.standardized_localities
        ],
        "all_routes_cover_every_site_once": all(
            tuple(sorted(order)) == tuple(range(n_sites))
            for order in config.route_orders
        ),
    }
    return config, ledger


def _all_spin_configurations(n_sites: int) -> np.ndarray:
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


def _rotate_state(
    state: np.ndarray,
    rotation: np.ndarray,
    n_sites: int,
) -> np.ndarray:
    tensor = np.asarray(state, dtype=np.complex128).reshape((2,) * n_sites)
    for axis in range(n_sites):
        tensor = np.tensordot(rotation, tensor, axes=(1, axis))
        tensor = np.moveaxis(tensor, 0, axis)
    return tensor.reshape(-1)


def _maximum_rotation_error(state: np.ndarray, *, seed: int) -> float:
    n_sites = int(round(np.log2(state.size)))
    norm = float(np.linalg.norm(state))
    if norm <= 0.0 or 1 << n_sites != state.size:
        raise ValueError("state must be a nonzero full computational-basis vector")
    rng = np.random.default_rng(seed)
    maximum = 0.0
    for _ in range(4):
        rotation = spin_half_su2_rotation(
            rng.normal(size=3), rng.uniform(-2.0 * np.pi, 2.0 * np.pi)
        )
        maximum = max(
            maximum,
            float(np.linalg.norm(_rotate_state(state, rotation, n_sites) - state) / norm),
        )
    return maximum


def _numerical_rank_spectrum(matrix: np.ndarray) -> dict[str, object]:
    values = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    if values.size == 0 or values[0] <= 0.0:
        raise ValueError("component matrix has zero numerical norm")
    ratios = values / values[0]
    return {
        "singular_value_ratios": [float(value) for value in ratios],
        "rank_at_1e_minus_10": int(np.count_nonzero(ratios > 1.0e-10)),
        "condition_number_on_retained_span": float(
            values[0] / values[np.count_nonzero(ratios > 1.0e-10) - 1]
        ),
    }


def support_aware_spatial_cg_oracle(
    lattice: str,
    *,
    seed: int = 20260916,
    tangent_count: int = 8,
) -> dict[str, object]:
    """Certify the N=8 state, random tangents, and projected route span."""

    if tangent_count < 1:
        raise ValueError("tangent_count must be positive")
    rows, columns = 2, 4
    n_sites = rows * columns
    config, ledger = spatial_cg_route_spec(
        lattice,
        rows,
        columns,
        feature_dim=3,
        maximum_intermediate_q=4,
        initialization_scale=0.4,
    )
    params = init_support_aware_spatial_cg(
        jax.random.PRNGKey(seed), config
    )
    query = jnp.linspace(
        -0.31, 0.27, config.n_support_features, dtype=jnp.float64
    )
    params = {**params, "geometry_query": query}
    configurations = jnp.asarray(_all_spin_configurations(n_sites))

    def coefficient_state(candidate):
        return support_aware_spatial_cg_amplitudes(
            candidate, configurations, config
        )

    state = np.asarray(coefficient_state(params), dtype=np.complex128)
    spin_squared, integer_basis = spin_squared_matrix(n_sites)
    if integer_basis != tuple(range(1 << n_sites)):
        raise RuntimeError("full computational basis order changed")
    state_residual = float(
        np.linalg.norm(spin_squared @ state) / np.linalg.norm(state)
    )
    rotation_error = _maximum_rotation_error(state, seed=seed + 1)

    tangent_s2 = []
    tangent_rotation = []
    keys = jax.random.split(jax.random.PRNGKey(seed + 2), tangent_count)
    for tangent_index in range(tangent_count):
        # ``tree_map`` cannot safely advance a Python iterator inside a JAX
        # trace, so fold the tangent index into one deterministic key per leaf.
        leaves, structure = jax.tree_util.tree_flatten(params)
        tangent_leaves = [
            jax.random.normal(
                jax.random.fold_in(keys[tangent_index], leaf_index),
                leaf.shape,
                dtype=leaf.dtype,
            )
            for leaf_index, leaf in enumerate(leaves)
        ]
        tangent = jax.tree_util.tree_unflatten(structure, tangent_leaves)
        _, tangent_state_device = jax.jvp(
            coefficient_state, (params,), (tangent,)
        )
        tangent_state = np.asarray(tangent_state_device, dtype=np.complex128)
        tangent_norm = float(np.linalg.norm(tangent_state))
        if tangent_norm <= 1.0e-14:
            raise RuntimeError("a random parameter tangent vanished")
        tangent_s2.append(
            float(np.linalg.norm(spin_squared @ tangent_state) / tangent_norm)
        )
        tangent_rotation.append(
            _maximum_rotation_error(tangent_state, seed=seed + 10 + tangent_index)
        )

    zero_query = {**params, "geometry_query": jnp.zeros_like(query)}
    nested_state = np.asarray(coefficient_state(zero_query))
    reference = np.asarray(
        support_aware_spatial_cg_component_matrix(
            zero_query, configurations, config
        )[:, config.reference_route]
    )
    nested_error = float(np.max(np.abs(nested_state - reference)))

    m0_basis = fixed_m_spin_basis(n_sites, 0)
    m0 = jnp.asarray(m0_basis, dtype=jnp.int32)
    components = np.asarray(
        support_aware_spatial_cg_component_matrix(params, m0, config)
    )
    if lattice == "square":
        group = rectangular_translation_group(rows, columns)
    else:
        group = triangular_hnf_translation_group(((columns, 0), (0, rows)))
    indices = physical_permutation_index_table(m0_basis, group)
    projected = np.column_stack(
        [
            np.asarray(reynolds_project_state(components[:, route], indices))
            for route in range(config.n_routes)
        ]
    )
    mixed_m0 = state[
        np.asarray(
            [
                sum(bit << (n_sites - site - 1) for site, bit in enumerate(row))
                for row in m0_basis
            ],
            dtype=np.int32,
        )
    ]
    projected_mixed = np.asarray(reynolds_project_state(mixed_m0, indices))
    closure = float(
        np.linalg.norm(
            np.asarray(reynolds_project_state(projected_mixed, indices))
            - projected_mixed
        )
        / np.linalg.norm(projected_mixed)
    )
    return {
        "lattice": lattice,
        "rows": rows,
        "columns": columns,
        "n_sites": n_sites,
        "route_ledger": ledger,
        "route_count": config.n_routes,
        "support_feature_count": config.n_support_features,
        "parameter_count": support_aware_spatial_cg_parameter_count(params),
        "backbone_parameter_count": support_aware_spatial_cg_parameter_count(params)
        - config.n_support_features,
        "nested_single_tree_max_abs_error": nested_error,
        "spin_squared_residual_ratio": state_residual,
        "global_rotation_residual_ratio": rotation_error,
        "maximum_tangent_spin_squared_residual_ratio": float(max(tangent_s2)),
        "maximum_tangent_global_rotation_residual_ratio": float(
            max(tangent_rotation)
        ),
        "tangent_count": tangent_count,
        "raw_component_span": _numerical_rank_spectrum(components),
        "translation_projected_component_span": _numerical_rank_spectrum(projected),
        "translation_projection_closure_error": closure,
        "passed": bool(
            nested_error < 1.0e-13
            and state_residual < 2.0e-11
            and rotation_error < 2.0e-11
            and max(tangent_s2) < 5.0e-10
            and max(tangent_rotation) < 5.0e-10
            and closure < 2.0e-12
            and _numerical_rank_spectrum(projected)["rank_at_1e_minus_10"] > 1
        ),
    }


__all__ = [
    "spatial_cg_route_spec",
    "support_aware_spatial_cg_oracle",
]
