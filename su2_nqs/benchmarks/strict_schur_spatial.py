"""Exact-small gates for strict spatially generated Schur/CG coefficients."""

from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from .clebsch_tree_exact import (
    coo_matvec,
    fixed_m_spin_basis,
    graph_hamiltonian_coo,
    physical_permutation_index_table,
    reynolds_project_state,
)
from .exact_diagonalization import spin_squared_matrix
from .su2_vit_architecture_oracle import spin_half_su2_rotation
from ..lattices.spatial_symmetry import (
    rectangular_translation_group,
    triangular_hnf_translation_group,
)
from ..lattices.j1j2 import square_j1j2_bonds
from ..models.strict_schur_spatial import (
    four_plaquette_amplitude_from_spinors,
    four_plaquette_amplitudes,
    four_plaquette_d4_plan,
    init_four_plaquette_reduced,
    StrictSchurSpatialPlan,
    init_static_schur_generator,
    init_strict_schur_reduced,
    square_plaquette_plan,
    static_schur_reduced_blocks,
    strict_schur_amplitude_from_spinors,
    strict_schur_amplitudes,
    strict_schur_parameter_count,
    strict_schur_spatial_amplitudes,
    strict_schur_spatial_descriptors,
    two_plaquette_plan,
    two_triangle_plan,
)
from ..su2.local_schur import cell_schur_unitarity_error


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


def _unit_reduced_basis(plan: StrictSchurSpatialPlan) -> tuple[tuple[jax.Array, ...], ...]:
    result = []
    for selected in range(plan.reduced_parameter_count):
        blocks = []
        offset = 0
        for shape in plan.reduced_shapes:
            size = int(np.prod(shape))
            flat = np.zeros(size, dtype=np.float64)
            if offset <= selected < offset + size:
                flat[selected - offset] = 1.0
            blocks.append(jnp.asarray(flat.reshape(shape)))
            offset += size
        result.append(tuple(blocks))
    return tuple(result)


def strict_schur_coefficient_basis(plan: StrictSchurSpatialPlan) -> np.ndarray:
    """Return full-basis columns associated with unit reduced coefficients."""

    configurations = jnp.asarray(_all_configurations(plan.n_sites))
    columns = [
        np.asarray(strict_schur_amplitudes(blocks, configurations, plan))
        for blocks in _unit_reduced_basis(plan)
    ]
    matrix = np.column_stack(columns)
    matrix.setflags(write=False)
    return matrix


def _rotate_state(state: np.ndarray, rotation: np.ndarray, n_sites: int) -> np.ndarray:
    tensor = np.asarray(state, dtype=np.complex128).reshape((2,) * n_sites)
    for axis in range(n_sites):
        tensor = np.tensordot(rotation, tensor, axes=(1, axis))
        tensor = np.moveaxis(tensor, 0, axis)
    return tensor.reshape(-1)


def _rotation_error(state: np.ndarray, *, seed: int) -> float:
    norm = float(np.linalg.norm(state))
    if norm <= 0.0:
        raise ValueError("state must be nonzero")
    rng = np.random.default_rng(seed)
    maximum = 0.0
    for _ in range(3):
        rotation = spin_half_su2_rotation(
            rng.normal(size=3), rng.uniform(-2.0 * np.pi, 2.0 * np.pi)
        )
        maximum = max(
            maximum,
            float(
                np.linalg.norm(_rotate_state(state, rotation, int(round(np.log2(state.size)))) - state)
                / norm
            ),
        )
    return maximum


def _multi_affinity_defect(
    blocks: tuple[jax.Array, ...],
    plan: StrictSchurSpatialPlan,
    *,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    spinors = rng.normal(size=(plan.n_sites, 2)) + 1.0j * rng.normal(
        size=(plan.n_sites, 2)
    )
    first = rng.normal(size=2) + 1.0j * rng.normal(size=2)
    second = rng.normal(size=2) + 1.0j * rng.normal(size=2)
    alpha = 0.41 - 0.23j
    beta = -0.37 + 0.19j
    maximum = 0.0
    for site in range(plan.n_sites):
        combined = spinors.copy()
        combined[site] = alpha * first + beta * second
        first_input = spinors.copy()
        first_input[site] = first
        second_input = spinors.copy()
        second_input[site] = second
        actual = complex(
            strict_schur_amplitude_from_spinors(blocks, combined, plan)
        )
        expected = alpha * complex(
            strict_schur_amplitude_from_spinors(blocks, first_input, plan)
        ) + beta * complex(
            strict_schur_amplitude_from_spinors(blocks, second_input, plan)
        )
        maximum = max(
            maximum,
            abs(actual - expected) / max(1.0, abs(actual), abs(expected)),
        )
    return float(maximum)


def _random_tangent(key: jax.Array, tree: object) -> object:
    leaves, structure = jax.tree_util.tree_flatten(tree)
    tangent = [
        jax.random.normal(
            jax.random.fold_in(key, index), leaf.shape, dtype=leaf.dtype
        )
        for index, leaf in enumerate(leaves)
    ]
    return jax.tree_util.tree_unflatten(structure, tangent)


def _projected_spin_residual(
    state: np.ndarray,
    plan: StrictSchurSpatialPlan,
) -> tuple[float, float]:
    basis = fixed_m_spin_basis(plan.n_sites, 0)
    integers = np.asarray(
        [
            sum(bit << (plan.n_sites - site - 1) for site, bit in enumerate(row))
            for row in basis
        ],
        dtype=np.int32,
    )
    sector = np.asarray(state)[integers]
    if plan.n_sites == 4:
        group = rectangular_translation_group(2, 2)
    elif plan.n_sites == 6:
        group = triangular_hnf_translation_group(((3, 0), (0, 2)))
    elif plan.n_sites == 8:
        group = rectangular_translation_group(2, 4)
    else:  # pragma: no cover - plans are validated by the public factory set
        raise ValueError("unsupported exact-small plan")
    indices = physical_permutation_index_table(basis, group)
    projected = np.asarray(reynolds_project_state(jnp.asarray(sector), indices))
    closure = float(
        np.linalg.norm(
            np.asarray(reynolds_project_state(jnp.asarray(projected), indices))
            - projected
        )
        / np.linalg.norm(projected)
    )
    spin_squared, spin_basis = spin_squared_matrix(
        plan.n_sites, n_up=plan.n_sites // 2
    )
    positions = {int(value): index for index, value in enumerate(spin_basis)}
    try:
        order = [positions[int(value)] for value in integers]
    except KeyError as error:  # pragma: no cover - independent basis invariant
        raise RuntimeError("fixed-M bases contain different configurations") from error
    spin_squared = spin_squared[np.ix_(order, order)]
    residual = float(
        np.linalg.norm(spin_squared @ projected) / np.linalg.norm(projected)
    )
    return closure, residual


def _plan_from_name(name: str) -> StrictSchurSpatialPlan:
    if name == "square_n4":
        return square_plaquette_plan()
    if name == "triangular_n6":
        return two_triangle_plan()
    if name == "square_n8":
        return two_plaquette_plan()
    raise ValueError("name must be square_n4, triangular_n6, or square_n8")


def strict_schur_spatial_oracle(
    name: str,
    *,
    seed: int = 20260917,
) -> dict[str, object]:
    """Certify completeness, multi-affinity, exact spin, and generator nesting."""

    plan = _plan_from_name(name)
    expected_rank = {
        4: 2,
        6: 5,
        8: 14,
    }[plan.n_sites]
    combinatorial_rank = math.comb(plan.n_sites, plan.n_sites // 2) - math.comb(
        plan.n_sites, plan.n_sites // 2 - 1
    )
    coefficient_basis = strict_schur_coefficient_basis(plan)
    gram_error = float(
        np.linalg.norm(
            coefficient_basis.conj().T @ coefficient_basis
            - np.eye(expected_rank)
        )
    )
    singular = np.linalg.svd(coefficient_basis, compute_uv=False)
    numerical_rank = int(np.count_nonzero(singular > singular[0] * 1.0e-12))
    spin_squared, full_basis = spin_squared_matrix(plan.n_sites)
    if tuple(full_basis) != tuple(range(1 << plan.n_sites)):
        raise RuntimeError("full computational basis order changed")
    basis_spin_residual = float(
        np.linalg.norm(spin_squared @ coefficient_basis)
        / np.linalg.norm(coefficient_basis)
    )

    blocks = init_strict_schur_reduced(jax.random.PRNGKey(seed), plan, scale=0.4)
    configurations = jnp.asarray(_all_configurations(plan.n_sites))
    state = np.asarray(strict_schur_amplitudes(blocks, configurations, plan))
    state_spin_residual = float(
        np.linalg.norm(spin_squared @ state) / np.linalg.norm(state)
    )
    outside_m0 = np.asarray(
        [row.sum() != plan.n_sites // 2 for row in np.asarray(configurations)]
    )
    magnetization_leakage = float(
        np.linalg.norm(state[outside_m0]) / np.linalg.norm(state)
    )
    rotation_error = _rotation_error(state, seed=seed + 1)
    multi_affinity = _multi_affinity_defect(blocks, plan, seed=seed + 2)
    projection_closure, projected_spin_residual = _projected_spin_residual(
        state, plan
    )

    generator = init_static_schur_generator(
        jax.random.PRNGKey(seed + 3), plan, hidden_dim=5, scale=0.3
    )
    nested_blocks = static_schur_reduced_blocks(generator, plan)
    nested_error = float(
        max(
            np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0.0)
            for actual, expected in zip(
                nested_blocks, generator["direct"], strict=True
            )
        )
    )
    active_generator = {
        **generator,
        "spatial": {
            **generator["spatial"],
            "residual_scale": jnp.asarray(0.37, dtype=jnp.float64),
        },
    }

    def generated_state(candidate):
        return strict_schur_spatial_amplitudes(candidate, configurations, plan)

    tangent = _random_tangent(jax.random.PRNGKey(seed + 4), active_generator)
    generated, tangent_state_device = jax.jvp(
        generated_state, (active_generator,), (tangent,)
    )
    generated = np.asarray(generated)
    tangent_state = np.asarray(tangent_state_device)
    generated_spin_residual = float(
        np.linalg.norm(spin_squared @ generated) / np.linalg.norm(generated)
    )
    tangent_spin_residual = float(
        np.linalg.norm(spin_squared @ tangent_state) / np.linalg.norm(tangent_state)
    )
    tangent_rotation_error = _rotation_error(tangent_state, seed=seed + 5)
    epsilon = 2.0e-5
    plus = jax.tree_util.tree_map(
        lambda value, direction: value + epsilon * direction,
        active_generator,
        tangent,
    )
    minus = jax.tree_util.tree_map(
        lambda value, direction: value - epsilon * direction,
        active_generator,
        tangent,
    )
    finite_difference = (
        np.asarray(generated_state(plus)) - np.asarray(generated_state(minus))
    ) / (2.0 * epsilon)
    finite_difference_error = float(
        np.linalg.norm(finite_difference - tangent_state)
        / np.linalg.norm(tangent_state)
    )

    scale_tangent = jax.tree_util.tree_map(jnp.zeros_like, generator)
    scale_tangent = {
        **scale_tangent,
        "spatial": {
            **scale_tangent["spatial"],
            "residual_scale": jnp.asarray(1.0, dtype=jnp.float64),
        },
    }
    _, geometry_tangent = jax.jvp(
        generated_state, (generator,), (scale_tangent,)
    )
    geometry_tangent_norm = float(np.linalg.norm(np.asarray(geometry_tangent)))

    local_errors = {
        cell.kind: cell_schur_unitarity_error(cell.kind) for cell in plan.cells
    }
    descriptor_shape = list(strict_schur_spatial_descriptors(plan).shape)
    passed = bool(
        expected_rank == combinatorial_rank == plan.reduced_parameter_count
        and numerical_rank == expected_rank
        and gram_error < 5.0e-12
        and basis_spin_residual < 5.0e-12
        and state_spin_residual < 5.0e-12
        and magnetization_leakage < 5.0e-14
        and rotation_error < 5.0e-12
        and multi_affinity < 5.0e-12
        and projection_closure < 5.0e-12
        and projected_spin_residual < 5.0e-12
        and nested_error < 1.0e-14
        and generated_spin_residual < 5.0e-12
        and tangent_spin_residual < 2.0e-10
        and tangent_rotation_error < 2.0e-10
        and finite_difference_error < 2.0e-8
        and geometry_tangent_norm > 1.0e-10
        and max(local_errors.values()) < 5.0e-12
    )
    return {
        "name": name,
        "n_sites": plan.n_sites,
        "cell_kinds": [cell.kind for cell in plan.cells],
        "cell_supports": [list(cell.support) for cell in plan.cells],
        "ledger_merge": None
        if plan.ledger_merge is None
        else {
            "parent": list(plan.ledger_merge.parent.sites),
            "left": list(plan.ledger_merge.left.sites),
            "right": list(plan.ledger_merge.right.sites),
        },
        "active_doubled_q": list(plan.active_q),
        "reduced_shapes": [list(shape) for shape in plan.reduced_shapes],
        "complete_singlet_rank_expected": expected_rank,
        "complete_singlet_rank_combinatorial": combinatorial_rank,
        "complete_singlet_rank_observed": numerical_rank,
        "coefficient_basis_gram_error": gram_error,
        "coefficient_basis_spin_squared_residual_ratio": basis_spin_residual,
        "state_spin_squared_residual_ratio": state_spin_residual,
        "magnetization_leakage_ratio": magnetization_leakage,
        "global_rotation_residual_ratio": rotation_error,
        "physical_leg_multi_affinity_defect": multi_affinity,
        "translation_projection_closure_error": projection_closure,
        "projected_spin_squared_residual_ratio": projected_spin_residual,
        "local_schur_unitarity_errors": local_errors,
        "spatial_descriptor_shape": descriptor_shape,
        "direct_reduced_parameter_count": plan.reduced_parameter_count,
        "generator_parameter_count": strict_schur_parameter_count(generator),
        "generator_nested_direct_max_abs_error": nested_error,
        "generated_state_spin_squared_residual_ratio": generated_spin_residual,
        "generator_tangent_spin_squared_residual_ratio": tangent_spin_residual,
        "generator_tangent_global_rotation_residual_ratio": tangent_rotation_error,
        "generator_jvp_central_difference_relative_error": finite_difference_error,
        "geometry_residual_scale_tangent_norm": geometry_tangent_norm,
        "passed": passed,
        "claim_boundary": (
            "Complete N<=8 singlet coefficient-space and geometry-only generator "
            "gate; not an N=16 energy, scaling, or SU2-Spatial-ViT win."
        ),
    }


def strict_schur_spatial_oracle_suite() -> dict[str, object]:
    """Run the complete N=4,6,8 promotion gate."""

    reports = {
        name: strict_schur_spatial_oracle(name)
        for name in ("square_n4", "triangular_n6", "square_n8")
    }
    return {
        "reports": reports,
        "passed": all(report["passed"] for report in reports.values()),
        "next_gate": (
            "Four complete plaquette carriers on square 4x4 with registered "
            "D=4 compression and a same-topology nonspatial nest."
        ),
    }


def _spin_raising_relative_norm(
    amplitudes: np.ndarray,
    basis: tuple[tuple[int, ...], ...],
) -> float:
    """Apply total ``S+`` from a complete fixed-``M`` basis without ``S2``."""

    vector = np.asarray(amplitudes)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (len(basis),) or norm <= np.finfo(np.float64).tiny:
        raise ValueError("a nonzero complete fixed-M vector is required")
    n_sites = len(basis[0])
    n_up = sum(basis[0])
    target = fixed_m_spin_basis(n_sites, 2 * (n_up + 1) - n_sites)
    positions = {configuration: index for index, configuration in enumerate(target)}
    raised = np.zeros(len(target), dtype=np.result_type(vector, np.complex128))
    for configuration, amplitude in zip(basis, vector, strict=True):
        for site, bit in enumerate(configuration):
            if bit == 0:
                candidate = configuration[:site] + (1,) + configuration[site + 1 :]
                raised[positions[candidate]] += amplitude
    return float(np.linalg.norm(raised) / norm)


def _four_plaquette_multi_affinity_defect(
    params: dict[str, object],
    *,
    seed: int,
) -> float:
    """Check affine dependence independently in all sixteen physical legs."""

    plan = four_plaquette_d4_plan()
    rng = np.random.default_rng(seed)
    spinors = rng.normal(size=(plan.n_sites, 2)) + 1.0j * rng.normal(
        size=(plan.n_sites, 2)
    )
    first = rng.normal(size=2) + 1.0j * rng.normal(size=2)
    second = rng.normal(size=2) + 1.0j * rng.normal(size=2)
    alpha, beta = 0.41 - 0.23j, -0.37 + 0.19j
    evaluate = jax.jit(
        lambda values: four_plaquette_amplitude_from_spinors(params, values, plan)
    )
    maximum = 0.0
    for site in range(plan.n_sites):
        combined = spinors.copy()
        combined[site] = alpha * first + beta * second
        first_input = spinors.copy()
        first_input[site] = first
        second_input = spinors.copy()
        second_input[site] = second
        actual = complex(evaluate(jnp.asarray(combined)))
        expected = alpha * complex(evaluate(jnp.asarray(first_input))) + beta * complex(
            evaluate(jnp.asarray(second_input))
        )
        maximum = max(
            maximum,
            abs(actual - expected) / max(1.0, abs(actual), abs(expected)),
        )
    return float(maximum)


def _four_plaquette_reduced_coefficients(
    params: dict[str, object],
) -> jax.Array:
    """Pack the unprojected full-singlet reduced coefficient matrices."""

    plan = four_plaquette_d4_plan()
    left_blocks, right_blocks = params["level1"]
    coefficients = []
    for (q, width), root in zip(
        plan.intermediate_widths, params["root"], strict=True
    ):
        selected = [
            index
            for index, layout in enumerate(plan.level1_layout)
            if layout.q_total == q
        ]
        left = jnp.concatenate(
            [left_blocks[index].reshape((width, -1)) for index in selected],
            axis=1,
        )
        right = jnp.concatenate(
            [right_blocks[index].reshape((width, -1)) for index in selected],
            axis=1,
        )
        coefficients.append(jnp.ravel(left.T @ root @ right))
    return jnp.concatenate(coefficients)


def _four_plaquette_generic_tangent_rank(
    params: dict[str, object],
) -> dict[str, float | int]:
    """Measure the raw-to-physical generic rank before lattice projection."""

    plan = four_plaquette_d4_plan()
    jacobian_tree = jax.jacrev(_four_plaquette_reduced_coefficients)(params)
    output_size = int(_four_plaquette_reduced_coefficients(params).size)
    jacobian = np.concatenate(
        [
            np.asarray(leaf).reshape((output_size, -1))
            for leaf in jax.tree_util.tree_leaves(jacobian_tree)
        ],
        axis=1,
    )
    singular = np.linalg.svd(jacobian, compute_uv=False)
    threshold = float(max(jacobian.shape) * np.finfo(np.float64).eps * singular[0])
    rank = int(np.count_nonzero(singular > threshold))
    return {
        "unprojected_complete_singlet_reduced_dimension": output_size,
        "raw_trainable_scalar_count": jacobian.shape[1],
        "internal_gauge_nullity": plan.internal_gauge_dimension,
        "unprojected_generic_reduced_coefficient_jacobian_rank_expected": (
            plan.generic_effective_tangent_dimension
        ),
        "unprojected_generic_reduced_coefficient_jacobian_rank_observed": rank,
        "jacobian_rank_threshold": threshold,
        "smallest_retained_singular_value": float(singular[rank - 1]),
        "largest_null_singular_value": float(singular[rank]),
        "retained_to_null_singular_value_ratio": float(
            singular[rank - 1] / singular[rank]
        ),
    }


def four_plaquette_n16_smoke(*, seed: int = 20260917) -> dict[str, object]:
    """Run the registered square-``4x4`` forward/JVP/``Hpsi`` bridge gate.

    This exhausts only the fixed-``M=0`` configuration basis.  It never
    diagonalizes the Hamiltonian and its random-state Rayleigh quotient is not
    an optimization or energy-quality result.
    """

    plan = four_plaquette_d4_plan()
    params = init_four_plaquette_reduced(jax.random.PRNGKey(seed), plan)
    tangent = _random_tangent(jax.random.PRNGKey(seed + 1), params)
    rank_report = _four_plaquette_generic_tangent_rank(params)
    multi_affinity = _four_plaquette_multi_affinity_defect(
        params, seed=seed + 2
    )
    basis = fixed_m_spin_basis(plan.n_sites, 0)
    configurations = jnp.asarray(np.asarray(basis, dtype=np.int32))

    def state_fn(candidate):
        return four_plaquette_amplitudes(candidate, configurations, plan)

    compiled_state_jvp = jax.jit(
        lambda candidate, direction: jax.jvp(
            state_fn, (candidate,), (direction,)
        )
    )
    started = time.perf_counter()
    state_device, tangent_device = compiled_state_jvp(params, tangent)
    state_device.block_until_ready()
    tangent_device.block_until_ready()
    forward_jvp_seconds = time.perf_counter() - started
    state = np.asarray(state_device)
    tangent_state = np.asarray(tangent_device)

    group = rectangular_translation_group(4, 4)
    permutation_indices = physical_permutation_index_table(basis, group)
    m2_basis = fixed_m_spin_basis(plan.n_sites, 2)
    m2_permutation_indices = physical_permutation_index_table(m2_basis, group)
    translation_m0_dimension = int(
        np.unique(np.min(permutation_indices, axis=0)).size
    )
    translation_m2_dimension = int(
        np.unique(np.min(m2_permutation_indices, axis=0)).size
    )
    translation_singlet_dimension = (
        translation_m0_dimension - translation_m2_dimension
    )
    probe_count = 112
    parameter_leaves, parameter_structure = jax.tree_util.tree_flatten(params)
    probe_leaves = [
        jax.random.normal(
            jax.random.fold_in(jax.random.PRNGKey(seed + 3), index),
            (probe_count,) + leaf.shape,
            dtype=leaf.dtype,
        )
        for index, leaf in enumerate(parameter_leaves)
    ]
    probe_tangents = jax.tree_util.tree_unflatten(
        parameter_structure, probe_leaves
    )

    def projected_tangent_fn(direction):
        _, candidate_tangent = jax.jvp(
            state_fn, (params,), (direction,)
        )
        return reynolds_project_state(candidate_tangent, permutation_indices)

    started = time.perf_counter()
    probe_microbatch = 8
    compiled_projected_tangent = jax.jit(jax.vmap(projected_tangent_fn))
    projected_tangent_rows = []
    for start in range(0, probe_count, probe_microbatch):
        stop = start + probe_microbatch
        directions = jax.tree_util.tree_map(
            lambda values: values[start:stop], probe_tangents
        )
        projected_tangent_rows.extend(
            np.asarray(compiled_projected_tangent(directions))
        )
    projected_rank_seconds = time.perf_counter() - started
    projected_tangent_matrix = np.stack(projected_tangent_rows, axis=1)
    projected_singular = np.linalg.svd(
        projected_tangent_matrix, compute_uv=False
    )
    projected_rank_threshold = float(
        max(projected_tangent_matrix.shape)
        * np.finfo(np.float64).eps
        * projected_singular[0]
    )
    projected_tangent_rank = int(
        np.count_nonzero(projected_singular > projected_rank_threshold)
    )
    projected_device = reynolds_project_state(state_device, permutation_indices)
    projected_tangent_device = reynolds_project_state(
        tangent_device, permutation_indices
    )
    projected_twice = reynolds_project_state(projected_device, permutation_indices)
    projected = np.asarray(projected_device)
    projection_closure = float(
        np.linalg.norm(np.asarray(projected_twice) - projected)
        / np.linalg.norm(projected)
    )

    bonds = square_j1j2_bonds(4, 4, j2=0.5, periodic=True)
    rows, columns, values, hamiltonian_basis = graph_hamiltonian_coo(
        plan.n_sites, 0, bonds
    )
    if hamiltonian_basis != basis:
        raise RuntimeError("Hamiltonian and amplitude basis order changed")

    def action_fn(candidate):
        candidate_state = state_fn(candidate)
        return coo_matvec(rows, columns, values, candidate_state)

    compiled_action_jvp = jax.jit(
        lambda candidate, direction: jax.jvp(
            action_fn, (candidate,), (direction,)
        )
    )
    started = time.perf_counter()
    action_device, tangent_action_device = compiled_action_jvp(params, tangent)
    action_device.block_until_ready()
    tangent_action_device.block_until_ready()
    hpsi_jvp_seconds = time.perf_counter() - started
    action = np.asarray(action_device)
    tangent_action = np.asarray(tangent_action_device)
    expected_tangent_action = np.asarray(
        coo_matvec(rows, columns, values, tangent_device)
    )
    hpsi_jvp_error = float(
        np.linalg.norm(tangent_action - expected_tangent_action)
        / np.linalg.norm(expected_tangent_action)
    )
    projected_action_from_h = np.asarray(
        coo_matvec(rows, columns, values, projected_device)
    )
    projected_action_from_p = np.asarray(
        reynolds_project_state(action_device, permutation_indices)
    )
    projected_hpsi_commutator_error = float(
        np.linalg.norm(projected_action_from_h - projected_action_from_p)
        / np.linalg.norm(projected_action_from_h)
    )
    projected_tangent_action_from_h = np.asarray(
        coo_matvec(rows, columns, values, projected_tangent_device)
    )
    projected_tangent_action_from_p = np.asarray(
        reynolds_project_state(tangent_action_device, permutation_indices)
    )
    projected_tangent_hpsi_commutator_error = float(
        np.linalg.norm(
            projected_tangent_action_from_h - projected_tangent_action_from_p
        )
        / np.linalg.norm(projected_tangent_action_from_h)
    )
    epsilon = 5.0e-6
    plus = jax.tree_util.tree_map(
        lambda value, direction: value + epsilon * direction, params, tangent
    )
    minus = jax.tree_util.tree_map(
        lambda value, direction: value - epsilon * direction, params, tangent
    )
    compiled_state = jax.jit(state_fn)
    compiled_action = jax.jit(action_fn)
    state_finite_difference = (
        np.asarray(compiled_state(plus)) - np.asarray(compiled_state(minus))
    ) / (2.0 * epsilon)
    action_finite_difference = (
        np.asarray(compiled_action(plus)) - np.asarray(compiled_action(minus))
    ) / (2.0 * epsilon)
    state_jvp_finite_difference_error = float(
        np.linalg.norm(state_finite_difference - tangent_state)
        / np.linalg.norm(tangent_state)
    )
    hpsi_jvp_finite_difference_error = float(
        np.linalg.norm(action_finite_difference - tangent_action)
        / np.linalg.norm(tangent_action)
    )
    norm_squared = np.vdot(state, state)
    energy = np.vdot(state, action) / norm_squared
    basis_array = np.asarray(basis, dtype=np.int8)
    hz_diagonal = np.zeros(len(basis), dtype=np.float64)
    for first, second, coupling in bonds:
        hz_diagonal += float(coupling) * np.where(
            basis_array[:, first] == basis_array[:, second], 0.25, -0.25
        )
    three_hz_energy = 3.0 * np.vdot(state, hz_diagonal * state) / norm_squared
    full_energy_vs_three_hz_error = float(abs(energy - three_hz_energy))
    variance = float(
        np.vdot(action - energy * state, action - energy * state).real
        / norm_squared.real
    )

    state_splus = _spin_raising_relative_norm(state, basis)
    tangent_splus = _spin_raising_relative_norm(tangent_state, basis)
    hpsi_splus = _spin_raising_relative_norm(action, basis)
    projected_splus = _spin_raising_relative_norm(projected, basis)
    projected_hpsi_splus = _spin_raising_relative_norm(
        projected_action_from_h, basis
    )
    finite = bool(
        np.all(np.isfinite(state))
        and np.all(np.isfinite(tangent_state))
        and np.all(np.isfinite(action))
        and np.all(np.isfinite(tangent_action))
    )
    passed = bool(
        finite
        and np.linalg.norm(state) > 1.0e-12
        and np.linalg.norm(tangent_state) > 1.0e-12
        and state_splus < 1.0e-11
        and tangent_splus < 1.0e-11
        and hpsi_splus < 1.0e-11
        and projected_splus < 1.0e-11
        and projected_hpsi_splus < 1.0e-11
        and projection_closure < 1.0e-11
        and projected_hpsi_commutator_error < 1.0e-11
        and projected_tangent_hpsi_commutator_error < 1.0e-11
        and multi_affinity < 5.0e-12
        and state_jvp_finite_difference_error < 2.0e-8
        and hpsi_jvp_finite_difference_error < 2.0e-8
        and full_energy_vs_three_hz_error < 1.0e-11
        and abs(energy.imag) < 1.0e-11
        and hpsi_jvp_error < 1.0e-11
        and rank_report[
            "unprojected_generic_reduced_coefficient_jacobian_rank_observed"
        ]
        == plan.generic_effective_tangent_dimension
        and translation_singlet_dimension == 107
        and projected_tangent_rank == translation_singlet_dimension
    )
    return {
        "system": "square_4x4_J2_over_J1_0.5_PBC",
        "n_sites": plan.n_sites,
        "fixed_m0_basis_size": len(basis),
        "cell_supports": [list(cell.support) for cell in plan.cells],
        "level1_support_routes": [
            {
                "parent": list(route.parent.sites),
                "left": list(route.left.sites),
                "right": list(route.right.sites),
            }
            for route in plan.level1_routes
        ],
        "root_support_route": {
            "parent": list(plan.root_route.parent.sites),
            "left": list(plan.root_route.left.sites),
            "right": list(plan.root_route.right.sites),
        },
        "local_multiplicities": {"q0": 2, "q2": 3, "q4": 1},
        "intermediate_multiplicities": {
            f"q{q}": multiplicity
            for q, multiplicity in plan.intermediate_multiplicities
        },
        "retained_widths": {f"q{q}": width for q, width in plan.intermediate_widths},
        "level1_block_count_per_route": len(plan.level1_layout),
        "level1_parameter_count_per_route": sum(
            int(np.prod(block.shape)) for block in plan.level1_layout
        ),
        "root_parameter_count": sum(int(np.prod(shape)) for shape in plan.root_shapes),
        "raw_trainable_scalar_count": plan.reduced_parameter_count,
        "internal_gauge_nullity": plan.internal_gauge_dimension,
        "unprojected_generic_effective_tangent_dimension": (
            plan.generic_effective_tangent_dimension
        ),
        **rank_report,
        "translation_invariant_m0_codomain_dimension": translation_m0_dimension,
        "translation_invariant_m2_codomain_dimension": translation_m2_dimension,
        "translation_invariant_singlet_codomain_dimension": (
            translation_singlet_dimension
        ),
        "translation_projected_random_tangent_probe_count": probe_count,
        "translation_projected_tangent_probe_microbatch": probe_microbatch,
        "translation_projected_random_tangent_rank_observed": (
            projected_tangent_rank
        ),
        "translation_projected_normalized_physical_tangent_dimension": (
            projected_tangent_rank - 1
        ),
        "translation_projected_tangent_rank_threshold": (
            projected_rank_threshold
        ),
        "translation_projected_smallest_retained_singular_value": float(
            projected_singular[projected_tangent_rank - 1]
        ),
        "translation_projected_largest_null_singular_value": float(
            projected_singular[projected_tangent_rank]
        ),
        "translation_projected_retained_to_null_singular_value_ratio": float(
            projected_singular[projected_tangent_rank - 1]
            / projected_singular[projected_tangent_rank]
        ),
        "finite_state_tangent_hpsi": finite,
        "physical_leg_multi_affinity_defect": multi_affinity,
        "state_norm": float(np.linalg.norm(state)),
        "tangent_state_norm": float(np.linalg.norm(tangent_state)),
        "state_spin_raising_relative_norm": state_splus,
        "tangent_spin_raising_relative_norm": tangent_splus,
        "hpsi_spin_raising_relative_norm": hpsi_splus,
        "translation_projected_spin_raising_relative_norm": projected_splus,
        "translation_projected_hpsi_spin_raising_relative_norm": (
            projected_hpsi_splus
        ),
        "translation_projection_closure_error": projection_closure,
        "translation_hpsi_commutator_relative_error": (
            projected_hpsi_commutator_error
        ),
        "translation_tangent_hpsi_commutator_relative_error": (
            projected_tangent_hpsi_commutator_error
        ),
        "hamiltonian_coo_nonzeros_with_duplicates": int(values.size),
        "random_state_rayleigh_quotient": float(energy.real),
        "random_state_three_hz_rayleigh_quotient": float(
            three_hz_energy.real
        ),
        "full_energy_vs_three_hz_abs_error": (
            full_energy_vs_three_hz_error
        ),
        "random_state_energy_variance": variance,
        "random_state_energy_imaginary_abs": float(abs(energy.imag)),
        "hpsi_jvp_linearity_relative_error": hpsi_jvp_error,
        "state_jvp_central_difference_relative_error": (
            state_jvp_finite_difference_error
        ),
        "hpsi_jvp_central_difference_relative_error": (
            hpsi_jvp_finite_difference_error
        ),
        "forward_jvp_wall_seconds_including_compile": forward_jvp_seconds,
        "hpsi_jvp_wall_seconds_including_compile": hpsi_jvp_seconds,
        "translation_projected_tangent_rank_wall_seconds_including_compile": (
            projected_rank_seconds
        ),
        "hamiltonian_diagonalization_performed": False,
        "passed": passed,
        "promotion_decision": "pass_n16_runtime_smoke_only" if passed else "fail_n16_runtime_smoke",
        "claim_boundary": (
            "Strict square-4x4 forward/JVP/Hpsi readiness only; the random-state "
            "Rayleigh quotient is not an optimized energy, efficiency, or physics win."
        ),
    }
__all__ = [
    "four_plaquette_n16_smoke",
    "strict_schur_coefficient_basis",
    "strict_schur_spatial_oracle",
    "strict_schur_spatial_oracle_suite",
]
