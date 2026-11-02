"""Exact-small capacity diagnostics for a shared-parameter fusion-tree atlas.

The input columns are states produced by one frozen model in several physical
fusion charts and transported into one common symmetry sector.  This module
does not construct charts and never changes neural parameters.  It compares
their equal coherent sum, the Hamiltonian Ritz state in their span, and the
target-projection capacity ceiling.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..operators.permutation import apply_site_permutation_fusion
from ..su2.fusion_tree import FusionTree
from ..su2.tree_rotations import recoupling_matrix_sparse


def _normalized(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.complex128)
    if value.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have finite nonzero norm")
    return value / norm


def _metrics(
    state: np.ndarray, hamiltonian: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    vector = _normalized(state, name="state")
    applied = hamiltonian @ vector
    energy = float(np.real(np.vdot(vector, applied)))
    variance = float(max(np.real(np.vdot(applied, applied)) - energy**2, 0.0))
    return {
        "energy": energy,
        "variance": variance,
        "fidelity": float(abs(np.vdot(target, vector)) ** 2),
        "normalization_error": abs(float(np.vdot(vector, vector).real) - 1.0),
    }


def transport_tree_chart_state(
    state: np.ndarray,
    tree: FusionTree,
    target_q: int,
    reference_physical_order: Sequence[int],
    chart_physical_order: Sequence[int],
) -> tuple[np.ndarray, dict[str, object]]:
    """Transport one abstract-tree state from a chart into a reference frame.

    Both charts use the same ordered binary tree and the same amplitude vector;
    only the assignment of physical sites to leaves changes.  The transport is
    evaluated independently through the left-associated permutation
    representation, so it remains an exact-small reference implementation.
    """

    vector = _normalized(state, name="state")
    n_sites = tree.n_leaves
    reference = tuple(int(site) for site in reference_physical_order)
    chart = tuple(int(site) for site in chart_physical_order)
    expected_sites = tuple(range(n_sites))
    if tuple(sorted(reference)) != expected_sites:
        raise ValueError("reference_physical_order must be a site permutation")
    if tuple(sorted(chart)) != expected_sites:
        raise ValueError("chart_physical_order must be a site permutation")
    if vector.size != len(tree.enumerate_states(int(target_q))):
        raise ValueError("state dimension does not match tree and target_q")

    reference_position = {site: leaf for leaf, site in enumerate(reference)}
    leaf_image = tuple(reference_position[chart[leaf]] for leaf in range(n_sites))
    inverse_image = [0] * n_sites
    for old, new in enumerate(leaf_image):
        inverse_image[new] = old

    left_to_tree, moves = recoupling_matrix_sparse(
        FusionTree.left_associated(n_sites), tree, int(target_q)
    )
    left_state = np.asarray(left_to_tree.T @ vector, dtype=np.complex128)
    transported_left = apply_site_permutation_fusion(
        left_state, leaf_image, target_q=int(target_q)
    )
    transported = np.asarray(
        left_to_tree @ transported_left, dtype=np.complex128
    )
    restored_left = apply_site_permutation_fusion(
        transported_left, tuple(inverse_image), target_q=int(target_q)
    )
    restored = np.asarray(left_to_tree @ restored_left, dtype=np.complex128)
    audit = {
        "leaf_image": list(leaf_image),
        "recoupling_move_count": len(moves),
        "input_normalization_error": abs(float(np.vdot(vector, vector).real) - 1.0),
        "transport_normalization_error": abs(
            float(np.vdot(transported, transported).real) - 1.0
        ),
        "inverse_recovery_error": float(np.linalg.norm(restored - vector)),
    }
    return transported, audit


def fusion_atlas_oracle(
    components: np.ndarray,
    hamiltonian: np.ndarray,
    target: np.ndarray,
    *,
    component_names: Sequence[str] | None = None,
    relative_singular_cutoff: float = 1.0e-12,
) -> dict[str, object]:
    """Analyze the exact span of target-independent fusion-chart states.

    ``components`` has shape ``(sector_dimension, chart_count)``.  Columns are
    normalized internally.  The target is used only for retrospective
    fidelity metrics and the explicitly labelled capacity ceiling; the
    Hamiltonian Ritz state is target independent.
    """

    matrix = np.asarray(components, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("components must contain at least one state column")
    h = np.asarray(hamiltonian, dtype=np.complex128)
    if h.shape != (matrix.shape[0], matrix.shape[0]):
        raise ValueError("hamiltonian shape does not match atlas components")
    if not 0.0 < relative_singular_cutoff < 1.0:
        raise ValueError("relative_singular_cutoff must lie between zero and one")
    labels = (
        tuple(str(index) for index in range(matrix.shape[1]))
        if component_names is None
        else tuple(str(name) for name in component_names)
    )
    if len(labels) != matrix.shape[1] or len(set(labels)) != len(labels):
        raise ValueError("component_names must be distinct and match chart_count")
    target_state = _normalized(target, name="target")
    if target_state.size != matrix.shape[0]:
        raise ValueError("target shape does not match atlas components")

    normalized_columns = np.column_stack(
        [
            _normalized(matrix[:, index], name=f"component {labels[index]}")
            for index in range(matrix.shape[1])
        ]
    )
    left, singular_values, _ = np.linalg.svd(normalized_columns, full_matrices=False)
    retained = singular_values > relative_singular_cutoff * singular_values[0]
    basis = left[:, retained]
    gram = normalized_columns.conj().T @ normalized_columns

    reduced_hamiltonian = basis.conj().T @ h @ basis
    reduced_hamiltonian = 0.5 * (
        reduced_hamiltonian + reduced_hamiltonian.conj().T
    )
    ritz_values, ritz_vectors = np.linalg.eigh(reduced_hamiltonian)
    ritz_state = basis @ ritz_vectors[:, 0]

    projected_target = basis @ (basis.conj().T @ target_state)
    projected_norm_squared = float(np.vdot(projected_target, projected_target).real)
    if projected_norm_squared <= 1.0e-28:
        raise ValueError("atlas span has numerically zero target projection")

    equal_state = np.sum(normalized_columns, axis=1)
    if np.linalg.norm(equal_state) <= 1.0e-14:
        equal_metrics = None
    else:
        equal_metrics = _metrics(equal_state, h, target_state)

    component_metrics = [
        {
            "name": labels[index],
            **_metrics(normalized_columns[:, index], h, target_state),
        }
        for index in range(matrix.shape[1])
    ]
    best_component = max(component_metrics, key=lambda row: float(row["fidelity"]))
    result: dict[str, object] = {
        "component_names": list(labels),
        "component_count": matrix.shape[1],
        "effective_rank": int(np.count_nonzero(retained)),
        "singular_values": [float(value) for value in singular_values],
        "gram_hermiticity_error": float(np.max(np.abs(gram - gram.conj().T))),
        "orthonormal_basis_error": float(
            np.max(
                np.abs(basis.conj().T @ basis - np.eye(basis.shape[1])),
                initial=0.0,
            )
        ),
        "component_metrics": component_metrics,
        "base_component": component_metrics[0],
        "best_component": best_component,
        "equal_coherent": equal_metrics,
        "hamiltonian_ritz": _metrics(ritz_state, h, target_state),
        "target_projection_ceiling": _metrics(projected_target, h, target_state),
        "target_projection_weight": projected_norm_squared,
        "target_used_to_construct_components": False,
        "target_used_by_hamiltonian_ritz": False,
        "target_used_by_projection_ceiling": True,
    }
    return result


__all__ = ["fusion_atlas_oracle", "transport_tree_chart_state"]
