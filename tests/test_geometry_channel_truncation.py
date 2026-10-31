"""Exact unit checks for matched-rank physical-prefix truncations."""

from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.geometry_channel_truncation import (
    aggregate_truncation_metrics,
    exact_truncation_metrics,
    nested_prefix_order,
    order_to_active_image,
    permute_bonds_to_leaf_order,
    permuted_fusion_state,
    retained_path_indices,
)
from su2_nqs.benchmarks.nested_block_spin import permute_computational_state
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.operators.permutation import (
    heisenberg_graph_fusion_matrix,
    site_permutation_fusion_matrix,
)
from su2_nqs.su2.fusion_path import enumerate_paths


def test_registered_n12_cutoffs_have_exact_dimensions() -> None:
    paths = enumerate_paths(12, 0)
    assert len(paths) == 132
    assert len(retained_path_indices(paths, {4: 2, 6: 2})) == 97
    assert len(retained_path_indices(paths, {4: 0, 6: 0})) == 10


def test_leaf_order_action_matches_computational_basis_and_hamiltonian_covariance() -> None:
    n_sites = 6
    paths = enumerate_paths(n_sites, 0)
    transform = coupled_basis_matrix(paths, 0, 0)
    rng = np.random.default_rng(891)
    state = rng.normal(size=len(paths)) + 1.0j * rng.normal(size=len(paths))
    state /= np.linalg.norm(state)
    order = (2, 0, 5, 1, 4, 3)
    actual = permuted_fusion_state(state, order, target_q=0)
    expected = transform.conj().T @ permute_computational_state(transform @ state, order)
    np.testing.assert_allclose(actual, expected, atol=2.0e-13)

    bonds = square_j1j2_bonds(2, 3, j1=1.0, j2=0.37, periodic=False)
    original = heisenberg_graph_fusion_matrix(n_sites, bonds, target_q=0).toarray()
    action = site_permutation_fusion_matrix(
        order_to_active_image(order), target_q=0
    ).toarray()
    reordered = heisenberg_graph_fusion_matrix(
        n_sites, permute_bonds_to_leaf_order(bonds, order), target_q=0
    ).toarray()
    np.testing.assert_allclose(reordered, action @ original @ action.T, atol=2.0e-13)


def test_same_physical_prefixes_give_same_projector_and_full_mask_is_identity() -> None:
    n_sites = 8
    paths = enumerate_paths(n_sites, 0)
    retained = np.asarray(retained_path_indices(paths, {4: 2, 6: 2}))
    first = nested_prefix_order(n_sites, (0, 1, 4, 5), (0, 1, 2, 4, 5, 6))
    second = first[:4][::-1] + first[4:6][::-1] + first[6:][::-1]
    projectors = []
    for order in (first, second):
        action = site_permutation_fusion_matrix(
            order_to_active_image(order), target_q=0
        ).toarray()
        columns = action.T[:, retained]
        projectors.append(columns @ columns.T)
    np.testing.assert_allclose(projectors[0], projectors[1], atol=3.0e-13)

    full = np.asarray(retained_path_indices(paths, {4: 4, 6: 6}))
    assert len(full) == len(paths)
    action = site_permutation_fusion_matrix(
        order_to_active_image(first), target_q=0
    ).toarray()
    columns = action.T[:, full]
    np.testing.assert_allclose(columns @ columns.T, np.eye(len(paths)), atol=3.0e-13)


def test_ritz_metric_reports_instead_of_hiding_variational_violation() -> None:
    hamiltonian = np.diag([-2.0, -1.0, 0.5])
    ground = np.asarray([1.0, 0.0, 0.0])
    valid = exact_truncation_metrics(
        ground,
        hamiltonian,
        (0, 1),
        exact_energy=-2.0,
        exact_singlet_gap=1.0,
        exact_singlet_bandwidth=2.5,
        n_sites=2,
    )
    assert valid["ritz_energy_gap"] == pytest.approx(0.0)
    assert valid["variational_bound_violation"] == pytest.approx(0.0)
    assert valid["variational_bound_pass"] is True

    inconsistent = exact_truncation_metrics(
        ground,
        hamiltonian,
        (0, 1),
        exact_energy=-1.9,
        exact_singlet_gap=1.0,
        exact_singlet_bandwidth=2.5,
        n_sites=2,
    )
    assert inconsistent["ritz_energy_gap"] == pytest.approx(-0.1)
    assert inconsistent["ritz_energy_error"] == pytest.approx(0.0)
    assert inconsistent["variational_bound_violation"] == pytest.approx(0.1)
    assert inconsistent["variational_bound_pass"] is False


def test_aggregate_rejects_nominally_unmatched_subspaces() -> None:
    common = {
        "projection_fidelity": 0.9,
        "discarded_weight": 0.1,
        "ritz_energy_error": 0.2,
        "ritz_energy_error_per_site": 0.1,
        "ritz_energy_error_over_singlet_gap": 0.4,
        "ritz_energy_error_over_singlet_bandwidth": 0.05,
        "target_projected_energy_error": 0.3,
        "variational_bound_violation": 0.0,
    }
    with pytest.raises(ValueError, match="matched retained dimension"):
        aggregate_truncation_metrics(
            [
                {**common, "retained_dimension": 3},
                {**common, "retained_dimension": 4},
            ]
        )
