from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.physical_interpretation import (
    FixedSzPairActions,
    error_channel_tomography,
    fixed_sector_to_full_state,
    hnf_oriented_bond_families,
    hnf_oriented_triangles,
    local_rdm_error_profile,
    triangular_cluster_supports,
)
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.operators.sz_graph_sparse import heisenberg_sz_graph_sparse
from su2_nqs.su2.sz_basis import enumerate_sz_configs


def test_cached_pair_actions_reconstruct_periodic_triangular_hamiltonian() -> None:
    supercell = ((4, 0), (0, 2))
    n_sites = 8
    bonds = triangular_hnf_j1j2_bonds(supercell, j1=1.0, j2=0.37)
    hamiltonian, basis = heisenberg_sz_graph_sparse(n_sites, bonds, n_up=4)
    rng = np.random.default_rng(912)
    state = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))
    state /= np.linalg.norm(state)
    actions = FixedSzPairActions(basis)
    actual = actions.weighted_bonds(state, bonds)
    np.testing.assert_allclose(actual, hamiltonian @ state, atol=3.0e-14)


def test_registered_hnf_support_counts_and_closure() -> None:
    supercell = ((4, 0), (0, 4))
    families = hnf_oriented_bond_families(supercell)
    assert tuple(len(family) for family in families) == (16, 16, 16)
    triangles = hnf_oriented_triangles(supercell)
    assert len(triangles) == 32
    assert len({tuple(sorted(value)) for value in triangles}) == 32

    supports = triangular_cluster_supports(supercell)
    assert {name: row[0].__len__() for name, row in supports.items()} == {
        "bond_2": 2,
        "triangle_3": 3,
        "rhombus_4": 4,
        "block_2x3_6": 6,
        "half_torus_8": 8,
    }
    assert {name: len(rows) for name, rows in supports.items()} == {
        "bond_2": 48,
        "triangle_3": 32,
        "rhombus_4": 48,
        "block_2x3_6": 96,
        "half_torus_8": 12,
    }


def test_sector_embedding_and_local_rdm_profile_are_exact_on_self() -> None:
    n_sites = 6
    basis = tuple(enumerate_sz_configs(n_sites, n_sites // 2))
    rng = np.random.default_rng(41)
    state = rng.normal(size=len(basis))
    state /= np.linalg.norm(state)
    full = fixed_sector_to_full_state(state, basis)
    assert np.linalg.norm(full) == pytest.approx(1.0, abs=2.0e-14)
    profile = local_rdm_error_profile(
        full,
        full,
        n_sites,
        {"pair": ((0, 1), (2, 3)), "half": ((0, 1, 2),)},
    )
    assert profile["pair"]["trace_distance_max"] < 2.0e-14
    assert profile["half"]["entropy_error_rms"] < 2.0e-14


def test_error_channel_tomography_recovers_a_known_exact_direction() -> None:
    state = np.asarray([1.0, 0.0, 0.0], dtype=np.complex128)
    exact = np.asarray([0.8, 0.6, 0.0], dtype=np.complex128)
    exact /= np.linalg.norm(exact)
    hamiltonian = -np.outer(exact, exact.conj())
    channels = {
        "correct": np.asarray([[0.0], [1.0], [0.0]], dtype=np.complex128),
        "orthogonal": np.asarray([[0.0], [0.0], [1.0]], dtype=np.complex128),
    }
    result = error_channel_tomography(state, exact, hamiltonian, channels)
    assert result["channels"]["correct"]["exact_error_fraction"] == pytest.approx(
        1.0, abs=2.0e-14
    )
    assert result["channels"]["correct"]["energy_error_recovery"] == pytest.approx(
        1.0, abs=2.0e-14
    )
    assert result["channels"]["orthogonal"]["exact_error_fraction"] < 2.0e-14
