"""Exact checks for physical nested-subregion total-spin FCS."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from su2_nqs.benchmarks.block_spin_fcs import (
    block_spin_fcs,
    subsystem_spin_eigenbasis,
)
from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.nested_block_spin import (
    compact_connected_subsets,
    endpoint_chord_screening,
    marginal_spin_distribution,
    nested_compact_chains,
    nested_distribution_summary,
    nested_spin_fcs,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.su2.fusion_path import enumerate_paths


def _single_group_distribution(
    wavefunction: np.ndarray, subset: tuple[int, ...]
) -> dict[int, float]:
    return {
        int(key[0]): float(probability)
        for key, probability in block_spin_fcs(wavefunction, (subset,)).items()
    }


def _apply_spin_projector(
    wavefunction: np.ndarray, subset: tuple[int, ...], doubled_spin: int
) -> np.ndarray:
    """Independent computational-basis action of one subsystem projector."""

    n_sites = int(round(np.log2(len(wavefunction))))
    remainder = tuple(site for site in range(n_sites) if site not in subset)
    order = subset + remainder
    inverse = tuple(int(value) for value in np.argsort(order))
    tensor = wavefunction.reshape((2,) * n_sites).transpose(order)
    matrix = tensor.reshape(2 ** len(subset), -1)
    labels, eigenvectors = subsystem_spin_eigenbasis(len(subset))
    selected = eigenvectors[:, labels == doubled_spin]
    projected = selected @ (selected.conj().T @ matrix)
    return (
        projected.reshape((2,) * n_sites)
        .transpose(inverse)
        .reshape(-1)
    )


@pytest.mark.parametrize(
    ("builder", "family_counts", "internal_edges", "chain_count"),
    [
        (square_j1j2_bonds, (17, 6, 7), (1, 4, 7), 56),
        (triangular_j1j2_bonds, (23, 13, 26), (1, 5, 9), 390),
    ],
)
def test_three_by_four_compact_atlas_is_geometry_determined(
    builder, family_counts, internal_edges, chain_count
) -> None:
    bonds = builder(3, 4, j1=1.0, j2=0.0, periodic=False)
    families = {
        scale: compact_connected_subsets(12, bonds, scale)
        for scale in (2, 4, 6)
    }
    assert tuple(len(families[scale]["subsets"]) for scale in families) == (
        family_counts
    )
    assert tuple(
        families[scale]["maximum_internal_edges"] for scale in families
    ) == internal_edges
    chains = nested_compact_chains(
        {scale: families[scale]["subsets"] for scale in families},
        scales=(2, 4, 6),
    )
    assert len(chains) == chain_count
    assert all(
        set(chain[0]) < set(chain[1]) < set(chain[2]) for chain in chains
    )


def test_nested_fcs_of_sequential_pair_singlets_is_deterministic() -> None:
    paths = enumerate_paths(6, 0)
    transform = coupled_basis_matrix(paths, 0, 0)
    path = (1, 0, 1, 0, 1, 0)
    full = transform[:, paths.index(path)]
    chain = ((0, 1), (0, 1, 2, 3), tuple(range(6)))
    direct = nested_spin_fcs(full, chain, coupled_basis=transform, paths=paths)
    reversed_result = nested_spin_fcs(
        full,
        chain,
        coupled_basis=transform,
        paths=paths,
        reverse_shells=True,
    )
    assert direct[(0, 0, 0)] == pytest.approx(1.0, abs=2.0e-14)
    assert sum(value for key, value in direct.items() if key != (0, 0, 0)) < 2.0e-14
    assert direct == pytest.approx(reversed_result, abs=2.0e-14)


def test_nested_fcs_matches_independent_projectors_and_complement() -> None:
    rng = np.random.default_rng(834)
    paths = enumerate_paths(8, 0)
    transform = coupled_basis_matrix(paths, 0, 0)
    amplitudes = rng.normal(size=len(paths)) + 1.0j * rng.normal(size=len(paths))
    amplitudes /= np.linalg.norm(amplitudes)
    full = transform @ amplitudes
    chain = ((0, 3), (0, 2, 3, 6), (0, 1, 2, 3, 6, 7))
    joint = nested_spin_fcs(full, chain, coupled_basis=transform, paths=paths)
    reversed_joint = nested_spin_fcs(
        full,
        chain,
        coupled_basis=transform,
        paths=paths,
        reverse_shells=True,
    )
    assert sum(joint.values()) == pytest.approx(1.0, abs=2.0e-14)
    assert joint == pytest.approx(reversed_joint, abs=2.0e-12)
    independent_joint = {}
    for key in product(*(sorted(set(path[len(subset) - 1] for path in paths)) for subset in chain)):
        projected = full
        for subset, doubled_spin in zip(chain, key, strict=True):
            projected = _apply_spin_projector(projected, subset, doubled_spin)
        independent_joint[key] = float(np.vdot(projected, projected).real)
    keys = sorted(joint.keys() | independent_joint.keys())
    np.testing.assert_allclose(
        [joint.get(key, 0.0) for key in keys],
        [independent_joint.get(key, 0.0) for key in keys],
        atol=2.0e-12,
    )
    for index, subset in enumerate(chain):
        expected = _single_group_distribution(full, subset)
        actual = marginal_spin_distribution(joint, index)
        keys = sorted(expected.keys() | actual.keys())
        np.testing.assert_allclose(
            [actual.get(key, 0.0) for key in keys],
            [expected.get(key, 0.0) for key in keys],
            atol=2.0e-12,
        )
    complement = tuple(sorted(set(range(8)) - set(chain[-1])))
    assert marginal_spin_distribution(joint, 2) == pytest.approx(
        _single_group_distribution(full, complement), abs=2.0e-12
    )


def test_endpoint_chord_screening_has_registered_sign_convention() -> None:
    result = endpoint_chord_screening(
        {0: 0.2, 2: 0.8},
        {0: 0.5, 2: 0.5},
        {0: 0.4, 2: 0.6},
        interpolation_weight=0.5,
    )
    assert result["singlet_excess"] == pytest.approx(0.2)
    assert result["casimir_suppression"] == pytest.approx(0.4)
    assert result["relative_casimir_suppression"] == pytest.approx(2.0 / 7.0)
    assert result["minimum_tail_suppression"] == pytest.approx(0.2)


def test_adjacent_sector_information_uses_pair_marginals() -> None:
    independent = {
        tuple(2 * bit for bit in outcome): 1.0 / 8.0
        for outcome in product((0, 1), repeat=3)
    }
    summary = nested_distribution_summary(independent)
    assert summary["joint_spin_sector_entropy"] == pytest.approx(np.log(8.0))
    assert summary["adjacent_sector_mutual_information"] == pytest.approx(
        [0.0, 0.0], abs=2.0e-14
    )
    assert summary["conditional_nonzero_to_singlet_healing"] == pytest.approx(
        [0.5, 0.5]
    )
