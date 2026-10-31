from __future__ import annotations

import itertools
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.vmc.cluster_observables import (
    cluster_state_projector_local_values,
    cluster_spin_sector_local_values,
    cluster_spin_squared_from_sector_values,
    prepare_cluster_spin_sectors,
    prepare_cluster_state_projector,
    prepare_spin_pair_observables,
    prepare_translated_cluster_bond_decomposition,
    prepare_translated_cluster_spin_sectors,
    spin_pair_observable_hpsi,
    spin_pair_observable_local_values,
    square_neel_structure_factor_spec,
    translated_rectangular_cluster_covers,
    triangular_120_structure_factor_spec,
)


def _bits(n_sites: int) -> np.ndarray:
    return np.asarray(
        [
            [(state >> (n_sites - 1 - site)) & 1 for site in range(n_sites)]
            for state in range(1 << n_sites)
        ],
        dtype=np.int32,
    )


def _table_amplitude(table: jax.Array, bits: jax.Array) -> jax.Array:
    weights = 1 << jnp.arange(bits.shape[0] - 1, -1, -1, dtype=jnp.int32)
    return table[jnp.sum(bits * weights)]


def _dense_pair_operator(
    n_sites: int,
    bonds: tuple[tuple[int, int, float], ...],
    *,
    constant: float = 0.0,
) -> np.ndarray:
    identity = np.eye(2, dtype=np.complex128)
    sx = 0.5 * np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = 0.5 * np.asarray([[0.0, 1.0j], [-1.0j, 0.0]], dtype=np.complex128)
    sz = np.asarray([[-0.5, 0.0], [0.0, 0.5]], dtype=np.complex128)
    dimension = 1 << n_sites
    result = float(constant) * np.eye(dimension, dtype=np.complex128)
    for first, second, coupling in bonds:
        for one_site in (sx, sy, sz):
            factors = [identity] * n_sites
            factors[first] = one_site
            factors[second] = one_site
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            result += coupling * term
    return result


def _projector_matrices(spec) -> np.ndarray:
    dimension = len(spec.local_basis_bits)
    result = np.zeros((len(spec.q_values), dimension, dimension), dtype=np.float64)
    connected = np.asarray(spec.connected_local_indices, dtype=np.int32)
    valid = np.asarray(spec.connected_valid, dtype=bool)
    rows = np.asarray(spec.projector_rows, dtype=np.float64)
    for state in range(dimension):
        result[:, state, connected[state, valid[state]]] = rows[:, state, valid[state]]
    return result


def test_all_translations_reduce_to_six_equal_weight_6x6_covers() -> None:
    covers = translated_rectangular_cluster_covers(6, 6)
    assert len(covers) == 6
    assert [item.translation_multiplicity for item in covers] == [6] * 6
    for cover in covers:
        assert len(cover.clusters) == 6
        assert all(len(cluster) == 6 for cluster in cover.clusters)
        assert sorted(itertools.chain.from_iterable(cover.clusters)) == list(range(36))

    sectors = prepare_translated_cluster_spin_sectors(6, 6)
    assert len(sectors.clusters) == 36
    incidence = np.bincount(
        np.asarray(tuple(itertools.chain.from_iterable(sectors.clusters))), minlength=36
    )
    np.testing.assert_array_equal(incidence, np.full(36, 6))


@pytest.mark.parametrize(
    ("geometry", "bonds"),
    (
        (
            "square",
            square_j1j2_bonds(6, 6, j1=1.0, j2=0.5, periodic=True),
        ),
        (
            "triangular",
            triangular_j1j2_bonds(6, 6, j1=1.0, j2=0.125, periodic=True),
        ),
    ),
)
def test_every_cover_and_translation_average_partition_total_hamiltonian(
    geometry: str, bonds: tuple[tuple[int, int, float], ...]
) -> None:
    del geometry
    spec = prepare_translated_cluster_bond_decomposition(6, 6, bonds)
    coefficients = np.asarray(spec.observables.coefficients)
    total = coefficients[spec.total_index]
    for intra, inter in spec.cover_mode_indices:
        np.testing.assert_array_equal(coefficients[intra] + coefficients[inter], total)
    np.testing.assert_allclose(
        coefficients[spec.average_intra_index]
        + coefficients[spec.average_inter_index],
        total,
        atol=2.0e-15,
        rtol=0.0,
    )
    assert sum(item.translation_multiplicity for item in spec.covers) == 36


def test_pair_observable_actions_match_independent_dense_complex_reference() -> None:
    n_sites = 4
    bonds = ((0, 1, 0.7), (1, 2, -0.2), (0, 3, 0.4), (0, 1, 0.3))
    second = ((0, 2, 2.0), (1, 3, -0.5))
    spec = prepare_spin_pair_observables(
        n_sites,
        (
            ("hamiltonian", 0.125, bonds),
            ("second", -0.25, second),
        ),
    )
    rng = np.random.default_rng(20260917)
    table = rng.normal(size=1 << n_sites) + 1.0j * rng.normal(size=1 << n_sites)
    table += 0.2 + 0.1j
    samples = _bits(n_sites)
    centers, actions = spin_pair_observable_hpsi(
        jnp.asarray(table),
        jnp.asarray(samples),
        spec,
        amplitude_fn=_table_amplitude,
    )
    np.testing.assert_allclose(centers, table, atol=2.0e-14, rtol=2.0e-14)
    dense_first = _dense_pair_operator(
        n_sites,
        ((0, 1, 1.0), (1, 2, -0.2), (0, 3, 0.4)),
        constant=0.125,
    )
    dense_second = _dense_pair_operator(n_sites, second, constant=-0.25)
    expected = np.stack((dense_first @ table, dense_second @ table), axis=1)
    np.testing.assert_allclose(actions, expected, atol=3.0e-13, rtol=3.0e-13)


def test_pair_local_values_are_jittable_and_fail_closed_at_a_node() -> None:
    samples = jnp.asarray(_bits(2))
    table = jnp.asarray([0.0, 1.0, -2.0, 3.0], dtype=jnp.float64)
    spec = prepare_spin_pair_observables(2, (("exchange", 0.0, ((0, 1, 1.0),)),))
    kernel = jax.jit(
        lambda candidate, states: spin_pair_observable_local_values(
            candidate, states, spec, amplitude_fn=_table_amplitude
        )
    )
    local = np.asarray(kernel(table, samples))
    assert np.isnan(local[0, 0])
    assert np.all(np.isfinite(local[1:, 0]))


def test_six_spin_projectors_have_exact_sector_ranks_and_resolve_identity() -> None:
    spec = prepare_cluster_spin_sectors(6, ((0, 1, 2, 3, 4, 5),))
    assert spec.q_values == (0, 2, 4, 6)
    projectors = _projector_matrices(spec)
    np.testing.assert_allclose(
        np.trace(projectors, axis1=1, axis2=2),
        (5.0, 27.0, 25.0, 7.0),
        atol=2.0e-12,
    )
    np.testing.assert_allclose(np.sum(projectors, axis=0), np.eye(64), atol=2.0e-12)
    for first in range(4):
        for second in range(4):
            target = projectors[first] if first == second else np.zeros((64, 64))
            np.testing.assert_allclose(
                projectors[first] @ projectors[second], target, atol=3.0e-12
            )


def test_sector_local_actions_and_s2_match_independent_pair_operator() -> None:
    n_sites = 6
    samples = _bits(n_sites)
    rng = np.random.default_rng(29)
    table = rng.normal(size=64) + 1.0j * rng.normal(size=64) + (0.3 - 0.2j)
    sector_spec = prepare_cluster_spin_sectors(n_sites, (tuple(range(6)),))
    local = cluster_spin_sector_local_values(
        jnp.asarray(table),
        jnp.asarray(samples),
        sector_spec,
        amplitude_fn=_table_amplitude,
    )
    projectors = _projector_matrices(sector_spec)
    expected_actions = np.stack(tuple(projector @ table for projector in projectors), axis=1)
    np.testing.assert_allclose(
        np.asarray(local[:, 0, :]) * table[:, None],
        expected_actions,
        atol=8.0e-13,
        rtol=8.0e-13,
    )
    np.testing.assert_allclose(np.sum(local[:, 0, :], axis=1), 1.0, atol=8.0e-13)

    all_pairs = tuple(
        (first, second, 2.0)
        for first in range(n_sites)
        for second in range(first + 1, n_sites)
    )
    pair_spec = prepare_spin_pair_observables(
        n_sites, (("cluster_s2", 0.75 * n_sites, all_pairs),)
    )
    pair_local = spin_pair_observable_local_values(
        jnp.asarray(table),
        jnp.asarray(samples),
        pair_spec,
        amplitude_fn=_table_amplitude,
    )[:, 0]
    sector_s2 = cluster_spin_squared_from_sector_values(local[:, 0, :], sector_spec.q_values)
    np.testing.assert_allclose(sector_s2, pair_local, atol=1.2e-12, rtol=1.2e-12)


def test_three_dimer_product_is_entirely_in_six_spin_singlet_sector() -> None:
    singlet = np.asarray([0.0, 1.0, -1.0, 0.0], dtype=np.float64) / np.sqrt(2.0)
    state = np.kron(np.kron(singlet, singlet), singlet)
    spec = prepare_cluster_spin_sectors(6, (tuple(range(6)),))
    projectors = _projector_matrices(spec)
    np.testing.assert_allclose(projectors[0] @ state, state, atol=2.0e-12)
    for projector in projectors[1:]:
        np.testing.assert_allclose(projector @ state, 0.0, atol=2.0e-12)


def test_cluster_state_projector_matches_independent_dense_complex_action() -> None:
    n_sites = 4
    cluster = (0, 2)
    local_state = np.asarray([0.0, 1.0, -1.0, 0.0], dtype=np.complex128)
    local_state /= np.linalg.norm(local_state)
    spec = prepare_cluster_state_projector(n_sites, (cluster,), local_state)
    rng = np.random.default_rng(91)
    table = rng.normal(size=16) + 1.0j * rng.normal(size=16) + 0.4
    samples = _bits(n_sites)
    local = np.asarray(
        cluster_state_projector_local_values(
            jnp.asarray(table),
            jnp.asarray(samples),
            spec,
            amplitude_fn=_table_amplitude,
        )[:, 0]
    )

    dense = np.zeros((16, 16), dtype=np.complex128)
    for row_index, row in enumerate(samples):
        x = 2 * row[cluster[0]] + row[cluster[1]]
        for column_index, column in enumerate(samples):
            if any(
                row[site] != column[site]
                for site in range(n_sites)
                if site not in cluster
            ):
                continue
            y = 2 * column[cluster[0]] + column[cluster[1]]
            dense[row_index, column_index] = (
                local_state[x] * np.conj(local_state[y])
            )
    np.testing.assert_allclose(local * table, dense @ table, atol=4.0e-13)
    expected = float(np.vdot(table, dense @ table).real / np.vdot(table, table).real)
    born = np.abs(table) ** 2 / np.vdot(table, table).real
    assert float(np.sum(born * local).real) == pytest.approx(expected, abs=3.0e-13)


def test_cluster_state_projector_validates_normalization_and_support() -> None:
    with pytest.raises(ValueError, match="normalized"):
        prepare_cluster_state_projector(2, ((0, 1),), (1.0, 1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="dimension"):
        prepare_cluster_state_projector(2, ((0, 1),), (1.0, 0.0))


def test_registered_ordering_vectors_follow_square_and_triangular_conventions() -> None:
    square = square_neel_structure_factor_spec(6, 6)
    square_weights = np.asarray(square.phase_weights)
    expected = np.asarray(
        [(-1.0) ** (row + column) for row in range(6) for column in range(6)]
    )
    np.testing.assert_allclose(square_weights, expected, atol=2.0e-15)
    assert square.normalization == 36.0

    triangular = triangular_120_structure_factor_spec(6, 6)
    pair_coefficients = {
        (first, second): value
        for first, second, value in zip(
            triangular.pair_first,
            triangular.pair_second,
            triangular.pair_coefficients,
            strict=True,
        )
    }
    origin = 0
    for neighbor in (6, 1, 7):
        assert pair_coefficients[tuple(sorted((origin, neighbor)))] == pytest.approx(
            -1.0, abs=2.0e-15
        )
    assert triangular.normalization == 36.0
    assert triangular.onsite_coefficient == pytest.approx(27.0)


def test_cluster_observable_boundaries_fail_closed() -> None:
    with pytest.raises(ValueError, match="tile"):
        translated_rectangular_cluster_covers(5, 6)
    with pytest.raises(ValueError, match="unique"):
        prepare_spin_pair_observables(
            2,
            (("same", 0.0, ((0, 1),)), ("same", 0.0, ((0, 1),))),
        )
    with pytest.raises(ValueError, match="same size"):
        prepare_cluster_spin_sectors(6, ((0, 1), (2, 3, 4)))
    with pytest.raises(ValueError, match="duplicate"):
        prepare_cluster_spin_sectors(6, ((0, 1, 2), (0, 1, 2)))
