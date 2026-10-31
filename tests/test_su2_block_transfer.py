"""Independent physical-spin oracles for offline SU(2) block initialization."""
from itertools import product
from dataclasses import replace

import numpy as np
import pytest

from su2_nqs.operators.su2_block_transfer import (
    BlockIsometry, canonicalize_block, merge_projected_blocks, project_block,
)
from su2_nqs.su2.clebsch_gordan import clebsch_gordan


def _operators(n):
    one = (np.array([[0, 1], [1, 0]], complex) / 2,
           np.array([[0, 1j], [-1j, 0]], complex) / 2,
           np.diag([-.5, .5]).astype(complex))
    result = []
    for site in range(n):
        row = []
        for component in one:
            value = np.ones((1, 1), complex)
            for i in range(n):
                value = np.kron(value, component if i == site else np.eye(2))
            row.append(value)
        result.append(tuple(row))
    return tuple(result)


def _hamiltonian(n, bonds):
    spin = _operators(n)
    return sum((strength * sum((spin[i][a] @ spin[j][a] for a in range(3)),
                               np.zeros((2**n, 2**n), complex)) for i, j, strength in bonds),
               np.zeros((2**n, 2**n), complex))


def _triangle():
    sectors = {1: np.zeros((8, 2, 2), complex), 3: np.zeros((8, 1, 4), complex)}
    for q, values in sectors.items():
        for a, q12 in enumerate((0, 2) if q == 1 else (2,)):
            for bits in product((0, 1), repeat=3):
                row = 4 * bits[0] + 2 * bits[1] + bits[2]
                m1, m2, m3 = (2 * b - 1 for b in bits)
                m = m1 + m2 + m3
                if abs(m) > q or abs(m1 + m2) > q12:
                    continue
                values[row, a, (m + q) // 2] = (
                    clebsch_gordan(1, m1, 1, m2, q12, m1 + m2)
                    * clebsch_gordan(q12, m1 + m2, 1, m3, q, m))
    return BlockIsometry(3, sectors)


def _children():
    triangle = _triangle()
    left_bonds = [(0, 1, 1.), (1, 2, .8), (0, 2, .6)]
    right_bonds = [(0, 1, .7), (1, 2, .9), (0, 2, 1.1)]
    cross = [(0, 0, .125), (0, 2, 1.), (1, 0, 1.), (1, 1, .125), (2, 1, .5)]
    left = project_block(triangle, _hamiltonian(3, left_bonds))
    right = project_block(triangle, _hamiltonian(3, right_bonds))
    physical = _hamiltonian(6, left_bonds + [(i + 3, j + 3, g) for i, j, g in right_bonds]
                            + [(i, j + 3, g) for i, j, g in cross])
    return left, right, cross, physical


def test_reduced_qr_uses_one_gauge_for_all_m_and_preserves_physical_maps():
    original = _triangle()
    transforms = {1: np.array([[2., .3j], [.4, -.8j]]), 3: np.array([[1.7j]])}
    raw = {q: np.einsum('pak,ab->pbk', u, transforms[q]) for q, u in original.sectors.items()}
    canonical, gauges, diagnostics = canonicalize_block(3, raw)
    assert diagnostics['gram_error'] < 3e-14
    assert diagnostics['ladder_error'] < 3e-14
    assert diagnostics['replay_error'] < 3e-14
    for q, u in canonical.sectors.items():
        np.testing.assert_allclose(np.einsum('pak,ab->pbk', u, gauges[q]), raw[q], atol=3e-14)
        np.testing.assert_allclose(np.diag(gauges[q]).imag, 0., atol=3e-14)
        assert np.all(np.diag(gauges[q]).real > 0)
        np.testing.assert_allclose(u[:, :, -1] @ u[:, :, -1].conj().T,
            original.sectors[q][:, :, -1] @ original.sectors[q][:, :, -1].conj().T, atol=3e-14)


@pytest.mark.parametrize('scale', [1e-40, 1., 1e40])
def test_qr_rank_and_su2_validation_are_invariant_to_uniform_gauge_scale(scale):
    original = _triangle()
    raw = {q: scale * u for q, u in original.sectors.items()}
    block, gauges, _ = canonicalize_block(3, raw)
    for q, u in block.sectors.items():
        np.testing.assert_allclose(u, original.sectors[q], atol=3e-14)
        np.testing.assert_allclose(gauges[q] / scale, np.eye(u.shape[1]), atol=3e-14)
    raw[1][:, :, 0] *= -1
    with pytest.raises(ValueError, match='Condon-Shortley'):
        canonicalize_block(3, raw)


def test_two_complete_triangles_reproduce_full_six_spin_H_and_spin_casimir():
    left, right, cross, physical = _children()
    result = merge_projected_blocks(left, right, cross, retain={0: 5, 2: 9, 4: 5, 6: 1})
    basis = result.block.basis
    np.testing.assert_allclose(basis.conj().T @ basis, np.eye(64), atol=3e-14)
    np.testing.assert_allclose(result.block.hamiltonian, basis.conj().T @ physical @ basis, atol=3e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(result.block.hamiltonian),
                               np.linalg.eigvalsh(physical), atol=3e-14)
    assert result.diagnostics['scalar_error'] < 3e-14
    spin = _operators(6)
    totals = [sum((spin[i][a] for i in range(6)), np.zeros((64, 64), complex)) for a in range(3)]
    casimir = sum((s @ s for s in totals), np.zeros((64, 64), complex))
    for q, a, m in result.block.labels:
        column = result.block.isometry.sectors[q][:, a, (m + q) // 2]
        np.testing.assert_allclose(casimir @ column, q * (q + 2) / 4 * column, atol=3e-14)
        np.testing.assert_allclose(totals[2] @ column, m / 2 * column, atol=3e-14)
    for i in range(6):
        for a in range(3):
            np.testing.assert_allclose(result.block.site_spin[i][a],
                                       basis.conj().T @ spin[i][a] @ basis, atol=3e-14)


def test_truncated_merge_keeps_q_positive_and_covariance_not_false_onsite_algebra():
    left, right, cross, physical = _children()
    result = merge_projected_blocks(left, right, cross, retain={0: 1, 2: 1, 4: 1})
    state = result.block
    assert state.isometry.sectors.keys() == {0, 2, 4}
    assert state.basis.shape == (64, 9)
    np.testing.assert_allclose(state.hamiltonian, state.basis.conj().T @ physical @ state.basis, atol=3e-14)
    spin = state.site_spin
    total = [sum((spin[i][a] for i in range(6)), np.zeros((9, 9), complex)) for a in range(3)]
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        np.testing.assert_allclose(total[a] @ total[b] - total[b] @ total[a], 1j * total[c], atol=3e-14)
        for i in range(6):
            np.testing.assert_allclose(total[a] @ spin[i][b] - spin[i][b] @ total[a],
                                       1j * spin[i][c], atol=3e-14)
    difference = state.basis.conj().T @ physical @ physical @ state.basis - state.hamiltonian @ state.hamiltonian
    assert np.linalg.eigvalsh(difference).min() > -3e-14
    assert np.linalg.norm(difference) < 3e-13  # complete child space: retained vectors are exact H eigenmultiplets
    # Incomplete child multiplets leave physical H leakage even after diagonalizing their projected H.
    triangle = _triangle()
    incomplete = BlockIsometry(3, {q: u[:, :1] for q, u in triangle.sectors.items()})
    truncated = project_block(incomplete, _hamiltonian(3, [(0, 1, 1.), (1, 2, .8), (0, 2, .6)]))
    assert truncated.diagnostics['physical_H2_minus_projected_H_squared_norm'] > 1e-4
    projected_result = merge_projected_blocks(truncated, right, cross, retain={0: 1, 2: 1, 4: 1})
    v, h = projected_result.block.basis, projected_result.block.hamiltonian
    np.testing.assert_allclose(h, v.conj().T @ physical @ v, atol=3e-14)
    assert np.linalg.norm(v.conj().T @ physical @ physical @ v - h @ h) > 1e-3
    assert any('H^2' in warning for warning in projected_result.warnings)


def test_degenerate_projectors_and_phase_are_deterministic():
    triangle = project_block(_triangle(), np.zeros((8, 8)))
    first = merge_projected_blocks(triangle, triangle, [], retain={0: 5, 2: 2})
    second = merge_projected_blocks(triangle, triangle, [], retain={0: 5, 2: 2})
    for q in first.reduced_maps:
        np.testing.assert_array_equal(first.reduced_maps[q], second.reduced_maps[q])
        np.testing.assert_allclose(first.energy_projectors[q][0], np.eye(len(first.channels[q])), atol=3e-14)
        np.testing.assert_allclose(first.retained_projectors[q] @ first.retained_projectors[q],
                                   first.retained_projectors[q], atol=3e-14)
        for column in first.reduced_maps[q].T:
            pivot = column[np.argmax(abs(column))]
            assert abs(pivot.imag) < 3e-14 and pivot.real > 0
    assert any('cuts a degenerate' in warning for warning in first.warnings)


def test_reference_bounds_bad_labels_and_magnetic_splitting_are_rejected():
    triangle = _triangle()
    with pytest.raises(ValueError, match='doubled'):
        canonicalize_block(3, {1.5: triangle.sectors[1]})
    bad = {q: u.copy() for q, u in triangle.sectors.items()}
    bad[1][:, :, 0] *= -1
    with pytest.raises(ValueError, match='Condon-Shortley'):
        canonicalize_block(3, bad)
    deficient = {1: np.repeat(triangle.sectors[1][:, :1], 2, axis=1)}
    with pytest.raises(ValueError, match='full column rank'):
        canonicalize_block(3, deficient)
    with pytest.raises(ValueError, match=r'not SU\(2\) scalar'):
        project_block(triangle, _operators(3)[0][2])
    left, right, cross, _ = _children()
    with pytest.raises(ValueError, match='limit 512'):
        merge_projected_blocks(left, right, cross, retain={0: 1}, max_effective_dimension=32)
    with pytest.raises(ValueError, match='unavailable'):
        merge_projected_blocks(left, right, cross, retain={1: 1})
    with pytest.raises(ValueError, match='exceeds the available'):
        merge_projected_blocks(left, right, cross, retain={0: 6})
    with pytest.raises(ValueError, match='cross bonds'):
        merge_projected_blocks(left, right, [(3, 0, 1.)], retain={0: 1})


def test_merged_boundary_operators_compose_without_dense_twelve_site_H():
    left, right, cross, _ = _children()
    block = merge_projected_blocks(left, right, cross, retain={0: 1, 2: 1}).block
    result = merge_projected_blocks(block, block, [(2, 0, .3)], retain={0: 1, 2: 1})
    assert result.block.basis.shape == (4096, 4)
    assert result.diagnostics['ladder_error'] < 3e-14
    assert result.diagnostics['gram_error'] < 3e-14
    assert result.diagnostics['scalar_error'] < 3e-14


def test_public_projected_block_rejects_stale_basis_labels_and_boundary_operators():
    left, right, cross, _ = _children()
    with pytest.raises(ValueError, match='basis/labels'):
        merge_projected_blocks(replace(left, labels=left.labels[::-1]), right, cross, retain={0: 1})
    with pytest.raises(ValueError, match='basis/labels'):
        merge_projected_blocks(replace(left, basis=-left.basis), right, cross, retain={0: 1})
    with pytest.raises(ValueError, match='projected H'):
        merge_projected_blocks(replace(left, hamiltonian=left.hamiltonian[:-1]), right, cross, retain={0: 1})
    swapped = (left.site_spin[1], left.site_spin[0], left.site_spin[2])
    # Swapping two sites preserves total spin and SU(2) covariance, but changes
    # the physical boundary Hamiltonian. Shape/covariance checks alone miss it.
    with pytest.raises(ValueError, match='physical-site projection'):
        merge_projected_blocks(replace(left, site_spin=swapped), right, cross, retain={0: 1})


def test_singlet_projection_cannot_hide_nonscalar_H_or_external_variance():
    singlet = np.zeros((4, 1, 1), complex)
    for a, b in product((0, 1), repeat=2):
        singlet[2*a+b, 0, 0] = clebsch_gordan(1, 2*a-1, 1, 2*b-1, 0, 0)
    block = BlockIsometry(2, {0: singlet})
    with pytest.raises(ValueError, match='before projection'):
        project_block(block, _operators(2)[0][2])
    child = project_block(block, np.zeros((4, 4)))
    result = merge_projected_blocks(child, child, [(1, 0, 1.)], retain={0: 1})
    h = _hamiltonian(4, [(1, 2, 1.)])
    psi = result.block.basis[:, 0]
    energy = np.vdot(psi, h @ psi)
    variance = np.vdot(h @ psi, h @ psi) - abs(energy)**2
    assert result.diagnostics['omitted_child_product_H_norm'] < 1e-14
    assert np.linalg.norm(result.block.hamiltonian) < 1e-14
    np.testing.assert_allclose(variance, 3/16, atol=3e-14)
    assert any('discarded-space' in warning for warning in result.warnings)


@pytest.mark.parametrize('tolerance', [np.nan, np.inf, -1., 0., True])
def test_nonfinite_or_nonpositive_validation_tolerance_is_rejected(tolerance):
    triangle = _triangle()
    with pytest.raises(ValueError, match='tolerance'):
        canonicalize_block(3, triangle.sectors, tolerance=tolerance)
    with pytest.raises(ValueError, match='tolerance'):
        project_block(triangle, np.zeros((8, 8)), tolerance=tolerance)
    child = project_block(triangle, np.zeros((8, 8)))
    with pytest.raises(ValueError, match='tolerance'):
        merge_projected_blocks(child, child, [], retain={0: 1}, tolerance=tolerance)


def test_dense_physical_limit_rejects_before_multiplet_gram_allocation(monkeypatch):
    import su2_nqs.operators.su2_block_transfer as module
    def allocation_is_forbidden(*args, **kwargs):
        raise AssertionError('must reject dimensions before multiplet validation allocates a Gram matrix')
    monkeypatch.setattr(module, '_validate_multiplets', allocation_is_forbidden)
    with pytest.raises(ValueError, match='dimension 1024'):
        project_block(BlockIsometry(12, {}), np.zeros((1, 1)))
