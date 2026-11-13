"""Independent dense small-spin oracles for target-q matrix-free merging."""
from dataclasses import replace

import numpy as np
import pytest

from su2_nqs.operators.su2_block_transfer import BlockIsometry, project_block
from su2_nqs.operators.su2_sector_merge import (
    ReducedBlockData, SectorMergeOperator, reduced_block_from_arrays,
    reduced_block_from_projected, solve_sector_merge, validate_reduced_block,
    compact_reduced_block, project_sector_maps,
)
from su2_nqs.su2.triangle_blocks import triangle_block_tables


def _spin(n):
    one = (np.array([[0, 1], [1, 0]], complex)/2,
           np.array([[0, 1j], [-1j, 0]], complex)/2,
           np.diag([-.5, .5]).astype(complex))
    result = []
    for i in range(n):
        row = []
        for s in one:
            a = np.ones((1, 1), complex)
            for j in range(n):
                a = np.kron(a, s if i == j else np.eye(2))
            row.append(a)
        result.append(tuple(row))
    return tuple(result)


def _hamiltonian(n, bonds):
    s = _spin(n)
    return sum((g*sum((s[i][a] @ s[j][a] for a in range(3)),
                       np.zeros((2**n, 2**n), complex)) for i, j, g in bonds),
               np.zeros((2**n, 2**n), complex))


def _children(complex_gauge=False):
    tables = triangle_block_tables()['three']
    mags = np.asarray([2*i.bit_count()-3 for i in range(8)])
    u = {q: np.stack([table*(mags == m)[:, None] for m in range(-q, q+1, 2)], axis=-1)
         for q, table in tables.items()}
    if complex_gauge:
        rotation = np.array([[1, 1j], [1j, 1]])/np.sqrt(2)
        u[1] = np.einsum('pak,ab->pbk', u[1], rotation)
    iso = BlockIsometry(3, u)
    lb = [(0, 1, 1.), (1, 2, .7), (0, 2, .4)]
    rb = [(0, 1, .8), (1, 2, 1.2), (0, 2, .6)]
    cross = [(0, 0, .125), (0, 2, .9), (1, 0, 1.), (2, 1, .45)]
    left = project_block(iso, _hamiltonian(3, lb))
    right = project_block(iso, _hamiltonian(3, rb))
    full = _hamiltonian(6, lb+[(i+3, j+3, g) for i, j, g in rb]
                        + [(i, j+3, g) for i, j, g in cross])
    return left, right, cross, full


@pytest.mark.parametrize('complex_gauge', [False, True])
def test_sparse_target_q_action_matches_independent_physical_H_all_m(complex_gauge):
    a, b, bonds, physical = _children(complex_gauge)
    left, right = map(reduced_block_from_projected, (a, b))
    product = np.kron(a.basis, b.basis)
    expected = product.conj().T @ physical @ product
    rng = np.random.default_rng(261024)
    for q in (0, 2, 4, 6):
        op = SectorMergeOperator(left, right, bonds, q)
        vector = rng.normal(size=op.dimension)+1j*rng.normal(size=op.dimension)
        reference = op.matvec(vector)
        for m in range(-q, q+1, 2):
            x = op.expand(vector, m)
            np.testing.assert_allclose(op.project(x, m), vector, atol=2e-13)
            np.testing.assert_allclose(op.product_action(x).ravel(), expected @ x.ravel(), atol=2e-13)
            np.testing.assert_allclose(op.project(op.product_action(x), m), reference, atol=2e-13)
        dense = op.dense_reference()
        np.testing.assert_allclose(dense, dense.conj().T, atol=2e-13)


def _physical_merged_basis(result, left, right, cross):
    product = np.kron(left.basis, right.basis)
    columns = []
    for q, a, m in result.block.labels:
        op = SectorMergeOperator(reduced_block_from_projected(left),
                                 reduced_block_from_projected(right), cross, q)
        columns.append(product @ op.expand(result.reduced_maps[q][:, a], m).ravel())
    return np.column_stack(columns)


@pytest.mark.parametrize('complex_gauge', [False, True])
def test_eigenmultiplets_and_streamed_boundary_operators_are_exact(complex_gauge):
    left, right, bonds, physical = _children(complex_gauge)
    result = solve_sector_merge(reduced_block_from_projected(left), reduced_block_from_projected(right),
                                bonds, retain={0: 2, 2: 2, 4: 1}, tol=1e-12)
    basis = _physical_merged_basis(result, left, right, bonds)
    np.testing.assert_allclose(basis.conj().T @ basis, np.eye(basis.shape[1]), atol=3e-13)
    projected = basis.conj().T @ physical @ basis
    positions = {label: i for i, label in enumerate(result.block.labels)}
    for q, energies in result.energies.items():
        for a, energy in enumerate(energies):
            for m in range(-q, q+1, 2):
                psi = basis[:, positions[q, a, m]]
                np.testing.assert_allclose(physical @ psi, energy*psi, atol=3e-13)
    for site, spins in zip(result.block.site_spin, _spin(6), strict=True):
        for observed, reference in zip(site, spins, strict=True):
            np.testing.assert_allclose(observed, basis.conj().T @ reference @ basis, atol=3e-13)
    assert validate_reduced_block(result.block)['vector_covariance_error'] < 3e-13
    assert max(np.max(x) for x in result.eigen_residuals.values()) < 3e-13
    # A second merge uses only reduced data, not the 12-spin computational basis.
    second = solve_sector_merge(result.block, result.block, [(2, 0, .2)],
                                 retain={0: 1, 2: 1}, tol=1e-11)
    assert second.block.n_sites == 12
    assert second.block.dimension == 4
    assert not second.diagnostics['physical_basis_constructed']


@pytest.mark.parametrize('complex_gauge', [False, True])
def test_forced_matrix_free_arpack_matches_dense_sector_eigenvalues(complex_gauge):
    a, b, cross, _ = _children(complex_gauge)
    l, r = map(reduced_block_from_projected, (a, b))
    result = solve_sector_merge(l, r, cross, retain={0: 1, 2: 2},
                                dense_reference_limit=0, tol=1e-12, maxiter=50)
    for q, values in result.energies.items():
        reference = np.linalg.eigvalsh(SectorMergeOperator(l, r, cross, q).dense_reference())
        np.testing.assert_allclose(values, reference[:len(values)], atol=2e-12)
        assert result.diagnostics['sectors'][q]['solver'] == 'matrix_free_eigsh'
    assert not result.diagnostics['dense_product_H_constructed']
    assert result.diagnostics['estimated_total_peak_bytes'] < 512*1024**2
    assert result.diagnostics['combined_embedding_reserve_bytes'] > 0


def test_complex_arpack_retention_boundary_without_extra_eigenpair_is_flagged():
    a, b, cross, _ = _children(True)
    l, r = map(reduced_block_from_projected, (a, b))
    # q=0 dimension is 5: complex ARPACK permits k <= 3, not k=4.
    result = solve_sector_merge(l, r, cross, retain={0: 3},
                                dense_reference_limit=0, tol=1e-12, maxiter=50)
    expected = np.linalg.eigvalsh(SectorMergeOperator(l, r, cross, 0).dense_reference())
    np.testing.assert_allclose(result.energies[0], expected[:3], atol=2e-12)
    assert any('degeneracy is unverified' in w for w in result.warnings)


@pytest.mark.parametrize('complex_gauge', [False, True])
def test_optional_krylov_dimension_keeps_eigenvalues_and_accounts_for_memory(complex_gauge):
    a, b, cross, _ = _children(complex_gauge)
    l, r = map(reduced_block_from_projected, (a, b))
    options = dict(retain={2: 1}, dense_reference_limit=0, tol=1e-12,
                   maxiter=1000, propagate_operators=False)
    small = solve_sector_merge(l, r, cross, krylov_dimension=6, **options)
    large = solve_sector_merge(l, r, cross, krylov_dimension=999, **options)
    default = solve_sector_merge(l, r, cross, **options)
    expected = np.linalg.eigvalsh(SectorMergeOperator(l, r, cross, 2).dense_reference())[:1]
    for result in (small, large, default):
        np.testing.assert_allclose(result.energies[2], expected, atol=2e-12)
        assert result.diagnostics['sectors'][2]['computed_eigenpair_count'] == 3
    assert small.diagnostics['sectors'][2]['krylov_dimension'] == 6
    assert large.diagnostics['sectors'][2]['krylov_dimension'] == 9
    assert large.diagnostics['sectors'][2]['krylov_dimension_requested'] == 999
    assert default.diagnostics['sectors'][2]['krylov_dimension'] == 9
    assert default.diagnostics['krylov_dimension_requested'] is None
    small_peak = small.diagnostics['estimated_total_peak_bytes']
    large_peak = large.diagnostics['estimated_total_peak_bytes']
    assert large_peak - small_peak == 9*(9-6)*16*4
    solve_sector_merge(l, r, cross, krylov_dimension=6,
                       memory_limit_bytes=small_peak, **options)
    with pytest.raises(MemoryError, match='ARPACK/output workspace'):
        solve_sector_merge(l, r, cross, krylov_dimension=9,
                           memory_limit_bytes=small_peak, **options)


@pytest.mark.parametrize('invalid', [True, 0, 1, -1, 6., '6', np.nan])
def test_invalid_krylov_dimension_is_rejected_before_solving(invalid):
    a, b, cross, _ = _children()
    l, r = map(reduced_block_from_projected, (a, b))
    with pytest.raises(ValueError, match='krylov_dimension must be an integer'):
        solve_sector_merge(l, r, cross, retain={0: 1}, krylov_dimension=invalid)


@pytest.mark.parametrize('complex_gauge,ncv', [(False, 4), (True, 5)])
def test_krylov_dimension_respects_real_and_complex_arpack_gap(complex_gauge, ncv):
    a, b, cross, _ = _children(complex_gauge)
    l, r = map(reduced_block_from_projected, (a, b))
    with pytest.raises(ValueError, match='krylov_dimension is too small'):
        solve_sector_merge(l, r, cross, retain={2: 2}, dense_reference_limit=0,
                           krylov_dimension=ncv)
    dense = solve_sector_merge(l, r, cross, retain={2: 2}, krylov_dimension=2)
    assert dense.diagnostics['sectors'][2]['krylov_dimension'] is None
    assert dense.diagnostics['sectors'][2]['krylov_dimension_requested'] == 2


def test_degenerate_retained_subspaces_have_reproducible_projectors():
    a, b, _, _ = _children()
    l, r = map(reduced_block_from_projected, (a, b))
    l = replace(l, h_by_q={q: np.zeros_like(h) for q, h in l.h_by_q.items()})
    r = replace(r, h_by_q={q: np.zeros_like(h) for q, h in r.h_by_q.items()})
    first = solve_sector_merge(l, r, [], retain={0: 2, 2: 1}, seed=1)
    second = solve_sector_merge(l, r, [], retain={0: 2, 2: 1}, seed=2)
    for q in first.reduced_maps:
        w, v = first.reduced_maps[q], second.reduced_maps[q]
        np.testing.assert_allclose(w @ w.conj().T, v @ v.conj().T, atol=2e-13)
    assert any('degenerate' in warning for warning in first.warnings)


def test_bad_magnetic_scalar_vector_input_and_explicit_bounds_are_rejected():
    a, b, cross, _ = _children()
    l, r = map(reduced_block_from_projected, (a, b))
    h = a.hamiltonian.copy(); h[0, 0] += .1
    with pytest.raises(ValueError, match='scalar common'):
        reduced_block_from_arrays(3, a.labels, h, a.site_spin)
    wrong = list(l.site_spin); wrong[0] = tuple(np.zeros_like(s) for s in wrong[0])
    with pytest.raises(ValueError, match='total-spin'):
        validate_reduced_block(replace(l, site_spin=tuple(wrong)))
    with pytest.raises(ValueError, match='canonical'):
        reduced_block_from_arrays(3, a.labels[::-1], a.hamiltonian, a.site_spin)
    with pytest.raises(ValueError, match='absent'):
        SectorMergeOperator(l, r, cross, 1)
    with pytest.raises(MemoryError, match='memory budget'):
        SectorMergeOperator(l, r, cross, 0, memory_limit_bytes=10)
    with pytest.raises(ValueError, match='output bound'):
        solve_sector_merge(l, r, cross, retain={0: 2, 2: 2}, max_output_dimension=2)
    with pytest.raises(ValueError, match='small-system bound'):
        SectorMergeOperator(l, r, cross, 2).dense_reference(max_dimension=2)
    with pytest.raises(ValueError, match='finite and positive'):
        solve_sector_merge(l, r, cross, retain={0: 1}, tol=np.nan)
    with pytest.raises(ValueError, match='dimension exceeds'):
        validate_reduced_block(ReducedBlockData(3, {1: 10**12}, {}, ()))


def test_scalar_child_eigen_residual_is_not_a_full_physical_variance():
    # Two retained dimer singlets have no projected boundary spin. Their product
    # is exact in the one-dimensional child space but has physical variance 3/16.
    child = ReducedBlockData(2, {0: 1}, {0: np.zeros((1, 1))},
                              tuple(tuple(np.zeros((1, 1)) for _ in range(3)) for _ in range(2)))
    result = solve_sector_merge(child, child, [(1, 0, 1.)], retain={0: 1})
    np.testing.assert_allclose(result.energies[0], 0., atol=1e-15)
    np.testing.assert_allclose(result.eigen_residuals[0], 0., atol=1e-15)
    singlet = np.array([0., -1., 1., 0.])/np.sqrt(2)
    psi = np.kron(singlet, singlet)
    h = _hamiltonian(4, [(1, 2, 1.)])
    np.testing.assert_allclose(np.vdot(h @ psi, h @ psi), 3/16, atol=1e-15)
    assert any('physical H variance' in warning for warning in result.warnings)


@pytest.mark.parametrize('complex_gauge', [False, True])
def test_compact_rank_one_storage_and_recoupling_match_dense_all_components(complex_gauge):
    a, b, cross, _ = _children(complex_gauge)
    l, r = map(reduced_block_from_projected, (a, b))
    lc, rc = map(compact_reduced_block, (l, r))
    assert not lc.site_spin and lc.site_reduced_spin is not None
    for q in (0, 2, 4, 6):
        dense = SectorMergeOperator(l, r, cross, q)
        compact = SectorMergeOperator(lc, rc, cross, q)
        np.testing.assert_allclose(compact.dense_reference(), dense.dense_reference(), atol=3e-13)
    exact = solve_sector_merge(l, r, cross, retain={0: 2, 2: 2, 4: 1}, tol=1e-12)
    actual = solve_sector_merge(lc, rc, cross, retain={0: 2, 2: 2, 4: 1}, tol=1e-12,
                                 compact_boundary=True)
    reference = compact_reduced_block(exact.block)
    for si, sr in zip(actual.block.site_reduced_spin, reference.site_reduced_spin, strict=True):
        for key in sr:
            np.testing.assert_allclose(si[key], sr[key], atol=5e-13)
    for q in actual.energies:
        np.testing.assert_allclose(actual.energies[q], exact.energies[q], atol=3e-13)
    # Compare another full q>0 retaining merge without reconstructing sites.
    dd = solve_sector_merge(exact.block, exact.block, [(0, 5, .3), (4, 2, .8)],
                             retain={0: 2, 2: 2, 4: 1}, tol=1e-12)
    cc = solve_sector_merge(actual.block, actual.block, [(0, 5, .3), (4, 2, .8)],
                             retain={0: 2, 2: 2, 4: 1}, tol=1e-12, compact_boundary=True)
    for q in cc.energies:
        np.testing.assert_allclose(cc.energies[q], dd.energies[q], atol=5e-12)
    expected = compact_reduced_block(dd.block)
    for si, sr in zip(cc.block.site_reduced_spin, expected.site_reduced_spin, strict=True):
        for key in sr:
            np.testing.assert_allclose(si[key], sr[key], atol=5e-11)
    assert validate_reduced_block(cc.block)['total_spin_error'] < 1e-12


def test_compact_input_rejects_wrong_reduced_adjoint_and_total_spin():
    a, _, _, _ = _children(True)
    block = compact_reduced_block(reduced_block_from_projected(a))
    corrupt = [{key: value.copy() for key, value in s.items()} for s in block.site_reduced_spin]
    corrupt[0][1, 3][0, 0] += .1
    with pytest.raises(ValueError, match='Hermiticity or total-spin'):
        validate_reduced_block(replace(block, site_reduced_spin=tuple(corrupt)))
    with pytest.raises(ValueError, match='only one reduced vector'):
        validate_reduced_block(replace(block, site_spin=a.site_spin))


@pytest.mark.parametrize('complex_gauge', [False, True])
def test_arbitrary_map_controls_keep_offdiagonal_H_and_exact_physical_operators(complex_gauge):
    a, b, cross, physical = _children(complex_gauge)
    l, r = map(reduced_block_from_projected, (a, b))
    lc, rc = map(compact_reduced_block, (l, r))
    rng = np.random.default_rng(91823)
    maps = {}; u = {}; product = np.kron(a.basis, b.basis)
    for q, count in ((0, 2), (2, 2), (4, 1)):
        op = SectorMergeOperator(lc, rc, cross, q)
        trial = rng.normal(size=(op.dimension, count))+1j*rng.normal(size=(op.dimension, count))
        maps[q] = np.linalg.qr(trial)[0]
        u[q] = np.stack([np.column_stack([product @ op.expand(maps[q][:, i], m).ravel()
                                          for i in range(count)]) for m in range(-q, q+1, 2)], axis=-1)
    expected = compact_reduced_block(reduced_block_from_projected(project_block(BlockIsometry(6, u), physical)))
    actual = project_sector_maps(lc, rc, cross, maps=maps)
    assert abs(actual.h_by_q[0][0, 1]) > .01
    for q in maps:
        np.testing.assert_allclose(actual.h_by_q[q], expected.h_by_q[q], atol=3e-13)
    for sa, se in zip(actual.site_reduced_spin, expected.site_reduced_spin, strict=True):
        for key in se:
            np.testing.assert_allclose(sa[key], se[key], atol=3e-13)
    with pytest.raises(ValueError, match='orthonormal'):
        project_sector_maps(lc, rc, cross, maps={0: maps[0]*1.1})
    with pytest.raises(ValueError, match='rows'):
        project_sector_maps(lc, rc, cross, maps={0: np.ones((1, 1))})
