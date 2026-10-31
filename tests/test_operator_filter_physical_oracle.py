"""Independent six-spin physical-CG oracle for offline polynomial filters."""
import numpy as np

from su2_nqs.frontier.operator_filter_learning import (
    build_boundary_residual_sketch, build_filter_sketch, filtered_columns,
    occupied_seed_rotations, sketch_ritz_columns,
)
from su2_nqs.operators.su2_block_transfer import BlockIsometry, project_block
from su2_nqs.operators.su2_sector_merge import (
    SectorMergeOperator, reduced_block_from_projected,
)
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.triangle_blocks import triangle_block_tables


def _physical_h(n, bonds):
    """Independent bit-swap S_i.S_j matrix, site zero most significant."""
    rows = np.arange(1 << n)
    result = np.zeros((1 << n, 1 << n), complex)
    for i, j, coupling in bonds:
        different = ((rows >> (n-1-i)) ^ (rows >> (n-1-j))) & 1
        result[rows, rows] += coupling*(.25-.5*different)
        mask = (1 << (n-1-i)) | (1 << (n-1-j))
        columns = rows[different == 1]
        result[columns ^ mask, columns] += .5*coupling
    return result


def _three_spin_sectors(rng):
    mags = np.asarray([2*i.bit_count()-3 for i in range(8)])
    sectors = {}
    for q, table in triangle_block_tables()['three'].items():
        width = table.shape[1]
        rotation = np.linalg.qr(rng.normal(size=(width, width))
                                +1j*rng.normal(size=(width, width)))[0]
        sectors[q] = np.stack([(table @ rotation)*(mags == m)[:, None]
                               for m in range(-q, q+1, 2)], axis=-1)
    return sectors


def _physical_coupled_columns(left, right, channels, q, m):
    columns = []
    for ql, a, qr, b in channels:
        vector = np.zeros(64, complex)
        for ml in range(-ql, ql+1, 2):
            mr = m-ml
            if abs(mr) <= qr and (mr+qr) % 2 == 0:
                cg = clebsch_gordan(ql, ml, qr, mr, q, m)
                vector += cg*np.kron(left[ql][:, a, (ml+ql)//2],
                                     right[qr][:, b, (mr+qr)//2])
        columns.append(vector)
    return np.column_stack(columns)


def physical_filter_oracle_errors():
    """All doubled-q/m channels, complex gauges, and a physical H2 caveat."""
    rng = np.random.default_rng(2610269501)
    l, r = _three_spin_sectors(rng), _three_spin_sectors(rng)
    lb = [(0, 1, .9), (1, 2, -.2), (0, 2, .43)]
    rb = [(0, 1, .81), (1, 2, 1.13), (0, 2, .17)]
    cross = [(0, 0, .33), (0, 2, -.27), (1, 0, 1.07), (2, 1, .125)]
    left = project_block(BlockIsometry(3, l), _physical_h(3, lb))
    right = project_block(BlockIsometry(3, r), _physical_h(3, rb))
    a, b = reduced_block_from_projected(left), reduced_block_from_projected(right)
    h = _physical_h(6, lb+[(i+3, j+3, c) for i, j, c in rb]
                    +[(i, j+3, c) for i, j, c in cross])
    s2 = 4.5*np.eye(64)+2*_physical_h(6, [(i, j, 1.) for i in range(6) for j in range(i+1, 6)])
    lv = np.concatenate([np.linalg.eigvalsh(x) for x in a.h_by_q.values()])
    rv = np.concatenate([np.linalg.eigvalsh(x) for x in b.h_by_q.values()])
    radius_cross = .75*sum(abs(c) for _, _, c in cross)
    lo, hi = lv.min()+rv.min()-radius_cross, lv.max()+rv.max()+radius_cross
    center, radius = (lo+hi)/2, (hi-lo)/2
    normalized_h = (h-center*np.eye(64))/radius
    assert np.max(abs(np.linalg.eigvalsh(normalized_h))) <= 1+2e-14
    errors = dict(projector=0., energy=0., norm=0., total_spin=0., sector_action=0., h2_leakage=0.)
    coefficients = np.array([-.23, .11, -.27])
    for q in (0, 2, 4, 6):
        op = SectorMergeOperator(a, b, cross, q)
        width = min(2, op.dimension)
        seed = np.linalg.qr(rng.normal(size=(op.dimension, width))
                            +1j*rng.normal(size=(op.dimension, width)))[0]
        sketch = build_filter_sketch(op.matvec, seed, lo, hi)
        w, report = filtered_columns(sketch, coefficients)
        _, ritz = sketch_ritz_columns(sketch)
        assert sketch.H_columns == 4*width
        assert report['minimum_filter_Gram_eigenvalue'] >= .01-2e-13
        assert ritz['raw_trace_energy'] <= report['raw_trace_energy']+2e-12
        for m in range(-q, q+1, 2):
            embedding = _physical_coupled_columns(l, r, op.channels, q, m)
            actual = embedding @ w
            power = embedding @ seed
            reference = power.copy()
            for coefficient in coefficients:
                power = normalized_h @ power
                reference += coefficient*power
            reference = np.linalg.qr(reference)[0]
            errors['projector'] = max(errors['projector'], float(np.max(abs(
                actual @ actual.conj().T-reference @ reference.conj().T))))
            errors['energy'] = max(errors['energy'], abs(float(np.trace(actual.conj().T @ h @ actual).real/width)
                                                        -report['raw_trace_energy']))
            errors['norm'] = max(errors['norm'], float(np.max(abs(actual.conj().T @ actual-np.eye(width)))))
            errors['total_spin'] = max(errors['total_spin'], float(np.max(abs(s2 @ actual-q*(q+2)/4*actual))))
            acted = np.column_stack([op.matvec(x) for x in seed.T])
            errors['sector_action'] = max(errors['sector_action'], float(np.max(abs(h @ embedding @ seed-embedding @ acted))))
            if q == 0:
                full_second = actual.conj().T @ h @ h @ actual
                projected_h = actual.conj().T @ h @ actual
                leakage = full_second-projected_h @ projected_h
                assert np.linalg.eigvalsh(leakage).min() >= -2e-13
                errors['h2_leakage'] = float(np.trace(leakage).real)
    return errors


def test_polynomial_filter_all_m_matches_independent_physical_cg_and_H():
    errors = physical_filter_oracle_errors()
    for key in ('projector', 'energy', 'norm', 'total_spin', 'sector_action'):
        assert errors[key] < 3e-12, (key, errors)
    # Exact first moments and legal multiplets do not make projected H squared
    # the full physical second moment of the filtered rank-two singlet space.
    assert errors['h2_leakage'] > 1e-5


def physical_boundary_residual_oracle_errors():
    """Check noncommuting words against independent six-spin physical H.

    The full-space reference constructs the commutator and two-exchange
    direction directly, without using the reduced builder's shared actions.
    Both child gauges and the initial rank-two seed are genuinely complex.
    """
    from projects.clebsch_tree_nqs.workflows.triangle_operator_filter_attention import internal_action

    rng = np.random.default_rng(2610269502)
    l, r = _three_spin_sectors(rng), _three_spin_sectors(rng)
    lb = [(0, 1, .9), (1, 2, -.2), (0, 2, .43)]
    rb = [(0, 1, .81), (1, 2, 1.13), (0, 2, .17)]
    cross = [(0, 0, .33), (0, 2, -.27), (1, 0, 1.07), (2, 1, .125)]
    left = reduced_block_from_projected(project_block(BlockIsometry(3, l), _physical_h(3, lb)))
    right = reduced_block_from_projected(project_block(BlockIsometry(3, r), _physical_h(3, rb)))
    h0 = _physical_h(6, lb+[(i+3, j+3, c) for i, j, c in rb])
    h = h0+_physical_h(6, [(i, j+3, c) for i, j, c in cross])
    s2 = 4.5*np.eye(64)+2*_physical_h(6, [(i, j, 1.) for i in range(6) for j in range(i+1, 6)])
    lv = np.concatenate([np.linalg.eigvalsh(x) for x in left.h_by_q.values()])
    rv = np.concatenate([np.linalg.eigvalsh(x) for x in right.h_by_q.values()])
    lo, hi = lv.min()+rv.min(), lv.max()+rv.max()
    center, radius = (lo+hi)/2, (hi-lo)/2
    cross_bound = .75*sum(abs(c) for _, _, c in cross)
    a = (h0-center*np.eye(64))/radius
    b = (h-h0)/cross_bound
    assert max(np.linalg.norm(a, 2), np.linalg.norm(b, 2)) <= 1+2e-14
    errors = dict(words=0., projector=0., energy=0., cached_energy=0., norm=0.,
                  total_spin=0., internal_action=0., seed_orthogonality=0.,
                  minimum_gram=1., maximum_gram=1.)
    coefficients = np.array([-1.7, .83, 1.21])
    for q in (0, 2, 4, 6):
        op = SectorMergeOperator(left, right, cross, q)
        width = min(2, op.dimension)
        seed = np.linalg.qr(rng.normal(size=(op.dimension, width))
                            +1j*rng.normal(size=(op.dimension, width)))[0]
        internal = lambda x: internal_action(left, right, op.channels, x[:, None])[:, 0]
        sketch = build_boundary_residual_sketch(op.matvec, internal, seed, lo, hi, cross_bound)
        assert sketch.H_columns == op.matvec_count == 5*width
        assert sketch.coefficient_bound == 2. and sketch.minimum_gram_bound == 1.
        w, report = filtered_columns(sketch, coefficients)
        _, ritz = sketch_ritz_columns(sketch)
        assert ritz['raw_trace_energy'] <= report['raw_trace_energy']+2e-12
        for signs in np.ndindex(2, 2, 2):
            corner = 2*(2*np.asarray(signs)-1)
            raw = sketch.basis @ np.concatenate([x*np.eye(width) for x in np.r_[1., corner]])
            eigen = np.linalg.eigvalsh(raw.conj().T @ raw)
            errors['minimum_gram'] = min(errors['minimum_gram'], float(eigen[0]))
            errors['maximum_gram'] = max(errors['maximum_gram'], float(eigen[-1]))
        for m in range(-q, q+1, 2):
            embedding = _physical_coupled_columns(l, r, op.channels, q, m)
            physical_seed = embedding @ seed
            complement = np.eye(64)-physical_seed @ physical_seed.conj().T
            directions = [complement @ b @ physical_seed,
                          complement @ (a @ b-b @ a) @ physical_seed/2,
                          complement @ b @ b @ physical_seed]
            physical_sketch = np.column_stack([physical_seed, *directions])
            errors['words'] = max(errors['words'], float(np.max(abs(embedding @ sketch.basis-physical_sketch))))
            errors['seed_orthogonality'] = max(errors['seed_orthogonality'],
                float(np.max(abs(physical_seed.conj().T @ physical_sketch[:, width:]))))
            reference = np.linalg.qr(physical_seed+sum(c*d for c, d in zip(coefficients, directions, strict=True)))[0]
            actual = embedding @ w
            errors['projector'] = max(errors['projector'], float(np.max(abs(
                actual @ actual.conj().T-reference @ reference.conj().T))))
            errors['energy'] = max(errors['energy'], abs(float(np.trace(actual.conj().T @ h @ actual).real/width)
                                                        -report['raw_trace_energy']))
            exact_cache = physical_sketch.conj().T @ ((h-center*np.eye(64)) @ physical_sketch)/(radius+cross_bound)
            errors['cached_energy'] = max(errors['cached_energy'], float(np.max(abs(exact_cache-sketch.energy))))
            errors['norm'] = max(errors['norm'], float(np.max(abs(actual.conj().T @ actual-np.eye(width)))))
            errors['total_spin'] = max(errors['total_spin'], float(np.max(abs(s2 @ actual-q*(q+2)/4*actual))))
            acted = np.column_stack([internal(x) for x in seed.T])
            errors['internal_action'] = max(errors['internal_action'],
                float(np.max(abs(h0 @ physical_seed-embedding @ acted))))
    return errors


def test_boundary_residual_words_all_m_match_independent_physical_cg_and_H():
    errors = physical_boundary_residual_oracle_errors()
    for key in ('words', 'projector', 'energy', 'cached_energy', 'norm',
                'total_spin', 'internal_action', 'seed_orthogonality'):
        assert errors[key] < 3e-12, (key, errors)
    assert errors['minimum_gram'] >= 1-3e-12
    assert errors['maximum_gram'] <= 37+3e-12


def physical_occupied_seed_oracle_errors():
    """Complex root singlet partial traces fix the right-ket conjugation."""
    rng = np.random.default_rng(2610269503)
    c = rng.normal(size=(5, 6))+1j*rng.normal(size=(5, 6))
    c *= np.sqrt(.37/np.vdot(c, c).real)
    width, q = 3, 3
    left, right, report = occupied_seed_rotations(c, width)
    magnetic = np.asarray([[clebsch_gordan(q, m, q, n, 0, 0)
                            for n in range(-q, q+1, 2)]
                           for m in range(-q, q+1, 2)])
    full = np.einsum('ab,mn->ambn', c, magnetic).reshape(5*(q+1), 6*(q+1))
    density_left = (full @ full.conj().T).reshape(5, q+1, 5, q+1)
    density_right = (full.T @ full.conj()).reshape(6, q+1, 6, q+1)
    multiplicity_left = np.einsum('amcm->ac', density_left)
    multiplicity_right = np.einsum('bndn->bd', density_right)
    expected_left = np.linalg.eigh(multiplicity_left)[1][:, -width:]
    expected_right = np.linalg.eigh(multiplicity_right)[1][:, -width:]
    ul = np.linalg.qr(rng.normal(size=(5, 5))+1j*rng.normal(size=(5, 5)))[0]
    ur = np.linalg.qr(rng.normal(size=(6, 6))+1j*rng.normal(size=(6, 6)))[0]
    rotated_left, rotated_right, rotated_report = occupied_seed_rotations(ul.conj().T @ c @ ur.conj(), width)
    wrong_right = np.linalg.eigh(c.conj().T @ c)[1][:, -width:]
    empty_left, empty_right, empty_report = occupied_seed_rotations(np.zeros((5, 6), complex), width)
    assert np.array_equal(empty_left, np.eye(5)[:, :width])
    assert np.array_equal(empty_right, np.eye(6)[:, :width])
    assert empty_report['exactly_zero_sector'] and empty_report['sector_weight'] == 0.
    projector = lambda w: w @ w.conj().T
    return dict(left_projector=float(np.max(abs(projector(left)-projector(expected_left)))),
        right_projector=float(np.max(abs(projector(right)-projector(expected_right)))),
        left_gauge=float(np.max(abs(projector(left)-projector(ul @ rotated_left)))),
        right_gauge=float(np.max(abs(projector(right)-projector(ur @ rotated_right)))),
        sector_weight=abs(report['sector_weight']-float(np.vdot(full, full).real)),
        gauge_weight=abs(report['sector_weight']-rotated_report['sector_weight']),
        discarded_wrong_right_convention_error=float(np.max(abs(projector(right)-projector(wrong_right)))),
        cutoff_gap_min=float(min(report['cutoff_gaps'])))


def test_occupied_donor_frames_match_complex_singlet_partial_traces_and_gauges():
    errors = physical_occupied_seed_oracle_errors()
    for key in ('left_projector', 'right_projector', 'left_gauge', 'right_gauge',
                'sector_weight', 'gauge_weight'):
        assert errors[key] < 3e-12, (key, errors)
    assert errors['discarded_wrong_right_convention_error'] > .1
    assert errors['cutoff_gap_min'] > 1e-4
