"""Independent finite-quadrature oracles; no ED or neural model required."""
from __future__ import annotations

import json

import numpy as np
import pytest

from su2_nqs.vmc.carrier_residual_geometry import (
    analyze_residual_geometry,
    leave_one_group_geometry,
    real_tangent_span_fraction,
)


def _fixture():
    rng = np.random.default_rng(2026100102)
    n = 9
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = (a + a.conj().T) / 2
    c = rng.normal(size=n) + 1j * rng.normal(size=n)
    reference = rng.normal(size=n) + 1j * rng.normal(size=n)
    psi = np.column_stack((c + .13 * reference,
        c + .2 * reference + .1 * (rng.normal(size=n) + 1j * rng.normal(size=n))))
    # Repeated configuration rows deliberately retain their multiplicity.
    indices = np.asarray([0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8])
    q_all = np.exp(rng.normal(size=n))
    return dict(carrier=c[indices], candidates=psi[indices], q=q_all[indices],
                reference=reference[indices], hcarrier=(h @ c)[indices],
                hcandidates=(h @ psi)[indices], href=(h @ reference)[indices],
                n_sites=4, groups=np.repeat(np.arange(4), 3)), (c, psi, reference, h)


def _assert_same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_same(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _assert_same(x, y)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        np.testing.assert_allclose(a, b, atol=3e-12, rtol=3e-12)
    else:
        assert a == b


def test_two_cross_terms_exact_with_repeated_rows_nonuniform_q():
    data, _ = _fixture()
    result = analyze_residual_geometry(**data)
    c, hc, q = data['carrier'], data['hcarrier'], data['q']
    inner = lambda a, b: np.sum(a.conj() * b / q)
    ec = inner(c, hc) / inner(c, c).real
    errors = []
    for k, arm in enumerate(result['arms']):
        p, hp = data['candidates'][:, k], data['hcandidates'][:, k]
        z = inner(p, p).real
        energy = inner(p, hp) / z
        variance = (inner(hp, hp).real / z - abs(energy) ** 2) / data['n_sites']
        out = arm['hamiltonian']
        delta = (energy.real - ec.real) / data['n_sites']
        decomposition = out['interference_sum_per_site'] + out['quadratic_per_site']
        errors.extend((abs(decomposition - delta), abs(out['variance_per_site'] - variance)))
        np.testing.assert_allclose(out['energy_real_per_site'], energy.real / data['n_sites'], atol=1e-13)
        np.testing.assert_allclose(out['energy_imag_per_site'], energy.imag / data['n_sites'], atol=1e-13)
        np.testing.assert_allclose(decomposition, delta, atol=1e-13)
        np.testing.assert_allclose(out['variance_per_site'], variance, atol=1e-13)
        assert out['hermiticity_cross_defect_abs_per_site'] > 1e-3
        assert abs(out['interference_left_per_site'] - out['interference_right_per_site']) > 1e-4
    print('max finite-Q energy/variance decomposition error:', max(errors))
    json.dumps(result, allow_nan=False)


def test_reference_innovation_split_reconstructs_every_term():
    data, _ = _fixture()
    result = analyze_residual_geometry(**data)
    for arm in result['arms']:
        reference = arm['reference']
        np.testing.assert_allclose(reference['projection_fraction'] + reference['innovation_fraction'], 1., atol=1e-14)
        out = arm['hamiltonian']
        split = out['reference_split']
        assert split['status'] == 'complete'
        for direction in ('left', 'right'):
            np.testing.assert_allclose(split[f'{direction}_reference_per_site'] + split[f'{direction}_innovation_per_site'],
                                       out[f'interference_{direction}_per_site'], atol=2e-13)
        np.testing.assert_allclose(sum(split[f'quadratic_{term}_per_site'] for term in ('reference', 'innovation', 'mixed')),
                                   out['quadratic_per_site'], atol=2e-13)
        np.testing.assert_allclose(split['quadratic_reference_innovation_cross_per_site']
            + split['quadratic_innovation_reference_cross_per_site'] + split['quadratic_cross_norm_correction_per_site'],
            split['quadratic_mixed_per_site'], atol=2e-13)
    without = dict(data)
    without.pop('href')
    split = analyze_residual_geometry(**without)['arms'][0]['hamiltonian']['reference_split']
    assert split['status'] == 'right_only_without_href'
    assert 'left_reference_per_site' not in split


def test_all_observables_independent_of_complex_ray_and_q_scales():
    data, _ = _fixture()
    expected = analyze_residual_geometry(**data)
    changed = dict(data)
    carrier_scale, reference_scale = 3 - 2j, -2 + 7j
    candidate_scales = np.asarray([-.1 + 4j, 2 - 3j])
    changed['carrier'] = data['carrier'] * carrier_scale
    changed['hcarrier'] = data['hcarrier'] * carrier_scale
    changed['candidates'] = data['candidates'] * candidate_scales
    changed['hcandidates'] = data['hcandidates'] * candidate_scales
    changed['reference'] = data['reference'] * reference_scale
    changed['href'] = data['href'] * reference_scale
    changed['q'] = data['q'] * 9e7
    _assert_same(expected, analyze_residual_geometry(**changed))


def test_complete_uniform_quadrature_has_hermitian_action():
    _, (c, psi, ref, h) = _fixture()
    result = analyze_residual_geometry(c, psi, np.ones(len(c)), reference=ref,
        hcarrier=h @ c, hcandidates=h @ psi, href=h @ ref)
    for arm in result['arms']:
        out = arm['hamiltonian']
        assert out['hermiticity_cross_defect_abs_per_site'] < 2e-14
        assert abs(out['energy_imag_per_site']) < 2e-14
        np.testing.assert_allclose(out['interference_left_per_site'], out['interference_right_per_site'], atol=2e-14)


def test_unit_fs_slope_and_curvature_reconstruct_fixed_rays():
    data, _ = _fixture()
    result = analyze_residual_geometry(**data)
    q = data['q']
    inner = lambda a, b: np.sum(a.conj() * b / q)
    zc = inner(data['carrier'], data['carrier']).real
    c, hc = data['carrier'] / np.sqrt(zc), data['hcarrier'] / np.sqrt(zc)
    ec = inner(c, hc).real / data['n_sites']
    for k, arm in enumerate(result['arms']):
        p, hp = data['candidates'][:, k], data['hcandidates'][:, k]
        b = inner(c, p)
        r, hr = p / b - c, hp / b - hc
        z = inner(r, r).real
        u, hu = r / np.sqrt(z), hr / np.sqrt(z)
        out = arm['hamiltonian']
        def energy(theta):
            v = np.cos(theta) * c + np.sin(theta) * u
            hv = np.cos(theta) * hc + np.sin(theta) * hu
            return (inner(v, hv).real / inner(v, v).real) / data['n_sites']
        for theta in (-.17, 0., .01, .23):
            expected = ec + np.sin(theta) * np.cos(theta) * out['unit_fs_slope_per_site']
            expected += np.sin(theta) ** 2 * out['unit_fs_curvature_per_site']
            np.testing.assert_allclose(energy(theta), expected, atol=2e-14)
        epsilon = 2e-6
        derivative = (energy(epsilon) - energy(-epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(derivative, out['unit_fs_slope_per_site'], atol=2e-10)


@pytest.mark.parametrize('rho', (2., -.7))
def test_real_scale_fit_preserves_opposite_direction(rho):
    c = np.asarray([1., 0., 0., 0.])
    r = np.asarray([0., .2 + .1j, -.1j, .3])
    psi = np.column_stack((c + r, (2 - 3j) * (c + rho * r)))
    pair = analyze_residual_geometry(c, psi, np.ones(4))['pairs'][0]
    np.testing.assert_allclose(pair['best_real_rho'], rho, atol=1e-14)
    assert pair['real_fit_defect'] < 1e-28
    np.testing.assert_allclose(pair['real_overlap'], np.sign(rho), atol=1e-14)
    np.testing.assert_allclose(pair['abs_overlap_squared'], 1., atol=1e-14)
    np.testing.assert_allclose(pair['positive_scale_fit_defect'], 0. if rho > 0 else 1., atol=1e-14)


def test_phase_rotated_tangent_is_not_a_real_rescaling():
    c = np.asarray([1., 0., 0.])
    r = np.asarray([0., .1, .2])
    pair = analyze_residual_geometry(c, np.column_stack((c + r, c + 1j * r)), np.ones(3))['pairs'][0]
    np.testing.assert_allclose(pair['normalized_gram'], [0., 1.], atol=2e-14)
    np.testing.assert_allclose(pair['real_fit_defect'], 1., atol=2e-14)


def test_single_row_concentration_and_loco_recompute_ray():
    c = np.ones(8)
    candidate = c.copy()
    candidate[0] += 2.
    groups = np.asarray([0, 1, 1, 2, 2, 3, 3, 3])
    result = leave_one_group_geometry(c, candidate[:, None], np.ones(8), groups=groups)
    arm = result['pooled']['arms'][0]
    np.testing.assert_allclose(arm['residual_concentration']['max_sample_fraction'], 7 / 8, atol=1e-14)
    removed = result['leave_one_group_out'][0]
    assert removed['omitted_group'] == 0
    assert removed['geometry']['arms'][0]['status'] == 'zero_or_unresolved_residual'
    assert removed['geometry']['arms'][0]['residual_concentration']['defined'] is False
    assert removed['geometry']['arms'][0]['canonical_residual_norm_squared'] is None


def test_zero_and_orthogonal_states_are_explicitly_undefined_and_nodes_safe():
    c = np.asarray([1., 0., 0.])
    psi = np.column_stack((np.zeros(3), 1j * c, np.asarray([0., 1., 0.]), np.asarray([1., .1, .2j])))
    h = np.asarray([[1., 1., 0.], [1., 0., 2.], [0., 2., -1.]])
    out = analyze_residual_geometry(c, psi, np.asarray([.2, 1., .3]), hcarrier=h @ c, hcandidates=h @ psi,
                                    reference=np.asarray([0., 1., 0.]), href=h[:, 1])
    assert [a['status'] for a in out['arms']] == ['zero_candidate_norm', 'zero_or_unresolved_residual',
        'near_orthogonal_to_carrier', 'defined']
    assert out['reference_status'] == 'defined'
    assert out['arms'][2]['canonical_residual_norm_squared'] is None
    assert out['arms'][3]['hamiltonian']['reference_split']['status'] == 'complete'
    json.dumps(out, allow_nan=False)


def test_real_tangent_span_uses_real_not_complex_coefficients():
    c = np.asarray([1., 0., 0.])
    tangent = np.asarray([0., 1., 0.])
    candidates = np.column_stack((c + .2 * tangent, c + .2j * tangent, c + np.asarray([0., 0., .2])))
    result = real_tangent_span_fraction(c, candidates, tangent[:, None], np.ones(3))
    assert result['real_rank'] == result['complex_rank'] == 1
    np.testing.assert_allclose([a['real_projection_fraction'] for a in result['arms']], [1., 0., 0.], atol=1e-14)
    np.testing.assert_allclose([a['complex_span_upper_bound'] for a in result['arms']], [1., 1., 0.], atol=1e-14)
    extended = real_tangent_span_fraction(c, candidates, np.column_stack((tangent, 2 * tangent, 1j * tangent, c)), np.ones(3))
    assert extended['real_rank'] == 2 and extended['complex_rank'] == 1
    assert extended['retained_probe_indices'] == [0, 1, 2]
    np.testing.assert_allclose([a['real_projection_fraction'] for a in extended['arms']], [1., 1., 0.], atol=1e-14)
    # Global physical-state phase/scale changes also multiply real-parameter JVPs.
    scale = 2 + 3j
    changed = real_tangent_span_fraction(c * scale, candidates * np.asarray([2j, -3., 4 - 1j]),
        tangent[:, None] * scale, np.ones(3) * 9)
    for expected, actual in zip(result['arms'], changed['arms'], strict=True):
        np.testing.assert_allclose(expected['real_projection_fraction'], actual['real_projection_fraction'], atol=1e-14)
        np.testing.assert_allclose(expected['complex_span_upper_bound'], actual['complex_span_upper_bound'], atol=1e-14)


def test_weighted_real_span_matches_independent_least_squares():
    rng = np.random.default_rng(42)
    count = 11
    c = rng.normal(size=count) + 1j * rng.normal(size=count)
    probes = rng.normal(size=(count, 3)) + 1j * rng.normal(size=(count, 3))
    psi = np.column_stack((c + .1 * probes[:, 0] - .05 * probes[:, 2],
                           c + .1j * probes[:, 0] + .07 * probes[:, 1]))
    q = np.exp(rng.normal(size=count))
    result = real_tangent_span_fraction(c, psi, probes, q)
    cw, tw = c / np.sqrt(q), probes / np.sqrt(q[:, None])
    tw -= cw[:, None] * (cw.conj() @ tw / np.vdot(cw, cw))[None, :]
    real = np.concatenate((tw.real, tw.imag), axis=0)
    for k, arm in enumerate(result['arms']):
        pw = psi[:, k] / np.sqrt(q)
        b = np.vdot(cw, pw) / np.vdot(cw, cw)
        r = pw / b - cw
        target = np.concatenate((r.real, r.imag))
        fit = real @ np.linalg.lstsq(real, target, rcond=None)[0]
        fraction = np.dot(fit, fit) / np.dot(target, target)
        np.testing.assert_allclose(arm['real_projection_fraction'], fraction, atol=3e-14)
        assert arm['real_projection_fraction'] <= arm['complex_span_upper_bound'] + 3e-14


def test_invalid_inputs_fail_without_silent_repairs():
    data, _ = _fixture()
    for change in ({'q': np.zeros(12)}, {'q': np.ones(11)}, {'n_sites': 0}, {'n_sites': True},
                   {'carrier': np.zeros(12)}, {'groups': np.zeros(12, dtype=float)},
                   {'relative_tolerance': 0}, {'hcandidates': None}, {'reference': None}):
        with pytest.raises(ValueError):
            analyze_residual_geometry(**{**data, **change})
    with pytest.raises(ValueError):
        leave_one_group_geometry(data['carrier'], data['candidates'], data['q'], groups=np.zeros(12, dtype=int))
