"""Independent physical-support and experiment-contract checks."""
import json
from itertools import combinations

import jax
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_conditional_carrier_resonance import (
    ARMS, CONFIG, PARAMETERS, carrier_columns, conditional_coherence,
    generator, load_case, local_indices, state_function,
    _rank_and_capture,
)
from su2_nqs.models.conditional_carrier_resonance import conditional_filter_matrix


def eight_site_basis():
    values = []
    for ups in combinations(range(8), 4):
        x = np.zeros(8, dtype=int)
        x[list(ups)] = 1
        values.append(x)
    return np.asarray(values)


def test_precontracted_columns_reproduce_direct_physical_filter():
    bits = eight_site_basis()
    local, _ = local_indices(bits, tuple(range(8)))
    c = np.random.default_rng(118).normal(size=len(bits))
    c /= np.linalg.norm(c)
    bank, _ = carrier_columns(np.eye(len(c)), c, bits, tuple(range(8)))
    theta = np.random.default_rng(119).normal(0, .05, 60)
    matrix = np.asarray(conditional_filter_matrix(theta))[np.ix_(local, local)]
    direct = matrix @ c
    direct /= np.linalg.norm(direct)
    np.testing.assert_allclose(state_function(c, bank, 'conditional')(theta), direct, atol=2e-14)
    for arm in ARMS:
        assert np.isfinite(np.asarray(jax.jacfwd(state_function(c, bank, arm))(np.zeros(PARAMETERS[arm])))).all()


def test_singlet_coherence_is_actual_oriented_dimer_matrix_element():
    from su2_nqs.su2.recoupling import tree_basis_matrix
    from su2_nqs.su2.fusion_tree import FusionTree
    cell, _ = tree_basis_matrix(FusionTree.balanced(4), 0, 0)
    active = (cell[:, 0] + cell[:, 1])/np.sqrt(2)
    full = np.kron(active, cell[:, 0])
    bits = eight_site_basis()
    integers = bits @ (1 << np.arange(7, -1, -1))
    measures = conditional_coherence(full[integers], bits, tuple(range(8)))
    np.testing.assert_allclose(sum(measures['spectator_probabilities']), 1, atol=2e-14)
    np.testing.assert_allclose(sum(measures['joint_singlet_triplet_coherence']), 1, atol=2e-14)


def test_cases_reuse_saved_references_without_hamiltonian_eigensolver():
    spec = json.loads(CONFIG.read_text())
    assert spec['remote_allowed'] is False
    assert tuple(spec['arms']) == ARMS
    assert spec['iterations'] == 120
    for case, expected in zip(spec['cases'], (14, 31), strict=True):
        phi, h, physical, bits = load_case(case)
        assert phi.shape == (12870, expected)
        np.testing.assert_allclose(phi.conj().T @ phi, np.eye(expected), atol=1e-11)
        assert physical.shape == (len(bits),)
        np.testing.assert_allclose(h, h.conj().T, atol=1e-11)
        assert np.linalg.norm(phi.conj().T @ physical) > .99


def test_triangle_basis_order_is_explicitly_reconciled():
    from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
    from projects.clebsch_tree_nqs.workflows.spin_conditional_carrier_resonance import ROOT
    case = json.loads(CONFIG.read_text())['cases'][0]
    phi, _, _, bits = load_case(case)
    canonical = fixed_m_spin_basis(16, 0)
    assert not np.array_equal(bits, canonical)
    indices = {tuple(x): i for i, x in enumerate(canonical)}
    with np.load(ROOT/case['sector_bundle']) as source:
        np.testing.assert_array_equal(phi, source[case['phi_key']][[indices[tuple(x)] for x in bits]])


def test_nulls_replay_same_conditional_state_not_separately_projected_factors():
    rng = np.random.default_rng(333)
    c = rng.normal(size=5); c /= np.linalg.norm(c)
    bank = rng.normal(size=(5, 60))
    for arm in ('factorized', 'diagonal_conditioned'):
        p = rng.normal(0, .05, PARAMETERS[arm])
        np.testing.assert_allclose(state_function(c, bank, arm)(p),
                                   state_function(c, bank, 'conditional')(generator(p, arm).reshape(-1)), atol=1e-14)


def test_real_parameter_tangent_does_not_use_complex_linear_span():
    result = _rank_and_capture(np.array([[1j]]), np.array([[1.]]))
    assert result['outside_reference_frobenius_fraction'] == 1.


def test_conditional_parameter_box_contains_null_images():
    spec = json.loads(CONFIG.read_text())
    for arm in ('factorized', 'diagonal_conditioned'):
        lo, hi = spec['parameter_bounds'][arm]
        for sign in (lo, hi):
            full = np.asarray(generator(np.full(PARAMETERS[arm], sign), arm))
            assert np.min(full) >= spec['parameter_bounds']['conditional'][0]
            assert np.max(full) <= spec['parameter_bounds']['conditional'][1]


def test_completed_artifact_replays_physical_h_and_preserves_seals():
    import pytest
    from projects.clebsch_tree_nqs.workflows.spin_conditional_carrier_resonance import ROOT, metrics
    from projects.clebsch_tree_nqs.workflows.spin import _prepare_case
    from su2_nqs.frontier.artifacts import canonical_sha256
    from su2_nqs.frontier.provenance import file_sha256
    directory = ROOT/'projects/clebsch_tree_nqs/results/spin_conditional_carrier_resonance_v1'
    if not (directory/'results.json').exists():
        pytest.skip('bounded numerical experiment has not been executed')
    reg = json.loads((directory/'registration.json').read_text())
    assert reg['sha256'] == canonical_sha256({k: v for k, v in reg.items() if k != 'sha256'})
    for path, digest in (reg['inputs'] | reg['sources']).items():
        assert file_sha256(ROOT/path) == digest, path
    result = json.loads((directory/'results.json').read_text())
    assert result['registration_sha256'] == reg['sha256']
    assert result['result_sha256'] == canonical_sha256({k: v for k, v in result.items() if k != 'result_sha256'})
    assert file_sha256(directory/'states.npz') == result['states_sha256']
    with np.load(directory/'states.npz') as states:
        for case in reg['config']['cases']:
            phi, h, physical, bits = load_case(case)
            c = phi.conj().T@physical; c /= np.linalg.norm(c)
            geometry = 'square' if case['name'].startswith('square') else 'triangular'
            data = _prepare_case(dict(geometry=geometry, rows=4, columns=4,
                    j2_over_j1=.5 if geometry == 'square' else .125,
                    exact_energy_anchor=16*case['reference_energy_per_site']), 'fixed_m0')
            order = {tuple(x): i for i, x in enumerate(bits)}
            indices = np.asarray([order[tuple(x)] for x in data['basis']])
            for arm in ARMS:
                bank, _ = carrier_columns(phi, c, bits, tuple(case['scrambled_support'] if arm == 'site_scrambled' else case['support']))
                fn = state_function(c, bank, arm)
                for row in result['cases'][case['name']]['arms'][arm]:
                    prefix = f"{case['name']}_{arm}_{row['start']['seed']}"
                    v = states[prefix+'_state']
                    np.testing.assert_allclose(fn(states[prefix+'_parameters']), v, atol=2e-13)
                    w = (phi@v)[indices]; hw = data['hamiltonian']@w
                    e = np.vdot(w, hw).real/np.vdot(w, w).real
                    var = np.linalg.norm(hw-e*w)**2/np.vdot(w, w).real
                    assert abs(e/16-row['energy_per_site']) < 2e-12
                    assert abs(var/16-row['variance_per_site']) < 2e-12
                    assert np.max(abs(w[data['tables']['gamma_a1']]-w)) < 2e-11
    assert result['n36_cost_profile']['full_H_calls'] == 0
    assert result['new_ed_diagonalizations'] == 0
    assert result['remote_jobs'] == 0
