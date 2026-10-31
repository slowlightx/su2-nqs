"""Exact physical density-matrix oracles: never twirl away an input defect."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import spin_symmetry_readout as workflow


def test_maximally_mixed_four_spin_reference_has_known_complete_spectra():
    report, arrays = workflow.density_readout(np.eye(16) / 16)
    for q, mult, p in ((0, 2, 2/16), (2, 3, 9/16), (4, 1, 5/16)):
        sector = report['sectors'][str(q)]
        assert sector['p'] == pytest.approx(p, abs=2e-15)
        np.testing.assert_allclose(sector['conditional_multiplicity_spectrum'], np.full(mult, 1/mult), atol=3e-15)
        np.testing.assert_allclose(arrays[f'mult_q{q}'], np.eye(mult) * p/mult, atol=2e-15)
    assert report['entropy_nats'] == pytest.approx(4*np.log(2), abs=3e-15)
    assert report['diagnostics']['physical_reconstruction_frobenius'] < 3e-15


def test_commuting_with_S_squared_alone_is_not_accepted_as_SU2_invariance():
    u, labels = workflow.schur_columns()
    vector = u[:, labels.index((2, 0, 0))]
    rho = np.outer(vector, vector.conj())
    s2 = workflow.subsystem_total_spin_squared(4)
    assert np.linalg.norm(rho@s2-s2@rho) < 1e-13
    with pytest.raises(ValueError, match='no projection repair'):
        workflow.density_readout(rho)


def test_arbitrary_complex_multiplicity_coherences_reconstruct_without_diagonalization_bias():
    rng = np.random.default_rng(42)
    u, labels = workflow.schur_columns()
    schur = np.zeros((16, 16), dtype=complex)
    expected = {}
    for q, mult, p in ((0, 2, .25), (2, 3, .60), (4, 1, .15)):
        x = rng.normal(size=(mult, mult)) + 1j*rng.normal(size=(mult, mult))
        reduced = x@x.conj().T
        reduced *= p/np.trace(reduced)
        expected[q] = reduced
        for m in range(-q, q+1, 2):
            indices = [labels.index((q, m, a)) for a in range(mult)]
            schur[np.ix_(indices, indices)] = reduced/(q+1)
    rho = u@schur@u.conj().T
    report, arrays = workflow.density_readout(rho)
    for q, expected_block in expected.items():
        np.testing.assert_allclose(arrays[f'mult_q{q}'], expected_block, atol=3e-15)
        actual = report['sectors'][str(q)]['conditional_multiplicity_spectrum']
        np.testing.assert_allclose(actual, np.linalg.eigvalsh(expected_block/np.trace(expected_block))[::-1], atol=3e-15)
    assert report['diagnostics']['entropy_sum_error'] < 3e-15
    assert np.max(abs(arrays['mult_q0'].imag)) > .005


@pytest.mark.parametrize('rho', [np.eye(16)/8, np.eye(15)/15,
    np.diag([-.1] + [1.1/15]*15), np.full((16, 16), np.nan)])
def test_nonphysical_density_matrices_fail_closed(rho):
    with pytest.raises(ValueError):
        workflow.density_readout(rho)


def test_analytic_dimer_and_multiplicity_Bell_controls_separate_q_from_multiplicity():
    basis = np.asarray(workflow.enumerate_sz_configs(16, 8), dtype=np.int8)
    cut = (0, 1, 4, 5)
    internal = tuple((4*r+c, 4*r+c+1) for r in range(4) for c in (0, 2))
    crossing = tuple((4*r+c, 4*r+(c+1)%4) for r in range(4) for c in (1, 3))
    states = (workflow.dimer_state(basis, internal), workflow.dimer_state(basis, crossing), workflow.multiplicity_bell_state(basis))
    reports = []
    for state in states:
        assert np.vdot(state, state).real == pytest.approx(1., abs=3e-15)
        rho = workflow.reduced_density_matrix(state, basis, cut)
        report, _ = workflow.density_readout(rho)
        reports.append(report)
    assert reports[0]['sectors']['0']['p'] == pytest.approx(1.)
    assert reports[0]['entropy_nats'] == pytest.approx(0., abs=3e-14)
    assert reports[1]['entropy_nats'] == pytest.approx(4*np.log(2), abs=3e-14)
    assert reports[2]['sectors']['0']['p'] == pytest.approx(1.)
    assert reports[2]['multiplicity_entropy'] == pytest.approx(np.log(2), abs=3e-14)
    assert reports[2]['magnetic_entropy'] == pytest.approx(0., abs=3e-14)
    assert reports[2]['spin_weight_entropy'] == pytest.approx(0., abs=3e-14)
    np.testing.assert_allclose(reports[2]['sectors']['0']['conditional_multiplicity_spectrum'], [.5, .5], atol=2e-15)


def test_readout_entrypoint_does_not_build_or_diagonalize_a_Hamiltonian():
    tree = ast.parse(Path(workflow.__file__).read_text())
    names = {x.id for x in ast.walk(tree) if isinstance(x, ast.Name)}
    attributes = {x.attr for x in ast.walk(tree) if isinstance(x, ast.Attribute)}
    assert not names.intersection({'draw', 'eigsh', 'heisenberg_sz_graph_sparse', 'build_corpus'})
    assert 'eigh' not in attributes and 'eigvalsh' in attributes
    spec = json.loads(workflow.CONFIG.read_text())
    assert spec['region'] == [0, 1, 4, 5]
    assert not any(spec[k] for k in ('new_ed', 'new_samples', 'new_hamiltonian_actions'))
    assert spec['square_missing_requested_references'] == [0., .8]


def test_saved_readout_density_matrix_replay_and_input_provenance():
    spec = json.loads(workflow.CONFIG.read_text())
    path = workflow.ROOT / spec['output']
    if not (path/'results.json').exists():
        pytest.skip('readout not yet computed')
    result = json.loads((path/'results.json').read_text())
    assert workflow.canonical_sha256({k: v for k, v in result.items() if k != 'result_sha256'}) == result['result_sha256']
    assert workflow.file_sha256(path/'readout.npz') == result['arrays_sha256']
    for name, sha in result['input_sha256'].items():
        assert workflow.file_sha256(workflow.ROOT/name) == sha
    with np.load(path/'readout.npz') as arrays:
        basis = arrays['basis']
        for name, report in result['states'].items():
            rho = workflow.reduced_density_matrix(arrays[name+'_state'], basis, spec['region'])
            np.testing.assert_allclose(rho, arrays[name+'_rho'], atol=5e-15)
            replay, _ = workflow.density_readout(rho)
            assert replay['entropy_nats'] == pytest.approx(report['entropy_nats'], abs=2e-14)
            for q in ('0', '2', '4'):
                np.testing.assert_allclose(replay['sectors'][q]['conditional_multiplicity_spectrum'], report['sectors'][q]['conditional_multiplicity_spectrum'], atol=5e-14)
    assert result['new_ED'] == result['new_H_actions'] == result['new_samples'] == 0
    assert result['seconds'] < 120
