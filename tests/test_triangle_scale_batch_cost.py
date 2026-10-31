"""Batch-cost wrapper preserves the sealed physical H action and its oracle."""
from itertools import combinations
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_scale_batch_cost import (
    checked_cost_parent, collect_accounting, make_scalar_connectivity_reference,
    relative_vector_error,
)
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.checkpoint import save_checkpoint
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action

jax.config.update('jax_enable_x64', True)


def test_real_batch_H_matches_scalar_connectivity_with_nodes_and_duplicate_bonds():
    states = np.zeros((6, 4), np.int32)
    for row, up in zip(states, combinations(range(4), 2), strict=True):
        row[list(up)] = 1
    powers = jnp.asarray([8, 4, 2, 1])
    table = np.linspace(-.8, 1.1, 16)+1j*np.linspace(.3, -.2, 16)
    table[states[0] @ np.asarray(powers)] = 0.
    table = jnp.asarray(table)
    amplitude = lambda params, row: params[jnp.dot(row, powers)]
    reference = make_scalar_connectivity_reference(amplitude, table)
    # The second graph shares physical queries but NOT coupling coefficients.
    for bonds in (((0, 1, 1.), (0, 1, .125), (2, 3, -.5), (1, 3, .25), (1, 2, .4)),
                  ((0, 1, 1.), (2, 3, -.5), (1, 2, .4))):
        graph = SzHeisenbergGraph(4, bonds, n_up=2)
        oracle = np.asarray([reference(graph, row) for row in states])
        expected = graph.to_dense() @ np.asarray(table)[states @ np.asarray(powers)]
        np.testing.assert_allclose(oracle[:, 1], expected, atol=2e-15, rtol=2e-14)
        action = make_amplitude_heisenberg_action(amplitude, bonds, n_sites=4)
        for batch in (1, 2, 4, 6):
            psi, hpsi = action(table, jnp.asarray(states[:batch]))
            assert relative_vector_error(psi, oracle[:batch, 0]) < 2e-14
            assert relative_vector_error(hpsi, oracle[:batch, 1]) < 2e-14
        assert oracle[0, 0] == 0 and abs(oracle[0, 1]) > 0


def _cost_parent(tmp_path):
    directory = tmp_path/'old_cost'
    directory.mkdir()
    source = tmp_path/'sealed_kernel.py'
    source.write_text('# fixed physical kernel\n')
    params = {arm: {'weights': jnp.arange(3, dtype=jnp.float64)}
              for arm in ('binary_truncated', 'ternary_joint')}
    registration = dict(config_sha256='old_config_sha',
        config={'parent': {'checkpoint_sha256': 'old_trained_checkpoint_sha'}},
        sources={'sealed_kernel.py': file_sha256(source)})
    result = dict(status='completed', stage='full', censored=False, H_finite=True,
        gradient_finite={arm: True for arm in params}, checkpoint_replay_exact=True,
        sampler_RNG_checkpoint_replay_exact=True, config_sha256='old_config_sha',
        parameter_counts={arm: 3 for arm in params})
    result['result_sha256'] = canonical_sha256(result)
    save_checkpoint(directory/'cost_checkpoint.pkl', params,
        metadata=dict(trained=False, source_checkpoint_sha256='old_trained_checkpoint_sha'))
    (directory/'registration.json').write_text(json.dumps(registration))
    (directory/'results.json').write_text(json.dumps(result))
    request = dict(directory=str(directory),
        checkpoint_sha256=file_sha256(directory/'cost_checkpoint.pkl'),
        registration_sha256=file_sha256(directory/'registration.json'),
        results_sha256=file_sha256(directory/'results.json'))
    return request, source


def test_completed_cost_parent_remains_sealed_and_not_resigned(tmp_path):
    request, source = _cost_parent(tmp_path)
    params, config = checked_cost_parent(request, root=tmp_path)
    assert set(params) == {'binary_truncated', 'ternary_joint'}
    assert config['parent']['checkpoint_sha256'] == 'old_trained_checkpoint_sha'
    source.write_text('# changed kernel\n')
    with pytest.raises(ValueError, match='sealed cost source changed'):
        checked_cost_parent(request, root=tmp_path)


def test_bad_old_result_cannot_be_relabelled_as_a_pass(tmp_path):
    request, _ = _cost_parent(tmp_path)
    path = tmp_path/'old_cost/results.json'
    result = json.loads(path.read_text())
    result['H_finite'] = False
    result['result_sha256'] = canonical_sha256({k: v for k, v in result.items() if k != 'result_sha256'})
    path.write_text(json.dumps(result))
    request['results_sha256'] = file_sha256(path)
    with pytest.raises(ValueError, match='completed full-H/gradient/checkpoint gate'):
        checked_cost_parent(request, root=tmp_path)


def test_hard_kill_records_partial_batch_evidence(tmp_path, monkeypatch):
    output = tmp_path/'batch'
    output.mkdir()
    (output/'results.json').write_text(json.dumps(dict(status='running', cases=[{'batch': 1}],
        current_operation='J2=0.125_batch=4_full_H_cold')))
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(output=str(output))))
    monkeypatch.setattr('subprocess.check_output', lambda *a, **k:
        '123|TIMEOUT|0:15|1800|cpu=4,gres/gpu=1,mem=64G|\n')
    report = collect_accounting(config, '123')
    assert report['status'] == 'censored_or_failed'
    assert report['completed_cases'] == 1
    assert report['last_operation'] == 'J2=0.125_batch=4_full_H_cold'
    assert not report['energy_estimate'] and report['training_steps'] == 0


def test_relative_error_never_repairs_a_zero_and_checks_shape():
    assert relative_vector_error(np.zeros(2), np.zeros(2)) == 0.
    assert relative_vector_error(np.ones(2), np.zeros(2)) == 1.
    assert relative_vector_error(np.full(2, 1e-210), np.zeros(2)) == 1.
    assert relative_vector_error(np.full(2, 1.1e-210), np.full(2, 1e-210)) > .09
    with pytest.raises(ValueError, match='matching'):
        relative_vector_error(np.zeros(2), np.zeros(3))
