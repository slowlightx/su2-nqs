"""Frozen synthetic inputs check audit seals and zero-query complement gates."""
import json

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import spin_rotation_disorder_analysis as workflow
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256


@pytest.fixture
def registered(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    # Source closure is separately validated in provenance tests; these data
    # fixtures are intentionally outside the repository package root.
    monkeypatch.setattr(workflow, 'transitive_local_source_manifest', lambda *a, **k: {})
    def sealed(path, data):
        path.write_text(json.dumps({**data, 'result_sha256': canonical_sha256(data)}))
    series = dict(records=[dict(label='synthetic',method='NQS',cut='pair',sites=[0,1],
                               energy=None,problem=dict(geometry='triangular',j2=.125))])
    (tmp_path/'series.json').write_text(json.dumps(series))
    np.savez(tmp_path/'series.npz', synthetic__pair__rho=np.eye(4)/4)
    directory = tmp_path/'audit'; directory.mkdir()
    streams = []
    rng = np.random.default_rng(260612132)
    for seed, nchain in [(1,2),(2,3)]:
        x = np.array([rng.permutation([0]*18+[1]*18) for _ in range(4*nchain)])
        x = x.reshape(4,nchain,36)
        np.savez(directory/f'stream{seed}.npz', states=x)
        proof = dict(seed=seed,params_sha256='fixed',registration_sha256='registered',
                     raw_sha256=file_sha256(directory/f'stream{seed}.npz'))
        sealed(directory/f'stream{seed}.json',proof)
        streams.append(proof)
    sealed(directory/'results.json',dict(streams=streams,params_sha256='fixed',
        registration_sha256='registered',energy=dict(samples=20,independent_chains=5)))
    config = dict(series_results='series.json',series_results_sha256=file_sha256(tmp_path/'series.json'),
        series_arrays='series.npz',series_arrays_sha256=file_sha256(tmp_path/'series.npz'),
        output='output',angle_points=5,seconds_cap=20,jackknife_caches=[],
        born_audits=[dict(label='synthetic',directory='audit',audit_sha256=file_sha256(directory/'results.json'))],
        regions=[dict(name='A4',sites=[0,1,6,7],contractible=True)])
    path = tmp_path/'config.json'; path.write_text(json.dumps(config))
    return path


def test_frozen_zero_query_workflow_and_complement(registered):
    result = workflow.analyze(registered)
    assert result['new_H_calls'] == result['new_samples'] == result['new_amplitude_queries'] == 0
    assert not result['phase_claim'] and not result['cusp_fit'] and not result['log_floor']
    assert result['born_records'][0]['chains'] == 5
    assert result['born_records'][0]['complement_identity_max_error'] < 1e-14
    assert result['born_records'][0]['region']['nn_cut_bonds'] > 0
    disk = json.loads((registered.parent/'output/results.json').read_text())
    assert disk.pop('result_sha256') == canonical_sha256(disk)
    with pytest.raises(FileExistsError):
        workflow.analyze(registered)


def test_changed_input_is_not_silently_used(registered):
    with (registered.parent/'series.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError,match='input seal changed'):
        workflow.analyze(registered)
    assert not (registered.parent/'output').exists()
