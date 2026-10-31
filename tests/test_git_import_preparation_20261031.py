"""Import safety gates use synthetic tokens only, never real credentials."""
import io
import json

import numpy as np
import pytest

from configs.git_import_20261031.prepare_source_import import (
    candidate_kind, denied_reason, external_fixture_risk, scan_bytes, scan_text)


def test_required_source_and_provenance_hashes_are_not_lost_as_secrets():
    for path in ('su2_nqs/vmc/local_energy.py',
                 'projects/clebsch_tree_nqs/SU2NQS_all.md',
                 'TRIANGULAR_J1J2_RESULTS.ipynb',
                 'configs/remote/example/runtime_packages.json',
                 'configs/remote/example/source_manifest.json'):
        assert candidate_kind(path)
    text=json.dumps({'sha256':'a'*64, 'runtime_packages':{'numpy':'2.2.4'}})
    assert not scan_text('configs/remote/example/source_manifest.json',text)


def test_credentials_and_scientific_artifacts_are_rejected_by_path_without_opening():
    for path in ('projects/clebsch_tree_nqs/results/checkpoint.pkl',
                 'projects/clebsch_tree_nqs/logs/SU2NQS_all.md',
                 '.aws/credentials', '.ssh/id_ed25519', '.codex/sessions/a.json',
                 '.agents/history.json', '.env', 'secrets.json', 'archive/a.py'):
        assert denied_reason(path)
        assert candidate_kind(path) is None
    assert candidate_kind('projects/clebsch_tree_nqs/initializations/g1_operator_vit_h_ring.npz')
    assert candidate_kind('projects/clebsch_tree_nqs/initializations/unreviewed.npz') is None


def test_synthetic_findings_never_include_values_or_context():
    synthetic='gh'+'p_'+('A'*40)
    text='api_'+'key = '+repr(synthetic)
    findings=scan_text('example.py',text)
    assert {x['rule'] for x in findings}=={'github_token','credential_assignment'}
    assert synthetic not in json.dumps(findings)
    assert all(set(x)=={'path','line','rule'} for x in findings)


def test_notebook_split_source_and_output_strings_are_scanned():
    first='gh'+'p_'+('B'*15);second='B'*25
    raw=json.dumps({'cells':[{'source':[first,second],
        'outputs':[{'text':[first,second]}]}]}).encode()
    findings,details=scan_bytes('TRIANGULAR_J1J2_RESULTS.ipynb',raw)
    assert any(x['rule']=='github_token' and 'notebook_string_field' in x for x in findings)
    assert details['notebook_cells']==1


def test_numeric_exception_forbids_object_payload_and_external_test_risk_is_explicit():
    path='projects/clebsch_tree_nqs/initializations/g1_operator_vit_h_ring.npz'
    blob=io.BytesIO();np.savez(blob,small=np.arange(4,dtype=np.float64))
    _,details=scan_bytes(path,blob.getvalue())
    assert details['numeric_arrays']['small']['shape']==[4]
    bad=io.BytesIO();np.savez(bad,untrusted=np.array([{'x':1}],dtype=object))
    with pytest.raises(ValueError):scan_bytes(path,bad.getvalue())
    assert external_fixture_risk('tests/test_replay.py', "p = ROOT / 'results' / 'state.pkl'")
    assert not external_fixture_risk('tests/test_exact.py', "def f():\n return 1\n")
