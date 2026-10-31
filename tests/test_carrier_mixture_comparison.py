"""Frozen nested-span comparison, with no favorable-outcome assumptions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import spin_carrier_mixture_comparison as workflow
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256


def _spec():
    return json.loads(workflow.CONFIG.read_text())


def test_protocol_defines_complex_nested_spans_and_one_primary_before_audit():
    spec=_spec(); parent=json.loads((workflow.ROOT/spec['parent_config']).read_text())
    assert spec['n_sites']==36 and spec['audit_blocks']*parent['chains']==128
    assert spec['subspaces']=={'old_mixture':[0,1], 'network_diagonal':[0,1,2], 'network_control':[0,1,3]}
    assert spec['primary_contrast']=='network_control-minus-old_mixture'
    assert spec['secondary_contrast']=='network_diagonal-minus-old_mixture'
    assert spec['audit_seed'] not in [parent[k] for k in parent if k.endswith('_seed')]
    assert spec['audit_seed']!=spec['bootstrap_seed']
    assert spec['relative_rank_tolerance']==1e-10
    assert spec['maximum_condition_number']==1e8
    workflow.validate_protocol(spec,parent)
    for key,value in (('n_sites',64),('audit_seed',parent['audit_seed']),
                      ('primary_contrast','network_diagonal-minus-old_mixture'),('audit_blocks',16)):
        with pytest.raises(ValueError):workflow.validate_protocol({**spec,key:value},parent)


def test_atomic_checkpoint_failure_preserves_previous_numeric_archive(tmp_path,monkeypatch):
    target=tmp_path/'partial.npz'
    workflow._atomic_npz(target,amplitudes=np.arange(4),metadata=np.asarray('fixed'))
    before=file_sha256(target)
    def interrupted(handle,**arrays):
        handle.write(b'incomplete archive')
        raise OSError('simulated interrupted write')
    monkeypatch.setattr(workflow.np,'savez_compressed',interrupted)
    with pytest.raises(OSError):workflow._atomic_npz(target,amplitudes=np.arange(8))
    assert file_sha256(target)==before
    assert sorted(p.name for p in tmp_path.iterdir())==['partial.npz']
    np.testing.assert_array_equal(np.load(target,allow_pickle=False)['amplitudes'],np.arange(4))


def test_saved_basis_has_consistent_total_H_and_never_drops_tail():
    spec=_spec(); parent=workflow.ROOT/spec['parent']
    basis,action,data=workflow.saved_development_basis(parent)
    assert basis.shape==action.shape==(128,4)
    np.testing.assert_array_equal(basis[:,0],data['carrier'])
    np.testing.assert_allclose(action[:,0],36*data['hcarrier'],atol=1e-15)
    np.testing.assert_array_equal(basis[:,2:],data['psi'][:,1:])
    assert len(np.unique(data['groups']))==16
    # Keep the previously observed dominant row, not a filtered fit corpus.
    assert data['groups'][123]==11


def test_fit_is_array_only_globally_optimal_and_retains_diagnostic_LOCO(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('fitting saved arrays must not evaluate physics')
    for name in ('setup','draw','make_amplitude_heisenberg_action'):
        monkeypatch.setattr(workflow,name,forbidden)
    spec=_spec()
    result=workflow.fit_saved_subspaces(spec,tmp_path,{'registration_sha256':'test-only'})
    assert result['viewed_audit_reclassified_as_development']
    assert not result['network_retrained'] and not result['full_nonlinear_network_optimum_claim']
    assert result['new_physical_calls']==0
    assert result['coefficients_selected_without_new_audit']
    assert not result['production_authorized'] and not result['n64_authorized']
    assert not (tmp_path/'samples.npz').exists()
    raw=dict(np.load(tmp_path/'coefficients.npz',allow_pickle=False))
    assert raw['coefficients'].shape==(4,6)
    assert file_sha256(tmp_path/'coefficients.npz')==result['coefficients_sha256']
    basis,action,data=workflow.saved_development_basis(workflow.ROOT/spec['parent'])
    for arm,indices in spec['subspaces'].items():
        fit=result['fits'][arm];column=raw['coefficients'][:,spec['arm_names'].index(arm)]
        assert fit['rank']==len(indices) and not fit['rank_truncated'] and not fit['ridge_added']
        assert fit['complex_coefficients'] and fit['statistics']['relative_optimality_residual']<1e-12
        excluded=np.setdiff1d(np.arange(4),indices)
        np.testing.assert_array_equal(column[excluded],np.zeros(len(excluded)))
        state=basis@column;hstate=action@column
        norm=np.sum(abs(state)**2/data['q'])
        energy=np.sum(state.conj()*hstate/data['q'])/norm
        np.testing.assert_allclose(fit['eigenvalues_total'][0],energy.real,atol=1e-10)
        records=result['leave_one_chain_out'][arm]
        assert not records['used_for_selection'] and len(records['records'])==16
        assert all(row['passed'] for row in records['records'])
    null=result['development_moments']['old_mixture']['energy_per_site']
    for name in ('network_diagonal','network_control'):
        assert result['development_moments'][name]['energy_per_site']<=null+1e-12


def test_completed_mixture_audit_seals_and_fixed_linear_H_replay():
    spec=_spec();folder=workflow.ROOT/spec['output']
    reg=json.loads((folder/'registration.json').read_text())
    assert reg['registration_sha256']==canonical_sha256({k:v for k,v in reg.items() if k!='registration_sha256'})
    for section in ('inputs','sources'):
        for name,digest in reg[section].items():assert file_sha256(workflow.ROOT/name)==digest,name
    fit=json.loads((folder/'fit/results.json').read_text())
    audit=json.loads((folder/'audit/results.json').read_text())
    inputs=json.loads((folder/'audit/inputs.json').read_text())
    assert inputs['fit_sha256']==file_sha256(folder/'fit/results.json')
    assert inputs['coefficients_sha256']==fit['coefficients_sha256']==file_sha256(folder/'fit/coefficients.npz')
    assert audit['fit_seal']==inputs
    assert not audit['coefficients_refit_on_audit'] and audit['no_best_arm_selection']
    assert not audit['network_retrained'] and not audit['production_authorized'] and not audit['n64_authorized']
    assert audit['independent_chains']==16 and audit['sample_count']==128
    assert audit['primary_contrast']==spec['primary_contrast']
    assert audit['unique_residual_H_centers']==384 and audit['shared_carrier_H_centers']==128
    for name,digest in audit['files'].items():assert file_sha256(folder/'audit'/name)==digest,name
    raw=dict(np.load(folder/'audit/values.npz',allow_pickle=False))
    coefficients=np.load(folder/'fit/coefficients.npz',allow_pickle=False)['coefficients']
    np.testing.assert_array_equal(raw['coefficients'],coefficients)
    np.testing.assert_allclose(raw['psi'],raw['basis']@coefficients,atol=1e-15)
    np.testing.assert_allclose(raw['hpsi'],raw['h_basis']@coefficients,atol=1e-15)
    for k,name in enumerate(spec['arm_names']):
        psi,hp,q=raw['psi'][:,k],raw['hpsi'][:,k],raw['q']
        norm=np.sum(abs(psi)**2/q);energy=np.sum(psi.conj()*hp/q)/norm
        variance=np.sum(abs(hp-energy*psi)**2/q)/norm/36
        np.testing.assert_allclose(audit['arms'][name]['energy_per_site']['value'],energy.real/36,atol=1e-13)
        np.testing.assert_allclose(audit['arms'][name]['variance_per_site']['value'],variance,atol=1e-12)
