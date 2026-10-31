"""Independent full-space operator audit and matched-work L16 design checks."""
from __future__ import annotations
import json
import jax
import numpy as np
import pytest
from fnqs.basis import all_bits
from fnqs.diagnostics.finite_chain import finite_bits,vector_action,full_correlations
from fnqs.hamiltonians import dense_hamiltonian,IsingCouplings
from fnqs.scripts.run_l16_pair import matched_schedules,prepare,audit
from fnqs.pilots.engine import init_model
from fnqs.variants.size_features import SizeFeatureConfig


@pytest.mark.parametrize('n',(4,6,8,10))
def test_bounded_action_and_correlations_against_independent_pauli(n):
    bits=finite_bits(n);np.testing.assert_array_equal(bits,all_bits(n))
    rng=np.random.default_rng(n+71);psi=rng.normal(size=1<<n)+1j*rng.normal(size=1<<n);psi/=np.linalg.norm(psi)
    for g in np.eye(4):
        np.testing.assert_allclose(vector_action(psi,n,g),dense_hamiltonian(n,IsingCouplings(*g))@psi,atol=2e-11,rtol=1e-13)
    corr=full_correlations(psi,bits)
    for r in range(n):
        labels=np.arange(1<<n);expected=0.
        for site in range(n):
            a=1-2*((labels>>(n-1-site))&1);b=1-2*((labels>>(n-1-(site+r)%n))&1)
            expected+=np.sum(abs(psi)**2*a*b)/n
        np.testing.assert_allclose(corr[r],expected,atol=1e-12)


def test_L16_extension_is_explicit_and_old_guard_remains():
    bits=finite_bits(16)
    assert bits.shape==(65536,16) and np.all(bits[0]==0) and np.all(bits[-1]==1)
    with pytest.raises(ValueError):all_bits(16)
    for n in (True,3,17,18):
        with pytest.raises(ValueError):finite_bits(n)
    with pytest.raises(ValueError):vector_action(np.ones(16),4,[1.,0.,0.,np.nan])


def test_matched_schedule_and_frozen_feature_protocol(tmp_path):
    output=tmp_path/'pair';prepare(output,rounds=2)
    d=json.loads((output/'protocol.json').read_text())
    assert d['model']==dict(hidden_dim=16,geometry_scale=64.,size_features='legacy')
    assert d['options']['response_weight']==0 and d['options']['shift']==.001
    for seed in d['seeds']:
        a,b=[r for arm in ('sequential','joint') for r in d['runs'] if r['seed']==seed and r['arm']==arm]
        assert [g for row in a['schedule'] for g in row]==[g for row in b['schedule'] for g in row]
        assert len(a['schedule'])==42 and len(b['schedule'])==2
        for key in ('born_samples','hamiltonian_actions','source_actions','task_evaluations'):
            assert a['cost'][key]==b['cost'][key]
        assert a['parent']==b['parent'] and a['parent_sha256']==b['parent_sha256']
    assert not d['reserved_confirmation_measured']
    with pytest.raises(FileExistsError):prepare(output,2)
    with pytest.raises(ValueError):matched_schedules([[1.,0.,0.,1.]],0)


def test_audit_chunking_JW_dense_energy_and_input_FD(tmp_path):
    cfg=SizeFeatureConfig(3,size_features='legacy');p=init_model(jax.random.PRNGKey(41),cfg)
    path=tmp_path/'raw.npz';g=[1.,0.,0.,1.03]
    d=audit(p,4,g,cfg,path,chunk=8)
    with np.load(path,allow_pickle=False) as raw:
        psi=raw['psi'];s=raw['score']
        np.testing.assert_allclose(d['energy_total'],psi@dense_hamiltonian(4,IsingCouplings(*g))@psi,atol=2e-12)
        np.testing.assert_allclose(d['chi_total'],np.sum(psi**2*s**2)-np.sum(psi**2*s)**2,atol=1e-12)
    assert d['normalization_error']<1e-12
    assert d['fd_score_relative_rms_error'][1]<d['fd_score_relative_rms_error'][0]
