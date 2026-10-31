import numpy as np
import pytest
from su2_nqs.vmc.correlated_audit import correlated_summary,paired_chain_intervals


def test_reweighted_complex_moments_overlap_and_kl_match_exact_state():
    rng=np.random.default_rng(8);a=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8));h=(a+a.conj().T)/2
    p=rng.normal(size=8)+1j*rng.normal(size=8);c=rng.normal(size=8)+1j*rng.normal(size=8)
    q=abs(p)**2/np.vdot(p,p).real;qc=abs(c)**2/np.vdot(c,c).real
    r=correlated_summary(np.log(c/p),h@p/p,h@c/c,n_sites=4,reference_weights=q)
    mean=np.vdot(c,h@c)/np.vdot(c,c);v=np.vdot(h@c,h@c)/np.vdot(c,c)-abs(mean)**2
    assert r['candidate_energy_per_site']==pytest.approx(mean.real/4,abs=2e-13)
    assert r['candidate_variance_per_site']==pytest.approx(v.real/4,abs=2e-13)
    fs=np.sqrt(1-abs(np.vdot(p,c))**2/(np.vdot(p,p).real*np.vdot(c,c).real))
    assert r['sampled_fs_sine']==pytest.approx(fs,abs=2e-13)
    assert r['sampled_forward_kl']==pytest.approx(np.sum(q*np.log(q/qc)),abs=2e-13)
    shifted=correlated_summary(np.log(c/p)+1000+2j,h@p/p,h@c/c,n_sites=4,reference_weights=q)
    for k in r:
        if k!='log_norm_ratio_estimate':assert shifted[k]==pytest.approx(r[k],abs=2e-11)


def test_paired_bootstrap_resamples_chains_not_snapshots():
    rng=np.random.default_rng(4);old=rng.normal(size=(2,8,6))+1j*rng.normal(size=(2,8,6));ratio=.05*rng.normal(size=old.shape)
    new=old+.04;zero=paired_chain_intervals(np.zeros_like(old),old,new,n_sites=4,seed=7,replicates=128)
    assert zero['energy_per_site']['difference']==pytest.approx(.01)
    assert zero['energy_per_site']['bootstrap_se']<1e-15
    assert zero['variance_per_site']['bootstrap_se']<1e-15
    a=paired_chain_intervals(ratio,old,new,n_sites=4,seed=7,replicates=128)
    b=paired_chain_intervals(*(np.repeat(x,2,axis=1) for x in (ratio,old,new)),n_sites=4,seed=7,replicates=128)
    for key in a:
        for field in a[key]:np.testing.assert_allclose(a[key][field],b[key][field],atol=2e-14)
    assert a['energy_per_site']['bootstrap_se']>0


def test_identity_and_invalid_audits():
    e=np.array([1+2j,-1j,.3]);r=correlated_summary(np.zeros(3),e,e,n_sites=2)
    assert r['delta_energy_per_site']==r['delta_variance_per_site']==0
    assert r['sampled_fs_sine']==0 and r['reweighting_ess_fraction']==1
    with pytest.raises(ValueError):correlated_summary([np.nan],e,e,n_sites=2)
    with pytest.raises(ValueError):correlated_summary(np.zeros(3),e,e,n_sites=2,reference_weights=[1,-1,1])
    with pytest.raises(ValueError):paired_chain_intervals(e,e,e,n_sites=2,seed=7)
