import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.su2_frozen_coupling_series_local import (
    SITES,iid_cached_panel,panel_summary)


def test_missing_deletions_do_not_invent_errors():
    names=('T','T0','T2','kappa','P110','P112','Ptotal0','Pstar','delta03_Pstar')
    values=np.array([.1,.2,-.1,-.525,.3,.4,.5,.2,-.05])
    result=panel_summary(values,None,names)
    assert not result['SE_available'] and 'jackknife_SE' not in result
    assert result['tau_q0']['value']==pytest.approx(2/3)
    leaves=np.repeat(values[None,:],4,axis=0)
    exact=panel_summary(values,leaves,names)
    assert max(exact['jackknife_SE'].values())==0
    with pytest.raises(ValueError):panel_summary(values,leaves[:3],names)


def test_iid_cache_matches_independent_exhaustive_physical_state():
    rng=np.random.default_rng(2026102803)
    bits=((np.arange(64)[:,None]>>np.arange(5,-1,-1))&1)
    selected=bits.sum(1)==3
    local=bits[selected]
    rows=np.zeros((len(local),36),int);rows[:,SITES]=local
    vector=np.zeros(64,complex)
    vector[selected]=rng.normal(size=len(local))+1j*rng.normal(size=len(local))
    vector[np.flatnonzero(selected)[0]]=0 # exact target node is retained
    vector/=np.linalg.norm(vector)
    phi=np.repeat(vector[None,:],len(rows),axis=0)
    x=rng.normal(size=(64,64))+1j*rng.normal(size=(64,64))
    operators=np.stack([np.eye(64),(x+x.conj().T)/2])
    point,leave=iid_cached_panel(rows,phi,np.ones(len(rows))/len(rows),operators,groups=4)
    expected=np.array([np.vdot(vector,op@vector).real for op in operators])
    np.testing.assert_allclose(point,expected,atol=2e-14)
    assert leave.shape==(4,2) and np.isfinite(leave).all()
    with pytest.raises(ValueError):iid_cached_panel(rows,phi,np.ones(len(rows)),operators,groups=True)
    with pytest.raises(ValueError):iid_cached_panel(rows,phi,np.zeros(len(rows)),operators,groups=4)
