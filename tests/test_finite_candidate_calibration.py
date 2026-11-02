import numpy as np
import pytest
from su2_nqs.vmc.finite_candidate_calibration import (
    common_proposal_FS,select_finite_grid,bilateral_direction_moments,
)


def test_nonlinear_finite_candidates_keep_parent_nodes_and_exact_quadrature():
    # Two changing local factors: the t² amplitude is absent from parent/tangent.
    t=.04;parent=np.array([1.,0.,0.,0.]);candidate=np.array([1.,t,t,t*t])
    candidate/=np.linalg.norm(candidate)
    assert candidate[-1]>0
    pair=np.column_stack((parent,candidate));q=np.ones(4)/4
    # Uniform exhaustive quadrature, repeated independent labels only for arithmetic.
    result=common_proposal_FS(np.tile(pair,(4,1)),np.tile(q,4),groups=4)
    exact=np.arccos(abs(parent@candidate))
    assert result['FS_radians']==pytest.approx(exact,abs=3e-15)
    assert result['candidate_mass_at_parent_nodes']>0
    assert result['parent_node_records']==12


def test_finite_grid_guard_cannot_reselect_and_accepts_complex_scale():
    grid=np.array([1.,.5,.25,.125,0.]);x=np.linspace(-1,1,32)
    base=np.ones(32,dtype=complex)
    # Registered actual amplitudes include nonlinear t² terms.
    p=np.column_stack([base+1j*.001*s*x+.0001*s*s*x*x for s in grid])
    phi=np.tile(p,(4,1));q=np.linspace(.1,.7,len(phi))
    i,report=select_finite_grid(phi,q,grid,split=64,groups=4)
    assert i==0 and report['accepted']
    other,_=select_finite_grid(phi*(.3+.7j),q*3,grid,split=64,groups=4)
    assert other==i
    broken=phi.copy();broken[64:,:-1]=1+20*(broken[64:,:-1]-1)
    selected,failed=select_finite_grid(broken,q,grid,split=64,groups=4)
    assert selected==4 and failed['proposed_index']==0
    assert not failed['guard_used_to_select_smaller_fraction']


def test_complex_nonhermitian_bilateral_common_Q_matches_finite_difference():
    rng=np.random.default_rng(271201);p=rng.normal(size=20)+1j*rng.normal(size=20)
    d=rng.normal(size=20)+1j*rng.normal(size=20);p[0]=0
    h=rng.normal(size=20)+1j*rng.normal(size=20)
    hd=rng.normal(size=20)+1j*rng.normal(size=20);q=rng.uniform(.1,1,20)
    result=bilateral_direction_moments(p,h,d,hd,q,groups=4,n_sites=6)
    def energy(t):
        a=p+t*d;b=h+t*hd
        return (np.vdot(a/np.sqrt(q),b/np.sqrt(q))/np.vdot(a/np.sqrt(q),a/np.sqrt(q))).real/6
    eps=1e-6;finite=(energy(eps)-energy(-eps))/(2*eps)
    assert result['values'][2]==pytest.approx(finite,abs=3e-11)
    assert result['values'][0]+result['values'][1]==pytest.approx(result['values'][2],abs=1e-16)
    assert result['first_order_node_safe'] and not result['global_tangent_metric_certified']
    with pytest.raises(ValueError):bilateral_direction_moments(p,h,d,hd,np.zeros(20),groups=4)
