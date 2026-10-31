"""Independent physical-matrix double commutators and exact gap checks."""
import numpy as np
import pytest

from su2_nqs.benchmarks.spin_oscillator_strength import (
    scalar_oscillator_strength, longitudinal_residual_bound,
)


def matrices(n):
    singles = [np.array([[0.,.5],[.5,0.]]),
               np.array([[0.,.5j],[-.5j,0.]]), np.diag([-.5,.5])]
    spins = []
    for site in range(n):
        axes = []
        for single in singles:
            op = np.array([[1.]],complex)
            for i in range(n):
                op = np.kron(op, single if i == site else np.eye(2))
            axes.append(op)
        spins.append(axes)
    return spins


@pytest.mark.parametrize('n', [2,4])
def test_scalar_sum_rule_without_SU2_assumption(n):
    rng = np.random.default_rng(260612133+n)
    v = rng.normal(size=(2**n,3))+1j*rng.normal(size=(2**n,3))
    rho=v@v.conj().T; rho/=np.trace(rho)
    spins=matrices(n)
    bonds=[(i,i+1,.7+.13*i) for i in range(n-1)]
    h=sum(j*sum(spins[a][axis]@spins[b][axis] for axis in range(3)) for a,b,j in bonds)
    c=np.array([[np.trace(rho@sum(spins[a][axis]@spins[b][axis]
                 for axis in range(3))).real for b in range(n)] for a in range(n)])
    phases=np.exp(1j*np.linspace(.2,2.4,n))
    report=scalar_oscillator_strength(c,bonds,phases)
    direct_f=direct_s=0.
    for axis in range(3):
        o=sum(phases[i]*spins[i][axis] for i in range(n))/np.sqrt(n)
        comm=h@o-o@h
        direct_f += .5*np.trace(rho@(o.conj().T@comm-comm@o.conj().T)).real
        direct_s += np.trace(rho@o.conj().T@o).real
    assert report['oscillator_strength']==pytest.approx(direct_f,abs=2e-13)
    assert report['structure_factor']==pytest.approx(direct_s,abs=2e-13)
    assert not report['strict_triplet_label'] and not report['new_spectral_information']


def test_two_site_exact_singlet_ratio_is_triplet_energy():
    c=np.array([[.75,-.75],[-.75,.75]])
    report=scalar_oscillator_strength(c,[(0,1,1.37)],np.array([1.,-1.]))
    assert report['static_ratio']==pytest.approx(1.37,abs=2e-14)
    # The helper itself never certifies the caller's state as a ground state.
    assert not report['exact_ground_state_gap_upper_bound']
    gamma=scalar_oscillator_strength(c,[(0,1,1.)],np.ones(2))
    assert gamma['static_ratio'] is None and gamma['spatially_constant_mode']


def test_parent_residual_cauchy_bound_for_actual_matrix_state():
    rng=np.random.default_rng(260612134)
    v=rng.normal(size=16)+1j*rng.normal(size=16); v/=np.linalg.norm(v)
    spins=matrices(4)
    h=sum(sum(spins[i][a]@spins[i+1][a] for a in range(3)) for i in range(3))
    o=sum(np.exp(.7j*i)*spins[i][2] for i in range(4))/2
    a=o.conj().T@o
    e=np.vdot(v,h@v).real; mean=np.vdot(v,a@v).real
    vh=np.vdot(v,h@h@v).real-e*e
    va=np.vdot(v,a@a@v).real-mean*mean
    covariance=np.vdot(v,(a-mean*np.eye(16))@(h-e*np.eye(16))@v)
    bound=longitudinal_residual_bound(vh,mean,va)
    assert abs(covariance)/mean <= bound+2e-14
    comm=h@o-o@h
    f=.5*np.vdot(v,(o.conj().T@comm-comm@o.conj().T)@v).real
    plus=np.vdot(o@v,(h-e*np.eye(16))@(o@v)).real
    minus=np.vdot(o.conj().T@v,(h-e*np.eye(16))@(o.conj().T@v)).real
    assert .5*(plus+minus)-f == pytest.approx(covariance.real,abs=2e-14)
    assert longitudinal_residual_bound(0.,mean,va)==0.


def test_chiral_eigen_parent_needs_plus_minus_momentum_symmetrization():
    # A three-spin Heisenberg triangle has a chiral q=1 ground doublet.
    # Even Var(H)=0 cannot make one O_k numerator equal its -k partner.
    spins=matrices(3)
    h=sum(sum(spins[i][a]@spins[j][a] for a in range(3))
          for i,j in [(0,1),(1,2),(0,2)])
    phase=np.exp(2j*np.pi*np.arange(3)/3)
    v=np.zeros(8,complex); v[[4,2,1]]=phase/np.sqrt(3)
    e=-.75
    np.testing.assert_allclose(h@v,e*v,atol=2e-14,rtol=0)
    o=sum(phase[i]*spins[i][2] for i in range(3))/np.sqrt(3)
    mean=np.vdot(v,o.conj().T@o@v).real
    comm=h@o-o@h
    f=.5*np.vdot(v,(o.conj().T@comm-comm@o.conj().T)@v).real/mean
    plus=np.vdot(o@v,(h-e*np.eye(8))@(o@v)).real/mean
    minus=np.vdot(o.conj().T@v,(h-e*np.eye(8))@(o.conj().T@v)).real/mean
    assert abs(plus-minus)>1.
    assert f==pytest.approx(.5*(plus+minus),abs=2e-14)
    assert longitudinal_residual_bound(0.,mean,1.)==0.


@pytest.mark.parametrize('args', [(-1.,1.,1.),(1.,0.,1.),(1.,1.,-1.),(np.inf,1.,1.)])
def test_invalid_variance_bound_not_repaired(args):
    with pytest.raises(ValueError):
        longitudinal_residual_bound(*args)


def test_invalid_spin_or_phase_data_not_repaired():
    with pytest.raises(ValueError):
        scalar_oscillator_strength(np.eye(2),[(0,1,1.)],np.ones(2))
    with pytest.raises(ValueError):
        scalar_oscillator_strength(.75*np.eye(2),[(True,1,1.)],np.ones(2))
    with pytest.raises(ValueError):
        scalar_oscillator_strength(.75*np.eye(2),[(0,1,1.)],np.array([np.inf,1.]))
