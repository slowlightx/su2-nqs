"""Dense independent H oracles; gauge, Q, units and saved-span invariance."""
import numpy as np
import pytest
from su2_nqs.vmc.residual_curvature import residual_response_gram,residual_curvature_direction


@pytest.mark.parametrize('complete',[False,True])
def test_response_gram_uses_full_complex_energy_derivative_and_retains_nodes(complete):
    rng=np.random.default_rng(18);a=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8));h=a+a.conj().T
    p=rng.normal(size=8)+1j*rng.normal(size=8);p[0]=0
    j=rng.normal(size=(8,3))+1j*rng.normal(size=(8,3));keep=np.arange(8) if complete else np.array([0,2,4,7])
    q=np.ones(len(keep)) if complete else np.array([.3,.8,2.,.7])
    x,hx,d,hd=p[keep],(h@p)[keep],j[keep],(h@j)[keep]
    fs,r,info=residual_response_gram(x,hx,d,hd,q,8)
    z=np.sum(abs(x)**2/q);beta=np.sum(x.conj()[:,None]*d/q[:,None],axis=0)/z
    t,ht=d-x[:,None]*beta,hd-hx[:,None]*beta
    def residual(c):
        psi=x+t@c;hp=hx+ht@c;norm=np.sum(abs(psi)**2/q)
        e=np.sum(psi.conj()*hp/q)/norm
        return hp-e*psi,e
    k=[];derivatives=[]
    for i in range(3):
        step=np.eye(3)[i]*1e-6;plus,ep=residual(step);minus,em=residual(-step)
        k.append((plus-minus)/2e-6);derivatives.append((ep-em).real/2e-6/8)
    k=np.stack(k,axis=1)
    np.testing.assert_allclose(r,(k.conj().T@(k/q[:,None])).real/z/8,atol=2e-9,rtol=2e-9)
    np.testing.assert_allclose(info['energy_slopes_per_site'],derivatives,atol=2e-9)
    assert np.linalg.eigvalsh(r).min()>-1e-12 and np.linalg.eigvalsh(fs).min()>-1e-12
    # A constant shift H -> H + cI must not change physical residual response.
    _,shifted,_=residual_response_gram(x,hx+7.1*x,d,hd+7.1*d,q,8)
    np.testing.assert_allclose(shifted,r,atol=2e-13)
    gauges=np.array([.4+.7j,-1.3j,1.1])
    f2,r2,_=residual_response_gram(x,hx,d+x[:,None]*gauges,hd+hx[:,None]*gauges,9*q,8)
    np.testing.assert_allclose(f2,fs,atol=3e-14);np.testing.assert_allclose(r2,r,atol=3e-13)


def test_residual_trust_changes_direction_not_just_step_and_is_basis_and_unit_invariant():
    p=np.array([1.,0,0,0]);hp=np.array([0.,.2,.2,0.])
    d=np.array([[0.,0],[1.,0],[0.,1],[0.,0]])
    hd=np.array([[.2,.2],[1.,0],[0.,12.],[0.,0]])
    b=np.eye(2);g=np.array([[-1.,-1.],[-2.,-1.],[-1.,-2.],[-.2,-.4],[-.3,-.4],[-.2,-.3]])
    masks=[np.ones(4,bool)];names=[str(i) for i in range(6)]
    c,info=residual_curvature_direction(p,hp,d,hd,b,np.ones(4),masks,g,names,4)
    assert info['common_descent'] and np.all(g@c<0) and np.linalg.norm(c)==pytest.approx(1.)
    assert c[0]>c[1]*3 # The high residual-action second direction is discouraged.
    assert info['relative_linear_residual']<1e-12 and info['simplex_kkt_scaled_violation']<1e-10
    u=np.array([[2.,.3],[-.4,1.2]])
    cu,_=residual_curvature_direction(p,hp,d@u,hd@u,b@u,np.ones(4),masks,
        (g@u)*np.array([.01,2,30,.4,7,100])[:,None],names,4)
    np.testing.assert_allclose(u@cu,c,atol=3e-12)
    opposite,stop=residual_curvature_direction(p,hp,d,hd,b,np.ones(4),masks,
        np.array([[1.,0],[-1.,0]]),['E','R'],4)
    assert not stop['common_descent'];np.testing.assert_allclose(opposite,0,atol=1e-12)


def test_invalid_response_and_trust_contracts_fail():
    p=np.ones(3);d=np.ones((3,2));m=[np.ones(3,bool)]
    with pytest.raises(ValueError):residual_response_gram(p,p,d,d,np.zeros(3),3)
    with pytest.raises(ValueError):residual_response_gram(p,p,d,d[:2],np.ones(3),3)
    with pytest.raises(ValueError):residual_curvature_direction(p,p,d,d,np.eye(2),np.ones(3),m,
        np.ones((2,2)),['E','R'],3,response_weight=0)
