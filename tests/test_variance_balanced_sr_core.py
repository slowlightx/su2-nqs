"""Self-contained exact oracles for complex energy/residual natural updates."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest
from su2_nqs.vmc.variance_balanced_sr import (
    product_cotangents, solve_balanced_forces, make_variance_balanced_update,
    residual_augmented_metric_rows,
)


def _physical_H(n, bonds):
    # Independent Sz-basis Heisenberg matrix, site zero most significant.
    h = np.zeros((2**n, 2**n), dtype=np.float64)
    for old in range(2**n):
        for i, j, coupling in bonds:
            a = (old >> (n - 1 - i)) & 1
            b = (old >> (n - 1 - j)) & 1
            h[old, old] += coupling * (0.25 if a == b else -0.25)
            if a != b:
                new = old ^ (1 << (n - 1 - i)) ^ (1 << (n - 1 - j))
                h[new, old] += 0.5 * coupling
    return h

def test_full_product_cotangents_match_direct_physical_singlet_finite_differences():
 from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
 from su2_nqs.su2.fusion_path import enumerate_paths
 u=coupled_basis_matrix(enumerate_paths(6,0),0,0);h=_physical_H(6,((0,1,1.),(1,2,.7),(2,3,1.1),(3,4,.9),(4,5,.8),(5,0,1.),(0,3,.125)))
 r=np.random.default_rng(8181);p=u@(r.normal(size=5)+1j*r.normal(size=5));d=u@(r.normal(size=(5,3))+1j*r.normal(size=(5,3)));q=r.uniform(.3,2,64)
 prod=np.column_stack([p,h@p]);ce,cv,point=product_cotangents(jnp.asarray(prod),jnp.asarray(q));jac=np.stack([d,h@d],axis=1)
 actual=np.array([np.real(np.einsum('nk,nkp->p',np.asarray(c),jac)) for c in (ce,cv)])
 def moments(theta):
  a=p+d@theta;b=h@a;z=(abs(a)**2/q).sum();e=np.real(np.sum(a.conj()*b/q))/z;v=(abs(b-e*a)**2/q).sum()/z
  return np.array([e,v])
 eps=1e-6;finite=np.column_stack([(moments(eps*np.eye(3)[k])-moments(-eps*np.eye(3)[k]))/(2*eps) for k in range(3)])
 np.testing.assert_allclose(actual,finite,atol=5e-10,rtol=2e-8)
 shifted=prod.copy();shifted[:,1]+=13*shifted[:,0];sce,scv,_=product_cotangents(jnp.asarray(shifted),jnp.asarray(q));sjac=jac.copy();sjac[:,1]+=13*sjac[:,0]
 shifted_g=np.array([np.real(np.einsum('nk,nkp->p',np.asarray(c),sjac)) for c in (sce,scv)])
 np.testing.assert_allclose(shifted_g,actual,atol=3e-13,rtol=1e-11)


def test_streamed_joint_update_matches_dense_autodiff_and_preserves_source():
 r=np.random.default_rng(9187);n=8;size=5;mat=r.normal(size=(n,size))+1j*r.normal(size=(n,size));fixed=r.normal(size=n)+1j*r.normal(size=n);h=r.normal(size=(n,n))+1j*r.normal(size=(n,n));h=(h+h.conj().T)/2
 m=jnp.asarray(mat);h=jnp.asarray(h);fixed=jnp.asarray(fixed);q=jnp.asarray(r.uniform(.2,2,n));params=jnp.asarray(r.normal(size=size));rows=jnp.arange(n)[:,None]
 def prepare(x):return 2*x
 def amp(bank,row):return fixed[row[0]]+m[row[0]]@(bank/2)
 def products(bank,rs,source,reference):
  p=fixed+m@(bank/2);return jnp.stack([p[rs[:,0]],(h@p)[rs[:,0]]],axis=1)
 parent=products(prepare(params),rows,None,None);moments=(jnp.asarray(0),jnp.zeros(size),jnp.zeros(size))
 update=make_variance_balanced_update(params,prepare=prepare,prepared_amplitude=amp,prepared_products=products,damping=.2,tangent_fs=.001,action_chunk=2)
 actual,mo,info=update(params,moments,rows,parent,q,jnp.zeros((n,2),complex),jnp.zeros((n,2),complex),.01)
 def loss(theta):
  p=fixed+m@theta;b=h@p;z=jnp.sum(abs(p)**2/q);e=jnp.real(jnp.sum(p.conj()*b/q))/z;v=jnp.sum(abs(b-e*p)**2/q)/z;return jnp.stack([e,v])
 ge=np.asarray(jax.jacrev(loss)(params));direction,di=solve_balanced_forces(m,parent[:,0],q,jnp.asarray(ge),damping=.2)
 scale=min(.01,.001/float(di['raw_fisher_norm']));expected=params-scale*direction
 np.testing.assert_allclose(actual,expected,atol=3e-13,rtol=3e-12);assert info['accepted'] and int(mo[0])==1
 assert info['applied_energy_directional_derivative']<0 and info['applied_variance_directional_derivative']<0
 np.testing.assert_array_equal(mo[1],moments[1]);assert info['full_H_replay_relative_error']==0


@pytest.mark.parametrize('chunk',[1,7,32])
def test_empirical_support_filter_matches_independent_dense_spectral_matrix(chunk):
 r=np.random.default_rng(881108);j=r.normal(size=(4,17))+1j*r.normal(size=(4,17));p=r.normal(size=4)+1j*r.normal(size=4);q=r.uniform(.2,2,4);g=r.normal(size=(2,17));ridge=.2
 z=(abs(p)**2/q).sum();u=p/np.sqrt(q*z);a=j/np.sqrt(q*z)[:,None];a-=np.outer(u,u.conj()@a);f=(a.conj().T@a).real
 val,vec=np.linalg.eigh(f);val=np.maximum(val,0.);filtered=(vec*(val/(val+ridge)**2))@vec.T
 d=(filtered@g.T).T;norm=np.sqrt(np.sum(g*d,axis=1));expected=(d/norm[:,None]).sum(axis=0)
 actual,info=solve_balanced_forces(*map(jnp.asarray,(j,p,q,g)),damping=ridge,parameter_chunk_size=chunk,support_filter=True)
 np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=2e-12);assert info['numerically_valid'] and info['empirical_support_filter'];assert not info['arbitrary_force_outside_sampled_rowspace_retained']
 assert info['relative_linear_residual']<1e-12


def test_residual_metric_matches_independent_dense_physical_singlet_and_energy_shift():
 from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
 from su2_nqs.su2.fusion_path import enumerate_paths
 u=coupled_basis_matrix(enumerate_paths(6,0),0,0).astype(complex)
 h=_physical_H(6,((0,1,1.),(1,2,.7),(2,3,1.1),(3,4,.9),(4,5,.8),(5,0,1.),(0,3,.125)))
 r=np.random.default_rng(88110813);j=u@(r.normal(size=(5,5))+1j*r.normal(size=(5,5)))
 c=r.normal(size=5)+1j*r.normal(size=5);p=j@c;hp=h@p;hj=h@j;q=r.uniform(.2,2,64)
 jj,pp,qq,note=residual_augmented_metric_rows(*map(jnp.asarray,(j,hj,p,hp,q)),weight=.7)
 z=np.sum(abs(p)**2/q);e=np.sum((p.conj()*hp).real/q)/z;res=hp-e*p
 alpha=(p.conj()/q)@j/z;b=hj-e*j-res[:,None]*alpha[None,:]
 width2=np.sum(abs(res)**2/q)/z;a=j-np.outer(p,alpha)
 expected=((a.conj().T/q)@a+.7*(b.conj().T/q)@b/width2)/z
 aa=np.asarray(jj)-np.outer(pp,(np.asarray(pp).conj()/qq)@jj/z)
 actual=(aa.conj().T/qq)@aa/z
 np.testing.assert_allclose(actual,expected,atol=3e-13,rtol=2e-12)
 np.testing.assert_allclose(np.asarray(jj)[64:]@c,0,atol=2e-13)
 shift=residual_augmented_metric_rows(*map(jnp.asarray,(j,hj+17*j,p,hp+17*p,q)),weight=.7)
 np.testing.assert_allclose(shift[0],jj,atol=2e-13,rtol=2e-12)
 assert np.linalg.eigvalsh(np.asarray(actual)).min()>-2e-13 and note['not_exact_energy_Hessian']


def test_residual_geometry_update_uses_physical_FS_not_augmented_norm():
 r=np.random.default_rng(811082);n=8;size=4;m=jnp.asarray(r.normal(size=(n,size))+1j*r.normal(size=(n,size)))
 fixed=jnp.asarray(r.normal(size=n)+1j*r.normal(size=n));h=r.normal(size=(n,n))+1j*r.normal(size=(n,n));h=jnp.asarray((h+h.conj().T)/2)
 params=jnp.asarray(r.normal(size=size));rows=jnp.arange(n)[:,None];q=jnp.ones(n);mom=(jnp.array(0),jnp.zeros(size),jnp.zeros(size))
 def amp(x,row):return fixed[row[0]]+m[row[0]]@x
 def products(x,rs,s,ref):
  p=fixed+m@x;return jnp.stack([p[rs[:,0]],(h@p)[rs[:,0]]],1)
 parent=products(params,rows,None,None)
 update=make_variance_balanced_update(params,prepare=lambda x:x,prepared_amplitude=amp,prepared_products=products,
  residual_metric_weight=1.,tangent_fs=.001,action_chunk=2,support_filter=True)
 got,_,info=update(params,mom,rows,parent,q,jnp.zeros((n,2)),jnp.zeros((n,2)),1.)
 d=np.asarray(m@(got-params));p=np.asarray(parent[:,0]);ray=np.vdot(p,d)/np.vdot(p,p)
 physical=np.linalg.norm(d-ray*p)/np.linalg.norm(p)
 np.testing.assert_allclose(physical,.001,atol=3e-14,rtol=3e-11)
 assert info['accepted'] and info['full_H_derivative_metric']
 assert info['applied_energy_directional_derivative']<0 and info['applied_variance_directional_derivative']<0
