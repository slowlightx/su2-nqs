"""Independent complete-Hilbert checks, including exact parent nodes."""
from itertools import combinations, product

import numpy as np
import pytest

from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
from su2_nqs.benchmarks.spin_response_moments import (
    swap_moment_plan, transpose_born_samples, spin_source_moments,
)


def reference(psi, n, bonds, phases, shift=0.):
    h = sum((j*spin_dot_matrix(n,a,b) for a,b,j in bonds), np.zeros((2**n,2**n)))
    s2 = 3*n/4*np.eye(2**n)+2*sum((spin_dot_matrix(n,a,b)
          for a,b in combinations(range(n),2)), np.zeros_like(h))
    bits = (np.arange(2**n)[:,None]>>np.arange(n-1,-1,-1))&1
    out = {}
    for name, phases in [('parent',None), *phases.items()]:
        phi = psi if phases is None else psi*((bits-.5)@phases)/np.sqrt(n)
        hp = (h-shift*np.eye(len(h)))@phi; sp = s2@phi
        diagonal = np.ones(len(psi)) if phases is None else ((bits-.5)@phases)/np.sqrt(n)
        residual = diagonal*((h-shift*np.eye(len(h)))@psi); commutator = hp-residual
        out.update({name+'__norm':np.vdot(phi,phi).real,
                    name+'__first':np.vdot(phi,hp).real, name+'__first_imag':np.vdot(phi,hp).imag,
                    name+'__second':np.vdot(hp,hp).real, name+'__spin':np.vdot(phi,sp).real,
                    name+'__spin_defect':np.linalg.norm(sp-(0 if phases is None else 2)*phi)**2,
                    name+'__commutator_first':np.vdot(phi,commutator).real,
                    name+'__commutator_second':np.linalg.norm(commutator)**2,
                    name+'__parent_residual_first':np.vdot(phi,residual).real,
                    name+'__parent_residual_second':np.linalg.norm(residual)**2,
                    name+'__commutator_parent_cross':2*np.vdot(commutator,residual).real})
    return out


@pytest.mark.parametrize('kind', ['complex_nodes','single_configuration','singlet_dimers'])
def test_exact_moments_cover_nodes_and_correct_triplet(kind):
    n=6 if kind=='complex_nodes' else 4
    x=np.array([v for v in product((0,1),repeat=n) if sum(v)==n//2],np.int8)
    rng=np.random.default_rng(14102026)
    if kind=='complex_nodes':
        psi=rng.normal(size=len(x))+1j*rng.normal(size=len(x));psi[::2]=0
    elif kind=='single_configuration':
        psi=np.zeros(len(x),complex);psi[1]=1
    else:
        psi=np.ones(len(x),complex)
        for i,j in ((0,1),(2,3)):
            psi*=np.where(x[:,i]!=x[:,j],np.where(x[:,i]==0,1,-1)/np.sqrt(2),0)
    psi/=np.linalg.norm(psi)
    lookup={tuple(v):a for a,v in enumerate(x)}
    plan=swap_moment_plan(x)
    queried=psi[[lookup[tuple(v)] for v in plan.configurations]]
    q=(abs(queried[plan.indices])**2).mean(1)
    np.testing.assert_allclose(q.sum(),1,atol=3e-15)
    active=q>0;plan=swap_moment_plan(x[active])
    queried=psi[[lookup[tuple(v)] for v in plan.configurations]]
    with np.errstate(divide='ignore'): logs=np.log(queried.astype(complex))
    bonds=[(i,i+1,1.) for i in range(n-1)] if kind!='singlet_dimers' else [(0,1,1.),(2,3,1.)]
    phases={'source':np.exp(2j*np.pi*np.arange(n)/n),'staggered':(-1.)**np.arange(n)}
    measured=spin_source_moments(plan,logs,bonds,phases,energy_shift=-1.5)
    full=np.zeros(2**n,complex);full[x@(1<<np.arange(n-1,-1,-1))]=psi
    exact=reference(full,n,bonds,phases,-1.5)
    for name,v in measured.items():
        np.testing.assert_allclose(q[active]@v,exact[name],atol=3e-13,rtol=0)
    if kind=='single_configuration':
        # A Born-only norm-of-Hphi estimator loses strictly positive weight.
        assert np.any((abs(psi[active])==0)&(measured['parent__second']>0))
    if kind=='singlet_dimers':
        norm=exact['staggered__norm'];mean=exact['staggered__first']/norm
        assert mean==pytest.approx(1.,abs=3e-14)
        assert abs(exact['staggered__second']/norm-mean**2)<3e-14
        assert exact['staggered__spin']/norm==pytest.approx(2.,abs=3e-14)
        assert exact['staggered__spin_defect']<1e-28


def test_sampling_matches_involution_pushforward_and_rejects_bad_labels():
    source=np.tile([0,0,1,1],(30000,1))
    sample=transpose_born_samples(source,np.random.default_rng(102014))
    labels=sample@np.array([8,4,2,1]);counts=np.bincount(labels,minlength=16)/len(labels)
    expected=np.zeros(16);expected[[3,5,6,9,10]]=[3/7,1/7,1/7,1/7,1/7]
    np.testing.assert_allclose(counts,expected,atol=.012,rtol=0)
    plan=swap_moment_plan(source[:1]);logs=np.zeros(len(plan.configurations))
    with pytest.raises(ValueError):spin_source_moments(plan,logs,[],{'x':np.ones(4)},target_q=1)
    with pytest.raises(ValueError):swap_moment_plan(np.array([[0,0],[0,1]]))


def test_n_mod_four_zero_even_spin_flip_parent_gives_odd_spin_source_bound():
    # N=8 shares N=36's rotation/bit-flip phase. A generic even-flip parent
    # is NOT a singlet. Its vector source has S=1,3; the next Casimir is 12.
    n=8;bits=np.array(list(product((0,1),repeat=n)));full=np.arange(2**n)
    total=[]
    for local in (np.array([[0,.5],[.5,0]]),np.array([[0,.5j],[-.5j,0]]),np.diag([-.5,.5])):
        matrix=np.zeros((2**n,2**n),complex)
        for i in range(n):
            tensor=np.ones((1,1))
            for j in range(n):tensor=np.kron(tensor,local if j==i else np.eye(2))
            matrix+=tensor
        total.append(matrix)
    s2=sum(s@s for s in total)
    rng=np.random.default_rng(801426);psi=rng.normal(size=2**n)+1j*rng.normal(size=2**n)
    psi[bits.sum(1)!=n//2]=0;psi=(psi+psi[full^255]);psi/=np.linalg.norm(psi)
    phi=psi*((bits-.5)@np.exp(2j*np.pi*np.arange(n)/n));phi/=np.linalg.norm(phi)
    np.testing.assert_allclose(phi[full^255],-phi,atol=1e-14)
    eig,u=np.linalg.eigh(s2);weights=abs(u.conj().T@phi)**2
    assert weights[(abs(eig-2)>1e-10)&(abs(eig-12)>1e-10)].sum()<1e-25
    nontriplet=weights[abs(eig-2)>1e-10].sum()
    defect=np.linalg.norm((s2-2*np.eye(2**n))@phi)**2
    assert abs(nontriplet-defect/100)<1e-13
