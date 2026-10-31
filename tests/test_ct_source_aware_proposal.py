"""Independent finite-state support, rejection and unknown-normalizer oracles."""
from copy import deepcopy
from itertools import combinations

import jax
jax.config.update("jax_enable_x64",True)
import numpy as np
import pytest

from projects.clebsch_tree_nqs.explorations.ct_source_aware_proposal_20261030 import (
    PositiveSpinSourceTilt,triangular_K_tilt)
from su2_nqs.benchmarks.exact_diagonalization import heisenberg_sz_matrix


def basis(n):
    rows=[]
    for ups in combinations(range(n),n//2):
        row=np.zeros(n,np.int32);row[list(ups)]=1;rows.append(row)
    return np.asarray(rows)[::-1]


@pytest.mark.parametrize("n",[2,4,6,8,10])
def test_complete_known_law_support_and_scale(n):
    rows=basis(n);coeff=np.exp(2j*np.pi*np.arange(n)/n)/np.sqrt(n)
    tilt=PositiveSpinSourceTilt(coeff,.1)
    raw=np.arange(1,len(rows)+1,dtype=float);raw/=raw.sum()
    # Explicit normalized half-identity/half-swap bridge plus uniform floor.
    ids={tuple(r):i for i,r in enumerate(rows)};push=np.zeros(len(rows))
    for i,row in enumerate(rows):
        changed=row.copy();changed[0],changed[1]=row[1],row[0]
        push[ids[tuple(changed)]]+=raw[i]
    q=.8*(raw+push)/2+.2/len(rows)
    relative=tilt.relative_density(q,rows);Z=relative.sum();law=relative/Z
    assert abs(law.sum()-1)<5e-16
    assert np.all(law>=q*tilt.lower_density_ratio_bound-1e-16)
    np.testing.assert_allclose(relative,q*(.1+abs((rows-.5)@coeff)**2),rtol=0,atol=1e-16)
    assert tilt.envelope>=np.max(tilt.weight(rows))


def test_rejection_frequencies_and_complete_rng_restore():
    rows=basis(4);q=np.asarray([1.,2.,3.,4.,2.,1.]);q/=q.sum()
    tilt=PositiveSpinSourceTilt(np.asarray([1.,1.,-1.,-1.])/2,.1)
    def draw(count,seed):
        return rows[np.random.default_rng(seed).choice(len(rows),count,p=q)],{}
    rng=np.random.default_rng(43030101);state=deepcopy(rng.bit_generator.state)
    samples,receipt=tilt.draw(draw,30_000,rng,batch_size=512,maximum_candidates=400_000)
    relative=tilt.relative_density(q,rows);expected=relative/relative.sum()
    empirical=np.asarray([np.mean(np.all(samples==row,axis=1)) for row in rows])
    np.testing.assert_allclose(empirical,expected,rtol=0,atol=.012)
    restored=np.random.default_rng();restored.bit_generator.state=state
    repeated,replay=tilt.draw(draw,30_000,restored,batch_size=512,maximum_candidates=400_000)
    np.testing.assert_array_equal(samples,repeated)
    assert receipt==replay and receipt["normalizer_known"] is False


def test_nodes_H_squared_and_energy_gradient_keep_unknown_Z_exactly_cancelled():
    rows=basis(4);tilt=PositiveSpinSourceTilt(np.asarray([1.,1.,-1.,-1.])/2,.1)
    qbase=np.asarray([1.,2.,4.,3.,2.,1.]);qbase/=qbase.sum()
    qt=tilt.relative_density(qbase,rows);Z=qt.sum();q=qt/Z
    factor=tilt.factor(rows);carrier=np.asarray([.7+.1j,.2,.3j,.4,-.5j,.8])
    psi=factor*carrier
    H,_=heisenberg_sz_matrix(4,[(0,1,1.),(1,2,.7),(2,3,1.1),(0,3,.2)],n_up=2)
    hp=H@psi;nodes=psi==0
    assert nodes.sum()>0 and np.sum(abs(hp[nodes])**2)>0 and np.all(qt[nodes]>0)
    norm=np.sum(q*abs(psi)**2/qt)
    energy=np.sum(q*psi.conj()*hp/qt)/norm
    h2=np.sum(q*abs(hp)**2/qt)/norm
    np.testing.assert_allclose(energy,np.vdot(psi,H@psi)/np.vdot(psi,psi),rtol=0,atol=1e-15)
    np.testing.assert_allclose(h2,np.vdot(hp,hp)/np.vdot(psi,psi),rtol=0,atol=1e-15)
    # An unknown common scale changes the raw norm estimate, not observables.
    np.testing.assert_allclose(norm,np.vdot(psi,psi).real/Z,rtol=0,atol=1e-15)
    J=np.asarray([[.1,.2j],[.3j,.2],[.2,.1],[.4j,-.2],[.3,-.1j],[.2j,.4]])
    gradient=2*np.real(np.sum(q[:,None]*J.conj()*(hp-energy*psi)[:,None]/qt[:,None],axis=0))/norm
    exact=2*np.real(J.conj().T@(hp-energy*psi))/np.vdot(psi,psi).real
    np.testing.assert_allclose(gradient,exact,rtol=0,atol=1e-15)
    def E(p):
        v=psi+J@p
        return float((np.vdot(v,H@v)/np.vdot(v,v)).real)
    for i in range(2):
        d=np.eye(2)[i]*1e-6
        np.testing.assert_allclose((E(d)-E(-d))/(2e-6),gradient[i],rtol=0,atol=2e-10)


def test_N36_K_bound_is_integer_occupation_extremum_not_sample_fit():
    tilt=triangular_K_tilt(6)
    assert tilt.factor_squared_bound==3.
    colours=np.asarray([(i//6+i%6)%3 for i in range(36)])
    rows=[]
    for a in range(13):
        for b in range(13):
            c=18-a-b
            if 0<=c<=12:
                row=np.zeros(36,np.int32)
                for color,count in enumerate((a,b,c)):
                    row[np.flatnonzero(colours==color)[:count]]=1
                rows.append(row)
    values=abs(tilt.factor(np.asarray(rows)))**2
    assert abs(values.max()-3)<2e-15
    assert np.max(values)<=tilt.envelope-tilt.floor


def test_invalid_labels_and_unsafe_bound_fail_without_repairs():
    with pytest.raises(ValueError):PositiveSpinSourceTilt(np.ones(3))
    with pytest.raises(ValueError):PositiveSpinSourceTilt(np.ones(4),0.)
    with pytest.raises(ValueError):triangular_K_tilt(4)
    tilt=PositiveSpinSourceTilt(np.asarray([1.,1.,-1.,-1.])/2,.1,.01)
    with pytest.raises(ArithmeticError):tilt.weight(basis(4))
    with pytest.raises(ValueError):tilt.factor(np.zeros((1,4),np.int32))
