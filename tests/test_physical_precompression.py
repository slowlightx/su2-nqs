"""Independent physical matrix oracle; no optimization or saved results."""
import numpy as np
import pytest
from scipy.linalg import expm

from su2_nqs.operators.physical_precompression import cyclic_three_spin_action


def test_cyclic_gate_matches_direct_exponential_and_total_spin():
    n=6;bits=np.arange(1<<n);sites=(0,2,5)
    def permutation(i,j):
        mask=(1<<(n-1-i))|(1<<(n-1-j))
        source=np.array([x^mask if ((x>>(n-1-i))&1)!=((x>>(n-1-j))&1) else x for x in bits])
        return np.eye(1<<n)[source]
    p=permutation(sites[0],sites[1]);q=permutation(sites[1],sites[2]);K=p@q-q@p
    S2=(3*n/4)*np.eye(1<<n)
    raising=np.zeros((1<<n,1<<n))
    for col in bits:
        for i in range(n):
            mask=1<<(n-1-i)
            if not col&mask:raising[col|mask,col]+=1.
    for i in range(n):
        for j in range(i):S2+=permutation(i,j)-.5*np.eye(1<<n)
    for theta in (0.,-.037,.06):
        U=cyclic_three_spin_action(np.eye(1<<n),bits,n_sites=n,sites=sites,theta=theta)
        np.testing.assert_allclose(U,expm(theta*K),atol=4e-16)
        np.testing.assert_allclose(U.T@U,np.eye(1<<n),atol=7e-16)
        np.testing.assert_allclose(S2@U,U@S2,atol=2e-15)
        np.testing.assert_allclose(raising@U,U@raising,atol=7e-16)
    rng=np.random.default_rng(271);v=rng.normal(size=(64,3))+1j*rng.normal(size=(64,3))
    full=cyclic_three_spin_action(v,bits,n_sites=n,sites=sites,theta=.03)
    row=np.array([x for x in bits if int(x).bit_count()==3])
    sector=cyclic_three_spin_action(v[row],row,n_sites=n,sites=sites,theta=.03)
    np.testing.assert_allclose(sector,full[row],atol=0.)


def test_cyclic_gate_refuses_missing_support():
    with pytest.raises(ValueError,match='closed'):
        cyclic_three_spin_action(np.ones(2),np.array([1,2]),n_sites=3,sites=(0,1,2),theta=.1)
    with pytest.raises(ValueError,match='sorted'):
        cyclic_three_spin_action(np.ones(2),np.array([2,1],dtype=np.uint32),n_sites=3,sites=(0,1,2),theta=.1)
