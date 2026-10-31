"""Independent regional density-matrix oracles, not training convergence tests."""
from math import comb, log

import numpy as np
import pytest

from su2_nqs.benchmarks.regional_su2_readout import regional_schur_columns, su2_multiplicity_readout
from su2_nqs.benchmarks.su2_entanglement_spectrum import subsystem_total_spin_squared
from projects.clebsch_tree_nqs.workflows.spin_symmetry_readout import density_readout


@pytest.mark.parametrize('n', [1, 2, 3, 4, 6, 7, 8])
def test_complete_schur_labels_match_independent_spin_operator(n):
    u, labels = regional_schur_columns(n)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(2**n), atol=3e-14)
    q = np.array([x[0] for x in labels])
    np.testing.assert_allclose(u.conj().T @ subsystem_total_spin_squared(n) @ u,
                               np.diag(q*(q+2)/4), atol=8e-14)
    r = su2_multiplicity_readout(np.eye(2**n)/2**n)
    for q in range(n%2,n+1,2):
        k=(n-q)//2
        d=comb(n,k)-(comb(n,k-1) if k else 0)
        s=r['sectors'][str(q)]
        assert s['multiplicity']==d
        assert s['p']==pytest.approx(d*(q+1)/2**n, abs=2e-14)
        np.testing.assert_allclose(s['conditional_multiplicity_spectrum'], np.full(d,1/d),atol=3e-14)
    assert r['entropy_nats']==pytest.approx(n*log(2),abs=3e-14)


def test_four_spin_existing_readout_replays_spectra_not_multiplicity_gauge():
    rng=np.random.default_rng(24)
    u, labels=regional_schur_columns(4)
    rho=np.zeros((16,16),complex)
    for q,d,p in [(0,2,.3),(2,3,.5),(4,1,.2)]:
        x=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
        r=x@x.conj().T; r*=p/np.trace(r)
        for m in range(-q,q+1,2):
            ids=[i for i,l in enumerate(labels) if l[:2]==(q,m)]
            rho+=u[:,ids]@(r/(q+1))@u[:,ids].conj().T
    old,_=density_readout(rho); new=su2_multiplicity_readout(rho)
    for q in ['0','2','4']:
        np.testing.assert_allclose(new['sectors'][q]['conditional_multiplicity_spectrum'],old['sectors'][q]['conditional_multiplicity_spectrum'],atol=5e-15)
    assert new['multiplicity_entropy']==pytest.approx(old['multiplicity_entropy'],abs=4e-15)


def test_kinematic_casimir_and_su2_breaking_are_not_confused():
    u,labels=regional_schur_columns(4)
    v=u[:,labels.index((2,0,0))]; rho=np.outer(v,v.conj())
    with pytest.raises(ValueError,match='no twirl'):
        su2_multiplicity_readout(rho)
    with pytest.raises(ValueError):
        su2_multiplicity_readout(np.eye(16)/8)
    with pytest.raises(ValueError):
        regional_schur_columns(10)
