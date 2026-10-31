import numpy as np
import pytest

from su2_nqs.benchmarks.su2_multiplicity_intervention import (
    pair_spin_projectors, pair_spin_dephase, chirality_unitary,
    compact6_intervention_operators, intervention_observables, triangle_multiplicity_identity)


def test_vector_chirality_fixed_CS_multiplicity_identity():
    report=triangle_multiplicity_identity()
    assert abs(abs(report['coefficient_c'])-np.sqrt(3))<1e-14
    assert max(v for k,v in report.items() if k.endswith('_error'))<2e-14
    bank=compact6_intervention_operators()
    assert np.linalg.norm(bank['vector_chirality_A_dot_B']-bank['vector_chirality_qtotal0']-bank['vector_chirality_qtotal2'])<2e-14
    p=pair_spin_projectors(6,(0,3))
    assert np.linalg.norm(sum(x@bank['vector_chirality_A_dot_B']@x for x in p.values()))<2e-14


def test_pair_projectors_independent_pauli_oracle_and_su2():
    sx=np.array([[0,1],[1,0]],complex)/2
    sy=np.array([[0,1j],[-1j,0]],complex)/2
    sz=np.diag([-.5,.5])
    dot=sum(np.kron(s,s) for s in (sx,sy,sz))
    p=pair_spin_projectors(2,(0,1))
    assert max(np.linalg.norm(p[0]-(np.eye(4)/4-dot)),np.linalg.norm(p[0]+p[2]-np.eye(4)))<1e-14
    for project in p.values():
        assert np.linalg.norm(project@project-project)<1e-14
        for s in (sx,sy,sz):
            total=np.kron(s,np.eye(2))+np.kron(np.eye(2),s)
            assert np.linalg.norm(total@project-project@total)<1e-14


def test_dephase_preserves_trace_all_joint_spins_and_triangle_energy():
    rng=np.random.default_rng(20261028)
    x=rng.normal(size=(64,5))+1j*rng.normal(size=(64,5))
    rho=x@x.conj().T;rho/=np.trace(rho)
    changed,prob=pair_spin_dephase(rho,(0,3))
    a,b=intervention_observables(rho),intervention_observables(changed)
    assert abs(sum(prob.values())-1)<1e-14
    assert abs(np.trace(changed)-1)<1e-14
    assert np.linalg.eigvalsh(changed).min()>-1e-14
    assert max(abs(a[k]-b[k]) for k in a if k.startswith('P_'))<2e-14
    for k in ('triangle_Casimir_energy','S_A_dot_S_B'):
        assert abs(a[k]-b[k])<2e-14
    u=chirality_unitary(6,(0,3,4),.4)
    assert np.linalg.norm(u.conj().T@u-np.eye(64))<1e-13
    for k,op in compact6_intervention_operators().items():
        if k.startswith('P_'):
            assert np.linalg.norm(u@op-op@u)<1e-13


@pytest.mark.parametrize('pair', [(0,0), (False,1),(-1,1),(0,6),(0,1,2)])
def test_invalid_physical_sites_rejected(pair):
    with pytest.raises(ValueError):pair_spin_projectors(6,pair)
