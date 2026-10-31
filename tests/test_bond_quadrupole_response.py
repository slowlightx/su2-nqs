"""Independent tensor-product oracles, including forbidden SWAP sectors."""
import itertools
import numpy as np
import pytest

from su2_nqs.benchmarks.replica_spin_response import (
    row_pi_bonds, double_spin_connections, double_spin_vector_action,
    operator_action_over_sample,
)
from su2_nqs.benchmarks.mps_rotation_covariance import charged_replica_matrix


def _dense(n, bonds):
    plus=np.array([[0.,0.],[1.,0.]])
    out=np.zeros((2**n,2**n),complex)
    for a,b,w in bonds:
        term=np.ones((1,1))
        for i in range(n):term=np.kron(term,plus if i in (a,b) else np.eye(2))
        out+=w*term
    return out


def test_charged_connections_and_Gram_trace_match_complex_dense_RDM():
    rng=np.random.default_rng(2026101501);n=4
    bonds=row_pi_bonds(0,4);Q=_dense(n,bonds)
    v=rng.normal(size=(16,3))+1j*rng.normal(size=(16,3))
    np.testing.assert_allclose(double_spin_vector_action(v,bonds),Q@v,atol=2e-15)
    np.testing.assert_allclose(double_spin_vector_action(v,bonds,delta_m=-4),Q.conj().T@v,atol=2e-15)
    m=np.array([2*i.bit_count()-4 for i in range(16)])
    a=np.zeros((16,16),complex);b=np.zeros_like(a)
    for q in np.unique(m):
        ids=np.flatnonzero(m==q);a[np.ix_(ids,ids)]=rng.normal(size=(len(ids),len(ids)))+1j*rng.normal(size=(len(ids),len(ids)))
        b[np.ix_(ids,ids)]=rng.normal(size=(len(ids),len(ids)))+1j*rng.normal(size=(len(ids),len(ids)))
    R=b@b.conj().T;rho=a@R@a.conj().T
    G=a.conj().T@Q@a
    actual=charged_replica_matrix(R,{'q':G},m,delta_m=4)[0,0]
    np.testing.assert_allclose(actual,np.trace(rho@Q.conj().T@rho@Q),rtol=2e-15,atol=1e-9)
    # The row cross terms need the correct complex order, not only a
    # nonnegative diagonal. A different rank-two bond pattern is independent.
    Qb=_dense(n,((0,2,.3),(1,3,-.7)))
    actual_matrix=charged_replica_matrix(R,{'q':G,'b':a.conj().T@Qb@a},m,delta_m=4)
    expected=np.array([[np.trace(rho@A.conj().T@rho@B) for B in (Q,Qb)] for A in (Q,Qb)])
    np.testing.assert_allclose(actual_matrix,expected,rtol=2e-15,atol=1e-9)
    with pytest.raises(ValueError,match='charge shift'):charged_replica_matrix(R,{'bad':G},m,delta_m=2)


def test_two_copy_charged_estimator_keeps_forbidden_plain_swaps():
    # Generic complex fixed-m=0 state. Off-diagonal O can rescue a plain
    # forbidden SWAP; postselecting zero-charge swaps would give zero here.
    rng=np.random.default_rng(2026101502);n=8;cut=4
    bits=np.array(list(itertools.product((0,1),repeat=n)),np.int8)
    allowed=np.flatnonzero(bits.sum(1)==4)
    psi=np.zeros(256,complex);psi[allowed]=rng.normal(size=len(allowed))+1j*rng.normal(size=len(allowed));psi/=np.linalg.norm(psi)
    x=np.repeat(bits[allowed],len(allowed),axis=0);y=np.tile(bits[allowed],(len(allowed),1))
    px=np.repeat(psi[allowed],len(allowed));py=np.tile(psi[allowed],len(allowed))
    u=x.copy();v=y.copy();u[:,:cut]=y[:,:cut];v[:,:cut]=x[:,:cut]
    actions=[]
    for rows,den,shift in ((u,px,-4),(v,py,4)):
        targets,src,coeff=double_spin_connections(rows,row_pi_bonds(0,4),delta_m=shift)
        packed=targets@(1<<np.arange(n-1,-1,-1));values=psi[packed]
        logs=np.full(values.shape,-np.inf+0j);nz=abs(values)>0;logs[nz]=np.log(values[nz])
        actions.append(operator_action_over_sample(logs,src,coeff,np.log(den)))
    w=abs(px*py)**2;record=actions[0]*actions[1]
    rho=psi.reshape(16,16)@psi.reshape(16,16).conj().T;Q=_dense(cut,row_pi_bonds(0,4))
    exact=np.trace(rho@Q.conj().T@rho@Q)
    np.testing.assert_allclose(np.sum(w*record),exact,atol=5e-17,rtol=1e-14)
    assert exact.real>1e-3
    assert np.max(abs(record[u.sum(1)==4]))==0
    other=((0,2,.3),(1,3,-.7))
    targets,src,coeff=double_spin_connections(v,other,delta_m=4)
    other_values=psi[targets@(1<<np.arange(n-1,-1,-1))]
    other_logs=np.full(other_values.shape,-np.inf+0j)
    nonzero=abs(other_values)>0;other_logs[nonzero]=np.log(other_values[nonzero])
    other_action=operator_action_over_sample(other_logs,src,coeff,np.log(py))
    cross_exact=np.trace(rho@Q.conj().T@rho@_dense(cut,other))
    np.testing.assert_allclose(np.sum(w*actions[0]*other_action),cross_exact,atol=5e-17,rtol=1e-14)
    reverse=[]
    for rows,den,shift in ((v,py,-4),(u,px,4)):
        targets,src,coeff=double_spin_connections(rows,row_pi_bonds(0,4),delta_m=shift)
        packed=targets@(1<<np.arange(n-1,-1,-1));values=psi[packed]
        logs=np.full(values.shape,-np.inf+0j);nz=abs(values)>0;logs[nz]=np.log(values[nz])
        reverse.append(operator_action_over_sample(logs,src,coeff,np.log(den)))
    averaged=(record+reverse[0]*reverse[1])/2
    np.testing.assert_allclose(np.sum(w*averaged),exact,atol=5e-17,rtol=1e-14)
    # Exact iid-pair distribution: conditional symmetry averaging cannot
    # increase the complex second moment. Persistent pairs remain ONE group.
    assert np.sum(w*abs(averaged)**2) <= np.sum(w*abs(record)**2)+1e-14


def test_physical_MPS_quadrupole_matches_dense_noncanonical_parent():
    from test_mps_rotation_covariance import _small_mps
    from su2_nqs.benchmarks.mps_rotation_covariance import finite_mps_prefix_rotation_covariance
    psi,state=_small_mps(True,True)
    report=finite_mps_prefix_rotation_covariance(psi,4,measured_charges=(0,4),
        leading_levels=16,spatial_strip_columns=4,spatial_levels=16,bond_quadrupole_rows=(0,))
    rho=state.reshape(16,16)@state.reshape(16,16).conj().T;Q=_dense(4,row_pi_bonds(0,4))
    exact=np.trace(rho@Q.conj().T@rho@Q).real
    np.testing.assert_allclose(report['two_copy_bond_quadrupole']['real'][0][0],exact,atol=3e-15,rtol=0)
    m=np.array([2*i.bit_count()-4 for i in range(16)]);vec={}
    for q in (0,4):
        ids=np.flatnonzero(m==q);_,v=np.linalg.eigh(rho[np.ix_(ids,ids)])
        vec[q]=np.zeros((16,len(ids)),complex);vec[q][ids]=v[:,::-1]
    expected=vec[4].conj().T@Q@vec[0]
    result=report['bond_quadrupole_transitions']['rows']['0']
    actual=np.array(result['real'])+1j*np.array(result['imag'])
    np.testing.assert_allclose(abs(actual)**2,abs(expected)**2,atol=3e-11,rtol=0)
