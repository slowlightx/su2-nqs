"""Independent three-spin physical-space retained-response oracle."""
import numpy as np
import pytest

from su2_nqs.operators.su2_block_transfer import BlockIsometry,project_block
from su2_nqs.operators.su2_sector_merge import reduced_block_from_projected
from su2_nqs.operators.retained_spin_response import retained_local_spin_response
from su2_nqs.su2.triangle_blocks import triangle_block_tables


@pytest.mark.parametrize('truncate',[False,True])
def test_actual_state_weighted_response_matches_complete_physical_projector(truncate):
    rng=np.random.default_rng(26102599)
    magnetization=np.asarray([2*x.bit_count()-3 for x in range(8)])
    sectors={}
    sigma={}
    for q,table in triangle_block_tables()['three'].items():
        width=table.shape[1]
        gauge=np.linalg.qr(rng.normal(size=(width,width))+1j*rng.normal(size=(width,width)))[0]
        if truncate and q==1:gauge=gauge[:,:1]
        sectors[q]=np.stack([(table@gauge)*(magnetization==m)[:,None]
                             for m in range(-q,q+1,2)],axis=-1)
        matrix=rng.normal(size=(gauge.shape[1],)*2)+1j*rng.normal(size=(gauge.shape[1],)*2)
        sigma[q]=matrix@matrix.conj().T
    norm=sum(np.trace(x).real for x in sigma.values());sigma={q:x/norm for q,x in sigma.items()}
    projected=project_block(BlockIsometry(3,sectors),np.zeros((8,8)))
    reduced=reduced_block_from_projected(projected)
    observed=retained_local_spin_response(reduced,sigma)
    rho_reduced=np.zeros((len(projected.labels),)*2,complex)
    for i,(q,a,m) in enumerate(projected.labels):
        for j,(qr,b,n) in enumerate(projected.labels):
            if q==qr and m==n:rho_reduced[i,j]=sigma[q][a,b]/(q+1)
    rho=projected.basis@rho_reduced@projected.basis.conj().T
    p=projected.basis@projected.basis.conj().T
    local=(np.array([[0,.5],[.5,0]]),np.array([[0,.5j],[-.5j,0]]),np.diag([-.5,.5]))
    for i,row in enumerate(observed['sites']):
        spins=[np.kron(np.kron(np.eye(2**i),s),np.eye(2**(2-i))) for s in local]
        kept=sum(np.trace(rho@s@p@s).real for s in spins)
        missing=sum(np.trace(rho@s@(np.eye(8)-p)@s).real for s in spins)
        assert row['retained_weight']==pytest.approx(kept,abs=3e-14)
        assert row['missing_weight']==pytest.approx(missing,abs=3e-14)
        assert kept+missing==pytest.approx(.75,abs=3e-14)
        if not truncate:assert kept==pytest.approx(.75,abs=3e-14)
    if truncate:assert observed['mean_retained_fraction']<.95
    with pytest.raises(ValueError,match='unit total trace'):
        retained_local_spin_response(reduced,{q:2*x for q,x in sigma.items()})


@pytest.mark.parametrize('q2_probability',[0.,.37])
def test_four_spin_high_q_omission_bound_with_mixed_input_sectors(q2_probability):
    """Drop q=4 exactly and test its physical response, not only tau loss."""
    from su2_nqs.su2.local_schur import cell_schur_tensor

    rng=np.random.default_rng(261025101)
    table=cell_schur_tensor('square_plaquette')
    sectors={};sigma={}
    for q,width,probability in ((0,2,1-q2_probability),(2,3,q2_probability)):
        gauge=np.linalg.qr(rng.normal(size=(width,width))
                           +1j*rng.normal(size=(width,width)))[0]
        sectors[q]=np.stack([table[:,q,4+m,:width]@gauge
                             for m in range(-q,q+1,2)],axis=-1)
        matrix=rng.normal(size=(width,width))+1j*rng.normal(size=(width,width))
        density=matrix@matrix.conj().T
        sigma[q]=probability*density/np.trace(density).real
    projected=project_block(BlockIsometry(4,sectors),np.zeros((16,16)))
    observed=retained_local_spin_response(reduced_block_from_projected(projected),sigma)
    rho_reduced=np.zeros((len(projected.labels),)*2,complex)
    for i,(q,a,m) in enumerate(projected.labels):
        for j,(qr,b,n) in enumerate(projected.labels):
            if q==qr and m==n:rho_reduced[i,j]=sigma[q][a,b]/(q+1)
    rho=projected.basis@rho_reduced@projected.basis.conj().T
    p=projected.basis@projected.basis.conj().T
    q4_basis=np.column_stack([table[:,4,4+m,0] for m in range(-4,5,2)])
    high_projector=q4_basis@q4_basis.T
    assert np.max(abs(p+high_projector-np.eye(16)))<3e-14
    assert observed['maximum_retained_q']==2
    assert observed['missing_above_retained_qmax_upper_bound']==pytest.approx(
        .75*q2_probability,abs=3e-14)
    local=(np.array([[0,.5],[.5,0]]),np.array([[0,.5j],[-.5j,0]]),np.diag([-.5,.5]))
    high_weights=[]
    for i,row in enumerate(observed['sites']):
        spins=[np.kron(np.kron(np.eye(2**i),s),np.eye(2**(3-i))) for s in local]
        kept=sum(np.trace(rho@s@p@s).real for s in spins)
        high=sum(np.trace(rho@s@high_projector@s).real for s in spins)
        high_weights.append(high)
        assert row['retained_weight']==pytest.approx(kept,abs=3e-14)
        assert row['missing_weight']==pytest.approx(high,abs=3e-14)
        assert kept+high==pytest.approx(.75,abs=3e-14)
        assert -3e-14<=high<=.75*q2_probability+3e-14
        assert row['missing_not_above_retained_qmax_lower_bound']==pytest.approx(0.,abs=3e-14)
    if q2_probability>0:assert max(high_weights)>.01
    else:assert np.max(np.abs(high_weights))<3e-14
