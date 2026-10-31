"""Independent <=8-spin physical oracles for a joint upper/root center."""
import numpy as np
import pytest

from su2_nqs.operators.su2_block_transfer import BlockIsometry, project_block
from su2_nqs.operators.su2_sector_merge import reduced_block_from_projected
from su2_nqs.operators.su2_three_block_center import (
    ThreeBlockSingletOperator, parent_anchored_center_ritz,
)
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.triangle_blocks import triangle_block_tables


def physical_action(vectors,n,bonds):
    rows=np.arange(2**n);value=np.asarray(vectors);result=np.zeros_like(value,dtype=complex)
    for i,j,g in bonds:
        bi=(rows>>(n-i-1))&1;bj=(rows>>(n-j-1))&1
        result += g*((bi-.5)*(bj-.5))[:,None]*value
        mask=bi!=bj;new=rows[mask]^(1<<(n-i-1))^(1<<(n-j-1))
        result[new] += .5*g*value[mask]
    return result


def child(n,bonds,seed=None):
    if n==2:
        sectors={q:np.zeros((4,1,q+1),complex) for q in (0,2)}
        for q,u in sectors.items():
            for bit in range(4):
                ml=2*((bit>>1)&1)-1;mr=2*(bit&1)-1;m=ml+mr
                if abs(m)<=q:u[bit,0,(m+q)//2]=clebsch_gordan(1,ml,1,mr,q,m)
    else:
        mags=np.asarray([2*x.bit_count()-3 for x in range(8)])
        sectors={q:np.stack([table*(mags==m)[:,None] for m in range(-q,q+1,2)],axis=-1)
                 for q,table in triangle_block_tables()['three'].items()}
    if seed is not None:
        rng=np.random.default_rng(seed)
        sectors={q:np.einsum('pam,ab->pbm',u,np.linalg.qr(
            rng.normal(size=(u.shape[1],)*2)+1j*rng.normal(size=(u.shape[1],)*2))[0])
            for q,u in sectors.items()}
    h=physical_action(np.eye(2**n),n,bonds)
    block=project_block(BlockIsometry(n,sectors),h)
    return reduced_block_from_projected(block),sectors


def example(complex_gauge,sizes=(3,3,2)):
    templates=([(0,1,1.),(1,2,.7),(0,2,.125)],[(0,1,.8),(1,2,1.2)],[(0,1,.6),(1,2,.4)])
    internal=tuple([(i,j,g) for i,j,g in rows if max(i,j)<n]
                   for rows,n in zip(templates,sizes,strict=True))
    blocks=[];sectors=[]
    for k,(n,bonds) in enumerate(zip(sizes,internal,strict=True)):
        b,u=child(n,bonds,26102440+k if complex_gauge else None);blocks.append(b);sectors.append(u)
    ab=[(0,min(2,sizes[1]-1),.9),(2,1,.125)];ae=[(1,0,.4),(2,1,1.1)]
    be=[(0,1,.75),(min(2,sizes[1]-1),0,.3)]
    op=ThreeBlockSingletOperator(*blocks,ab,ae,be)
    nb=sizes[0];ne=sum(sizes[:2])
    full=list(internal[0])+[(i+nb,j+nb,g) for i,j,g in internal[1]]+[(i+ne,j+ne,g) for i,j,g in internal[2]]
    full += [(i,j+nb,g) for i,j,g in ab]+[(i,j+ne,g) for i,j,g in ae]+[(i+nb,j+ne,g) for i,j,g in be]
    columns=[]
    for qa,a,qb,b,qe,e in op.channels:
        vector=np.zeros(256,complex)
        for ma in range(-qa,qa+1,2):
            for mb in range(-qb,qb+1,2):
                me=-ma-mb
                if abs(me)>qe or (me-qe)%2:continue
                cg=clebsch_gordan(qa,ma,qb,mb,qe,-me)*clebsch_gordan(qe,-me,qe,me,0,0)
                vector += cg*np.kron(np.kron(sectors[0][qa][:,a,(ma+qa)//2],
                    sectors[1][qb][:,b,(mb+qb)//2]),sectors[2][qe][:,e,(me+qe)//2])
        columns.append(vector)
    basis=np.column_stack(columns)
    return op,basis,full


@pytest.mark.parametrize('complex_gauge',[False,True])
@pytest.mark.parametrize('sizes',[(3,3,2),(3,2,3)])
def test_three_block_all_pair_H_matches_physical_H_and_singlet(complex_gauge,sizes):
    op,basis,bonds=example(complex_gauge,sizes)
    assert op.dimension==14
    assert np.max(abs(basis.conj().T@basis-np.eye(14)))<3e-14
    reference=basis.conj().T@physical_action(basis,8,bonds)
    actual=np.column_stack([op.matvec(e) for e in np.eye(14)])
    assert np.max(abs(actual-reference))<5e-13
    assert np.max(abs(actual-actual.conj().T))<5e-13
    # Total spin S^2 annihilates every retained singlet.
    spin_squared=6*basis+2*physical_action(basis,8,[(i,j,1.) for i in range(8) for j in range(i+1,8)])
    assert np.max(abs(spin_squared))<5e-13


@pytest.mark.parametrize('complex_gauge',[False,True])
@pytest.mark.parametrize('sizes',[(3,3,2),(3,2,3)])
def test_joint_center_ritz_updates_upper_not_only_root_and_factors_exactly(complex_gauge,sizes):
    op,basis,bonds=example(complex_gauge,sizes);rng=np.random.default_rng(26102449)
    parent=rng.normal(size=14)
    if complex_gauge:parent=parent+1j*rng.normal(size=14)
    parent=parent/np.linalg.norm(parent)
    widths={q:op.blocks[2].q_widths[q] for q in op.q_indices}
    upper,root,_=op.factor(parent,widths=widths)
    assert np.max(abs(op.encode(upper,root)-parent))<5e-15
    new,info=parent_anchored_center_ritz(op,parent,krylov_vectors=3)
    newupper,newroot,notes=op.factor(new,widths=widths)
    assert np.max(abs(op.encode(newupper,newroot)-new))<5e-15
    assert notes['maximum_factorization_error']<5e-15
    assert any(np.linalg.norm((np.eye(len(upper[q]))-upper[q]@upper[q].conj().T)@newupper[q])>.01
               for q in widths)
    hpsi=physical_action((basis@new)[:,None],8,bonds).ravel()
    assert info['raw_energy_after']==pytest.approx(np.vdot(basis@new,hpsi).real,abs=5e-13)
    assert info['raw_energy_after']<info['raw_energy_before']-.01
    assert info['center_H_matvecs']==3
    assert not info['physical_full_H_variance_known']


def test_bounded_center_and_fixed_Krylov_controls():
    b,_=child(2,[(0,1,1.)])
    with pytest.raises(ValueError,match='dimension'):
        ThreeBlockSingletOperator(b,b,b,[],[],[],max_dimension=1)
    op=ThreeBlockSingletOperator(b,b,b,[],[],[])
    with pytest.raises(MemoryError):
        ThreeBlockSingletOperator(b,b,b,[],[],[],memory_limit_bytes=1)
    for k in (0,9,True,1.5):
        with pytest.raises(ValueError):parent_anchored_center_ritz(op,np.ones(op.dimension),krylov_vectors=k)
    with pytest.raises(ValueError):op.matvec(np.ones(op.dimension+1))


def test_cached_parent_product_preserves_fixed_trial_space_and_work_count():
    op,_,_=example(True);rng=np.random.default_rng(26102451)
    parent=rng.normal(size=op.dimension);parent=parent/np.linalg.norm(parent)
    direct,first=parent_anchored_center_ritz(op,parent,krylov_vectors=4)
    hp=op.matvec(parent)
    cached,second=parent_anchored_center_ritz(op,parent,krylov_vectors=4,initial_product=hp)
    assert np.max(abs(direct-cached))<2e-14
    assert first['center_H_matvecs']==4
    assert second['center_H_matvecs']==3
    assert second['previously_evaluated_parent_H_products']==1


@pytest.mark.parametrize('q',[0,1,2,3,4])
def test_orient_root_center_swap_matches_explicit_CG_amplitudes(q):
    """Independent magnetic oracle includes the odd-q singlet swap sign."""
    from projects.clebsch_tree_nqs.workflows.triangle_boundary_response_enrichment import (
        oriented_root_coefficients,
    )
    rng=np.random.default_rng(26102591+q)
    gamma=rng.normal(size=(2,3))+1j*rng.normal(size=(2,3))
    gamma/=np.linalg.norm(gamma)
    channels=tuple((q,a,q,b) for a in range(2) for b in range(3))
    left=oriented_root_coefficients(gamma.reshape(-1,1),channels,center_on_left=True)[q]
    right=oriented_root_coefficients(gamma.reshape(-1,1),channels,center_on_left=False)[q]
    direct=np.zeros((2,q+1,3,q+1),complex)
    exchanged=np.zeros_like(direct)
    for a in range(2):
        for b in range(3):
            for mi,m in enumerate(range(-q,q+1,2)):
                for ni,n in enumerate(range(-q,q+1,2)):
                    direct[a,mi,b,ni]=gamma[a,b]*clebsch_gordan(q,m,q,n,0,0)
                    exchanged[a,mi,b,ni]=right[b,a]*clebsch_gordan(q,n,q,m,0,0)
    assert np.array_equal(left,gamma)
    assert np.max(abs(direct-exchanged))<2e-15
