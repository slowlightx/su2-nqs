"""Independent full-physical checks of whole-complement downward transport."""
import numpy as np
import pytest

from test_su2_conditional_response import _fixture, _physical_merge, _count, _physical_cut_action
from su2_nqs.operators.su2_complement import (
    descend_singlet_complement, restore_singlet_parent, make_fixed_complement_cut_response,
)
from su2_nqs.operators.su2_three_block_center import ThreeBlockSingletOperator
from su2_nqs.su2.clebsch_gordan import clebsch_gordan


def _singlet(a, e, coefficients):
    vector = np.zeros(next(iter(a.values())).shape[0]*next(iter(e.values())).shape[0],complex)
    for q,c in coefficients.items():
        for m in range(-q,q+1,2):
            vector += clebsch_gordan(q,m,q,-m,0,0)*np.einsum('pa,ab,rb->pr',
                a[q][:,:,(m+q)//2],c,e[q][:,:,(-m+q)//2]).ravel()
    return vector


def _prepared():
    args,(pl,pr,pb,pd),psi,ab,ad=_fixture()
    step=descend_singlet_complement(args['a_source'],args['b'],args['d'],
        args['parent_maps'],args['root_coefficients'],[(0,2,.37)])
    pa=_physical_merge(pl,pr,args['a_maps'])
    pe=_physical_merge(pb,pd,step.complement_maps)
    return args,(pl,pr,pb,pd,pa,pe),psi,ab,ad,step


def test_downward_complement_and_changed_center_restore_exact_complex_state():
    args,(pl,pr,pb,pd,pa,pe),psi,_,_,step=_prepared()
    np.testing.assert_allclose(_singlet(pa,pe,step.coefficients),psi,atol=5e-13)
    rng=np.random.default_rng(2610269101)
    changed={q:c+.1*(rng.normal(size=c.shape)+1j*rng.normal(size=c.shape))
             for q,c in step.coefficients.items()}
    norm=np.sqrt(sum(np.vdot(c,c).real for c in changed.values()))
    changed={q:c/norm for q,c in changed.items()}
    maps,roots,notes=restore_singlet_parent(step,changed,args['a_source'].q_widths,
        parent_widths={q:w.shape[1] for q,w in args['parent_maps'].items()})
    pc=_physical_merge(pa,pb,maps)
    np.testing.assert_allclose(_singlet(pc,pd,roots),_singlet(pa,pe,changed),atol=5e-13)
    assert notes['no_singular_value_truncation']


def test_whole_complement_response_matches_physical_gradient_projection():
    args,(pl,pr,_,_,_,pe),psi,ab,ad,step=_prepared()
    bonds=ab+[(i,3+j,g) for i,j,g in ad]
    response=make_fixed_complement_cut_response(a_children=args['a_children'],a_maps=args['a_maps'],
        environment=step.complement,coefficients=step.coefficients,cross_bonds=bonds,target_q=2)
    ua=_physical_merge(pl,pr,{2:np.eye(_count(pl,pr,2))})[2]
    action=_physical_cut_action(psi,ab,ad)
    expected=np.sqrt(3)*ua[:,:,-1].conj().T@action.reshape(64,64)@pe[2][:,:,0].conj()
    np.testing.assert_allclose(response.to_dense(),expected,atol=5e-13)
    assert response.diagnostics['whole_complement_sites']==6
    assert not response.diagnostics['physical_bath_leakage_included']


def test_fixed_complement_center_full_H_action_matches_physical_bonds():
    args,(pl,pr,_,_,_,pe),_,ab,ad,step=_prepared()
    bonds=ab+[(i,3+j,g) for i,j,g in ad]
    le=[(i,j,g) for i,j,g in bonds if i<3]
    re=[(i-3,j,g) for i,j,g in bonds if i>=3]
    op=ThreeBlockSingletOperator(*args['a_children'],step.complement,(),le,re)
    complete=_physical_merge(pl,pr,{q:np.eye(_count(pl,pr,q)) for q in step.coefficients})
    physical=[]
    for i in range(op.dimension):
        co={q:np.zeros((_count(pl,pr,q),step.complement.q_widths[q]),complex) for q in step.coefficients}
        for q,ids in op.q_indices.items():
            where=np.flatnonzero(ids==i)
            if len(where): co[q].flat[int(where[0])]=1
        physical.append(_singlet(complete,pe,co))
    basis=np.column_stack(physical)
    rng=np.random.default_rng(2610269102)
    c=rng.normal(size=op.dimension)+1j*rng.normal(size=op.dimension)
    psi=basis@c
    action=_physical_cut_action(psi,ab,ad)
    ids=np.arange(4096); i,j=6,11
    bi,bj=(ids>>(11-i))&1,(ids>>(11-j))&1
    action+=.37*(bi-.5)*(bj-.5)*psi
    different=bi!=bj; flipped=ids[different]^((1<<(11-i))|(1<<(11-j)))
    action[flipped]+=.37*.5*psi[different]
    np.testing.assert_allclose(op.matvec(c),basis.conj().T@action,atol=8e-13)
    np.testing.assert_allclose(basis.conj().T@basis,np.eye(op.dimension),atol=5e-13)


@pytest.mark.parametrize('bad_q',[True,2.,-2])
def test_complement_response_rejects_invalid_doubled_labels(bad_q):
    args,_,_,ab,ad,step=_prepared()
    with pytest.raises((TypeError,ValueError)):
        make_fixed_complement_cut_response(a_children=args['a_children'],a_maps=args['a_maps'],
            environment=step.complement,coefficients=step.coefficients,
            cross_bonds=ab+[(i,3+j,g) for i,j,g in ad],target_q=bad_q)


def test_complement_response_rejects_nonfinite_or_missing_coefficients():
    args,_,_,ab,ad,step=_prepared()
    for bad in ({q:c for q,c in step.coefficients.items() if q!=2},
                {q:np.full_like(c,np.nan) for q,c in step.coefficients.items()}):
        with pytest.raises(ValueError):
            make_fixed_complement_cut_response(a_children=args['a_children'],a_maps=args['a_maps'],
                environment=step.complement,coefficients=bad,
                cross_bonds=ab+[(i,3+j,g) for i,j,g in ad],target_q=2)
