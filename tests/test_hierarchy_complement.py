"""All left/right cuts of a complex N12 singlet checked in physical space."""
import numpy as np
import pytest

from test_su2_conditional_response import _fixture, _physical_merge, _physical_cut_action
from test_su2_complement import _singlet
from su2_nqs.operators.su2_complement import hierarchy_singlet_complement, swap_coupled_children
from su2_nqs.operators.su2_conditional_response import _channels
from su2_nqs.operators.su2_sector_merge import project_sector_maps


def _tree():
    args, physical, psi, ab, ad = _fixture()
    left, right = args['a_children']
    c = project_sector_maps(args['a_source'], args['b'], ab,
        maps=args['parent_maps'], compact_boundary=True)
    labels = _channels(c.q_widths, args['d'].q_widths, 0)
    root_map = np.asarray([args['root_coefficients'][q][a,b] for q,a,_,b in labels])[:,None]
    root_cross = ad+[(7,2,.37)]
    root = project_sector_maps(c, args['d'], root_cross, maps={0:root_map}, compact_boundary=True)
    blocks = [left,right,args['b'],args['d'],args['a_source'],c,root]
    supports = [(0,1,2),(3,4,5),(6,7,8),(9,10,11),tuple(range(6)),tuple(range(9)),tuple(range(12))]
    merges = [(0,1,4),(4,2,5),(5,3,6)]
    maps = {0:args['a_maps'],1:args['parent_maps'],2:{0:root_map}}
    basis = list(physical)
    basis.append(_physical_merge(basis[0],basis[1],maps[0]))
    basis.append(_physical_merge(basis[4],basis[2],maps[1]))
    bonds = [(i,6+j,g) for i,j,g in ab]+[(i,9+j,g) for i,j,g in ad]+[(7,11,.37)]
    hpsi = _physical_cut_action(psi,ab,ad)
    ids=np.arange(4096); bi=(ids>>4)&1; bj=ids&1
    hpsi += .37*(bi-.5)*(bj-.5)*psi
    mask=bi!=bj; flip=ids[mask]^((1<<4)|1)
    hpsi[flip] += .185*psi[mask]
    return dict(blocks=blocks,merges=merges,maps=maps,supports=supports,bonds=bonds),basis,psi,np.vdot(psi,hpsi)


@pytest.mark.parametrize('target',range(6))
def test_every_cut_preserves_complex_state_and_complete_physical_energy(target):
    tree,basis,psi,expected_energy=_tree()
    result=hierarchy_singlet_complement(**tree,target=target)
    pe=basis[3] if set(tree['supports'][target]) <= set(tree['supports'][5]) else basis[5]
    lookup={p:(l,r) for l,r,p in tree['merges']}
    for parent,side,step in result.steps:
        l,r=lookup[parent]; sibling=r if side=='left' else l
        pe=_physical_merge(basis[sibling],pe,step.complement_maps)
    physical=_singlet(basis[target],pe,result.coefficients)
    order=tree['supports'][target]+tuple(result.complement_sites)
    physical=physical.reshape((2,)*12).transpose(np.argsort(order)).ravel()
    np.testing.assert_allclose(physical,psi,atol=8e-13,rtol=0.)
    target_lookup={s:i for i,s in enumerate(tree['supports'][target])}
    env_lookup={s:i for i,s in enumerate(result.complement_sites)}
    cross=[]
    for i,j,g in tree['bonds']:
        if i in target_lookup and j in env_lookup: cross.append((target_lookup[i],env_lookup[j],g))
        elif j in target_lookup and i in env_lookup: cross.append((target_lookup[j],env_lookup[i],g))
    labels=_channels(tree['blocks'][target].q_widths,result.complement.q_widths,0)
    root=np.asarray([result.coefficients[q][a,b] for q,a,_,b in labels])[:,None]
    contracted=project_sector_maps(tree['blocks'][target],result.complement,cross,
        maps={0:root},compact_boundary=True)
    np.testing.assert_allclose(contracted.h_by_q[0][0,0],expected_energy,atol=1e-12,rtol=0.)
    assert result.diagnostics['complement_sites']==12-len(tree['supports'][target])


def test_child_swap_is_involution_with_nontrivial_exchange_phase():
    tree,_,_,_=_tree()
    left,right=tree['blocks'][4],tree['blocks'][2]
    original=tree['maps'][1]
    swapped=swap_coupled_children(left.q_widths,right.q_widths,original)
    restored=swap_coupled_children(right.q_widths,left.q_widths,swapped)
    for q in original: np.testing.assert_array_equal(restored[q],original[q])
    # Condon--Shortley sign is physically checked by all-target test, not just inversion.


@pytest.mark.parametrize('target',[True,-1,6,2.5])
def test_invalid_target_rejected(target):
    tree,_,_,_=_tree()
    with pytest.raises(ValueError): hierarchy_singlet_complement(**tree,target=target)


@pytest.mark.parametrize('tolerance',[np.nan,np.inf,0.,-1.,1.])
def test_invalid_tolerance_rejected(tolerance):
    tree,_,_,_=_tree()
    with pytest.raises(ValueError):
        hierarchy_singlet_complement(**tree,target=0,tolerance=tolerance)
    with pytest.raises(ValueError):
        swap_coupled_children(tree['blocks'][0].q_widths,tree['blocks'][1].q_widths,
            tree['maps'][0],tolerance=tolerance)


@pytest.mark.parametrize('bad_support',[(0,0,2),(0,1.5,2),(0,True,2),(0,-1,2)])
def test_invalid_physical_support_rejected(bad_support):
    tree,_,_,_=_tree()
    tree['supports'][0]=bad_support
    with pytest.raises(ValueError): hierarchy_singlet_complement(**tree,target=0)
