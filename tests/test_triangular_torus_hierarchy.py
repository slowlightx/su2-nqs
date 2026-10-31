"""Geometry-ledger checks and an independent complete-multiplet N8 oracle."""
import numpy as np
import pytest

from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.lattices.triangular_torus_hierarchy import (
    build_merge_bond_ledger, reconnect_triangular_hierarchy, weighted_layer_cuts,
)
from su2_nqs.operators.su2_block_transfer import BlockIsometry, project_block
from su2_nqs.operators.su2_sector_merge import reduced_block_from_projected, solve_sector_merge
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.quantum_numbers import coupling_range

PAIRS = ((0,5),(1,4),(2,3),(6,7),(8,13),(9,12),(10,11),(14,15))
GROUPS = ((0,5,6,7),(1,2,3,4),(8,13,14,15),(9,10,11,12))


@pytest.mark.parametrize("j2,old_expected,new_expected", [(0., (320.,224.), (288.,208.)),
                                                          (.125, (380.,266.), (342.,247.))])
def test_explicit_reconnection_reuses_every_leaf_and_root_cut_and_routes_all_bonds(j2, old_expected, new_expected):
    old = triangular_schur_hierarchy(12)
    new = reconnect_triangular_hierarchy(old, pairs18=PAIRS, groups36=GROUPS)
    assert new.blocks is old.blocks
    assert all(a is b and a.sites == b.sites for a, b in zip(old.blocks, new.blocks))
    assert set(new.coarse_supports[28]) == set(old.coarse_supports[28])
    assert set(new.coarse_supports[29]) == set(old.coarse_supports[29])
    assert new.coarse_supports[28] != old.coarse_supports[28]  # local coordinate order really changes
    bonds = triangular_j1j2_bonds(12,12,j2=j2,periodic=True)
    assert tuple(weighted_layer_cuts(old,bonds).values()) == old_expected
    assert tuple(weighted_layer_cuts(new,bonds).values()) == new_expected
    ledger = build_merge_bond_ledger([b.sites for b in new.blocks],new.binary_merges,bonds)
    recovered = []
    for sites, local in zip(ledger.supports[:16],ledger.leaf_bonds):
        recovered.extend((*sorted((sites[a],sites[b])),g) for a,b,g in local)
    for (left,right,_), local in zip(new.binary_merges,ledger.merge_bonds):
        recovered.extend((*sorted((ledger.supports[left][a],ledger.supports[right][b])),g) for a,b,g in local)
    assert sorted(recovered) == sorted(ledger.physical_bonds)
    assert len(recovered) == (432 if j2 == 0 else 864)
    assert triangular_schur_hierarchy(12) is old  # opt-in helper did not change the default


def _physical_merge(left, right, maps):
    result = {}
    for q,w in maps.items():
        out = np.zeros((next(iter(left.values())).shape[0]*next(iter(right.values())).shape[0],w.shape[1],q+1),complex)
        row = 0
        for ql,ul in sorted(left.items()):
            for qr,ur in sorted(right.items()):
                if q not in coupling_range(ql,qr): continue
                for a in range(ul.shape[1]):
                    for b in range(ur.shape[1]):
                        for k,m in enumerate(range(-q,q+1,2)):
                            vector = np.zeros(out.shape[0],complex)
                            for ml in range(-ql,ql+1,2):
                                mr=m-ml
                                if abs(mr)<=qr and (mr-qr)%2==0:
                                    vector += clebsch_gordan(ql,ml,qr,mr,q,m)*np.kron(ul[:,a,(ml+ql)//2],ur[:,b,(mr+qr)//2])
                            out[:,:,k] += vector[:,None]*w[row][None,:]
                        row += 1
        result[q] = out
    return result


def _physical_H(vectors,n,bonds):
    rows=np.arange(2**n); value=np.asarray(vectors); out=np.zeros_like(value,dtype=complex)
    for i,j,g in bonds:
        bi=(rows>>(n-i-1))&1; bj=(rows>>(n-j-1))&1
        out += g*((bi-.5)*(bj-.5))[:,None]*value
        mask=bi!=bj; new=rows[mask]^(1<<(n-i-1))^(1<<(n-j-1))
        out[new] += .5*g*value[mask]
    return out


def _exact_N8_tree(merges,bonds):
    leaf={1:np.eye(2,dtype=complex)[:,None,:]}
    block=reduced_block_from_projected(project_block(BlockIsometry(1,leaf),np.zeros((2,2))))
    blocks=[block]*8; physical=[leaf]*8
    ledger=build_merge_bond_ledger([(i,) for i in range(8)],merges,bonds)
    for k,(l,r,parent) in enumerate(merges):
        widths={}
        for ql,dl in blocks[l].q_widths.items():
            for qr,dr in blocks[r].q_widths.items():
                for q in coupling_range(ql,qr): widths[q]=widths.get(q,0)+dl*dr
        retain={0:widths[0]} if parent==14 else widths
        merged=solve_sector_merge(blocks[l],blocks[r],ledger.merge_bonds[k],retain=retain,
                                  compact_boundary=True,max_output_dimension=128)
        blocks.append(merged.block)
        physical.append(_physical_merge(physical[l],physical[r],merged.reduced_maps))
    local=physical[-1][0][:,:,0]
    rows=np.arange(256); global_rows=np.zeros(256,dtype=int)
    for k,site in enumerate(ledger.supports[-1]): global_rows += ((rows>>(7-k))&1)<<(7-site)
    global_basis=np.zeros_like(local); global_basis[global_rows]=local
    return global_basis,blocks[-1].h_by_q[0]


def test_new_tree_reuses_existing_physical_mergers_and_matches_exact_N8_H_all_singlets():
    old=((0,1,8),(2,3,9),(4,5,10),(6,7,11),(8,9,12),(10,11,13),(12,13,14))
    new=((0,2,8),(1,3,9),(4,6,10),(5,7,11),(8,9,12),(10,11,13),(12,13,14))
    bonds=[(i,(i+1)%8,1.) for i in range(8)]+[(i,(i+2)%8,.125) for i in range(8)]
    a,ha=_exact_N8_tree(old,bonds); b,hb=_exact_N8_tree(new,bonds)
    assert a.shape == b.shape == (256,14)
    for u,h in ((a,ha),(b,hb)):
        np.testing.assert_allclose(u.conj().T@u,np.eye(14),atol=8e-13)
        np.testing.assert_allclose(_physical_H(u,8,bonds),u@h,atol=8e-13)
        spin_squared=6*u+2*_physical_H(u,8,[(i,j,1.) for i in range(8) for j in range(i+1,8)])
        np.testing.assert_allclose(spin_squared,0,atol=8e-13)
    f=b.conj().T@a
    np.testing.assert_allclose(f.conj().T@f,np.eye(14),atol=8e-13)
    np.testing.assert_allclose(hb,f@ha@f.conj().T,atol=8e-13)
    np.testing.assert_allclose(np.linalg.eigvalsh(ha),np.linalg.eigvalsh(hb),atol=8e-13)


def test_finite_width_reconnection_is_not_a_state_preserving_tensor_copy():
    singlet=np.array([0.,-1.,1.,0.])/np.sqrt(2)
    old=np.kron(singlet,singlet)
    new=np.zeros(16)
    for bit in range(16):
        bits=[(bit>>(3-i))&1 for i in range(4)]
        new[bit]=singlet[2*bits[0]+bits[2]]*singlet[2*bits[1]+bits[3]]
    # A width-one singlet block maps to a different physical ray when the
    # pairing is changed. Two such fixed four-spin halves form an N8 example.
    assert abs(np.vdot(old,new))**2 == pytest.approx(.25,abs=2e-15)
    assert abs(np.vdot(np.kron(old,old),np.kron(new,new)))**2 == pytest.approx(.0625,abs=2e-15)


def test_invalid_reconnections_and_duplicate_bond_aggregation():
    old=triangular_schur_hierarchy(12)
    with pytest.raises(ValueError,match="L=12"):
        reconnect_triangular_hierarchy(triangular_schur_hierarchy(6),pairs18=PAIRS,groups36=GROUPS)
    crossed=((0,8),(1,4),(2,3),(5,13),(6,7),(9,12),(10,11),(14,15))
    with pytest.raises(ValueError,match="72"):
        reconnect_triangular_hierarchy(old,pairs18=crossed,groups36=GROUPS)
    with pytest.raises(ValueError,match="whole"):
        reconnect_triangular_hierarchy(old,pairs18=PAIRS,groups36=((0,1,2,3),(4,5,6,7),(8,9,10,11),(12,13,14,15)))
    with pytest.raises(ValueError,match="partition"):
        build_merge_bond_ledger([(0,1),(1,2)],[(0,1,2)],[(0,1,1.)])
    ledger=build_merge_bond_ledger([(0,),(1,),(2,)],[(0,1,3),(3,2,4)],
        [(0,1,.5),(1,0,-.5),(0,2,.3),(2,0,.2)])
    assert ledger.physical_bonds == ((0,2,.5),)
    assert ledger.merge_bonds == ((),((0,0,.5),))
