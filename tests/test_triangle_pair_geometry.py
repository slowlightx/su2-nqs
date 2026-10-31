"""Exact torus distances/orientation and independent physical embedding."""
import numpy as np
import pytest
from itertools import permutations

from su2_nqs.benchmarks.multipartite_geometry import partial_trace_sites
from su2_nqs.benchmarks.su2_six_body_chirality import six_body_chirality_operators
from su2_nqs.benchmarks.triangle_pair_geometry import (
    periodic_triangle,triangle_pair_geometry,ccw_triangle,
    frozen_six_site_order,disjoint_elementary_triangle_pairs,
)


def test_same_anchor_near_and_far_have_exact_distances_and_CCW_faces():
    a=(12,18,19)
    near=triangle_pair_geometry(a,(13,20,14))
    far=triangle_pair_geometry(a,(14,21,15))
    assert near['shortest_center_distance_squared_numerator_over9']==21
    assert far['shortest_center_distance_squared_numerator_over9']==57
    assert near['A']['orientation']==near['B']['orientation']==far['B']['orientation']==1
    assert not near['same_triangle_flavor'] and not far['same_triangle_flavor']
    assert frozen_six_site_order(a,(14,21,15))==(12,14,15,18,19,21)


def test_periodic_translation_preserves_pair_geometry():
    shift=lambda s: ((s//6+4)%6)*6+(s%6+5)%6
    a,b=(12,18,19),(14,21,15)
    original=triangle_pair_geometry(a,b)
    translated=triangle_pair_geometry(tuple(map(shift,a)),tuple(map(shift,b)))
    assert translated['shortest_center_distance_squared_numerator_over9']==57
    assert translated['A']['orientation']==original['A']['orientation']
    assert translated['A']['flavor']==original['A']['flavor']


def test_name_is_not_a_distance_sequence():
    assert len(disjoint_elementary_triangle_pairs([12,13,14,18,19,20]))==1
    assert len(disjoint_elementary_triangle_pairs([12,13,14,15,16,17]))==0
    assert len(disjoint_elementary_triangle_pairs([20,21,15,8,7,13]))==0
    assert len(disjoint_elementary_triangle_pairs([20,21,15,8,7,13,14]))==0
    pairs=disjoint_elementary_triangle_pairs([12,13,14,15,18,19,20,21])
    assert len(pairs)==6
    assert sorted(set(p['shortest_center_distance_squared_numerator_over9'] for p in pairs))==[21,36,57]


def test_invalid_and_overlapping_triangles_are_not_disjoint_six_spin_probes():
    with pytest.raises(ValueError):periodic_triangle((0,1,2))
    with pytest.raises(ValueError):periodic_triangle((0,True,7))
    assert triangle_pair_geometry((0,6,7),(0,7,1))['overlap']==[0,7]
    with pytest.raises(ValueError):frozen_six_site_order((0,6,7),(0,7,1))
    assert periodic_triangle(ccw_triangle((12,13,19)))['orientation']==1


def test_frozen_T_after_partial_trace_matches_literal_8_spin_chirality_product():
    # No geometry/fusion assumption in this oracle: literal physical8 spin
    # matrices and random complex full state independently fix orientation.
    sites=[12,13,14,15,18,19,20,21]
    a,b=(12,18,19),(14,21,15)
    order=frozen_six_site_order(a,b)
    keep=tuple(sites.index(s) for s in order)
    rng=np.random.default_rng(202610281109)
    psi=rng.normal(size=256)+1j*rng.normal(size=256);psi/=np.linalg.norm(psi)
    rho=np.outer(psi,psi.conj())
    six=partial_trace_sites(rho,keep)
    target=six_body_chirality_operators()['T']
    pauli=[np.array([[0,1],[1,0]],complex),np.array([[0,1j],[-1j,0]],complex),np.diag([-1,1])]
    literal=np.zeros((256,256),complex)
    for abc in permutations(range(3)):
        sa=(-1)**sum(abc[i]>abc[j] for i in range(3) for j in range(i+1,3))
        for xyz in permutations(range(3)):
            sb=(-1)**sum(xyz[i]>xyz[j] for i in range(3) for j in range(i+1,3))
            components={sites.index(s):p for s,p in zip(a,abc)}
            components.update({sites.index(s):p for s,p in zip(b,xyz)})
            term=np.ones((1,1),complex)
            for i in range(8):term=np.kron(term,pauli[components[i]] if i in components else np.eye(2))
            literal+=sa*sb*term/12
    np.testing.assert_allclose(np.trace(six@target),np.vdot(psi,literal@psi),atol=3e-16)
