"""Independent dense controls for the bounded physical state-span audit."""
import numpy as np
import pytest

from projects.clebsch_tree_nqs.explorations.topological_branch_audit_20261013_v1 import (
    restriction, toric_code_states, whiten,
)
from su2_nqs.benchmarks.spin_rotation_calibration import manifold_region_algebra, singlet_covering_state


def test_actual_Gram_rejects_duplicate_states_and_keeps_complex_coherence():
    first = singlet_covering_state(4,[(0,1),(2,3)])
    second = singlet_covering_state(4,[(0,2),(1,3)])
    states = np.column_stack((first,second))
    q,gram,eigenvalues = whiten(states)
    np.testing.assert_allclose(q.conj().T@q,np.eye(2),atol=2e-14,rtol=0)
    np.testing.assert_allclose(eigenvalues,[.5,1.5],atol=2e-14,rtol=0)
    with pytest.raises(ValueError):
        whiten(np.column_stack((first,first)))
    mixed = states@np.array([[1,.2j],[.3+1j,.7]],complex)
    original = manifold_region_algebra(states,[0,1])
    transformed = manifold_region_algebra(mixed,[0,1])
    assert transformed['local_algebra_defect_frobenius'] == pytest.approx(original['local_algebra_defect_frobenius'],abs=2e-14)


def test_singlet_one_site_code_pass_is_not_topology_and_physical_swap_is_complete():
    states = np.column_stack((singlet_covering_state(4,[(0,1),(2,3)]),
                              singlet_covering_state(4,[(0,2),(1,3)])))
    assert manifold_region_algebra(states,[0])['local_algebra_defect_frobenius'] < 2e-14
    assert manifold_region_algebra(states,[0,1])['local_algebra_defect_frobenius'] == pytest.approx(np.sqrt(2/3),abs=2e-14)
    packed = np.arange(16)
    bits = (packed[:,None] >> np.arange(3,-1,-1)) & 1
    pullback = bits[:,[1,0,2,3]] @ (1 << np.arange(3,-1,-1))
    q,_,_ = whiten(states)
    result = restriction(q,q[pullback])
    matrix = np.array(result['matrix_real'])+1j*np.array(result['matrix_imaginary'])
    np.testing.assert_allclose(np.linalg.eigvalsh(matrix),[-1,1],atol=2e-14,rtol=0)
    assert result['closure_residual_frobenius'] < 2e-14
    assert result['unitarity_error'] < 2e-14


def test_exact_Z2_code_separates_contractible_algebra_and_logical_loop():
    states,stars,plaquettes,logical_z = toric_code_states()
    np.testing.assert_array_equal(states.conj().T@states,np.eye(4))
    packed = np.arange(len(states))
    for star in stars:
        np.testing.assert_array_equal(states[packed^star],states)
    for mask in plaquettes:
        sign = np.array([(-1.)**int((int(x)&mask).bit_count()) for x in packed])
        np.testing.assert_array_equal(sign[:,None]*states,states)
    assert manifold_region_algebra(states,[0,1,2])['local_algebra_defect_frobenius'] == 0
    logical = manifold_region_algebra(states,[0,2,4])
    assert logical['local_algebra_defect_frobenius'] > .5
    sign = np.array([(-1.)**int((int(x)&logical_z[0]).bit_count()) for x in packed])
    matrix = restriction(states,sign[:,None]*states)
    np.testing.assert_array_equal(matrix['matrix_real'],np.diag([1,-1,1,-1]))
