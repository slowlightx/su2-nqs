"""Independent dense spin-half oracle for the compressed magnetic control."""
from math import comb

import numpy as np
import pytest

from su2_nqs.benchmarks.coplanar_entanglement_control import (
    coplanar_collective_singlet_entanglement,
)


def test_N6_control_matches_independent_spin_half_singlet_and_exact_partial_trace():
    report = coplanar_collective_singlet_entanglement(2,1)
    # Each ordered pair is projected onto spin one, then three spin-one
    # irreps form the normalized epsilon singlet. Explicit triplet tensors
    # and Levi-Civita construction are independent of the CG implementation.
    down = np.array([1,0], dtype=complex)
    up = np.array([0,1], dtype=complex)
    spherical = [np.kron(down,down), (np.kron(down,up)+np.kron(up,down))/np.sqrt(2),
                 np.kron(up,up)]
    cartesian = [(spherical[0]-spherical[2])/np.sqrt(2),
                 1j*(spherical[0]+spherical[2])/np.sqrt(2), spherical[1]]
    state = np.zeros(64, dtype=complex)
    for i,j,k,sign in [(0,1,2,1),(1,2,0,1),(2,0,1,1),
                       (2,1,0,-1),(0,2,1,-1),(1,0,2,-1)]:
        state += sign*np.kron(np.kron(cartesian[i],cartesian[j]),cartesian[k])/np.sqrt(6)
    # Region sites 0,2,4; complement sites 1,3,5.
    matrix = state.reshape((2,)*6).transpose(0,2,4,1,3,5).reshape(8,8)
    rho = matrix @ matrix.conj().T
    np.testing.assert_allclose(report['region_rho'],rho,atol=4e-15,rtol=0)
    assert report['strict_su2_verified']
    assert report['physical_raising_commutator_relative_norm'] < 4e-15


@pytest.mark.parametrize('n,a', [(4,2),(6,3),(12,6)])
def test_magnetic_rotor_counting_is_multiplicity_not_just_magnetic_degeneracy(n,a):
    report = coplanar_collective_singlet_entanglement(n,a)
    assert report['normalization_error'] < 3e-14
    assert report['cg_orthonormality_max_error'] < 2e-14
    assert report['physical_raising_commutator_relative_norm'] < 4e-14
    # Three sublattice spins: low S has 2S+1 independent SU2 multiplets.
    for q,sector in report['q_sectors'].items():
        if int(q) <= a:
            assert sector['multiplicity_dimension'] == int(q)+1
    assert sum(s['weight'] for s in report['q_sectors'].values()) == pytest.approx(1.,abs=4e-14)


def test_illegal_half_odd_global_collective_spins_are_rejected():
    for args in [(3,1),(4,0),(4,4),(True,1),(4,1.5)]:
        with pytest.raises(ValueError):
            coplanar_collective_singlet_entanglement(*args)
