"""Positive-rotor bound and independent Cartesian / spin-two analytic checks."""
import numpy as np
import pytest

from su2_nqs.benchmarks.rotor_spectrum_template import (
    conditional_magnetic_rotor_test, integer_spin_generators,
    positive_rotor_spin2_bound, quadratic_rotor_levels,
)


def test_spin_generators_obey_independent_Casimir_and_lie_algebra():
    for q in (0,2,4,6):
        x,y,z=integer_spin_generators(q)
        np.testing.assert_allclose(x @ y-y @ x,1j*z,atol=3e-15,rtol=0)
        np.testing.assert_allclose(x @ x+y @ y+z @ z,
                                  q*(q+2)/4*np.eye(q+1),atol=4e-15,rtol=0)
    for q in (1,-2,True,2.0):
        with pytest.raises(ValueError):
            integer_spin_generators(q)


def test_general_triaxial_bound_matches_independent_analytic_levels():
    rng=np.random.default_rng(20261014)
    for coefficients in ([1,1,1],[0,0,1],[0,1,1],[0,1,100],
                         *rng.lognormal(0,3,size=(250,3))):
        a,b,c=np.asarray(coefficients,dtype=float)
        spin1=np.sort([a+b,a+c,b+c])
        discriminant=a*a+b*b+c*c-a*b-a*c-b*c
        spin2=np.sort([2*(a+b+c)-2*np.sqrt(max(0.,discriminant)),
                       2*(a+b+c)+2*np.sqrt(max(0.,discriminant)),
                       4*a+b+c,a+4*b+c,a+b+4*c])
        # Eigenvalue backward error scales with the largest inertia, not a
        # tiny lowest eigenvalue under extreme triaxial anisotropy.
        absolute_error=64*np.finfo(float).eps*max(1.,max(coefficients))
        np.testing.assert_allclose(quadratic_rotor_levels(coefficients,2),spin1,rtol=0,atol=absolute_error)
        np.testing.assert_allclose(quadratic_rotor_levels(coefficients,4),spin2,rtol=0,atol=absolute_error)
        result=positive_rotor_spin2_bound(coefficients)
        assert result['bound_slack'] >= -1e-11*max(1.,max(coefficients))
        np.testing.assert_allclose(result['variational_trial_expectations'],[3,3,0],atol=5e-15,rtol=0)


def test_actual_magnetic_ratio_is_conditional_and_never_a_qsl_certificate():
    result=conditional_magnetic_rotor_test({'0':[3.0816631571356723],
        '2':[3.1126137366211832], '4':[5.777704455779893]})
    assert result['positive_rotor_template_rejected_if_premises_hold']
    assert result['excess_above_bound_if_premises_hold']>2.6
    assert result['ratio_if_premises_hold']>80
    assert not result['premises_verified'] and not result['qsl_certified']
    assert not result['bulk_magnetic_order_excluded']


def test_the_inertia_constraints_are_required_and_invalid_values_rejected():
    for values in ([-1,1,1],[1,np.inf,1],[1,2]):
        with pytest.raises(ValueError):
            positive_rotor_spin2_bound(values)


def test_low_template_failure_can_coexist_with_a_higher_strict_rotor():
    from projects.clebsch_tree_nqs.explorations.rotor_bound_check_20261014 import rotor_boundary_counterexample
    epsilon,delta=.0600706062,2.4682233359
    result=rotor_boundary_counterexample(epsilon,delta)
    # Independent physical operators for a rotor (S<=2, all body copies)
    # tensored with two literal spin halves. Identify total-spin sectors by
    # the actual combined Casimir rather than copying the coupling table.
    generators=[[],[],[]]
    for q in (0,2,4):
        for axis,operator in enumerate(integer_spin_generators(q)):
            generators[axis].extend([operator]*(q+1))
    def direct_sum(blocks):
        matrix=np.zeros((sum(len(block) for block in blocks),)*2,dtype=complex)
        start=0
        for block in blocks:
            stop=start+len(block)
            matrix[start:stop,start:stop]=block
            start=stop
        return matrix
    rotor=[direct_sum(blocks) for blocks in generators]
    halves=[np.array([[0,.5],[.5,0]],complex),np.array([[0,.5j],[-.5j,0]],complex),
            np.diag([-.5,.5]).astype(complex)]
    edge=[np.kron(s,np.eye(2))+np.kron(np.eye(2),s) for s in halves]
    total=[np.kron(j,np.eye(4))+np.kron(np.eye(35),s) for j,s in zip(rotor,edge)]
    casimir=sum(j @ j for j in total)
    hamiltonian=np.kron(delta/2*sum(j @ j for j in rotor),np.eye(4))
    hamiltonian+=np.kron(np.eye(35),epsilon/2*sum(s @ s for s in edge))
    np.testing.assert_allclose(casimir @ hamiltonian-hamiltonian @ casimir,0,atol=4e-14,rtol=0)
    values,vectors=np.linalg.eigh(casimir)
    q_labels=np.rint(np.sqrt(np.maximum(0,1+4*values))-1).astype(int)
    for q,minimum in result['physical_spin_minima'].items():
        u=vectors[:,q_labels==int(q)]
        actual=np.linalg.eigvalsh(u.conj().T @ hamiltonian @ u)[0]
        np.testing.assert_allclose(actual,minimum,atol=3e-14,rtol=0)
    assert result['positive_so3_rotor_is_present'] and result['strict_su2_symmetry']
    assert result['lowest_positive_rotor_template_fails']
