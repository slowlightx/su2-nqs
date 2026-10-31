from itertools import permutations

import numpy as np

from su2_nqs.benchmarks.su2_multiplicity_intervention import pair_spin_projectors
from su2_nqs.benchmarks.su2_six_body_chirality import six_body_chirality_operators,six_body_operator_diagnostics


def test_independent_six_pauli_epsilon_oracle():
    # Explicit epsilon products, independent of the commutator implementation.
    pauli=[np.array([[0,1],[1,0]],complex),np.array([[0,1j],[-1j,0]],complex),np.diag([-1,1])]
    oracle=np.zeros((64,64),complex)
    for abc in permutations(range(3)):
        sign_a=(-1)**sum(abc[i]>abc[j] for i in range(3) for j in range(i+1,3))
        for xyz in permutations(range(3)):
            sign_b=(-1)**sum(xyz[i]>xyz[j] for i in range(3) for j in range(i+1,3))
            components=dict(zip((0,3,4),abc));components.update(zip((1,5,2),xyz))
            term=np.ones((1,1),complex)
            for i in range(6):term=np.kron(term,pauli[components[i]])
            oracle+=sign_a*sign_b*term/12
    np.testing.assert_allclose(six_body_chirality_operators()['T'],oracle,atol=2e-15)


def test_six_body_support_spin_resolution_and_dephasing():
    diagnostics=six_body_operator_diagnostics()
    for key,value in diagnostics.items():
        if key.endswith('_error') or key.startswith('max_') or key=='hermiticity_frobenius':
            assert value<2e-14,(key,value)
    assert abs(diagnostics['norm_frobenius']-4)<2e-14
    t=six_body_chirality_operators()['T']
    for pair in ((0,3),(3,4),(0,4)):
        ps=pair_spin_projectors(6,pair)
        assert np.linalg.norm(sum(p@t@p for p in ps.values()))<2e-14
