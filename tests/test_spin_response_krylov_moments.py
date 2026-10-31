"""All-support importance third moments agree with an independent dense H."""
from itertools import product

import numpy as np

from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
from su2_nqs.benchmarks.spin_response_krylov_moments import two_action_plan, third_source_moments


def test_exact_third_moment_with_complex_parent_nodes_and_repeated_bonds():
    n = 6
    bits = np.array([x for x in product((0, 1), repeat=n) if sum(x) == 3], np.int8)
    powers = 1 << np.arange(n-1, -1, -1)
    words = bits @ powers
    rng = np.random.default_rng(141036)
    psi = rng.normal(size=len(bits)) + 1j*rng.normal(size=len(bits))
    psi[::2] = 0
    psi /= np.linalg.norm(psi)
    lookup = {int(word): k for k, word in enumerate(words)}
    bonds = [(0, 1, .7), (0, 1, .3), (1, 3, .2), (3, 5, -.1), (2, 4, 1.1)]
    plan = two_action_plan(bits, bonds)
    queried = psi[[lookup[int(w)] for w in plan.configurations @ powers]]
    density = np.mean(abs(queried[plan.proposal_indices])**2, axis=1)
    active = density > 0
    plan = two_action_plan(bits[active], bonds)
    queried = psi[[lookup[int(w)] for w in plan.configurations @ powers]]
    with np.errstate(divide='ignore'):
        logs = np.log(queried.astype(complex))
    phase = np.exp(2j*np.pi*np.arange(n)/n)
    moments = third_source_moments(plan, logs, {'k': phase}, energy_shift=-1.2)
    h = sum((c*spin_dot_matrix(n, i, j) for i,j,c in bonds), np.zeros((2**n, 2**n)))
    d = h + 1.2*np.eye(2**n)
    full = np.zeros(2**n, complex); full[words] = psi
    for name, source in [('parent', full), ('k', full*((np.array(list(product((0,1), repeat=n)))-.5)@phase/np.sqrt(n)))]:
        vector = source.copy()
        for power in range(4):
            target = np.vdot(source, vector)
            observed = density[active] @ moments[name+'__m'+str(power)]
            np.testing.assert_allclose(observed, target.real, atol=8e-14, rtol=0)
            vector = d @ vector
        np.testing.assert_allclose(density[active] @ moments[name+'__m3_imag'], 0., atol=8e-14)
