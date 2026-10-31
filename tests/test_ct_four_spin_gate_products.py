import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.dressing.multiplicity_gate import plaquette_unitary
from su2_nqs.dressing.gate_action import apply_local_gate
from su2_nqs.vmc.ct_four_spin_gate_products import (
    four_spin_entry_pairs, make_gate_products_from_queries,
)


def _setup(sites=(5, 0, 3, 2)):
    n = 6
    rows = ((np.arange(64)[:, None] >> np.arange(5, -1, -1)) & 1).astype(np.int32)
    powers = 2**np.arange(5, -1, -1)
    images = np.array([np.arange(n), (np.arange(n)+3) % n,
                       np.arange(n)[::-1], ((np.arange(n)+3) % n)[::-1]])
    bonds = tuple((i, (i+1) % n, .7) for i in range(n))
    bonds += tuple((i, (i+1) % n, -.2) for i in range(n))  # repeated, signed labels
    bonds += tuple((i, i+3, .13) for i in range(3))
    # The default has outside edge1--4 and noncanonical local order.
    rng = np.random.default_rng(61028)
    state = rng.normal(size=64)+1j*rng.normal(size=64)
    state[::3] = 0  # exact source nodes
    state /= np.linalg.norm(state)

    def raw(bank, bits):
        return bank[bits @ jnp.asarray(powers)]

    def factory(edges):
        edge_array = jnp.asarray(edges)
        def query(bank, bits):
            def changed(edge):
                a, b = edge
                return raw(bank, bits.at[a].set(bits[b]).at[b].set(bits[a]))
            return raw(bank, bits), jax.vmap(changed)(edge_array)
        return query

    h = np.zeros((64, 64))
    for old, row in enumerate(rows):
        for a, b, weight in bonds:
            opposite = row[a] != row[b]
            h[old, old] += weight*(-.25 if opposite else .25)
            if opposite:
                changed = row.copy(); changed[a], changed[b] = row[b], row[a]
                h[int(changed @ powers), old] += .5*weight
    features = np.zeros((64, 70), complex)
    density = np.zeros(64)
    pairs = four_spin_entry_pairs()
    for index, row in enumerate(rows):
        for image in images:
            y = row[image]
            new = int(y[list(sites)] @ np.array([8, 4, 2, 1]))
            count = sum(old.bit_count() == new.bit_count() for old in range(16))
            for entry, (new_code, old) in enumerate(pairs):
                if new_code != new:
                    continue
                z = y.copy()
                z[list(sites)] = (old >> np.arange(3, -1, -1)) & 1
                value = state[int(z @ powers)]
                features[index, entry] += value/len(images)
                density[index] += abs(value)**2/(len(images)*count)
    pushed = np.zeros(64)
    total_abs = sum(abs(weight) for _, _, weight in bonds)
    for index, row in enumerate(rows):
        for a, b, weight in bonds:
            z = row.copy(); z[a], z[b] = row[b], row[a]
            pushed[index] += abs(weight)/total_abs*density[int(z @ powers)]
    return rows, images, bonds, sites, state, raw, factory, h, features, density, pushed


@pytest.mark.parametrize('chunk', [1, 2, 4])
def test_all_entries_full_H_and_positive_guides_against_independent_dense_oracle(chunk):
    rows, images, bonds, sites, state, raw, factory, h, expected, q, qp = _setup()
    action, metadata = make_gate_products_from_queries(raw, factory, images, bonds, sites,
                                                      n_sites=6, projection_chunk=chunk)
    phi, hp, actual_q, actual_qp = map(np.asarray, action(jnp.asarray(state), rows))
    assert metadata['entry_count'] == 70
    assert metadata['outside_bond_labels'] == 1
    assert four_spin_entry_pairs() == tuple(sorted(four_spin_entry_pairs()))
    np.testing.assert_allclose(phi, expected, atol=3e-15, rtol=0)
    np.testing.assert_allclose(hp, h @ expected, atol=3e-15, rtol=0)
    np.testing.assert_allclose(actual_q, q, atol=3e-15, rtol=0)
    np.testing.assert_allclose(actual_qp, qp, atol=3e-15, rtol=0)
    assert abs(actual_q.sum()-1) < 3e-15 and abs(actual_qp.sum()-1) < 3e-15
    unitary = plaquette_unitary([.37, -.23, .19, .41])
    coefficients = np.array([unitary[a, b] for a, b in four_spin_entry_pairs()])
    psi, hpsi = phi @ coefficients, hp @ coefficients
    np.testing.assert_allclose(hpsi, h @ psi, atol=3e-15, rtol=0)
    support = np.any(abs(phi)+abs(hp) > 1e-14, axis=1)
    assert np.all((actual_q+actual_qp)[support] > 0)
    # Empty fully polarized sectors correctly have no guide mass and no H
    # support; no uniform floor or fabricated nonzero probability is needed.
    assert np.all(abs(psi[actual_q == 0]) < 1e-14)
    assert np.max(abs(psi[actual_q > 0])**2/actual_q[actual_q > 0]) <= 6+1e-12
    # A coherent target node retains Hpsi and its H² contribution.
    cancel = hp[7].conj()-phi[7].conj()*(phi[7] @ hp[7].conj())/np.vdot(phi[7], phi[7]).real
    cancel /= np.linalg.norm(cancel)
    assert abs(phi[7] @ cancel) < 3e-15
    assert abs(hp[7] @ cancel) > 1e-7
    assert .5*(actual_q[7]+actual_qp[7]) > 0


def test_invalid_graph_images_and_proposal_invariance_are_rejected():
    rows, images, bonds, sites, state, raw, factory, *_ = _setup()
    with pytest.raises(ValueError):
        make_gate_products_from_queries(raw, factory, images, bonds+((0, 2, .3),), sites, n_sites=6)
    # H cancellations must not hide a non-invariant absolute-weight proposal.
    with pytest.raises(ValueError):
        make_gate_products_from_queries(raw, factory, images, bonds+((0, 2, .3), (0, 2, -.3)), sites, n_sites=6)
    with pytest.raises(ValueError):
        make_gate_products_from_queries(raw, factory, images, bonds, (0, 0, 1, 2), n_sites=6)
    with pytest.raises(ValueError):
        make_gate_products_from_queries(raw, factory, images, bonds, sites, n_sites=6, projection_chunk=3)


def test_no_outside_edges_and_gate_before_projection_noncommuting_witness():
    rows, images, bonds, sites, state, raw, factory, h, expected, q, qp = _setup((0, 2, 4, 5))
    action, metadata = make_gate_products_from_queries(raw, factory, images, bonds, sites, n_sites=6)
    phi, hp, actual_q, actual_qp = map(np.asarray, action(state, rows))
    assert metadata['outside_bond_labels'] == 0
    np.testing.assert_allclose(phi, expected, atol=3e-15, rtol=0)
    np.testing.assert_allclose(hp, h @ expected, atol=3e-15, rtol=0)
    np.testing.assert_allclose(actual_q, q, atol=3e-15, rtol=0)
    np.testing.assert_allclose(actual_qp, qp, atol=3e-15, rtol=0)
    gate = plaquette_unitary([.37, -.23, .19, .41])
    coefficients = np.array([gate[a, b] for a, b in four_spin_entry_pairs()])
    powers = 2**np.arange(5, -1, -1)
    def project(vector):
        return np.mean([vector[rows[:, image] @ powers] for image in images], axis=0)
    correct = project(apply_local_gate(state, gate, sites))
    wrong = apply_local_gate(project(state), gate, sites)
    wrong_h = apply_local_gate(h @ project(state), gate, sites)
    np.testing.assert_allclose(phi @ coefficients, correct, atol=3e-15, rtol=0)
    np.testing.assert_allclose(hp @ coefficients, h @ correct, atol=3e-15, rtol=0)
    assert np.max(abs(correct-wrong)) > 1e-3
    assert np.max(abs(h @ correct-wrong_h)) > 1e-3
