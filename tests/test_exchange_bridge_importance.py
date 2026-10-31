"""Independent finite-support checks for the fixed-parent importance audit.

These tests deliberately include repeated bonds, negative couplings, null
exchanges, complex amplitudes, and exact target nodes. No neural training or
N=36 calculation is needed to establish these sampling identities.
"""
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.analytic_plaquette_carrier import aligned_plaquette_a1_local_data
from su2_nqs.benchmarks.coherent_plaquette_sampling import plaquette_frame_component_amplitudes
from su2_nqs.vmc.exchange_bridge_importance import (
    importance_audit,
    make_amplitude_heisenberg_action,
    make_exchange_bridge,
    make_coherent_exchange_bridge_products,
    make_exchange_record_bridge,
)
from su2_nqs.vmc.frame_metropolis import frame_proposal
from su2_nqs.vmc.resumable_checkpoint import (
    load_resumable_checkpoint,
    save_resumable_checkpoint,
)
from su2_nqs.vmc.transition_moments import estimate_transition_moments

jax.config.update("jax_enable_x64", True)

FRAMES = (((0, 1, 2, 3), (4, 5, 6, 7)), ((1, 2, 3, 4), (5, 6, 7, 0)))
FRAME_WEIGHTS = np.asarray([.3, .7])
# Repeated bonds must retain their total proposal probability and H coupling.
BONDS = ((0, 1, 1.), (1, 2, .25), (0, 1, .6),
         (4, 6, -.3), (3, 7, .1), (1, 5, .4))
EDGES = tuple((i, j) for i, j, _ in BONDS)
EDGE_PROBABILITIES = np.abs([coupling for _, _, coupling in BONDS])
EDGE_PROBABILITIES /= EDGE_PROBABILITIES.sum()
POWERS = 1 << np.arange(7, -1, -1)


def _basis():
    bits = np.zeros((70, 8), dtype=np.int32)
    for row, up in enumerate(combinations(range(8), 4)):
        bits[row, list(up)] = 1
    return bits


def _setup(bridge_fraction=.4):
    bits = _basis()
    local, _ = aligned_plaquette_a1_local_data()
    log_base, draw_base = frame_proposal(
        FRAMES, FRAME_WEIGHTS, abs(local)**2, uniform_fraction=.1,
    )
    density, draw = make_exchange_bridge(
        log_base, draw_base, EDGES, EDGE_PROBABILITIES,
        n_sites=8, bridge_fraction=bridge_fraction,
    )
    core = abs(plaquette_frame_component_amplitudes(bits, FRAMES))**2 @ FRAME_WEIGHTS
    base = .9 * core + .1 / 70
    # Forward push is independent of the production reverse-density formula.
    indices = {tuple(row): i for i, row in enumerate(bits)}
    pushed = np.zeros(70)
    for old, row in enumerate(bits):
        for (i, j), probability in zip(EDGES, EDGE_PROBABILITIES, strict=True):
            new = row.copy()
            new[i], new[j] = row[j], row[i]
            pushed[indices[tuple(new)]] += probability * base[old]
    mixed = (1 - bridge_fraction) * base + bridge_fraction * pushed
    return bits, density, draw, base, pushed, mixed


def _dense_hamiltonian():
    """Independent tensor-product spin matrices, bit zero = spin down."""
    identity = np.eye(2)
    z = np.diag([-.5, .5])
    plus = np.asarray([[0., 0.], [1., 0.]])
    minus = plus.T
    result = np.zeros((256, 256), dtype=np.complex128)
    for i, j, coupling in BONDS:
        for left, right, coefficient in ((z, z, 1.), (plus, minus, .5), (minus, plus, .5)):
            term = np.ones((1, 1))
            for site in range(8):
                term = np.kron(term, left if site == i else right if site == j else identity)
            result += coupling * coefficient * term
    return result


def _complex_table_with_nodes(bits):
    core = plaquette_frame_component_amplitudes(bits, FRAMES) @ FRAME_WEIGHTS
    psi = core + 1e-3j * np.arange(len(bits))
    psi[::7] = 0.
    table = np.zeros(256, dtype=np.complex128)
    table[bits @ POWERS] = psi
    return table


def test_bridge_density_matches_forward_push_with_duplicates_and_null_swaps():
    bits, density, _, base, pushed, mixed = _setup()
    log_mixed, log_base, log_pushed = jax.jit(density)(jnp.asarray(bits))
    for observed, expected in ((log_mixed, mixed), (log_base, base), (log_pushed, pushed)):
        np.testing.assert_allclose(np.exp(observed), expected, atol=2e-15, rtol=2e-14)
        assert abs(np.exp(observed).sum() - 1) < 3e-14
    assert np.all(mixed >= .1 / 70 * (1 - 1e-13))


def test_bridge_draw_frequencies_use_full_mixture_and_preserve_fixed_m():
    bits, _, draw, _, _, mixed = _setup()
    states, used_bridge = jax.jit(lambda key: draw(key, 40000))(jax.random.PRNGKey(2026100211))
    states = np.asarray(states)
    observed = np.bincount(states @ POWERS, minlength=256)[bits @ POWERS] / len(states)
    np.testing.assert_allclose(observed, mixed, atol=.006, rtol=0)
    assert np.all(states.sum(axis=1) == 4)
    assert np.asarray(used_bridge).shape == (40000,)
    assert abs(np.asarray(used_bridge).mean() - .4) < .012


def test_direct_amplitude_action_matches_tensor_spin_hamiltonian_at_nodes():
    bits = _basis()
    table = _complex_table_with_nodes(bits)
    amplitude = lambda params, row: params[jnp.sum(row * jnp.asarray(POWERS))]
    action = make_amplitude_heisenberg_action(amplitude, BONDS, n_sites=8)
    psi, hpsi = action(jnp.asarray(table), jnp.asarray(bits))
    dense = _dense_hamiltonian()
    expected = (dense @ table)[bits @ POWERS]
    np.testing.assert_allclose(dense, dense.conj().T, atol=0, rtol=0)
    np.testing.assert_allclose(psi, table[bits @ POWERS], atol=0, rtol=0)
    np.testing.assert_allclose(hpsi, expected, atol=3e-15, rtol=2e-14)
    nodes = np.asarray(psi) == 0
    assert nodes.sum() == 10
    assert np.linalg.norm(np.asarray(hpsi)[nodes]) > .01
    assert np.isfinite(hpsi).all()


def test_exhaustive_proposal_weighted_gram_recovers_norm_energy_and_h_squared():
    bits, density, _, _, _, _ = _setup()
    probability = np.exp(np.asarray(density(jnp.asarray(bits))[0]))
    table = _complex_table_with_nodes(bits)
    codes = bits @ POWERS
    psi = table[codes]
    hpsi = (_dense_hamiltonian() @ table)[codes]
    states = np.stack((psi, hpsi), axis=1)
    estimate = estimate_transition_moments(
        states, states[:, None, :], probability,
        observation_weights=probability,
    )
    gram = estimate.moments[0]
    expected = states.conj().T @ states
    np.testing.assert_allclose(gram, expected, atol=3e-15, rtol=2e-14)
    norm = gram[0, 0].real
    energy = gram[0, 1] / norm
    second_moment = gram[1, 1].real / norm
    np.testing.assert_allclose(energy, np.vdot(psi, hpsi) / np.vdot(psi, psi), atol=2e-15)
    np.testing.assert_allclose(second_moment, np.vdot(hpsi, hpsi).real / np.vdot(psi, psi).real, atol=2e-15)
    residual = np.linalg.norm(hpsi - energy * psi)**2 / norm
    assert second_moment - abs(energy)**2 == pytest.approx(residual, abs=3e-15)
    # An estimator that multiplies |psi|^2 by a local-energy ratio at nodes
    # loses this positive second-moment contribution.
    node_fraction = np.sum(abs(hpsi[psi == 0])**2) / np.sum(abs(hpsi)**2)
    assert node_fraction > .03


def test_bridge_draw_checkpoint_replays_exactly_without_parameter_updates(tmp_path):
    _, _, draw, _, _, _ = _setup()
    executable = jax.jit(lambda key: draw(key, 32))
    key = jax.random.PRNGKey(2026100212)
    next_key, sample_key = jax.random.split(key)
    first = executable(sample_key)
    path = tmp_path / "audit.pkl"
    identity = {"kind": "fixed_parent_importance", "n_sites": 8}
    save_resumable_checkpoint(
        path, np.asarray([1., 2.]), optimizer_state=None, completed_steps=0,
        next_sample_index=32, run_identity=identity, history=[],
        metadata={"key": np.asarray(next_key), "states": np.asarray(first[0])},
    )
    restored = load_resumable_checkpoint(path, expected_run_identity=identity)
    assert restored["completed_steps"] == 0
    assert restored["optimizer_state"] is None
    left = executable(next_key)
    right = executable(jnp.asarray(restored["metadata"]["key"]))
    for actual, expected in zip(left, right, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_ratio_audit_retains_joint_covariance_complex_energy_and_scale_invariance():
    bits, _, _, _, _, probability_full = _setup()
    table = _complex_table_with_nodes(bits)
    codes = bits @ POWERS
    psi_full = table[codes]
    hpsi_full = (_dense_hamiltonian() @ table)[codes]
    indices = np.random.default_rng(2026100213).choice(70, 512, p=probability_full)
    psi, hpsi, probability = psi_full[indices], hpsi_full[indices], probability_full[indices]
    groups = np.repeat(np.arange(16), 32)
    kwargs = dict(n_sites=8, bootstrap_replicates=200, seed=2026100214)
    observed = importance_audit(psi, hpsi, probability, groups, **kwargs)
    scaled = importance_audit((2-3j)*psi, (2-3j)*hpsi, probability, groups, **kwargs)

    z = abs(psi)**2 / probability
    numerator = psi.conj()*hpsi / probability
    second = abs(hpsi)**2 / probability
    energy, moment = numerator.sum()/z.sum(), second.sum()/z.sum()
    expected = np.asarray([energy.real/8, energy.imag/8, moment, (moment-abs(energy)**2)/8])
    u_energy = (numerator-energy*z)/z.mean()
    u_moment = (second-moment*z)/z.mean()
    influences = np.stack((u_energy.real/8, u_energy.imag/8, u_moment,
                           (u_moment-2*np.real(energy.conjugate()*u_energy))/8), axis=1)
    iid_se = np.std(influences, axis=0, ddof=1)/np.sqrt(512)
    leave = []
    for group in range(16):
        keep = groups != group
        e = numerator[keep].sum()/z[keep].sum()
        m = second[keep].sum()/z[keep].sum()
        leave.append([e.real/8, e.imag/8, m, (m-abs(e)**2)/8])
    leave = np.asarray(leave)
    jackknife_se = np.sqrt(15/16 * np.sum((leave-leave.mean(axis=0))**2, axis=0))
    names = ("energy_per_site", "imaginary_energy_per_site", "second_moment_total", "variance_per_site")
    for i, name in enumerate(names):
        result = observed["metrics"][name]
        assert result["value"] == pytest.approx(expected[i], abs=3e-14)
        assert result["iid_delta_se"] == pytest.approx(iid_se[i], abs=3e-14)
        assert result["group_jackknife_se"] == pytest.approx(jackknife_se[i], abs=3e-14)
        for key in ("value", "iid_delta_se", "group_jackknife_se", "group_bootstrap_se", "group_bootstrap_95"):
            np.testing.assert_allclose(result[key], scaled["metrics"][name][key], atol=3e-14, rtol=3e-13)
    assert abs(expected[1]) > 1e-5  # Sample Im(E) must not be silently projected out.
    assert observed["exact_node_samples"] > 0
    assert observed["unseen_tail_coverage_certified"] is False
    assert observed["ratio_estimates_finite_sample_biased"] is True
    residual = np.sum(abs(hpsi-energy*psi)**2/probability)/z.sum()/8
    assert observed["stable_residual_variance_per_site"] == pytest.approx(residual, abs=3e-14)


def _four_spin_bridge_fixture(images, probabilities=None):
    rows = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.int32)
    powers = jnp.asarray([8, 4, 2, 1])
    def query_factory(edges):
        edges = jnp.asarray(edges)
        def query(table, bits):
            swaps = jnp.broadcast_to(bits, (len(edges), 4))
            index = jnp.arange(len(edges))
            swaps = swaps.at[index, edges[:, 0]].set(bits[edges[:, 1]])
            swaps = swaps.at[index, edges[:, 1]].set(bits[edges[:, 0]])
            return table[bits @ powers], table[swaps @ powers]
        return query
    # Repeated reversed edge, negative edge, explicitly zero H label.
    bonds = [(0, 1, 1.), (1, 0, -.25), (1, 2, -.5), (2, 3, .75), (3, 0, .4), (0, 2, 0.)]
    if probabilities is None:
        probabilities = np.array([.1, .2, .15, .2, .25, .1])
    products, meta = make_coherent_exchange_bridge_products(
        query_factory, images, bonds, probabilities, n_sites=4,
        projection_chunk=1 if len(images) == 1 else 2)
    dense = np.zeros((16, 16))
    for old, row in enumerate(rows):
        for a, b, coupling in bonds:
            dense[old, old] += coupling * (.25 if row[a] == row[b] else -.25)
            if row[a] != row[b]:
                changed = row.copy(); changed[a], changed[b] = row[b], row[a]
                dense[changed @ np.asarray(powers), old] += .5 * coupling
    return rows, bonds, probabilities, dense, products, meta


def test_projected_bridge_coherence_precedes_square_and_inverse_images_are_correct():
    images = np.array([np.roll(np.arange(4), k) for k in range(4)], np.int32)
    rows, bonds, probabilities, dense, products, meta = _four_spin_bridge_fixture(images)
    rng = np.random.default_rng(2610241)
    table = rng.normal(size=16) + 1j * rng.normal(size=16)
    powers = np.array([8, 4, 2, 1])
    projected = np.mean(table[rows[:, images] @ powers], axis=1)
    expected_push = np.zeros(16)
    for (a, b, _), probability in zip(bonds, probabilities, strict=True):
        swapped = rows.copy(); swapped[:, a], swapped[:, b] = rows[:, b], rows[:, a]
        expected_push += probability * abs(projected[swapped @ powers]) ** 2
    psi, hp, push = map(np.asarray, products(jnp.asarray(table), jnp.asarray(rows)))
    np.testing.assert_allclose(psi, projected, atol=7e-16, rtol=2e-14)
    np.testing.assert_allclose(hp, dense @ projected, atol=1e-15, rtol=2e-14)
    np.testing.assert_allclose(push, expected_push, atol=1e-15, rtol=2e-14)
    # H need not commute with the supplied image average: direct H(Ppsi), not P(Hpsi).
    wrong_order = np.mean((dense @ table)[rows[:, images] @ powers], axis=1)
    assert np.max(abs(wrong_order - hp)) > .1
    assert meta['physical_unique_edges'] == 5
    assert meta['raw_query_unique_edges'] == 6  # inverse-image diagonal closure
    assert abs(push.sum() - np.vdot(psi, psi).real) < 5e-14


def test_unknown_common_Z_bridge_node_variance_and_cross_H_hermiticity_exhaustive():
    images = np.arange(4, dtype=np.int32)[None]
    rows, bonds, probabilities, dense, products, _ = _four_spin_bridge_fixture(images)
    tables = np.zeros((2, 16), complex)
    tables[0, [3, 12]] = [2. + .3j, -.7j]
    tables[1, [3, 5]] = [-.4 + .8j, .9 - .2j]
    evaluated = [tuple(map(np.asarray, products(jnp.asarray(t), jnp.asarray(rows)))) for t in tables]
    psi, hp = np.column_stack([e[0] for e in evaluated]), np.column_stack([e[1] for e in evaluated])
    q0 = np.sum(abs(psi) ** 2, axis=1)
    qpush = np.sum([e[2] for e in evaluated], axis=0)
    # Independently push forward the unnormalized density, retaining nulls/duplicates.
    forward = np.zeros(16)
    powers = np.array([8, 4, 2, 1])
    for old, row in enumerate(rows):
        for (a, b, _), probability in zip(bonds, probabilities, strict=True):
            swapped = row.copy(); swapped[a], swapped[b] = row[b], row[a]
            forward[swapped @ powers] += probability * q0[old]
    np.testing.assert_allclose(qpush, forward, atol=2e-15)
    assert qpush.sum() == pytest.approx(q0.sum(), abs=5e-15)
    qmix = .5 * (q0 + qpush)
    positive = qmix > 0
    np.testing.assert_allclose(hp[~positive], 0., atol=0)
    assert np.linalg.norm(hp[q0 == 0]) > .1
    sampling_probability = qmix[positive] / q0.sum()
    weights = sampling_probability / qmix[positive]
    gram = psi[positive].conj().T @ (weights[:, None] * psi[positive])
    cross_h = psi[positive].conj().T @ (weights[:, None] * hp[positive])
    h2 = hp[positive].conj().T @ (weights[:, None] * hp[positive])
    np.testing.assert_allclose(cross_h, cross_h.conj().T, atol=2e-15)
    for k in range(2):
        energy = cross_h[k, k] / gram[k, k]
        variance = h2[k, k].real / gram[k, k].real - abs(energy) ** 2
        reference = np.linalg.norm(hp[:, k] - energy * psi[:, k]) ** 2 / np.vdot(psi[:, k], psi[:, k]).real
        assert variance == pytest.approx(reference, abs=3e-15)
        # Arbitrary common Q normalization cancels, without declaring raw moments unbiased.
        scaled_cross = cross_h[k, k] / 37.
        assert scaled_cross / (gram[k, k] / 37.) == pytest.approx(energy, abs=2e-15)
    assert np.sum(abs(hp[q0 == 0]) ** 2) > .1


def test_record_bridge_replays_preserves_chain_order_and_fixed_spin():
    rows, bonds, p, _, _, _ = _four_spin_bridge_fixture(np.arange(4)[None])
    base = np.repeat(rows[[3, 5, 6, 9, 10, 12]], 4000, axis=0)
    push = make_exchange_record_bridge([(a, b) for a, b, _ in bonds], p,
                                      n_sites=4, bridge_fraction=.5)
    key = jax.random.PRNGKey(2610242)
    changed, use, labels = map(np.asarray, push(key, jnp.asarray(base)))
    replay = push(key, jnp.asarray(base))
    for actual, repeated in zip((changed, use, labels), replay, strict=True):
        np.testing.assert_array_equal(actual, repeated)
    assert np.all(changed.sum(axis=1) == 2)
    np.testing.assert_array_equal(changed[~use], base[~use])
    for k in np.flatnonzero(use):
        a, b, _ = bonds[labels[k]]
        expected = base[k].copy(); expected[a], expected[b] = base[k, b], base[k, a]
        np.testing.assert_array_equal(changed[k], expected)
    np.testing.assert_allclose(np.bincount(labels, minlength=len(p)) / len(labels), p, atol=.009, rtol=0)
    assert abs(use.mean() - .5) < .012


def test_coherent_bridge_rejects_incomplete_H_support_and_invalid_probabilities():
    factory = lambda edges: None
    with pytest.raises(ValueError, match='every nonzero H edge'):
        make_coherent_exchange_bridge_products(factory, np.arange(4)[None],
            [(0, 1, 1.), (1, 2, 1.)], [1., 0.], n_sites=4, projection_chunk=1)
    for p in ([.2], [np.nan], [-1.], [2.]):
        with pytest.raises(ValueError, match='probabilities'):
            make_exchange_record_bridge([(0, 1)], p, n_sites=4)


def test_H_connection_bridge_has_pointwise_node_second_moment_Cauchy_bound():
    absolute_j = np.abs([1., -.25, -.5, .75, .4, 0.])
    rows, _, _, dense, products, _ = _four_spin_bridge_fixture(
        np.arange(4, dtype=np.int32)[None], absolute_j / absolute_j.sum())
    tables = np.zeros((2, 16), complex)
    tables[0, [3, 12]] = [2. + .3j, -.7j]
    tables[1, [3, 5]] = [-.4 + .8j, .9 - .2j]
    triples = [tuple(map(np.asarray, products(jnp.asarray(t), jnp.asarray(rows)))) for t in tables]
    psi, hp = (np.column_stack([t[k] for t in triples]) for k in (0, 1))
    q0 = np.sum(abs(psi) ** 2, axis=1)
    qpush = np.sum([t[2] for t in triples], axis=0)
    off_diagonal = hp - np.diag(dense)[:, None] * psi
    lhs = np.sum(abs(off_diagonal) ** 2, axis=1)
    # Cauchy--Schwarz for each state, then sum endpoints. Null swaps may only
    # enlarge RHS. Original duplicate labels make this a conservative bound.
    rhs = .25 * absolute_j.sum() ** 2 * qpush
    assert np.min(rhs - lhs) >= -2e-14
    nodes = q0 == 0
    np.testing.assert_allclose(hp[nodes], off_diagonal[nodes], atol=0)
    assert np.any(lhs[nodes] > .01)
    positive = .5 * (q0 + qpush) > 0
    node_positive = nodes & positive
    assert np.all(lhs[node_positive] / (.5 * qpush[node_positive])
                  <= .5 * absolute_j.sum() ** 2 * (1 + 1e-14))
