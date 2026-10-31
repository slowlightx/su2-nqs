"""Physical-oracle tests: internal q coherence is NOT sampled as a hidden label."""
import numpy as np
import pytest

from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.reduced_tree_environment import ReducedSingletTree
from su2_nqs.vmc.raw_tree_born import RawTreeBornSampler
from su2_nqs.su2.quantum_numbers import coupling_range


def _four_spin_sampler():
    singlet = np.array([0., -1., 1., 0.])/np.sqrt(2)
    triplet = np.array([[1., 0., 0.], [0., 1/np.sqrt(2), 0.],
                        [0., 1/np.sqrt(2), 0.], [0., 0., 1.]])
    iso = {0: singlet[:, None, None], 2: triplet[:, None, :]}
    coefficients = np.array([np.cos(.61), np.exp(.73j)*np.sin(.61)])
    kernels = {(0, 0, 0): coefficients[:1].reshape(1, 1, 1),
               (2, 2, 0): coefficients[1:].reshape(1, 1, 1)}
    state = ReducedSingletTree(FusionTree.balanced(2), ({0: np.eye(1), 2: np.eye(1)},)*2,
                              (kernels,), np.ones(1))
    sampler = RawTreeBornSampler(state, (iso, iso), ((0, 1), (2, 3)))
    physical = (coefficients[0]*np.kron(singlet, singlet)
        +coefficients[1]*(np.kron(triplet[:, 0], triplet[:, 2])
            -np.kron(triplet[:, 1], triplet[:, 1])+np.kron(triplet[:, 2], triplet[:, 0]))/np.sqrt(3))
    return sampler, physical


def test_exact_physical_draws_retain_complex_cross_q_interference_and_replay_probability():
    sampler, physical = _four_spin_sampler()
    np.testing.assert_allclose(np.vdot(physical, physical), 1., atol=8e-15)
    rng = np.random.default_rng(26102742)
    counts = np.zeros(16, int)
    for _ in range(4096):
        draw = sampler.draw(rng)
        index = int(draw.bits@(1 << np.arange(3, -1, -1)))
        counts[index] += 1
        np.testing.assert_allclose(draw.amplitude, physical[index], atol=8e-15)
        np.testing.assert_allclose(draw.log_probability, np.log(abs(physical[index])**2), atol=2e-13)
    probabilities = abs(physical)**2
    np.testing.assert_array_equal(counts[probabilities == 0], 0)
    # Fixed seed and a conservative binomial envelope, not exact frequency equality.
    assert np.all(abs(counts/4096-probabilities) <= 6*np.sqrt(probabilities*(1-probabilities)/4096)+1/4096)


def test_sampler_rejects_noncanonical_input_and_explicit_memory_overflow():
    sampler, _ = _four_spin_sampler()
    state = sampler.state
    bad = ReducedSingletTree(state.tree, state.leaf_grams,
        ({key: 2*value for key, value in state.kernels[0].items()},), state.root_vector)
    singlet = np.array([0., -1., 1., 0.])/np.sqrt(2)
    triplet = np.array([[1., 0., 0.], [0., 1/np.sqrt(2), 0.],
                        [0., 1/np.sqrt(2), 0.], [0., 0., 1.]])
    iso = {0: singlet[:, None, None], 2: triplet[:, None, :]}
    with pytest.raises(ValueError, match='isometric upper'):
        RawTreeBornSampler(bad, (iso, iso), ((0, 1), (2, 3)))
    with pytest.raises(MemoryError):
        RawTreeBornSampler(state, (iso, iso), ((0, 1), (2, 3)), memory_limit_bytes=1)


def test_deep_entangled_N8_sampler_matches_independent_physical_CG_expansion():
    from test_triangular_torus_hierarchy import _physical_merge
    singlet = np.array([0., -1., 1., 0.])/np.sqrt(2)
    triplet = np.array([[1., 0., 0.], [0., 1/np.sqrt(2), 0.],
                        [0., 1/np.sqrt(2), 0.], [0., 0., 1.]])
    iso = {0: singlet[:, None, None], 2: triplet[:, None, :]}
    widths = {0: 2, 2: 3, 4: 1}; lower = {}
    for q, d in widths.items():
        pairs = [(ql, qr) for ql in (0, 2) for qr in (0, 2) if q in coupling_range(ql, qr)]
        for index, (ql, qr) in enumerate(pairs):
            lower[ql, qr, q] = np.eye(d)[:, index, None, None]
    rng = np.random.default_rng(26102743)
    coefficients = rng.normal(size=14)+1j*rng.normal(size=14)
    coefficients /= np.linalg.norm(coefficients)
    root = {}; offset = 0
    for q, d in widths.items():
        root[q, q, 0] = coefficients[offset:offset+d*d].reshape(1, d, d); offset += d*d
    state = ReducedSingletTree(FusionTree.balanced(4), ({0: np.eye(1), 2: np.eye(1)},)*4,
                              (lower, lower, root), np.ones(1))
    sampler = RawTreeBornSampler(state, (iso,)*4, ((0, 1), (2, 3), (4, 5), (6, 7)))
    four = _physical_merge(iso, iso, {q: np.eye(d) for q, d in widths.items()})
    physical = _physical_merge(four, four, {0: coefficients[:, None]})[0][:, 0, 0]
    np.testing.assert_allclose(np.vdot(physical, physical), 1., atol=8e-15)
    for _ in range(512):
        draw = sampler.draw(rng)
        index = int(draw.bits@(1 << np.arange(7, -1, -1)))
        np.testing.assert_allclose(draw.amplitude, physical[index], atol=8e-15)
        np.testing.assert_allclose(draw.log_probability, np.log(abs(physical[index])**2), atol=2e-12)
