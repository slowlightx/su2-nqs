"""Exact physical support oracles for conditioned patch proposals."""
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.su2.fusion_tree import FusionNode, FusionTree, Leaf
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.reduced_tree_environment import ReducedSingletTree
from su2_nqs.vmc.conditioned_patch_proposal import (
    make_conditioned_patch_proposal, physical_leaf_diagonals,
)
from su2_nqs.vmc.exchange_bridge_importance import make_exchange_bridge

jax.config.update("jax_enable_x64", True)


def _basis(n):
    return np.asarray(list(product((0, 1), repeat=n)), dtype=np.int32)


def _oracle(patches, probabilities, images, n_up, floor):
    n = np.size(patches)
    bits = _basis(n)
    powers = 1 << np.arange(len(patches[0])-1, -1, -1)
    codes = bits[:, patches] @ powers
    weights = np.prod(probabilities[np.arange(len(patches))[None, :], codes], axis=1)
    valid = bits.sum(axis=1) == n_up
    normalization = weights[valid].sum()
    weights = np.where(valid, weights/normalization, 0.)
    pushed = np.zeros_like(weights)
    global_powers = 1 << np.arange(n-1, -1, -1)
    for image in images:
        destination = np.zeros_like(bits)
        destination[:, image] = bits
        pushed[destination @ global_powers] += weights/len(images)
    return bits, (1-floor)*pushed+floor*valid/valid.sum(), normalization


@pytest.mark.parametrize("floor", [0., .07, 1.])
def test_exact_normalization_and_active_permutation_push(floor):
    patches = np.asarray([[0, 3, 4], [1, 2, 5]])
    p = np.asarray([[0, .05, .10, .25, .20, .15, .15, .10],
                    [.15, .10, .05, .10, .25, .20, .15, 0]])
    # Not a group: the explicit-list mixture must still be exact.
    images = np.asarray([np.arange(6), [2, 5, 1, 3, 0, 4], [4, 2, 0, 1, 5, 3]])
    log_q, _, metadata = make_conditioned_patch_proposal(
        patches, p, images, n_up=3, uniform_fraction=floor, image_batch_size=2)
    bits, expected, normalization = _oracle(patches, p, images, 3, floor)
    actual = np.exp(jax.jit(log_q)(jnp.asarray(bits)))
    np.testing.assert_allclose(actual, expected, atol=3e-16, rtol=3e-14)
    assert abs(actual.sum()-1) < 4e-15
    assert abs(np.exp(metadata["log_product_condition_probability"])-normalization) < 2e-16
    assert metadata["full_fixed_Sz_support"] == (floor > 0)
    assert not metadata["coherent_projection_density"]
    invalid = jnp.asarray([[.5, .5, 0, 1, 0, 1], [2, 0, 0, 1, 0, 0]])
    assert np.all(np.isneginf(log_q(invalid)))


def test_draw_frequency_exact_density_and_reproducibility():
    patches = np.asarray([[0, 2], [3, 1]])
    p = np.asarray([[.12, .38, .22, .28], [.05, .2, .5, .25]])
    images = np.asarray([np.arange(4), [2, 0, 3, 1]])
    log_q, draw, _ = make_conditioned_patch_proposal(patches, p, images, n_up=2, uniform_fraction=.17)
    key = jax.random.PRNGKey(20261023)
    samples = np.asarray(jax.jit(lambda k: draw(k, 40000))(key))
    assert np.all(samples.sum(axis=1) == 2)
    bits = _basis(4)
    expected = np.exp(log_q(bits))
    actual = np.bincount(samples @ np.asarray([8, 4, 2, 1]), minlength=16)/len(samples)
    np.testing.assert_allclose(actual, expected, atol=.005, rtol=0)
    np.testing.assert_array_equal(draw(key, 20), draw(key, 20))


def test_reachability_is_enforced_before_sampling_not_repaired():
    # Every patch is all down or all up. The middle patch is forced up.
    p = np.asarray([[.8, 0, 0, .2], [0, 0, 0, 1.], [.3, 0, 0, .7]])
    log_q, draw, _ = make_conditioned_patch_proposal(
        [[0, 1], [2, 3], [4, 5]], p, n_up=4, uniform_fraction=0)
    samples = np.asarray(draw(jax.random.PRNGKey(17), 5000))
    assert np.all(samples[:, 2:4] == 1)
    assert np.all(samples[:, ::2] == samples[:, 1::2])
    assert np.all(samples.sum(axis=1) == 4)
    assert np.isfinite(log_q(samples)).all()
    expected_first_up = (.2*.3)/(.2*.3+.8*.7)
    assert abs(samples[:, 0].mean()-expected_first_up) < .02


def test_positive_floor_covers_base_nodes_and_exchange_bridge_is_normalized():
    p = np.asarray([[0, .5, .5, 0], [0, .5, .5, 0]])
    log_q, draw, _ = make_conditioned_patch_proposal([[0, 1], [2, 3]], p, n_up=2, uniform_fraction=.03)
    bits = _basis(4)
    fixed = bits.sum(axis=1) == 2
    probability = np.exp(log_q(bits))
    np.testing.assert_allclose(probability[[3, 12]], .03/6, atol=1e-16)
    bridge, bridge_draw = make_exchange_bridge(
        log_q, draw, [(0, 2), (1, 2)], [.4, .6], n_sites=4, bridge_fraction=.4)
    mixed = np.exp(bridge(bits)[0])
    assert abs(mixed.sum()-1) < 1e-14
    assert np.all(mixed[fixed] > 0)
    sample, _ = bridge_draw(jax.random.PRNGKey(7), 100)
    assert np.all(np.asarray(sample).sum(axis=1) == 2)


def test_invalid_inputs_and_reproducible_provenance():
    p = np.full((2, 4), .25)
    kwargs = dict(n_up=2)
    _, _, first = make_conditioned_patch_proposal([[0, 1], [2, 3]], p, **kwargs)
    _, _, second = make_conditioned_patch_proposal([[0, 1], [2, 3]], p.copy(), **kwargs)
    assert first["proposal_sha256"] == second["proposal_sha256"]
    _, _, changed = make_conditioned_patch_proposal([[0, 1], [2, 3]], p, uniform_fraction=.1, **kwargs)
    assert changed["proposal_sha256"] != first["proposal_sha256"]
    for patches, probs, options in (
        ([[0, 0], [2, 3]], p, kwargs),
        ([[0, 1], [2, 3]], p*.9, kwargs),
        ([[0, 1], [2, 3]], p+1e-5j, kwargs),
        ([[0, 1], [2, 3]], p, dict(n_up=2.)),
        ([[0, 1], [2, 3]], p, dict(n_up=2, images=[[0, 0, 1, 2]])),
        ([[0, 1], [2, 3]], p, dict(n_up=2, image_batch_size=0)),
        ([[0, 1], [2, 3]], [[1, 0, 0, 0], [1, 0, 0, 0]], kwargs),
    ):
        with pytest.raises(ValueError):
            make_conditioned_patch_proposal(patches, probs, **options)


def test_raw_physical_diagonal_preserves_complex_coherence_and_magnetic_factors():
    # Two complete spin-pair blocks. Compare with an explicit four-spin ket.
    tree = FusionTree(FusionNode(Leaf(0), Leaf(1)))
    singlet = np.asarray([0, -1, 1, 0])/np.sqrt(2)
    triplet = np.asarray([1, 1/np.sqrt(2), 1/np.sqrt(2), 1])
    scale_left = {0: .7+.4j, 2: 1.3-.1j}
    scale_right = {0: 1.1-.2j, 2: .9+.5j}
    coeff = {0: .8+.6j, 2: -.3+.7j}
    features = tuple({0: (singlet*scales[0])[:, None], 2: (triplet*scales[2])[:, None]}
                     for scales in (scale_left, scale_right))
    state = ReducedSingletTree(tree,
        tuple({q: np.asarray([[abs(scales[q])**2]]) for q in (0, 2)} for scales in (scale_left, scale_right)),
        ({(q, q, 0): np.asarray([[[coeff[q]]]]) for q in (0, 2)},), np.ones(1))
    local, diagnostic = physical_leaf_diagonals(state, features)
    down = np.asarray([1, 0, 0, 0])
    zero = np.asarray([0, 1, 1, 0])/np.sqrt(2)
    up = np.asarray([0, 0, 0, 1])
    triplet_pair = (np.kron(down, up)-np.kron(zero, zero)+np.kron(up, down))/np.sqrt(3)
    psi = (coeff[0]*scale_left[0]*scale_right[0]*np.kron(singlet, singlet)
           + coeff[2]*scale_left[2]*scale_right[2]*triplet_pair)
    tensor = psi.reshape(4, 4)
    norm = np.vdot(psi, psi).real
    np.testing.assert_allclose(local[0], np.sum(abs(tensor)**2, axis=1)/norm, atol=3e-16, rtol=1e-14)
    np.testing.assert_allclose(local[1], np.sum(abs(tensor)**2, axis=0)/norm, atol=3e-16, rtol=1e-14)
    assert abs(diagnostic["raw_tree_norm_squared"]-norm) < 1e-14
    assert not diagnostic["coherent_A1_diagonal"]
    bad = ({q: w.copy() for q, w in features[0].items()}, features[1])
    bad[0][0][0] = .1
    with pytest.raises(ValueError, match="magnetic"):
        physical_leaf_diagonals(state, bad)


def test_complex_nondiagonal_multiplicity_environment_matches_six_spin_partial_trace():
    bits = _basis(3)
    magnetic = 2*bits.sum(axis=1)-3
    schur = np.zeros((8, 2))
    for x, row in enumerate(bits):
        m0, m1, m2 = 2*row-1
        if abs(magnetic[x]) != 1:
            continue
        for a, q12 in enumerate((0, 2)):
            if abs(m0+m1) <= q12:
                schur[x, a] = (clebsch_gordan(1, int(m0), 1, int(m1), q12, int(m0+m1))
                    * clebsch_gordan(q12, int(m0+m1), 1, int(m2), 1, int(magnetic[x])))
    left = np.asarray([[1+.2j, .4-.7j], [.3+.1j, .8+.2j]])
    right = np.asarray([[.7+.5j, -.4+.1j], [.2-.6j, 1.2-.2j]])
    core = np.asarray([[.5+.2j, .3-.4j], [-.7+.6j, .2+.8j]])
    tree = FusionTree(FusionNode(Leaf(0), Leaf(1)))
    state = ReducedSingletTree(tree, ({1: left.conj().T@left}, {1: right.conj().T@right}),
                               ({(1, 1, 0): core[None]},), np.ones(1))
    fl, fr = schur@left, schur@right
    local, _ = physical_leaf_diagonals(state, ({1: fl}, {1: fr}))
    tensor = np.zeros((8, 8), dtype=complex)
    for x in range(8):
        for y in range(8):
            if abs(magnetic[x]) == 1 and magnetic[x]+magnetic[y] == 0:
                tensor[x, y] = (fl[x]@core@fr[y]
                    * clebsch_gordan(1, int(magnetic[x]), 1, int(magnetic[y]), 0, 0))
    norm = np.sum(abs(tensor)**2)
    np.testing.assert_allclose(local[0], np.sum(abs(tensor)**2, axis=1)/norm, atol=5e-16, rtol=1e-14)
    np.testing.assert_allclose(local[1], np.sum(abs(tensor)**2, axis=0)/norm, atol=5e-16, rtol=1e-14)
