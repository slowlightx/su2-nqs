"""Independent physical-space checks for coherent patch-product projection."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.lattices.spatial_symmetry import rectangular_gamma_a1_group
from su2_nqs.lattices.triangular_hnf import hnf_point_group_permutations, hnf_translation_permutations
from su2_nqs.models.patch_product import patch_product_orbit, make_projected_patch_product

jax.config.update('jax_enable_x64', True)


def _bits(n):
    return (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1


def _square_fixture():
    patches = tuple(tuple((2*r+i)*4+2*c+j for i in range(2) for j in range(2))
                    for r in range(2) for c in range(2))
    source = rectangular_gamma_a1_group(2, 2)
    target = rectangular_gamma_a1_group(4, 4)
    return patches, source, target


def test_exact_coherent_square_projection_and_full_gradient():
    patches, source, target = _square_fixture()
    orbit = patch_product_orbit(patches, source, target)
    powers = jnp.asarray([8, 4, 2, 1])

    def raw(p, bits):
        table = p[:16]+1j*p[16:]
        return table[bits @ powers]

    compressed = make_projected_patch_product(raw, orbit)
    # Independent explicit two-stage sum: no orbit canonicalization.
    pids = np.asarray(target)[:, np.asarray(patches)]
    images = jnp.asarray(pids[:, :, np.asarray(source)])

    def naive(p, bits):
        table = p[:16]+1j*p[16:]
        values = table[bits[images] @ powers]
        return jnp.mean(jnp.prod(jnp.mean(values, axis=2), axis=1))

    rng = np.random.default_rng(2718)
    p = jnp.asarray(rng.normal(size=32))
    bits = jnp.asarray(rng.integers(0, 2, size=(24, 16)))
    observed = jax.jit(jax.vmap(compressed, in_axes=(None, 0)))(p, bits)
    expected = jax.vmap(naive, in_axes=(None, 0))(p, bits)
    np.testing.assert_allclose(observed, expected, rtol=2e-12, atol=2e-12)
    loss_c = lambda x: jnp.sum(jnp.abs(jax.vmap(compressed, in_axes=(None, 0))(x, bits[:3]))**2)
    loss_n = lambda x: jnp.sum(jnp.abs(jax.vmap(naive, in_axes=(None, 0))(x, bits[:3]))**2)
    np.testing.assert_allclose(jax.grad(loss_c)(p), jax.grad(loss_n)(p), rtol=2e-11, atol=2e-11)
    assert sum(orbit.counts) == len(target)
    assert len(orbit.representatives) < len(target)
    assert orbit.parent_raw_queries_per_amplitude == len(orbit.representatives)*4*len(source)
    assert float(jnp.max(jnp.abs(jnp.imag(observed)))) > 1e-3
    # Projection acts on complex amplitudes, not probabilities or magnitudes.
    absolute_parent = make_projected_patch_product(lambda x, b: jnp.abs(raw(x, b)), orbit)
    assert not np.allclose(jax.vmap(absolute_parent, in_axes=(None, 0))(p, bits), jnp.abs(observed))


def test_source_projection_is_not_silently_replaced_by_raw_copies():
    patches, source, target = _square_fixture()
    projected = patch_product_orbit(patches, source, target)
    unprojected = patch_product_orbit(patches, (tuple(range(4)),), target)
    raw = lambda p, b: p[b @ jnp.asarray([8, 4, 2, 1])]
    p = jnp.asarray(np.random.default_rng(2).normal(size=16))
    b = jnp.asarray(np.random.default_rng(3).integers(0, 2, (32, 16)))
    strong = jax.vmap(make_projected_patch_product(raw, projected), in_axes=(None, 0))(p, b)
    weak = jax.vmap(make_projected_patch_product(raw, unprojected), in_axes=(None, 0))(p, b)
    assert np.max(np.abs(np.asarray(strong-weak))) > .01


def test_four_dimers_complete_support_singlet_and_translation_invariance():
    patches = ((0, 1), (2, 3), (4, 5), (6, 7))
    source = ((0, 1),)  # A singlet is odd under SWAP, not the source A1 irrep!
    target = tuple(tuple((i+s) % 8 for i in range(8)) for s in range(8))
    orbit = patch_product_orbit(patches, source, target)

    def singlet(p, b):
        return p*(b[0]-b[1])/jnp.sqrt(2.)

    amplitude = make_projected_patch_product(singlet, orbit)
    bits = _bits(8)
    psi = np.asarray(jax.vmap(amplitude, in_axes=(None, 0))(1.+.2j, jnp.asarray(bits)))
    reference = np.zeros(256, complex)
    for g in target:
        for row, b in enumerate(bits):
            transformed = b[np.asarray(g)]
            reference[row] += np.prod([(1.+.2j)*(transformed[i]-transformed[j])/np.sqrt(2.)
                                      for i, j in patches])/len(target)
    np.testing.assert_allclose(psi, reference, atol=2e-14)
    assert np.linalg.norm(psi) > .1
    psi = psi/np.linalg.norm(psi)
    # Total S+ and Sz annihilate the state in the fixed bit convention.
    raised = np.zeros_like(psi)
    for index, b in enumerate(bits):
        for site in np.flatnonzero(b == 0):
            raised[index | (1 << (7-site))] += psi[index]
    np.testing.assert_allclose(raised, 0., atol=2e-14)
    np.testing.assert_allclose((bits.sum(1)-4)*psi, 0., atol=2e-14)
    permutation = bits[:, np.asarray(target[1])] @ (1 << np.arange(7, -1, -1))
    np.testing.assert_allclose(psi[permutation], psi, atol=2e-14)


def test_group_order_and_copy_order_do_not_change_orbits():
    patches, source, target = _square_fixture()
    a = patch_product_orbit(patches, source, target)
    b = patch_product_orbit(tuple(reversed(patches)), tuple(reversed(source)), tuple(reversed(target)))
    assert a.representatives == b.representatives
    assert a.counts == b.counts


def test_actual_triangular_6_to_12_rectangular_patch_orbit_count():
    # Compose independently generated point maps and translations; the helper
    # certifies closure.  No 2**36 physical vector or network allocation.
    def group(length):
        cell = ((length, 0), (0, length))
        return tuple(sorted({tuple(t[p[i]] for i in range(length**2))
                             for _, t in hnf_translation_permutations(cell)
                             for _, _, p in hnf_point_group_permutations(cell)}))

    patches = tuple(tuple((6*r+i)*12+6*c+j for i in range(6) for j in range(6))
                    for r in range(2) for c in range(2))
    orbit = patch_product_orbit(patches, group(6), group(12))
    assert len(orbit.source_images) == 432
    assert orbit.target_group_size == 1728
    assert len(orbit.representatives) == 108
    assert set(orbit.counts) == {16}
    assert orbit.parent_raw_queries_per_amplitude == 186624


@pytest.mark.parametrize('patches,source,target', [
    (((0, 1), (1, 2)), ((0, 1),), (tuple(range(4)),)),
    (((0, 1),), ((0, 1), (0, 1)), ((0, 1),)),
    (((0, 1, 2),), ((0, 1, 2), (1, 2, 0)), ((0, 1, 2),)),
    (((0, 1),), ((1, 0),), ((0, 1),)),
])
def test_invalid_partitions_and_groups_rejected(patches, source, target):
    with pytest.raises(ValueError):
        patch_product_orbit(patches, source, target)


def test_amplitude_input_contract():
    orbit = patch_product_orbit(((0, 1),), ((0, 1),), ((0, 1),))
    f = make_projected_patch_product(lambda p, b: p*(b[0]-b[1]), orbit)
    assert f(1., jnp.asarray([2, 0])) == 0
    with pytest.raises(ValueError):
        f(1., jnp.asarray([0., 1.]))
    bad = make_projected_patch_product(lambda p, b: jnp.ones(2), orbit)
    with pytest.raises(ValueError, match='scalar'):
        bad(1., jnp.asarray([0, 1]))
