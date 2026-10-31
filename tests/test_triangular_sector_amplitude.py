"""Independent charge, phase, support and derivative checks for opt-in CT sectors."""
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.lattices.triangular_hnf import hnf_site_permutation, triangular_hnf_j1j2_bonds
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank, init_arity_pair
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, init_mixed_clebsch_family
from su2_nqs.models.triangular_sector_amplitude import (
    make_prepared_sector_amplitude, make_prepared_target_root_amplitude,
    make_sector_prepare, pack_sector_parameters, triangular_sector_projector,
    make_prepared_target_root_exchange_queries, make_prepared_sector_exchange_queries,
)
from su2_nqs.models.triangular_target_root import (
    make_target_root_amplitude, singlet_root_coefficients, target_root_layout,
)
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def example():
    config = MixedClebschConfig(6, 9, 3)
    joint = init_mixed_clebsch_family(jax.random.PRNGKey(11201), config)["ternary_free"]
    rng = np.random.default_rng(11202)
    roots = tuple(jnp.asarray(rng.normal(size=(dl, dr)))
                  for _, _, dl, dr in target_root_layout(config, 2))
    supports = triangular_schur_hierarchy(6).blocks
    while True:
        bits = rng.permutation(np.r_[np.ones(18, np.int32), np.zeros(18, np.int32)])
        if all(abs(np.sum(2*bits[list(block.sites)]-1)) <= 3 for block in supports):
            break
    return config, joint, roots, bits


@pytest.mark.parametrize("arity", ["ternary_joint", "binary_truncated"])
def test_prepared_target_matches_legacy_and_preserves_all_native_derivatives(example, arity):
    config, joint, roots, bits = example
    native = (joint if arity == "ternary_joint" else
              init_arity_pair(joint, config, 2)[0]["binary_truncated"])
    params = pack_sector_parameters(native, roots)
    assert len(params["backbone"]["top"]) == len(native["top"]) - 1
    assert "six" in params["backbone"] if arity == "binary_truncated" else "six" not in params["backbone"]
    prepare = make_sector_prepare(config, arity=arity, binary_rank=2)
    prepared_raw = make_prepared_target_root_amplitude(config, 2, 0)
    legacy_raw = make_target_root_amplitude(config, 2, 0)

    def original(p):
        lower = dict(p["backbone"])
        lower["top"] = lower["top"] + (native["top"][-1],)
        return legacy_raw(arity_clebsch_bank(lower, config, arity, 2), p["root"], bits)

    candidate = lambda p: prepared_raw(prepare(p), bits)
    direction = jax.tree.map(lambda x: jnp.sin(jnp.arange(x.size).reshape(x.shape)+.3), params)
    actual = jax.jit(lambda p, d: jax.jvp(candidate, (p,), (d,)))(params, direction)
    reference = jax.jit(lambda p, d: jax.jvp(original, (p,), (d,)))(params, direction)
    np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=1e-16)
    grad = jax.jit(jax.grad(candidate))(params)
    for name in params["backbone"]:
        norm = sum(float(jnp.vdot(x, x).real) for x in jax.tree.leaves(grad["backbone"][name]))
        assert norm > 0., name
    assert sum(float(jnp.vdot(x, x).real) for x in grad["root"]) > 0.


def test_singlet_replay_and_triplet_physical_spin_squared(example):
    config, joint, roots, bits = example
    for q, root in ((0, singlet_root_coefficients(joint, config)), (2, roots)):
        prepare = make_sector_prepare(config)
        params = pack_sector_parameters(joint, root)
        raw = jax.jit(jax.vmap(make_prepared_target_root_amplitude(config, q, 0), in_axes=(None, 0)))
        # Direct physical S²=M²+N/2+sum_(unlike unordered pairs) swap at M=0.
        rows = [bits]
        for up in np.flatnonzero(bits):
            for down in np.flatnonzero(1-bits):
                row = bits.copy()
                row[up], row[down] = row[down], row[up]
                rows.append(row)
        values = np.asarray(raw(prepare(params), np.asarray(rows)))
        if q == 0:
            legacy = make_target_root_amplitude(config, 0, 0)(joint, root, bits)
            np.testing.assert_allclose(values[0], legacy, rtol=2e-12, atol=1e-16)
        residual = (18-q*(q+2)/4)*values[0] + values[1:].sum()
        assert np.linalg.norm(values) > 0.
        assert abs(residual)/np.linalg.norm(values) < 3e-12
        wrong_m = bits.copy()
        wrong_m[np.flatnonzero(wrong_m)[0]] = 0
        assert float(raw(prepare(params), wrong_m[None, :])[0]) == 0.


@pytest.mark.parametrize("momentum", [((1, 3), (1, 3)), ((-1, 6), (-1, 6))])
def test_projected_amplitude_exact_active_translation_phase_and_h_graph(example, momentum):
    config, joint, roots, bits = example
    projector = triangular_sector_projector(config, momentum)
    prepare = make_sector_prepare(config)
    prepared = prepare(pack_sector_parameters(joint, roots))
    # chunk5 forces the exact zero-weight padded-tail branch.
    projected = jax.jit(make_prepared_sector_amplitude(config, 2, 0, momentum, chunk=5))
    a = complex(projected(prepared, bits))
    assert abs(a) > 1e-16
    for translation in ((1, 0), (0, 1)):
        image = hnf_site_permutation(((6, 0), (0, 6)), translation=translation)
        transformed = complex(projected(prepared, bits[np.asarray(image)]))
        phase = np.exp(-2j*np.pi*sum(Fraction(*k)*t for k, t in zip(momentum, translation)))
        np.testing.assert_allclose(transformed, phase*a, rtol=1e-10, atol=2e-16)
    # The full J1-J2 operator is independently invariant under every image;
    # spatial projection hence commutes with H, not just with a selected bond.
    bonds = triangular_hnf_j1j2_bonds(((6, 0), (0, 6)), j1=1., j2=.125)
    original = {tuple(sorted((i, j))): coupling for i, j, coupling in bonds}
    for image in projector.site_permutations:
        changed = {tuple(sorted((image[i], image[j]))): coupling for i, j, coupling in bonds}
        assert changed == original


def test_exact_fraction_api_and_invalid_quantum_numbers(example):
    config = example[0]
    assert triangular_sector_projector(config, (Fraction(1, 3), Fraction(1, 3))).group_order == 216
    for momentum in ((.333333333, .333333333), ((1, 4), (0, 1)), (True, 0), ((1, 0), 0)):
        with pytest.raises((ValueError, TypeError)):
            triangular_sector_projector(config, momentum)
    for q, m in ((-2, 0), (1, 1), (2, 1), (38, 0), (True, 0)):
        with pytest.raises((ValueError, TypeError)):
            make_prepared_target_root_amplitude(config, q, m)
    with pytest.raises(ValueError):
        make_prepared_sector_amplitude(config, 2, 0, ((1, 3), (1, 3)), chunk=0)
    prepare = make_sector_prepare(config)
    params = pack_sector_parameters(example[1], example[2])
    projected = make_prepared_sector_amplitude(config, 2, 0, ((1, 3), (1, 3)))
    with pytest.raises(ValueError):
        projected(prepare(params), jnp.zeros((35,), dtype=jnp.int32))


def test_target_root_cached_exchanges_match_independent_full_recontraction_and_jvp(example):
    config,joint,roots,bits = example
    pairs = ((0,6),(0,21),(0,3),(1,2),(0,21))
    prepare = make_sector_prepare(config)
    params = pack_sector_parameters(joint,roots)
    cached = make_prepared_target_root_exchange_queries(config,2,0,pairs)
    raw = make_prepared_target_root_amplitude(config,2,0)
    rows = [bits]
    for a,b in pairs:
        changed = bits.copy()
        changed[a],changed[b] = changed[b],changed[a]
        rows.append(changed)
    def fast(p):
        value,swaps = cached(prepare(p),bits)
        return jnp.r_[value,swaps]
    slow = lambda p:jax.vmap(raw,in_axes=(None,0))(prepare(p),jnp.asarray(rows))
    direction = jax.tree.map(lambda x:jnp.cos(jnp.arange(x.size).reshape(x.shape)+.7),params)
    actual = jax.jit(lambda p,d:jax.jvp(fast,(p,),(d,)))(params,direction)
    expected = jax.jit(lambda p,d:jax.jvp(slow,(p,),(d,)))(params,direction)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=1e-16)
    np.testing.assert_array_equal(np.asarray(actual)[...,2],np.asarray(actual)[...,5])
    with pytest.raises(ValueError):
        make_prepared_target_root_exchange_queries(config,2,0,((0,0),))


def test_complex_projected_cached_exchanges_track_original_not_image_pair_indices(example):
    config,joint,roots,bits = example
    momentum = ((-1,6),(-1,6))
    # Incomplete orbit and duplicate reversed pair exercise inverse-image
    # lookup; treating an image-frame bond as the original one fails this test.
    pairs = ((0,1),(1,0))
    prepared = make_sector_prepare(config)(pack_sector_parameters(joint,roots))
    projected = jax.jit(make_prepared_sector_amplitude(config,2,0,momentum,chunk=5))
    cached = jax.jit(make_prepared_sector_exchange_queries(config,2,0,momentum,pairs,chunk=5))
    value,swaps = cached(prepared,bits)
    changed = bits.copy()
    changed[0],changed[1] = changed[1],changed[0]
    np.testing.assert_allclose(value,projected(prepared,bits),rtol=2e-11,atol=2e-16)
    np.testing.assert_allclose(swaps,np.repeat(projected(prepared,changed),2),rtol=2e-11,atol=2e-16)
    # Independent physical-H action directly re-evaluates the full projected
    # amplitude, not the ancestor cache. Repeated bonds must add, not replace.
    bonds = ((0,1,1.),(1,0,.125))
    physical = make_amplitude_heisenberg_action(projected,bonds,n_sites=36)
    reference_psi,reference_hpsi = physical(prepared,jnp.asarray(bits)[None,:])
    different = bits[0] != bits[1]
    cached_hpsi = (1.+.125)*(jnp.where(different,-.25,.25)*value +
                           jnp.where(different,.5*swaps[0],0.))
    np.testing.assert_allclose(value,reference_psi[0],rtol=2e-11,atol=2e-16)
    np.testing.assert_allclose(cached_hpsi,reference_hpsi[0],rtol=2e-11,atol=2e-16)
