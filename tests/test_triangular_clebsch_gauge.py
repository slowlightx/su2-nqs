"""Physical checks of offline canonical coordinates for the production CT bank."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.models.triangular_clebsch_gauge import (
    triangular_bank_reduced_tree, triangular_edge_spectrum, gauge_triangular_edge,
    canonicalize_triangular_bank, compress_triangular_edge, orthogonal_live_expansion, triangular_raw_overlap,
)
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, init_mixed_clebsch_family, make_mixed_clebsch_amplitude,
    make_mixed_projected_amplitude, mixed_clebsch_layout,
)
from su2_nqs.su2.reduced_tree_environment import reduced_tree_environments
from su2_nqs.su2.triangle_blocks import triangle_block_tables

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def example():
    config = MixedClebschConfig(multiplicity_cap=9, maximum_q=3)
    bank = init_mixed_clebsch_family(jax.random.PRNGKey(2026102101), config)["ternary_free"]
    rng = np.random.default_rng(2026102102)
    rows = np.asarray([rng.permutation([0]*18+[1]*18) for _ in range(5)], np.int32)
    evaluate = jax.jit(jax.vmap(make_mixed_clebsch_amplitude(config), in_axes=(None, 0)))
    return config, bank, rows, evaluate


def test_nine_site_leaf_metric_uses_fixed_m_not_multiplet_degeneracy(example):
    config, bank, _, _ = example
    state = triangular_bank_reduced_tree(bank, config)
    bits = (np.arange(512)[:, None] >> np.arange(8, -1, -1)) & 1
    mags = np.sum(2*bits-1, axis=1)
    for q, weight in zip(config.local_q, bank["local"][0], strict=True):
        physical = triangle_block_tables()["nine"][q][mags == 1] @ np.asarray(weight)
        np.testing.assert_allclose(physical.conj().T @ physical, state.leaf_grams[0][q], atol=2e-12)


def test_qr_root_center_preserves_raw_and_complete_A1_amplitudes_and_singlet(example):
    config, bank, rows, evaluate = example
    canonical, report = canonicalize_triangular_bank(bank, config)
    env = reduced_tree_environments(triangular_bank_reduced_tree(canonical, config))
    root = triangular_bank_reduced_tree(canonical, config).tree.root
    for node, sectors in env.upward.items():
        if node != root:
            for gram in sectors.values():
                np.testing.assert_allclose(gram, np.eye(len(gram)), atol=3e-12, rtol=0)
    assert report["maximum_isometry_defect"] < 3e-12
    np.testing.assert_allclose(evaluate(canonical, rows), evaluate(bank, rows), atol=3e-17, rtol=3e-11)
    group = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    projected = jax.jit(make_mixed_projected_amplitude(config, group, chunk=4))
    np.testing.assert_allclose(projected(canonical, rows[0]), projected(bank, rows[0]), atol=3e-17, rtol=3e-11)
    # Literal S+ Ward sum from every allowed spin lowering, not label inspection.
    bra = np.random.default_rng(71).permutation([0]*17+[1]*19)
    columns = np.repeat(bra[None, :], 19, axis=0)
    columns[np.arange(19), np.flatnonzero(bra)] = 0
    values = np.asarray(evaluate(canonical, columns))
    assert abs(values.sum()) / np.linalg.norm(values) < 3e-11


def test_physical_environment_compression_is_gauge_invariant_and_matches_raw_norm(example):
    config, bank, _, _ = example
    before = triangular_edge_spectrum(bank, config, 0)
    gauges = {q: np.diag(np.geomspace(.08, 8., len(block.multiplicity_weights)))
              for q, block in before.blocks.items()}
    gauged = gauge_triangular_edge(bank, config, 0, gauges)
    after = triangular_edge_spectrum(gauged, config, 0)
    for q in before.blocks:
        np.testing.assert_allclose(after.blocks[q].multiplicity_weights,
                                   before.blocks[q].multiplicity_weights, atol=2e-11, rtol=0)
    # The singular values of the coefficient matrix are NOT gauge invariant.
    a = np.linalg.svd(bank["local"][0][0], compute_uv=False)
    b = np.linalg.svd(gauged["local"][0][0], compute_uv=False)
    assert np.max(abs(a/a.sum()-b/b.sum())) > .01
    keep = {q: max(1, len(block.multiplicity_weights)//2) for q, block in before.blocks.items()}
    reduced, report = compress_triangular_edge(gauged, config, 0, keep)
    oldnorm = reduced_tree_environments(triangular_bank_reduced_tree(gauged, config)).norm_squared
    newnorm = reduced_tree_environments(triangular_bank_reduced_tree(reduced, config)).norm_squared
    np.testing.assert_allclose(newnorm/oldnorm, report["raw_squared_fidelity"], atol=3e-11, rtol=0)
    cross = triangular_raw_overlap(gauged, reduced, config)
    np.testing.assert_allclose(abs(cross)**2/(oldnorm*newnorm), report["raw_squared_fidelity"], atol=3e-11, rtol=0)
    assert 0 < report["raw_discarded_weight"] < 1


def test_orthogonal_expansion_preserves_parent_and_has_live_new_downstream_tangent(example):
    config, bank, rows, evaluate = example
    target = MixedClebschConfig(multiplicity_cap=12, maximum_q=3)
    expanded, report = orthogonal_live_expansion(bank, config, target, jax.random.PRNGKey(101))
    wide = jax.jit(jax.vmap(make_mixed_clebsch_amplitude(target), in_axes=(None, 0)))
    np.testing.assert_allclose(wide(expanded, rows), evaluate(bank, rows), atol=3e-17, rtol=3e-11)
    assert report["local_complement_orthogonality_error"] < 3e-12
    direction = jax.tree.map(jnp.zeros_like, expanded)
    top = list(direction["top"])
    groups = list(top[0])
    old_layout = mixed_clebsch_layout(config)[0]
    old_widths = {key: group["shape"][1:] for group in old_layout["groups"] for key in group["keys"]}
    for i, group in enumerate(mixed_clebsch_layout(target)[0]["groups"]):
        value = groups[i]
        for k, key in enumerate(group["keys"]):
            if key[0] == 1 and value.shape[2] > old_widths[key][1]:
                value = value.at[k, 0, old_widths[key][1], 0].set(1.)
        groups[i] = value
    top[0] = tuple(groups)
    direction["top"] = tuple(top)
    tangent = jax.jit(lambda p, d: jax.jvp(lambda z: wide(z, rows), (p,), (d,))[1])(expanded, direction)
    assert np.linalg.norm(tangent) > 1e-12
    epsilon = 1e-4
    plus = jax.tree.map(lambda p, d: p+epsilon*d, expanded, direction)
    minus = jax.tree.map(lambda p, d: p-epsilon*d, expanded, direction)
    np.testing.assert_allclose(tangent, (wide(plus, rows)-wide(minus, rows))/(2*epsilon), atol=3e-14, rtol=2e-6)


def test_invalid_edge_gauge_and_multiplicity_are_rejected(example):
    config, bank, _, _ = example
    with pytest.raises(ValueError):
        triangular_edge_spectrum(bank, config, True)
    with pytest.raises(ValueError):
        gauge_triangular_edge(bank, config, 0, {1: np.eye(9)})
    with pytest.raises(ValueError):
        compress_triangular_edge(bank, config, 0, {1: -1, 3: 1})


def test_tiny_gram_eigenvalue_with_finite_schmidt_weight_fails_closed(example):
    """Inverse parent scaling must not turn a finite physical weight into zero."""
    config, bank, _, _ = example
    canonical, _ = canonicalize_triangular_bank(bank, config)
    original = triangular_edge_spectrum(canonical, config, 0)
    gauges = {q: np.eye(len(block.multiplicity_weights)) for q, block in original.blocks.items()}
    gauges[1][0, 0] = 1e-7
    scaled = gauge_triangular_edge(canonical, config, 0, gauges)
    # Independent physical double-layer overlap certifies that the state is
    # unchanged despite G eigenvalue~1e-14 and inverse environment scale.
    norm = triangular_raw_overlap(canonical, canonical, config).real
    np.testing.assert_allclose(triangular_raw_overlap(scaled, scaled, config), norm, rtol=2e-13)
    with pytest.raises(ArithmeticError, match="lost normalized physical weight"):
        triangular_edge_spectrum(scaled, config, 0)
    with pytest.raises(ArithmeticError, match="lost normalized physical weight"):
        compress_triangular_edge(scaled, config, 0, {q: 2 for q in original.blocks})
    repaired, _ = canonicalize_triangular_bank(scaled, config)
    recovered = triangular_edge_spectrum(repaired, config, 0)
    for q in original.blocks:
        np.testing.assert_allclose(recovered.blocks[q].multiplicity_weights,
                                   original.blocks[q].multiplicity_weights, atol=3e-13)
