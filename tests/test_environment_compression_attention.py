"""Independent physical checks for a static reduced-subspace controller."""
import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.models.environment_compression_attention import (
    EnvironmentCompressionConfig, init_environment_compression,
    predict_environment_block, reduced_rank_one_projector, reduced_frame_projector,
    environment_memory_weights, predict_environment_memory,
)
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.su2.local_schur import cell_schur_tensor

jax.config.update("jax_enable_x64", True)


def test_arbitrary_learned_compression_is_idempotent_and_preserves_singlet():
    """All magnetic copies are retained, with the same reduced projector."""
    block = np.asarray(reduced_rank_one_projector(jnp.asarray([.1, .8, .3])))
    table = cell_schur_tensor("triangular_triangle")
    physical = sum(table[:, 1, m+3, :2] @ block @ table[:, 1, m+3, :2].T
                   for m in (-1, 1))
    physical += sum(np.outer(table[:, 3, m+3, 0], table[:, 3, m+3, 0])
                    for m in (-3, -1, 1, 3))
    np.testing.assert_allclose(physical @ physical, physical, atol=2e-14)
    basis = coupled_basis_matrix(enumerate_paths(6, 0), 0, 0)
    acted = np.kron(physical, np.eye(8)) @ basis
    np.testing.assert_allclose(basis @ (basis.conj().T @ acted), acted, atol=2e-14)
    # Keeping a reduced vector for only one m would fail this invariant.


def test_equal_count_predictors_and_finite_trainable_gradients():
    tokens = jnp.asarray(np.random.default_rng(12).normal(size=(15, 16)))
    pool = jnp.asarray(np.eye(3).repeat(5, axis=1)/5)
    mean = jnp.asarray([.4, .4, 0.])
    counts = []
    for mixing in ("pointwise", "attention", "hierarchical_attention"):
        config = EnvironmentCompressionConfig(16, mixing=mixing)
        params = init_environment_compression(jax.random.PRNGKey(13), config, mean)
        counts.append(sum(x.size for x in jax.tree.leaves(params)))
        np.testing.assert_array_equal(predict_environment_block(params, tokens, config, pool), mean)
        params["readout"] = jnp.ones((8, 3))*.01
        gradient = jax.grad(lambda p: jnp.sum(predict_environment_block(p, tokens, config, pool)**2))(params)
        assert all(np.isfinite(v).all() for v in jax.tree.leaves(gradient))
        assert np.linalg.norm(gradient["embedding"]) > 0
    assert len(set(counts)) == 1


def test_multiplicity_spectrum_alone_does_not_determine_retained_orientation():
    sigma = np.diag([.8, .2])
    theta = .7
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    changed = rotation @ sigma @ rotation.T
    np.testing.assert_allclose(np.linalg.eigvalsh(sigma), np.linalg.eigvalsh(changed), atol=1e-15)
    p0 = np.asarray(reduced_rank_one_projector(jnp.asarray([.8, .2, 0.])))
    p1 = np.asarray(reduced_rank_one_projector(jnp.asarray([changed[0, 0], changed[1, 1], 2**.5*changed[0, 1]])))
    assert np.linalg.norm(p0-p1) > .8
    assert np.trace(p0 @ changed) < np.trace(p1 @ changed)-.1


def test_three_local_bonds_are_complete_for_real_three_spin_scalar_rdm():
    """An operator-level control independent of trained neural coefficients."""
    table = cell_schur_tensor("triangular_triangle")
    components = []
    for a, b in ((0, 0), (1, 1), (0, 1)):
        block = sum(np.outer(table[:, 1, m+3, a], table[:, 1, m+3, b]) for m in (-1, 1))/2
        if a != b:
            block = (block+block.T)/np.sqrt(2)
        components.append(block)
    components.append(sum(np.outer(table[:, 3, m+3, 0], table[:, 3, m+3, 0]) for m in (-3, -1, 1, 3))/4)
    observables = [np.eye(8)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        permutation = np.zeros((8, 8))
        for old in range(8):
            bits = [(old >> (2-i)) & 1 for i in range(3)]
            bits[a], bits[b] = bits[b], bits[a]
            new = 4*bits[0]+2*bits[1]+bits[2]
            permutation[new, old] = 1
        observables.append(.5*permutation-.25*np.eye(8))
    design = np.array([[np.trace(obs @ block) for block in components] for obs in observables])
    assert np.linalg.matrix_rank(design, tol=1e-12) == 4
    np.testing.assert_allclose(np.linalg.solve(design, design), np.eye(4), atol=2e-15)
    # Complex/chiral singlets have a further imaginary multiplicity component.
    # All three real bond expectations miss it, while a scalar commutator sees it.
    coherent = sum(np.outer(table[:, 1, m+3, 0], table[:, 1, m+3, 1]) for m in (-1, 1))/2
    imaginary = 1j*(coherent-coherent.T)
    base = .35*(components[0]+components[1])+.3*components[3]
    plus, minus = base+.1*imaginary, base-.1*imaginary
    assert np.linalg.eigvalsh(plus).min() > 0
    np.testing.assert_allclose([np.trace(obs @ plus) for obs in observables],
                               [np.trace(obs @ minus) for obs in observables], atol=2e-15)
    chirality = 1j*(observables[1] @ observables[3]-observables[3] @ observables[1])
    assert abs(np.trace(chirality @ (plus-minus))) > .1


def test_five_spin_rank_two_compression_preserves_exact_ten_spin_singlet():
    raw = jnp.asarray(np.random.default_rng(48).normal(size=(5, 2)))
    reduced = np.asarray(reduced_frame_projector(raw.ravel(), 5, 2))
    np.testing.assert_allclose(reduced @ reduced, reduced, atol=2e-14)
    np.testing.assert_allclose(np.trace(reduced), 2, atol=2e-14)
    physical = np.zeros((32, 32))
    for q in (1, 3, 5):
        for m in range(-q, q+1, 2):
            u = coupled_basis_matrix(enumerate_paths(5, q), q, m).real
            physical += u @ (reduced if q == 1 else np.eye(u.shape[1])) @ u.T
    basis = coupled_basis_matrix(enumerate_paths(10, 0), 0, 0).real
    acted = np.einsum("ij,jbk->ibk", physical, basis.reshape(32, 32, 42)).reshape(1024, 42)
    np.testing.assert_allclose(basis @ (basis.T @ acted), acted, atol=2e-14)
    config = EnvironmentCompressionConfig(12, output_dim=10, width=10)
    params = init_environment_compression(jax.random.PRNGKey(49), config, raw.ravel())
    output = predict_environment_block(params, jnp.ones((45, 12)), config, jnp.ones((3, 45))/45)
    assert output.shape == (10,)
    loss = lambda v: jnp.sum(reduced_frame_projector(v, 5, 2)*jnp.arange(25).reshape(5, 5))
    assert np.isfinite(jax.grad(loss)(raw.ravel())).all()


def test_environment_memory_excludes_parent_and_preserves_positive_reduced_blocks():
    """Attention here is exactly a Gaussian kernel, with no hidden teacher."""
    rng = np.random.default_rng(2026102201)
    features = jnp.asarray(rng.normal(size=(6, 7)))
    metric = jnp.asarray(rng.normal(size=(7, 3)))
    frames = rng.normal(size=(6, 42, 4))
    densities = np.einsum("bik,bjk->bij", frames, frames)
    densities /= np.trace(densities, axis1=1, axis2=2)[:, None, None]
    groups = np.repeat(np.arange(3), 2)
    allowed = jnp.asarray(groups[:, None] != groups[None, :])
    weights = environment_memory_weights(features, features, metric, allowed=allowed)
    np.testing.assert_allclose(np.asarray(weights).sum(1), 1, atol=2e-15)
    np.testing.assert_array_equal(np.asarray(weights)[~np.asarray(allowed)], 0)
    embedded = np.asarray(features @ metric)
    kernel = np.exp(-np.sum((embedded[:, None]-embedded[None, :])**2, axis=-1)/2)
    kernel *= np.asarray(allowed)
    kernel /= kernel.sum(1, keepdims=True)
    np.testing.assert_allclose(weights, kernel, atol=2e-15)
    predicted = np.asarray(predict_environment_memory(features, features, metric,
                                                     jnp.asarray(densities), allowed=allowed))
    assert np.linalg.eigvalsh(predicted).min() > -2e-15
    np.testing.assert_allclose(np.trace(predicted, axis1=1, axis2=2), 1, atol=2e-15)
    gradient = jax.grad(lambda a: jnp.sum(predict_environment_memory(
        features, features, a, jnp.asarray(densities), allowed=allowed)**2))(metric)
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 1e-6


def test_complete_nine_site_compression_commutes_with_total_spin_raising():
    """A 42-dimensional q=1 physical Schur-frame proposal is strictly scalar."""
    from su2_nqs.su2.triangle_blocks import triangle_block_tables
    rng = np.random.default_rng(2026102202)
    frame = np.linalg.qr(rng.normal(size=(42, 12)))[0]
    reduced = frame @ frame.T
    magnetizations = np.array([2*i.bit_count()-9 for i in range(512)])
    physical = np.zeros((512, 512))
    for q, table in triangle_block_tables()["nine"].items():
        block = reduced if q == 1 else np.eye(table.shape[1])
        for m in range(-q, q+1, 2):
            u = table*np.asarray(magnetizations == m)[:, None]
            physical += u @ block @ u.T
    raising = np.zeros((512, 512))
    for old in range(512):
        for site in range(9):
            bit = 1 << (8-site)
            if not old & bit:
                raising[old | bit, old] = 1
    np.testing.assert_allclose(physical @ physical, physical, atol=2e-14)
    np.testing.assert_allclose(physical @ raising, raising @ physical, atol=2e-14)
