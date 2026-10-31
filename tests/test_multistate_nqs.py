from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.multihead_fusion_gru import (
    MultiHeadFusionGRUConfig,
    init_multihead_fusion_gru,
    multihead_log_psi,
    multihead_path_probabilities,
    sample_multihead,
)
from su2_nqs.models.persistent_head_fusion_gru import (
    PersistentHeadFusionGRUConfig,
    init_persistent_head_fusion_gru,
    persistent_head_log_psi,
    persistent_head_probabilities,
)
from su2_nqs.su2.fusion_path import enumerate_paths, is_valid_path
from su2_nqs.vmc.multistate import (
    exact_ky_fan_loss,
    exact_projector_infidelity_loss,
    generalized_ritz_analysis,
    ky_fan_subspace_energy,
    subspace_projector_overlap,
    subspace_principal_fidelities,
)


def test_every_shared_core_head_is_exhaustively_normalized() -> None:
    config = MultiHeadFusionGRUConfig(8, 4, embedding_dim=5, hidden_dim=9)
    params = init_multihead_fusion_gru(jax.random.PRNGKey(17), config)
    paths = jnp.asarray(enumerate_paths(config.n, config.target_q))
    probabilities = multihead_path_probabilities(params, paths, config)
    assert probabilities.shape == (len(paths), config.n_heads)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(probabilities, axis=0)), np.ones(config.n_heads), atol=2.0e-14
    )


def test_selected_head_sampler_respects_exact_path_mask() -> None:
    config = MultiHeadFusionGRUConfig(8, 3, embedding_dim=4, hidden_dim=7)
    params = init_multihead_fusion_gru(jax.random.PRNGKey(2), config)
    samples = np.asarray(
        sample_multihead(params, jax.random.PRNGKey(9), 128, 1, config)
    )
    assert all(is_valid_path(tuple(int(value) for value in row), 0) for row in samples)
    with pytest.raises(ValueError, match="out of range"):
        sample_multihead(params, jax.random.PRNGKey(9), 2, 3, config)


def test_persistent_state_tokens_define_separate_normalized_heads() -> None:
    config = PersistentHeadFusionGRUConfig(
        8, 3, embedding_dim=4, hidden_dim=7, head_embedding_scale=0.2
    )
    params = init_persistent_head_fusion_gru(jax.random.PRNGKey(99), config)
    paths = jnp.asarray(enumerate_paths(8, 0))
    probabilities = persistent_head_probabilities(params, paths, config)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(probabilities, axis=0)), np.ones(3), atol=3.0e-14
    )
    logs = np.asarray(
        jax.vmap(lambda path: persistent_head_log_psi(params, path, config))(paths)
    )
    assert np.max(np.abs(logs[:, 0] - logs[:, 1])) > 1.0e-4


def test_persistent_token_row_permutation_only_permutes_heads() -> None:
    config = PersistentHeadFusionGRUConfig(6, 3, embedding_dim=3, hidden_dim=6)
    params = init_persistent_head_fusion_gru(jax.random.PRNGKey(101), config)
    paths = jnp.asarray(enumerate_paths(6, 0))
    original = np.asarray(
        jax.vmap(lambda path: persistent_head_log_psi(params, path, config))(paths)
    )
    permutation = np.asarray([2, 0, 1])
    changed = dict(params)
    embeddings = list(params["embeddings"])
    embeddings[-1] = embeddings[-1][permutation]
    changed["embeddings"] = tuple(embeddings)
    permuted = np.asarray(
        jax.vmap(lambda path: persistent_head_log_psi(changed, path, config))(paths)
    )
    np.testing.assert_allclose(permuted, original[:, permutation], atol=3.0e-14)


def test_ky_fan_energy_is_invariant_under_nonsingular_head_mixing() -> None:
    rng = np.random.default_rng(31)
    raw = rng.normal(size=(9, 3)) + 1.0j * rng.normal(size=(9, 3))
    h_raw = rng.normal(size=(9, 9)) + 1.0j * rng.normal(size=(9, 9))
    hamiltonian = h_raw + h_raw.conj().T
    mixing = np.asarray(
        [[1.0, 0.2, 0.0], [0.1j, 0.8, -0.3], [0.0, 0.4, 1.3]],
        dtype=np.complex128,
    )
    first = float(ky_fan_subspace_energy(jnp.asarray(raw), jnp.asarray(hamiltonian)))
    second = float(
        ky_fan_subspace_energy(jnp.asarray(raw @ mixing), jnp.asarray(hamiltonian))
    )
    assert first == pytest.approx(second, abs=2.0e-12)


def test_ky_fan_energy_fails_closed_for_duplicate_columns() -> None:
    hamiltonian = jnp.diag(jnp.asarray([-5.0, -4.0, 1.0, 2.0]))
    first = jnp.asarray([1.0, 0.0, 0.0, 0.0])
    duplicate = jnp.stack([first, first], axis=1)
    assert np.isinf(float(ky_fan_subspace_energy(duplicate, hamiltonian)))


def test_generalized_ritz_and_projector_overlap_match_exact_span() -> None:
    hamiltonian = np.diag([-4.0, -2.0, 1.0, 3.0])
    states = np.asarray(
        [[1.0, 1.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.complex128,
    )
    result = generalized_ritz_analysis(states, hamiltonian)
    assert result["rank"] == 2
    np.testing.assert_allclose(result["ritz_values"], [-4.0, -2.0], atol=2.0e-13)
    np.testing.assert_allclose(result["ritz_residuals"], [0.0, 0.0], atol=2.0e-13)
    exact = np.eye(4, 2, dtype=np.complex128)
    assert subspace_projector_overlap(result["orthonormal_basis"], exact) == pytest.approx(1.0)
    np.testing.assert_allclose(
        subspace_principal_fidelities(result["orthonormal_basis"], exact),
        np.ones(2),
        atol=2.0e-13,
    )


def test_projector_infidelity_and_principal_angles_resolve_a_bad_direction() -> None:
    theta = 0.3
    target = np.eye(4, 2, dtype=np.complex128)
    trial = np.asarray(
        [
            [1.0, 0.0],
            [0.0, np.cos(theta)],
            [0.0, np.sin(theta)],
            [0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    fidelities = subspace_principal_fidelities(target, trial)
    np.testing.assert_allclose(fidelities, [np.cos(theta) ** 2, 1.0], atol=2.0e-14)
    assert subspace_projector_overlap(target, trial) == pytest.approx(np.mean(fidelities))

    # A direct dummy log table closes the JAX oracle definition.
    paths = jnp.arange(4, dtype=jnp.int32)[:, None]
    amplitudes = jnp.asarray(trial)
    logs = jnp.where(jnp.abs(amplitudes) > 0, jnp.log(amplitudes), -jnp.inf + 0j)
    log_fn = lambda table, path: table[path[0]]
    loss = exact_projector_infidelity_loss(logs, paths, jnp.asarray(target), log_fn)
    assert float(loss) == pytest.approx(1.0 - np.cos(theta) ** 2)


def test_exact_multistate_loss_has_finite_gradient() -> None:
    config = MultiHeadFusionGRUConfig(6, 2, embedding_dim=3, hidden_dim=5)
    paths = jnp.asarray(enumerate_paths(6, 0))
    hamiltonian = jnp.diag(jnp.linspace(-2.0, 2.0, len(paths)))
    params = init_multihead_fusion_gru(jax.random.PRNGKey(4), config)
    log_fn = lambda candidate, path: multihead_log_psi(candidate, path, config)
    value, gradient = jax.value_and_grad(exact_ky_fan_loss)(
        params, paths, hamiltonian, log_fn
    )
    assert np.isfinite(float(value))
    assert all(
        np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(gradient)
    )
