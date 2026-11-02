from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.models.fusion_gru import (
    FusionGRUConfig,
    init_fusion_gru,
    log_psi,
    path_probabilities,
    sample,
)
from su2_nqs.models.hubbard_fusion_gru import (
    HubbardFusionGRUConfig,
    hubbard_log_psi,
    hubbard_probabilities,
    hubbard_reachability_table,
    init_hubbard_fusion_gru,
    sample_hubbard_fusion,
    transfer_hubbard_fusion_gru_parameters,
)
from su2_nqs.models.sz_gru import SzGRUConfig, init_sz_gru, sample_sz, sz_log_psi
from su2_nqs.su2.fusion_path import enumerate_paths, is_valid_path
from su2_nqs.su2.hubbard_fusion import (
    decode_hubbard_fusion_state,
    encode_hubbard_fusion_state,
    enumerate_hubbard_fusion_states,
    is_valid_hubbard_fusion_state,
)
from su2_nqs.su2.sz_basis import enumerate_sz_configs


def test_fusion_gru_is_exhaustively_normalized_to_machine_precision():
    config = FusionGRUConfig(n=10, target_q=0, embedding_dim=5, hidden_dim=9)
    params = init_fusion_gru(jax.random.PRNGKey(10), config)
    paths = jnp.asarray(enumerate_paths(config.n, config.target_q))
    probabilities = path_probabilities(params, paths, config)
    np.testing.assert_allclose(float(probabilities.sum()), 1.0, atol=1e-12)


def test_hubbard_reachability_agrees_with_exact_sector_prefixes():
    n_sites, n_electrons, target_q = 4, 4, 0
    table = hubbard_reachability_table(n_sites, n_electrons, target_q)
    states = enumerate_hubbard_fusion_states(n_sites, n_electrons, target_q)
    prefixes = {(0, 0)}
    for used in range(n_sites + 1):
        remaining = n_sites - used
        exact = {
            (
                n_electrons - sum(state.occupations[:used]),
                0 if used == 0 else state.path[used - 1],
            )
            for state in states
        }
        actual = {
            (charge_left, current_q)
            for charge, current_q in prefixes
            for charge_left in (n_electrons - charge,)
            if 0 <= charge_left <= 2 * n_sites
            if table[remaining, charge_left, current_q]
        }
        assert actual == exact
        updated = set()
        for charge, current_q in prefixes:
            for occupation, delta_q in ((0, 0), (1, -1), (1, 1), (2, 0)):
                next_q = current_q + delta_q
                if next_q >= 0:
                    updated.add((charge + occupation, next_q))
        prefixes = updated


def test_hubbard_fusion_gru_is_exhaustively_normalized():
    config = HubbardFusionGRUConfig(
        n_sites=4,
        n_electrons=4,
        target_q=0,
        embedding_dim=5,
        hidden_dim=9,
    )
    params = init_hubbard_fusion_gru(jax.random.PRNGKey(101), config)
    states = enumerate_hubbard_fusion_states(4, 4, 0)
    labels = jnp.asarray([encode_hubbard_fusion_state(state) for state in states])
    probabilities = hubbard_probabilities(params, labels, config)
    np.testing.assert_allclose(float(probabilities.sum()), 1.0, atol=1e-12)
    invalid = labels[0].at[-1].set(2)
    assert jnp.isneginf(jnp.real(hubbard_log_psi(params, invalid, config)))


def test_hubbard_fusion_gru_size_transfer_copies_shared_labels_and_core():
    source_config = HubbardFusionGRUConfig(4, 4, 0, 3, 5)
    target_config = HubbardFusionGRUConfig(6, 6, 0, 3, 5)
    source = init_hubbard_fusion_gru(jax.random.PRNGKey(210), source_config)
    target = transfer_hubbard_fusion_gru_parameters(
        source, source_config, target_config, jax.random.PRNGKey(211)
    )
    for name in ("h0", "gru", "probability", "phase"):
        for source_leaf, target_leaf in zip(
            jax.tree_util.tree_leaves(source[name]),
            jax.tree_util.tree_leaves(target[name]),
            strict=True,
        ):
            np.testing.assert_array_equal(target_leaf, source_leaf)
    for source_embedding, target_embedding in zip(
        source["embeddings"], target["embeddings"], strict=True
    ):
        np.testing.assert_array_equal(
            target_embedding[: source_embedding.shape[0]], source_embedding
        )


def test_hubbard_fusion_gru_sampling_matches_probabilities_and_sector():
    config = HubbardFusionGRUConfig(
        n_sites=4,
        n_electrons=3,
        target_q=1,
        embedding_dim=4,
        hidden_dim=7,
    )
    params = init_hubbard_fusion_gru(jax.random.PRNGKey(102), config)
    states = enumerate_hubbard_fusion_states(4, 3, 1)
    labels = tuple(encode_hubbard_fusion_state(state) for state in states)
    expected = np.asarray(
        hubbard_probabilities(params, jnp.asarray(labels), config)
    )
    draws = np.asarray(
        sample_hubbard_fusion(
            params, jax.random.PRNGKey(103), 100_000, config
        )
    )
    decoded = [decode_hubbard_fusion_state(tuple(row)) for row in draws]
    assert all(
        is_valid_hubbard_fusion_state(
            state, n_electrons=config.n_electrons, target_q=config.target_q
        )
        for state in decoded
    )
    counts = Counter(map(tuple, draws))
    empirical = np.asarray([counts[state] / len(draws) for state in labels])
    np.testing.assert_allclose(empirical, expected, atol=7e-3)


def test_fusion_gru_sampling_frequencies_match_exact_probabilities():
    config = FusionGRUConfig(n=8, target_q=0, embedding_dim=4, hidden_dim=7)
    params = init_fusion_gru(jax.random.PRNGKey(11), config)
    paths = enumerate_paths(config.n, config.target_q)
    expected = np.asarray(path_probabilities(params, jnp.asarray(paths), config))
    draws = np.asarray(sample(params, jax.random.PRNGKey(12), 100_000, config))
    assert all(is_valid_path(tuple(row), 0) for row in draws)
    counts = Counter(map(tuple, draws))
    empirical = np.asarray([counts[path] / len(draws) for path in paths])
    np.testing.assert_allclose(empirical, expected, atol=7e-3)


def test_sz_gru_is_normalized_and_samples_only_the_requested_sector():
    config = SzGRUConfig(n=8, n_up=4, embedding_dim=5, hidden_dim=9)
    params = init_sz_gru(jax.random.PRNGKey(13), config)
    basis = jnp.asarray(enumerate_sz_configs(config.n, config.n_up))
    logs = jax.vmap(lambda bits: sz_log_psi(params, bits, config))(basis)
    np.testing.assert_allclose(float(jnp.exp(2 * jnp.real(logs)).sum()), 1.0, atol=1e-12)
    draws = np.asarray(sample_sz(params, jax.random.PRNGKey(14), 2_000, config))
    np.testing.assert_array_equal(draws.sum(axis=1), np.full(2_000, config.n_up))


def test_fusion_model_has_zero_spin_contamination_while_generic_sz_model_need_not():
    n = 6
    fusion_config = FusionGRUConfig(n=n, target_q=0, embedding_dim=4, hidden_dim=7)
    fusion_params = init_fusion_gru(jax.random.PRNGKey(15), fusion_config)
    paths = enumerate_paths(n, 0)
    logs = jax.vmap(lambda path: log_psi(fusion_params, path, fusion_config))(jnp.asarray(paths))
    physical_vector = coupled_basis_matrix(paths, 0, 0) @ np.asarray(jnp.exp(logs))
    spin_squared, _ = spin_squared_matrix(n)
    assert abs(np.vdot(physical_vector, spin_squared @ physical_vector)) < 1e-12

    sz_config = SzGRUConfig(n=n, n_up=n // 2, embedding_dim=4, hidden_dim=7)
    sz_params = init_sz_gru(jax.random.PRNGKey(16), sz_config)
    sz_basis = enumerate_sz_configs(n, n // 2)
    sz_logs = jax.vmap(lambda bits: sz_log_psi(sz_params, bits, sz_config))(jnp.asarray(sz_basis))
    sz_vector = np.zeros(2**n, dtype=np.complex128)
    for bits, amplitude in zip(sz_basis, np.asarray(jnp.exp(sz_logs))):
        index = sum(bit << (n - site - 1) for site, bit in enumerate(bits))
        sz_vector[index] = amplitude
    contamination = float(np.real(np.vdot(sz_vector, spin_squared @ sz_vector)))
    assert contamination > 1e-4
