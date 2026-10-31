import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.model_evaluation import parameter_count
from su2_nqs.benchmarks.optimization import build_exact_problem, evaluate_exact
from su2_nqs.models.full_hilbert_gru import (
    FullHilbertGRUConfig,
    full_hilbert_log_psi,
    init_full_hilbert_gru,
    sample_full_hilbert,
)
from su2_nqs.models.fusion_gru import FusionGRUConfig, init_fusion_gru
from su2_nqs.models.sz_gru import SzGRUConfig, init_sz_gru
from su2_nqs.models.symmetry_blind_gru import (
    SymmetryBlindGRUConfig,
    init_symmetry_blind_gru,
    sample_symmetry_blind,
    symmetry_blind_log_psi,
)
from su2_nqs.models.token_position_gru import (
    TokenPositionGRUConfig,
    init_token_position_gru,
    sample_token_position,
    token_position_log_psi,
)


def _all_bits(n: int) -> jax.Array:
    return jnp.asarray(
        [
            [(state >> (n - site - 1)) & 1 for site in range(n)]
            for state in range(1 << n)
        ],
        dtype=jnp.int32,
    )


def test_full_hilbert_gru_is_exhaustively_normalized_and_unrestricted():
    config = FullHilbertGRUConfig(6, embedding_dim=3, hidden_dim=5)
    params = init_full_hilbert_gru(jax.random.PRNGKey(12), config)
    logs = jax.vmap(lambda bits: full_hilbert_log_psi(params, bits, config))(
        _all_bits(config.n)
    )
    np.testing.assert_allclose(np.sum(np.exp(2.0 * np.real(logs))), 1.0, atol=1e-12)

    samples = np.asarray(
        sample_full_hilbert(params, jax.random.PRNGKey(13), 512, config)
    )
    assert samples.shape == (512, 6)
    assert np.all((samples == 0) | (samples == 1))
    assert len(np.unique(np.sum(samples, axis=1))) > 1


def test_full_control_matches_fusion_and_sz_parameter_budget():
    key = jax.random.PRNGKey(4)
    n = 8
    dimensions = dict(embedding_dim=4, hidden_dim=8)
    counts = {
        parameter_count(init_full_hilbert_gru(key, FullHilbertGRUConfig(n, **dimensions))),
        parameter_count(init_fusion_gru(key, FusionGRUConfig(n, 0, **dimensions))),
        parameter_count(init_sz_gru(key, SzGRUConfig(n, n // 2, **dimensions))),
        parameter_count(
            init_symmetry_blind_gru(key, SymmetryBlindGRUConfig(n, **dimensions))
        ),
    }
    assert len(counts) == 1


def test_full_exact_problem_has_correct_basis_operators_and_normalization():
    problem = build_exact_problem(4, "full", embedding_dim=2, hidden_dim=4)
    assert problem.basis.shape == (16, 4)
    params = problem.init(jax.random.PRNGKey(0))
    metrics = evaluate_exact(problem, params)
    np.testing.assert_allclose(metrics["normalization"], 1.0, atol=1e-12)
    assert metrics["spin_squared"] >= -1e-12
    np.testing.assert_allclose(problem.hamiltonian, problem.hamiltonian.T, atol=1e-14)


def test_symmetry_blind_control_is_normalized_without_quantum_number_input():
    config = SymmetryBlindGRUConfig(5, embedding_dim=3, hidden_dim=5)
    params = init_symmetry_blind_gru(jax.random.PRNGKey(21), config)
    logs = jax.vmap(lambda bits: symmetry_blind_log_psi(params, bits, config))(
        _all_bits(config.n)
    )
    np.testing.assert_allclose(np.sum(np.exp(2.0 * np.real(logs))), 1.0, atol=1e-12)
    samples = np.asarray(
        sample_symmetry_blind(params, jax.random.PRNGKey(22), 512, config)
    )
    assert len(np.unique(np.sum(samples, axis=1))) > 1

    problem = build_exact_problem(4, "blind", embedding_dim=2, hidden_dim=4)
    metrics = evaluate_exact(problem, problem.init(jax.random.PRNGKey(2)))
    np.testing.assert_allclose(metrics["normalization"], 1.0, atol=1e-12)


def test_token_position_control_is_normalized_and_charge_blind():
    config = TokenPositionGRUConfig(6, embedding_dim=4, hidden_dim=7)
    params = init_token_position_gru(jax.random.PRNGKey(31), config)
    logs = jax.vmap(lambda bits: token_position_log_psi(params, bits, config))(
        _all_bits(config.n)
    )
    np.testing.assert_allclose(np.sum(np.exp(2.0 * np.real(logs))), 1.0, atol=1e-12)
    samples = np.asarray(
        sample_token_position(params, jax.random.PRNGKey(32), 512, config)
    )
    assert samples.shape == (512, 6)
    assert len(np.unique(np.sum(samples, axis=1))) > 1

    problem = build_exact_problem(4, "tokenpos", embedding_dim=3, hidden_dim=6)
    metrics = evaluate_exact(problem, problem.init(jax.random.PRNGKey(3)))
    np.testing.assert_allclose(metrics["normalization"], 1.0, atol=1e-12)
