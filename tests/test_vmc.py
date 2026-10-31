import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.models.fusion_gru import FusionGRUConfig, init_fusion_gru, log_psi, sample
from su2_nqs.models.hubbard_fusion_gru import (
    HubbardFusionGRUConfig,
    hubbard_log_psi,
    init_hubbard_fusion_gru,
)
from su2_nqs.operators.heisenberg_chain import HeisenbergChain
from su2_nqs.operators.hubbard import HubbardHamiltonian
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.vmc.adam import train_adam, train_exact_adam
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.local_energy import local_energies


def _small_problem(seed=20):
    config = FusionGRUConfig(n=4, target_q=0, embedding_dim=4, hidden_dim=8)
    params = init_fusion_gru(jax.random.PRNGKey(seed), config)
    operator = HeisenbergChain(config.n, config.target_q)
    log_fn = lambda candidate, path: log_psi(candidate, path, config)
    sample_fn = lambda candidate, key, count: sample(candidate, key, count, config)
    return config, params, operator, log_fn, sample_fn


def test_local_energy_reproduces_exact_variational_energy():
    config, params, operator, log_fn, _ = _small_problem()
    basis = enumerate_paths(config.n, config.target_q)
    logs = np.asarray([log_fn(params, np.asarray(path)) for path in basis])
    amplitudes = np.exp(logs)
    probabilities = np.abs(amplitudes) ** 2
    locals_ = np.asarray(local_energies(params, basis, log_fn, operator.get_conn))
    sampled_formula = np.sum(probabilities * locals_)
    exact_formula = np.vdot(amplitudes, operator.to_dense() @ amplitudes)
    np.testing.assert_allclose(sampled_formula, exact_formula, atol=2e-13)


def test_adam_vmc_converges_to_small_chain_exact_diagonalization():
    config, params, operator, log_fn, sample_fn = _small_problem(seed=0)
    exact_energy = np.linalg.eigvalsh(operator.to_dense())[0]
    _, history = train_adam(
        params,
        jax.random.PRNGKey(2),
        sample_fn=sample_fn,
        log_psi_fn=log_fn,
        get_conn_fn=operator.get_conn,
        steps=60,
        batch_size=128,
        learning_rate=3e-2,
    )
    assert history[-1]["energy"] < exact_energy + 5e-3
    assert history[-1]["variance"] < 1e-2


def test_exact_adam_reuses_shared_update_and_reports_compile_time():
    params = {"x": jnp.asarray(3.0)}
    trained, report = train_exact_adam(
        params,
        loss_fn=lambda candidate: (candidate["x"] - 1.0) ** 2,
        steps=100,
        learning_rate=5.0e-2,
        record_every=25,
    )
    assert abs(float(trained["x"]) - 1.0) < 2.0e-2
    assert report["initial_loss"] == 4.0
    assert report["compile_seconds"] >= 0.0
    assert report["training_seconds"] >= 0.0
    assert [row["step"] for row in report["history"]] == [25.0, 50.0, 75.0, 100.0]


def test_checkpoint_round_trip(tmp_path):
    _, params, _, _, _ = _small_problem()
    destination = tmp_path / "state.pkl"
    save_checkpoint(destination, params, metadata={"step": 3})
    restored = load_checkpoint(destination)
    assert restored["metadata"] == {"step": 3}
    for actual, expected in zip(
        jax.tree_util.tree_leaves(restored["params"]),
        jax.tree_util.tree_leaves(params),
    ):
        np.testing.assert_array_equal(actual, expected)


def test_hubbard_local_energy_reproduces_exact_variational_energy():
    config = HubbardFusionGRUConfig(
        n_sites=4,
        n_electrons=4,
        target_q=0,
        embedding_dim=4,
        hidden_dim=7,
    )
    params = init_hubbard_fusion_gru(jax.random.PRNGKey(104), config)
    operator = HubbardHamiltonian(
        n_sites=4,
        n_electrons=4,
        target_q=0,
        bonds=((0, 1), (0, 2), (1, 3), (2, 3)),
        interaction=4.0,
    )
    basis = jnp.asarray(operator.basis)
    log_fn = lambda candidate, state: hubbard_log_psi(candidate, state, config)
    logs = np.asarray(jax.vmap(lambda state: log_fn(params, state))(basis))
    amplitudes = np.exp(logs)
    probabilities = np.abs(amplitudes) ** 2
    locals_ = np.asarray(
        local_energies(params, operator.basis, log_fn, operator.get_conn)
    )
    sampled_formula = np.sum(probabilities * locals_)
    exact_formula = np.vdot(amplitudes, operator.to_dense() @ amplitudes)
    np.testing.assert_allclose(sampled_formula, exact_formula, atol=3e-13)
