import jax
import jax.numpy as jnp

from su2_nqs.vmc.distillation import (
    exact_fidelity_distillation_loss,
    exact_graph_fidelity_distillation_loss,
    sampled_distillation_loss,
    train_exact_fidelity_distillation,
    widen_gru_hidden,
)
from su2_nqs.models._core import init_gru_core


def test_sampled_distillation_prefers_target_distribution_and_ignores_global_phase():
    samples = jnp.asarray([[0], [1]], dtype=jnp.int32)
    weights = jnp.asarray([0.8, 0.2])
    phases = jnp.asarray([0.2, -0.1])

    def log_psi(logits, sample):
        log_probability = jax.nn.log_softmax(logits)[sample[0]]
        return 0.5 * log_probability + 1j * (phases[sample[0]] + 0.7)

    matched, metrics = sampled_distillation_loss(
        jnp.log(weights), samples, weights, phases, log_psi
    )
    reversed_loss, _ = sampled_distillation_loss(
        jnp.log(weights[::-1]), samples, weights, phases, log_psi
    )
    assert matched < reversed_loss
    assert metrics["phase_loss"] < 1e-12


def test_widen_gru_preserves_old_hidden_subspace():
    params = init_gru_core(
        jax.random.PRNGKey(0), embedding_sizes=(2,), embedding_dim=2,
        hidden_dim=3, output_dim=2
    )
    widened = widen_gru_hidden(params, 5)
    assert widened["h0"].shape == (5,)
    assert widened["gru"]["wz"].shape[1] == 5
    assert widened["probability"]["w"].shape[0] == 5


def test_exact_fidelity_loss_is_normalized_and_global_phase_invariant():
    basis = jnp.asarray([[0], [1]], dtype=jnp.int32)
    target = jnp.asarray([jnp.sqrt(0.8), 1j * jnp.sqrt(0.2)])

    def log_amplitude(params, state):
        index = state[0]
        log_probability = jax.nn.log_softmax(params["logits"])[index]
        return 0.5 * log_probability + 1j * params["phases"][index]

    params = {
        "logits": jnp.log(jnp.asarray([0.8, 0.2])),
        "phases": jnp.asarray([0.4, 0.4 + 0.5 * jnp.pi]),
    }
    loss, metrics = exact_fidelity_distillation_loss(
        params, basis, target * jnp.exp(0.73j), log_amplitude
    )
    assert loss < 2e-12
    assert metrics["infidelity"] < 2e-12
    assert jnp.abs(metrics["student_normalization"] - 1.0) < 2e-12


def test_exact_fidelity_distillation_fits_complex_two_state_target():
    basis = jnp.asarray([[0], [1]], dtype=jnp.int32)
    target = jnp.asarray(
        [jnp.sqrt(0.73) * jnp.exp(0.2j), jnp.sqrt(0.27) * jnp.exp(-0.35j)]
    )

    def log_amplitude(params, state):
        index = state[0]
        log_probability = jax.nn.log_softmax(params["logits"])[index]
        return 0.5 * log_probability + 1j * params["phases"][index]

    initial = {
        "logits": jnp.zeros(2, dtype=jnp.float64),
        "phases": jnp.zeros(2, dtype=jnp.float64),
    }
    _, history = train_exact_fidelity_distillation(
        initial,
        basis=basis,
        target=target,
        log_psi_fn=log_amplitude,
        steps=200,
        learning_rate=0.05,
        record_every=50,
    )
    assert history[-1]["fidelity_squared"] > 1.0 - 2e-9
    assert history[-1]["target_to_student_born_kl"] < 2e-9


def test_graph_fidelity_has_the_same_target_ray_and_reports_ordinary_fidelity():
    basis = jnp.asarray([[0], [1]], dtype=jnp.int32)
    target = jnp.asarray([jnp.sqrt(0.6), 1j * jnp.sqrt(0.4)])
    hamiltonian = jnp.asarray([[0.0, 0.2], [0.2, 1.0]])

    def log_amplitude(params, state):
        index = state[0]
        log_probability = jax.nn.log_softmax(params["logits"])[index]
        return 0.5 * log_probability + 1j * params["phases"][index]

    params = {
        "logits": jnp.log(jnp.asarray([0.6, 0.4])),
        "phases": jnp.asarray([0.3, 0.3 + 0.5 * jnp.pi]),
    }
    loss, metrics = exact_graph_fidelity_distillation_loss(
        params,
        basis,
        target * jnp.exp(-0.4j),
        hamiltonian,
        0.3,
        log_amplitude,
    )
    assert loss < 2e-12
    assert metrics["graph_infidelity"] < 2e-12
    assert metrics["infidelity"] < 2e-12
