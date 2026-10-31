import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.reduced_channel_attention import init_channel_attention, channel_attention_coefficients
from su2_nqs.operators.cell_spin_channels import cell_casimir_action, cell_spin_projector_action, weighted_cell_exchange_action, four_site_spin_modes


def _independent_spin_matrices(n):
    pauli = (np.asarray([[0, 1], [1, 0]]) / 2, np.asarray([[0, 1j], [-1j, 0]]) / 2, np.diag([-0.5, 0.5]))
    matrices = []
    for i in range(n):
        triplet = []
        for s in pauli:
            value = np.ones((1, 1))
            for j in range(n):
                value = np.kron(value, s if i == j else np.eye(2))
            triplet.append(value)
        matrices.append(triplet)
    return matrices


def test_channel_transition_preserves_global_spin_but_not_cell_spin():
    spin = _independent_spin_matrices(4)
    eye = np.eye(16)
    pairs = {(i, j): sum(spin[i][k] @ spin[j][k] for k in range(3)) for i in range(4) for j in range(i + 1, 4)}
    pair = lambda v, i, j: pairs[tuple(sorted((i, j)))] @ v
    a, b = (0, 1), (2, 3)
    uniform = weighted_cell_exchange_action(eye, pair, a, b, (1, 1), (1, 1))
    transition = weighted_cell_exchange_action(eye, pair, a, b, (1, -1), (1, -1))
    ca = cell_casimir_action(eye, pair, a)
    total = sum(sum(spin[i][k] for i in range(4)) @ sum(spin[i][k] for i in range(4)) for k in range(3))
    assert np.max(np.abs(ca @ uniform - uniform @ ca)) < 1e-13
    assert np.linalg.norm(ca @ transition - transition @ ca) > 1
    assert np.max(np.abs(total @ transition - transition @ total)) < 1e-13
    assert np.max(np.abs(transition - transition.conj().T)) < 1e-13
    # Two dimer singlets -> coupled local triplets with total S=0.
    singlet = np.asarray([0, -1, 1, 0]) / np.sqrt(2)
    state = np.kron(singlet, singlet)
    image = transition @ state
    assert np.linalg.norm(image) > 1
    assert np.linalg.norm(ca @ image - 2 * image) < 1e-13
    assert np.linalg.norm(total @ image) < 1e-13


def test_casimir_projectors_against_independent_spectral_decomposition():
    spin = _independent_spin_matrices(4)
    eye = np.eye(16)
    pair = lambda v, i, j: sum(spin[i][k] @ spin[j][k] for k in range(3)) @ v
    total = sum(sum(spin[i][k] for i in range(4)) @ sum(spin[i][k] for i in range(4)) for k in range(3))
    eig, vec = np.linalg.eigh(total)
    projectors = []
    for q in (0, 2, 4):
        p = cell_spin_projector_action(eye, pair, tuple(range(4)), q)
        columns = vec[:, np.abs(eig - q * (q + 2) / 4) < 1e-10]
        assert np.max(np.abs(p - columns @ columns.conj().T)) < 2e-14
        assert np.max(np.abs(p @ p - p)) < 2e-14
        projectors.append(p)
    assert np.max(np.abs(sum(projectors) - eye)) < 2e-14
    assert np.max(np.abs(four_site_spin_modes() @ four_site_spin_modes().T - np.eye(4))) < 1e-14
    with pytest.raises(ValueError):
        cell_spin_projector_action(eye, pair, (0, 1), 1)
    with pytest.raises(ValueError):
        cell_casimir_action(eye, pair, (0.5, 1))


def test_attention_zero_start_and_trainable_conditional_coefficients():
    jax.config.update("jax_enable_x64", True)
    contexts = jnp.asarray(list(itertools.product(range(4), (0, 2, 4), (0, 2, 4))))
    coordinates = jnp.asarray(((0., 0.), (1., 0.), (0., 1.), (1., 1.)))
    p = init_channel_attention(jax.random.PRNGKey(41), 4)
    f = jax.jit(channel_attention_coefficients)
    assert np.max(np.abs(f(p, contexts, coordinates))) == 0
    p["attention"]["head"] = jnp.ones(4).at[0].set(0.2)
    values = np.asarray(f(p, contexts, coordinates))
    assert np.isfinite(values).all()
    assert np.ptp(values[:9]) > 1e-4
    g = jax.grad(lambda params: jnp.sum(f(params, contexts, coordinates) ** 2))(p)
    assert np.linalg.norm(g["attention"]["query"]) > 1e-8


def test_isospectral_control_fixes_carrier_and_all_residual_inner_products():
    from projects.clebsch_tree_nqs.workflows.spin_channel_attention import _scramble_fixing_carrier

    rng = np.random.default_rng(41)
    carrier = rng.normal(size=14)
    carrier /= np.linalg.norm(carrier)
    rotation = _scramble_fixing_carrier(carrier, 32)
    bank = rng.normal(size=(36, 14))
    scrambled = bank @ rotation.T
    assert np.max(np.abs(rotation.T @ rotation - np.eye(14))) < 2e-14
    assert np.max(np.abs(rotation @ carrier - carrier)) < 2e-14
    assert np.max(np.abs(bank @ bank.T - scrambled @ scrambled.T)) < 1e-13


def test_joint_and_staged_start_at_same_carrier_and_checkpoint(tmp_path):
    from su2_nqs.benchmarks.channel_attention import train_channel_schedule

    contexts = np.asarray(list(itertools.product(range(4), (0, 2, 4), (0, 2, 4))))
    coords = np.asarray(((0., 0.), (1., 0.), (0., 1.), (1., 1.)))
    rng = np.random.default_rng(15)
    carrier = np.asarray([1., 0., 0.])
    bank = rng.normal(size=(36, 3)) / 50
    h = np.asarray([[0., .1, .2], [.1, -.4, 0.], [.2, 0., .6]])
    for schedule in ("joint", "staged", "local_only"):
        _, info = train_channel_schedule(carrier, bank, h, contexts, coords,
            seed=24, width=4, steps=3, warmup=1, learning_rate=.001,
            schedule=schedule, checkpoint=tmp_path / f"{schedule}.pkl")
        assert info["history"][0]["energy_per_site"] == 0
        assert info["checkpoint_replay_max_error"] == 0
