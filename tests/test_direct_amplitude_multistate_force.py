from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.direct_amplitude_multistate_force import (
    direct_amplitude_ky_fan_horizontal_force,
    prepare_direct_amplitude_ky_fan_force_reference,
)
from su2_nqs.vmc.resonating_projected_agp import (
    ResonatingProjectedAGPAction,
    solve_three_root_resonating_subspace,
)


def _problem(seed: int = 2026091450):
    rng = np.random.default_rng(seed)
    dimension, bank_size = 9, 4
    raw = rng.normal(size=(dimension, dimension))
    hamiltonian = 0.5 * (raw + raw.T)
    parameters = rng.normal(size=(dimension, bank_size))
    # An exact node exercises the direct-amplitude path.
    parameters[0, 0] = 0.0
    proposal = rng.random(dimension) + 0.2
    proposal /= np.sum(proposal)
    return hamiltonian, parameters, proposal


def _action(hamiltonian: np.ndarray, amplitudes: np.ndarray):
    actions = hamiltonian @ amplitudes
    return ResonatingProjectedAGPAction(
        amplitudes=amplitudes,
        hamiltonian_actions=actions,
        center_exact_nodes=amplitudes == 0.0,
        minimum_connected_cancellation_ratio=np.ones_like(amplitudes),
    )


def _mean_ky_fan(
    hamiltonian: np.ndarray,
    amplitudes: np.ndarray,
    proposal: np.ndarray,
) -> float:
    result = solve_three_root_resonating_subspace(
        _action(hamiltonian, amplitudes),
        proposal,
        observation_weights=proposal,
    )
    return float(np.mean(result.energies))


def test_exhaustive_direct_force_matches_directional_finite_difference() -> None:
    hamiltonian, parameters, proposal = _problem()
    reference = prepare_direct_amplitude_ky_fan_force_reference(
        _action(hamiltonian, parameters),
        proposal,
        observation_weights=proposal,
    )
    observed = direct_amplitude_ky_fan_horizontal_force(
        jnp.asarray(parameters), reference, lambda value: value
    )
    rng = np.random.default_rng(2026091451)
    direction = rng.normal(size=parameters.shape)
    direction /= np.linalg.norm(direction)
    step = 2.0e-6
    finite_difference = (
        _mean_ky_fan(hamiltonian, parameters + step * direction, proposal)
        - _mean_ky_fan(hamiltonian, parameters - step * direction, proposal)
    ) / (2.0 * step)
    automatic = float(np.sum(np.asarray(observed.gradient) * direction))
    assert observed.energy == pytest.approx(
        _mean_ky_fan(hamiltonian, parameters, proposal), abs=2.0e-14
    )
    assert observed.surrogate_value_at_reference == pytest.approx(0.0, abs=2.0e-14)
    assert reference.maximum_horizontality_error < 3.0e-14
    np.testing.assert_allclose(automatic, finite_difference, rtol=3.0e-7, atol=3.0e-9)


def test_carrier_basis_gauge_directions_have_zero_force() -> None:
    hamiltonian, parameters, proposal = _problem(2026091452)
    reference = prepare_direct_amplitude_ky_fan_force_reference(
        _action(hamiltonian, parameters),
        proposal,
        observation_weights=proposal,
    )
    observed = direct_amplitude_ky_fan_horizontal_force(
        jnp.asarray(parameters), reference, lambda value: value
    )
    mixing = np.asarray(
        [[0.2, -0.1, 0.3, 0.0], [0.1, 0.4, 0.0, -0.2],
         [-0.3, 0.1, 0.2, 0.1], [0.0, -0.2, 0.1, -0.1]]
    )
    gauge_direction = parameters @ mixing
    derivative = float(np.sum(np.asarray(observed.gradient) * gauge_direction))
    assert abs(derivative) < 2.0e-13


def test_direct_force_contracts_fail_closed() -> None:
    hamiltonian, parameters, proposal = _problem(2026091453)
    action = _action(hamiltonian, parameters)
    with pytest.raises(ValueError, match="positive"):
        prepare_direct_amplitude_ky_fan_force_reference(
            action, np.zeros_like(proposal), observation_weights=proposal
        )
    reference = prepare_direct_amplitude_ky_fan_force_reference(
        action, proposal, observation_weights=proposal
    )
    with pytest.raises(ValueError, match="shape"):
        direct_amplitude_ky_fan_horizontal_force(
            jnp.asarray(parameters), reference, lambda value: value[:, :3]
        )
    with pytest.raises(TypeError, match="real parameters"):
        direct_amplitude_ky_fan_horizontal_force(
            jnp.asarray(parameters, dtype=jnp.complex128),
            reference,
            lambda value: value,
        )
    with pytest.raises(ValueError, match="frozen reference"):
        direct_amplitude_ky_fan_horizontal_force(
            jnp.asarray(parameters + 1.0e-3),
            reference,
            lambda value: value,
        )


def test_finite_sample_force_is_horizontal_to_the_full_carrier_bank() -> None:
    rng = np.random.default_rng(2026091454)
    full_dimension, sample_count, bank_size = 13, 8, 5
    raw = rng.normal(size=(full_dimension, full_dimension))
    hamiltonian = 0.5 * (raw + raw.T)
    full_amplitudes = rng.normal(size=(full_dimension, bank_size))
    # A simultaneous node of every carrier is legal for direct amplitudes.
    full_amplitudes[0, :] = 0.0
    rows = np.arange(sample_count)
    amplitudes = full_amplitudes[rows]
    actions = (hamiltonian @ full_amplitudes)[rows]
    proposal = rng.random(sample_count) + 0.1
    proposal /= np.sum(proposal)
    action = ResonatingProjectedAGPAction(
        amplitudes=amplitudes,
        hamiltonian_actions=actions,
        center_exact_nodes=amplitudes == 0.0,
        minimum_connected_cancellation_ratio=np.ones_like(amplitudes),
    )
    reference = prepare_direct_amplitude_ky_fan_force_reference(
        action,
        proposal,
        observation_weights=None,
        maximum_condition_number=1.0e10,
    )
    observed = direct_amplitude_ky_fan_horizontal_force(
        jnp.asarray(amplitudes), reference, lambda value: value
    )
    mixing = rng.normal(size=(bank_size, bank_size))
    gauge_direction = amplitudes @ mixing
    derivative = float(np.sum(np.asarray(observed.gradient) * gauge_direction))
    assert abs(derivative) < 2.0e-12
    scaled = amplitudes / reference.column_rescales[None, :]
    projected = np.einsum(
        "x,xa,xb->ab",
        reference.measure,
        scaled.conj(),
        reference.horizontal_residual,
    )
    assert np.max(np.abs(projected)) < 2.0e-12


def test_lowest_three_force_rejects_an_unresolved_fourth_root() -> None:
    amplitudes = np.eye(4, dtype=np.float64)
    hamiltonian = np.diag([0.0, 1.0, 2.0, 2.0])
    proposal = np.full(4, 0.25)
    with pytest.raises(FloatingPointError, match="Ky-Fan boundary"):
        prepare_direct_amplitude_ky_fan_force_reference(
            _action(hamiltonian, amplitudes),
            proposal,
            observation_weights=proposal,
        )


def test_complex_amplitude_force_matches_real_parameter_finite_difference() -> None:
    rng = np.random.default_rng(2026091455)
    dimension, bank_size = 10, 4
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    hamiltonian = 0.5 * (raw + raw.conj().T)
    params = {
        "real": rng.normal(size=(dimension, bank_size)),
        "imag": rng.normal(size=(dimension, bank_size)),
    }

    def amplitudes_fn(values):
        return values["real"] + 1j * values["imag"]

    amplitudes = np.asarray(amplitudes_fn(params))
    proposal = rng.random(dimension) + 0.2
    proposal /= np.sum(proposal)
    reference = prepare_direct_amplitude_ky_fan_force_reference(
        _action(hamiltonian, amplitudes),
        proposal,
        observation_weights=proposal,
    )
    observed = direct_amplitude_ky_fan_horizontal_force(
        {name: jnp.asarray(value) for name, value in params.items()},
        reference,
        amplitudes_fn,
    )
    direction = {
        name: rng.normal(size=value.shape) for name, value in params.items()
    }
    norm = np.sqrt(sum(np.sum(value**2) for value in direction.values()))
    direction = {name: value / norm for name, value in direction.items()}
    step = 2.0e-6
    plus = {name: params[name] + step * direction[name] for name in params}
    minus = {name: params[name] - step * direction[name] for name in params}
    finite_difference = (
        _mean_ky_fan(hamiltonian, np.asarray(amplitudes_fn(plus)), proposal)
        - _mean_ky_fan(hamiltonian, np.asarray(amplitudes_fn(minus)), proposal)
    ) / (2.0 * step)
    automatic = sum(
        float(np.sum(np.asarray(observed.gradient[name]) * direction[name]))
        for name in params
    )
    np.testing.assert_allclose(automatic, finite_difference, rtol=5.0e-7, atol=5.0e-9)
