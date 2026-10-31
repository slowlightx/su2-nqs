from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.exact_log_derivative_sr import (
    exact_log_derivative_natural_gradient,
    exact_log_derivative_natural_gradient_from_data,
)


jax.config.update("jax_enable_x64", True)


def test_two_level_natural_gradient_points_downhill() -> None:
    result = exact_log_derivative_natural_gradient(
        jnp.asarray([0.0], dtype=jnp.float64),
        log_amplitude_fn=lambda value: jnp.asarray(
            [value[0] + 0.0j, 0.0 + 0.0j]
        ),
        hamiltonian=jnp.diag(jnp.asarray([0.0, 1.0], dtype=jnp.float64)),
        damping=1.0e-3,
    )
    assert result.energy == pytest.approx(0.5, abs=1.0e-14)
    assert result.variance == pytest.approx(0.25, abs=1.0e-14)
    assert result.gradient[0] == pytest.approx(-0.5, abs=1.0e-14)
    assert result.metric[0, 0] == pytest.approx(0.25, abs=1.0e-14)
    assert result.descent_direction[0] > 0.0
    assert result.numerical_rank == 1
    assert result.relative_linear_residual < 1.0e-12


def test_global_log_norm_and_phase_directions_have_zero_metric() -> None:
    result = exact_log_derivative_natural_gradient(
        jnp.zeros((2,), dtype=jnp.float64),
        log_amplitude_fn=lambda value: jnp.asarray(
            [value[0] + 1.0j * value[1], value[0] + 1.0j * value[1]]
        ),
        hamiltonian=jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float64),
        damping=1.0e-3,
    )
    np.testing.assert_allclose(result.metric, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.gradient, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.descent_direction, 0.0, atol=1.0e-14)
    assert result.numerical_rank == 0


def test_cached_derivative_entry_point_matches_direct_evaluation() -> None:
    parameters = jnp.asarray([0.2], dtype=jnp.float64)
    function = lambda value: jnp.asarray([value[0] + 0.0j, 0.0 + 0.0j])
    matrix = jnp.diag(jnp.asarray([0.0, 1.0], dtype=jnp.float64))
    direct = exact_log_derivative_natural_gradient(
        parameters,
        log_amplitude_fn=function,
        hamiltonian=matrix,
        damping=1.0e-3,
    )
    cached = exact_log_derivative_natural_gradient_from_data(
        function(parameters),
        jax.jacfwd(function)(parameters),
        hamiltonian=matrix,
        damping=1.0e-3,
    )
    assert cached.energy == pytest.approx(direct.energy, abs=1.0e-14)
    np.testing.assert_allclose(cached.gradient, direct.gradient, atol=1.0e-14)
    np.testing.assert_allclose(
        cached.descent_direction, direct.descent_direction, atol=1.0e-14
    )


def test_large_global_log_shift_is_numerically_invariant() -> None:
    matrix = jnp.asarray([[0.0, 1.0], [1.0, 0.2]], dtype=jnp.float64)
    derivatives = jnp.asarray([[1.0], [0.0]], dtype=jnp.complex128)
    reference = exact_log_derivative_natural_gradient_from_data(
        jnp.asarray([0.2, -0.1], dtype=jnp.complex128),
        derivatives,
        hamiltonian=matrix,
        damping=1.0e-3,
    )
    shifted = exact_log_derivative_natural_gradient_from_data(
        jnp.asarray([1000.2, 999.9], dtype=jnp.complex128),
        derivatives,
        hamiltonian=matrix,
        damping=1.0e-3,
    )
    assert shifted.energy == pytest.approx(reference.energy, abs=1.0e-13)
    assert shifted.variance == pytest.approx(reference.variance, abs=1.0e-13)
    np.testing.assert_allclose(shifted.gradient, reference.gradient, atol=1.0e-13)


@pytest.mark.parametrize("damping", [0.0, -1.0, np.inf])
def test_invalid_damping_is_rejected(damping: float) -> None:
    with pytest.raises(ValueError):
        exact_log_derivative_natural_gradient(
            jnp.zeros((1,), dtype=jnp.float64),
            log_amplitude_fn=lambda value: value.astype(jnp.complex128),
            hamiltonian=jnp.zeros((1, 1), dtype=jnp.float64),
            damping=damping,
        )
