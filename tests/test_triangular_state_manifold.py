from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.triangular_state_manifold import (
    TRIANGULAR_N16_RHOMBUS_SUPPORTS,
    phase_magnitude_tomography,
    triangular_rhombus_schur_config,
)
from su2_nqs.models.carrier_anchored_schur import (
    carrier_anchored_additive_state,
)
from su2_nqs.models.su2_four_cell_schur_vit import (
    exhaustive_four_cell_actions,
    init_su2_four_cell_schur_vit,
    su2_four_cell_schur_log_amplitudes,
)


jax.config.update("jax_enable_x64", True)


def test_rhombus_config_keeps_complete_n16_autoregressive_normalization() -> None:
    config = triangular_rhombus_schur_config(
        embedding_dim=8, head_count=2, depth=1
    )
    assert config.n_sites == 16
    assert len(TRIANGULAR_N16_RHOMBUS_SUPPORTS) == 4
    params = init_su2_four_cell_schur_vit(jax.random.PRNGKey(91), config)
    logs = su2_four_cell_schur_log_amplitudes(
        params, exhaustive_four_cell_actions(config), config
    )
    assert logs.shape == (1430,)
    np.testing.assert_allclose(
        np.sum(np.exp(2.0 * np.asarray(logs.real))), 1.0, atol=4.0e-14, rtol=0.0
    )


def test_additive_anchor_recovers_carrier_and_has_orthogonal_scale_tangent() -> None:
    carrier = jnp.asarray([1.0, 2.0j, -0.5, 0.25j])
    residual = jnp.asarray([0.2j, -0.7, 1.1j, 0.3])
    state = carrier_anchored_additive_state(carrier, residual, 0.0)
    normalized = carrier / jnp.linalg.norm(carrier)
    np.testing.assert_allclose(state, normalized, atol=1.0e-15, rtol=0.0)
    tangent = jax.jvp(
        lambda scale: carrier_anchored_additive_state(carrier, residual, scale),
        (jnp.asarray(0.0),),
        (jnp.asarray(1.0),),
    )[1]
    np.testing.assert_allclose(jnp.vdot(normalized, tangent), 0.0, atol=2.0e-15)
    changed = carrier_anchored_additive_state(carrier, residual, 0.2)
    assert np.linalg.norm(np.asarray(changed - normalized)) > 1.0e-3


def test_phase_magnitude_tomography_separates_the_two_error_sources() -> None:
    hamiltonian = np.asarray([[0.0, -1.0], [-1.0, 0.0]])
    exact = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    magnitude_error = np.asarray([2.0, 1.0]) / np.sqrt(5.0)
    report = phase_magnitude_tomography(exact, magnitude_error, hamiltonian)
    assert abs(report["candidate_phase_exact_magnitude_energy_error_total"]) < 1.0e-14
    assert report["exact_phase_candidate_magnitude_energy_error_total"] > 1.0e-2
    phase_error = np.asarray([1.0, -1.0]) / np.sqrt(2.0)
    report = phase_magnitude_tomography(exact, phase_error, hamiltonian)
    assert abs(report["magnitude_fidelity"] - 1.0) < 1.0e-14
    assert report["candidate_phase_exact_magnitude_energy_error_total"] > 1.0
