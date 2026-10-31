import numpy as np
import pytest

from su2_nqs.benchmarks.hubbard_ground import (
    hubbard_fock_vector,
    hubbard_ground_state_metrics,
    hubbard_variational_state_metrics,
)
from su2_nqs.operators.hubbard import hubbard_reduced_fusion_sparse


def test_hubbard_ground_metrics_preserve_normalization_and_moment_identity():
    matrix, states = hubbard_reduced_fusion_sparse(
        2, 2, 0, ((0, 1),), interaction=4.0
    )
    values, vectors = np.linalg.eigh(matrix.toarray())
    ground = np.asarray(vectors[:, 0], dtype=np.complex128)
    residual = np.linalg.norm(matrix @ ground - values[0] * ground)
    metrics = hubbard_ground_state_metrics(
        matrix, states, values[0], ground, residual, 0.0
    )
    np.testing.assert_allclose(metrics["normalization"], 1.0, atol=2e-14)
    np.testing.assert_allclose(
        metrics["local_moment_per_site"],
        1.0 - 2.0 * metrics["double_occupancy_per_site"],
        atol=2e-14,
    )
    np.testing.assert_allclose(metrics["fock_normalization"], 1.0, atol=2e-14)
    assert metrics["staggered_spin_structure_factor"] >= 0.0
    assert metrics["staggered_charge_structure_factor"] >= 0.0


def test_hubbard_fock_vector_rejects_a_mismatched_coefficient_count():
    _, states = hubbard_reduced_fusion_sparse(
        2, 2, 0, ((0, 1),), interaction=4.0
    )
    with pytest.raises(ValueError, match="coefficient count"):
        hubbard_fock_vector(states, np.ones(len(states) + 1))


def test_variational_metrics_normalize_scale_and_match_exact_state():
    matrix, states = hubbard_reduced_fusion_sparse(
        2, 2, 0, ((0, 1),), interaction=4.0
    )
    values, vectors = np.linalg.eigh(matrix.toarray())
    ground = np.asarray(vectors[:, 0], dtype=np.complex128)
    metrics = hubbard_variational_state_metrics(
        matrix, states, 3.0j * ground, ground, float(values[0])
    )
    np.testing.assert_allclose(metrics["normalization"], 1.0, atol=2e-14)
    np.testing.assert_allclose(metrics["energy_error"], 0.0, atol=2e-14)
    np.testing.assert_allclose(metrics["variance_per_site"], 0.0, atol=2e-14)
    np.testing.assert_allclose(metrics["ground_state_fidelity"], 1.0, atol=2e-14)
