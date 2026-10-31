import math

import numpy as np
import pytest

from su2_nqs.benchmarks.spectral_certificate import (
    residual_gap_reference_weight_certificate,
)


def test_residual_gap_bound_is_conservative_against_an_exact_diagonal_example():
    eigenvalues = np.asarray([0.0, 1.0, 3.0])
    probabilities = np.asarray([0.98, 0.02, 0.0])
    energy = float(probabilities @ eigenvalues)
    residual = math.sqrt(float(probabilities @ (eigenvalues - energy) ** 2))
    complement_gap = float(np.min(np.abs(eigenvalues[1:] - energy)))
    certificate = residual_gap_reference_weight_certificate(
        residual, complement_gap
    )
    assert certificate.physical_weight_lower_bound <= probabilities[0]
    assert certificate.passes
    assert certificate.residual_to_gap_upper == pytest.approx(
        residual / complement_gap
    )


def test_residual_gap_certificate_uses_conservative_confidence_bounds():
    nominal = residual_gap_reference_weight_certificate(0.1, 0.5)
    conservative = residual_gap_reference_weight_certificate(0.12, 0.45)
    assert conservative.raw_weight_lower_bound < nominal.raw_weight_lower_bound
    assert nominal.raw_weight_lower_bound == pytest.approx(0.96)


def test_residual_gap_certificate_fails_closed_for_unresolved_complement():
    with pytest.raises(ValueError, match="positive"):
        residual_gap_reference_weight_certificate(0.01, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        residual_gap_reference_weight_certificate(-0.01, 0.5)
    failed = residual_gap_reference_weight_certificate(1.0, 0.5)
    assert failed.physical_weight_lower_bound == 0.0
    assert failed.raw_weight_lower_bound < 0.0
    assert not failed.passes
