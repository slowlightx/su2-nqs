"""Exact checks for observable-resolved imaginary-time helpers."""

from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.observable_flow import (
    exact_imaginary_time_flow,
    full_reorthogonalized_ritz_flow,
    scalar_moments,
)


def test_exact_imaginary_time_is_normalized_and_monotone() -> None:
    hamiltonian = np.asarray([[0.0, 0.2], [0.2, 1.0]])
    ground, points = exact_imaginary_time_flow(
        hamiltonian, np.asarray([0.3, 1.0]), (0.0, 0.2, 1.0, 8.0)
    )
    np.testing.assert_allclose([np.linalg.norm(p.state) for p in points], 1.0)
    assert all(a.energy >= b.energy - 1.0e-14 for a, b in zip(points, points[1:]))
    assert all(
        a.ground_fidelity <= b.ground_fidelity + 1.0e-14
        for a, b in zip(points, points[1:])
    )
    assert abs(np.vdot(ground, points[-1].state)) ** 2 > 0.999


def test_nested_ritz_energy_is_variationally_monotone() -> None:
    hamiltonian = np.diag((-1.0, 0.5, 2.0))
    points = tuple(
        full_reorthogonalized_ritz_flow(
            hamiltonian, np.asarray([1.0, 2.0, 3.0]), 2
        )
    )
    assert len(points) == 3
    assert all(a.energy >= b.energy - 1.0e-14 for a, b in zip(points, points[1:]))
    assert points[-1].energy == pytest.approx(-1.0, abs=1.0e-12)


def test_scalar_moments_separate_mean_and_connected_part() -> None:
    state = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    operator = np.diag((-1.0, 1.0))
    moments = scalar_moments(state, operator, normalization=2.0)
    assert moments["mean_real"] == pytest.approx(0.0)
    assert moments["raw_second"] == pytest.approx(0.25)
    assert moments["connected_second"] == pytest.approx(0.25)
