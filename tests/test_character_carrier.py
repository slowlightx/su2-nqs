import numpy as np
import pytest

from su2_nqs.benchmarks.character_carrier import (
    build_cyclic_character_carrier,
    compress_character_operator,
    cyclic_character_projector,
    lift_normalized_character_state,
)


def _cyclic_shift(period: int) -> np.ndarray:
    shift = np.zeros((period, period), dtype=np.complex128)
    for column in range(period):
        shift[(column + 1) % period, column] = 1.0
    return shift


def test_complex_character_convention_and_all_k_completeness() -> None:
    period = 5
    translation = np.kron(_cyclic_shift(period), np.eye(2))
    projectors = []
    for momentum in range(period):
        projector, eigenvalue, audit = cyclic_character_projector(
            translation, period, momentum
        )
        projectors.append(projector)
        assert audit["projector_idempotency_error"] < 2.0e-15
        np.testing.assert_allclose(
            translation @ projector,
            eigenvalue * projector,
            atol=2.0e-15,
        )
    np.testing.assert_allclose(
        sum(projectors), np.eye(2 * period), atol=2.0e-15
    )
    # k=1 is deliberately complex and fixes the sign convention.
    assert abs(projectors[1][2, 0] - np.exp(2.0j * np.pi / period) / period) < 2.0e-15


def test_projected_column_carrier_is_deterministic_and_complete() -> None:
    period = 4
    translation = np.kron(_cyclic_shift(period), np.eye(3))
    first = build_cyclic_character_carrier(translation, period, 1)
    second = build_cyclic_character_carrier(translation, period, 1)

    assert first.dimension == 3
    assert first.pivot_indices == second.pivot_indices
    np.testing.assert_array_equal(first.columns, second.columns)
    np.testing.assert_allclose(
        first.columns.conj().T @ first.columns, np.eye(3), atol=2.0e-14
    )
    np.testing.assert_allclose(
        first.columns @ first.columns.conj().T,
        first.projector,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        translation @ first.columns,
        first.character_eigenvalue * first.columns,
        atol=2.0e-14,
    )


def test_operator_compression_is_independently_spectral_and_covariant() -> None:
    period = 5
    translation = np.kron(_cyclic_shift(period), np.eye(2))
    carrier = build_cyclic_character_carrier(translation, period, 2)
    generator = np.random.default_rng(1701)
    raw = generator.normal(size=translation.shape) + 1.0j * generator.normal(
        size=translation.shape
    )
    raw = 0.5 * (raw + raw.conj().T)
    operator = np.zeros_like(raw)
    power = np.eye(len(raw), dtype=np.complex128)
    for _ in range(period):
        operator += power @ raw @ power.conj().T / period
        power = power @ translation

    compressed = compress_character_operator(operator, translation, carrier)
    assert compressed.audit["operator_translation_commutator_error"] < 2.0e-14
    assert compressed.audit["carrier_invariant_action_error"] < 2.0e-14
    assert compressed.audit["projected_operator_reconstruction_error"] < 2.0e-14
    assert compressed.audit["independent_projector_spectrum_error"] < 2.0e-14

    coefficients = np.asarray((1.0, 1.0j), dtype=np.complex128) / np.sqrt(2.0)
    state = lift_normalized_character_state(coefficients, carrier)
    np.testing.assert_allclose(np.vdot(state, state), 1.0, atol=2.0e-14)
    np.testing.assert_allclose(
        translation @ state,
        carrier.character_eigenvalue * state,
        atol=2.0e-14,
    )


def test_carrier_rejects_invalid_representation_and_coefficients() -> None:
    invalid = np.asarray(((1.0, 0.2), (0.0, 1.0)), dtype=np.complex128)
    with pytest.raises(ValueError, match="unitary"):
        build_cyclic_character_carrier(invalid, 2, 0)

    carrier = build_cyclic_character_carrier(_cyclic_shift(4), 4, 0)
    with pytest.raises(ValueError, match="normalized"):
        lift_normalized_character_state(np.asarray((2.0 + 0.0j,)), carrier)


def test_compression_rejects_nonhermitian_operator() -> None:
    translation = _cyclic_shift(3)
    carrier = build_cyclic_character_carrier(translation, 3, 1)
    bad = np.asarray(
        ((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        dtype=np.complex128,
    )
    with pytest.raises(ValueError, match="Hermitian"):
        compress_character_operator(bad, translation, carrier)
