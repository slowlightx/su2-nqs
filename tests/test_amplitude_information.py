"""Exact tests for basis-dependent Born-amplitude information diagnostics."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from su2_nqs.benchmarks.amplitude_information import (
    amplitude_information_profile,
    compare_to_kinematic_reference,
    conditional_kl_to_reference,
    conditioned_cut_amplitude_information,
    cut_amplitude_information,
)


def test_independent_uniform_bits_have_zero_cut_information() -> None:
    sequences = tuple(product((0, 1), repeat=4))
    probabilities = np.full(len(sequences), 1.0 / len(sequences))
    result = amplitude_information_profile(sequences, probabilities)
    assert result["middle_cmi_bits"] == pytest.approx(0.0, abs=1.0e-14)
    assert result["uniform_support_KL_bits"] == pytest.approx(0.0)
    assert all(row["numerical_rank"] == 1 for row in result["cuts"])


def test_copy_distribution_has_one_bit_of_middle_information() -> None:
    sequences = ((0, 0), (1, 1))
    result = cut_amplitude_information(sequences, (0.5, 0.5), cut=1)
    assert result["cmi_bits"] == pytest.approx(1.0)
    assert result["numerical_rank"] == 2
    assert result["effective_singular_rank"] == pytest.approx(2.0)


def test_kinematic_comparison_is_explicitly_signed() -> None:
    sequences = ((0, 0), (0, 1), (1, 0), (1, 1))
    target = amplitude_information_profile(
        sequences, (0.5, 0.0, 0.0, 0.5)
    )
    reference = amplitude_information_profile(sequences, (0.25,) * 4)
    comparison = compare_to_kinematic_reference(target, reference)
    assert comparison["middle_cmi_excess_bits"] == pytest.approx(1.0)
    assert comparison["uniform_support_KL_bits"] == pytest.approx(0.0)


def test_conditional_kl_obeys_chain_rule_and_mask_interpretation() -> None:
    sequences = ((0, 0), (0, 1), (1, 0), (1, 1))
    result = conditional_kl_to_reference(
        sequences,
        (0.45, 0.05, 0.05, 0.45),
        sequences,
        (0.25, 0.25, 0.25, 0.25),
        cut=1,
    )
    assert result["prefix_KL_bits"] == pytest.approx(0.0, abs=1.0e-14)
    assert result["conditional_future_KL_bits"] == pytest.approx(
        result["total_KL_bits"]
    )
    assert result["chain_rule_error"] < 1.0e-14


def test_conditional_kl_reference_must_cover_target_support() -> None:
    with pytest.raises(ValueError, match="positive weight"):
        conditional_kl_to_reference(
            ((0, 0), (1, 1)),
            (0.5, 0.5),
            ((0, 0), (1, 1)),
            (1.0, 0.0),
            cut=1,
        )


def test_conserved_boundary_resolves_kinematic_mutual_information() -> None:
    sequences = tuple(row for row in product((0, 1), repeat=4) if sum(row) == 2)
    probabilities = np.full(len(sequences), 1.0 / len(sequences))
    labels = tuple(sum(row[:2]) for row in sequences)
    result = conditioned_cut_amplitude_information(
        sequences, probabilities, labels, cut=2
    )
    assert result["raw_cmi_bits"] == pytest.approx(result["boundary_entropy_bits"])
    assert result["conditional_cmi_bits"] == pytest.approx(0.0, abs=1.0e-14)
    assert result["decomposition_error"] < 1.0e-14
    assert result["boundary_is_prefix_deterministic"]
    assert result["boundary_is_suffix_deterministic"]


def test_wrong_boundary_label_fails_suffix_determinism_and_closure() -> None:
    sequences = tuple(product((0, 1), repeat=4))
    probabilities = np.full(len(sequences), 1.0 / len(sequences))
    labels = tuple(row[0] for row in sequences)
    result = conditioned_cut_amplitude_information(
        sequences, probabilities, labels, cut=2
    )
    assert result["boundary_is_prefix_deterministic"]
    assert not result["boundary_is_suffix_deterministic"]
    assert result["decomposition_error"] > 0.5


def test_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="unique"):
        cut_amplitude_information(((0, 0), (0, 0)), (0.5, 0.5), cut=1)
    with pytest.raises(ValueError, match="normalized"):
        cut_amplitude_information(((0, 0), (1, 1)), (0.2, 0.2), cut=1)
    with pytest.raises(ValueError, match="leave non-empty"):
        cut_amplitude_information(((0, 0), (1, 1)), (0.5, 0.5), cut=0)
