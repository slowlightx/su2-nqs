import numpy as np
import pytest

from su2_nqs.benchmarks.fusion_action_spectrum import (
    compare_fusion_action_cut_spectra,
    fusion_action_cut_matrix,
    fusion_action_cut_spectrum,
)


def _binary_actions(length: int) -> np.ndarray:
    return np.asarray(
        [
            [(state >> (length - position - 1)) & 1 for position in range(length)]
            for state in range(1 << length)
        ],
        dtype=np.int32,
    )


def test_product_amplitudes_have_rank_one_at_every_action_cut() -> None:
    actions = _binary_actions(4)
    local = (
        np.asarray([1.0, 2.0j]),
        np.asarray([0.7, -0.4]),
        np.asarray([1.2j, 0.9]),
        np.asarray([-0.8, 0.3j]),
    )
    values = np.asarray(
        [np.prod([local[position][action] for position, action in enumerate(row)]) for row in actions]
    )
    for cut in range(5):
        spectrum = fusion_action_cut_spectrum(values, actions, cut)
        assert spectrum["numeric_rank"] == 1
        assert spectrum["effective_rank"] == pytest.approx(1.0, abs=2.0e-14)
        assert spectrum["entropy_bits"] == pytest.approx(0.0, abs=2.0e-14)
        assert spectrum["r90"] == spectrum["r99"] == spectrum["r999"] == 1


def test_known_bell_spectrum_and_relative_predictive_growth() -> None:
    actions = _binary_actions(2)
    product = np.ones(4)
    bell = np.asarray([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
    spectrum = fusion_action_cut_spectrum(bell, actions, 1)
    assert spectrum["numeric_rank"] == 2
    assert spectrum["entropy_bits"] == pytest.approx(1.0)
    assert spectrum["effective_rank"] == pytest.approx(2.0)
    assert spectrum["effective_schmidt_rank"] == pytest.approx(2.0)
    assert spectrum["participation_rank"] == pytest.approx(2.0)
    assert spectrum["r90"] == spectrum["r99"] == spectrum["r999"] == 2

    rows = compare_fusion_action_cut_spectra(
        {"base": product, "bell": bell}, actions, base_name="base"
    )
    middle = rows[1]["spectra"]["bell"]["relative_to_base"]
    assert middle["numeric_rank_ratio"] == 2.0
    assert middle["effective_rank_ratio"] == pytest.approx(2.0)
    assert middle["entropy_bits_delta"] == pytest.approx(1.0)


def test_invalid_prefix_suffix_combinations_are_explicit_zeros() -> None:
    actions = np.asarray(((0, 0), (0, 1), (1, 0)), dtype=np.int32)
    values = np.asarray((2.0, 3.0, 5.0))
    cut = fusion_action_cut_matrix(values, actions, 1)
    np.testing.assert_array_equal(cut.values, np.asarray(((2.0, 3.0), (5.0, 0.0))))
    assert cut.prefix_keys == ((0,), (1,))
    assert cut.suffix_keys == ((0,), (1,))
    assert cut.invalid_combination_count == 1
    assert cut.invalid_combination_fraction == 0.25


def test_capture_ranks_follow_squared_singular_weight() -> None:
    actions = np.asarray(((0, 0), (1, 1), (2, 2)), dtype=np.int32)
    values = np.sqrt(np.asarray((0.80, 0.15, 0.05)))
    spectrum = fusion_action_cut_spectrum(values, actions, 1)
    assert spectrum["numeric_rank"] == 3
    assert spectrum["r90"] == 2
    assert spectrum["r99"] == 3
    assert spectrum["r999"] == 3
    expected_entropy = -np.sum(np.asarray((0.8, 0.15, 0.05)) * np.log2((0.8, 0.15, 0.05)))
    assert spectrum["entropy_bits"] == pytest.approx(expected_entropy)
    assert sum(spectrum["schmidt_probabilities"]) == pytest.approx(1.0)


def test_spectrum_is_scale_invariant_and_validates_unique_sequences() -> None:
    actions = _binary_actions(3)
    values = np.arange(1.0, 9.0) + 1.0j * np.arange(8.0)
    first = fusion_action_cut_spectrum(values, actions, 2)
    second = fusion_action_cut_spectrum((-3.0 + 4.0j) * values, actions, 2)
    np.testing.assert_allclose(
        first["normalized_singular_values"],
        second["normalized_singular_values"],
        atol=2.0e-15,
    )
    duplicate = actions.copy()
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="uniquely"):
        fusion_action_cut_spectrum(values, duplicate, 1)
    with pytest.raises(ValueError, match="positive"):
        fusion_action_cut_spectrum(np.zeros(len(actions)), actions, 1)
