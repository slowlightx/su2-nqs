import numpy as np
import pytest

from su2_nqs.benchmarks.conditional_response import (
    complex_martingale_decomposition,
    complex_martingale_intervention,
    complex_response_metrics,
    conditional_kl_decomposition,
)


def _binary_kl(q, p):
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sum(q * np.log2(q / p)))


def test_conditional_kl_obeys_both_chain_rules_for_product_distribution():
    actions = np.asarray(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.int32)
    p_first = np.asarray((0.75, 0.25))
    p_second = np.asarray((0.6, 0.4))
    q_first = np.asarray((0.4, 0.6))
    q_second = np.asarray((0.2, 0.8))
    reference = np.outer(p_first, p_second).ravel()
    target = np.outer(q_first, q_second).ravel()

    result = conditional_kl_decomposition(reference, target, actions)

    expected = _binary_kl(q_first, p_first) + _binary_kl(q_second, p_second)
    assert result["kl_target_reference_bits"] == pytest.approx(expected, abs=2e-15)
    assert result["chain_kl_bits"] == pytest.approx(expected, abs=2e-15)
    assert result["chain_rule_error"] < 2e-15
    assert result["entropy_chain_rule_error"] < 2e-15
    assert result["decisions"][0]["kl_contribution_bits"] == pytest.approx(
        _binary_kl(q_first, p_first), abs=2e-15
    )
    assert result["decisions"][1]["kl_contribution_bits"] == pytest.approx(
        _binary_kl(q_second, p_second), abs=2e-15
    )


def test_complex_response_separates_probability_and_phase_changes():
    reference = np.asarray((1.0, 1.0), dtype=np.complex128) / np.sqrt(2.0)
    target = np.asarray((1.0, -1.0), dtype=np.complex128) / np.sqrt(2.0)

    result = complex_response_metrics(reference, target)

    assert result["amplitude_fidelity_squared"] == pytest.approx(1.0, abs=2e-15)
    assert result["overlap_squared"] == pytest.approx(0.0, abs=2e-15)
    assert result["conditional_phase_coherence"] == pytest.approx(0.0, abs=2e-15)
    assert result["probability_kl_target_reference_bits"] == pytest.approx(
        0.0, abs=2e-15
    )
    assert result["born_reweighting_ess_fraction"] == pytest.approx(1.0, abs=2e-15)


def test_conditional_kl_rejects_nonunique_complete_action_strings():
    with pytest.raises(ValueError, match="identify states uniquely"):
        conditional_kl_decomposition(
            np.asarray((0.5, 0.5)),
            np.asarray((0.5, 0.5)),
            np.asarray(((0, 1), (0, 1))),
        )


def test_common_zero_support_does_not_destroy_reweighting_ess():
    result = complex_response_metrics(
        np.asarray((1.0, 0.0, 0.0)),
        np.asarray((1.0, 0.0, 0.0)),
    )
    assert result["target_probability_outside_reference_support"] == 0.0
    assert result["born_reweighting_ess_fraction"] == pytest.approx(1.0)


def test_true_target_support_leakage_is_infinite_kl_and_zero_ess():
    actions = np.asarray(((0,), (1,)), dtype=np.int32)
    decomposition = conditional_kl_decomposition(
        np.asarray((1.0, 0.0)), np.asarray((0.5, 0.5)), actions
    )
    response = complex_response_metrics(
        np.asarray((1.0, 0.0)), np.asarray((1.0, 1.0))
    )
    assert np.isinf(decomposition["kl_target_reference_bits"])
    assert np.isinf(decomposition["chain_kl_bits"])
    assert decomposition["chain_rule_error"] == 0.0
    assert response["target_probability_outside_reference_support"] == pytest.approx(0.5)
    assert response["born_reweighting_ess_fraction"] == 0.0


def test_complex_martingale_reconstructs_amplitude_and_phase_response():
    actions = np.asarray(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.int32)
    reference = np.asarray((1.0, 2.0j, -1.5, 0.7j), dtype=np.complex128)
    target = np.asarray((0.4j, -1.2, 0.3 + 0.8j, -0.9j), dtype=np.complex128)
    reference /= np.linalg.norm(reference)
    target /= np.linalg.norm(target)

    result = complex_martingale_decomposition(reference, target, actions)
    reconstructed = complex_martingale_intervention(reference, result, (0, 1))

    assert result["overlap"] == pytest.approx(np.vdot(reference, target), abs=3e-15)
    assert reconstructed == pytest.approx(target, abs=3e-15)
    assert result["maximum_state_reconstruction_error"] < 3e-15
    assert result["maximum_increment_orthogonality_error"] < 3e-15
    assert result["maximum_increment_mean_error"] < 3e-15
    assert result["normalization_sum_error"] < 3e-15

    probabilities = np.abs(reference) ** 2
    increments = result["increments"]
    conventional_gram = increments.conj() @ (probabilities[:, None] * increments.T)
    off_diagonal = conventional_gram - np.diag(np.diag(conventional_gram))
    assert np.max(np.abs(off_diagonal)) < 3e-15


def test_empty_complex_intervention_is_reference_up_to_global_phase():
    actions = np.asarray(((0,), (1,)), dtype=np.int32)
    reference = np.asarray((1.0, 2.0j), dtype=np.complex128)
    target = np.asarray((2.0 + 0.2j, -0.3j), dtype=np.complex128)
    result = complex_martingale_decomposition(reference, target, actions)

    empty = complex_martingale_intervention(reference, result, ())
    normalized_reference = reference / np.linalg.norm(reference)

    assert abs(np.vdot(normalized_reference, empty)) == pytest.approx(1.0, abs=2e-15)


def test_empty_intervention_is_defined_for_orthogonal_target_and_full_is_target():
    actions = np.asarray(((0,), (1,)), dtype=np.int32)
    reference = np.asarray((1.0, 1.0), dtype=np.complex128) / np.sqrt(2.0)
    target = np.asarray((1.0j, -1.0j), dtype=np.complex128) / np.sqrt(2.0)
    result = complex_martingale_decomposition(reference, target, actions)

    assert result["overlap"] == pytest.approx(0.0j, abs=2e-15)
    assert complex_martingale_intervention(reference, result, ()) == pytest.approx(
        reference, abs=2e-15
    )
    assert complex_martingale_intervention(reference, result, (0,)) == pytest.approx(
        target, abs=2e-15
    )


def test_common_zero_support_reconstructs_but_any_true_leakage_is_rejected():
    actions = np.asarray(((0, 0), (0, 1), (1, 0)), dtype=np.int32)
    reference = np.asarray((1.0, 2.0j, 0.0), dtype=np.complex128)
    target = np.asarray((0.2j, -0.7, 0.0), dtype=np.complex128)
    result = complex_martingale_decomposition(reference, target, actions)
    full = complex_martingale_intervention(reference, result, (0, 1))
    assert full == pytest.approx(target / np.linalg.norm(target), abs=3e-15)

    target[-1] = 1e-12
    with pytest.raises(ValueError, match="support must be contained"):
        complex_martingale_decomposition(reference, target, actions)


def test_complex_intervention_rejects_noninteger_decisions():
    actions = np.asarray(((0,), (1,)), dtype=np.int32)
    result = complex_martingale_decomposition(
        np.asarray((1.0, 1.0)), np.asarray((1.0, 2.0j)), actions
    )
    with pytest.raises(TypeError, match="must contain integers"):
        complex_martingale_intervention(np.asarray((1.0, 1.0)), result, (0.5,))


def test_complex_martingale_rejects_target_support_leakage():
    actions = np.asarray(((0,), (1,)), dtype=np.int32)
    with pytest.raises(ValueError, match="support must be contained"):
        complex_martingale_decomposition(
            np.asarray((1.0, 0.0)),
            np.asarray((1.0, 1.0)),
            actions,
        )
