"""Exact two-state physical-distance oracles, independent of any optimizer."""
import json

import numpy as np
import pytest

from su2_nqs.vmc.matched_state_step import matched_state_step


def _linear(rate=1.):
    calls = []
    def candidate(alpha):
        calls.append(alpha)
        return np.array([1., rate*alpha], dtype=np.complex128)
    return candidate, calls


def test_two_unequal_directions_match_measured_FS_not_parameter_length_with_fixed_work():
    old = np.array([1., 0.], complex)
    target = .01
    answers = []
    for rate in (1., 7.):
        candidate, calls = _linear(rate)
        answer = matched_state_step(candidate, old, np.ones(2), target, 1., 18,
                                    relative_tolerance=.004)
        assert answer.matched and answer.target_bracketed and answer.status == 'matched'
        assert answer.evaluation_count == len(calls) == 19
        assert len(answer.records) == 19 and calls[0] == 1.
        assert answer.distance['fs'] <= target
        assert answer.radius_shortfall <= .004*target
        assert answer.distance['fs'] == pytest.approx(np.arctan(rate*answer.alpha), abs=3e-16)
        np.testing.assert_array_equal(answer.amplitudes, np.array([1., rate*answer.alpha]))
        # Selection is the closest UNDER-target point across all evaluations.
        eligible = [row['distance']['fs'] for row in answer.records if row['valid'] and row['within_radius']]
        assert answer.distance['fs'] == max(eligible)
        json.dumps(answer.as_dict(), allow_nan=False)
        answers.append(answer)
    assert answers[0].alpha/answers[1].alpha == pytest.approx(7., rel=.004)


def test_unbracketed_target_is_flagged_not_silently_called_matched():
    candidate, calls = _linear(.001)
    answer = matched_state_step(candidate, np.array([1., 0.]), np.ones(2), .01, 1., 9)
    assert not answer.matched and not answer.target_bracketed
    assert answer.status == 'target_not_bracketed_by_evaluated_points'
    assert answer.alpha == 1.
    assert answer.distance['fs'] == pytest.approx(np.arctan(.001), abs=3e-16)
    assert len(calls) == answer.evaluation_count == 10


@pytest.mark.parametrize('gauge', [False, True])
def test_null_and_complex_gauge_directions_return_parent_with_same_evaluation_budget(gauge):
    old = np.array([1., 0., .2j], complex)
    q = np.array([.2, 2., 3.])
    calls = []
    def candidate(alpha):
        calls.append(alpha)
        scale = (1+alpha)*np.exp(.9j*alpha) if gauge else 1.
        return scale*old
    result = matched_state_step(candidate, old, q, .01, 1., 12)
    assert result.alpha == 0 and not result.matched
    assert result.status == 'zero_or_gauge_on_evaluated_points'
    assert result.distance['fs'] == 0
    np.testing.assert_array_equal(result.amplitudes, old)
    assert len(calls) == result.evaluation_count == 13
    assert max(record['distance']['fs'] for record in result.records) < 1e-14


def test_finite_quadrature_uses_full_support_and_allows_opening_parent_nodes():
    candidate, calls = _linear()
    q = np.array([1., 4.])
    result = matched_state_step(candidate, np.array([1., 0.]), q, .02, 1., 16)
    assert result.matched
    assert result.distance['opened_node_samples'] == 1
    # A tangent at the exact node has nonzero norm weighted by 1/q[1].
    assert result.distance['fs'] == pytest.approx(np.arctan(result.alpha/2), abs=3e-16)
    scaled, _ = _linear()
    other = matched_state_step(scaled, np.array([1., 0.]), 7.3*q, .02, 1., 16)
    assert result.alpha == other.alpha
    assert result.distance['fs'] == pytest.approx(other.distance['fs'], abs=3e-16)
    assert len(calls) == 17


def test_nonmonotone_path_can_find_an_interior_bracket_even_if_endpoint_is_parent():
    calls = []
    def candidate(alpha):
        calls.append(alpha)
        return np.array([1., .3*np.sin(np.pi*alpha)], complex)
    answer = matched_state_step(candidate, np.array([1., 0.]), np.ones(2), .05, 1., 16)
    assert answer.matched and answer.target_bracketed
    assert answer.records[0]['distance']['fs'] < 1e-14
    assert answer.distance['fs'] <= .05
    assert len(calls) == 17


def test_nonfinite_or_zero_proposals_shrink_bracket_but_keep_the_fixed_work_budget():
    calls = []
    def candidate(alpha):
        calls.append(alpha)
        if alpha > .5:
            return np.array([np.nan, 0.])
        if alpha == .5:
            return np.zeros(2)
        return np.array([1., alpha])
    result = matched_state_step(candidate, np.array([1., 0.]), np.ones(2), .1, 1., 14)
    assert result.matched
    assert not result.records[0]['valid'] and not result.records[1]['valid']
    assert result.records[0]['reason'] == 'nonfinite_candidate'
    assert result.records[1]['reason'] == 'zero_candidate_norm'
    assert len(calls) == 15
    json.dumps(result.as_dict(), allow_nan=False)


def test_underresolved_bisection_or_all_invalid_candidates_do_not_fake_matching():
    candidate, calls = _linear()
    result = matched_state_step(candidate, np.array([1., 0.]), np.ones(2), .01, 1., 2)
    assert not result.matched and result.target_bracketed
    assert result.alpha == 0.
    assert result.status == 'radius_below_target_after_fixed_budget'
    assert len(calls) == 3
    invalid_calls = []
    def invalid(alpha):
        invalid_calls.append(alpha)
        return np.zeros(2)
    failed = matched_state_step(invalid, np.array([1., 0.]), np.ones(2), .01, 1., 4)
    assert failed.status == 'no_finite_candidate_in_evaluations'
    assert not failed.matched and failed.alpha == 0
    assert len(invalid_calls) == failed.evaluation_count == 5


def test_invalid_contracts_raise_before_work_or_propagate_callback_errors():
    candidate, calls = _linear()
    with pytest.raises(ValueError):
        matched_state_step(candidate, np.array([1., 0.]), np.array([1., 0.]), .01, 1., 4)
    with pytest.raises(ValueError):
        matched_state_step(candidate, np.array([1., 0.]), np.ones(2), .01, 1., True)
    with pytest.raises(ValueError):
        matched_state_step(candidate, np.array([1., 0.]), np.ones(2), .01, 1., 4, relative_tolerance=1.)
    assert calls == []
    with pytest.raises(ValueError):
        matched_state_step(lambda _: np.ones(3), np.array([1., 0.]), np.ones(2), .01, 1., 4)
    def broken(_):
        raise RuntimeError('real model failure')
    with pytest.raises(RuntimeError, match='real model failure'):
        matched_state_step(broken, np.array([1., 0.]), np.ones(2), .01, 1., 4)
