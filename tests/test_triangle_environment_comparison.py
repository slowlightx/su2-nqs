"""Estimator checks independent of CT training and group tables."""
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_environment_comparison import (
    overlap_statistics, paired_overlap_difference,
)


def test_overlap_keeps_coherent_sign_and_ignores_global_scale():
    labels = np.tile(np.arange(4), 3)
    ratios = np.array([1., -1., 2., -2.]*3)
    a = overlap_statistics(ratios, labels)
    b = overlap_statistics(ratios*(2-3j), labels)
    assert a['sampled_squared_overlap'] == 0
    assert a['sampled_fs_sine'] == 1
    for key in ('sampled_squared_overlap', 'sampled_infidelity', 'reweighting_ess_fraction'):
        np.testing.assert_allclose(a[key], b[key], atol=2e-15)


def test_constant_physical_state_has_zero_distance_and_unit_ess():
    result = overlap_statistics(np.full(16, 2+3j), np.tile(np.arange(4), 4))
    assert abs(result['sampled_infidelity']) < 2e-15
    assert abs(result['reweighting_ess_fraction']-1) < 2e-15
    assert result['chain_jackknife_infidelity_SE'] < 2e-15


def test_block_jackknife_is_invariant_to_record_order():
    labels = np.tile(np.arange(8), 4)
    ratios = np.exp(.15*np.random.default_rng(9).normal(size=len(labels)))
    shuffled = np.random.default_rng(10).permutation(len(labels))
    a = overlap_statistics(ratios, labels)
    b = overlap_statistics(ratios[shuffled], labels[shuffled])
    np.testing.assert_allclose(a['chain_jackknife_infidelity_SE'],
                               b['chain_jackknife_infidelity_SE'], atol=2e-15)


def test_paired_error_keeps_exact_candidate_correlation_and_difference_sign():
    labels = np.tile(np.arange(8), 4)
    a = np.exp(.15*np.random.default_rng(19).normal(size=len(labels)))
    identical = paired_overlap_difference(a, a*2j, labels)
    np.testing.assert_allclose(identical['left_infidelity_minus_right_infidelity'], 0, atol=2e-15)
    assert identical['paired_chain_jackknife_SE'] < 2e-15
    improved = paired_overlap_difference(a, np.ones(len(a)), labels)
    assert improved['left_infidelity_minus_right_infidelity'] > 0
    np.testing.assert_allclose(improved['paired_chain_jackknife_SE'],
        overlap_statistics(a, labels)['chain_jackknife_infidelity_SE'], atol=2e-15)


@pytest.mark.parametrize('ratios,labels', [([0.]*4, range(4)), ([1., np.nan, 2., 3.], range(4)),
                                           ([1.]*4, [0, 0, 0, 1])])
def test_bad_overlap_inputs_fail_closed(ratios, labels):
    with pytest.raises(ValueError):
        overlap_statistics(np.array(ratios), np.array(labels))
