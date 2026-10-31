"""Source weights and energy references of the sampled response Ritz readout."""
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_response_krylov_readout import summarize


def test_two_poles_are_referenced_to_parent_and_preserve_source_weights():
    ids = np.repeat(np.arange(8), 4)
    poles = np.array([.15, 1.6875]); weight = np.array([25/41, 16/41])
    values = {}
    for name in ('parent', 'K', 'minusK', 'M1', 'M2', 'M3'):
        for degree in range(4):
            moment = .3**degree if name == 'parent' else weight @ poles**degree
            values[name+'__m'+str(degree)] = np.full(len(ids), moment)
        values[name+'__m3_imag'] = np.zeros(len(ids))
    result = summarize(values, ids)
    assert result['independent_chains'] == 8
    for star in ('Kstar', 'Mstar'):
        row = result['response_stars'][star]['metrics']
        np.testing.assert_allclose(row['lowest_ritz_from_parent']['value'], -.15, atol=3e-14)
        np.testing.assert_allclose(row['upper_ritz_from_parent']['value'], 1.3875, atol=3e-14)
        np.testing.assert_allclose(row['lowest_source_weight']['value'], weight[0], atol=3e-14)
        # A negative approximate-parent difference is retained, not clipped.
        assert row['lowest_ritz_from_parent']['chain_SE'] < 3e-14
