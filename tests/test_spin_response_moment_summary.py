"""Nonlinear chain aggregation and approximate-parent residual accounting."""
import numpy as np
from projects.clebsch_tree_nqs.workflows.spin_response_moment_readout import summarize


def test_pooled_stars_keep_ratios_paired_and_width_decomposition():
    values = {}
    for name in ('parent','K','minusK','M1','M2','M3'):
        e = .1 if name == 'parent' else 1.1
        fields = dict(norm=1, first=e, first_imag=0, second=e*e+.2,
                      spin=0 if name=='parent' else 2, spin_defect=0,
                      commutator_first=e-.1, commutator_second=(e-.1)**2,
                      parent_residual_first=.1, parent_residual_second=.21,
                      commutator_parent_cross=2*.1*(e-.1))
        for field, value in fields.items():
            # Correlated fluctuations of norm and all raw moments cancel.
            values[name+'__'+field] = value*np.linspace(.8, 1.2, 8)
    result = summarize(values, np.arange(8), -20, 36)['metrics']
    assert abs(result['Kstar__centroid_from_parent']['value']-1)<1e-14
    assert result['Kstar__centroid_from_parent']['chain_SE']<1e-14
    assert abs(result['Kstar__width_squared']['value']-.2)<1e-14
    assert abs(result['Kstar__weighted_parent_residual_norm_squared']['value']-.2)<1e-14
    assert abs(result['Kstar__width_decomposition_error']['value'])<1e-14
    assert result['Kstar__outside_triplet_weight_upper_estimate']['value']==0
