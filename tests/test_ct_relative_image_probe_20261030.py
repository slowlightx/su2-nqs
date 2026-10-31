"""Independent exact one-sided group/H identities and conditional noise."""
import numpy as np
from projects.clebsch_tree_nqs.explorations.ct_relative_image_probe_20261030 import exact_identities,one_sided_stats


def test_exact_projector_H_H2_and_parameter_derivatives_with_negative_control():
    result=exact_identities()
    for case in result['cases']:
        assert case['P_idempotence_max']<2e-14
        assert case['H_commutator_max']<2e-14
        assert case['moment_max_error']<2e-14
        assert case['derivative_max_error']<2e-13
        assert case['quotient_finite_difference_max_error']<2e-8
        assert case['noncommuting_H_negative_control_discrepancy']>1e-6


def test_conditional_norm_moment_noise_matches_explicit_image_quadrature():
    a=np.array([[1.,-.3,.2],[.4,.1,-.2]])
    q=np.array([[.4,.3,.3],[.6,.1,.3]])
    stats=one_sided_stats(a,q,0)
    for i in range(2):
        values=a[i]/(3*q[i]*a[i,0]);mean=np.dot(q[i],values)
        variance=np.dot(q[i],abs(values-mean)**2)
        np.testing.assert_allclose(mean,stats['per_row_exact_mean_real'][i],atol=2e-15)
        np.testing.assert_allclose(variance,stats['per_row_conditional_variance'][i],atol=2e-15)
