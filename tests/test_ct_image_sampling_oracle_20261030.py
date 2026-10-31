"""Exact pair quadrature, including phase cancellations and exact zeros."""
import numpy as np
import pytest
from projects.clebsch_tree_nqs.explorations.ct_image_sampling_oracle_20261030 import norm_pair_oracle


def test_pair_variance_formula_matches_complete_complex_quadrature():
    a=np.array([1.+.2j,-.4+.1j,0.,.3-.2j]); eps=.05
    q=(1-eps)*abs(a)/abs(a).sum()+eps/len(a)
    law=q[:,None]*q[None,:]; z=a.conj()[:,None]*a[None,:]/law
    mean=np.sum(law*z); var=np.sum(law*abs(z-mean)**2)/abs(mean)**2
    got=norm_pair_oracle(a)['proposals']['magnitude_oracle_with_5percent_uniform']
    assert got['pair_relative_variance']==pytest.approx(var,abs=3e-14)
    np.testing.assert_allclose(mean,abs(a.sum())**2,atol=3e-15)
    assert got['minimum_positive_q']>0


def test_projected_nodes_and_raw_zeros_are_never_repaired():
    assert norm_pair_oracle(np.zeros(4))['status']=='zero_raw_amplitudes'
    assert norm_pair_oracle(np.array([1.,-1.,0.,0.]))['status']=='exact_projected_node'
    assert norm_pair_oracle(np.array([1.,0.,0.,0.]))['zero_images']==3
