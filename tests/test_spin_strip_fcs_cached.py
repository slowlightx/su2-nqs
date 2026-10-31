import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_strip_fcs_cached import (
    prefix_chain_histograms, fcs_metrics,
)


def test_whole_chains_are_not_pooled_as_independent_snapshots():
    bits=np.array([[[0,0,1,1],[0,1,0,1]],[[0,0,1,1],[1,1,0,0]]])
    labels,hist=prefix_chain_histograms([bits,bits],2)
    np.testing.assert_array_equal(labels,[-2,0,2])
    np.testing.assert_allclose(hist,[[1,0,0],[0,.5,.5],[1,0,0],[0,.5,.5]],atol=0,rtol=0)
    values,report=fcs_metrics(labels,hist.mean(0),2)
    assert len(values)==4 and not report['strict_su2_verified']
    assert not report['schmidt_spectrum_measured'] and not report['qsl_specific']


def test_zero_variance_is_a_trivial_dimer_counterexample_not_qsl():
    _,report=fcs_metrics(np.array([-2,0,2]),np.array([0.,1.,0.]),2)
    assert report['variance_sz']==0 and report['fourth_cumulant_sz']==0
    assert report['magnetic_number_entropy_nats']==0 and report['pi_rotation_real']==1


@pytest.mark.parametrize('bits,cut',[(np.ones((2,1,4),int),2),(np.zeros((2,4),int),2),(np.zeros((2,1,4)),2),(np.zeros((2,1,4),int),True)])
def test_invalid_physical_inputs_fail_closed(bits,cut):
    with pytest.raises(ValueError):prefix_chain_histograms([bits],cut)
