import numpy as np

from su2_nqs.benchmarks.finite_audit_statistics import paired_chain_summary, residual_tail_diagnostics
from su2_nqs.benchmarks.finite_scalar_dressing import finite_step_moments


def test_tail_contributions_sum_to_untrimmed_physical_moments():
    rng=np.random.default_rng(731)
    p=rng.normal(size=24)+1j*rng.normal(size=24);p[0]=0
    d=rng.normal(size=24)+1j*rng.normal(size=24)
    matrix=rng.normal(size=(24,24));matrix=(matrix+matrix.T)/2
    data=dict(psi=p,direction=d,hpsi=matrix@p,hdirection=matrix@d,proposal_squared=abs(p)**2+.1)
    m=finite_step_moments(*data.values(),.5)
    thresholds=np.quantile(abs(data['hpsi'])**2/data['proposal_squared'],[.9,.99])
    tail=residual_tail_diagnostics(data,.5,reference_energy=0.,thresholds=thresholds)
    assert sum(r['count'] for r in tail['strata'])==24
    np.testing.assert_allclose(sum(r['energy_change_contribution'] for r in tail['strata']),m['energy_change_total'],atol=2e-13)
    np.testing.assert_allclose(sum(r['variance_change_contribution'] for r in tail['strata']),m['variance_change_total'],atol=2e-13)
    summary=paired_chain_summary(data,np.tile(np.arange(6),4),.5,replicates=100,seed=34)
    assert summary['independent_chain_count']==6
    assert summary['paired_chain_bootstrap_se']['energy_change_total']>0
    zero=paired_chain_summary(data,np.tile(np.arange(6),4),0.,replicates=100,seed=34)
    assert zero['energy_change_total']==zero['variance_change_total']==0
    assert zero['paired_chain_bootstrap_se']['energy_change_total']==0
