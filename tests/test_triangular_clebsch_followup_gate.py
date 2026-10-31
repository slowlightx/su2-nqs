"""Continuation cannot mistake ordinary reweighted old energies for H(new)."""
from copy import deepcopy

from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_followup_gate import gate_failures


def records():
    h=[dict(accepted=True,FS_sine=.02,ESS_fraction=.99,energy_before_update_per_site=-.4,
        variance_before_update_per_site=.2,top_one_percent_residual_weight=.2) for _ in range(2)]
    replay=dict(checkpoint_rng_sampler_replay_exact=True,checkpoint_optimizer_exact=True)
    born=dict(energy_per_site=-.39,energy_chain_SE=.01,variance_per_site=.21)
    correlated=dict(energy_per_site=-.41,energy_chain_SE=.01,weight_ESS_fraction=.98)
    return h,replay,born,correlated


def test_gate_does_not_require_two_noisy_updates_to_lower_energy():
    assert gate_failures(*records())==[]


def test_gate_rejects_step_tail_support_replay_and_order_of_magnitude_failures():
    for modify in (
        lambda h,r,b,c:h.clear(),
        lambda h,r,b,c:[x.update(accepted=False) for x in h],
        lambda h,r,b,c:h[0].update(FS_sine=.2),
        lambda h,r,b,c:h[0].update(ESS_fraction=.5),
        lambda h,r,b,c:h[0].update(top_one_percent_residual_weight=.95),
        lambda h,r,b,c:r.update(checkpoint_optimizer_exact=False),
        lambda h,r,b,c:b.update(energy_per_site=-4.),
        lambda h,r,b,c:b.update(variance_per_site=30.),
        lambda h,r,b,c:c.update(weight_ESS_fraction=.02),
    ):
        args=deepcopy(records());modify(*args)
        assert gate_failures(*args)
