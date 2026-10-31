import jax
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_reproduction import run
from su2_nqs.vmc.checkpoint import load_checkpoint


def test_exact_training_resumes_parameters_adam_and_rng(tmp_path):
    spec=dict(rows=2,columns=2,geometry='square',multiplicity_cap=2,maximum_q=2,
              j2_over_j1=0.5,expectation='exact_full_support',seed=30,calibration_seed=31,
              steps=3,sr_warm_steps=1,sr_samples=8,sr_damping=0.01,
              sr_adam_learning_rate=0.005,adam_learning_rate=0.001,gradient_clip=0.001,
              milestones=[1,2,3],exact_energy_anchor=-2.)
    continuous=run(spec,tmp_path/'a',120)
    paused=run(spec,tmp_path/'b',120,until_step=2)
    assert paused['completed_steps'] == 2
    resumed=run(spec,tmp_path/'b',120)
    assert resumed['completed_steps'] == continuous['completed_steps'] == 3
    a=load_checkpoint(tmp_path/'a/checkpoint.pkl'); b=load_checkpoint(tmp_path/'b/checkpoint.pkl')
    for field in ('params','optimizer_state'):
        for x,y in zip(jax.tree.leaves(a[field]),jax.tree.leaves(b[field]),strict=True):
            np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(a['metadata']['rng'],b['metadata']['rng'])
    assert continuous['new_eigensolve'] is False
