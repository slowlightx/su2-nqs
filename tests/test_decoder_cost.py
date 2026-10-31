"""Small physical action oracles for the cost-only N36 measurement path."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin import _prepare_case
from projects.clebsch_tree_nqs.workflows.spin_decoder_cost import physical_config,amplitude_functions,benchmark,batched_host_amplitudes
from projects.clebsch_tree_nqs.workflows.spin_vap_decoder import init_decoder
from su2_nqs.benchmarks.spatial_vit_sampling import make_full_heisenberg_ratios


def test_complete_A1_and_full_H_ratios_against_independent_small_physical_matrices():
    data=_prepare_case(dict(geometry='square',rows=2,columns=4,j2_over_j1=.5,exact_energy_anchor=0),'fixed_m0')
    config=physical_config(2,4,2);params=init_decoder(198,config)
    funcs=amplitude_functions(config,data['groups']['gamma_a1'])
    basis=jnp.asarray(data['basis'])
    raw=np.asarray(jax.jit(jax.vmap(funcs['raw'],in_axes=(None,0)))(params,basis))
    projected=np.asarray(jax.jit(jax.vmap(funcs['a1'],in_axes=(None,0)))(params,basis))
    np.testing.assert_allclose(projected,np.mean(raw[data['tables']['gamma_a1']],axis=0),atol=3e-16,rtol=2e-13)
    for name,vector in [('raw',raw),('a1',projected)]:
        assert np.linalg.norm(data['spin_squared']@vector)/np.linalg.norm(vector)<2e-12
        selected=np.argsort(abs(vector))[-8:]
        ratios=make_full_heisenberg_ratios(lambda p,x:jnp.log(funcs[name](p,x)),data['bonds'])
        measured=np.asarray(ratios(params,basis[selected]))
        expected=(data['hamiltonian']@vector)[selected]/vector[selected]
        np.testing.assert_allclose(measured,expected,atol=2e-12,rtol=2e-12)


def test_timing_blocks_replay_and_host_chunks_do_not_change_amplitudes():
    fn=lambda p,x:jnp.sum(x,axis=1)*p['scale']
    p={'scale':jnp.asarray(2.)};states=np.arange(24,dtype=np.int32).reshape(8,3)
    report,executable,values=benchmark(fn,p,states,3)
    assert report['finite'] and report['replay_max']==0
    assert len(report['execution_seconds'])==3 and min(report['execution_seconds'])>0
    assert report['peak_rss_after_bytes']>=report['peak_rss_before_bytes']>0
    got=batched_host_amplitudes(executable,p,states[:3],batch=8)
    np.testing.assert_array_equal(got,values[:3])


def test_nine_cell_layout_preserves_all_doubled_spin_channels_without_N16_transfer():
    config=physical_config(6,6,2)
    assert config.n_cells==9 and config.n_sites==36
    assert config.maximum_intermediate_q is None and config.target_q==config.target_m2==0
    assert config.feature_dim==2 and config.projector is None
    with pytest.raises(TypeError): physical_config(6.,6,2)
    with pytest.raises(ValueError): physical_config(3,6,2)
