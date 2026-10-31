"""Fixed-cache probes and genuine transverse joint-W first-order directions."""
import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.results.triangle_capacity_20261017.diagnose_saved_states import (
    sign_comparison,transverse_joint_direction,cached_energy_slope,
)
from su2_nqs.models.triangular_arity_clebsch import init_arity_pair,six_edge_unfoldings
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig,init_mixed_clebsch_family,mixed_clebsch_bank,make_mixed_clebsch_amplitude,
)
from su2_nqs.operators.triangular_a1_scalar import _cycle_configuration

jax.config.update('jax_enable_x64',True)


def test_joint_normal_is_not_a_dead_first_order_multilinear_channel():
    cfg=MixedClebschConfig(multiplicity_cap=9,maximum_q=3)
    source=init_mixed_clebsch_family(jax.random.PRNGKey(1710),cfg)
    pair,_=init_arity_pair(mixed_clebsch_bank(source['ternary_c3'],cfg,'ternary_c3'),cfg,8)
    bank=pair['ternary_joint']
    direction,note=transverse_joint_direction(bank,cfg,0,1711)
    assert note['left_tangent_orthogonality']<2e-13
    assert note['right_tangent_orthogonality']<2e-13
    assert note['fixed_CG_roundtrip_max_absolute_error']<2e-13
    assert np.linalg.matrix_rank(six_edge_unfoldings(jax.tree.map(lambda a,b:a+b,bank,direction),cfg)[0][2],tol=1e-10)==9
    raw=make_mixed_clebsch_amplitude(cfg)
    rng=np.random.default_rng(1712)
    rows=jnp.asarray([rng.permutation([0]*18+[1]*18) for _ in range(12)])
    evaluate=jax.jit(jax.vmap(raw,in_axes=(None,0)))
    psi=evaluate(bank,rows)
    tangent=jax.jit(lambda b,v:jax.jvp(lambda x:evaluate(x,rows),(b,),(v,))[1])(bank,direction)
    # Only ONE local bank changes: this independent variation is exactly
    # linear, not a small finite-difference approximation or new train step.
    direct=evaluate(jax.tree.map(lambda a,b:a+b,bank,direction),rows)-psi
    np.testing.assert_allclose(tangent,direct,rtol=2e-10,atol=2e-17)
    # Low maximum_q raw trees have genuine block-magnetization nodes;
    # the GPU diagnostic uses saved Born draws, not these uniform rows.
    supported=np.asarray(psi)!=0
    assert supported.sum()>=4
    score=np.asarray(tangent)[supported]/np.asarray(psi)[supported]
    assert np.std(score)>1e-8


def test_frozen_R4_is_hermitian_not_a_diagonal_configuration_frequency():
    n=4;cycle=(0,1,2,3);matrix=np.zeros((1<<n,1<<n))
    for old in range(1<<n):
        bits=tuple((old>>(n-i-1))&1 for i in range(n))
        for order in (cycle,tuple(reversed(cycle))):
            new=_cycle_configuration(bits,order)
            packed=sum(bit<<(n-i-1) for i,bit in enumerate(new))
            matrix[packed,old]+=.5
    np.testing.assert_array_equal(matrix,matrix.T)
    rng=np.random.default_rng(1713);psi=rng.normal(size=1<<n);psi/=np.linalg.norm(psi)
    local=(matrix@psi)/psi
    np.testing.assert_allclose(np.dot(psi**2,local),psi@matrix@psi,atol=2e-15)
    assert np.max(abs(matrix-np.diag(np.diag(matrix))))==1.
    # Alternating bits have identical clockwise/counterclockwise images,
    # so the two half weights COMBINE to one, not two duplicate entries.
    assert matrix[0b1010,0b0101]==1.


def test_sign_partitions_do_not_confuse_global_gauge_with_new_nodes():
    a=np.array([1.,-2.,3.,-4.,1.,-2.,3.,-4.]);labels=np.repeat([0,1],4)
    assert sign_comparison(a,-a,labels)['opposite_sign']['mean']==0
    b=-a.copy();b[0]=a[0];b[5]=0.
    result=sign_comparison(a,b,labels)
    assert result['opposite_sign']['mean']==1/8
    assert result['other_state_node_mass']['mean']==1/8
    assert 'directional_importance_overlap' not in result


def test_cached_slope_keeps_complex_conjugation_and_whole_chain_covariance():
    score=np.array([1+1j,-1+2j,2-1j,-2-2j,1-2j,2+1j,-1-1j,-2+2j])
    energy=np.array([-1+2j,2-1j,1+1j,-2-2j,-1+1j,1-2j,2+2j,-2-1j])
    labels=np.tile(np.arange(4),2)
    result=cached_energy_slope(score,energy,labels)
    expected=2*np.real(np.mean(np.conj(score-score.mean())*(energy-energy.mean())))/36
    np.testing.assert_allclose(result['energy_per_site_directional_derivative'],expected,atol=1e-16)
    assert result['chains']==4 and result['records']==8
    assert result['whole_chain_jackknife_SE']>0
    assert not np.isclose(expected,2*np.real(np.mean((score-score.mean())*(energy-energy.mean())))/36)


def test_default_SR_Adam_matches_preexisting_frozen_runtime_bitwise():
    # The previous production snapshot is evidence, not overwritten to make
    # a new shared optimizer pass. Default options must preserve all values.
    import tarfile
    import types
    from pathlib import Path
    from su2_nqs.vmc.physical_sr_adam import make_physical_sr_adam
    from su2_nqs.vmc.adam import _manual_adam_init
    root=Path(__file__).resolve().parents[1]
    archive=root/'projects/clebsch_tree_nqs/results/triangle_arity_followup_20261016/source_snapshot.tar.gz'
    with tarfile.open(archive) as data:
        source=data.extractfile('su2_nqs/vmc/physical_sr_adam.py').read()
    legacy=types.ModuleType('su2_nqs.vmc.frozen_triangle_SR_Adam')
    legacy.__package__='su2_nqs.vmc'
    exec(compile(source,'frozen_triangle_SR_Adam.py','exec'),legacy.__dict__)
    p=jnp.asarray([.17,-.12,.08],jnp.float64);state=_manual_adam_init(p)
    rows=jnp.asarray([[0,0],[0,1],[1,0],[1,1]]*4)
    energies=jnp.asarray([-.5,.2,-.1,.3]*4)
    def amplitude(theta,row):
        a,b=2*row[0]-1,2*row[1]-1
        return jnp.exp(theta@jnp.asarray([a,b,a*b]))
    previous=legacy.make_physical_sr_adam(amplitude,p,damping=.05,score_chunk=2,maximum_fs_sine=.025)
    current=make_physical_sr_adam(amplitude,p,damping=.05,score_chunk=2,maximum_fs_sine=.025)
    for warm in (True,True,False,False):
        before=previous(p,state,rows,energies,1e-4,warm=warm)
        after=current(p,state,rows,energies,1e-4,warm=warm)
        for a,b in zip(jax.tree.leaves(before[:2]),jax.tree.leaves(after[:2]),strict=True):
            np.testing.assert_array_equal(a,b)
        for key in before[2]:np.testing.assert_array_equal(before[2][key],after[2][key])
        p,state=before[:2]
