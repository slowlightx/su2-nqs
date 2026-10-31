"""Exact small SU2 inclusion, native-axis migration, and weighted SR oracles."""
from itertools import product
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.explorations.ct12_live_w28_20261030 import (
    orthogonal_additions,expand_active_bank,lift_inherited_tree,projected_streaming_operators)
from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig,mixed_clebsch_layout
from su2_nqs.models.triangular_clebsch_active_layout import mixed_active_layout


def test_independent_eight_spin_completion_keeps_singlet_state_and_live_new_root_gate():
    rng=np.random.default_rng(26103071);frames={};wide={};coeff={}
    for q in (0,2,4):
        width=len(enumerate_paths(4,q));old=np.linalg.qr(rng.normal(size=(width,1)))[0]
        frames[q]=old;wide[q]=np.column_stack((old,orthogonal_additions(old,width-1,rng,scale=1.)))
        coeff[q]=np.zeros((width,width));coeff[q][0,0]=rng.normal()
    norm=np.sqrt(sum(np.vdot(v,v).real for v in coeff.values()));coeff={q:c/norm for q,c in coeff.items()}
    def physical(full):
        state=np.zeros((16,16));tangent=np.zeros((16,16))
        for q in (0,2,4):
            for m in range(-q,q+1,2):
                ul=coupled_basis_matrix(enumerate_paths(4,q),q,m).real
                ur=coupled_basis_matrix(enumerate_paths(4,q),q,-m).real
                f=wide[q] if full else frames[q]
                c=coeff[q] if full else coeff[q][:1,:1]
                state+=(-1)**((q-m)//2)/np.sqrt(q+1)*(ul@f@c@f.T@ur.T)
                if full and q==2:
                    d=np.zeros_like(c);d[1,0]=1
                    tangent+=(-1)**((q-m)//2)/np.sqrt(q+1)*(ul@f@d@f.T@ur.T)
        return state.ravel(),tangent.ravel()
    old,_=physical(False);new,tangent=physical(True)
    np.testing.assert_allclose(old,new,atol=2e-14)
    np.testing.assert_allclose(np.linalg.norm(new),1,atol=2e-14)
    assert np.linalg.norm(tangent)> .99
    singlet=coupled_basis_matrix(enumerate_paths(8,0),0,0)
    np.testing.assert_allclose(singlet@(singlet.T@new),new,atol=2e-14)
    np.testing.assert_allclose(singlet@(singlet.T@tangent),tangent,atol=2e-14)


def _small_bank():
    cfg=MixedClebschConfig(6,9,3);profile=({0:2,2:2},{0:2,2:2},{0:1});rng=np.random.default_rng(26103072)
    from su2_nqs.su2.triangle_blocks import triangle_block_tables,nine_site_output
    local=tuple(tuple(rng.normal(size=(triangle_block_tables()['nine'][q].shape[1],nine_site_output(q,9).width))*.1
                      for q in cfg.local_q) for _ in range(4))
    layout=mixed_active_layout(cfg,profile)
    top=tuple(tuple(rng.normal(size=g['shape'])*.1 for g in row['groups']) for row in layout)
    return cfg,profile,dict(local=local,top=top)


def test_actual_layout_inclusion_and_all_old_moments_exactly_migrate():
    from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank,make_feature_lut_amplitude
    cfg,profile,bank=_small_bank();newcfg=MixedClebschConfig(6,10,3)
    expanded,note=expand_active_bank(bank,cfg,profile,newcfg,seed=26103073)
    moved=lift_inherited_tree(bank,cfg,profile,expanded,newcfg)
    for old,new,embedded in zip(jax.tree.leaves(bank['local']),jax.tree.leaves(expanded['local']),jax.tree.leaves(moved['local']),strict=True):
        np.testing.assert_array_equal(new[:,:old.shape[1]],old)
        np.testing.assert_array_equal(embedded[:,:old.shape[1]],old)
        assert not np.any(embedded[:,old.shape[1]:])
    oldlayout=mixed_active_layout(cfg,profile);newlayout=mixed_clebsch_layout(newcfg)
    for oldrow,newrow,oldvalues,newvalues in zip(oldlayout,newlayout,bank['top'],moved['top'],strict=True):
        lookup={k:a for g,w in zip(oldrow['groups'],oldvalues,strict=True) for k,a in zip(g['keys'],w,strict=True)}
        for group,values in zip(newrow['groups'],newvalues,strict=True):
            for key,a in zip(group['keys'],values,strict=True):
                inherited=lookup[key];np.testing.assert_array_equal(a[tuple(slice(n) for n in inherited.shape)],inherited)
                outside=a.copy();outside[tuple(slice(n) for n in inherited.shape)]=0;assert not np.any(outside)
    rows=np.array([np.random.default_rng(26103074+i).permutation([0]*18+[1]*18) for i in range(4)])
    def evaluate(p,cfg,profile):
        raw=make_feature_lut_amplitude(cfg,active_top_widths=profile)
        return jax.jit(jax.vmap(raw,in_axes=(None,0)))(prepare_feature_lut_bank(p,cfg),jnp.asarray(rows))
    np.testing.assert_allclose(evaluate(bank,cfg,profile),evaluate(expanded,newcfg,None),atol=2e-18,rtol=3e-12)
    assert note['same_state_by_old_output_new_child_zero_closure']


def test_streamed_full_support_F_not_two_F_matches_existing_dense_SR_including_nodes():
    from su2_nqs.vmc.importance_amplitude_sr import solve_projected_block_sr
    from su2_nqs.vmc.matrix_free_lsqr import damped_lsqr
    rng=np.random.default_rng(26103075);jac=rng.normal(size=(7,9));theta=jnp.asarray(rng.normal(size=9))
    rows=jnp.arange(7)[:,None];offset=jac@np.asarray(theta);psi=offset.copy();psi[2]=0.
    shift=jnp.asarray(psi-offset)
    batch=lambda x,r:jnp.asarray(jac)[r[:,0]]@x+shift[r[:,0]]
    q=jnp.asarray(np.exp(rng.normal(size=7)));hp=jnp.asarray(rng.normal(size=7));psi=jnp.asarray(psi)
    forward,adjoint=projected_streaming_operators(batch,theta,rows,psi,q)
    norm=jnp.sum(psi**2/q);scale=1/jnp.sqrt(q*norm);u=psi*scale
    a=(np.eye(7)-np.outer(u,u))@(jac*np.asarray(scale)[:,None])
    v=rng.normal(size=9);np.testing.assert_allclose(adjoint(forward(jnp.asarray(v))),a.T@a@v,atol=2e-13)
    energy=jnp.sum(psi*hp/q)/norm;b=2*(hp-energy*psi)*scale;rhs=jnp.r_[b,jnp.zeros(7)]
    result=damped_lsqr(forward,adjoint,rhs,damping=.3,max_iterations=24,relative_tolerance=1e-11)
    dense,gradient,_=solve_projected_block_sr(jnp.asarray(jac),psi,hp,q,damping=.3,parameter_chunk_size=4)
    np.testing.assert_allclose(result.solution,dense,atol=2e-11,rtol=2e-11)
    np.testing.assert_allclose(adjoint(rhs),gradient,atol=2e-13)
    assert np.linalg.norm(np.asarray(jac)[2])>0 # genuine derivative support at a psi node
    assert float(result.relative_normal_residual)<1e-10


def test_new_only_physical_Gram_matching_matches_independent_raw_overlap_and_old_axes():
    from projects.clebsch_tree_nqs.explorations.ct12_w28_tangent_strength_20261030 import match_new_Gram_RMS
    from su2_nqs.models.triangular_clebsch_gauge import triangular_raw_overlap
    cfg,profile,bank=_small_bank();newcfg=MixedClebschConfig(6,10,3)
    wide,_=expand_active_bank(bank,cfg,profile,newcfg,seed=26103076)
    changed,notes=match_new_Gram_RMS(wide,newcfg,{1:9,3:9},profile)
    before=triangular_raw_overlap(wide,wide,newcfg);after=triangular_raw_overlap(changed,changed,newcfg)
    np.testing.assert_allclose(notes['root_raw_norm'],before,atol=2e-14,rtol=2e-12)
    np.testing.assert_allclose(after,before,atol=2e-14,rtol=2e-12)
    for a,b in zip(jax.tree.leaves(wide['local']),jax.tree.leaves(changed['local']),strict=True):
        np.testing.assert_array_equal(a[:,:9],b[:,:9])
    for row in notes['records']:
        if row['new_RMS_after'] is not None:
            np.testing.assert_allclose(row['new_RMS_after'],row['old_RMS'],atol=2e-14,rtol=2e-12)


def test_streamed_complex_amplitude_real_coordinate_adjoint_including_source_node():
    """Independent complex linear oracle: real-coordinate metric is Re(A†A)."""
    from su2_nqs.vmc.matrix_free_lsqr import damped_lsqr
    rng=np.random.default_rng(26103078)
    jac=rng.normal(size=(6,8))+1j*rng.normal(size=(6,8))
    theta=rng.normal(size=8);psi=jac@theta;psi[1]=0.
    shift=psi-jac@theta;density=np.exp(rng.normal(size=6))
    rows=jnp.arange(6)[:,None]
    batch=lambda x,r:jnp.asarray(jac)[r[:,0]]@x+jnp.asarray(shift)[r[:,0]]
    forward,adjoint=projected_streaming_operators(
        batch,jnp.asarray(theta),rows,jnp.asarray(psi),jnp.asarray(density))
    scale=1/np.sqrt(density*np.sum(abs(psi)**2/density));u=psi*scale
    a=(np.eye(6)-np.outer(u,u.conj()))@(jac*scale[:,None])
    real_a=np.concatenate((a.real,a.imag),axis=0)
    direction=rng.normal(size=8);weights=rng.normal(size=12)
    np.testing.assert_allclose(forward(jnp.asarray(direction)),real_a@direction,atol=3e-13)
    np.testing.assert_allclose(adjoint(jnp.asarray(weights)),real_a.T@weights,atol=3e-13)
    damp=.3
    result=damped_lsqr(forward,adjoint,jnp.asarray(weights),damping=damp,
                       max_iterations=24,relative_tolerance=1e-11)
    oracle=np.linalg.solve(real_a.T@real_a+damp*np.eye(8),real_a.T@weights)
    np.testing.assert_allclose(result.solution,oracle,atol=2e-11,rtol=2e-11)
    assert np.linalg.norm(jac[1])>0


def test_shape_budget_matches_existing_legal_active_layout_without_tensor_allocation():
    from projects.clebsch_tree_nqs.explorations.ct12_width_shape_budget_20261030 import width_shape_plan
    profile=tuple({0:1} if i==14 else {q:(20 if i<12 else 28) for q in range(0,11,2)} for i in range(15))
    counted=width_shape_plan(28,profile,runtime_cap=28)
    layout=mixed_active_layout(MixedClebschConfig(12,28,10),profile)
    assert counted['parameters']==10_432_128
    assert counted['local_parameters']==53_024
    for count,row in zip(counted['upper_rows'],layout,strict=True):
        assert count['parameters']==sum(int(np.prod(group['shape'])) for group in row['groups'])
    assert counted['existing_API_executable']
    hypothetical=width_shape_plan(20,tuple({0:1} if i==14 else {q:28 for q in range(0,11,2)} for i in range(15)),runtime_cap=28)
    assert hypothetical['parameters']==18_146_416
    assert not hypothetical['existing_API_executable']
