"""Independent dense projective geometry and native prepared-Jacobian oracles."""
from functools import partial

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint
from su2_nqs.vmc.importance_amplitude_sr import solve_projected_block_sr,make_importance_amplitude_sr
from su2_nqs.vmc.physical_sr_adam import make_physical_sr_adam


def dense(j,p,h,q,ridge):
    norm=np.sum(p*p/q);u=p/np.sqrt(q*norm);d=j/np.sqrt(q*norm)[:,None]
    a=d-np.outer(u,u@d);energy=np.sum(p*h/q)/norm
    b=2*(h-energy*p)/np.sqrt(q*norm);g=a.T@b
    direction=np.linalg.solve(a.T@a+ridge*np.eye(j.shape[1]),g)
    return direction,g,a,energy


@pytest.mark.parametrize('parameters,chunk',[(1,8),(17,1),(17,6),(17,17),(17,64),(34,8)])
def test_projective_blocks_match_independent_dense_parameter_solve(parameters,chunk):
    rng=np.random.default_rng(27102701+parameters);n=9
    p=rng.normal(size=n);p[[0,4]]=0.;h=rng.normal(size=n);q=rng.uniform(.05,2.,n)
    j=rng.normal(size=(n,parameters));j[:,0]=p  # exact normalization gauge, including nodes.
    ridge=.17;expected,g,a,energy=dense(j,p,h,q,ridge)
    solve=jax.jit(partial(solve_projected_block_sr,damping=ridge,parameter_chunk_size=chunk))
    result,gradient,info=solve(*map(jnp.asarray,(j,p,h,q)))
    np.testing.assert_allclose(result,expected,atol=3e-14,rtol=2e-12)
    np.testing.assert_allclose(gradient,g,atol=2e-14,rtol=2e-13)
    np.testing.assert_allclose(info['raw_fisher_norm'],np.linalg.norm(a@expected),atol=3e-14,rtol=2e-12)
    assert bool(info['numerically_valid']) and int(info['exact_node_records'])==2
    assert float(info['relative_linear_residual'])<3e-12
    assert float(info['sampled_energy_total'])==pytest.approx(energy,abs=2e-15)
    assert int(info['maximum_normalized_jacobian_block_entries'])==n*min(chunk,parameters)
    if parameters>1:
        product=a@expected;weight=product**2
        b=2*(h-energy*p)/np.sqrt(q*np.sum(p*p/q));force_mass=abs(product*b)
        np.testing.assert_allclose(info['chosen_direction_tangent_square_ESS'],weight.sum()**2/(weight@weight),rtol=2e-12)
        np.testing.assert_allclose(info['chosen_direction_tangent_square_maximum_fraction'],weight.max()/weight.sum(),rtol=2e-12)
        np.testing.assert_allclose(info['chosen_direction_absolute_force_projection_ESS'],force_mass.sum()**2/(force_mass@force_mass),rtol=2e-12)
        np.testing.assert_allclose(info['chosen_direction_absolute_force_projection_maximum_fraction'],force_mass.max()/force_mass.sum(),rtol=2e-12)
        np.testing.assert_allclose(info['chosen_direction_signed_force_projection_sum'],expected@g,rtol=2e-12)


def test_exact_node_metric_and_force_are_not_lost_by_log_score_reweighting():
    # Complete two-state quadrature, H|psi> has nonzero weight exactly at psi=0.
    p=np.array([1.,0.]);h=np.array([0.,2.]);j=np.array([[0.],[1.]])
    q=np.ones(2)/2;ridge=.5
    direction,g,info=solve_projected_block_sr(*map(jnp.asarray,(j,p,h,q)),damping=ridge)
    np.testing.assert_allclose(g,[4.],rtol=0,atol=0)
    np.testing.assert_allclose(direction,[4/1.5],rtol=2e-15,atol=0)
    assert info['exact_node_records']==1 and info['raw_fisher_norm']==pytest.approx(4/1.5)
    # Literal physical normalized energy derivative agrees independently.
    step=1e-6
    def energy(theta):
        state=np.array([1.,theta]);ham=np.array([[0.,2.],[2.,0.]])
        return state@ham@state/(state@state)
    assert (energy(step)-energy(-step))/(2*step)==pytest.approx(float(g[0]),rel=2e-12)


def test_fixed_scale_and_energy_shift_invariance_without_Q_or_amplitude_floor():
    rng=np.random.default_rng(27102703);j=rng.normal(size=(9,7));p=rng.normal(size=9)
    h=rng.normal(size=9);q=rng.uniform(.03,.7,9)
    first=solve_projected_block_sr(*map(jnp.asarray,(j,p,h,q)),damping=.5,parameter_chunk_size=3)
    for scale,qscale,offset in [(-3.,.1,17.),(1e-40,1e-20,-3.)]:
        actual=solve_projected_block_sr(*map(jnp.asarray,(scale*j,scale*p,scale*(h+offset*p),q*qscale)),
            damping=.5,parameter_chunk_size=3)
        np.testing.assert_allclose(actual[0],first[0],atol=3e-14,rtol=3e-12)
        np.testing.assert_allclose(actual[1],first[1],atol=3e-14,rtol=3e-12)


def test_Born_limit_matches_old_F_force_ridge_and_direct_SR_update_exactly():
    # Finite Born quadrature may include signed amplitudes. Q=psi², so u
    # contains signs; the old log|psi| covariance is recovered nevertheless.
    rng=np.random.default_rng(27102704);features=rng.normal(size=(12,5));features[:,0]=1
    p=jnp.array([.01,-.02,.015,.004,-.007]);rows=jnp.arange(12)[:,None]
    signs=jnp.where(jnp.arange(12)%3==0,-1.,1.)
    def amp(theta,row):return signs[row[0]]*jnp.exp(jnp.asarray(features)[row[0]]@theta)
    psi=np.asarray(jax.vmap(lambda x:amp(p,x))(rows));energies=rng.normal(size=12)
    jac=features*psi[:,None];ridge=.5
    d,g,_=solve_projected_block_sr(*map(jnp.asarray,(jac,psi,energies*psi,psi**2)),damping=ridge,parameter_chunk_size=3)
    centered=features-features.mean(0);expected_g=2*centered.T@(energies-energies.mean())/12
    expected=np.linalg.solve(centered.T@centered/12+ridge*np.eye(5),expected_g)
    np.testing.assert_allclose(g,expected_g,atol=3e-15,rtol=2e-14)
    np.testing.assert_allclose(d,expected,atol=3e-15,rtol=2e-14)
    old=make_physical_sr_adam(amp,p,damping=ridge,score_chunk=3,sr_solver='dual_force',
        parameter_chunk_size=3,sr_use_adam=False,sr_tangent_fs_radius=.003,maximum_fs_sine=.02)
    new=make_importance_amplitude_sr(amp,p,damping=ridge,score_chunk=3,parameter_chunk_size=3)
    moments=_manual_adam_init(p);a,ma,ia=old(p,moments,rows,jnp.asarray(energies),.001,warm=True)
    b,mb,ib=new(p,moments,rows,jnp.asarray(energies*psi),jnp.asarray(psi**2),.001)
    assert ia['accepted'] and ib['accepted']
    for x,y in zip(jax.tree.leaves((a,ma)),jax.tree.leaves((b,mb)),strict=True):
        np.testing.assert_allclose(x,y,atol=2e-15,rtol=0)
    np.testing.assert_allclose(ia['energy_gradient_norm'],ib['energy_gradient_norm'],atol=3e-15,rtol=2e-14)


def fixture():
    params=dict(matrix=jnp.array([[.7,.2],[-.15,.9]]),vector=jnp.array([.13,-.17]),gauge=jnp.array(.11))
    rows=jnp.array([[0,0],[0,1],[1,0],[1,1]]*3)
    def prepare(p):return dict(weights=p['matrix']@p['vector'],
        pair=jnp.sin(p['matrix'][0,1])+p['vector'][0]*p['vector'][1],gauge=p['gauge'])
    def prepared(b,x):
        a,c=2*x[0]-1,2*x[1]-1
        return (1-2*x[0]*x[1])*jnp.exp(b['weights']@jnp.array([a,c])+b['pair']*a*c+b['gauge'])
    def amplitude(p,x):return prepared(prepare(p),x)
    a,c=np.asarray(2*rows[:,0]-1),np.asarray(2*rows[:,1]-1);ac=a*c
    matrix,vector=np.asarray(params['matrix']),np.asarray(params['vector'])
    score=np.column_stack((np.ones(len(rows)),a*vector[0],a*vector[1]+ac*np.cos(matrix[0,1]),
        c*vector[0],c*vector[1],a*matrix[0,0]+c*matrix[1,0]+ac*vector[1],
        a*matrix[0,1]+c*matrix[1,1]+ac*vector[0]))
    psi=np.asarray(jax.vmap(lambda x:amplitude(params,x))(rows))
    return params,rows,prepare,prepared,amplitude,psi,score*psi[:,None]


@pytest.mark.parametrize('chunk',[1,2,5])
def test_prepared_pullback_matches_all_native_columns_and_dense_oracle(chunk,tmp_path):
    p,rows,prepare,prepared,amplitude,psi,jac=fixture();flat,unravel=ravel_pytree(p)
    rng=np.random.default_rng(27102705);h=rng.normal(size=len(rows));q=rng.uniform(.05,1.,len(rows))
    direction,gradient,a,_=dense(jac,psi,h,q,.5)
    moments=(jnp.array(41),unravel(jnp.linspace(.001,.003,len(flat))),unravel(jnp.linspace(.01,.03,len(flat))))
    expected_scale=min(.03,.003/np.linalg.norm(a@direction));expected=np.asarray(flat)-expected_scale*direction
    outputs=[]
    for callbacks in ({},dict(prepare=prepare,prepared_amplitude=prepared)):
        update=make_importance_amplitude_sr(amplitude,p,score_chunk=chunk,parameter_chunk_size=3,**callbacks)
        proposed,state,info=update(p,moments,rows,jnp.asarray(h),jnp.asarray(q),.03)
        assert info['accepted'] and info['proposal_only_requires_fresh_finite_guard']
        np.testing.assert_allclose(ravel_pytree(proposed)[0],expected,atol=3e-15,rtol=0)
        assert float(info['sampled_tangent_FS'])<=.003*(1+2e-15)
        assert int(state[0])==42
        for x,y in zip(jax.tree.leaves(state[1:]),jax.tree.leaves(moments[1:]),strict=True):np.testing.assert_array_equal(x,y)
        assert info['raw_energy_directional_derivative']==pytest.approx(float(gradient@(expected-flat)),abs=4e-15)
        outputs.append((proposed,state,info))
    for x,y in zip(jax.tree.leaves(outputs[0]),jax.tree.leaves(outputs[1]),strict=True):
        np.testing.assert_allclose(x,y,atol=3e-14,rtol=3e-12)
    path=tmp_path/'state.pkl';save_checkpoint(path,outputs[1][0],optimizer_state=outputs[1][1])
    checkpoint=load_checkpoint(path)
    update=make_importance_amplitude_sr(amplitude,p,score_chunk=chunk,prepare=prepare,prepared_amplitude=prepared)
    restored=update(checkpoint['params'],checkpoint['optimizer_state'],rows,jnp.asarray(h),jnp.asarray(q),.001)
    direct=update(outputs[1][0],outputs[1][1],rows,jnp.asarray(h),jnp.asarray(q),.001)
    for x,y in zip(jax.tree.leaves(restored),jax.tree.leaves(direct),strict=True):np.testing.assert_array_equal(x,y)


def test_JIT_invalid_Q_does_not_repair_density_or_commit_counter():
    p=jnp.array([.2]);rows=jnp.arange(4)[:,None]
    amplitude=lambda p,row:1+p[0]*(row[0]-1.5)
    update=make_importance_amplitude_sr(amplitude,p)
    moments=_manual_adam_init(p)
    for q in (jnp.array([1.,0.,1.,1.]),jnp.array([1.,jnp.nan,1.,1.])):
        proposed,state,info=update(p,moments,rows,jnp.ones(4),q,.03)
        assert not info['accepted'] and not info['numerically_valid']
        np.testing.assert_array_equal(proposed,p);assert int(state[0])==0


def test_zero_force_concentration_is_explicitly_undefined_not_regularized():
    j=jnp.array([[0.,1.],[1.,2.],[2.,-1.]])
    _,g,info=solve_projected_block_sr(j,jnp.ones(3),jnp.zeros(3),jnp.ones(3),damping=.5)
    np.testing.assert_array_equal(g,np.zeros(2))
    assert info['numerically_valid'] and info['raw_fisher_norm']==0
    assert not info['chosen_direction_tangent_concentration_defined']
    assert not info['chosen_direction_force_concentration_defined']
    assert info['chosen_direction_tangent_square_ESS']==0
    assert info['chosen_direction_absolute_force_projection_ESS']==0


@pytest.mark.parametrize('defect',['shape','float32','complex','Qzero','Qnan','zero_norm','ridge','chunk'])
def test_public_solver_rejects_invalid_contract(defect):
    j=jnp.ones((4,3));p=jnp.ones(4);h=jnp.arange(4,dtype=jnp.float64);q=jnp.ones(4)
    options=dict(damping=.5,parameter_chunk_size=2)
    if defect=='shape':j=j[0]
    if defect=='float32':j=j.astype(jnp.float32)
    if defect=='complex':h=h.astype(jnp.complex128)
    if defect=='Qzero':q=q.at[1].set(0)
    if defect=='Qnan':q=q.at[1].set(jnp.nan)
    if defect=='zero_norm':p=p*0
    if defect=='ridge':options['damping']=False
    if defect=='chunk':options['parameter_chunk_size']=True
    with pytest.raises(ValueError):solve_projected_block_sr(j,p,h,q,**options)


def test_x64_and_preparation_contract_boundaries():
    p=jnp.array([1.]);amp=lambda p,x:p[0]
    with pytest.raises(ValueError):make_importance_amplitude_sr(amp,p,prepare=lambda p:p)
    with pytest.raises(ValueError):make_importance_amplitude_sr(amp,p.astype(jnp.float32))
    try:
        jax.config.update('jax_enable_x64',False)
        with pytest.raises(RuntimeError):make_importance_amplitude_sr(amp,p)
    finally:jax.config.update('jax_enable_x64',True)


def test_compiled_workspace_does_not_materialize_second_full_Jacobian():
    count,parameters,chunk=16,131071,1024
    function=jax.jit(partial(solve_projected_block_sr,damping=.5,parameter_chunk_size=chunk))
    compiled=function.lower(jax.ShapeDtypeStruct((count,parameters),jnp.float64),
        *[jax.ShapeDtypeStruct((count,),jnp.float64)]*3).compile()
    memory=compiled.memory_analysis();raw_bytes=count*parameters*8
    assert memory.temp_size_in_bytes<raw_bytes*.5,memory
