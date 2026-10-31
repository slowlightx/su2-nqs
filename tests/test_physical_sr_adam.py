import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from su2_nqs.vmc.physical_sr_adam import make_physical_sr_adam
from su2_nqs.vmc.adam import _manual_adam_init

jax.config.update('jax_enable_x64',True)


def test_sr_adam_descent_and_atomic_optimizer_commit():
    p=jnp.array([.2],dtype=jnp.float64)
    def amplitude(p,x): return jnp.exp(p[0]*(2*x[0]-1))
    x=jnp.asarray([[0],[1],[0],[1]])
    e=jnp.asarray([1.,-1.,1.,-1.])
    update=make_physical_sr_adam(amplitude,p,maximum_fs_sine=.05)
    old=_manual_adam_init(p)
    candidate,opt,d=update(p,old,x,e,.005,warm=True)
    assert d['accepted'] and opt[0] == 1
    assert candidate[0] > p[0]  # More weight on the lower-energy bit1.
    assert d['relative_SR_residual'] < 1e-7
    assert d['FS_sine'] <= .05
    reject=make_physical_sr_adam(amplitude,p,maximum_fs_sine=1e-12,halvings=1)
    q,unchanged,diagnostics=reject(p,old,x,e,1.,warm=False)
    assert not diagnostics['accepted']
    np.testing.assert_array_equal(q,p)
    for a,b in zip(jax.tree.leaves(old),jax.tree.leaves(unchanged),strict=True):np.testing.assert_array_equal(a,b)


def test_adam_rematerialization_preserves_independent_full_score_gradient_and_optimizer():
    from su2_nqs.vmc.adam import _manual_adam_update
    p=jnp.asarray([.17,-.12,.08],dtype=jnp.float64)
    x=jnp.asarray([[0,0],[0,1],[1,0],[1,1]]*4)
    e=jnp.asarray([-.5,.2,-.1,.3]*4)
    features=jnp.column_stack((2*x[:,0]-1,2*x[:,1]-1,(2*x[:,0]-1)*(2*x[:,1]-1)))
    def amplitude(theta,row):
        a,b=2*row[0]-1,2*row[1]-1
        return jnp.exp(theta @ jnp.array([a,b,a*b]))
    independent_gradient=2*features.T @ (e-e.mean())/len(e)
    independent_gradient *= jnp.minimum(1.,.001/jnp.linalg.norm(independent_gradient))
    initial=_manual_adam_init(p)
    expected,expected_moments=_manual_adam_update(p,independent_gradient,initial,.001)
    results=[]
    for remat in (False,True):
        fn=make_physical_sr_adam(amplitude,p,score_chunk=2,rematerialize_adam=remat)
        candidate,moments,d=fn(p,initial,x,e,.001,warm=False)
        assert d['accepted'] and d['update_scale'] == 1
        np.testing.assert_allclose(candidate,expected,atol=2e-15,rtol=0)
        for a,b in zip(jax.tree.leaves(moments),jax.tree.leaves(expected_moments),strict=True):
            np.testing.assert_allclose(a,b,atol=2e-15,rtol=0)
        results.append((candidate,moments))
    for a,b in zip(jax.tree.leaves(results[0]),jax.tree.leaves(results[1]),strict=True):
        np.testing.assert_allclose(a,b,atol=2e-15,rtol=0)


def test_coordinate_adam_can_reverse_SR_descent_but_direct_SR_preserves_it():
    # Four-state independent exact Born oracle at theta=0. A positive metric
    # and exact energy force can produce an uphill SR->Adam coordinate step.
    # Integer features and energies make the target g=(1,2) exact in binary
    # arithmetic on both ARM and x86; no Cholesky rounding enters the oracle.
    f=np.array([[1.,6.],[-1.,-6.],[0.,1.],[0.,-1.]])
    S=f.T@f/4;g=np.array([1.,2.]);e=np.array([1.,-1.,-4.,4.])
    rows=jnp.array([[0,0],[0,1],[1,0],[1,1]])
    p=jnp.zeros(2,dtype=jnp.float64)
    def amplitude(theta,x):return jnp.exp(theta@jnp.asarray(f)[2*x[0]+x[1]])
    def energy(theta):
        w=jax.vmap(lambda row:amplitude(theta,row))(rows)**2
        return jnp.sum(w*jnp.asarray(e))/w.sum()
    np.testing.assert_allclose(jax.grad(energy)(p),g,atol=2e-15,rtol=0)
    natural=np.linalg.solve(S+.01*np.eye(2),g)
    np.testing.assert_allclose(natural,[12.51/.4401,-1.98/.4401],atol=2e-13,rtol=0)
    baseline=make_physical_sr_adam(amplitude,p,damping=.01,maximum_fs_sine=.05)
    old=_manual_adam_init(p)
    bad,_,info=baseline(p,old,rows,jnp.asarray(e),1e-5,warm=True)
    assert info['accepted'] and energy(bad)>energy(p)
    assert info['applied_energy_directional_derivative']>0
    direct=make_physical_sr_adam(amplitude,p,damping=.01,maximum_fs_sine=.05,
                                 sr_use_adam=False,sr_tangent_fs_radius=.02)
    good,moments,diagnostics=direct(p,old,rows,jnp.asarray(e),1e-5,warm=True)
    np.testing.assert_allclose(good,p-1e-5*natural,atol=2e-15,rtol=0)
    assert diagnostics['accepted'] and energy(good)<energy(p)
    assert diagnostics['applied_energy_directional_derivative']<0
    assert int(moments[0])==1
    for x,y in zip(jax.tree.leaves(moments[1:]),jax.tree.leaves(old[1:]),strict=True):
        np.testing.assert_array_equal(x,y)
    # Tangent radius limits a large proposed SR step without changing its ray.
    bounded=make_physical_sr_adam(amplitude,p,damping=.01,maximum_fs_sine=.05,
                                  sr_use_adam=False,sr_tangent_fs_radius=1e-4)
    q,_,d=bounded(p,old,rows,jnp.asarray(e),.1,warm=True)
    assert d['accepted'] and d['FS_sine']<=1.0001e-4 and energy(q)<energy(p)


def test_default_sr_adam_matches_independent_full_matrix_and_numpy_adam_oracle():
    # Independent parameter-space S solve, not the production sample-dual
    # Woodbury expression, and literal NumPy Adam equations. Include nonzero
    # accumulated moments and both warm/plain phases; default sr_use_adam=True.
    rows=jnp.asarray([[0,0],[0,1],[1,0],[1,1]]*4)
    features=np.asarray([[2*x[0]-1,2*x[1]-1,(2*x[0]-1)*(2*x[1]-1)] for x in rows])
    energies=np.asarray([-.5,.2,-.1,.3]*4)
    def amplitude(p,row):
        a,b=2*row[0]-1,2*row[1]-1
        return jnp.exp(p@jnp.array([a,b,a*b]))
    theta=np.array([.17,-.12,.08]);p=jnp.asarray(theta);state=_manual_adam_init(p)
    count=0;m=np.zeros(3);v=np.zeros(3);eta=1e-4
    centered=features-features.mean(0);gradient=2*centered.T@(energies-energies.mean())/len(rows)
    covariance=centered.T@centered/len(rows)
    update=make_physical_sr_adam(amplitude,p,damping=.01,score_chunk=2)
    for warm in (True,True,False,False):
        direction=(np.linalg.solve(covariance+.01*np.eye(3),gradient) if warm
                   else gradient*min(1.,.001/np.linalg.norm(gradient)))
        count+=1;m=.9*m+.1*direction;v=.999*v+.001*direction**2
        theta-=eta*(m/(1-.9**count))/(np.sqrt(v/(1-.999**count))+1e-8)
        p,state,info=update(p,state,rows,jnp.asarray(energies),eta,warm=warm)
        assert info['accepted'] and info['update_scale']==1
        np.testing.assert_allclose(p,theta,atol=2e-15,rtol=0)
        np.testing.assert_allclose(state[1],m,atol=2e-15,rtol=0)
        np.testing.assert_allclose(state[2],v,atol=2e-15,rtol=0)
        assert int(state[0])==count


@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('parameter_chunk_size', [None, 3])
def test_dual_force_matches_independent_dense_ridge_with_inherited_adam_moments(prepared, parameter_chunk_size):
    p,rows,e,prepare,prepared_amplitude,amplitude,scores,unravel=_nonlinear_prepared_fixture()
    flat=np.asarray(ravel_pytree(p)[0]);ridge=.05;eta=1e-4
    m=np.linspace(-.003,.004,len(flat));v=np.linspace(.002,.008,len(flat))
    state=(jnp.asarray(3),unravel(jnp.asarray(m)),unravel(jnp.asarray(v)))
    a=scores-scores.mean(0);g=2*a.T@(e-e.mean())/len(rows)
    direction=np.linalg.solve(a.T@a/len(rows)+ridge*np.eye(len(flat)),g)
    expected_m=.9*m+.1*direction;expected_v=.999*v+.001*direction**2
    expected=flat-eta*(expected_m/(1-.9**4))/(np.sqrt(expected_v/(1-.999**4))+1e-8)
    callbacks=dict(prepare=prepare,prepared_amplitude=prepared_amplitude) if prepared else {}
    update=make_physical_sr_adam(amplitude,p,damping=ridge,score_chunk=2,
                                sr_solver='dual_force',parameter_chunk_size=parameter_chunk_size,**callbacks)
    q,moments,info=update(p,state,rows,jnp.asarray(e),eta,warm=True)
    assert info['accepted'] and int(moments[0])==4
    np.testing.assert_allclose(ravel_pytree(q)[0],expected,atol=3e-14,rtol=0)
    np.testing.assert_allclose(ravel_pytree(moments[1])[0],expected_m,atol=3e-14,rtol=0)
    np.testing.assert_allclose(ravel_pytree(moments[2])[0],expected_v,atol=3e-14,rtol=0)
    assert info['relative_SR_residual']<1e-12


def test_dual_force_nearly_collinear_scores_against_dense_parameter_ridge():
    # Strongly correlated physical score columns expose Woodbury subtraction,
    # while the oracle directly solves the independent 3x3 parameter metric.
    features=np.array([[30,30.0001,1],[-30,-29.9999,-1],
                       [30,29.9999,-1],[-30,-30.0001,1]],float)
    rows=jnp.arange(4)[:,None];p=jnp.zeros(3,jnp.float64)
    energies=np.array([.2,-.1,-.4,.3]);ridge=.05;eta=1e-5
    def amplitude(theta,row):return jnp.exp(theta@jnp.asarray(features)[row[0]])
    a=features-features.mean(0);g=2*a.T@(energies-energies.mean())/4
    expected=np.linalg.solve(a.T@a/4+ridge*np.eye(3),g)
    update=make_physical_sr_adam(amplitude,p,damping=ridge,sr_use_adam=False,
                                sr_solver='dual_force')
    q,_,info=update(p,_manual_adam_init(p),rows,jnp.asarray(energies),eta,warm=True)
    assert info['accepted']
    np.testing.assert_allclose(-np.asarray(q)/eta,expected,atol=2e-12,rtol=2e-11)
    assert info['relative_SR_residual']<1e-10
    with pytest.raises(ValueError,match='solver'):
        make_physical_sr_adam(amplitude,p,sr_solver='unknown')


def test_default_sr_adam_bitwise_matches_frozen_optimizer_archive():
    import tarfile,types
    from pathlib import Path
    import pytest
    archive=Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/results/square_optimizer_conditions_20261016/source_snapshot.tar.gz'
    if not archive.exists():
        pytest.skip('optional research source archive absent; mandatory independent matrix/NumPy oracle still runs')
    with tarfile.open(archive) as tar:
        source=tar.extractfile('su2_nqs/vmc/physical_sr_adam.py').read().decode()
    old=types.ModuleType('su2_nqs.vmc._frozen_square_SR_Adam');old.__package__='su2_nqs.vmc'
    exec(compile(source,'frozen_physical_sr_adam.py','exec'),old.__dict__)
    rows=jnp.array([[0,0],[0,1],[1,0],[1,1]]*4);e=jnp.array([-.5,.2,-.1,.3]*4)
    p=jnp.array([.17,-.12,.08],dtype=jnp.float64)
    def amplitude(p,x):
        a,b=2*x[0]-1,2*x[1]-1
        return jnp.exp(p@jnp.array([a,b,a*b]))
    for radius in (.05,1e-4):
        reference=old.make_physical_sr_adam(amplitude,p,maximum_fs_sine=radius,score_chunk=2)
        default=make_physical_sr_adam(amplitude,p,maximum_fs_sine=radius,score_chunk=2)
        a,b=p,p;sa,sb=_manual_adam_init(p),_manual_adam_init(p)
        for warm in (True,True,False,False):
            a,sa,da=reference(a,sa,rows,e,.001,warm=warm)
            b,sb,db=default(b,sb,rows,e,.001,warm=warm)
            for x,y in zip(jax.tree.leaves((a,sa)),jax.tree.leaves((b,sb)),strict=True):
                np.testing.assert_array_equal(x,y)
            for key in da:np.testing.assert_array_equal(da[key],db[key])


def _nonlinear_prepared_fixture():
    # Nonlinear, shared preparation with a parameter in every original column,
    # including an exactly uniform normalization gauge. The repeated states
    # define an empirical real metric independently of how scores are computed.
    params=dict(matrix=jnp.array([[.7,.2],[-.15,.9]],dtype=jnp.float64),
                vector=jnp.array([.13,-.17],dtype=jnp.float64),
                gauge=jnp.array(.11,dtype=jnp.float64))
    rows=jnp.asarray([[0,0],[0,1],[1,0],[1,1]]*3)
    energies=np.array([.3,-.1,.2,-.4]*3)
    def prepare(p):
        return dict(weights=p['matrix']@p['vector'],
                    pair=jnp.sin(p['matrix'][0,1])+p['vector'][0]*p['vector'][1],
                    gauge=p['gauge'])
    def prepared_amplitude(bank,row):
        a,b=2*row[0]-1,2*row[1]-1
        # Nontrivial fixed real signs must survive forward overlap evaluation.
        sign=1.-2.*row[0]*row[1]
        return sign*jnp.exp(bank['weights']@jnp.array([a,b])+bank['pair']*a*b+bank['gauge'])
    def amplitude(p,row):return prepared_amplitude(prepare(p),row)
    features=np.array([[2*x[0]-1,2*x[1]-1,(2*x[0]-1)*(2*x[1]-1)] for x in rows])
    a,b,c=features.T;matrix=np.asarray(params['matrix']);v=np.asarray(params['vector'])
    # ravel_pytree(dict) orders keys gauge,matrix,vector. This analytic score
    # matrix does NOT use AD, the prepared pullback or the sample-dual solve.
    scores=np.column_stack((np.ones(len(rows)),a*v[0],a*v[1]+c*np.cos(matrix[0,1]),
                            b*v[0],b*v[1],a*matrix[0,0]+b*matrix[1,0]+c*v[1],
                            a*matrix[0,1]+b*matrix[1,1]+c*v[0]))
    flat,unravel=ravel_pytree(params)
    np.testing.assert_array_equal(flat,np.r_[params['gauge'],matrix.ravel(),v])
    return params,rows,energies,prepare,prepared_amplitude,amplitude,scores,unravel


@pytest.mark.parametrize('sr_use_adam',[False,True])
@pytest.mark.parametrize('warm',[False,True])
@pytest.mark.parametrize('rematerialize',[False,True])
def test_prepared_scores_match_literal_and_independent_original_parameter_metric(
    sr_use_adam,warm,rematerialize,
):
    p,rows,e,prepare,prepared_amplitude,amplitude,scores,unravel=_nonlinear_prepared_fixture()
    flat=np.asarray(ravel_pytree(p)[0]);ridge=.017;eta=1e-4
    old_m=np.linspace(-.003,.004,len(flat));old_v=np.linspace(.002,.008,len(flat))
    state=(jnp.array(3,dtype=jnp.int64),unravel(jnp.asarray(old_m)),unravel(jnp.asarray(old_v)))
    centered=scores-scores.mean(0)
    force=2*centered.T@(e-e.mean())/len(e)
    fisher=centered.T@centered/len(e)
    direction=(np.linalg.solve(fisher+ridge*np.eye(len(flat)),force) if warm
               else force*min(1.,.001/np.linalg.norm(force)))
    if warm and not sr_use_adam:
        expected=flat-eta*direction
        expected_m,expected_v=old_m,old_v
    else:
        expected_m=.9*old_m+.1*direction
        expected_v=.999*old_v+.001*direction**2
        expected=flat-eta*(expected_m/(1-.9**4))/(np.sqrt(expected_v/(1-.999**4))+1e-8)
    options=dict(damping=ridge,score_chunk=2,sr_use_adam=sr_use_adam,
                 rematerialize_adam=rematerialize)
    literal=make_physical_sr_adam(amplitude,p,**options)
    prepared=make_physical_sr_adam(amplitude,p,prepare=prepare,
        prepared_amplitude=prepared_amplitude,**options)
    actual=[]
    for update in (literal,prepared):
        q,moments,info=update(p,state,rows,jnp.asarray(e),eta,warm=warm)
        assert info['accepted'] and info['update_scale']==1
        np.testing.assert_allclose(ravel_pytree(q)[0],expected,atol=3e-14,rtol=0)
        np.testing.assert_allclose(ravel_pytree(moments[1])[0],expected_m,atol=3e-14,rtol=0)
        np.testing.assert_allclose(ravel_pytree(moments[2])[0],expected_v,atol=3e-14,rtol=0)
        np.testing.assert_allclose(info['raw_energy_directional_derivative'],
                                   force@(expected-flat),atol=3e-14,rtol=0)
        np.testing.assert_allclose(info['energy_gradient_norm'],np.linalg.norm(force),atol=3e-14,rtol=0)
        assert int(moments[0])==4
        if warm:
            assert info['relative_SR_residual']<1e-12
        actual.append((q,moments,info))
    for a,b in zip(jax.tree.leaves(actual[0]),jax.tree.leaves(actual[1]),strict=True):
        np.testing.assert_allclose(a,b,atol=3e-14,rtol=0)


def test_prepared_chain_rule_preserves_every_original_column_and_physical_JVP():
    p,rows,_,prepare,prepared_amplitude,amplitude,scores,unravel=_nonlinear_prepared_fixture()
    flat=ravel_pytree(p)[0]
    bank,pullback=jax.vjp(lambda values:prepare(unravel(values)),flat)
    prepared_score=jax.grad(lambda value,row:jnp.log(jnp.abs(prepared_amplitude(value,row))))
    actual=jax.vmap(lambda row:pullback(prepared_score(bank,row))[0])(rows)
    literal=jax.vmap(jax.grad(lambda values,row:jnp.log(jnp.abs(amplitude(unravel(values),row)))),
                     in_axes=(None,0))(flat,rows)
    np.testing.assert_allclose(actual,scores,atol=3e-15,rtol=0)
    np.testing.assert_allclose(literal,scores,atol=3e-15,rtol=0)
    assert np.all(np.linalg.norm(np.asarray(actual),axis=0)>0)
    tangent=jnp.linspace(-.07,.09,len(flat));dp=unravel(tangent)
    _,dbank=jax.jvp(prepare,(p,),(dp,))
    psi,dpsi=jax.jvp(lambda value:jax.vmap(amplitude,in_axes=(None,0))(value,rows),(p,),(dp,))
    _,prepared_jvp=jax.jvp(lambda value:jax.vmap(prepared_amplitude,in_axes=(None,0))(value,rows),
                           (bank,),(dbank,))
    np.testing.assert_allclose(dpsi,prepared_jvp,atol=3e-15,rtol=0)
    np.testing.assert_allclose(dpsi/psi,scores@np.asarray(tangent),atol=3e-15,rtol=0)
    # The analytic gauge is exactly constant. Differentiating log(exp(...))
    # carries ordinary floating-point roundoff in BOTH execution paths; retain
    # the independent all-column tolerance rather than changing the legacy
    # arithmetic to force a numerically exact unit derivative.
    np.testing.assert_array_equal((scores-scores.mean(0))[:,0],np.zeros(len(rows)))
    np.testing.assert_allclose(actual-actual.mean(0),scores-scores.mean(0),atol=3e-15,rtol=0)


def test_prepared_signed_ratios_preserve_sign_reversal_in_actual_overlap_guard():
    p=jnp.array([0.],dtype=jnp.float64);rows=jnp.array([[-1.],[1.]])
    e=jnp.array([1.,-1.]);initial=_manual_adam_init(p)
    def prepare(theta):return dict(weight=3*theta[0])
    def prepared_amplitude(bank,row):return 1+bank['weight']/3*row[0]
    def amplitude(theta,row):return 1+theta[0]*row[0]
    expected_theta=1.1*2/1.01
    values=np.array([1-expected_theta,1+expected_theta])
    expected_fs=np.sqrt(1-values.sum()**2/(2*(values@values)))
    abs_only_fs=np.sqrt(1-np.abs(values).sum()**2/(2*(values@values)))
    assert expected_fs>.8 and abs_only_fs<.8
    for radius in (.8,.99):
        for prepared in (False,True):
            options=dict(prepare=prepare,prepared_amplitude=prepared_amplitude) if prepared else {}
            update=make_physical_sr_adam(amplitude,p,damping=.01,score_chunk=1,
                sr_use_adam=False,maximum_fs_sine=radius,minimum_ess_fraction=.1,halvings=1,**options)
            q,moments,info=update(p,initial,rows,e,1.1,warm=True)
            assert bool(info['accepted'])==(radius==.99)
            if radius==.99:
                np.testing.assert_allclose(q,[expected_theta],atol=3e-14,rtol=0)
                np.testing.assert_allclose(info['FS_sine'],expected_fs,atol=3e-14,rtol=0)
            else:
                np.testing.assert_array_equal(q,p)
                for a,b in zip(jax.tree.leaves(moments),jax.tree.leaves(initial),strict=True):
                    np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('callbacks',[
    dict(prepare=lambda p:p),dict(prepared_amplitude=lambda bank,row:bank[0]),
    dict(prepare=7,prepared_amplitude=lambda bank,row:bank[0]),
    dict(prepare=lambda p:p,prepared_amplitude=7),
])
def test_prepared_execution_requires_a_callable_callback_pair(callbacks):
    with pytest.raises(ValueError,match='supplied together|must be callable'):
        make_physical_sr_adam(lambda p,row:jnp.exp(p[0]),jnp.array([.1],dtype=jnp.float64),**callbacks)
