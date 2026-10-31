from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.adaptive_projected_pair_carrier import (
    batched_singlet_agp_amplitudes,
    generic_adaptive_pair_control,
    triangular_physical_adaptive_pair_carrier,
    triangular_physical_pairing_bank,
)
from su2_nqs.benchmarks.projected_singlet_agp import (
    singlet_agp_determinant_amplitude,
)
from su2_nqs.benchmarks.qsl_three_arm import (
    triangular_pi_flux_operator_components,
)
from su2_nqs.benchmarks.xdiag_physical_seed_krylov import (
    SMALL_CASES,
    configuration_from_repository_integer,
    fixed_sz_sparse_hamiltonian,
    physical_seed_amplitudes,
    spin_raising_relative_norm,
)


def test_batched_agp_matches_scalar_repository_convention() -> None:
    configurations = np.asarray(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.int8,
    )
    pairing = np.asarray(
        [
            [0.0, 0.3, -0.2, 0.7],
            [0.3, 0.0, 0.5, -0.4],
            [-0.2, 0.5, 0.0, 0.9],
            [0.7, -0.4, 0.9, 0.0],
        ]
    )
    expected = np.asarray(
        [
            singlet_agp_determinant_amplitude(configuration, pairing)
            for configuration in configurations
        ]
    )
    np.testing.assert_allclose(
        batched_singlet_agp_amplitudes(configurations, pairing),
        expected,
        atol=2.0e-15,
    )


def test_zero_deformation_is_registered_pi_zero_flux_singlet() -> None:
    case = SMALL_CASES["square4"]
    _hamiltonian, basis = fixed_sz_sparse_hamiltonian(case, 0.125)
    configurations = np.asarray(
        [
            configuration_from_repository_integer(value, case.n_sites)
            for value in basis
        ],
        dtype=np.int8,
    )
    family = triangular_physical_adaptive_pair_carrier(configurations, case)
    observed = family.amplitudes(np.zeros(family.parameter_count))
    expected = physical_seed_amplitudes(configurations, case)[:, 0]
    observed /= np.linalg.norm(observed)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(observed, expected, atol=7.0e-15)
    assert spin_raising_relative_norm(observed, basis, case.n_sites)[0] < 2.0e-13
    assert family.deformation_labels == (
        "A1_J1_pair",
        "A1_J2_pair",
        "E2_J1_pair",
    )


def test_generic_control_is_deterministic_and_scale_gauge_orthogonal() -> None:
    case = SMALL_CASES["square4"]
    _hamiltonian, basis = fixed_sz_sparse_hamiltonian(case, 0.075)
    configurations = np.asarray(
        [
            configuration_from_repository_integer(value, case.n_sites)
            for value in basis
        ],
        dtype=np.int8,
    )
    physical = triangular_physical_adaptive_pair_carrier(configurations, case)
    first = generic_adaptive_pair_control(physical, seed=20260920)
    second = generic_adaptive_pair_control(physical, seed=20260920)
    for left, right in zip(
        first.deformation_pairings, second.deformation_pairings, strict=True
    ):
        np.testing.assert_array_equal(left, right)
        assert abs(np.vdot(first.base_pairing, left)) < 2.0e-13
        np.testing.assert_allclose(
            np.linalg.norm(left), np.linalg.norm(first.base_pairing), atol=2.0e-14
        )
    gram = np.asarray(
        [
            [np.vdot(left, right) for right in first.deformation_pairings]
            for left in first.deformation_pairings
        ]
    )
    target = np.eye(first.parameter_count) * np.linalg.norm(
        first.base_pairing
    ) ** 2
    np.testing.assert_allclose(gram, target, atol=5.0e-14)


def test_operator_vocabulary_accepts_an_identical_carrier_override() -> None:
    reference = triangular_pi_flux_operator_components(0.125)
    overridden = triangular_pi_flux_operator_components(
        0.125, carrier_override=reference.carrier
    )
    np.testing.assert_allclose(overridden.carrier, reference.carrier, atol=0.0)
    np.testing.assert_allclose(
        overridden.component_matrix, reference.component_matrix, atol=2.0e-13
    )


def test_support_free_pairing_bank_scales_to_n36() -> None:
    base, directions, labels = triangular_physical_pairing_bank(6)
    assert base.shape == (36, 36)
    assert len(directions) == len(labels) == 3
    assert all(direction.shape == base.shape for direction in directions)
    assert np.max(np.abs(base - base.T)) < 1.0e-13
    assert all(
        np.max(np.abs(direction - direction.T)) < 1.0e-13
        for direction in directions
    )
    assert all(
        abs(float(np.vdot(base, direction))) < 1.0e-12
        for direction in directions
    )


def test_cached_sector_objective_matches_raw_agp_and_has_scale_gauge() -> None:
    from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import _pair_sector_functions
    # Four spins and the COMPLETE fixed-Sz support. Identity U isolates the
    # objective algebra from an assumed symmetry projector and from saved data.
    bits=np.asarray([[0,0,1,1],[0,1,0,1],[0,1,1,0],
                     [1,0,0,1],[1,0,1,0],[1,1,0,0]],np.int8)
    h=np.diag(np.linspace(-1,1,6))
    c, objective=_pair_sector_functions(bits,np.eye(6),h)
    p=np.array([.3,-.2,.7,.5,-.4,.9])
    matrix=np.zeros((4,4));upper=np.triu_indices(4,1)
    matrix[upper]=p;matrix[(upper[1],upper[0])]=p
    raw=batched_singlet_agp_amplitudes(bits,matrix);raw/=np.linalg.norm(raw)
    np.testing.assert_allclose(c(p),raw,atol=2e-14,rtol=0)
    e,g=objective(p)
    assert abs(float(e)-np.vdot(raw,h@raw).real/4)<2e-14
    assert abs(np.asarray(g)@p)<2e-13
    direction=np.array([1.,-.3,.4,-.2,.7,.1]);eps=1e-5
    fd=(float(objective(p+eps*direction)[0])-float(objective(p-eps*direction)[0]))/(2*eps)
    assert abs(fd-np.asarray(g)@direction)<2e-9
    np.testing.assert_allclose(c(2*p),c(p),atol=2e-14,rtol=0)


def test_tiled_complete_pair_projection_matches_independent_numpy_and_gradient() -> None:
    import jax
    import jax.numpy as jnp
    from su2_nqs.benchmarks.projected_singlet_agp import ProjectedSingletAGP, square_gamma_a1_group
    from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
    from su2_nqs.su2.sz_basis import enumerate_sz_configs
    from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import _tiled_pair_amplitudes

    bits=np.asarray(enumerate_sz_configs(4,2),np.int8)
    p=np.array([.3,-.2,.7,.5,-.4,.9]);upper=np.triu_indices(4,1)
    matrix=np.zeros((4,4));matrix[upper]=p;matrix[(upper[1],upper[0])]=p
    group=square_gamma_a1_group(2);cfg=ProjectedSingletAGPConfig(4,group)
    ref=ProjectedSingletAGP(matrix,group).amplitudes(bits)
    weights=jnp.array([.2,-.3,.7,.9,-.4,.1])
    for tile in (1,4,9):
        fn=jax.jit(lambda a: _tiled_pair_amplitudes(a,jnp.asarray(bits),cfg,tile_size=tile))
        np.testing.assert_allclose(fn(jnp.asarray(p)),ref,atol=2e-14,rtol=0)
        derivative=jax.grad(lambda a: jnp.dot(fn(a),weights))(jnp.asarray(p))
        direction=np.arange(1,7)/10;eps=1e-5
        fd=float((fn(p+eps*direction)@weights-fn(p-eps*direction)@weights)/(2*eps))
        assert abs(fd-np.asarray(derivative)@direction)<2e-10


def test_tiled_full_pair_H_gradient_matches_independent_reference_with_nodes() -> None:
    import jax.numpy as jnp
    from su2_nqs.benchmarks.projected_singlet_agp import ProjectedSingletAGP, exact_small_heisenberg_reference
    from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
    from su2_nqs.su2.sz_basis import enumerate_sz_configs
    from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries
    from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import _pair_query_functions

    bits=np.asarray(enumerate_sz_configs(4,2),np.int8)
    group=(tuple(range(4)),);cfg=ProjectedSingletAGPConfig(4,group)
    bonds=((0,1,1.),(1,2,.7),(2,3,1.),(0,3,.3))
    plan=compile_physical_h_queries(bits,bonds)
    action,gradient=_pair_query_functions(cfg,plan,amplitude_scale=.7,tile_size=4)
    # One product of two singlets has physical nodes. Direct amplitudes and
    # full-H derivatives remain legal there; no log derivative is introduced.
    p=np.array([1.,0,0,0,0,1.]);upper=np.triu_indices(4,1)
    matrix=np.zeros((4,4));matrix[upper]=p;matrix[(upper[1],upper[0])]=p
    reference=exact_small_heisenberg_reference(ProjectedSingletAGP(matrix,group),4,bonds)
    a,b=action(jnp.asarray(p))
    np.testing.assert_allclose(a,reference.amplitudes/.7,atol=3e-14,rtol=0)
    np.testing.assert_allclose(b,reference.hamiltonian_actions/.7,atol=3e-14,rtol=0)
    assert np.any(np.asarray(a)==0)
    e,g=gradient(jnp.asarray(p),jnp.ones(len(bits)))
    expected=np.vdot(reference.amplitudes,reference.hamiltonian_actions).real/np.vdot(reference.amplitudes,reference.amplitudes).real/4
    assert abs(float(e)-expected)<3e-14
    assert np.all(np.isfinite(g))
    direction=np.array([.1,.2,-.3,.4,.2,-.1]);eps=1e-5
    def value(candidate):
        x,y=action(jnp.asarray(candidate))
        return float(jnp.vdot(x,y).real/jnp.vdot(x,x).real/4)
    assert abs((value(p+eps*direction)-value(p-eps*direction))/(2*eps)-np.asarray(g)@direction)<2e-9


def test_tiled_pair_value_validation_does_not_repair_bad_configurations() -> None:
    import pytest
    from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
    from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import _tiled_pair_amplitudes
    cfg=ProjectedSingletAGPConfig(4,(tuple(range(4)),))
    with pytest.raises(ValueError,match='half-filled'):
        _tiled_pair_amplitudes(np.ones(6),np.array([[1,1,1,0]]),cfg)
    with pytest.raises(ValueError,match='tile_size'):
        _tiled_pair_amplitudes(np.ones(6),np.array([[1,1,0,0]]),cfg,tile_size=0)
    with pytest.raises(ValueError,match='integer binary'):
        _tiled_pair_amplitudes(np.ones(6),np.array([[1.5,.5,0,0]]),cfg)


def test_blockwise_pair_moments_reproduce_global_energy_gradient_not_mean_ratios() -> None:
    from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
    from su2_nqs.su2.sz_basis import enumerate_sz_configs
    from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries
    from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import (
        _pair_query_functions, _pair_moment_functions)
    cfg=ProjectedSingletAGPConfig(4,(tuple(range(4)),))
    bits=np.asarray(enumerate_sz_configs(4,2),np.int8)
    bonds=((0,1,1.),(1,2,.7),(2,3,1.),(0,3,.3))
    p=np.array([.3,-.2,.7,.5,-.4,.9]);q=np.linspace(.2,1.8,6)
    n=d=0.;gn=np.zeros(6);gd=np.zeros(6)
    for begin,end in ((0,2),(2,6)):
        plan=compile_physical_h_queries(bits[begin:end],bonds)
        nf,df=_pair_moment_functions(cfg,plan,amplitude_scale=.7,tile_size=4)
        ((num,(den,_,_,_)),g)=nf(p,q[begin:end])
        n+=float(num);d+=float(den);gn+=np.asarray(g);gd+=np.asarray(df(p,q[begin:end]))
    _,ref=_pair_query_functions(cfg,compile_physical_h_queries(bits,bonds),amplitude_scale=.7,tile_size=4)
    e,g=ref(p,q)
    assert abs(n/d/4-float(e))<2e-14
    np.testing.assert_allclose((gn-(n/d)*gd)/d/4,g,atol=3e-13,rtol=0)


def test_pair_chain_audit_uses_pooled_ratios_and_matches_identical_states() -> None:
    from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import (
        _pair_weighted_moments,_pair_chain_comparison)
    a=np.array([1.,0.,2.,3.,1.,4.]);b=np.array([-.3,1.,-.9,-1.5,-.2,-2.3]);q=np.linspace(.1,1.,6)
    ref=_pair_weighted_moments(a,b,q,4)
    audit=_pair_chain_comparison({'a':(a,b),'b':(2*a,2*b)},q,3,4)
    assert abs(audit['states']['a']['energy_per_site']-ref[0])<1e-15
    assert abs(audit['paired']['b-minus-a']['energy'])<1e-15
    assert audit['paired']['b-minus-a']['energy_chain_SE']<1e-15
    assert audit['paired']['b-minus-a']['variance_chain_SE']<1e-15
    # Hpsi can be nonzero at a node and MUST enter the residual second moment.
    assert ref[1]>_pair_weighted_moments(a,b*np.where(a==0,0,1),q,4)[1]


def test_dynamic_pair_full_h_kernel_matches_exact_action_and_global_gradient() -> None:
    import jax.numpy as jnp
    from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
    from su2_nqs.su2.sz_basis import enumerate_sz_configs
    from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries
    from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import (
        _pair_dynamic_functions, _pair_padded_query_arguments, _pair_query_functions)
    bits=np.asarray(enumerate_sz_configs(4,2),np.int8)
    cfg=ProjectedSingletAGPConfig(4,(tuple(range(4)),))
    bonds=((0,1,1.),(1,2,.7),(2,3,1.),(0,3,.3))
    # The up=(2,3),down=(0,1) determinant has rank one: its value is zero,
    # but its cofactors/tangent are nonzero. A rank-zero minor would not test this.
    p=np.array([.3,1.,1.,1.,1.,.9]);q=np.linspace(.2,1.8,6)
    fns=_pair_dynamic_functions(cfg,amplitude_scale=.7,tile_size=4)
    plan=compile_physical_h_queries(bits,bonds)
    reference,refgrad=_pair_query_functions(cfg,plan,amplitude_scale=.7,tile_size=4)
    ra,rb=reference(p);re,rg=refgrad(p,q)
    for bucket in (1,5,16):
        args=_pair_padded_query_arguments(plan,bucket=bucket)
        a,b=fns['action'](p,*args)
        np.testing.assert_allclose(a,ra,atol=2e-14,rtol=0)
        np.testing.assert_allclose(b,rb,atol=2e-14,rtol=0)
        ((n,(d,_,_,_)),gn)=fns['numerator'](p,*args,q)
        gd=fns['norm_gradient'](p,jnp.asarray(bits),q)
        assert abs(float(n/d/4)-float(re))<2e-14
        np.testing.assert_allclose((gn-(n/d)*gd)/d/4,rg,atol=2e-13,rtol=0)
    jac=np.asarray(fns['center_jacobian'](p,jnp.asarray(bits)))
    # Some zero amplitudes have nonzero tangent; these are never log-repaired.
    assert np.any((np.asarray(ra)==0)&(np.linalg.norm(jac,axis=1)>0))


def test_registered_pair_coupling_preserves_zero_and_rejects_invalid_values() -> None:
    import pytest
    from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import _pair_registered_coupling
    for coupling in (0, .125, .175):
        assert _pair_registered_coupling({'j2_over_j1':coupling}) == coupling
    for invalid in (None, True, -1., float('nan'), float('inf'), '0'):
        with pytest.raises(ValueError,match='finite nonnegative'):
            _pair_registered_coupling({'j2_over_j1':invalid})


def test_pair_coupling_changes_actual_triangular_h_action_not_only_metadata() -> None:
    from su2_nqs.benchmarks.projected_singlet_agp import ProjectedSingletAGP, exact_small_heisenberg_reference
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
    from su2_nqs.su2.sz_basis import enumerate_sz_configs
    from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries
    from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import (
        _pair_dynamic_functions,_pair_padded_query_arguments,_pair_registered_coupling)
    bits=np.asarray(enumerate_sz_configs(6,3),np.int8)
    p=np.random.default_rng(2026100701).normal(size=15)
    matrix=np.zeros((6,6));upper=np.triu_indices(6,1)
    matrix[upper]=p;matrix[(upper[1],upper[0])]=p
    group=(tuple(range(6)),);cfg=ProjectedSingletAGPConfig(6,group)
    fns=_pair_dynamic_functions(cfg,amplitude_scale=.8,tile_size=4)
    actions=[]
    for raw in (0,.125):
        coupling=_pair_registered_coupling({'j2_over_j1':raw})
        bonds=triangular_j1j2_bonds(2,3,j1=1.,j2=coupling,periodic=True)
        plan=compile_physical_h_queries(bits,bonds)
        a,b=fns['action'](p,*_pair_padded_query_arguments(plan,bucket=8))
        ref=exact_small_heisenberg_reference(ProjectedSingletAGP(matrix,group),6,bonds)
        np.testing.assert_allclose(a,ref.amplitudes/.8,atol=2e-13,rtol=0)
        np.testing.assert_allclose(b,ref.hamiltonian_actions/.8,atol=5e-13,rtol=0)
        actions.append(np.asarray(b))
    assert np.linalg.norm(actions[0]-actions[1]) > 1e-4
    # The physical H kernels, not an audit label, contain the different J2 data.
    bonds0=triangular_j1j2_bonds(6,6,j1=1.,j2=0.,periodic=True)
    bonds1=triangular_j1j2_bonds(6,6,j1=1.,j2=.125,periodic=True)
    assert len(bonds0)==108 and len(bonds1)==216


def test_importance_amplitude_metric_matches_normalized_state_and_keeps_nodes() -> None:
    from su2_nqs.vmc.sr import importance_amplitude_metric
    from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import _pair_sine_fs
    a=np.array([1.,0.,-2.,3.],complex)
    jac=np.array([[1.,.4],[0.,2.],[-2.,-.3],[3.,.7]],complex)
    q=np.array([.2,1.3,.7,2.1])
    metric=importance_amplitude_metric(a,jac,q)
    v=np.array([.1,-.3]);eps=1e-4
    finite=_pair_sine_fs(a,a+eps*(jac@v),q)**2/eps**2
    assert abs(finite-v@metric@v/2)<1e-6
    assert abs(metric[0,0])<3e-15  # pure amplitude-scale gauge
    assert np.linalg.eigvalsh(metric).min()>-2e-14
    np.testing.assert_allclose(importance_amplitude_metric(2*a,2*jac,7*q),metric,atol=2e-14,rtol=0)
    suppressed=jac.copy();suppressed[1]=0
    assert metric[1,1]>importance_amplitude_metric(a,suppressed,q)[1,1]


def test_importance_amplitude_metric_rejects_unsafe_density() -> None:
    import pytest
    from su2_nqs.vmc.sr import importance_amplitude_metric
    for q in (np.array([1.,0.]),np.array([1.,np.nan])):
        with pytest.raises(ValueError):
            importance_amplitude_metric(np.ones(2),np.ones((2,3)),q)
    with pytest.raises(ValueError):
        importance_amplitude_metric(np.zeros(2),np.ones((2,3)),np.ones(2))


def test_canonical_hermitian_force_equals_exact_gradient_including_rank_one_nodes() -> None:
    import jax.numpy as jnp
    from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
    from su2_nqs.su2.sz_basis import enumerate_sz_configs
    from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries
    from su2_nqs.vmc.sr import importance_hermitian_force
    from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import (
        _pair_dynamic_functions,_pair_padded_query_arguments,_pair_query_functions)
    bits=np.asarray(enumerate_sz_configs(4,2),np.int8)
    cfg=ProjectedSingletAGPConfig(4,(tuple(range(4)),))
    bonds=((0,1,1.),(1,2,.7),(2,3,1.),(0,3,.3))
    p=np.array([.3,1.,1.,1.,1.,.9]);q=np.ones(6)
    fns=_pair_dynamic_functions(cfg,amplitude_scale=.7,tile_size=4)
    plan=compile_physical_h_queries(bits,bonds)
    a,b=fns['action'](p,*_pair_padded_query_arguments(plan))
    jac=fns['center_jacobian'](p,jnp.asarray(bits))
    force=importance_hermitian_force(a,jac,b,q,n_sites=4)
    _,refgrad=_pair_query_functions(cfg,plan,amplitude_scale=.7,tile_size=4)
    _,expected=refgrad(p,q)
    np.testing.assert_allclose(force,expected,atol=3e-13,rtol=0)
    assert abs(force@p)<3e-13
    assert np.any(np.asarray(a)==0)
    # Explicitly removing node contributions changes the physical force.
    filtered=np.asarray(jac).copy();filtered[np.asarray(a)==0]=0
    assert np.linalg.norm(force-importance_hermitian_force(a,filtered,b,q,n_sites=4))>1e-4


def test_canonical_force_obeys_sampled_cauchy_bound_and_removes_both_gauges() -> None:
    from su2_nqs.vmc.sr import importance_amplitude_metric,importance_hermitian_force
    rng=np.random.default_rng(617);a=rng.normal(size=8)+1j*rng.normal(size=8);a[0]=0
    jac=rng.normal(size=(8,6))+1j*rng.normal(size=(8,6));jac[:,0]=a;jac[:,1]=1j*a
    b=rng.normal(size=8)+1j*rng.normal(size=8);q=np.exp(rng.normal(size=8))
    S=importance_amplitude_metric(a,jac,q);g=importance_hermitian_force(a,jac,b,q,n_sites=6)
    norm=np.sum(abs(a)**2/q);mean=np.sum(a.conj()*b/q)/norm
    variance=np.sum(abs(b-mean*a)**2/q)/norm
    assert max(abs(g[:2]))<2e-14
    for _ in range(20):
        d=rng.normal(size=6);fs=np.sqrt(max(0.,d@S@d/2))
        assert abs(g@d)<=2*np.sqrt(variance)/6*fs+2e-13
    np.testing.assert_allclose(importance_hermitian_force(2*a,2*jac,2*b,7*q,n_sites=6),g,atol=2e-14,rtol=0)


def test_generic_primitive_seed_opens_tangent_without_breaking_physical_singlet_a1() -> None:
    """Complete N12 support, no ED or optimization; independent S+ cancellation."""
    import jax.numpy as jnp
    from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
    from su2_nqs.su2.sz_basis import enumerate_sz_configs
    from su2_nqs.vmc.sr import importance_amplitude_metric
    from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import (
        _pair_dynamic_functions, _pair_generic_perturbation)
    n = 12
    group = tuple(sorted({tuple((s*i+t)%n for i in range(n))
        for s in (-1,1) for t in range(n)}))
    cfg = ProjectedSingletAGPConfig(n,group)
    separation = np.arange(n)[:,None]-np.arange(n)[None,:]
    # Antiperiodic primitive kernel: its fermionic projective signs compensate
    # the odd site-permutation parity of a one-site physical translation.
    pairing = sum(c*np.cos((2*k+1)*np.pi*separation/n)
        for k,c in enumerate([.6,-.9,.31,.22,-.1,.7]))
    p = pairing[np.triu_indices(n,1)]
    delta = _pair_generic_perturbation(p,relative_norm=.1,seed=314159)
    assert abs(delta@p)<1e-13
    assert abs(np.linalg.norm(delta)/np.linalg.norm(p)-.1)<1e-14
    np.testing.assert_array_equal(delta,_pair_generic_perturbation(p,relative_norm=.1,seed=314159))
    bits = np.asarray(enumerate_sz_configs(n,6),np.int8)
    targets = np.asarray(enumerate_sz_configs(n,7),np.int8)
    lookup = {tuple(x):i for i,x in enumerate(bits)}
    fns = _pair_dynamic_functions(cfg,amplitude_scale=1.,tile_size=8)
    ranks = []
    for values in (p,p+delta):
        a = np.asarray(fns['amplitude'](values,jnp.asarray(bits)))
        assert np.linalg.norm(a)>1.
        raised = []
        for target in targets:
            predecessors = []
            for site in np.flatnonzero(target):
                old = target.copy(); old[site] = 0
                predecessors.append(lookup[tuple(old)])
            raised.append(sum(a[predecessors]))
        assert np.linalg.norm(raised)/np.linalg.norm(a)<2e-12
        for image in (tuple((i+1)%n for i in range(n)),tuple((-i)%n for i in range(n))):
            permuted = np.asarray(fns['amplitude'](values,jnp.asarray(bits[:,image])))
            assert np.linalg.norm(permuted-a)/np.linalg.norm(a)<2e-12
        jac = np.concatenate([np.asarray(fns['center_jacobian'](values,jnp.asarray(bits[i:i+8])))
            for i in range(0,len(bits),8)])
        eig = np.linalg.eigvalsh(importance_amplitude_metric(a,jac,np.ones(len(bits))))
        ranks.append(int(np.count_nonzero(eig>eig[-1]*1e-8)))
    assert ranks == [3,11]


def test_projected_pairing_local_gauges_do_not_change_the_physical_tangent() -> None:
    """Independent determinant identities, before and after spatial restoration."""
    import jax.numpy as jnp
    from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
    from su2_nqs.su2.sz_basis import enumerate_sz_configs
    from su2_nqs.vmc.sr import importance_amplitude_metric
    from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import _pair_dynamic_functions
    n = 8; upper = np.triu_indices(n,1)
    raw = np.random.default_rng(271828).normal(size=(n,n))
    F = .5*(raw+raw.T); np.fill_diagonal(F,0.)
    p = F[upper]; bits = np.asarray(enumerate_sz_configs(n,4),np.int8)
    # A half-filled determinant has disjoint up/down sets. Its derivative
    # under DF+FD is (Tr D)*psi; under FDF it is (Tr DF)*psi=0 because Fii=0.
    scale = np.stack([(np.diag(e)@F+F@np.diag(e))[upper] for e in np.eye(n)],axis=1)
    nambu = np.stack([(F@np.diag(e)@F)[upper] for e in np.eye(n)],axis=1)
    ring = tuple(sorted({tuple((s*i+t)%n for i in range(n))
        for s in (-1,1) for t in range(n)}))
    for group in ((tuple(range(n)),),ring):
        cfg = ProjectedSingletAGPConfig(n,group)
        fns = _pair_dynamic_functions(cfg,amplitude_scale=1.,tile_size=8)
        a = np.asarray(fns['amplitude'](p,jnp.asarray(bits)))
        jac = np.concatenate([np.asarray(fns['center_jacobian'](p,jnp.asarray(bits[i:i+8])))
            for i in range(0,len(bits),8)])
        assert np.linalg.norm(jac@scale-a[:,None])/np.linalg.norm(a)<2e-12
        assert np.linalg.norm(jac@nambu)/np.linalg.norm(a)<2e-12
        metric = importance_amplitude_metric(a,jac,np.ones(len(bits)))
        for gauge in (scale,nambu):
            assert np.linalg.norm(gauge.T@metric@gauge)<2e-11
