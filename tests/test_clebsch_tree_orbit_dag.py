"""Independent numerical equivalence tests; no optimization results used."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.literature_clebsch_tree import LiteratureClebschTreeConfig, init_literature_clebsch_tree
from su2_nqs.models.clebsch_tree_projected import make_projected_clebsch
from su2_nqs.lattices.geometry_tree import square_geometry_tree
from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from projects.clebsch_tree_nqs.workflows.spin import _groups

jax.config.update('jax_enable_x64', True)
from su2_nqs.models.clebsch_tree_orbit_dag import make_orbit_dag_clebsch, make_clebsch_execution


@pytest.mark.parametrize('length', [2, 4])
@pytest.mark.parametrize('fold', [False, True])
def test_all_amplitudes_tangent_and_full_H(length, fold):
    tree, order = square_geometry_tree(length, length)
    cfg = LiteratureClebschTreeConfig(tree, multiplicity_cap=4, maximum_q=8)
    p, _ = init_literature_clebsch_tree(jax.random.PRNGKey(79), cfg)
    group = _groups('square', length, length)['gamma_a1']
    baseline, _ = make_projected_clebsch(cfg, order, group, orbit_chunk=2, fold_root=fold)
    dag, report = make_orbit_dag_clebsch(cfg, order, group, fold_root=fold)
    basis = np.asarray(fixed_m_spin_basis(length * length, 0))
    x = basis if length == 2 else basis[::389]
    old, new = jax.jit(baseline)(p, x), jax.jit(dag)(p, x)
    np.testing.assert_allclose(old, new, rtol=4e-12, atol=2e-13)
    d = jax.tree.map(lambda v: jnp.sin(jnp.arange(v.size)).reshape(v.shape), p)
    old_jvp = jax.jvp(lambda z: baseline(z, x[:2]), (p,), (d,))[1]
    new_jvp = jax.jvp(lambda z: dag(z, x[:2]), (p,), (d,))[1]
    np.testing.assert_allclose(old_jvp, new_jvp, rtol=4e-12, atol=2e-13)
    loss = lambda f, z: jnp.sum(f(z, x[:2])**2)
    a, b = jax.grad(lambda z: loss(baseline, z))(p), jax.grad(lambda z: loss(dag, z))(p)
    for aa, bb in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(aa, bb, rtol=4e-12, atol=2e-13)
    bonds = square_j1j2_bonds(length, length, j2=.5, periodic=True)
    h0 = make_amplitude_heisenberg_action(lambda z, row: baseline(z, row[None])[0], bonds,
                                        n_sites=length * length)
    h1 = make_amplitude_heisenberg_action(lambda z, row: dag(z, row[None])[0], bonds,
                                        n_sites=length * length)
    for aa, bb in zip(h0(p, x[:2]), h1(p, x[:2]), strict=True):
        np.testing.assert_allclose(aa, bb, rtol=4e-12, atol=2e-13)
    assert all(r['unique_ordered_nodes'] <= r['image_count'] * r['original_nodes_per_image']
               for r in report['layers'])


def test_distinct_ordered_subtrees_not_identified_and_invalid_support():
    tree, order = square_geometry_tree(4, 4)
    cfg = LiteratureClebschTreeConfig(tree, multiplicity_cap=4, maximum_q=8)
    p, _ = init_literature_clebsch_tree(jax.random.PRNGKey(83), cfg)
    dag, report = make_orbit_dag_clebsch(cfg, order,
                            _groups('square', 4, 4)['gamma_a1'], fold_root=False)
    # Every undirected NN bond appears in both orientations, not one.
    assert report['layers'][0]['unique_ordered_nodes'] == 64
    invalid = np.zeros((2, 16), dtype=np.int32)
    invalid[1, 0] = 2
    np.testing.assert_array_equal(dag(p, invalid), 0.)


def test_complete_eight_site_support_and_independent_H_connectivity():
    tree, order = square_geometry_tree(2, 4)
    cfg = LiteratureClebschTreeConfig(tree, multiplicity_cap=4, maximum_q=8)
    p, _ = init_literature_clebsch_tree(jax.random.PRNGKey(91), cfg)
    group = _groups('square', 2, 4)['gamma_a1']
    old, _ = make_projected_clebsch(cfg, order, group, orbit_chunk=2)
    dag, _ = make_orbit_dag_clebsch(cfg, order, group)
    basis = np.asarray(fixed_m_spin_basis(8, 0))
    a, b = np.asarray(jax.jit(old)(p, basis)), np.asarray(jax.jit(dag)(p, basis))
    np.testing.assert_allclose(a, b, rtol=4e-12, atol=2e-13)
    index = {tuple(row): i for i, row in enumerate(basis)}
    bonds = square_j1j2_bonds(2, 4, j2=.5, periodic=True)
    graph = SzHeisenbergGraph(8, tuple(bonds), n_up=4)
    independent = []
    for row in basis:
        connected, elements = graph.get_conn(tuple(row))
        independent.append(sum(value*b[index[tuple(target)]]
                               for target, value in zip(connected, elements, strict=True)))
    action = make_amplitude_heisenberg_action(lambda z, row: dag(z, row[None])[0],
                                             bonds, n_sites=8)
    np.testing.assert_allclose(action(p, basis)[1], independent, rtol=4e-12, atol=2e-13)
    # Independent physical raising operation, not a label-only singlet check.
    raised = []
    for bra in np.asarray(fixed_m_spin_basis(8, 2)):
        value = 0.
        for site in np.flatnonzero(bra):
            ket = bra.copy()
            ket[site] = 0
            value += b[index[tuple(ket)]]
        raised.append(value)
    assert np.linalg.norm(raised) < 2e-12 * np.linalg.norm(b)
    for image in group:
        transformed = np.asarray([b[index[tuple(row[np.asarray(image)])]] for row in basis])
        np.testing.assert_allclose(transformed, b, rtol=4e-12, atol=2e-13)


def test_factory_keeps_legacy_default_and_rejects_unregistered_kernel():
    tree,order=square_geometry_tree(2,4)
    cfg=LiteratureClebschTreeConfig(tree,multiplicity_cap=4,maximum_q=8)
    p,_=init_literature_clebsch_tree(jax.random.PRNGKey(112),cfg)
    group=_groups('square',2,4)['gamma_a1']
    original,_=make_projected_clebsch(cfg,order,group,orbit_chunk=2)
    default,report=make_clebsch_execution(cfg,order,group,orbit_chunk=2)
    bits=np.asarray(fixed_m_spin_basis(8,0))[:8]
    np.testing.assert_array_equal(jax.jit(default)(p,bits),jax.jit(original)(p,bits))
    assert report['execution_kernel']=='legacy'
    for kernel in ('truncated_orbit','approximate',None):
        with pytest.raises(ValueError,match='kernel'):
            make_clebsch_execution(cfg,order,group,kernel=kernel)


def test_training_execution_settings_do_not_smuggle_scientific_changes():
    from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_training import _execution_settings
    spec=dict(projection_chunk=8,score_chunk=1,H_sample_chunk=32,fold_root_swap=True)
    assert _execution_settings(spec)==('legacy',8,1,32)
    assert _execution_settings(dict(spec,execution=dict(kernel='orbit_dag',score_chunk=4)))==('orbit_dag',8,4,32)
    for execution in (dict(learning_rate=.01),dict(score_chunk=0),dict(projection_chunk=True),
                      dict(kernel='approximate'),dict(kernel=[])):
        with pytest.raises(ValueError):
            _execution_settings(dict(spec,execution=execution))
    with pytest.raises(ValueError,match='validated'):
        _execution_settings(dict(spec,fold_root_swap=False,execution=dict(kernel='orbit_dag')))


def test_explicit_DAG_migration_restores_trained_optimizer_RNG_and_chains(tmp_path):
    import json
    import shutil
    from pathlib import Path
    from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_sampled import cost
    from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_training import train
    from su2_nqs.vmc.checkpoint import load_checkpoint
    from su2_nqs.frontier.provenance import file_sha256
    root=Path(__file__).resolve().parents[1]
    spec=dict(geometry='square',rows=4,columns=4,j2_over_j1=.5,multiplicity_cap=2,
        maximum_q=4,projection_chunk=4,require_gpu=False,seed=122,
        calibration_seed=123,sampling_seed=124)
    gate=cost(spec,tmp_path/'cost')
    spec.update(cost_parent=dict(directory=str(tmp_path/'cost'),checkpoint_sha256=gate['checkpoint_sha256'],
        registration_sha256=gate['registration_sha256']),training_seed=125,chains=4,steps=2,
        sr_warm_steps=1,warm_samples=8,adam_samples=8,burn_proposals=8,
        proposals_between_snapshots=4,score_chunk=2,H_sample_chunk=4,
        sr_damping=.01,maximum_FS_sine=.2,warm_lr=.005,adam_lr=.001,
        reset_moments_at_phase_switch=True,milestones=[0,1,2],maximum_training_seconds=180,
        fold_root_swap=True)
    old_dir=tmp_path/'legacy'
    parent=train(spec,old_dir,1,180)
    frozen=tmp_path/'frozen'
    frozen.mkdir()
    for name in ('checkpoint.pkl','registration.json'):
        shutil.copy2(old_dir/name,frozen/name)
    assert file_sha256(frozen/'checkpoint.pkl')==parent['checkpoint_sha256']
    baseline=train(spec,old_dir,2,180)
    new_spec=dict(spec,execution=dict(kernel='orbit_dag'),output=str(tmp_path/'dag'),
        training_resume_from=dict(directory=str(frozen),checkpoint_sha256=parent['checkpoint_sha256'],
            registration_sha256=parent['registration_sha256'],source_workspace=str(root),
            same_projected_physical_state_proved=True))
    migrated=train(new_spec,tmp_path/'dag',2,180)
    assert migrated['completed_steps']==baseline['completed_steps']==2
    old_cp,new_cp=load_checkpoint(old_dir/'checkpoint.pkl'),load_checkpoint(tmp_path/'dag/checkpoint.pkl')
    for name in ('params','optimizer_state'):
        for a,b in zip(jax.tree.leaves(old_cp[name]),jax.tree.leaves(new_cp[name]),strict=True):
            np.testing.assert_allclose(a,b,atol=3e-10,rtol=3e-10)
    for name in ('sampler_states','rng'):
        np.testing.assert_array_equal(old_cp['metadata'][name],new_cp['metadata'][name])
    assert file_sha256(frozen/'checkpoint.pkl')==parent['checkpoint_sha256']
    replay=json.loads((tmp_path/'dag/execution_replay.json').read_text())
    assert replay['execution']['kernel']=='orbit_dag' and replay['independent_legacy_backend']
    assert set(replay['maximum_relative_errors'])=={'amplitude','H_product','parameter_JVP','full_parameter_gradient'}
    assert max(replay['maximum_relative_errors'].values())<2e-11
    registration=json.loads((tmp_path/'dag/registration.json').read_text())
    assert 'su2_nqs/models/clebsch_tree_orbit_dag.py' in registration['source_manifest']
