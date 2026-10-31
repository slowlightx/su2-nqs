"""Same physical trained parent, deliberately different constrained tangents."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train import trained_parent_family,arity_parent_family,resolve_arm_spec
from su2_nqs.frontier.artifacts import canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig,init_mixed_clebsch_family,mixed_clebsch_bank,
    make_mixed_clebsch_amplitude,
)
from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.vmc.checkpoint import save_checkpoint

jax.config.update('jax_enable_x64',True)


def fixture_parent(directory):
    cfg=MixedClebschConfig(multiplicity_cap=9,maximum_q=3)
    family=init_mixed_clebsch_family(jax.random.PRNGKey(1601),cfg)
    # A noninitial endpoint: arbitrary top and local C3 coefficients remain
    # strict intertwiners; no energy optimization is needed for this oracle.
    p=jax.tree.map(lambda v:v+.003*jnp.sin(v),family['ternary_c3'])
    config=dict(model=dict(length=6,multiplicity_cap=9,maximum_q=3),arm='ternary_c3',j2_over_j1=.125)
    registration=dict(config=config,sources={},cost_registration_sha256='cost')
    registration['sha256']=canonical_sha256(registration)
    (directory/'registration.json').write_text(json.dumps(registration))
    key=jax.random.PRNGKey(1602);states=jnp.zeros((2,36),jnp.int32)
    opt=_manual_adam_init(p)
    metadata=dict(registration_sha256=registration['sha256'],params_sha256=pytree_sha256(p),
        completed_steps=0,history=[],optimizer_step=0,optimizer_sha256=pytree_sha256(opt),
        rng=np.asarray(key),sampler_states=np.asarray(states),sampler_rng_sha256=pytree_sha256((key,states)))
    save_checkpoint(directory/'checkpoint.pkl',p,optimizer_state=opt,metadata=metadata)
    audit=dict(registration_sha256=registration['sha256'],params_sha256=pytree_sha256(p),energy_per_site=-.39)
    (directory/'terminal_audit.json').write_text(json.dumps(audit))
    request=dict(directory=str(directory),registration_sha256=registration['sha256'],
        checkpoint_sha256=file_sha256(directory/'checkpoint.pkl'),params_sha256=pytree_sha256(p),
        completed_steps=0,audit_sha256=file_sha256(directory/'terminal_audit.json'))
    return cfg,family,dict(config,trained_parent=request),p


def test_trained_parent_migration_preserves_raw_amplitudes_and_gradients(tmp_path):
    cfg,family,spec,parent=fixture_parent(tmp_path)
    migrated,note=trained_parent_family(family,cfg,spec)
    assert note['optimizer_reset'] and note['sampler_reset'] and not note['tangent_spaces_identical']
    assert pytree_sha256(migrated['ternary_c3'])==pytree_sha256(parent)
    amp=make_mixed_clebsch_amplitude(cfg)
    rng=np.random.default_rng(1603)
    bits=jnp.asarray([rng.permutation([0]*18+[1]*18) for _ in range(8)])
    values={mode:jax.jit(jax.vmap(amp,in_axes=(None,0)))(mixed_clebsch_bank(p,cfg,mode),bits)
            for mode,p in migrated.items()}
    assert np.linalg.norm(values['ternary_c3'])>1e-15
    for v in values.values():np.testing.assert_allclose(v,values['ternary_c3'],rtol=3e-12,atol=3e-17)
    # The C3 tangent pushed through its exact synthesis has the same physical
    # derivative in the free parameterization (without claiming all free
    # directions have constrained preimages).
    tangent=jax.tree.map(lambda v:jnp.cos(v)*.001,parent)
    bank,dbank=jax.jvp(lambda p:mixed_clebsch_bank(p,cfg,'ternary_c3'),(parent,),(tangent,))
    a=amp(bank,bits[0])
    da=jax.jvp(lambda p:amp(mixed_clebsch_bank(p,cfg,'ternary_c3'),bits[0]),(parent,),(tangent,))[1]
    db=jax.jvp(lambda b:amp(b,bits[0]),(bank,),(dbank,))[1]
    np.testing.assert_allclose(da,db,rtol=2e-12,atol=1e-17)


@pytest.mark.parametrize('field',['checkpoint_sha256','params_sha256','audit_sha256','registration_sha256'])
def test_trained_parent_rejects_changed_seals(tmp_path,field):
    cfg,family,spec,_=fixture_parent(tmp_path)
    spec['trained_parent'][field]='0'*64
    with pytest.raises(ValueError):trained_parent_family(family,cfg,spec)


def test_arity_pair_requires_fresh_audit_and_rejects_parameter_budget_mismatch(tmp_path):
    cfg,_,spec,_=fixture_parent(tmp_path)
    request=spec.pop('trained_parent')
    spec['arity_comparison']=dict(parent=request,binary_rank=2,maximum_parameter_mismatch_fraction=1.)
    family,note=arity_parent_family(cfg,spec)
    assert set(family)=={'binary_truncated','ternary_joint'}
    assert note['fresh_initial_audit_required'] and note['physical_parent_changed_by_truncation']
    assert note['same_physical_initial_state']
    assert note['parameter_mismatch_fraction']>0
    spec['arity_comparison']['maximum_parameter_mismatch_fraction']=0.
    with pytest.raises(ValueError,match='parameter budgets'):
        arity_parent_family(cfg,spec)
    spec['arity_comparison']['maximum_parameter_mismatch_fraction']=1.
    spec['arity_comparison']['maximum_local_parameter_mismatch_fraction']=0.
    with pytest.raises(ValueError,match='LOCAL parameter budgets'):
        arity_parent_family(cfg,spec)


def test_arity_config_cannot_override_training_budget_or_optimizer():
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/configs/remote/triangle_arity_20261016/comparison.json'
    spec=json.loads(path.read_text())
    a=resolve_arm_spec(spec,'binary_truncated');b=resolve_arm_spec(spec,'ternary_joint')
    assert {k for k in a if a[k]!=b[k]}=={'arm','terminal_audit_seed','output'}
    assert a['steps']==20 and a['samples']==128 and a['audit_samples']==512
    spec['arm_overrides']['ternary_joint']['learning_rate']=.1
    with pytest.raises(ValueError,match='shared training'):
        resolve_arm_spec(spec,'ternary_joint')


def test_arity_deployment_includes_static_cost_oracle_and_its_dependencies():
    from pathlib import Path
    from su2_nqs.frontier.provenance import transitive_local_source_manifest
    root=Path(__file__).resolve().parents[1]
    files=set((root/'projects/clebsch_tree_nqs/remote/triangle_arity_sync_20261016.txt').read_text().splitlines())
    entries=(root/'projects/clebsch_tree_nqs/workflows/spin_triangular_clebsch_train.py',
             root/'projects/clebsch_tree_nqs/workflows/spin_triangular_decoder_cost.py',
             root/'su2_nqs/vmc/local_energy.py')
    needed=transitive_local_source_manifest(entries,root=root,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    assert set(needed)<=files
    assert all((root/path).is_file() for path in files)
