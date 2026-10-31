"""Pure preflight scope, natural-scale comparisons, and inherited directions."""
from copy import deepcopy
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.explorations.ct12_w28_gpu_preflight_20261030 import (
    TASK,validate_plan,relative_error,inherited_local_direction,root_direction)
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig,mixed_clebsch_layout
from su2_nqs.models.triangular_clebsch_active_layout import mixed_active_layout


def _plan():
    return dict(task=TASK,length=12,j2=.125,target_q=0,target_m2=0,maximum_q=10,
        multiplicity_cap=28,active_real_parameters=20748160,complete_A1_images=1728,
        physical_H_bonds=864,rows=4,row_indices=[0,1,2,3],maximum_updates=0,
        scientific_fit_records_consumed=0,scientific_RNG_consumed=False,wall_cap_seconds=14400,
        maximum_peak_device_bytes=70*1024**3,nodelist=None,full_dense_Jacobian_allowed=False,
        required_phases=['amplitude','full_H','derivatives','fisher'],projection_chunk=16,
        phase_caps_seconds={key:600 for key in ('amplitude','full_H','derivatives','fisher','H_JVP','full_H_four')},
        execution_authorized=False)


def test_scope_is_true_width_full_H_zero_updates_and_explicit_release():
    plan=_plan();validate_plan(plan)
    with pytest.raises(PermissionError):validate_plan(plan,executing=True)
    for field,value in [('multiplicity_cap',20),('active_real_parameters',1495080),
                        ('complete_A1_images',16),('maximum_updates',1),('scientific_RNG_consumed',True),
                        ('full_dense_Jacobian_allowed',True),('nodelist','some-node')]:
        changed=deepcopy(plan);changed[field]=value
        with pytest.raises(ValueError):validate_plan(changed)


def test_replay_comparison_uses_vector_scale_not_unit_floor():
    assert relative_error(np.array([2e-30]),np.array([1e-30]))==1.
    with pytest.raises(ArithmeticError):relative_error([np.nan],[1.])
    with pytest.raises(ArithmeticError):relative_error([1.,2.],[1.])


def test_inherited_local_direction_and_active_root_keys_are_not_frozen_LUTs():
    cfg=MixedClebschConfig(6,9,3);profile=({0:2,2:3},{0:3,2:2},{0:1})
    layout=mixed_active_layout(cfg,profile)
    bank=dict(local=((jnp.ones((4,2)),jnp.ones((3,2))),),
        top=tuple(tuple(jnp.zeros(group['shape']) for group in row['groups']) for row in layout))
    local=inherited_local_direction(bank)
    assert sum(float(jnp.sum(abs(x))) for x in jax.tree.leaves(local))==1.
    assert float(local['local'][0][0][0,0])==1.
    root=root_direction(bank,cfg,new_copy=None,profile=profile)
    assert sum(float(jnp.sum(abs(x))) for x in jax.tree.leaves(root))==1.
    # Mismatched q widths cause different root buckets: active-key lookup is essential.
    assert len(layout[-1]['groups'])!=len(mixed_clebsch_layout(cfg)[-1]['groups'])
