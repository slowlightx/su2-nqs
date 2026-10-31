"""Bounded memory-route relaunch: immutable parents, real lazy NTK and replay."""
from copy import deepcopy
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import (
    initial_model, validate_spec,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'projects/clebsch_tree_nqs/configs/remote/vit_relaunch_20261022'


@pytest.mark.parametrize('case', ['j0', 'j0125'])
def test_vit6_relaunch_changes_runtime_not_failed_expansion_physics(case):
    old = json.loads((BASE.parent/'vit6_gcnn_quality_20261021'/f'{case}.json').read_text())
    new = json.loads((BASE/f'vit6_{case}.json').read_text())
    validate_spec(new)
    for key in ('model', 'model_family', 'coordinate_convention', 'spin_inversion_even',
                'j2_over_j1', 'sampling', 'sr', 'repair_parameters_from', 'capacity_expansion'):
        assert new[key] == old[key]
    assert new['output'] != old['output']
    assert old['netket']['on_the_fly'] is False
    assert new['netket'] == dict(old['netket'], on_the_fly=True)
    assert new['stages'] == [dict(projection='gamma_a1', steps=16)]
    assert new['quality_milestones'] == [0, 2, 4, 8, 16]
    assert not set(old['audit']['seeds']) & set(new['audit']['seeds'])
    assert new['budget_release']['new_training_full_H_calls'] == 16*8192
    assert not new['budget_release']['fully_matrix_free_claim']


@pytest.mark.parametrize('case,boundary,extra', [('j0', 32, 16), ('j0125', 160, 64)])
def test_vit12_relaunch_preserves_full_state_protocol(case, boundary, extra):
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket import validate_continuation_protocol
    plan = json.loads((BASE/f'vit12_{case}_plan.json').read_text())
    old = json.loads((ROOT/f'projects/clebsch_tree_nqs/results/remote_review_20261021/vit12_{case}/registration.json').read_text())
    assert plan['parent']['registration_sha256'] == old['result_sha256']
    child = deepcopy(old['config']); child.update(deepcopy(plan['child_overrides']))
    child['netket_continue_from'] = dict(plan['continuation'], completed_steps=boundary)
    validate_continuation_protocol(old['config'], child)
    assert plan['parent']['completed_steps'] == boundary
    assert sum(s['steps'] for s in child['stages']) == boundary+extra
    assert child['model']['mlp_ratio'] == 4
    assert child['sampling']['chains']*child['sampling']['snapshots_per_update'] == 4096


def test_relaunch_wrapper_has_no_submission_or_automatic_long_release():
    text = (BASE/'run.sbatch').read_text()
    assert 'sbatch ' not in text and 'scancel ' not in text
    assert 'srun ' not in text
    assert '--resume --until-step' in text
    assert 'peak_bytes' in text and 'unique_local_sample_partitions' in text
    assert "s['netket']['on_the_fly'] is True" in text


def test_lazy_full_A1_NTK_matches_explicit_and_replays_complete_update(tmp_path):
    nk = pytest.importorskip('netket')
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket import (
        make_driver, full_update, driver_payload, restore_driver,
    )
    from su2_nqs.vmc.checkpoint import save_checkpoint, load_checkpoint
    jax.config.update('jax_enable_x64', True)
    spec = dict(task='spin_spatial_vit_quality', preregistered_before_training=True,
        backend='netket_ntk', model_family='literature', spin_inversion_even=True,
        j2_over_j1=.125, initialization_seed=201, training_seed=211,
        model=dict(geometry='square', rows=2, columns=2, patch_rows=1, patch_columns=1,
                   embedding_dim=4, head_count=2, depth=1, mlp_ratio=4),
        stages=[dict(projection='gamma_a1', steps=3)], quality_milestones=[0, 3],
        maximum_training_seconds=120., maximum_audit_seconds_per_checkpoint=120.,
        sampling=dict(chains=4, snapshots_per_update=4, proposals_between_snapshots=4, burn_proposals=8),
        audit=dict(seeds=[221, 231], chains=4, snapshots=4, proposals_between_snapshots=4,
                   burn_proposals=8, query_batch=4), sr=dict(damping=.01, step_size=.005),
        netket=dict(chunk_size=4, chunk_size_bwd=1, on_the_fly=True,
            initialization='balanced_spin_half', exchange_kernel='uniform_all_pairs',
            replicate_initial_params=True,
            sampled_trust=dict(maximum_fs_sine=.001, minimum_ess_fraction=.95, halvings=8)),
        momenta=dict(Gamma=[0, 0]), regions=dict(compact4=dict(sites=[0, 1, 2, 3], left=[0, 1])))
    assert nk.__version__ == '3.22.3'
    params, _ = initial_model(spec)
    lazy, _, _, images = make_driver(spec, params, 'gamma_a1')
    explicit_spec = deepcopy(spec); explicit_spec['netket']['on_the_fly'] = False
    explicit, _, _, _ = make_driver(explicit_spec, params, 'gamma_a1')
    assert lazy.use_ntk and lazy.on_the_fly
    assert explicit.use_ntk and not explicit.on_the_fly
    assert images > 1
    left = full_update(lazy, spec=spec); right = full_update(explicit, spec=explicit_spec)
    assert left['energy'] == pytest.approx(right['energy'], abs=2e-12)
    for a, b in zip(jax.tree_util.tree_leaves(lazy.state.parameters),
                    jax.tree_util.tree_leaves(explicit.state.parameters), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=0)
    state, optimizer = driver_payload(lazy)
    path = tmp_path/'lazy.pkl'
    save_checkpoint(path, state, optimizer_state=optimizer, metadata={})
    replay, _, _, _ = make_driver(spec, params, 'gamma_a1', step=1)
    restore_driver(replay, load_checkpoint(path))
    before_rng = jax.random.key_data(lazy.state.sampler_state.rng)
    np.testing.assert_array_equal(before_rng, jax.random.key_data(replay.state.sampler_state.rng))
    left = full_update(lazy, spec=spec); right = full_update(replay, spec=spec)
    assert left['energy'] == pytest.approx(right['energy'], abs=2e-12)
    for a, b in zip(jax.tree_util.tree_leaves(lazy.state.parameters),
                    jax.tree_util.tree_leaves(replay.state.parameters), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=0)
    np.testing.assert_array_equal(lazy.state.sampler_state.σ, replay.state.sampler_state.σ)
    np.testing.assert_array_equal(jax.random.key_data(lazy.state.sampler_state.rng),
                                 jax.random.key_data(replay.state.sampler_state.rng))
