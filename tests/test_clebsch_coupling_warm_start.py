"""Cross-H warm starts must not masquerade as complete same-H continuations."""
import copy

import pytest
from pathlib import Path
import json

from su2_nqs.frontier.artifacts import pytree_sha256
from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_sampled import cost
from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_training import train
from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_audit import audit
from su2_nqs.vmc.checkpoint import load_checkpoint

from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_sampled import validate_initialization_coupling


def test_only_explicit_cross_H_parameter_import_is_allowed():
    target = dict(geometry='square', j2_over_j1=0.)
    parent = dict(geometry='square', j2_over_j1=.5)
    declaration = dict(parent_j2_over_j1=.5, target_j2_over_j1=0., reset_optimizer_sampler_rng=True)
    validate_initialization_coupling(target, parent, dict(coupling_warm_start=declaration))
    for broken in ({}, dict(coupling_warm_start={}),
                   dict(coupling_warm_start=dict(declaration, reset_optimizer_sampler_rng=False)),
                   dict(coupling_warm_start=dict(declaration, parent_j2_over_j1=.125))):
        with pytest.raises(ValueError):
            validate_initialization_coupling(target, parent, broken)
    with pytest.raises(ValueError):
        validate_initialization_coupling(target, dict(parent, geometry='triangular'),
                                        dict(coupling_warm_start=declaration))


def test_same_H_transfer_remains_unchanged_and_cannot_claim_cross_H_reset():
    spec = dict(geometry='square', j2_over_j1=0.)
    validate_initialization_coupling(spec, copy.deepcopy(spec), {})
    with pytest.raises(ValueError):
        validate_initialization_coupling(spec, spec, dict(coupling_warm_start={}))


def test_actual_cross_H_cost_training_and_unreferenced_audit(tmp_path):
    spec = dict(geometry='square', rows=2, columns=2, j2_over_j1=.5,
                multiplicity_cap=2, maximum_q=2, projection_chunk=2,
                require_gpu=False, seed=120, calibration_seed=121, sampling_seed=122)
    parent = tmp_path/'parent'
    gate = cost(spec, parent)
    source = load_checkpoint(parent/'checkpoint.pkl')
    target = dict(spec, j2_over_j1=0., seed=130, calibration_seed=131, sampling_seed=132,
        initialization_parent=dict(directory=str(parent), registration_sha256=gate['registration_sha256'],
            checkpoint_sha256=gate['checkpoint_sha256'], params_sha256=pytree_sha256(source['params']),
            coupling_warm_start=dict(parent_j2_over_j1=.5, target_j2_over_j1=0.,
                                     reset_optimizer_sampler_rng=True)))
    new_gate = cost(target, tmp_path/'new_cost')
    assert new_gate['status'] == 'passed'
    assert new_gate['validation']['H_relative_error'] < 2e-11
    target.update(cost_parent=dict(directory=str(tmp_path/'new_cost'),
        checkpoint_sha256=new_gate['checkpoint_sha256'], registration_sha256=new_gate['registration_sha256']),
        training_seed=133, chains=4, steps=1, sr_warm_steps=1, warm_samples=8, adam_samples=8,
        burn_proposals=8, proposals_between_snapshots=4, score_chunk=1, H_sample_chunk=4,
        sr_damping=.01, maximum_FS_sine=.05, warm_lr=.005, adam_lr=.001,
        reset_moments_at_phase_switch=True, milestones=[0, 1], maximum_training_seconds=120)
    trained = train(target, tmp_path/'training', 1, 120)
    assert trained['completed_steps'] == 1
    cp = load_checkpoint(tmp_path/'training/checkpoint.pkl')
    assert cp['metadata']['completed_steps'] == 1
    measured = audit(dict(training_directory=str(tmp_path/'training'), endpoint_step=1,
        require_gpu=False, streams=2, chains_per_stream=4, snapshots_per_stream=2,
        burn_proposals=8, proposals_between_snapshots=4, audit_seeds=[141, 142],
        bootstrap_seed=143, H_sample_chunk=4), tmp_path/'audit')
    assert measured['statistics']['literature_energy_per_site'] is None
    assert measured['statistics']['energy_minus_literature_per_site'] is None
