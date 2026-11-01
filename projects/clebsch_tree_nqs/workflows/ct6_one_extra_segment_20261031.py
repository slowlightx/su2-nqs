"""One conditional CT6 J2=.125 continuation, not a fifth campaign segment.

The CPU-only gate consumes NO new draws/H products. It independently checks
the CURRENT segment04 terminal products before the unchanged native consumer
can prepare/replay/train/audit one additional <=32-attempt, <=24h allocation.
Learned control values, all W, moments and next RNG are inherited; the previous
monitor history is archived and only its window/streaks reset, as registered.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from math import comb
import os
from pathlib import Path
import shutil
import sys

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from . import ct6_adaptive_campaign as campaign

base = campaign.base
ROOT = Path(__file__).resolve().parents[3]
TASK = 'CT6_J0125_conditional_one_extra_32_v1'
PARENT_DIRECTORY = '/home/yx789/nqs-wavefunctions/nqs/ct6_adaptive_campaign_20261029_v2/ct6_j0125/segment04/run01'
PARENT_REGISTRATION_SHA256 = '793072088cc14cf0fc87b35a640d0183cae5452c48cf276228ac4f7cd087923c'
PARENT_JOB = 763126
TEMPLATE = ROOT/'configs/remote/ct6_adaptive_campaign_20261029_v2/ct6_j0125_segment04.json'


def make_plan() -> dict:
    """Build the separately authorized profile without widening the campaign."""
    plan = json.loads(TEMPLATE.read_text())
    for name in ('campaign_segments_per_coupling', 'campaign_A100_hours_upper_per_coupling', 'segment_index'):
        del plan[name]
    plan.update(task=TASK, parent_directory=PARENT_DIRECTORY,
        expected_parent_registration_sha256=PARENT_REGISTRATION_SHA256,
        expected_parent_job=PARENT_JOB, expected_parent_segment=4,
        automatic_next_allocation=False, no_successor=True, maximum_extra_allocations=1,
        maximum_extra_A100_hours=24, one_extra_index=1,
        parent_science_gate='current_terminal_deltaE_plus_2maxSE_negative_variance_nonharm_coverage',
        execution_authorized=True,
        authorization_scope='one_extra_32_only_if_current_segment04_terminal_improves_beyond_two_SE',
        seeds=dict(zip(base.required_seed_roles(), range(4101031001, 4101031013))))
    return plan


def validate_plan(plan: dict, *, executing: bool = False) -> None:
    """Reject broadened budgets, another parent/model, and campaign disguise."""
    expected = make_plan()
    if set(plan) != set(expected) or any(plan[k] != v for k, v in expected.items()
                                       if k != 'execution_authorized'):
        raise ValueError('exact separately authorized one-extra CT6/.125 profile required')
    if type(plan['execution_authorized']) is not bool:
        raise ValueError('explicit boolean authorization required')
    if executing and not plan['execution_authorized']:
        raise PermissionError('one-extra execution not authorized')


def independent_products(path: Path, audit: dict, plan: dict) -> dict:
    """Independent NumPy, node-faithful Q/ratios/IID and delete-128-group SE.

    Raw Hpsi is retained at psi=0; no floors, dropped rows, or real-energy
    replacement. The larger of the two paired SE estimates gates each metric.
    Ratio errors are finite-sample diagnostics, not unseen-tail certificates.
    """
    with np.load(path, allow_pickle=False) as f:
        rows, p, h, q, guide = (np.asarray(f[k]) for k in
            ('rows', 'psi', 'hpsi', 'proposal', 'guide_components'))
    n, groups = 8192, 128
    if (rows.shape != (n, 36) or rows.dtype.kind not in 'iu' or not np.isin(rows, (0, 1)).all()
            or not np.all(rows.sum(1) == 18) or p.shape != (n, 2) or h.shape != p.shape
            or q.shape != (n,) or guide.shape != (n, 2, 2)
            or not all(np.isfinite(x).all() for x in (rows, p, h, q, guide))
            or np.iscomplexobj(q) or np.any(q <= 0)):
        raise ValueError('all8192 finite Sz=0 paired node-faithful products required')
    scale = float(np.max(abs(guide)))
    if (np.max(abs(guide.imag)) > 2e-12*scale if np.iscomplexobj(guide) else False):
        raise ValueError('guide density has nonzero imaginary part')
    guide = guide.real
    if np.any(guide < 0):
        raise ValueError('nonnegative orbit/push guides required without clipping')
    uniform = plan['epsilon']/comb(36, 18)
    reconstructed = (1-plan['epsilon'])*guide.mean(axis=(1, 2))+uniform
    relative_q = float(np.max(abs(q-reconstructed)/reconstructed))
    if np.any(q < uniform) or relative_q > 2e-12:
        raise ArithmeticError('full-support Q mixture relative replay failed')
    z = abs(p)**2/q[:, None]
    numerator = p.conj()*h/q[:, None]
    second = abs(h)**2/q[:, None]
    norm = z.mean(0)
    if np.any(norm <= 0):
        raise ArithmeticError('nonzero sampled endpoint norms required')
    energy = numerator.mean(0)/norm
    h2 = second.mean(0)/norm
    residual = abs(h-energy*p)**2/q[:, None]
    point = np.stack((energy.real/36, energy.imag/36,
                      residual.mean(0)/norm/36, norm), axis=1)
    # Independent block sums give exactly the same delete-group ratio as
    # rereading all other rows, while avoiding 128 full-array materializations.
    zgroup = z.reshape(groups, n//groups, 2).sum(1)
    egroup = numerator.reshape(groups, n//groups, 2).sum(1)
    leave = []
    for group in range(groups):
        keep = np.ones(n, dtype=bool); keep[group*(n//groups):(group+1)*(n//groups)] = False
        zn = (z.sum(0)-zgroup[group])/(n-n//groups)
        en = (numerator.sum(0)-egroup[group])/(n-n//groups)/zn
        vn = (abs(h[keep]-en*p[keep])**2/q[keep, None]).mean(0)/zn
        leave.append(np.stack((en.real/36, en.imag/36, vn/36, zn), axis=1))
    leave = np.asarray(leave)
    def jackknife(values: np.ndarray) -> np.ndarray:
        return np.sqrt((groups-1)/groups*np.sum((values-values.mean(0))**2, axis=0))
    endpoint_jk = jackknife(leave)
    paired_jk = jackknife(leave[:, 1]-leave[:, 0])
    ue = (numerator-energy*z)/norm
    um = (second-h2*z)/norm
    influence = np.stack((ue.real/36, ue.imag/36,
        (um-2*np.real(energy.conj()*ue))/36, z-norm), axis=-1)
    endpoint_iid = influence.std(0, ddof=1)/np.sqrt(n)
    paired_iid = (influence[:, 1]-influence[:, 0]).std(0, ddof=1)/np.sqrt(n)
    delta = point[1]-point[0]
    def concentration(values: np.ndarray) -> dict:
        total = values.sum(0)
        if np.any(total <= 0):
            raise ValueError('nonzero norm/H2/residual contribution totals required')
        return dict(effective_contributions=(total**2/np.sum(values**2, axis=0)).tolist(),
            maximum_sample_fraction=(values.max(0)/total).tolist(),
            maximum_group_fraction=(values.reshape(groups, n//groups, 2).sum(1).max(0)/total).tolist())
    recomputed = dict(endpoint_values=point.tolist(), candidate_minus_control=delta.tolist(),
        endpoint_iid_delta_SE=endpoint_iid.tolist(), paired_iid_delta_SE=paired_iid.tolist(),
        endpoint_independent_block_jackknife_SE=endpoint_jk.tolist(),
        paired_independent_block_jackknife_SE=paired_jk.tolist(),
        norm_concentration=concentration(z), H2_concentration=concentration(second),
        residual_concentration=concentration(residual),
        exact_target_node_records=np.sum(p == 0, axis=0).tolist(),
        target_node_H2_contributions=np.where(p == 0, second, 0).sum(0).tolist(),
        norm_weight_maximum=z.max(0).tolist(), sample_records=n, independent_groups=groups)
    m = audit['moments']; errors = []
    for key, actual in recomputed.items():
        pairs = ((actual[k], m[key][k]) for k in actual) if isinstance(actual, dict) else [(actual, m[key])]
        for a, b in pairs:
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ArithmeticError('terminal moments shape/finite replay failed: '+key)
            # Relative tolerance matters for the large H²/node weights; the
            # separate Q check above is strictly relative even at 1e-11.
            err = float(np.max(abs(a-b)/np.maximum(1., abs(a))))
            errors.append(err)
            if err > 2e-10:
                raise ArithmeticError('independent terminal moment replay failed: '+key)
    se = np.maximum(paired_iid, paired_jk)
    if not delta[0]+2*se[0] < 0:
        raise ValueError('current terminal energy gain is not beyond2maxSE')
    if delta[2]-2*se[2] > 0:
        raise ValueError('current terminal variance resolves harm beyond2maxSE')
    if not base.monitor_coverage(plan, recomputed):
        raise ValueError('independently recomputed terminal coverage fails')
    return dict(status='passed', delta=delta.tolist(), paired_iid_SE=paired_iid.tolist(),
        paired_delete128_group_SE=paired_jk.tolist(), paired_maximum_SE=se.tolist(),
        energy_upper_two_SE=float(delta[0]+2*se[0]),
        variance_lower_two_SE=float(delta[2]-2*se[2]),
        maximum_scaled_moment_replay_error=max(errors), maximum_relative_Q_error=relative_q,
        recomputed=recomputed, node_records_retained=True, new_draws=0, new_H_centers=0,
        normalized_ratios_have_finite_sample_bias=True, unseen_tail_precision_not_certified=True)


def _prefix_safety(src: Path, progress: dict, frozen: dict) -> str:
    """Allow a resource-bounded prefix, never bypass a scientific/partial stop."""
    attempts = progress['attempts']; reason = frozen.get('reason')
    if reason != progress.get('stop_reason'):
        raise ValueError('frozen/committed stop reason differs')
    if any(p.is_dir() and p.name[4:].isdigit() and int(p.name[4:]) > attempts
           for p in src.glob('step*')):
        raise ValueError('uncommitted attempted streams prohibit inherited RNG reuse')
    if attempts == 32 and reason is None:
        return 'completed_registered32_attempts'
    if reason == 'wall_gate_terminal_and_monitor_reserves_preserved':
        return 'clean_resource_bounded_prefix'
    if reason == 'independent_monitor_coverage_or_reserved_resource_gate' and progress['monitor_history']:
        decision = progress['monitor_history'][-1]; m = decision['monitor']
        adverse = (m['delta_energy']-2*m['energy_se'] > 0 or
                   m['delta_variance']-2*m['variance_se'] > 0)
        resource = base.SRResourceEnvelope(**decision['resources'])
        if (m['coverage_ok'] is True and m['geometry_ok'] is True and not adverse
                and decision['reason'] not in ('geometry_failure', 'resolved_harm',
                    'missed_resolved_prediction', 'coverage_not_verified_hold_and_diagnose')
                and decision['can_train_next_block'] is False
                and not resource.permits(decision['next_controls']['fit_records'])):
            return 'clean_monitor_reserved_resource_bounded_prefix'
    raise ValueError('unknown/scientific/partial parent stop cannot authorize another allocation')


def inspect_parent(plan: dict) -> dict:
    """Read only immutable completed terminal evidence; no device H or draws."""
    validate_plan(plan, executing=True)
    src = Path(plan['parent_directory'])
    names = ['registration.json', 'checkpoint.pkl', 'initial.pkl', 'parent.pkl',
             'audit_result.json', 'endpoint_frozen.json', 'terminal_audit/audit.json',
             'terminal_audit/products.npz', 'terminal_audit/draw.npz', 'terminal_audit/draw.json']
    if file_sha256(src/'registration.json') != plan['expected_parent_registration_sha256']:
        raise ValueError('fixed current segment04 registration SHA differs')
    reg = json.loads((src/'registration.json').read_text()); old = reg['plan']
    campaign.validate_plan(old, executing=True)
    if (old['segment_index'] != 4 or old['j2'] != .125 or old['model'] != plan['model']
            or old['arity'] != plan['arity'] or old['binary_rank'] != plan['binary_rank']):
        raise ValueError('only current binaryw28 J2=.125 campaign segment04 is eligible')
    for name, digest in reg['parent_seals'].items():
        path = src/'parent_snapshot'/name
        if file_sha256(path) != digest:
            raise ValueError('parent snapshot lineage seal mismatch: '+name)
        names.append('parent_snapshot/'+name)
    for name, digest in reg['sources'].items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError('inherited production source changed: '+name)
    seals = {name: file_sha256(src/name) for name in names}
    state, controls, progress = base.load_adaptive_state(src/'checkpoint.pkl', plan=old, registration=reg)
    initial, initial_controls, initial_progress = base.load_adaptive_state(src/'initial.pkl', plan=old, registration=reg)
    bound, bound_meta = base.legacy.load_state(src/'parent.pkl')
    snapshot_reg = json.loads((src/'parent_snapshot/registration.json').read_text())
    snapshot, snapshot_controls, snapshot_progress = base.load_adaptive_state(
        src/'parent_snapshot/checkpoint.pkl', plan=snapshot_reg['plan'], registration=snapshot_reg)
    if (pytree_sha256(initial[:2]) != pytree_sha256(bound[:2])
            or pytree_sha256(initial[:2]) != pytree_sha256(snapshot[:2])
            or canonical_sha256(initial[2]) != canonical_sha256(bound[2])
            or canonical_sha256(initial[2]) != canonical_sha256(snapshot[2])
            or initial_progress != base.initial_progress()
            or bound_meta['absolute_completed_steps'] != reg['parent_global_step']
            or reg['parent_global_step'] != snapshot_reg['parent_global_step']+snapshot_progress['accepted_updates']
            or reg['original_parameter_sha256'] != pytree_sha256(initial[0])
            or reg['original_optimizer_sha256'] != pytree_sha256(initial[1])
            or reg['original_rng_sha256'] != canonical_sha256(initial[2])
            or initial[2]['next_step'] != reg['initial_native_cursor']
            or any(getattr(initial_controls, k) != getattr(snapshot_controls, k)
                   for k in ('fs_radius', 'damping', 'fit_records'))
            or initial_controls.looks != 0):
        raise ValueError('terminal audit control/initial native state lineage mismatch')
    final = json.loads((src/'audit_result.json').read_text())
    frozen = json.loads((src/'endpoint_frozen.json').read_text())
    audit = json.loads((src/'terminal_audit/audit.json').read_text())
    attempts = progress['attempts']; accepted = progress['accepted_updates']
    hashes = dict(parameter_sha256=pytree_sha256(state[0]), optimizer_sha256=pytree_sha256(state[1]),
        rng_sha256=canonical_sha256(state[2]), checkpoint_sha256=seals['checkpoint.pkl'],
        controller_sha256=base.sr_control_checkpoint(base.policy_from_plan(old), controls)['sha256'])
    if (not 1 <= attempts <= 32 or not 1 <= accepted <= attempts
            or final.get('status') != 'complete_fixed_prefix_independent_audit'
            or frozen.get('status') != 'fixed_attempt_prefix_frozen'
            or frozen.get('requested_attempts') != 32
            or frozen.get('fixed_attempt_budget_completed') is not (attempts == 32)
            or final.get('independent_from_training_geometry_and_adaptation') is not True
            or frozen.get('no_best_endpoint_selection') is not True
            or final.get('terminal_audit') != audit
            or any(final.get(k) != v or frozen.get(k) != v for k, v in hashes.items())
            or any(d.get('attempts') != attempts or d.get('accepted_updates') != accepted
                   or d.get('absolute_completed_steps') != reg['parent_global_step']+accepted
                   for d in (final, frozen))
            or frozen.get('native_next_cursor') != state[2]['next_step']
            or audit.get('status') != 'complete' or audit.get('records') != 8192
            or audit.get('moments', {}).get('independent_groups') != 128
            or audit.get('seed') != old['seeds']['audit']
            or audit.get('statistics_seed') != old['seeds']['audit_statistics']
            or audit.get('endpoint_params_sha256') != hashes['parameter_sha256']
            or audit.get('comparison_parent_global_step') != reg['parent_global_step']
            or audit.get('no_endpoint_selection') is not True
            or audit.get('full_support_epsilon') != .125):
        raise ValueError('completed current terminal products/state/controller identity required')
    prefix = _prefix_safety(src, progress, frozen)
    draw_receipt = json.loads((src/'terminal_audit/draw.json').read_text())
    before, after = draw_receipt['rng_before'], draw_receipt['rng_after']
    if (draw_receipt.get('records') != 8192 or draw_receipt.get('n_sites') != 36
            or draw_receipt.get('uniform_fraction') != .125
            or draw_receipt.get('empirical_branch_weights_used') is not False
            or before['records'] != 0 or before['batches'] != 0
            or after['records'] != 8192 or after['batches'] != 1
            or any(x.get('n_sites') != 36 or x.get('epsilon') != .125 for x in (before, after))
            or set(before['stream_identities']) != {'branch', 'base', 'uniform'}
            or before['stream_identities'] != after['stream_identities']
            or any(before['stream_identities'][role]['entropy'] != old['seeds']['audit']
                   or before['stream_identities'][role]['spawn_key'] != [index]
                   or before['stream_identities'][role]['pool_size'] != 4
                   for index, role in enumerate(('branch', 'base', 'uniform')))):
        raise ValueError('terminal draw stream/count/fixed-proposal provenance differs')
    with np.load(src/'terminal_audit/draw.npz', allow_pickle=False) as draw, np.load(
            src/'terminal_audit/products.npz', allow_pickle=False) as products:
        if not np.array_equal(draw['rows'], products['rows']):
            raise ValueError('all terminal drawn rows must match node-faithful H products')
    science = independent_products(src/'terminal_audit/products.npz', audit, plan)
    future = deepcopy(state[2]); history = set(reg.get('campaign_seed_history', [])) | set(old['seeds'].values())
    used = history | set(future['consumed_role_seeds']) | set(future['excluded_seeds'])
    roles_preview = []
    for index in range(future['next_step'], future['next_step']+32):
        roles, future = base.c6.role_seeds(future, index)
        if set(roles.values()) & history:
            raise ValueError('inherited future RNG collides with historical role seeds')
        used.update(roles.values()); roles_preview.append(roles)
    if used & set(plan['seeds'].values()):
        raise ValueError('new one-extra role seeds collide with inherited/future streams')
    if {name: file_sha256(src/name) for name in names} != seals:
        raise ValueError('parent moved while reading terminal evidence')
    return dict(status='passed', task=TASK, plan_sha256=canonical_sha256(plan),
        source_sha256=file_sha256(Path(__file__)), code_commit=os.environ.get('NQS_CODE_COMMIT'),
        parent_directory=str(src), parent_job=PARENT_JOB, parent_files=seals,
        parent_attempts=attempts, parent_accepted_updates=accepted,
        parent_global_step=reg['parent_global_step']+accepted, prefix_classification=prefix,
        native_next_cursor=state[2]['next_step'], inherited_state_hashes=hashes,
        inherited_control_values={k:getattr(controls, k) for k in ('fs_radius', 'damping', 'fit_records')},
        first32_native_roles_sha256=canonical_sha256(roles_preview), independent_review=science,
        new_draws=0, new_H_centers=0, no_successor=True,
        terminal_audit_consumed_for_one_extra_authorization_not_untouched_future_precision=True)


def checked_gate(plan: dict, output: Path) -> dict:
    """Persist one CPU receipt; never retry a refusal or modify a prior seal."""
    path = output/'parent_gate.json'
    if path.exists():
        gate = json.loads(path.read_text())
        if gate.get('status') != 'passed':
            return gate
        if (gate['plan_sha256'] != canonical_sha256(plan)
                or gate['source_sha256'] != file_sha256(Path(__file__))
                or gate['code_commit'] != os.environ.get('NQS_CODE_COMMIT')
                or any(file_sha256(Path(plan['parent_directory'])/name) != digest
                       for name, digest in gate['parent_files'].items())):
            raise ValueError('previous CPU gate/source/parent seal changed')
        return gate
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('new one-extra output root required before CPU gate')
    output.mkdir(parents=True, exist_ok=True)
    try:
        gate = inspect_parent(plan)
    except (OSError, ValueError, ArithmeticError, KeyError, TypeError) as error:
        gate = dict(status='refused_no_training', task=TASK, plan_sha256=canonical_sha256(plan),
            parent_directory=plan['parent_directory'], error=str(error),
            new_draws=0, new_H_centers=0, no_successor=True)
    atomic_json(path, gate)
    return gate


def bind_parent(plan: dict, out: Path) -> dict:
    """Strong gate, then reuse unchanged exact native-state parent binding."""
    gate = checked_gate(plan, out.parent)
    if gate['status'] != 'passed':
        raise PermissionError('CPU independent terminal gate has not passed')
    binding = campaign.bind_parent(plan, out)
    snapshot = out/'parent_snapshot'
    shutil.copy2(out.parent/'parent_gate.json', snapshot/'one_extra_parent_gate.json')
    for name in ('initial.pkl', 'parent.pkl'):
        shutil.copy2(Path(plan['parent_directory'])/name, snapshot/('control_'+name))
    binding['parent_seals'].update({p.name:file_sha256(p) for p in snapshot.iterdir()})
    binding.update(one_extra_authorization=gate, no_successor=True)
    return binding


initial_controls = campaign.initial_controls
resources_for_next_block = campaign.resources_for_next_block


def run(plan_path: Path, output: Path, phase: str) -> dict:
    """Gate before all GPU phases; failed dependency is a clean no-work exit."""
    plan = json.loads(plan_path.read_text()); validate_plan(plan, executing=True)
    gate = checked_gate(plan, output)
    if phase == 'gate' or gate['status'] != 'passed':
        return gate
    if phase not in ('prepare', 'replay', 'train', 'audit'):
        raise ValueError('unknown one-extra phase')
    return base.run(plan_path, output/'run01', phase, profile=sys.modules[__name__])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('gate', 'prepare', 'replay', 'train', 'audit'), required=True)
    args = parser.parse_args(); jax.config.update('jax_enable_x64', True)
    result = run(args.plan, args.output, args.phase)
    print(json.dumps({k:result[k] for k in ('status', 'attempts', 'accepted_updates', 'new_draws', 'new_H_centers')
                      if k in result}), flush=True)


if __name__ == '__main__':
    main()
