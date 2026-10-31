"""Bind completed frozen confirmation, phase reviews, costs and validation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import jax
import numpy as np

from fnqs.artifacts import write_json
from fnqs.initialization import initialize_run
from fnqs.models import GRUConfig
from fnqs.pilots.lineage import ROOT, inclusive_cost, read_node, sha
from fnqs.scripts.run_frozen_confirmation import verified


def read(path: Path) -> dict:
    """Read a completed JSON record without changing its source artifact."""
    return json.loads(path.read_text())


def record(output: Path) -> dict:
    """Verify sealed inputs and assemble a descriptive, post-confirmation receipt."""
    protocol = verified(output)
    pipeline = read(output / 'pipeline_complete.json')
    assert pipeline['complete']
    for name, digest in pipeline['products'].items():
        assert sha(output / name) == digest
    seal = read(output / 'endpoint_seal.json')
    assert seal['confirmation_references_opened'] is False
    confirmation = read(output / 'confirmation.json')
    scaling = read(output / 'scaling_stability.json')
    replay = read(output / 'independent_replay.json')
    response = read(output / 'response_supplement/review.json')
    peaks = read(output / 'peak_reference_check.json')
    assert replay['complete'] and len(confirmation['rows']) == 9

    endpoints = [output / 'runs' / f'L{n}_seed_{seed}' / 'fast/node.json'
                 for seed in protocol['seeds'] for n in protocol['sizes']]
    training = inclusive_cost(endpoints)
    assert training == confirmation['training_cost'] == read(output / 'training_receipt.json')['cost']
    seen = set()
    residuals = []
    for endpoint in endpoints:
        pending = [endpoint]
        while pending:
            path = pending.pop().resolve()
            if path in seen:
                continue
            seen.add(path)
            read_node(path)
            node = read(path)
            residuals += [float(row['true_sr_residual']) for row in node['history']]
            pending += [Path(parent['path']) for parent in node['parents']]
    assert len(seen) == 27 and max(residuals) <= 1e-6

    audit_cost = dict(born_samples=0, basis_rows=0, hamiltonian_actions=0, source_actions=0)
    all_points = []
    held_points = []
    energy_rows = fd_rows = 0
    for endpoint in endpoints:
        points = read(endpoint.parent.parent / 'audit.json')['points']
        assert len(points) == 41
        all_points += points
        held_points += [point for point in points if point['split'] == 'reserved_confirmation']
        for point in points:
            for key in audit_cost:
                audit_cost[key] += int(point[key])
            if point['born_samples']:
                energy_rows += point['energy_rows']
                fd_rows += point['fd_rows']
    assert len(held_points) == 360 and len(all_points) == replay['points'] == 369
    audit_cost.update(local_energy_rows=energy_rows, fd_rows=fd_rows)
    worst = max(held_points, key=lambda point: point['chi_upper_error'])

    # Reconstruct the three pre-training initializations; these are model-only calls.
    initializations = []
    for seed in protocol['seeds']:
        params, _, _key, _ = initialize_run(seed, GRUConfig(16, 64))
        digest = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(params):
            array = np.asarray(leaf)
            digest.update(str((array.shape, str(array.dtype))).encode())
            digest.update(array.tobytes())
        initializations.append(dict(seed=seed, parameter_sha256=digest.hexdigest()))
    assert len({item['parameter_sha256'] for item in initializations}) == 3

    checks = ROOT / 'fnqs/results/phase_checks_v18d'
    phase_receipt = read(checks / 'receipt.json')
    for stage in (1, 2):
        assert sha(checks / f'phase{stage}.json') == phase_receipt[f'phase{stage}_sha256']
    assert sha(ROOT / 'fnqs/scripts/check_other_phases.py') == phase_receipt['source_sha256']
    phase1, phase2 = read(checks / 'phase1.json'), read(checks / 'phase2.json')
    assert phase1['artifact_integrity_pass'] and phase2['artifact_integrity_pass']
    assert phase2['transitive_holdouts_pass'] and phase2['matched_adaptation_budget_pass']
    tests = read(ROOT / 'fnqs/results/test_validation_v18.json')
    for name, digest in tests['log_sha256'].items():
        assert sha(ROOT / 'fnqs/results' / name) == digest

    # The original scheduler's accepted-node pass rewrote timing.json with skip
    # times. Recover actual elapsed markers from untouched training logs instead.
    log_paths = [ROOT / 'fnqs/results/frozen_training_v18.log'] + [
        ROOT / f'fnqs/results/parallel_seed_{seed}_v18.log' for seed in protocol['seeds'][1:]]
    timings = {}
    for path in log_paths:
        for run, seconds in re.findall(r'(L\d+_seed_\d+) fast 100 seconds ([0-9.]+)', path.read_text()):
            assert run not in timings
            timings[run] = float(seconds)
    assert len(timings) == 9
    pooled = next(row for row in scaling['fits'] if row['seed'] == 'pooled')
    response_values = [row['response_relative_residual'] for row in response['rows']]
    profile = read(ROOT / 'fnqs/results/audit_resource_profile_v18.json')

    bindings = [output / name for name in pipeline['products']] + [
        output / 'protocol.json', output / 'endpoint_seal.json', output / 'training_receipt.json',
        output / 'pipeline_declaration.json', output / 'pipeline_complete.json',
        output / 'confirmation_profiles.png', output / 'scaling_and_peak_stability.png',
        output / 'response_supplement/protocol.json',
        checks / 'receipt.json', checks / 'phase1.json', checks / 'phase2.json',
        ROOT / 'fnqs/results/test_validation_v18.json',
        ROOT / 'fnqs/results/audit_resource_profile_v18.json']
    bindings += list(output.glob('parallel_seed_*/protocol.json')) + log_paths
    result = dict(
        kind='completed_frozen_confirmation_review_v18', confirmation_work_complete=True,
        phase0_scientific_gate=False,
        original_gate=scaling['original_gate'],
        additional_stability=dict(window_and_fit_form_stable=scaling['window_and_fit_form_stable'],
            minimum_size_drop_check_available=scaling['minimum_size_drop_check_available'],
            common_systematic_bias_not_in_bootstrap=True),
        independent_initializations=dict(count=3, records=initializations,
            paired_across_sizes=True, independent_optimization_per_size=True,
            parent_parameters_from_method_selection=False, model_only_verification=True),
        observable_confirmation=confirmation['rows'],
        pooled_descriptive_fit=dict(hc=pooled['primary']['hc'], nu=pooled['primary']['nu'],
            eta=pooled['eta']['eta'], hc_spread=pooled['hc_spread'],
            nu_spread=pooled['nu_spread'], eta_spread=pooled['eta_spread'],
            reliable_universality_estimate=False),
        worst_chi_point={key: worst[key] for key in ('n', 'couplings', 'chi_relative_error',
            'chi_upper_error', 'chi_se', 'reference_chi_total')},
        response_relative_residual_range=[min(response_values), max(response_values)],
        response_residual_is_not_chi_accuracy_bound=True,
        peak_reference_comparisons=peaks['comparisons'],
        independent_replay={key: value for key, value in replay.items()
                            if key != 'validated_sha256'},
        actual_training_cost=training,
        true_sr_residual_max=max(residuals), acceptance_guard=1e-6,
        all_sr_solves_meet_requested_1e_minus_7=all(value <= 1e-7 for value in residuals),
        separate_validation_costs=dict(initial_confirmation=audit_cost,
            physical_response_supplement=response['audit_cost'], independent_replay=replay['cost'],
            phase1_review=phase1['validation_cost'], phase2_review=phase2['validation_cost'],
            resource_profile={key: profile[key] for key in
                              ('born_samples', 'hamiltonian_actions', 'source_actions')}),
        cost_convention='logical operator actions, not FLOPs; old Phase 1/2 training is not new training',
        actual_training_elapsed_log_markers=dict(seconds_by_run=timings,
            fair_wall_benchmark=False, includes_compilation=True, concurrent_tests_and_workers=True,
            timing_json_after_skip_is_not_training_time=True),
        phase1={key: value for key, value in phase1.items()
                if key not in ('validated_sha256', 'rows')},
        phase2={key: value for key, value in phase2.items()
                if key not in ('validated_sha256', 'rows')},
        test_validation=tests,
        source_sha256=sha(Path(__file__)),
        input_sha256={str(path.relative_to(ROOT)): sha(path) for path in bindings},
        next_step='new development protocol for fixed L32 response convergence, then fresh frozen confirmation',
        v18_reserved_fields_available_for_future_confirmation=False,
        existing_artifacts_resealed=False)
    assert not scaling['phase0_scientific_gate']
    assert not phase1['science_pass'] and not phase2['science_pass']
    runtime = ROOT / 'fnqs/results/runtime_v18_analysis'
    runtime.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__), runtime / Path(__file__).name)
    write_json(ROOT / 'fnqs/results/progression_v18.json', result)
    write_json(runtime / 'manifest.json', dict(source_sha256=result['source_sha256'],
        progression_sha256=sha(ROOT / 'fnqs/results/progression_v18.json'),
        post_confirmation_aggregation_only=True, new_training=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'fnqs/results/phase0_frozen_confirmation_v18')
    args = parser.parse_args()
    result = record(args.output.resolve())
    print(json.dumps(dict(complete=result['confirmation_work_complete'],
                          phase0_scientific_gate=result['phase0_scientific_gate'],
                          phase1_integrity=result['phase1']['artifact_integrity_pass'],
                          phase2_integrity=result['phase2']['artifact_integrity_pass'])))


if __name__ == '__main__':
    main()
