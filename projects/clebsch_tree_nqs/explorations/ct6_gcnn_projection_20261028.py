"""Conditional CT6 time projection from independent paired terminal audits.

This is a scenario calculation, not a convergence guarantee or a predictive
confidence interval. No training means are fitted, and no inverse-variance
average of different endpoint absolute energies is reported as a new state.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT/'projects/clebsch_tree_nqs/results'
NEW = RESULTS/'ct_sector_excitation_pilot_20261028/completed_CT6'
OUTPUT = RESULTS/'ct6_adaptive_production_20261028'


def run() -> dict:
    """Project a constant recent rate, then explicitly halve it as stress test."""
    old_base = RESULTS/'ct_status_20261028/ct6_independent_review'
    latest = json.loads((NEW/'independent_review.json').read_text())
    records = {}
    sources = [NEW/'independent_review.json']
    for tag, previous_name, previous_steps, gcnn_se in (
        ('ct6_j0', 'ct6_j0_independent.json', 8, .000003),
        ('ct6_j0125', 'ct6_j0125_old_independent.json', 16, .000009),
    ):
        previous_path = old_base/previous_name
        sources.append(previous_path)
        previous = json.loads(previous_path.read_text())
        now = latest['runs'][tag]
        stage_n = np.asarray((previous_steps, 8), float)
        changes = np.asarray((previous['moments']['candidate_minus_control'][0], now['paired_energy_change']))
        errors = np.asarray((previous['moments']['paired_independent_block_jackknife_SE'][0],
                             now['paired_energy_change_SE']))
        rates, rate_se = -changes/stage_n, errors/stage_n
        weights = 1/rate_se**2
        recent_rate = float(np.dot(weights, rates)/weights.sum())
        recent_se = float(1/np.sqrt(weights.sum()))
        transitions = [json.loads((NEW/tag/f'step{i}/transition.json').read_text()) for i in range(1, 9)]
        sources.extend(NEW/tag/f'step{i}/transition.json' for i in range(1, 9))
        warm_seconds = float(np.median([item['wall_seconds'] for item in transitions[1:]]))
        gap = now['energy_per_site']-now['gcnn_energy_per_site']
        rate_band = [max(recent_rate-1.96*recent_se, 0.), recent_rate+1.96*recent_se]
        step_band = [float(gap/rate_band[1]), float(gap/rate_band[0]) if rate_band[0] else None]
        central_steps = float(gap/recent_rate)
        records[tag] = dict(latest_step=now['absolute_step'], latest_energy_per_site=now['energy_per_site'],
            latest_energy_SE=now['energy_block_SE'], target_gcnn_mean=now['gcnn_energy_per_site'],
            published_gcnn_SE=gcnn_se, remaining_gap_to_central_value=gap,
            paired_stages_steps=stage_n.tolist(), paired_stage_changes=changes.tolist(),
            paired_stage_SE=errors.tolist(), recent_gain_per_step=rates.tolist(),
            recent_gain_per_step_SE=rate_se.tolist(), pooled_recent_rate=recent_rate,
            pooled_rate_SE=recent_se, conditional_rate_only_band=rate_band,
            constant_recent_rate_steps_to_target=central_steps,
            latest_eight_step_rate_only_steps_to_target=float(gap/rates[-1]),
            latest_eight_step_rate_total_hours_with_25percent_overhead=float(
                1.25*gap/rates[-1]*warm_seconds/3600),
            conditional_rate_only_steps_band=step_band,
            median_warm_update_seconds=warm_seconds,
            constant_rate_update_GPU_hours=central_steps*warm_seconds/3600,
            constant_rate_total_hours_with_25percent_overhead=1.25*central_steps*warm_seconds/3600,
            half_rate_stress_total_hours_with_25percent_overhead=2.5*central_steps*warm_seconds/3600,
            proposed_next_fixed_attempts=32, proposed_allocation_hours=24,
            extrapolation_not_validated=True)
    report = dict(task='conditional_recent_paired_audit_CT6_GCNN_projection', runs=records,
        method='two consecutive disjoint independent paired audit stages, inverse-variance rate heuristic',
        limitation='rate-only band omits endpoint energy uncertainty, rate evolution, algorithmic bias and capacity floors; not a predictive interval',
        mixed_protocol_note='J.125 older stage used4096 records/ridge.5, latest used5120/ridge.05; pooled rate is only a scenario heuristic, not a stationary optimization-law fit. Latest8-only scenario is reported separately.',
        target_precision_note='time projection concerns reaching the published central energy, not reproducing the much smaller published statistical error; a precision-matched audit needs a separate budget',
        historical_128_step_note='older128-step width28 audits did not resolve energy gain versus width20; do not assume recent IID-SR rate persists indefinitely',
        no_new_state_energy_created=True, no_training_mean_fit=True,
        proposed_total_requested_A100_hours=48, no_automatic_150_step_dependency_chain=True,
        source_sha256={str(path.relative_to(ROOT)):file_sha256(path) for path in sources})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT/'conditional_gcnn_projection.json', report)
    return report


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
