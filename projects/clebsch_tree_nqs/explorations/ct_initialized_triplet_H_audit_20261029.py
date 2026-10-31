"""Actual full-H audit of frozen, supervised CT6 triplet initializations.

No new training or endpoint choice. Reuse the128 held-out IID rows already
used for the reported overlap confirmation, explicitly NOT a second fresh
audit. They were never used by initialization. Positive Q and target nodes
are retained for the physical H-squared moment. Nothing runs remotely.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.models.triangular_sector_amplitude import pack_sector_parameters
from su2_nqs.models.triangular_target_root import collective_spin_root_coefficients
from projects.clebsch_tree_nqs.workflows.ct_sector_excitation_pilot import make_runtime, deadline
from projects.clebsch_tree_nqs.workflows.triangle_iid_orbit_audit import paired_iid_moments
from .ct_physical_spin_seed_20261029 import PARENTS


def run(directory: Path) -> dict:
    """Evaluate both pre-frozen CT endpoints and compare to the physical seed."""
    output = directory / 'initialized_CT_full_H_audit'
    if output.exists():
        raise FileExistsError('new immutable physics-audit directory required')
    report = json.loads((directory / 'results.json').read_text())
    frozen = json.loads((directory / 'endpoint_frozen.json').read_text())
    checkpoint = directory / 'initialized.pkl'
    assert report['fixed_attempts_completed']
    assert file_sha256(checkpoint) == frozen['checkpoint_sha256']
    final = load_checkpoint(checkpoint)['params']
    assert pytree_sha256(final) == frozen['parameter_sha256']
    tag = 'ct6_j0' if report['j2'] == 0 else 'ct6_j0125'
    arity = 'ternary_joint' if report['j2'] == 0 else 'binary_truncated'
    parent = load_checkpoint(PARENTS / tag / 'checkpoint.pkl')
    assert pytree_sha256(parent['params']) == report['parent_parameter_sha256']
    cfg = MixedClebschConfig(6, 28, 10)
    initial = pack_sector_parameters(parent['params'], tuple(jnp.asarray(x) for x in
        collective_spin_root_coefficients(arity_clebsch_bank(parent['params'], cfg, arity, 8), cfg, 2)))
    plan = dict(model=dict(length=6, multiplicity_cap=28, maximum_q=10),
        j2=report['j2'], arity=arity, binary_rank=8, epsilon=.125,
        projection_chunk=4, H_chunk=1, action_chunk=1,
        sector=dict(target_q=2, target_m2=0, momentum=((1, 3), (1, 3))))
    with np.load(directory / 'audit_source_products.npz') as raw:
        rows, Q = raw['rows'], raw['proposal']
        source, source_H = raw['psi'][:, 1], raw['hpsi'][:, 1]
    with np.load(directory / 'heldout_CT_amplitudes.npz') as raw:
        assert np.array_equal(raw['rows'], rows) and np.array_equal(raw['proposal'], Q)
        expected = raw['psi']
    output.mkdir()
    registration = dict(task='frozen_initialized_CT_actual_physical_H_audit', j2=report['j2'],
        initial_parameter_sha256=pytree_sha256(initial), final_parameter_sha256=pytree_sha256(final),
        initialized_checkpoint_sha256=file_sha256(checkpoint), records=len(rows), groups=16,
        reused_heldout_overlap_panel=True, independent_of_initialization_fit_and_calibration=True,
        a_new_fresh_confirmation_panel=False, no_updates_or_endpoint_selection=True,
        source_products_sha256=file_sha256(directory / 'audit_source_products.npz'),
        statistics_seed=report['seeds']['statistics'] + 401,
        wall_cap_seconds=900, doubled_target_q=2, doubled_target_m2=0,
        momentum=[[1, 3], [1, 3]], full_native_nonroot_W_and_target_root=True)
    atomic_json(output / 'registration.json', registration)
    start = perf_counter()
    with deadline(900):
        runtime = make_runtime(plan, parent)
        parts = []
        elapsed = []
        for arm, params in enumerate((initial, final)):
            tick = perf_counter()
            # Bound host memory and retain all physical H connections.
            values = []
            for i in range(0, len(rows), 16):
                p, hp = runtime['H'](params, rows[i:i+16])
                values.append(np.column_stack((p, hp)))
                print(tag, 'fullH', arm, i+16, '/', len(rows), flush=True)
            parts.append(np.concatenate(values))
            elapsed.append(perf_counter() - tick)
        products = np.stack(parts, axis=1)
        replay = float(np.max(abs(products[:, :, 0] - expected)))
        assert np.allclose(products[:, :, 0], expected, rtol=2e-11, atol=2e-16)
        # Independent complete recontraction at fixed rows, not another cache.
        oracle = []
        for arm, params in enumerate((initial, final)):
            p, hp = runtime['generic_H'](params, rows[:2])
            oracle.append(float(np.max(abs(np.column_stack((p, hp)) - products[:2, arm]))))
        assert max(oracle) < 2e-12
        labels = np.repeat(np.arange(16), len(rows)//16)
        moments = paired_iid_moments(products[:, :, 0], products[:, :, 1], Q, labels,
            n_sites=36, seed=registration['statistics_seed'])
        anchor = paired_iid_moments(np.column_stack((source, products[:, 1, 0])),
            np.column_stack((source_H, products[:, 1, 1])), Q, labels,
            n_sites=36, seed=registration['statistics_seed'] + 1)
        np.savez_compressed(output / 'frozen_CT_H_products.npz', rows=rows, proposal=Q,
            psi=products[:, :, 0], hpsi=products[:, :, 1])
        result = dict(status='completed_actual_physical_H_audit', registration=registration,
            moments= moments, physical_seed_vs_initialized_CT=anchor,
            H_wall_seconds_by_arm=elapsed, amplitude_replay_maximum_absolute_error=replay,
            independent_generic_H_maximum_absolute_error=max(oracle),
            runtime_work=runtime['work'], wall_seconds=perf_counter()-start,
            physical_energy_optimization_steps=0, not_a_converged_gap=True,
            endpoint_checkpoint_unchanged=file_sha256(checkpoint)==frozen['checkpoint_sha256'])
        atomic_json(output / 'results.json', result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    r = run(args.directory)
    print(json.dumps(dict(status=r['status'], moments=r['moments']['endpoint_values'],
        delta=r['moments']['candidate_minus_control'], wall_seconds=r['wall_seconds']), indent=2))
