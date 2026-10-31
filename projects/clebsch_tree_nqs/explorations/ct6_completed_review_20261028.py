"""Fetch and independently review the two completed, frozen CT6 continuations.

Read-only remote access; no new samples, Hamiltonian calls, or job changes.
Uses the previous independent NumPy audit oracle, not the training estimator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import jax
jax.config.update('jax_enable_x64', True)
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[3]
BASE = '/home/yx789/nqs-wavefunctions/nqs/ct6_verified_continuation_20261028'
FILES = ('checkpoint.pkl', 'registration.json', 'endpoint_frozen.json',
         'audit_result.json', 'train_result.json', 'training_progress.json',
         'replay_result.json', 'device_replay.json', 'cost.json',
         'terminal_audit/audit.json', 'terminal_audit/products.npz',
         'terminal_audit/draw.npz', *(f'step{i}/transition.json' for i in range(1, 9)))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(output: Path) -> None:
    """Copy only explicit frozen files; verify remote and local SHA256."""
    if output.exists():
        raise FileExistsError('new immutable completed-review directory required')
    query = ('import hashlib,json;from pathlib import Path\n'
             f'base=Path({BASE!r});names={FILES!r}\n'
             'print(json.dumps({tag:{name:hashlib.sha256((base/tag/"run01"/name).read_bytes()).hexdigest() '
             'for name in names} for tag in ("ct6_j0","ct6_j0125")}))\n')
    reply = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                            'unicorn', 'python3 -'], input=query, text=True,
                           capture_output=True, timeout=90, check=True)
    manifest = json.loads(reply.stdout)
    output.mkdir(parents=True)
    atomic_json(output/'remote_manifest.json', manifest)
    for tag, names in manifest.items():
        destination = output/tag
        destination.mkdir()
        command = ['rsync', '-aR', *[f'unicorn:{BASE}/{tag}/run01/./{name}' for name in names],
                   str(destination)+'/']
        subprocess.run(command, check=True, timeout=240)
        for name, expected in names.items():
            if digest(destination/name) != expected:
                raise ArithmeticError('remote/local frozen artifact checksum mismatch: '+name)


def review(output: Path) -> dict:
    """Recompute paired moments and bind all eight parameter/RNG transitions."""
    oracle_path = ROOT/'projects/clebsch_tree_nqs/results/ct_status_20261028/ct6_independent_review/recompute.py'
    spec = importlib.util.spec_from_file_location('previous_independent_ct6_oracle', oracle_path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    records = {}
    for tag, gcnn in (('ct6_j0', -.560313), ('ct6_j0125', -.515386)):
        directory = output/tag
        registration = json.loads((directory/'registration.json').read_text())
        audit = json.loads((directory/'terminal_audit/audit.json').read_text())
        frozen = json.loads((directory/'endpoint_frozen.json').read_text())
        saved = load_checkpoint(directory/'checkpoint.pkl')
        meta = saved['metadata']
        params_hash = pytree_sha256(saved['params'])
        assert audit['status'] == 'complete' and audit['records'] == 8192
        assert audit['endpoint_params_sha256'] == params_hash == frozen['parameter_sha256']
        assert digest(directory/'checkpoint.pkl') == frozen['checkpoint_sha256']
        assert meta['parameter_sha256'] == params_hash
        assert pytree_sha256(saved['optimizer_state']) == meta['optimizer_sha256']
        assert canonical_sha256(meta['rng']) == meta['rng_sha256']
        with np.load(directory/'terminal_audit/products.npz') as data:
            rows, psi, hpsi, density, guide = [data[name] for name in
                ('rows', 'psi', 'hpsi', 'proposal', 'guide_components')]
        assert rows.shape == (8192, 36) and np.all(rows.sum(1) == 18)
        assert np.isfinite(np.r_[psi.ravel(), hpsi.ravel(), density]).all()
        assert np.all(density > 0)
        with np.load(directory/'terminal_audit/draw.npz') as draw:
            assert np.array_equal(rows, draw['rows'])
        recomputed = oracle.moments(psi, hpsi, density, audit['moments']['independent_groups'])
        keys = ('endpoint_values', 'candidate_minus_control', 'endpoint_iid_delta_SE',
                'paired_iid_delta_SE', 'endpoint_independent_block_jackknife_SE',
                'paired_independent_block_jackknife_SE')
        differences = {key: float(np.max(abs(np.asarray(recomputed[key]) -
                          np.asarray(audit['moments'][key])))) for key in keys}
        if max(differences.values()) > 2e-10:
            raise ArithmeticError('independent moment recomputation failed')
        expected_q = .875/2*guide.real.mean(1).sum(1) + .125/float(__import__('math').comb(36, 18))
        if not np.allclose(density, expected_q, rtol=2e-13, atol=0):
            raise ArithmeticError('full-support importance density replay failed')
        steps = [json.loads((directory/f'step{i}/transition.json').read_text()) for i in range(1, 9)]
        for left, right in zip(steps[:-1], steps[1:], strict=True):
            assert left['candidate_params_sha256'] == right['parent_params_sha256']
            assert left['rng_after_sha256'] == right['rng_before_sha256']
        assert steps[-1]['candidate_params_sha256'] == params_hash
        assert steps[-1]['rng_after_sha256'] == canonical_sha256(meta['rng'])
        assert all(step['accepted'] for step in steps)
        energy = recomputed['endpoint_values'][1][0]
        se = recomputed['endpoint_independent_block_jackknife_SE'][1][0]
        delta = recomputed['candidate_minus_control'][0]
        paired_se = recomputed['paired_independent_block_jackknife_SE'][0]
        records[tag] = dict(absolute_step=meta['absolute_completed_steps'],
            parameter_sha256=params_hash, checkpoint_sha256=digest(directory/'checkpoint.pkl'),
            records=8192, moments=recomputed, maximum_numeric_difference=max(differences.values()),
            numeric_differences=differences, parameter_optimizer_rng_and_eight_updates_verified=True,
            proposal_maximum_relative_error=float(np.max(abs(density-expected_q)/density)),
            energy_per_site=energy, energy_block_SE=se, gcnn_energy_per_site=gcnn,
            difference_from_gcnn_per_site=energy-gcnn, paired_energy_change=delta,
            paired_energy_change_SE=paired_se, paired_energy_change_significance=abs(delta)/paired_se,
            variance_relative_change=recomputed['candidate_minus_control'][2]/recomputed['endpoint_values'][0][2],
            maximum_SR_residual=max(step['SR']['relative_SR_residual'] for step in steps),
            independent_guard_FS_radians=[step['calibration']['guard']['FS_radians'] for step in steps],
            parameter_count=sum(int(np.size(x)) for x in jax.tree_util.tree_leaves(saved['params'])))
    result = dict(status='completed_frozen_CT6_audits_independently_verified', runs=records,
        oracle_sha256=digest(oracle_path), new_samples=0, new_H_queries=0, remote_jobs_changed=False,
        best_mean_reporting_note='new paired gains are reliable; new absolute means are not lower than old466/492 means, so best-result plot keeps old endpoints')
    atomic_json(output/'independent_review.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    if args.fetch:
        fetch(args.output)
    result = review(args.output)
    print(json.dumps({tag: {k: v for k, v in row.items() if k != 'moments'}
                      for tag, row in result['runs'].items()}, indent=2))
