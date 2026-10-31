"""Cost-only size extension of the separate finite-width ClebschTree.

The existing L=6*2**k cover and CG convention are reused unchanged. This is
not Spatial-ViT, an energy optimization, or an equilibrated-state audit.
Transferred nine-site tensors and compatible eighteen-site merges are
warm starts only: newly opened coarse charges change the physical state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import signal
import subprocess
from math import prod
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank, init_arity_pair
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, init_mixed_clebsch_family, mixed_clebsch_bank,
    mixed_clebsch_layout, mixed_parameter_count, make_mixed_clebsch_amplitude,
    make_mixed_projected_amplitude,
)
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.physical_exchange import run_exchange_sweeps


def scale_structure(config: MixedClebschConfig, j2: float) -> dict:
    """Exact topology/array counts, not a throughput or memory prediction."""
    if not np.isfinite(j2):
        raise ValueError('finite J2/J1 required')
    hierarchy = triangular_schur_hierarchy(config.length)
    layout = mixed_clebsch_layout(config)
    bonds = triangular_j1j2_bonds(config.length, config.length, j2=float(j2), periodic=True)
    top_count = sum(prod(g['shape']) for row in layout for g in row['groups'])
    return dict(length=config.length, n_sites=hierarchy.n_sites,
        layer_counts=list(hierarchy.layer_counts), nine_site_blocks=len(hierarchy.blocks),
        top_merges=len(layout), top_real_parameters=top_count,
        retained_q_max=int(config.maximum_q), multiplicity_cap=int(config.multiplicity_cap),
        full_A1_images=len(triangular_hnf_gamma_a1_group(((config.length, 0), (0, config.length)))),
        Hamiltonian_bonds=len(bonds), strict_total_q=0,
        physical_state_preserved_by_size_change=False)


def balanced_block_probe_states(config: MixedClebschConfig, *, count: int, seed: int) -> np.ndarray:
    """Synthetic non-equilibrium inputs with each coarse-pair m=0.

    Uniform fixed-Sz rows often lie at EXACT nodes of a finite-q TTN. A
    block-balanced probe avoids mistaking that structural support for float
    underflow. It is neither a Born draw nor invalid-sample repair.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('positive integer probe count required')
    tree = triangular_schur_hierarchy(config.length)
    rng = np.random.default_rng(seed)
    rows = np.zeros((count, tree.n_sites), np.int32)
    for index, block in enumerate(tree.blocks):
        n_up = 4+index % 2
        for row in rows:
            row[np.asarray(block.sites)] = rng.permutation([0]*(9-n_up)+[1]*n_up)
    return rows


def scale_arity_pair(params: dict, source: MixedClebschConfig, target: MixedClebschConfig,
                     *, source_arm: str, rank: int, key: jax.Array) -> tuple[dict, dict]:
    """Tile trained local banks, copy same-shape bottom binary maps, then lift.

    Both target arms start at the SAME compressed physical bank. No old
    optimizer/sampler/RNG is reused across Hilbert spaces. The source root
    is NOT copied into a larger non-root edge: doing so would force every
    source-sized supercell to spin zero and can kill intercell tangents.
    """
    if source.length != 6 or target.length <= source.length:
        raise ValueError('registered transfer is from L6 to a larger family member')
    if (source.multiplicity_cap, source.maximum_q) != (target.multiplicity_cap, target.maximum_q):
        raise ValueError('size-only transfer must retain all charge and width budgets')
    old = arity_clebsch_bank(params, source, source_arm, rank)
    family = init_mixed_clebsch_family(key, target)
    bank = mixed_clebsch_bank(family['ternary_c3'], target, 'ternary_c3')
    target_tree = triangular_schur_hierarchy(target.length)
    local = tuple(old['local'][block.base_index] for block in target_tree.blocks)
    old_layout, new_layout = mixed_clebsch_layout(source), mixed_clebsch_layout(target)
    top = list(bank['top'])
    copied = []
    for index, row in enumerate(new_layout):
        if row['left'] >= len(target_tree.blocks) or row['right'] >= len(target_tree.blocks):
            continue
        candidate = row['left'] % 4 // 2
        reference = old_layout[candidate]
        if [(g['keys'], g['shape']) for g in row['groups']] != [
                (g['keys'], g['shape']) for g in reference['groups']]:
            raise ValueError('incompatible lower coarse merge layout')
        top[index] = old['top'][candidate]
        copied.append(index)
    common = dict(local=local, top=tuple(top))
    pair, truncation = init_arity_pair(common, target, rank)
    return pair, dict(source_length=6, target_length=target.length,
        source_arm=source_arm, copied_nine_site_banks=len(local),
        copied_eighteen_site_merges=copied, new_coarse_tensors_random=True,
        source_root_copied=False, parent_physical_state_preserved=False,
        same_target_initial_physical_state=True, fresh_target_audit_required=True,
        optimizer_reset_required=True, sampler_reset_required=True, rng_reset_required=True,
        truncation=truncation)


def checked_scale_parent(request: dict, *, target_j2: float) -> dict:
    """Load one audited immutable same-H parent, not a live training output."""
    checkpoint = Path(request['checkpoint'])
    directory = checkpoint.parent
    registration = json.loads((directory/'registration.json').read_text())
    seal = registration['sha256']
    if (seal != request['registration_sha256'] or seal != canonical_sha256({
            k: v for k, v in registration.items() if k != 'sha256'})):
        raise ValueError('source parent registration mismatch')
    parent_spec = registration['config']
    if (parent_spec['model'] != request['model'] or parent_spec['arm'] != request['arm']
            or parent_spec['j2_over_j1'] != target_j2):
        raise ValueError('source physical model/H identity mismatch')
    if file_sha256(checkpoint) != request['checkpoint_sha256']:
        raise ValueError('trusted source checkpoint hash mismatch')
    saved = load_checkpoint(checkpoint)
    if (pytree_sha256(saved['params']) != request['params_sha256']
            or saved['metadata']['params_sha256'] != request['params_sha256']
            or saved['metadata']['registration_sha256'] != seal):
        raise ValueError('trusted source parameters/registration mismatch')
    audit_path = directory/'terminal_audit.json'
    if file_sha256(audit_path) != request['audit_sha256']:
        raise ValueError('source independent audit seal mismatch')
    audit = json.loads(audit_path.read_text())
    if (audit['params_sha256'] != request['params_sha256']
            or audit['registration_sha256'] != seal):
        raise ValueError('source independent audit state mismatch')
    return saved


def collect_accounting(config_path: Path, job_id: str) -> dict:
    """An independent afterany CPU job records hard kills/late XLA signals.

    A Python signal handler is best effort during a native compile/kernel.
    Therefore a missing final file is NEVER interpreted as successful cost
    completion. Slurm TIMEOUT/OOM/CANCELLED records an explicit censored row.
    """
    if not job_id.isdigit():
        raise ValueError('numeric Slurm job id required')
    spec = json.loads(config_path.read_text())
    output = Path(spec['output'])
    raw = subprocess.check_output([
        '/usr/local/slurm/current/bin/sacct', '-j', job_id, '-X', '-n', '-P',
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'], text=True)
    records = [line.split('|') for line in raw.splitlines() if line.strip()]
    main = next((r for r in records if r[0] == job_id), None)
    if main is None:
        raise RuntimeError('completed-job Slurm accounting not yet available')
    state = main[1].split()[0]
    final_path = output/'results.json'
    final = json.loads(final_path.read_text()) if final_path.exists() else None
    progress_path = output/'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
    complete = bool(state == 'COMPLETED' and main[2] == '0:0' and final
        and (final.get('status') == 'completed' or final.get('stage') == 'structure'))
    censored = bool(state in ('TIMEOUT', 'OUT_OF_MEMORY', 'CANCELLED', 'PREEMPTED', 'NODE_FAIL')
        or (final and final.get('censored')))
    result = dict(job_id=job_id, Slurm_state=state, exit_code=main[2],
        Slurm_elapsed_seconds=int(main[3]), Slurm_allocated_resources=main[4],
        config_sha256=file_sha256(config_path), status='completed' if complete else (
            'censored' if censored else 'failed_validation_or_runtime'),
        censored=censored, final_result_present=final is not None,
        last_operation=None if progress is None else progress.get('current_operation'),
        last_progress_elapsed_seconds=None if progress is None else progress.get('elapsed_seconds'),
        operation_start_elapsed_seconds=None if progress is None else progress.get('operation_start_elapsed_seconds'),
        operation_started_utc=None if progress is None else progress.get('operation_started_utc'),
        complete_H_gradient_proven=complete and spec.get('task') == 'triangle_ClebschTree_size_extension_cost_only'
            and final.get('stage') == 'full', accounting_raw=raw)
    atomic_json(output/'accounting.json', result)
    if final is None and censored:
        atomic_json(final_path, dict(result, last_progress=progress))
    return result


def run(config_path: Path, *, stage: str) -> dict:
    """Execute bounded synthetic-input costs; NEVER interpret them as energies."""
    jax.config.update('jax_enable_x64', True)
    spec = json.loads(config_path.read_text())
    output = Path(spec['output'])
    if output.exists():
        raise FileExistsError('cost outputs are immutable; use a new directory')
    target = MixedClebschConfig(**spec['model'])
    if spec.get('require_gpu') and jax.default_backend() != 'gpu':
        raise RuntimeError('registered GPU cost run cannot fall back to CPU')
    output.mkdir(parents=True)
    started = perf_counter()
    result = dict(config_sha256=file_sha256(config_path), stage=stage,
        structure=scale_structure(target, spec['j2_over_j1']),
        backend=jax.default_backend(), devices=[str(x) for x in jax.devices()],
        new_training_steps=0, energy_estimate=False, equilibrium_claim=False,
        created_utc=datetime.now(timezone.utc).isoformat())
    def deadline(signum, frame):
        result.update(status='censored', censored=True,
            censor_reason='Slurm deadline/termination signal '+signal.Signals(signum).name,
            elapsed_seconds=perf_counter()-started)
        atomic_json(output/'results.json', result)
        raise SystemExit(2)
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, deadline)
    signal.signal(signal.SIGTERM, deadline)
    root = Path(__file__).resolve().parents[3]
    atomic_json(output/'registration.json', dict(config=spec, stage=stage,
        config_sha256=result['config_sha256'], sources=transitive_local_source_manifest(
            # vmc.__init__ resolves the public local_energy/submodule name
            # collision with import_module; AST-only discovery cannot see it.
            [Path(__file__), root/'su2_nqs/vmc/local_energy.py'], root=root,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))))
    if stage == 'structure':
        atomic_json(output/'results.json', result)
        return result
    result.update(current_operation='checked_parent_and_size_initialization',
        operation_start_elapsed_seconds=perf_counter()-started,
        operation_started_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output/'progress.json', result)
    request = spec['parent']
    saved = checked_scale_parent(request, target_j2=spec['j2_over_j1'])
    source = MixedClebschConfig(**request['model'])
    pair, note = scale_arity_pair(saved['params'], source, target,
        source_arm=request['arm'], rank=spec['binary_rank'],
        key=jax.random.PRNGKey(spec['seed']))
    result['transfer'] = note
    result['parameter_counts'] = {a: mixed_parameter_count(p) for a, p in pair.items()}
    result['timings'] = {}
    def timed(name, fn, *args):
        if perf_counter()-started > spec['seconds_cap']:
            result.update(status='censored', censored=True,
                censor_reason='registered cost cap reached before '+name)
            atomic_json(output/'results.json', result)
            raise SystemExit(2)
        result['current_operation'] = name
        result['operation_start_elapsed_seconds'] = perf_counter()-started
        result['operation_started_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(output/'progress.json', result)
        begin = perf_counter(); value = fn(*args); jax.block_until_ready(value)
        first = perf_counter()-begin
        begin = perf_counter(); second = fn(*args); jax.block_until_ready(second)
        result['timings'][name] = dict(first_seconds=first, warm_seconds=perf_counter()-begin)
        result['elapsed_seconds'] = perf_counter()-started
        atomic_json(output/'progress.json', result)
        return value
    raw = make_mixed_clebsch_amplitude(target)
    bank = arity_clebsch_bank(pair['ternary_joint'], target, 'ternary_joint', spec['binary_rank'])
    states = jnp.asarray(balanced_block_probe_states(target,
        count=spec['chains'], seed=spec['seed']+1))
    result['probe_distribution'] = 'synthetic nine-site block-balanced; NOT Born equilibrium'
    raw_values = timed('raw_batch', jax.jit(jax.vmap(raw, in_axes=(None, 0))), bank, states)
    result['raw_finite_nonzero'] = bool(np.isfinite(raw_values).all() and np.any(np.asarray(raw_values)!=0))
    if not result['raw_finite_nonzero']:
        raise FloatingPointError('transferred raw amplitude underflow/node probe failed')
    if stage in ('projected', 'full'):
        images = np.asarray(triangular_hnf_gamma_a1_group(((target.length, 0), (0, target.length))), np.int32)
        projected = make_mixed_projected_amplitude(target, images, chunk=spec['projection_chunk'])
        amp = jax.jit(projected)
        value = timed('complete_A1_amplitude', amp, bank, states[0])
        symmetry = amp(bank, states[0, images[len(images)//3]])
        result['A1_relative_difference'] = float(abs(value-symmetry)/max(abs(value), abs(symmetry), 1e-280))
        if not np.isfinite(value) or result['A1_relative_difference'] > 1e-10:
            raise ArithmeticError('complete A1 amplitude covariance check failed')
    if stage == 'full':
        bonds = tuple(triangular_j1j2_bonds(target.length, target.length,
            j2=spec['j2_over_j1'], periodic=True))
        action = make_amplitude_heisenberg_action(projected, bonds, n_sites=target.length**2)
        psi, hpsi = timed('complete_H_one_configuration', action, bank, states[:1])
        result['H_finite'] = bool(np.isfinite(psi).all() and np.isfinite(hpsi).all())
        if not result['H_finite'] or np.any(np.asarray(psi) == 0):
            raise FloatingPointError('synthetic full-H ratio probe is nonfinite or at a node')
        result['synthetic_H_ratio_not_Born_energy'] = [None if a == 0 else float(b/a)
            for a, b in zip(np.asarray(psi), np.asarray(hpsi), strict=True)]
    if stage in ('projected', 'full'):
        # The shared bank gives the same H cost at the common physical state.
        # Gradients still depend on the binary/joint parameterization, so both
        # complete parameter gradients are timed independently.
        result['gradient_finite'] = {}
        for arm, p in pair.items():
            loss = lambda z: projected(arity_clebsch_bank(z, target, arm, spec['binary_rank']), states[0])
            gradient = timed('full_parameter_A1_gradient_'+arm, jax.jit(jax.grad(loss)), p)
            result['gradient_finite'][arm] = bool(all(np.isfinite(x).all() for x in jax.tree.leaves(gradient)))
            if not result['gradient_finite'][arm]:
                raise FloatingPointError('nonfinite full parameter gradient: '+arm)
    if stage == 'full':
        edges = [(a, b) for a, b, _ in bonds]
        sampler = jax.jit(lambda p, k, x: run_exchange_sweeps(p, k, x,
            amplitude_fn=projected, particle_kind='spin', sweeps=spec['proposal_attempts'],
            exchange_edges=edges))
        sampled = timed('unbalanced_MCMC_cost_probe', sampler, bank,
            jax.random.PRNGKey(spec['seed']+2), states)
        result['sampler_diagnostics'] = {k: float(v) for k, v in sampled[-1].items()}
        # Cost checkpoint with exact state and RNG replay; not a trained state.
        save_checkpoint(output/'cost_checkpoint.pkl', pair, metadata=dict(
            sampler_states=np.asarray(sampled[0]), rng=np.asarray(sampled[2]),
            trained=False, source_checkpoint_sha256=request['checkpoint_sha256']))
        restored = load_checkpoint(output/'cost_checkpoint.pkl')
        result['checkpoint_replay_exact'] = pytree_sha256(restored['params']) == pytree_sha256(pair)
        restart = sampler(bank, jnp.asarray(restored['metadata']['rng']),
            jnp.asarray(restored['metadata']['sampler_states']))
        replay = sampler(bank, sampled[2], sampled[0])
        result['sampler_RNG_checkpoint_replay_exact'] = all(
            np.array_equal(a, b) for a, b in zip(jax.tree.leaves(restart), jax.tree.leaves(replay), strict=True))
    result.update(status='completed', censored=False)
    result['elapsed_seconds'] = perf_counter()-started
    result['result_sha256'] = canonical_sha256(result)
    atomic_json(output/'results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('structure', 'raw', 'projected', 'full'), default='structure')
    parser.add_argument('--collect-job', default=None)
    args = parser.parse_args()
    report = collect_accounting(args.config, args.collect_job) if args.collect_job else run(args.config, stage=args.stage)
    print(json.dumps(report, indent=2))
