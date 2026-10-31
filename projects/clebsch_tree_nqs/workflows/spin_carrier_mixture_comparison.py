"""Optimized old-state mixing versus fixed learned-shape additions.

The old audit is explicitly development data.  A new audit is generated only
after full-complex subspace coefficients have been sealed.  This workflow
does not retrain the network or claim full nonlinear-manifold optimization.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import tempfile
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import (
    file_sha256, numerical_runtime_identity, transitive_local_source_manifest,
)
from su2_nqs.models.schur_cutset_attention import make_branch_bank
from su2_nqs.vmc.carrier_subspace import fit_carrier_subspace
from su2_nqs.vmc.cutset_training import paired_chain_audit, weighted_state_distance
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from .spin_decoder_cost import ROOT, peak_rss_bytes
from .spin_schur_carrier_control import setup, draw

jax.config.update('jax_enable_x64', True)
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_carrier_mixture_comparison.json'


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Replace a numeric checkpoint only after a complete fsynced archive."""
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary): os.unlink(temporary)
        raise


def validate_parent_seals(spec: dict) -> dict:
    """Validate original provenance before creating a new seal of old data."""
    parent = ROOT/spec['parent']
    old = json.loads((parent.parent/'registration.json').read_text())
    body = {key:value for key, value in old.items() if key != 'registration_sha256'}
    if canonical_sha256(body) != old['registration_sha256']:
        raise ValueError('original registration canonical checksum failed')
    if json.loads((ROOT/spec['parent_config']).read_text()) != old['config']:
        raise ValueError('parent config differs from the original registration')
    for category in ('inputs', 'sources'):
        for name, digest in old[category].items():
            if file_sha256(ROOT/name) != digest:
                raise ValueError(f'original registered {category} changed: {name}')
    identity = dict(registration_sha256=old['registration_sha256'], length=6)
    for stage in ('prepare', 'train', 'audit'):
        result = json.loads((parent/stage/'results.json').read_text())
        if result['identity'] != {**identity, 'stage':stage}:
            raise ValueError(f'original {stage} identity mismatch')
        for name, digest in result.get('files', {}).items():
            if file_sha256(parent/stage/name) != digest:
                raise ValueError(f'original {stage} artifact changed: {name}')
    audit_inputs = json.loads((parent/'audit/inputs.json').read_text())
    if audit_inputs['identity'] != {**identity, 'stage':'audit'}:
        raise ValueError('original audit input identity mismatch')
    for name, digest in audit_inputs['files'].items():
        if file_sha256(parent/name) != digest:
            raise ValueError(f'original audit input changed: {name}')
    sampling = json.loads((parent/'audit/sampling.json').read_text())
    if sampling['samples_sha256'] != file_sha256(parent/'audit/samples.npz'):
        raise ValueError('original audit sampling seal changed')
    return old


def validate_protocol(spec: dict, parent_spec: dict) -> None:
    """Fail closed on accidental changes to this bounded N36 comparison."""
    expected = dict(n_sites=36,
        basis_names=['carrier', 'reference_scaled', 'raw_diagonal', 'raw_control'],
        arm_names=['carrier', 'old_mixture', 'network_diagonal', 'network_control', 'raw_diagonal', 'raw_control'],
        subspaces=dict(old_mixture=[0, 1], network_diagonal=[0, 1, 2], network_control=[0, 1, 3]),
        primary_contrast='network_control-minus-old_mixture',
        secondary_contrast='network_diagonal-minus-old_mixture', audit_blocks=8, H_microbatch=2,
        maximum_residual_H_attempted_centers=386, maximum_carrier_H_attempted_centers=256)
    for name, value in expected.items():
        if spec.get(name) != value: raise ValueError(f'fixed comparison protocol mismatch: {name}')
    if parent_spec['chains'] != 16 or parent_spec['j2_over_j1'] != .125:
        raise ValueError('expected the unchanged16-chain triangular N36 J2/J1=.125 parent')
    old_seeds = {value for key, value in parent_spec.items() if 'seed' in key and isinstance(value, int)}
    old_seeds |= {value+case['seed_offset'] for value in tuple(old_seeds) for case in parent_spec['cases']}
    if (spec['audit_seed'] in old_seeds or spec['bootstrap_seed'] in old_seeds or
        spec['audit_seed'] == spec['bootstrap_seed']):
        raise ValueError('audit/bootstrap seed must be new and distinct')


def registration(spec: dict) -> dict:
    """Seal viewed development data, parent checkpoints and all dependencies."""
    parent = ROOT/spec['parent']; old_spec_path = ROOT/spec['parent_config']
    old_spec = json.loads(old_spec_path.read_text())
    validate_protocol(spec, old_spec)
    original_registration = validate_parent_seals(spec)
    interface_path = ROOT/old_spec['interface_config']
    interface = json.loads(interface_path.read_text())
    carrier_parent = ROOT/interface['parent']/'triangular'
    paths = [old_spec_path, interface_path, parent.parent/'registration.json']
    paths += [carrier_parent/name for name in ('results.json', 'parameters.json', 'parameters.npz')]
    paths += [parent/name for name in (
        'prepare/results.json', 'prepare/initial.json', 'prepare/initial.npz',
        'prepare/metric.npz', 'train/results.json', 'train/diagonal.json',
        'train/diagonal.npz', 'train/control.json', 'train/control.npz',
        'train/metric_values.npz', 'audit/results.json', 'audit/inputs.json',
        'audit/samples.npz', 'audit/sampling.json', 'audit/values.npz',
        'audit/reference_H.npz', 'audit/diagonal_H.npz', 'audit/control_H.npz')]
    result = dict(config=spec, parent_registration_sha256=original_registration['registration_sha256'],
        inputs={str(p.relative_to(ROOT)):file_sha256(p) for p in paths},
        runtime=numerical_runtime_identity(), sources=transitive_local_source_manifest(
            (Path(__file__),), root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    result['registration_sha256'] = canonical_sha256(result)
    return result


def saved_development_basis(parent: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read C, D0/scale, and both physical endpoints, with total-H units."""
    data = dict(np.load(parent/'audit/values.npz', allow_pickle=False))
    ref = dict(np.load(parent/'audit/reference_H.npz', allow_pickle=False))
    sample_sha = file_sha256(parent/'audit/samples.npz')
    if str(ref['samples_sha256']) != sample_sha:
        raise ValueError('reference H and viewed audit use different configurations')
    if data['psi'].shape != (len(data['q']), 3) or data['hpsi'].shape != data['psi'].shape:
        raise ValueError('expected the original carrier/diagonal/control columns')
    values = np.column_stack((data['psi'][:, 0], ref['amplitudes'], data['psi'][:, 1:]))
    actions = np.column_stack((data['hpsi'][:, 0], ref['h_amplitudes'], data['hpsi'][:, 1:]))
    for k, arm in enumerate(('diagonal', 'control'), start=2):
        endpoint = dict(np.load(parent/'audit'/f'{arm}_H.npz', allow_pickle=False))
        if str(endpoint['samples_sha256']) != sample_sha:
            raise ValueError('endpoint H and viewed audit use different configurations')
        if not np.allclose(values[:, 0]+endpoint['amplitudes']-values[:, 1], values[:, k], rtol=0, atol=1e-14):
            raise ValueError('saved endpoint amplitude replay failed')
        if not np.allclose(actions[:, 0]+endpoint['h_amplitudes']-actions[:, 1], actions[:, k], rtol=0, atol=1e-13):
            raise ValueError('saved endpoint H replay failed')
    if not np.isfinite(values).all() or not np.isfinite(actions).all():
        raise ValueError('nonfinite development amplitudes or H actions')
    return values, actions, data


def _point_moments(psi: np.ndarray, hp: np.ndarray, q: np.ndarray, n_sites: int) -> dict:
    """Raw finite-Q diagnostics only, including the imaginary energy."""
    norm = np.sum(abs(psi)**2/q)
    if not np.isfinite(norm) or norm <= 0:
        raise FloatingPointError('zero or nonfinite finite-Q norm')
    energy = np.sum(psi.conj()*hp/q)/norm
    variance = np.sum(abs(hp-energy*psi)**2/q)/norm
    return dict(energy_per_site=float(energy.real/n_sites),
        imaginary_energy_per_site=float(energy.imag/n_sites), variance_per_site=float(variance/n_sites))


def fit_saved_subspaces(spec: dict, output: Path, registered: dict) -> dict:
    """Optimize three fixed complex spans; all LOCO fits are diagnostic only."""
    parent = ROOT/spec['parent']; values, actions, data = saved_development_basis(parent)
    q = data['q']; groups = data['groups']; n = spec['n_sites']
    names = spec['arm_names']; coefficients = np.zeros((values.shape[1], len(names)), dtype=np.complex128)
    coefficients[0, names.index('carrier')] = 1
    coefficients[2, names.index('raw_diagonal')] = 1
    coefficients[3, names.index('raw_control')] = 1
    fits = {}; sensitivity = {}
    options = dict(n_sites=n, relative_rank_tolerance=spec['relative_rank_tolerance'],
                   maximum_condition_number=spec['maximum_condition_number'])
    for arm, indices in spec['subspaces'].items():
        fit = fit_carrier_subspace(values, actions, q, indices=indices, **options)
        coefficients[:, names.index(arm)] = fit.coefficients
        fits[arm] = fit.as_dict(); records = []
        for group in np.unique(groups):
            keep = groups != group
            try:
                leave = fit_carrier_subspace(values[keep], actions[keep], q[keep], indices=indices, **options)
                held = ~keep
                records.append(dict(group=int(group), passed=True, fit=leave.as_dict(),
                    full_development_distance_to_pooled=weighted_state_distance(
                        values@fit.coefficients, values@leave.coefficients, q),
                    held_out_chain_moments=_point_moments(values[held]@leave.coefficients,
                        actions[held]@leave.coefficients, q[held], n)))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                records.append(dict(group=int(group), passed=False, error=repr(error)))
        sensitivity[arm] = dict(records=records, used_for_selection=False,
            description='Whole-chain delete/refit sensitivity, NOT untouched validation or a confidence interval.')
    if not np.isfinite(coefficients).all():
        raise FloatingPointError('nonfinite fitted coefficients')
    # The independent development Gram is a distance diagnostic, not an energy density.
    metric = dict(np.load(parent/'prepare/metric.npz', allow_pickle=False))
    metric_endpoints = dict(np.load(parent/'train/metric_values.npz', allow_pickle=False))
    scale = json.loads((parent/'prepare/results.json').read_text())['residual_scale']
    metric_basis = np.column_stack((metric['carrier'], metric['reference']/scale,
        metric_endpoints['diagonal'], metric_endpoints['control']))
    distances = {arm:weighted_state_distance(metric['carrier'], metric_basis@coefficients[:, i], metric['q'])
                 for i, arm in enumerate(names)}
    _atomic_npz(output/'coefficients.npz', coefficients=coefficients,
        arm_names=np.asarray(names), basis_names=np.asarray(spec['basis_names']))
    inherited = json.loads((parent/'train/results.json').read_text())
    prepared = json.loads((parent/'prepare/results.json').read_text())
    result = dict(identity=dict(registration_sha256=registered['registration_sha256'], stage='fit'),
        fits=fits, leave_one_chain_out=sensitivity, development_metric_distances=distances,
        development_moments={arm:_point_moments(values@coefficients[:, i], actions@coefficients[:, i], q, n)
                             for i, arm in enumerate(names)},
        primary_contrast=spec['primary_contrast'], secondary_contrast=spec['secondary_contrast'],
        basis_names=spec['basis_names'], arm_names=names, viewed_audit_reclassified_as_development=True,
        network_retrained=False, full_nonlinear_network_optimum_claim=False,
        coefficients_selected_without_new_audit=True, new_physical_calls=0,
        inherited_network_training=dict(parameter_count=prepared['parameter_count'],
            attempted_steps_per_arm=len(inherited['history']),
            accepted_steps_per_arm=sum(row['accepted_pair'] for row in inherited['history']),
            training_seconds=inherited['seconds'], arm_seconds=inherited['arm_seconds'],
            shared_gradient_compile_seconds=inherited['shared_gradient_compile_seconds'],
            training_work_ledger=inherited['ledger'],
            parent_training_result_sha256=file_sha256(parent/'train/results.json'),
            interpretation='Prior full-network training cost is inherited; three complex mixing coefficients are NOT the cost or capacity of the learned network.'),
        coefficients_sha256=file_sha256(output/'coefficients.npz'),
        production_authorized=False, n64_authorized=False)
    return result


def run(path: Path, stage: str, *, resume: bool = False) -> None:
    """Run one sealed fit or a budget-limited, resumable new-chain audit."""
    spec = json.loads(path.read_text()); output = ROOT/spec['output']
    registered = json.loads((output/'registration.json').read_text())
    if registered != registration(spec):
        raise ValueError('registered sources, parent artifacts, runtime or protocol changed')
    if resume and stage != 'audit':
        raise ValueError('--resume applies only to the expensive audit stage')
    dest = output/stage; dest.mkdir(parents=True, exist_ok=True)
    if (dest/'results.json').exists():
        raise FileExistsError('completed stage is immutable')
    if (dest/'progress.json').exists() and not resume:
        raise FileExistsError('interrupted audit requires explicit --resume')
    previous = json.loads((dest/'progress.json').read_text()) if (dest/'progress.json').exists() else None
    prior_seconds = float(previous['seconds']) if previous else 0.
    ledger = dict(previous['ledger']) if previous else dict(residual_H_attempted_centers=0,
        residual_H_completed_centers=0, shared_carrier_H_centers=0, carrier_H_attempted_centers=0)
    if resume and (dest/'failure.json').exists():
        failed = json.loads((dest/'failure.json').read_text())
        prior_seconds = max(prior_seconds, float(failed['seconds']))
        for key, value in failed['ledger'].items(): ledger[key] = max(ledger.get(key, 0), value)
    start = perf_counter(); wall_cap = spec[stage+'_wall_seconds']
    def progress(event: str, *, announce: bool = True, **extra):
        elapsed = prior_seconds+perf_counter()-start
        atomic_json(dest/'progress.json', dict(event=event, seconds=elapsed,
            peak_rss_bytes=peak_rss_bytes(), ledger=ledger, **extra))
        if announce: print(stage, event, round(elapsed, 2), 's', round(peak_rss_bytes()/2**30, 2), 'GiB', flush=True)
        if peak_rss_bytes() > spec['peak_rss_limit_gib']*2**30:
            raise MemoryError('registered RSS limit')
        if elapsed > wall_cap: raise TimeoutError('cumulative registered wall limit')
    def alarm(_signum, _frame): raise TimeoutError('registered stage wall limit')
    remaining = wall_cap-prior_seconds
    if remaining <= 0: raise TimeoutError('registered wall budget exhausted')
    old_alarm = signal.signal(signal.SIGALRM, alarm); signal.alarm(max(1, int(remaining)))
    try:
        progress('start_or_resume')
        if stage == 'fit':
            report = fit_saved_subspaces(spec, dest, registered)
            report.update(seconds=perf_counter()-start, peak_rss_bytes=peak_rss_bytes(), ledger=ledger)
            atomic_json(dest/'results.json', report); progress('fit_complete_coefficients_sealed')
            return
        fit_path = output/'fit/results.json'; fit = json.loads(fit_path.read_text())
        if fit['identity'] != dict(registration_sha256=registered['registration_sha256'], stage='fit'):
            raise ValueError('fit identity does not match registration')
        if file_sha256(output/'fit/coefficients.npz') != fit['coefficients_sha256']:
            raise ValueError('sealed coefficients changed')
        seal = dict(registration_sha256=registered['registration_sha256'],
            fit_sha256=file_sha256(fit_path), coefficients_sha256=fit['coefficients_sha256'])
        if (dest/'inputs.json').exists():
            if json.loads((dest/'inputs.json').read_text()) != seal:
                raise ValueError('audit fit/endpoint seal changed')
        else: atomic_json(dest/'inputs.json', seal)
        coeff_data = dict(np.load(output/'fit/coefficients.npz', allow_pickle=False))
        coefficients = coeff_data['coefficients']
        if (coeff_data['arm_names'].tolist() != spec['arm_names'] or
            coeff_data['basis_names'].tolist() != spec['basis_names'] or
            coefficients.shape != (4, len(spec['arm_names'])) or not np.isfinite(coefficients).all()):
            raise ValueError('invalid coefficient bank')
        parent = ROOT/spec['parent']; parent_spec = json.loads((ROOT/spec['parent_config']).read_text())
        prepared = json.loads((parent/'prepare/results.json').read_text())
        trained = json.loads((parent/'train/results.json').read_text())
        initial, cfg, residual, carrier, bonds, group_order = setup(parent_spec, 6)
        if pytree_sha256(initial) != prepared['initial_sha256']:
            raise ValueError('original physical initialization changed')
        generate = jax.jit(lambda p: make_branch_bank(p, cfg))
        banks = dict(reference=generate(initial))
        for arm in ('diagonal', 'control'):
            endpoint = load_parameter_artifact(parent/'train'/f'{arm}.json', initial,
                expected_identity={**trained['identity'], 'arm':arm})
            if pytree_sha256(endpoint) != trained['endpoint_sha256'][arm]:
                raise ValueError('endpoint parameter identity mismatch')
            banks[arm] = generate(endpoint)
        progress('original_model_and_endpoints_replayed')
        if (dest/'samples.npz').exists():
            data = dict(np.load(dest/'samples.npz', allow_pickle=False))
            embedded_info = json.loads(str(data.pop('_sampling_json')))
            embedded_info['samples_sha256'] = file_sha256(dest/'samples.npz')
            if (dest/'sampling.json').exists():
                info = json.loads((dest/'sampling.json').read_text())
                if info != embedded_info: raise ValueError('sampling sidecar and archive disagree')
            else:
                # Recover an interruption between two atomic writes, without redrawing.
                info = embedded_info; atomic_json(dest/'sampling.json', info)
            if info['samples_sha256'] != file_sha256(dest/'samples.npz') or info['audit_seal'] != seal:
                raise ValueError('saved new-chain samples or endpoint seal changed')
            ledger['shared_carrier_H_centers'] = max(ledger['shared_carrier_H_centers'], len(data['states']))
        else:
            proposed_count = parent_spec['chains']*spec['audit_blocks']
            if ledger['carrier_H_attempted_centers']+proposed_count > spec['maximum_carrier_H_attempted_centers']:
                raise RuntimeError('registered carrier-H attempted-center budget exhausted')
            ledger['carrier_H_attempted_centers'] += proposed_count
            progress('new_chains_begin_after_fit_seal')
            data, info = draw(parent_spec, spec['n_sites'], carrier, bonds, prepared['carrier_scale'],
                              spec['audit_seed'], spec['audit_blocks'])
            info.update(audit_seal=seal, audit_seed=spec['audit_seed'], blocks=spec['audit_blocks'])
            _atomic_npz(dest/'samples.npz', **data, _sampling_json=np.asarray(json.dumps(info, sort_keys=True)))
            info['samples_sha256'] = file_sha256(dest/'samples.npz')
            atomic_json(dest/'sampling.json', info)
            ledger['shared_carrier_H_centers'] += len(data['states'])
        if len(data['states']) != parent_spec['chains']*spec['audit_blocks']:
            raise ValueError('unexpected registered audit sample count')
        progress('new_audit_samples_saved')
        scale = prepared['residual_scale']
        action = make_amplitude_heisenberg_action(lambda b, x:residual(b, x[None])[0]/scale,
                                                  bonds, n_sites=spec['n_sites'])
        def hvalues(bank: dict, label: str) -> tuple[np.ndarray, np.ndarray]:
            file = dest/f'{label}_H.npz'; amps = []; hs = []; done = 0
            bank_hash = pytree_sha256(bank)
            if file.exists():
                saved = dict(np.load(file, allow_pickle=False))
                amps = [saved['amplitudes']]; hs = [saved['h_amplitudes']]; done = len(amps[0])
                if (str(saved['samples_sha256']) != info['samples_sha256'] or
                    str(saved['bank_sha256']) != bank_hash or
                    str(saved['registration_sha256']) != registered['registration_sha256'] or
                    hs[0].shape != amps[0].shape or done > len(data['states']) or
                    not np.isfinite(amps[0]).all() or not np.isfinite(hs[0]).all()):
                    raise ValueError('partial H checkpoint identity/shape/values mismatch')
            for i in range(done, len(data['states']), spec['H_microbatch']):
                x = data['states'][i:i+spec['H_microbatch']]
                if ledger['residual_H_attempted_centers']+len(x) > spec['maximum_residual_H_attempted_centers']:
                    raise RuntimeError('registered residual-H attempted-center budget exhausted')
                ledger['residual_H_attempted_centers'] += len(x)
                progress(f'{label}_H_batch_started_{i}', announce=False)
                a, h = map(np.asarray, action(bank, x))
                if not np.isfinite(a).all() or not np.isfinite(h).all():
                    raise FloatingPointError('nonfinite residual full-H action')
                amps.append(a); hs.append(h)
                _atomic_npz(file, amplitudes=np.concatenate(amps), h_amplitudes=np.concatenate(hs),
                    samples_sha256=info['samples_sha256'], bank_sha256=bank_hash,
                    registration_sha256=registered['registration_sha256'])
                ledger['residual_H_completed_centers'] += len(x)
                progress(f'{label}_H_saved_{i+len(x)}', announce=(i+len(x)) % 16 == 0)
            return np.concatenate(amps), np.concatenate(hs)
        ref, href = hvalues(banks['reference'], 'reference')
        dv, hdv = hvalues(banks['diagonal'], 'diagonal')
        cv, hcv = hvalues(banks['control'], 'control')
        # An archive commit can precede an interruption before its progress
        # write. Validated checkpoint lengths, not a stale progress counter,
        # therefore define completed work; attempted calls stay conservative.
        committed_centers = len(ref)+len(dv)+len(cv)
        if ledger['residual_H_attempted_centers'] < committed_centers:
            raise ValueError('committed H checkpoints exceed the charged attempted work')
        ledger['residual_H_completed_centers'] = committed_centers
        progress('committed_H_checkpoint_counts_reconciled')
        c = data['carrier']; hc = data['hcarrier']*spec['n_sites']
        basis = np.column_stack((c, ref, c+dv-ref, c+cv-ref))
        hbasis = np.column_stack((hc, href, hc+hdv-href, hc+hcv-href))
        psi = basis@coefficients; hp = hbasis@coefficients
        _atomic_npz(dest/'values.npz', **data, basis=basis, h_basis=hbasis, psi=psi, hpsi=hp,
            coefficients=coefficients, basis_names=np.asarray(spec['basis_names']), arm_names=np.asarray(spec['arm_names']))
        report = paired_chain_audit(psi, hp, data['q'], data['groups'], n_sites=spec['n_sites'],
            bootstrap_replicates=spec['bootstrap_replicates'], seed=spec['bootstrap_seed'], arm_names=spec['arm_names'])
        report.update(identity=dict(registration_sha256=registered['registration_sha256'], stage='audit'),
            fit_seal=seal, sampling=info, group_order=group_order,
            primary_contrast=spec['primary_contrast'], secondary_contrast=spec['secondary_contrast'],
            no_best_arm_selection=True, coefficients_refit_on_audit=False,
            parent_distances={name:weighted_state_distance(c, psi[:, k], data['q'])
                              for k, name in enumerate(spec['arm_names'])},
            ledger=ledger, unique_residual_H_centers=3*len(c), shared_carrier_H_centers=len(c),
            expected_raw_projected_amplitude_centers=3*len(c)*(1+len(bonds)),
            seconds=prior_seconds+perf_counter()-start, peak_rss_bytes=peak_rss_bytes(),
            unchanged_architecture=True, network_retrained=False,
            inherited_network_training=fit['inherited_network_training'],
            production_authorized=False, n64_authorized=False,
            files={p.name:file_sha256(p) for p in dest.glob('*.npz')})
        progress('audit_statistics_complete_before_seal')
        atomic_json(dest/'results.json', report)
        progress('new_audit_complete_no_production_release')
        print(json.dumps({name:report['paired'][name] for name in (spec['primary_contrast'], spec['secondary_contrast'])}, indent=2), flush=True)
    except Exception as error:
        atomic_json(dest/'failure.json', dict(error=repr(error), seconds=prior_seconds+perf_counter()-start, ledger=ledger))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('register', 'fit', 'audit'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); spec = json.loads(args.config.read_text())
    if args.stage == 'register':
        if args.resume: parser.error('--resume is only for audit')
        path = ROOT/spec['output']/'registration.json'
        if path.exists(): raise FileExistsError('registration already sealed')
        atomic_json(path, registration(spec)); print(path)
    else:
        run(args.config, args.stage, resume=args.resume)


if __name__ == '__main__':
    main()
