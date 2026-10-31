"""Post-hoc carrier/reference/innovation geometry from SEALED saved arrays.

No new configurations, neural amplitudes, JVPs, Hamiltonian actions, training,
or best-step selection. The old audit is now viewed data, never a new gate.
The CLI reads the existing matched-training artifact layout and preserves it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.vmc.carrier_residual_geometry import (
    analyze_residual_geometry, leave_one_group_geometry, real_tangent_span_fraction,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT/'projects/clebsch_tree_nqs/results/spin_schur_control_training_v1'
ARMS = ('diagonal', 'control')


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def configuration_orbits(states: np.ndarray, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label exact configurations and spatial orbits WITHOUT reweighting them.

    Site zero remains the most significant bit. Repeated observations retain
    their original Q weights; an orbit is not an independent statistical unit.
    """
    bits, permutations = np.asarray(states), np.asarray(images)
    if bits.ndim != 2 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError('states must be a matrix of physical binary strings')
    if (permutations.ndim != 2 or permutations.shape[1] != bits.shape[1]
            or not np.issubdtype(permutations.dtype, np.integer)
            or not np.all(np.sort(permutations, axis=1) == np.arange(bits.shape[1]))):
        raise ValueError('images must contain valid site permutations')
    keys = [np.packbits(row, bitorder='big').tobytes().hex() for row in bits]
    orbits = []
    for row in bits:
        packed = np.packbits(row[permutations], axis=1, bitorder='big')
        orbits.append(min(image.tobytes() for image in packed).hex())
    return np.asarray(keys), np.asarray(orbits)


def label_mass(fractions: np.ndarray, labels: np.ndarray) -> dict:
    """Aggregate the ORIGINAL nonnegative row contributions, not unique means."""
    values = np.asarray(fractions, dtype=float)
    if values.shape != np.asarray(labels).shape or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('need matching labels and nonnegative finite row fractions')
    if not np.isclose(values.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError('row fractions must sum to one')
    unique, inverse = np.unique(labels, return_inverse=True)
    totals = np.bincount(inverse, weights=values)
    return dict(unique_labels=len(unique), max_fraction=float(totals.max()),
                top_label=str(unique[np.argmax(totals)]),
                effective_label_count=float(1/np.sum(totals**2)))


def _scalar_summary(report: Any) -> Any:
    """Remove long row profiles from LOCO JSON; retain all scalar diagnostics."""
    if isinstance(report, dict):
        return {k:_scalar_summary(v) for k,v in report.items() if k not in ('row_fractions', 'group_fractions')}
    if isinstance(report, list):
        return [_scalar_summary(v) for v in report]
    return report


def load_saved_corpora(source: Path) -> tuple[dict, dict, dict, list[Path]]:
    """Replay units/order at explicit boundaries; never recompute a wavefunction."""
    reg = read_json(source/'registration.json'); spec = reg['config']
    if reg['registration_sha256'] != canonical_sha256({k:v for k,v in reg.items() if k!='registration_sha256'}):
        raise ValueError('invalid original registration seal')
    for section in ('inputs','sources'):
        for name,digest in reg[section].items():
            if file_sha256(ROOT/name)!=digest: raise ValueError('original registered input/source changed')
    if spec['arms'] != list(ARMS):
        raise ValueError('this artifact adapter expects the two named estimator arms')
    folder = source/'n36'; prepare = read_json(folder/'prepare/results.json')
    audit_result = read_json(folder/'audit/results.json')
    for stage in ('prepare', 'audit'):
        result = read_json(folder/stage/'results.json')
        for name,digest in result['files'].items():
            if file_sha256(folder/stage/name) != digest:
                raise ValueError(f'changed sealed {stage} array: {name}')
    for name,digest in read_json(folder/'audit/inputs.json')['files'].items():
        if file_sha256(folder/name) != digest:
            raise ValueError('changed sealed endpoint input')
    metric = read_arrays(folder/'prepare/metric.npz')
    endpoint = read_arrays(folder/'train/metric_values.npz')
    fresh = read_arrays(folder/'prepare/calibration.npz')
    oldroot = ROOT/spec['known_metric_parent']/'n36'
    old_result = read_json(oldroot/'results.json')
    if old_result['parameters_sha256'] != prepare['initial_sha256']:
        raise ValueError('saved JVPs use a different initial parameter point')
    for name in ('calibration_values.npz','audit_values.npz'):
        if file_sha256(oldroot/name)!=old_result['artifacts'][name]:
            raise ValueError('old JVP artifact changed')
    old = read_arrays(oldroot/'audit_values.npz')
    old_cal = read_arrays(oldroot/'calibration_values.npz')
    old_scale = old_result['carrier_scale']/prepare['carrier_scale']
    if not np.array_equal(metric['states'][128:],old['states']):
        raise ValueError('known corpus lost exact row alignment')
    if not np.array_equal(metric['states'][:128],fresh['states']):
        raise ValueError('fresh corpus lost exact row alignment')
    replay = dict(carrier=float(np.max(abs(metric['carrier'][128:]-old['carrier']*old_scale))),
                  reference=float(np.max(abs(metric['reference'][128:]-old['amplitudes']))))
    if replay['carrier'] > 1e-12 or replay['reference'] > 1e-15:
        raise ValueError('known carrier/reference units do not replay')
    candidates = np.column_stack([endpoint[a] for a in ARMS])
    corpora = {}
    for k,name,data in ((0,'fresh_calibration',fresh),(1,'known_development',old)):
        select = metric['corpus'] == k
        corpora[name] = dict(states=metric['states'][select], q=metric['q'][select],
            carrier=metric['carrier'][select], candidates=candidates[select],
            reference=metric['reference'][select], groups=data['groups'])
    # This is a fixed equal-Gram mixture, NOT one new Q sample stream.
    corpora['combined_development'] = dict(states=metric['states'],q=metric['q'],
        carrier=metric['carrier'],candidates=candidates,reference=metric['reference'])
    audit = read_arrays(folder/'audit/values.npz')
    ref = read_arrays(folder/'audit/reference_H.npz')
    corpora['viewed_audit'] = dict(states=audit['states'],q=audit['q'],carrier=audit['psi'][:,0],
        candidates=audit['psi'][:,1:],reference=ref['amplitudes'],
        hcarrier=audit['hpsi'][:,0],hcandidates=audit['hpsi'][:,1:],href=ref['h_amplitudes'],
        groups=audit['groups'])
    for k,arm in enumerate(ARMS,1):
        end = read_arrays(folder/'audit'/(arm+'_H.npz'))
        replay[arm+'_amplitude'] = float(np.max(abs(audit['psi'][:,k]-(audit['carrier']+end['amplitudes']-ref['amplitudes']))))
        replay[arm+'_H'] = float(np.max(abs(audit['hpsi'][:,k]-(36*audit['hcarrier']+end['h_amplitudes']-ref['h_amplitudes']))))
    if max(v for k,v in replay.items() if k.endswith(('_H','_amplitude'))) > 1e-12:
        raise ValueError('saved endpoint H identity failed')
    probes = dict(calibration=old_cal,known=old,carrier_rescale=old_scale,
                  names=old_result['direction_names'])
    paths = [source/'registration.json',folder/'prepare/results.json',folder/'prepare/metric.npz',
        folder/'prepare/calibration.npz',folder/'train/results.json',folder/'train/metric_values.npz',
        folder/'audit/results.json',folder/'audit/inputs.json',folder/'audit/values.npz',
        folder/'audit/reference_H.npz',oldroot/'results.json',oldroot/'calibration_values.npz',
        oldroot/'audit_values.npz']+[folder/'audit'/(a+'_H.npz') for a in ARMS]
    meta = dict(replay=replay,residual_scale=prepare['residual_scale'],
                previous_gate=audit_result['gate'],initial_sha256=prepare['initial_sha256'])
    return corpora,probes,meta,paths


def data_kwargs(data: dict) -> dict:
    return {key:data[key] for key in ('reference','hcarrier','hcandidates','href','groups') if key in data}


def run(source: Path, output: Path) -> dict:
    """Bounded analysis-only entry point. Refuses to overwrite completed output."""
    if (output/'results.json').exists():
        raise FileExistsError('analysis result already sealed; use a different output')
    start = perf_counter()
    corpora,probes,meta,paths = load_saved_corpora(source)
    manifest = dict(inputs={str(p.relative_to(ROOT)):file_sha256(p) for p in paths},
        sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')),
        protocol='Post-hoc saved-array geometry, not preregistered validation. No new physical samples or H calls.')
    manifest['sha256'] = canonical_sha256(manifest)
    atomic_json(output/'manifest.json',manifest)
    reports = {}
    for name,data in corpora.items():
        args = (data['carrier'],data['candidates'],data['q'])
        kwargs = dict(n_sites=36,**data_kwargs(data))
        if 'groups' in data:
            full = leave_one_group_geometry(*args,**kwargs)
        else:
            full = dict(pooled=analyze_residual_geometry(*args,**kwargs),
                        leave_one_group_out=[], reason='mixture is a fixed development metric; no pooled-chain inference')
        reports[name] = _scalar_summary(full)
    # Separate corpus projections of PRE-TRAINING directions: not full Jacobian rank.
    probe_reports = {}
    for name,data in (('prior_calibration',probes['calibration']),('known_development',probes['known'])):
        c = data['carrier']*probes['carrier_rescale']; t = data['tangents'];q=data['q']
        projected = t-c[:,None]*(np.sum(c.conj()[:,None]*t/q[:,None],axis=0)/np.sum(abs(c)**2/q))
        ref = data['amplitudes']-c*np.sum(c.conj()*data['amplitudes']/q)/np.sum(abs(c)**2/q)
        innovation = projected-ref[:,None]*np.sum(ref.conj()[:,None]*projected/q[:,None],axis=0)/np.sum(abs(ref)**2/q)
        tangent_norms = np.sum(abs(projected)**2/q[:,None],axis=0)
        if np.any(tangent_norms <= 0): raise ValueError('undefined saved tangent')
        normalized = innovation/np.sqrt(q[:,None]*tangent_norms[None,:])
        norms = np.linalg.norm(normalized,axis=0)
        sv = np.linalg.svd(np.concatenate((normalized.real,normalized.imag),axis=0),compute_uv=False)
        probe_reports[name] = dict(innovation_norm_fraction=(norms**2).tolist(),
            real_innovation_singular_values=sv.tolist(),
            normalization='each carrier-orthogonal parent JVP has unit Q norm before reference removal',
            interpretation='Only six saved random JVPs, not all15315 parameter directions; no energy conclusion.')
    known = corpora['known_development']
    finite_span = real_tangent_span_fraction(known['carrier'],known['candidates'],probes['known']['tangents'],known['q'])
    # All observations/weights are kept, including the largest residual event.
    observed = corpora['viewed_audit'];c=observed['carrier'];v=observed['candidates'];q=observed['q']
    z=np.sum(abs(c)**2/q); b=np.sum(c.conj()[:,None]*v/q[:,None],axis=0)/z
    residual=v-c[:,None]*b
    mass=abs(residual)**2/q[:,None];fraction=mass/mass.sum(axis=0)
    images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),dtype=int)
    configs,orbits=configuration_orbits(observed['states'],images)
    reference=observed['reference']-c*np.sum(c.conj()*observed['reference']/q)/z
    force=observed['hcarrier']-c*np.sum(c.conj()*observed['hcarrier']/q)/z
    def normalized_mass(x):
        w=abs(x)**2/q
        return w/w.sum()
    profiles=dict(states=observed['states'],q=q,groups=observed['groups'],
        carrier=normalized_mass(c),reference=normalized_mass(reference),force=normalized_mass(force),
        **{arm:fraction[:,k] for k,arm in enumerate(ARMS)})
    events={}
    for k,arm in enumerate(ARMS):
        i=int(np.argmax(fraction[:,k]));ratio=None if c[i]==0 else v[i,k]/c[i]
        events[arm]=dict(row=i,chain=int(observed['groups'][i]),state=observed['states'][i].tolist(),
            weighted_fraction=float(fraction[i,k]),raw_squared_fraction=float(abs(residual[i,k])**2/np.sum(abs(residual[:,k])**2)),
            proposal_midrank_percent=float(100*(np.sum(q<q[i])+.5*np.sum(q==q[i]))/len(q)),
            carrier_weight_fraction=float(profiles['carrier'][i]),
            endpoint_over_carrier=None if ratio is None else [float(ratio.real),float(ratio.imag)],
            configuration=label_mass(fraction[:,k],configs),orbit=label_mass(fraction[:,k],orbits))
    chart = chart_replay(corpora,meta,probes)
    np.savez_compressed(output/'profiles.npz',**profiles)
    result=dict(manifest_sha256=manifest['sha256'],arms=list(ARMS),corpora=reports,
        probe_names=probes['names'],pretraining_probes=probe_reports,six_probe_span=finite_span,
        events=events,chart=chart,replay=meta['replay'],source_gate_passed=meta['previous_gate']['passed'],
        resources=dict(new_physical_samples=0,new_neural_amplitude_calls=0,new_H_calls=0,new_JVP_calls=0,
                       training_steps=0,size_increase=False),
        post_hoc=True,new_variational_win_claim=False,production_authorized=False,
        seconds=perf_counter()-start,files={'profiles.npz':file_sha256(output/'profiles.npz')})
    result['sha256']=canonical_sha256(result)
    atomic_json(output/'results.json',result)
    print(json.dumps(dict(replay=result['replay'],chart=chart,events=events,seconds=result['seconds']),indent=2))
    return result


def chart_replay(corpora: dict, meta: dict, probes: dict) -> dict:
    """Check the new coordinate interface on OLD physical endpoints, not train."""
    import jax
    jax.config.update('jax_enable_x64',True)
    from su2_nqs.models.carrier_residual_chart import (
        make_residual_chart_anchor, fit_residual_chart_coefficients,
        residual_chart_amplitudes, additive_ray_parameters,
    )
    metric=corpora['combined_development']; audit=corpora['viewed_audit'];scale=meta['residual_scale']
    anchor=make_residual_chart_anchor(metric['carrier'],metric['reference'],metric['q'])
    zero=np.zeros_like(metric['carrier'],dtype=complex)
    zero_coeff=fit_residual_chart_coefficients(anchor,zero)
    initialized=residual_chart_amplitudes(anchor,zero_coeff,metric['carrier'],metric['reference'],zero)
    if not np.array_equal(np.asarray(initialized.amplitudes),metric['carrier']):
        raise ValueError('chart initialization changed the strong carrier')
    rows={}
    for k,arm in enumerate(ARMS):
        f=scale*(metric['candidates'][:,k]-metric['carrier'])
        coeff=fit_residual_chart_coefficients(anchor,f)
        converted=additive_ray_parameters(coeff,scale)
        if not bool(converted.valid): raise ValueError('old endpoint outside the carrier chart')
        mix,gain,carrier_coefficient=converted.mix,converted.gain,converted.carrier_coefficient
        row={}
        for name,data in (('development',metric),('viewed_audit',audit)):
            ref=data['reference'] if name=='development' else data['reference']*scale
            difference=scale*(data['candidates'][:,k]-data['carrier'])
            state,innovation,_=residual_chart_amplitudes(anchor,coeff,data['carrier'],ref,difference,mix=mix,gain=gain)
            state,innovation=np.asarray(state),np.asarray(innovation)
            error=float(np.max(abs(state*complex(carrier_coefficient)-data['candidates'][:,k])))
            relative=error/max(float(np.max(abs(data['candidates'][:,k]))),1e-300)
            if relative>1e-11:raise ValueError('coordinate chart changed the physical endpoint')
            row[name+'_relative_amplitude_replay_error']=relative
            if name=='viewed_audit':
                hp,_,_=residual_chart_amplitudes(anchor,coeff,data['hcarrier'],data['href']*scale,
                    scale*(data['hcandidates'][:,k]-data['hcarrier']),mix=mix,gain=gain)
                row['H_relative_replay_error']=float(np.max(abs(np.asarray(hp)*complex(carrier_coefficient)-data['hcandidates'][:,k]))/
                    max(float(np.max(abs(data['hcandidates'][:,k]))),1e-300))
                if row['H_relative_replay_error']>1e-11: raise ValueError('coordinate chart changed saved H action')
        rows[arm]=row
    # Linearized chart acts on ALREADY SAVED model JVPs. Fit once on prior64,
    # apply the same global coefficients to known256, without new JVP calls.
    cal=probes['calibration']; scale_c=probes['carrier_rescale']
    probe_anchor=make_residual_chart_anchor(cal['carrier']*scale_c,cal['amplitudes'],cal['q'])
    probe_transfer={name:[] for name in ('prior_calibration','known_development')}
    for k in range(cal['tangents'].shape[1]):
        coefficient=fit_residual_chart_coefficients(probe_anchor,cal['tangents'][:,k])
        for name,data in (('prior_calibration',cal),('known_development',probes['known'])):
            c=data['carrier']*scale_c;ref=data['amplitudes'];q=data['q'];t=data['tangents'][:,k]
            innovation=np.asarray(residual_chart_amplitudes(probe_anchor,coefficient,c,ref,t).innovation)
            zc=np.sum(abs(c)**2/q)
            ref_perp=ref-c*np.sum(c.conj()*ref/q)/zc
            t_perp=t-c*np.sum(c.conj()*t/q)/zc
            zi=np.sum(abs(innovation)**2/q)
            span_norm=abs(np.sum(c.conj()*innovation/q))**2/zc+abs(np.sum(ref_perp.conj()*innovation/q))**2/np.sum(abs(ref_perp)**2/q)
            probe_transfer[name].append(dict(
                remaining_norm_relative_to_original_carrier_perp=float(zi/np.sum(abs(t_perp)**2/q)),
                anchor_plane_fraction=None if zi==0 else float(span_norm/zi)))
    return dict(endpoints=rows,coefficients_fit_once_on='combined_development',
        initial_carrier_exact=True,prior64_fitted_JVP_transfer=probe_transfer,
        saved_endpoints_preserved=True,global_manifold_equivalence_not_asserted=True,
        shape_gain_initially_nonzero=True,new_training=False,
        interpretation='Tied coordinates exactly replay old states. Untying mix/gain is a future training choice, not a proven equivalent family or a win.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    run(args.source,args.output or args.source/'structure')


if __name__=='__main__':
    main()
