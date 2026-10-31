"""Zero-new-state analysis of saved candidate quality sequences.

Uses independent saved physical amplitudes/coherent regional RDMs. No ED,
training, new samples, amplitude queries, SU2 twirling, or spectrum repair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from su2_nqs.benchmarks.singlet_physics_readout import (
    compact6_channel_readout, importance_jackknife_observables,
    singlet_structure_factor_records, triangular_torus_structure_records,
)
from su2_nqs.benchmarks.spin_operator_information import three_bond_readout
from su2_nqs.benchmarks.spin_partition_readout import spin_partition_readout
from su2_nqs.benchmarks.tensor_network_readout import physical_spin_block_readout

ROOT = Path(__file__).resolve().parents[3]


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(_plain(value), indent=2, allow_nan=False)+'\n')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measurements(center: dict, deletes: list[dict]) -> dict:
    count = len(deletes)
    result = {}
    for name, value in center.items():
        x = np.array([row[name] for row in deletes])
        se = np.sqrt((count-1)*np.mean((x-x.mean())**2))
        result[name] = dict(value=value, chain_SE=se, jackknife_values=x)
    return result


def _paired(before: dict, after: dict) -> dict:
    result = {}
    for name in before.keys() & after.keys():
        jack = np.asarray(after[name]['jackknife_values'])-np.asarray(before[name]['jackknife_values'])
        result[name] = dict(value=after[name]['value']-before[name]['value'],
            chain_SE=float(np.sqrt((len(jack)-1)*np.mean((jack-jack.mean())**2))),
            jackknife_values=jack)
    return result


def _independent_difference(before: dict, after: dict) -> dict:
    """Different audit RNG/chain sets: add variances, never fake pairing."""
    result={}
    for name in before.keys() & after.keys():
        value=float(after[name]['value']-before[name]['value'])
        se=float(np.hypot(before[name]['chain_SE'],after[name]['chain_SE']))
        result[name]=dict(value=value,independent_chain_SE=se,
            absolute_signal_over_SE=abs(value)/se if se>0 else None,
            paired=False)
    return result


def _regional(rho: np.ndarray, left: list[int], *, compact: bool) -> dict:
    block = physical_spin_block_readout(rho)
    joint = spin_partition_readout(rho, left)
    values = {'spin_squared': block['spin_squared_mean'],
        **{f'p_q{q}': p for q, p in block['p_q'].items()},
        **{f'singlet_pair_q{q}': p for q, p in joint['singlet_pair_channels'].items()},
        'cross_block_spin_dot': joint['cross_block_spin_dot']}
    if rho.shape == (64,64):
        triple = three_bond_readout(rho, left)
        values.update(high_q1=triple['high_q1'], high_q3=triple['high_q3'],
            three_bridge_raw=triple['orbits']['Y']['raw'],
            three_bridge_connected=triple['orbits']['Y']['connected'])
    if compact:
        values.update(compact6_channel_readout(rho))
    return values


def _region_sequence(data, prefix: str, labels: list[str], *, name: str, left: list[int]) -> dict:
    states = {}
    diagnostics = {}
    for label in labels:
        key = prefix+name+'_'+label
        if key+'_rho' not in data.files:
            key = label
        rho = data[key+'_rho']
        jack = data[key+'_jackknife_rhos']
        states[label] = _measurements(_regional(rho, left, compact=name=='compact6'),
            [_regional(r, left, compact=name=='compact6') for r in jack])
        check = physical_spin_block_readout(rho)
        diagnostics[label] = {k: check[k] for k in (
            'su2_factorization_frobenius', 'strict_su2_eligible', 'minimum_eigenvalue')}
    pairs = [('previous_endpoint', 'endpoint'), ('original_carrier', 'endpoint')]
    if 'two_Lanczos' in labels:
        pairs = [('projected_VMC', 'two_Lanczos'), ('attention_initial', 'attention_final')]
    return dict(states=states, diagnostics=diagnostics,
        paired={after+'-minus-'+before: _paired(states[before], states[after])
            for before, after in pairs if before in labels and after in labels},
        strict_multiplicity_spectrum_reported=False,
        warning='Actual coherent sampled RDM, no twirl. Regional local channels are not global susceptibilities.')


def _structure_sequence(bits, q, groups, amplitudes: dict[str, np.ndarray]) -> dict:
    records = triangular_torus_structure_records(bits, 6, 6, strict_singlet=True)
    states = {label: importance_jackknife_observables(records, psi, q, groups)
        for label, psi in amplitudes.items()}
    pairs = [('previous_endpoint', 'endpoint'), ('original_carrier', 'endpoint')]
    if 'two_Lanczos' in states:
        pairs = [('projected_VMC', 'two_Lanczos'), ('attention_initial', 'attention_final')]
    paired = {after+'-minus-'+before: _paired(states[before]['observables'], states[after]['observables'])
        for before, after in pairs if before in states and after in states}
    for state in states.values():
        means = state['observables']
        state['raw_M_spread'] = max(means[k]['value'] for k in ('M1','M2','M3'))-min(
            means[k]['value'] for k in ('M1','M2','M3'))
        state['M_pair_differences'] = {
            f'{a}-minus-{b}': _paired({a: means[b]}, {a: means[a]})[a]
            for a, b in (('M1', 'M2'), ('M2', 'M3'), ('M1', 'M3'))}
    return dict(states=states, paired=paired,
        strict_singlet_independently_established=True,
        M_star_average_uses_exact_A1_but_raw_M_retained=True,
        warning='3Sz diagonal scalar estimator valid only for independently proved singlets. Gamma/Parseval identities do not certify Born equilibrium.')


def run(output: Path) -> dict:
    started = time.perf_counter()
    pair = ROOT/'projects/clebsch_tree_nqs/results/spin_pair_n36_unlocked_v1'
    old = ROOT/'projects/clebsch_tree_nqs/results/spin_nqs_region_readout_v1'
    small = ROOT/'projects/clebsch_tree_nqs/results/spin_competing_state_readout_v1'
    paths = [pair/'samples_states_readouts.npz', pair/'results.json', pair/'completion.json',
        old/'products.npz', old/'samples.npz', old/'compact6.npz', old/'winding6.npz',
        old/'registration.json', old/'energy_correlations.json', small/'readout.npz', small/'results.json']
    source_paths = [Path(__file__), ROOT/'su2_nqs/benchmarks/singlet_physics_readout.py',
        ROOT/'su2_nqs/benchmarks/spin_operator_information.py', ROOT/'su2_nqs/benchmarks/spin_partition_readout.py',
        ROOT/'su2_nqs/benchmarks/tensor_network_readout.py']
    output.mkdir(parents=True, exist_ok=True)
    if (output/'completion.json').exists():
        raise FileExistsError('completed saved-state analysis is immutable; choose a fresh output')
    _write(output/'registration.json', dict(inputs={str(p.relative_to(ROOT)): _sha(p) for p in paths},
        sources={str(p.relative_to(ROOT)): _sha(p) for p in source_paths},
        new_training=0, new_samples=0, new_ED=0, new_amplitude_queries=0,
        analysis_uses_existing_audits=True, phase_claim=False,
        protocol='Fixed cached quality sequences; all raw M, coherent RDMs, paired WHOLE-chain error; no endpoint or feature selection.'))
    sealed = json.loads((pair/'completion.json').read_text())
    for name, digest in sealed['outputs'].items():
        if _sha(pair/name) != digest:
            raise ValueError('sealed pair artifact changed')
    labels = ['original_carrier', 'previous_endpoint', 'finite_quadrature_control',
        'locked_hermitian_control', 'endpoint']
    data = np.load(pair/'samples_states_readouts.npz')
    pair_result = json.loads((pair/'results.json').read_text())
    audits = []
    for index in range(2):
        prefix = f'audit{index}_'
        x, q = data[prefix+'configurations'], data[prefix+'density']
        structure = _structure_sequence(x, q, np.arange(len(x))%16,
            {label: data[prefix+label+'_psi'] for label in labels})
        regions = {name: _region_sequence(data, prefix, labels, name=name, left=left)
            for name, left in (('compact6', [0,1,2]), ('winding6', [0,2,4]))}
        audits.append(dict(structure=structure, regions=regions))
    x = np.concatenate([data[f'audit{i}_configurations'] for i in range(2)])
    q = np.concatenate([data[f'audit{i}_density'] for i in range(2)])
    groups = np.concatenate([np.arange(1024)%16+16*i for i in range(2)])
    pooled = _structure_sequence(x, q, groups, {label: np.concatenate(
        [data[f'audit{i}_{label}_psi'] for i in range(2)]) for label in labels})
    oldlabels = ['projected_VMC', 'two_Lanczos', 'attention_initial', 'attention_final']
    old_samples, old_products = np.load(old/'samples.npz'), np.load(old/'products.npz')
    old_structure = _structure_sequence(old_samples['states'], old_samples['q'], old_samples['groups'],
        {label: old_products['psi'][:, i] for i, label in enumerate(oldlabels)})
    old_regions = {name: _region_sequence(np.load(old/(name+'.npz')), '', oldlabels,
        name=name, left=left) for name, left in (('compact6',[0,1,2]),('winding6',[0,2,4]))}
    # Anchoring removes a common-C offset, not sampling bias or missing tails.
    anchored = []
    for audit in audits:
        row = {}
        for name in ('compact6','winding6'):
            current = audit['regions'][name]['paired']['endpoint-minus-original_carrier']
            krylov = old_regions[name]['paired']['two_Lanczos-minus-projected_VMC']
            row[name] = {key: dict(value=current[key]['value']-krylov[key]['value'],
                independent_chain_SE=float(np.hypot(current[key]['chain_SE'],krylov[key]['chain_SE'])),
                current_within_family_shift=audit['regions'][name]['paired']['endpoint-minus-previous_endpoint'][key],
                krylov_quality_shift=krylov[key]) for key in current.keys() & krylov.keys()}
        anchored.append(row)
    # Exact saved singlet calibration; K is deliberately absent on 4x4.
    references = json.loads((small/'results.json').read_text())
    small_data = np.load(small/'readout.npz')
    calibrations = {}
    coords = np.array([divmod(i,4) for i in range(16)])
    allowed = {'Gamma':(0.,0.), 'M1':(.5,0.), 'M2':(0.,.5), 'M3':(.5,.5)}
    diagrecords = singlet_structure_factor_records(small_data['basis'], coords, allowed, strict_singlet=True)
    for label, metadata in references['states'].items():
        if metadata['physical']['total_spin_squared'] > 1e-9 or label+'_state' not in small_data.files:
            continue
        v = small_data[label+'_state']; weights = abs(v)**2/np.vdot(v,v).real
        correlations = np.array(metadata['physical']['spin_correlation_matrix'])
        structural = {key: float(weights@values) for key, values in diagrecords.items()}
        direct = {key: float(np.vdot(np.exp(2j*np.pi*(coords@k)), correlations@np.exp(2j*np.pi*(coords@k))).real/16)
            for key, k in allowed.items()}
        regionkey = label+'_strip_6_rho'
        if regionkey not in small_data.files:
            continue
        calibrations[label] = dict(provenance=metadata['provenance'], structure_factors=structural,
            full_three_component_comparison_maximum_error=max(abs(structural[k]-direct[k]) for k in direct),
            strip6_readout=_regional(small_data[regionkey], [0,1,2], compact=False),
            warning='4x4 excludes allowed K; strip6 differs from N36 regions, not finite-size phase inference.')
    result = dict(pair_quality_sequence=dict(audits=audits,pooled_structure=pooled,
        energy_variance=pair_result['pooled'], saved_milestones_have_no_independent_readouts=True),
        old_carrier_krylov_neural_sequence=dict(structure=old_structure, regions=old_regions,
            energy_variance=json.loads((old/'energy_correlations.json').read_text())['audit']['arms']),
        common_carrier_anchored_candidate_differences=anchored, exact_N16_calibrations=calibrations,
        new_training=0,new_samples=0,new_ED=0,new_full_H_calls=0,new_amplitude_queries=0,
        strict_multiplicity_spectrum_reported=False, phase_claim=False,
        local_operator_identities=['chi_triangle^2=(3/16)P(S_triangle=1/2)=3/32-(B01+B12+B20)/8',
            'sum_a E2_a^2=(3/4)P(S_triangle=1/2)=3/8-(B01+B12+B20)/2=4chi_triangle^2',
            'chi_A chi_B=det[B_ai,bj], with oriented disjoint triples; signed sum of six bridge matchings',
            'For triangle q=1 multiplicity C^2, normalized E2 components and chi form three pseudospin axes.'],
        redundancy_ledger={
            'triangle_total_spin_weight': 'p(q=1) is fixed by all three triangle pair correlations.',
            'local_E2_norm_squared': 'Two-body only; exactly 4*chi_squared, not independent nematic information.',
            'local_chi_squared': 'Two-body only; exactly (3/16)*p(q=1), not independent chirality evidence.',
            'intertriangle_E2_dot': 'Four-body disjoint-dimer channel; organizes but does not add information beyond complete dimer correlations.',
            'disjoint_triangle_chi_chi': 'Pure six-body, orthogonal to all full-support operators on at most four sites; signed bridge resonance, not GMN or topology.',
            'joint_spin_high_part': 'Hilbert-Schmidt remainder outside complete two/four-body scalar span, not a phase witness.',
        },
        scope=['Global K/ALL M from singlet isotropy, local coherent compact6/winding6 readouts.',
            'Local E2 norm and local chi^2 are redundant pair information.',
            'Intertriangle E2 correlation reorganizes four-body dimer information; disjoint chi-chi is six-body.',
            'No full-lattice E2 susceptibility, long-range dimer matrix, strict sampled entanglement spectrum, or phase identification.'],
        warnings=['M-star identities and static finite-size proxies do not establish long-range order.',
            'Different cached proposals/chains may have different bias; a common-C subtraction is not equilibration proof.',
            'Independent chain SE excludes systematic Born sampling/tail bias.',
            'Within-family refinement drift is empirical, not a certified error bound.'], seconds=time.perf_counter()-started)
    _write(output/'results.json', result)
    _write(output/'completion.json', dict(status='complete', seconds=time.perf_counter()-started,
        results_sha256=_sha(output/'results.json'), registration_sha256=_sha(output/'registration.json')))
    return result


def _cached_moment_sequence(arrays: dict, density: np.ndarray, groups: np.ndarray,
        *, n_sites: int) -> dict:
    """Reuse node-safe full-H moments with explicit whole-chain IDs."""
    from .spin_adaptive_pair_carrier import _pair_weighted_moments
    states={}
    for label,(a,b) in arrays.items():
        values=_pair_weighted_moments(a,b,density,n_sites)
        jack=[_pair_weighted_moments(a[groups!=c],b[groups!=c],density[groups!=c],n_sites)
            for c in np.unique(groups)]
        states[label]=_measurements(dict(zip(('energy_per_site','variance_per_site'),values)),
            [dict(zip(('energy_per_site','variance_per_site'),row)) for row in jack])
    pairs=[('previous_endpoint','endpoint'),('original_carrier','endpoint')]
    return dict(states=states,paired={after+'-minus-'+before:_paired(states[before],states[after])
        for before,after in pairs if before in states and after in states})


def _cached_residual_tails(a: np.ndarray, b: np.ndarray, q: np.ndarray,
        groups: np.ndarray, *, n_sites: int) -> dict:
    """Concentration of the ACTUAL residual moment, with nodes retained.

    This is descriptive saved-data analysis; no trimming defines an energy,
    variance, physical state, or subsequent training decision.
    """
    weights=abs(a)**2/q; norm=weights.sum()
    energy=float(np.sum(a.conj()*b/q).real/norm)
    residual=abs(b-energy*a)**2/q
    total=float(residual.sum())
    ordered=np.sort(residual)
    def share(fraction):
        count=max(1,int(np.ceil(len(a)*fraction)))
        return float(ordered[-count:].sum()/total) if total else 0.
    group_contributions=np.array([residual[groups==c].sum() for c in np.unique(groups)])
    return dict(energy_per_site=energy/n_sites,variance_per_site=total/norm/n_sites,
        top_0p1_percent_residual_share=share(.001),top_1_percent_residual_share=share(.01),
        top_5_percent_residual_share=share(.05),
        largest_record_residual_share=float(ordered[-1]/total) if total else 0.,
        largest_chain_residual_share=float(group_contributions.max()/total) if total else 0.,
        effective_residual_contributions=float(total**2/(residual@residual)) if total else 0.,
        exact_node_records=int(np.count_nonzero(a==0)),
        residual_share_on_exact_nodes=float(residual[a==0].sum()/total) if total else 0.,
        no_records_trimmed=True,unseen_tail_coverage_certified=False)


def _cached_early_selector(groups: np.ndarray) -> np.ndarray:
    """Select the first half IN EACH saved chain, preserving temporal order."""
    early=np.zeros(len(groups),bool)
    for label in np.unique(groups):
        indices=np.flatnonzero(groups==label)
        if len(indices)<4 or len(indices)%2:
            raise ValueError('each audit chain requires an even count of at least four records')
        early[indices[:len(indices)//2]]=True
    return early


def run_pair_source(source: Path, output: Path) -> dict:
    """Analyze any completed N36 pair run, with ZERO new physical queries.

    Independent audit streams are distinct chains, not distinct optimized
    seeds. Explicit IDs avoid accidental covariance across different streams.
    Original training outputs and their completion seal are never rewritten.
    """
    started=time.perf_counter(); source=source.resolve(); output=output.resolve()
    if source==output or source in output.parents:
        raise ValueError('diagnostics output must be separate from the sealed source run')
    sealed=json.loads((source/'completion.json').read_text())
    for name,digest in sealed['outputs'].items():
        if _sha(source/name)!=digest:
            raise ValueError('sealed pair artifact changed: '+name)
    metadata=json.loads((source/'results.json').read_text())
    spec=metadata['config']; chains=spec['chains']; regions=spec['regions']
    if spec['n_sites']!=36 or metadata['parameter_count']!=630:
        raise ValueError('only saved complete N36 projected singlet AGP audits are supported')
    labels=list(json.loads((source/'frozen_endpoint.json').read_text())['checkpoints'])
    count=len(metadata['audits'])
    if count<1:
        raise ValueError('at least one completed independent audit required')
    identity=None
    if spec.get('starting_checkpoint'):
        from su2_nqs.vmc.checkpoint import load_checkpoint
        starting_path=ROOT/spec['starting_checkpoint']['path']
        if _sha(starting_path)!=spec['starting_checkpoint']['sha256']:
            raise ValueError('registered starting checkpoint seal changed')
        before=load_checkpoint(starting_path);restored=load_checkpoint(source/'previous_endpoint.pkl')
        pa=np.asarray(before['params']['pairing_real']);pb=np.asarray(restored['params']['pairing_real'])
        if not np.array_equal(pa,pb) or before['metadata']['amplitude_scale']!=restored['metadata']['amplitude_scale']:
            raise ValueError('warm-start physical pairing/amplitude scale changed')
        identity=dict(pairing_array_equal=True,maximum_pairing_error=float(np.max(abs(pa-pb))),
            parameter_shape=list(pa.shape),parameter_dtype=str(pa.dtype),
            amplitude_scale_equal=True,amplitude_scale=before['metadata']['amplitude_scale'],
            starting_H_j2=before['metadata'].get('j2'),audit_H_j2=restored['metadata'].get('j2'),
            physical_definition='The same packed symmetric F, scale, N36 and full432 GammaA1 decoder. J2 occurs only in H, not in amplitudes.',
            new_amplitude_queries=0)
    output.mkdir(parents=True,exist_ok=False)
    from shutil import copy2
    copy2(Path(__file__),output/'workflow_snapshot.py')
    dependencies=[Path(__file__),ROOT/'projects/clebsch_tree_nqs/workflows/spin_adaptive_pair_carrier.py',
        ROOT/'su2_nqs/benchmarks/singlet_physics_readout.py',
        ROOT/'su2_nqs/benchmarks/spin_operator_information.py',
        ROOT/'su2_nqs/benchmarks/spin_partition_readout.py',
        ROOT/'su2_nqs/benchmarks/tensor_network_readout.py',
        ROOT/'su2_nqs/benchmarks/regional_su2_readout.py']
    for dependency in dependencies:
        snapshot=output/'source_snapshots'/dependency.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        copy2(dependency,snapshot)
    paths=[source/'completion.json',source/'results.json',source/'samples_states_readouts.npz',
        source/'registration.json',source/'frozen_endpoint.json']
    _write(output/'registration.json',dict(inputs={str(p):_sha(p) for p in paths},
        source_workflow_sha256=_sha(Path(__file__)),j2_over_j1=spec['j2_over_j1'],
        dependency_sources={str(p.relative_to(ROOT)):_sha(p) for p in dependencies},
        new_training=0,new_samples=0,new_ED=0,new_full_H_calls=0,new_amplitude_queries=0,
        protocol='Saved independent audit chains only; no tail trimming or endpoint selection.'))
    with np.load(source/'samples_states_readouts.npz') as data:
        audits=[]; pooled_x=[]; pooled_q=[]; pooled_groups=[]
        pooled_arrays={label:([],[]) for label in labels}
        for index in range(count):
            prefix=f'audit{index}_'; bits=data[prefix+'configurations']; q=data[prefix+'density']
            groups=np.arange(len(q))%chains
            arrays={label:(data[prefix+label+'_psi'],data[prefix+label+'_Hpsi']) for label in labels}
            structure=_structure_sequence(bits,q,groups,{label:a for label,(a,_) in arrays.items()})
            region_readouts={name:_region_sequence(data,prefix,labels,name=name,left=region['left'])
                for name,region in regions.items()}
            audits.append(dict(structure=structure,regions=region_readouts,
                energy_variance=_cached_moment_sequence(arrays,q,groups,n_sites=36)))
            pooled_x.append(bits);pooled_q.append(q);pooled_groups.append(groups+index*chains)
            for label,(a,b) in arrays.items():
                pooled_arrays[label][0].append(a);pooled_arrays[label][1].append(b)
        bits=np.concatenate(pooled_x);q=np.concatenate(pooled_q);groups=np.concatenate(pooled_groups)
        arrays={label:(np.concatenate(a),np.concatenate(b)) for label,(a,b) in pooled_arrays.items()}
        pooled=_structure_sequence(bits,q,groups,{label:a for label,(a,_) in arrays.items()})
        tails={label:_cached_residual_tails(a,b,q,groups,n_sites=36) for label,(a,b) in arrays.items()}
        early=_cached_early_selector(groups); halves={}
        for name,keep in (('early',early),('late',~early)):
            subset={label:(a[keep],b[keep]) for label,(a,b) in arrays.items()}
            halves[name]=dict(energy_variance=_cached_moment_sequence(subset,q[keep],groups[keep],n_sites=36),
                structure=_structure_sequence(bits[keep],q[keep],groups[keep],
                    {label:a for label,(a,_) in subset.items()}))
        drifts={label:dict(
            energy_variance=_paired(halves['early']['energy_variance']['states'][label],
                halves['late']['energy_variance']['states'][label]),
            structure=_paired(halves['early']['structure']['states'][label]['observables'],
                halves['late']['structure']['states'][label]['observables'])) for label in labels}
    result=dict(j2_over_j1=spec['j2_over_j1'],audits=audits,pooled_structure=pooled,
        pooled_energy_variance=_cached_moment_sequence(arrays,q,groups,n_sites=36),
        residual_tails=tails,early_late=dict(halves=halves,late_minus_early=drifts),
        independent_audit_streams=count,independent_chains=count*chains,optimized_initializations=1,
        cross_coupling_warm_start=spec.get('starting_checkpoint'),
        warm_start_physical_identity=identity,
        new_training=0,new_samples=0,new_ED=0,new_full_H_calls=0,new_amplitude_queries=0,
        phase_claim=False,strict_multiplicity_spectrum_reported=False,equilibrium_certified=False,
        warnings=['All means are finite-sample self-normalized ratios.',
            'Whole-chain SE does not bound unseen tails or equilibrium bias.',
            'Finite-size m120 proxy is not an infinite-system onsite magnetization.',
            'Independent audit streams are not independent optimized initializations.'],
        seconds=time.perf_counter()-started)
    _write(output/'results.json',result)
    _write(output/'completion.json',dict(status='complete',outputs={
        str(path.relative_to(output)):_sha(path) for path in output.rglob('*')
        if path.is_file() and path.name!='completion.json'}))
    return result


def run_vit_source(source: Path, output: Path) -> dict:
    """Saved full-ViT Born audits: fixed cuts, scalar K/M and six-spin drift.

    No state/sampling/optimizer changes or amplitude queries. Read original
    seals rather than re-registering an archived workflow with today's code.
    Strict multiplicity spectra are not inferred from noninvariant RDMs.
    """
    from .spin_spatial_vit_quality import read_sealed
    started=time.perf_counter(); source=source.resolve(); output=output.resolve()
    if source==output or source in output.parents:
        raise ValueError('keep derived analysis outside the sealed source run')
    registration=read_sealed(source/'registration.json'); spec=registration['config']
    if spec['model']['rows']!=6 or spec['model']['columns']!=6 or spec['model']['geometry']!='triangular':
        raise ValueError('this physical comparison requires the triangular N36 K/all-M torus')
    paths=[source/'registration.json']; reports=[]; seen_seeds=set()
    for folder in sorted((source/'audits').glob('step*')):
        if not (folder/'results.json').is_file():
            continue
        report=read_sealed(folder/'results.json')
        checkpoint=source/'quality'/f'step{report["step"]:05d}.pkl'
        if (report['registration_sha256']!=registration['result_sha256'] or
                _sha(checkpoint)!=report['checkpoint_sha256']):
            raise ValueError('frozen audit/checkpoint identity changed')
        seeds={stream['seed'] for stream in report['streams']}
        if seen_seeds & seeds:
            raise ValueError('different milestones must use disjoint audit seeds')
        seen_seeds.update(seeds); paths.append(folder/'results.json')
        regional={}
        for name,cut in spec['regions'].items():
            manifest=read_sealed(folder/(name+'.json')); raw=folder/(name+'.npz')
            if _sha(raw)!=manifest['raw_sha256'] or manifest['result_sha256']!=report['regions'][name]['result_sha256']:
                raise ValueError('saved coherent regional artifact changed')
            paths.extend([folder/(name+'.json'),raw])
            with np.load(raw) as data:
                if list(data['sites'])!=cut['sites']:
                    raise ValueError('physical region/site order differs')
                rho=data['rho']; deletes=data['jackknife_rhos']
                values=_measurements(_regional(rho,cut['left'],compact=name=='compact6'),
                    [_regional(r,cut['left'],compact=name=='compact6') for r in deletes])
                diagnostic=physical_spin_block_readout(rho)
            regional[name]=dict(sites=cut['sites'],left=cut['left'],observables=values,
                su2_factorization_frobenius=diagnostic['su2_factorization_frobenius'],
                strict_su2_eligible=diagnostic['strict_su2_eligible'],rho=rho)
        reports.append(dict(step=report['step'],energy=report['energy'],
            structure_factors=report['structure_factors'],total_spin_squared=report['total_spin_squared'],
            regions=regional,audit_seeds=sorted(seeds),strict_su2=False))
    if len(reports)<2:
        raise ValueError('at least two completed, independently audited quality milestones required')
    differences=[]
    for before,after in zip(reports[:-1],reports[1:]):
        regions={}
        for name in spec['regions']:
            a,b=before['regions'][name],after['regions'][name]
            distance=float(.5*np.sum(abs(np.linalg.eigvalsh(b['rho']-a['rho']))))
            regions[name]=dict(observables=_independent_difference(a['observables'],b['observables']),
                raw_RDM_trace_distance=distance,trace_distance_uncertainty_not_estimated=True)
        differences.append(dict(before_step=before['step'],after_step=after['step'],regions=regions))
    for report in reports:
        for region in report['regions'].values(): region.pop('rho')
    if output.exists():
        raise FileExistsError('derived comparison is immutable; choose a new output')
    output.mkdir(parents=True)
    _write(output/'registration.json',dict(inputs={str(p):_sha(p) for p in paths},
        source_workflow_sha256=_sha(Path(__file__)),hamiltonian=spec['j2_over_j1'],
        new_samples=0,new_training=0,new_ED=0,new_full_H_calls=0,new_amplitude_queries=0))
    result=dict(states=reports,adjacent_quality_changes=differences,
        geometry='triangular',rows=6,columns=6,boundary='PBC',j2_over_j1=spec['j2_over_j1'],
        independent_milestone_audits=True,strict_multiplicity_spectrum_reported=False,
        new_samples=0,new_training=0,new_ED=0,new_full_H_calls=0,new_amplitude_queries=0,
        phase_claim=False,seconds=time.perf_counter()-started,
        warnings=['Same candidate family, different optimization quality: not competing phases.',
            'Scalar K/M includes coherent spin-flip ratios, NOT 3SzSz for this nonsinglet.',
            'Whole-chain errors exclude equilibrium and missing-tail bias.',
            'Signed high_q3 is outside the complete two/four-body scalar operator span, not a probability or phase witness.',
            'Connected three-bridge cumulants and the linear high-order projector remainder are different observables.',
            'Local E2/chiral channels are not full-lattice susceptibilities.'])
    _write(output/'results.json',result)
    _write(output/'completion.json',dict(status='complete',outputs={
        p.name:_sha(p) for p in output.iterdir() if p.is_file()}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'projects/clebsch_tree_nqs/results/spin_saved_state_physics_v1')
    parser.add_argument('--pair-source',type=Path,help='Analyze any sealed N36 pair run without new queries')
    parser.add_argument('--vit-source',type=Path,help='Analyze independently audited full-ViT quality milestones without new queries')
    args = parser.parse_args()
    if args.pair_source and args.vit_source:
        parser.error('choose a single saved-state family')
    result = (run_vit_source(args.vit_source,args.output) if args.vit_source else
              run_pair_source(args.pair_source,args.output) if args.pair_source else run(args.output))
    print(json.dumps(dict(seconds=result['seconds'], phase_claim=result['phase_claim'],
        new_samples=result['new_samples'], new_full_H_calls=result['new_full_H_calls'])))
