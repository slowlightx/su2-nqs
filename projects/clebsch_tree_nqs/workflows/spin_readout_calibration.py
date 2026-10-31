"""Registered saved-state/read-only-iPEPS calibration without a DMRG dependency.

Reuse the common iPEPS adapter, physical RDM readouts and refinement metrics.
One configuration manages all points; no tensor optimization or new ED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.readout_quality import (
    regional_readout_distance, compare_with_refinement,
    physical_problem_signature, physical_state_similarity,
)
from su2_nqs.benchmarks.tensor_network_readout import physical_spin_block_readout
from su2_nqs.benchmarks.tensor_network_observables import regional_physical_observables
from su2_nqs.benchmarks.su2_entanglement_spectrum import reduced_density_matrix

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_readout_calibration.json'
SCREEN_CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_candidate_state_screen.json'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def point_directory(spec: dict, point: dict) -> Path:
    """Return a reused immutable artifact or this campaign's coupling output."""
    return ROOT/point['reuse'] if 'reuse' in point else ROOT/spec['output']/f'ipeps_j{point["j2"]:g}'


def register(config: Path = CONFIG) -> dict:
    """Freeze couplings, source tensors, budgets and code BEFORE contraction."""
    spec = json.loads(config.read_text())
    if spec['task']!='spin_readout_calibration' or any(spec[k] for k in
            ('new_ed','new_training','new_dmrg','new_sampling','new_gpu_jobs')):
        raise ValueError('only read-only calibration is permitted')
    if spec['chis']!=[24,32] or spec['seconds_per_new_tensor_cap']!=600:
        raise ValueError('declared bounded contraction protocol changed')
    output = ROOT/spec['output']
    output.mkdir(parents=True, exist_ok=True)
    inputs = {str(config.resolve()): _sha(config)}
    for point in spec['points']:
        state = Path(spec['source_directory'])/point['state']
        inputs[str(state)] = _sha(state)
        log = Path(str(state).replace('_state.json', '.out'))
        inputs[str(log)] = _sha(log)
    previous = ROOT/spec['saved_n16']
    for name in ('results.json','readout.npz'):
        inputs[str(previous/name)] = _sha(previous/name)
    sources = [Path(__file__), ROOT/'projects/clebsch_tree_nqs/workflows/spin_ipeps_reference.py']
    sources.extend(ROOT/'su2_nqs/benchmarks'/f'{name}.py' for name in
                   ('readout_quality','tensor_network_readout','tensor_network_observables',
                    'regional_su2_readout','su2_entanglement_spectrum'))
    registration = dict(config=spec, inputs=inputs, sources={str(p.relative_to(ROOT)): _sha(p) for p in sources},
                        new_tensors=sum('reuse' not in p for p in spec['points']),
                        maximum_new_contraction_seconds=600*sum('reuse' not in p for p in spec['points']),
                        phase_claim=False, certified_finite_D_error=False,
                        external_DMRG_required=False)
    target = output/'registration.json'
    if target.exists():
        if json.loads(target.read_text())!=registration:
            raise ValueError('registered protocol/source changed; do not reseal old results')
    else:
        _json(target, registration)
        for source in sources:
            archive = output/'source_snapshots'/source.relative_to(ROOT)
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, archive)
    return registration


def contract_point(config: Path, index: int) -> dict:
    """Contract exactly one registered point using the shared iPEPS adapter."""
    registration = register(config)
    spec = registration['config']
    if not 0 <= index < len(spec['points']):
        raise ValueError('point index is outside the registered coupling list')
    point = spec['points'][index]
    target = point_directory(spec, point)
    if 'reuse' in point:
        return json.loads((target/'results.json').read_text())
    from projects.clebsch_tree_nqs.workflows.spin_ipeps_reference import run
    return run(Path(spec['source_directory'])/point['state'], Path(spec['checkout']),
               target, tuple(spec['chis']), j2=point['j2'])


def analyze(config: Path = CONFIG) -> dict:
    """Compare frozen endpoints and empirical refinement; no new contraction."""
    registration = register(config)
    spec = registration['config']; output = ROOT/spec['output']
    if (output/'results.json').exists():
        raise FileExistsError('completed calibration is immutable')
    start = perf_counter(); records = []; densities = []; inputs = {}
    for point in spec['points']:
        directory = point_directory(spec, point)
        info = json.loads((directory/'results.json').read_text())
        if (info['registration']['j2']!=point['j2'] or
                info['registration']['state_sha256']!=registration['inputs'][str(Path(spec['source_directory'])/point['state'])] or
                info['rdms_sha256']!=_sha(directory/'rdms.npz')):
            raise ValueError('coupling/tensor/RDM seal mismatch')
        for name in ('results.json','rdms.npz'):
            inputs[str(directory/name)] = _sha(directory/name)
        with np.load(directory/'rdms.npz',allow_pickle=False) as arrays:
            coarse, fine = [arrays[f'chi{chi}_physical_compact4'] for chi in spec['chis']]
        stages = [next(s for s in info['records'] if s['chi']==chi) for chi in spec['chis']]
        records.append(dict(j2=point['j2'], artifact=str(directory.relative_to(ROOT)),
                            stages=stages, refinement=regional_readout_distance(fine,coarse),
                            compact4_observables=regional_physical_observables(fine),
                            strict_su2_eligible=physical_spin_block_readout(fine)['strict_su2_eligible']))
        densities.append((coarse,fine))
    adjacent = []
    for i in range(len(records)-1):
        comparison = compare_with_refinement(densities[i][1],densities[i+1][1],
                          refinements_a=[densities[i][0]],refinements_b=[densities[i+1][0]])
        adjacent.append(dict(j2_a=records[i]['j2'],j2_b=records[i+1]['j2'],
                             same_hamiltonian=False, calibration_only=True, **comparison))
    previous = ROOT/spec['saved_n16']
    saved = json.loads((previous/'results.json').read_text())
    states = saved['states']; cuts = saved['config']['regions']; controls = {}
    quality = {}; seed_comparison = {}
    with np.load(previous/'readout.npz',allow_pickle=False) as arrays:
        def density(name: str, region: str) -> np.ndarray:
            key = name+'_'+region+'_rho'
            return arrays[key] if key in arrays else reduced_density_matrix(
                arrays[name+'_state'],arrays['basis'],cuts[region])
        for a,b in [('collective_three_sublattice','dimer_internal'),
                    ('dimer_internal','dimer_four_crossing'),('triangular_j0','triangular_j0.125')]:
            controls[a+'__'+b] = {region:regional_readout_distance(density(a,region),density(b,region))
                                  for region in cuts}
        names = ['triangular_j0.125_carrier','triangular_j0.125_two_lanczos',
                 'triangular_optimized_pi_flux_jastrow_two_lanczos',
                 'triangular_pi_flux_fixed_sz_gamma_a1_spatial_vit',
                 'triangular_vap_2026092901_terminal','triangular_vap_2026092902_terminal']
        for name in names:
            quality[name] = dict(quality=states[name]['quality'], provenance=states[name]['provenance'],
                regions={region:regional_readout_distance(density(name,region),density('triangular_j0.125',region))
                         for region in cuts})
        for region in cuts:
            a,b = 'triangular_vap_2026092901','triangular_vap_2026092902'
            seed_comparison[region] = compare_with_refinement(density(a+'_terminal',region),density(b+'_terminal',region),
                refinements_a=[density(a+'_parent',region)],refinements_b=[density(b+'_parent',region)])
    result = dict(task=spec['task'],status='complete',registration=registration,
                  contracted_inputs=inputs,ipeps_records=records,adjacent_coupling_calibration=adjacent,
                  n16=dict(controls=controls,quality_to_saved_same_sector_reference=quality,
                           two_VAP_seeds=seed_comparison,
                           warning='Analytic collective/dimer states are distinguishable controls, not competing ground states. N16 triangular torus misses K. Accurate VAP seeds converge to the same state.'),
                  seconds=perf_counter()-start,new_ED=0,new_training=0,new_DMRG=0,new_samples=0,new_GPU_jobs=0,
                  phase_claim=False,size_evolution_released=False,
                  disposition='D5 ordered-family contraction calibration and N16 controls only. No strict spectrum for broken SU2; no same-H competing pair or finite-D convergence certificate. PRL-specific external LE/HE data deferred.')
    _json(output/'results.json',result)
    return result


def screen_candidates(config: Path = SCREEN_CONFIG) -> dict:
    """Screen saved physical states and fixed carriers on one labelled torus.

    A depth-four polynomial filter is a bounded quality diagnostic, NOT a
    new neural optimization or a phase classifier. Reuse the saved complete
    S=0/Gamma-A1 columns and Hamiltonian; no many-body eigensolve is performed.
    """
    from dataclasses import asdict
    from itertools import combinations
    from su2_nqs.benchmarks.adaptive_projected_pair_carrier import batched_singlet_agp_amplitudes
    from su2_nqs.benchmarks.observable_flow import normalized_state, full_reorthogonalized_ritz_flow
    from su2_nqs.benchmarks.paired_z2_carrier import paired_z2_kernel_bank
    from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
    from su2_nqs.frontier.provenance import transitive_local_source_manifest
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from projects.clebsch_tree_nqs.workflows.spin_competing_state_readout import align_basis_vector, _observables

    spec = json.loads(config.read_text())
    if (spec['task']!='spin_candidate_state_screen' or
            any(spec[k] for k in ('new_ed','new_neural_training','new_dmrg','new_sampling','new_gpu_jobs')) or
            (spec['rows'],spec['columns'],spec['j1'],spec['j2'])!=(4,4,1.,.125) or
            spec['maximum_krylov_depth']!=4 or spec['report_depths']!=[0,1,2,4]):
        raise ValueError('only the declared fixed-H saved-state screen is permitted')
    output = ROOT/spec['output']; output.mkdir(parents=True,exist_ok=True)
    if (output/'results.json').exists():
        raise FileExistsError('completed candidate screen is immutable')
    source_dirs = {key:ROOT/path for key,path in spec['sources'].items()}
    inputs = {str(config.relative_to(ROOT)):_sha(config)}
    for key,directory in source_dirs.items():
        for name in ('results.json','readout.npz' if key=='small_readout' else 'states.npz'):
            inputs[str((directory/name).relative_to(ROOT))] = _sha(directory/name)
    inputs.update({path:_sha(ROOT/path) for path in spec['metadata_only_excluded']})
    sources = transitive_local_source_manifest(
        [Path(__file__),ROOT/'projects/clebsch_tree_nqs/workflows/spin_competing_state_readout.py'],root=ROOT)
    registration = dict(config=spec,inputs=inputs,sources=sources,
        external_DMRG_required=False,maximum_Ritz_dimension=5,
        warning='Mean-field Z2/Dirac names are carrier labels, not identified projected phases. Fixed N16 misses K; no size/phase release.')
    target = output/'registration.json'
    if target.exists():
        if json.loads(target.read_text())!=registration:
            raise ValueError('registered protocol/source changed; do not reseal')
    else:
        _json(target,registration)
        for path in sources:
            archive = output/'source_snapshots'/path
            archive.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/path,archive)

    def timeout(*_):
        raise TimeoutError('bounded saved-state screening exceeded its CPU wall cap')
    old_handler = signal.signal(signal.SIGALRM,timeout)
    signal.alarm(spec['wall_seconds_cap']); start = perf_counter()
    try:
        arrays = {key:dict(np.load(directory/('readout.npz' if key=='small_readout' else 'states.npz'),
                                   allow_pickle=False)) for key,directory in source_dirs.items()}
        for key in ('qsl','a1','loop','backflow','resonating','manifold'):
            metadata = json.loads((source_dirs[key]/'results.json').read_text())['config']
            if (metadata['geometry']!=spec['geometry'] or
                    (spec['j2'] not in metadata['couplings'] if key=='a1' else metadata['j2_over_j1']!=spec['j2'])):
                raise ValueError('saved artifact is not on the common physical Hamiltonian')
        basis = arrays['qsl']['basis']; phi = arrays['a1']['physical_a1_basis']
        h = arrays['a1']['coupling_0.125_h_a1']
        if basis.shape!=(12870,16) or phi.shape!=(12870,14) or h.shape!=(14,14):
            raise ValueError('unexpected saved physical support/projector')
        bonds = triangular_j1j2_bonds(4,4,j1=spec['j1'],j2=spec['j2'],periodic=True)
        signature = physical_problem_signature(16,bonds,geometry=spec['geometry'],boundary=spec['physical_boundary'])
        actions = FixedSzPairActions(basis); h_calls = 0
        def full_h(vector):
            nonlocal h_calls
            h_calls += 1
            return actions.weighted_bonds(vector,bonds)
        h_phi = np.column_stack([full_h(column) for column in phi.T])
        checks = dict(physical_a1_orthonormality=float(np.linalg.norm(phi.conj().T@phi-np.eye(14))),
                      full_H_intertwining=float(np.linalg.norm(h_phi-phi@h)),
                      saved_H_hermiticity=float(np.linalg.norm(h-h.conj().T)))
        if max(checks.values())>1e-10:
            raise ValueError('saved complete projector/H failed independent physical action')
        reference = normalized_state(arrays['qsl']['exact_ground'])
        reference_energy = float(np.vdot(reference,full_h(reference)).real)
        states = {}; provenance = {}; flows = {}; reconstructions = {}
        def add(name,vector,*,family,role,source,projection_weight=None):
            if name in states:
                raise ValueError('duplicate state identifier')
            states[name] = normalized_state(vector)
            provenance[name] = dict(family=family,role=role,source=source,
                problem_signature=signature,projection_weight=projection_weight)
        add('saved_reference',reference,family='saved_reference',role='quality_reference_not_new_ED',source='qsl:exact_ground')
        initial = {'pi_flux':normalized_state(arrays['a1']['coupling_0.125_carrier_a1'])}
        for kernel in paired_z2_kernel_bank('triangular_n16'):
            raw = normalized_state(batched_singlet_agp_amplitudes(basis,kernel.pairing))
            coefficient = phi.conj().T@raw
            initial[kernel.label] = normalized_state(coefficient)
            raw_obs,_ = _observables(raw,actions)
            diagnostics = asdict(kernel); diagnostics.pop('pairing')
            diagnostics.update(raw_total_spin_squared=raw_obs['total_spin_squared'],
                explicit_A1_projection_weight=float(np.linalg.norm(coefficient)**2),
                meaning='Fixed BdG kernel; mean-field IGG/gap does not establish projected Z2 order.')
            if abs(raw_obs['total_spin_squared'])>1e-9:
                raise ValueError('fixed paired determinant violates singlet sign conventions')
            reconstructions[kernel.label] = diagnostics
        for family,source in [('collective_order_A1','collective_three_sublattice_state'),('dimer_A1','dimer_internal_state')]:
            raw = align_basis_vector(arrays['small_readout'][source],arrays['small_readout']['basis'],basis)
            coefficient = phi.conj().T@normalized_state(raw)
            initial[family] = normalized_state(coefficient)
            reconstructions[family] = dict(explicit_A1_projection_weight=float(np.linalg.norm(coefficient)**2),
                meaning='New explicitly projected finite control, not an ordered ground-state candidate.')
        if set(initial)!=set(spec['primary_families']):
            raise ValueError('primary candidate families changed after registration')
        reduced_h_calls = 0
        class CountedReducedH:
            shape = h.shape
            def __matmul__(self,vector):
                nonlocal reduced_h_calls
                reduced_h_calls += 1 if vector.ndim==1 else vector.shape[1]
                return h@vector
        for family in spec['primary_families']:
            flows[family] = []
            for point in full_reorthogonalized_ritz_flow(CountedReducedH(),initial[family],4):
                if point.depth not in spec['report_depths']:
                    continue
                name = f'{family}_k{point.depth}'
                add(name,phi@point.state,family=family,
                    role='analytic_control' if family.endswith('_A1') else 'fixed_carrier_quality_flow',
                    source='saved A1 pi-flux' if family=='pi_flux' else family,
                    projection_weight=reconstructions.get(family,{}).get('explicit_A1_projection_weight'))
                flows[family].append(name)
            if len(flows[family])!=4:
                raise ValueError('unexpected early Krylov breakdown; do not invent missing refinements')
        for name in ('optimized_pi_flux_jastrow_two_lanczos','pi_flux_fixed_sz_gamma_a1_spatial_vit',
                     'pi_flux_exact_su2_a1_operator_vit'):
            add(name,arrays['qsl'][name],family=name,role='saved_neural_or_VMC_audit',source='qsl:'+name)
        for seed in (2026092901,2026092902):
            for stage in ('parent','terminal'):
                name = f'triangular_vap_{seed}_{stage}'
                vector = align_basis_vector(arrays['small_readout'][name+'_state'],arrays['small_readout']['basis'],basis)
                add(name,vector,family='VAP',role='saved_same_ansatz_precision_control',source='small_readout:'+name)
        for key in arrays['loop']:
            if '_physical_seed' in key and key.endswith('_a1'):
                add(key,phi@arrays['loop'][key],family=key.split('_physical_seed')[0],
                    role='saved_loop_seed_audit_not_new_phase',source='loop:'+key)
        for source,keys in [('backflow',('compact_j1_best_physical_a1','extended_j1j2_best_physical_a1')),
                            ('resonating',('combined_physical_a1',)),
                            ('manifold',('standalone_best','operator_anchored_best'))]:
            for key in keys:
                vector = arrays[source][key]
                add(source+'_'+key,phi@vector if vector.shape==(14,) else vector,
                    family=source,role='saved_neural_audit_no_refinement_certificate',source=source+':'+key)
        for source in ('loop','backflow','resonating'):
            if physical_state_similarity(phi@arrays[source]['exact_a1'],reference)['fidelity']<1-1e-10:
                raise ValueError('historical coefficient basis is not the common physical basis')
        records = {}; densities = {}; payload = {'basis':basis,'physical_a1_basis':phi,'h_a1':h}
        for name,vector in states.items():
            image = full_h(vector); energy = float(np.vdot(vector,image).real)
            variance = float(np.linalg.norm(image-energy*vector)**2)
            observables,corr = _observables(vector,actions)
            quality = dict(energy_per_site=energy/16,variance_per_site=variance/16,
                energy_error_per_site=(energy-reference_energy)/16,
                **physical_state_similarity(vector,reference),
                full_S0_A1_projection_residual=float(np.linalg.norm(vector-phi@(phi.conj().T@vector))))
            regions = {}; densities[name] = {}; payload[name+'_state'] = vector
            for region,sites in spec['regions'].items():
                rho = reduced_density_matrix(vector,basis,sites)
                readout = physical_spin_block_readout(rho)
                readout['spin_squared_pair_closure_error'] = abs(readout['spin_squared_mean']-corr[np.ix_(sites,sites)].sum())
                regions[region] = readout; densities[name][region] = rho
                payload[name+'_'+region+'_rho'] = rho
            records[name] = dict(provenance=provenance[name],quality=quality,observables=observables,regions=regions)
        comparisons = []
        for a,b in combinations(spec['primary_families'],2):
            stages = {str(depth):dict(
                similarity=physical_state_similarity(states[f'{a}_k{depth}'],states[f'{b}_k{depth}']),
                regions={r:regional_readout_distance(densities[f'{a}_k{depth}'][r],densities[f'{b}_k{depth}'][r])
                         for r in spec['regions']}) for depth in spec['report_depths']}
            refinement = {r:compare_with_refinement(densities[f'{a}_k4'][r],densities[f'{b}_k4'][r],
                refinements_a=[densities[f'{a}_k2'][r]],refinements_b=[densities[f'{b}_k2'][r]]) for r in spec['regions']}
            comparisons.append(dict(family_a=a,family_b=b,same_H_and_boundary=True,
                includes_analytic_control=a.endswith('_A1') or b.endswith('_A1'),
                stages=stages,endpoint_against_each_own_refinement=refinement,
                warning='Passing an empirical margin is not a ground-phase identification or finite-bias certificate.'))
        seeds = {r:compare_with_refinement(densities['triangular_vap_2026092901_terminal'][r],
                densities['triangular_vap_2026092902_terminal'][r],
                refinements_a=[densities['triangular_vap_2026092901_parent'][r]],
                refinements_b=[densities['triangular_vap_2026092902_parent'][r]]) for r in spec['regions']}
        np.savez_compressed(output/'states_readouts.npz',**payload)
        checks['maximum_RDM_trace_error'] = max(r['trace_error'] for s in records.values() for r in s['regions'].values())
        checks['maximum_spin_readout_closure_error'] = max(r['spin_squared_pair_closure_error'] for s in records.values() for r in s['regions'].values())
        result = dict(task=spec['task'],status='complete',registration=registration,
            physical_problem=dict(signature=signature,geometry=spec['geometry'],j1=1.,j2=.125,
                boundary=spec['physical_boundary'],bonds=bonds,reference_energy_per_site=reference_energy/16,
                warning='N16 torus misses K. Region8 wraps a direction. No bulk-QSL identification or system-size scaling.'),
            states=records,primary_quality_flows=flows,reconstructions=reconstructions,
            primary_pair_comparisons=comparisons,VAP_seed_precision_comparison=seeds,validation=checks,
            state_bundle=dict(path=str((output/'states_readouts.npz').relative_to(ROOT)),sha256=_sha(output/'states_readouts.npz')),
            excluded=dict(D5_iPEPS='Infinite ordered branch, different physical boundary/size and uncontrolled finite-D bias.',
                paired_development='Different triangular N12/square geometry; no saved trained final wavefunction.',
                N36='Sampling summaries, not a saved compatible finite-region density; no new sampling in this task.'),
            resources=dict(seconds=perf_counter()-start,physical_full_H_vector_calls=h_calls,
                cached_reduced_H_vector_equivalents=reduced_h_calls,maximum_Ritz_dimension=5,
                single_particle_BdG_reconstructions=3,new_ED=0,new_neural_training=0,new_DMRG=0,new_samples=0,new_GPU_jobs=0),
            phase_claim=False,size_evolution_released=False,
            disposition='Candidate-state/quality-dependence calibration only. Actual pair eligibility must retain both endpoint quality changes and avoid interpreting analytic controls or seed differences as phases.')
        _json(output/'results.json',result)
        return result
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,old_handler)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register',action='store_true')
    mode.add_argument('--point',type=int)
    mode.add_argument('--analyze',action='store_true')
    mode.add_argument('--screen-candidates',action='store_true')
    parser.add_argument('--screen-config',type=Path,default=SCREEN_CONFIG)
    args = parser.parse_args()
    if args.screen_candidates:
        result = screen_candidates(args.screen_config)
        print('Candidate screen complete:',len(result['states']),'states,',result['resources'])
    elif args.register:
        print('Registered',register(args.config)['maximum_new_contraction_seconds'],'CPU seconds maximum')
    elif args.analyze:
        print('Calibration complete',analyze(args.config)['seconds'])
    else:
        contract_point(args.config,args.point)
