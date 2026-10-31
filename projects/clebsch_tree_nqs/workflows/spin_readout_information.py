"""Saved-state higher-order information and coherence-aware sampling calibration.

One reusable workflow, no new state, head, training or Hamiltonian eigensolve.
The restricted-sector check intentionally separates full-region operator
nonredundancy from information available on this particular small manifold.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.spin_operator_information import joint_spin_operator_split, _low_order_span
from su2_nqs.benchmarks.adaptive_projected_pair_carrier import triangular_physical_adaptive_pair_carrier
from su2_nqs.benchmarks.xdiag_physical_seed_krylov import SMALL_CASES
from su2_nqs.benchmarks.su2_entanglement_spectrum import reduced_density_matrix
from su2_nqs.benchmarks.readout_quality import regional_readout_distance
from su2_nqs.vmc.regional_density import (regional_query_plan, conditional_regional_matrices,
                                        regional_density_summary, regional_observable_summary)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_readout_information.json'


def restricted_span(basis: np.ndarray, columns: np.ndarray, sites: list[int], splits: dict) -> dict:
    """Compress regional operators to an EXISTING saved S0/A1 isometry, no ED."""
    n = basis.shape[1]; m = len(sites); rest = [i for i in range(n) if i not in sites]
    a = basis[:, sites]@(1 << np.arange(m-1, -1, -1))
    e = basis[:, rest]@(1 << np.arange(len(rest)-1, -1, -1))
    tensors = np.zeros((2**m, 2**len(rest), columns.shape[1]), complex)
    tensors[a, e] = columns
    # T_abij contracts the environment only; no 2^N by 2^N operator.
    t = np.stack([tensors[:, :, i].conj()@tensors[:, :, j].T
                  for i in range(columns.shape[1]) for j in range(columns.shape[1])]).reshape(columns.shape[1]**2, -1)
    _, bank, _ = _low_order_span(m)
    compressed = t@bank
    u, s, _ = np.linalg.svd(compressed, full_matrices=False)
    rank = int(np.count_nonzero(s > 1e-11*s[0])); u = u[:, :rank]
    result = {}
    for name, split in splits.items():
        target = t@split.full.reshape(-1)
        remainder = target-u@(u.conj().T@target)
        result[name] = dict(relative_residual=float(np.linalg.norm(remainder)/np.linalg.norm(target)),
                           absolute_residual=float(np.linalg.norm(remainder)))
    return dict(saved_subspace_dimension=columns.shape[1], restricted_low_order_rank=rank,
        targets=result, warning='Redundancy on this saved finite-size symmetry sector is not redundancy on all regional states or larger systems.')


def run(config: Path = DEFAULT) -> dict:
    spec = json.loads(config.read_text()); out = ROOT/spec['output']
    if out.exists(): raise FileExistsError('immutable readout execution already exists')
    candidate, trusted = [ROOT/spec[k] for k in ('candidate_archive', 'trusted_archive')]
    screen = json.loads((candidate/'results.json').read_text())
    old = json.loads((trusted/'results.json').read_text())
    bundle = ROOT/screen['state_bundle']['path']
    if file_sha256(bundle) != screen['state_bundle']['sha256'] or file_sha256(trusted/'readout.npz') != old['readout_sha256']:
        raise ValueError('saved physical inputs changed')
    sources = transitive_local_source_manifest((Path(__file__),), root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    out.mkdir(parents=True)
    for p in sources:
        dst = out/'source_snapshots'/p; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/p, dst)
    atomic_json(out/'registration.json', dict(config=spec, sources=sources,
        inputs={str(p.relative_to(ROOT)):file_sha256(p) for p in (config, bundle, candidate/'results.json', trusted/'results.json', trusted/'readout.npz')},
        new_ED=0, new_states=0, new_training=0))
    start = perf_counter()
    splits = {str(q):joint_spin_operator_split(6, (0, 2, 4), (q, q, 0)) for q in (1, 3)}
    records = {}; rhos = {}
    with np.load(bundle) as z:
        basis = z['basis']; subspace = z['physical_a1_basis']; psi = z['saved_reference_state']
        for name, source in screen['states'].items():
            rho = z[name+'_strip_6_rho']; rhos[name] = rho
            records[name] = dict(quality=source['quality'], signals={q:s.measure(rho) for q, s in splits.items()})
    controls = ('triangular_j0','triangular_j0.075','triangular_j0.125','triangular_j0.175',
        'dimer_internal','dimer_four_crossing','multiplicity_bell_q0','collective_two_sublattice','collective_three_sublattice')
    with np.load(trusted/'readout.npz') as z:
        for name in controls:
            rho = z[name+'_strip_6_rho']; rhos[name] = rho
            records[name] = dict(quality=old['states'][name].get('quality'), signals={q:s.measure(rho) for q,s in splits.items()})
    def difference(a, b):
        return {q:{k:records[a]['signals'][q][k]-records[b]['signals'][q][k] for k in ('full','low','high')} for q in splits}
    pairs = []
    for a, b in itertools.combinations(screen['primary_quality_flows'], 2):
        separation = difference(a+'_k4', b+'_k4')
        oa, ob = difference(a+'_k4', a+'_k2'), difference(b+'_k4', b+'_k2')
        pairs.append(dict(family_a=a, family_b=b, channels={q:dict(
            separation=separation[q]['high'], refinement_a=abs(oa[q]['high']), refinement_b=abs(ob[q]['high']),
            refinement_sum=abs(oa[q]['high'])+abs(ob[q]['high']),
            exceeds_refinement_sum=abs(separation[q]['high'])>abs(oa[q]['high'])+abs(ob[q]['high'])+1e-12) for q in splits}))
    # Calibrate coherent enumeration on the saved N16 state, exactly summing Q.
    lookup = {tuple(b):i for i,b in enumerate(basis)}
    q = abs(psi)**2+spec['calibration_uniform_floor']/len(psi)
    calibrations = {}
    rng = np.random.default_rng(spec['calibration_seed'])
    draw_ids = rng.choice(len(basis), spec['calibration_chains']*spec['calibration_records'], p=q/q.sum())
    for cut, sites in spec['calibration_regions'].items():
        plan = regional_query_plan(basis, sites)
        ids = np.array([lookup[tuple(b)] for b in plan.configurations])
        a = conditional_regional_matrices(plan, psi[ids], q[ids])
        exact = np.einsum('c,cab->ab', q/q.sum(), a); exact /= np.trace(exact)
        with np.load(bundle) as z: expected = z['saved_reference_'+cut+'_rho']
        error = float(np.max(abs(exact-expected)))
        if error > 2e-13: raise ValueError('saved-state coherent estimator exact replay failed')
        sampled = regional_density_summary(a[draw_ids], np.tile(np.arange(spec['calibration_chains']), spec['calibration_records']))
        # No Markov chain: each calibration group is an independent IID stream.
        qs = range(len(sites)%2, len(sites)+1, 2)
        from su2_nqs.benchmarks.spin_partition_readout import _spin_projectors
        observables = {str(q):regional_observable_summary(sampled, op) for q,op in _spin_projectors(len(sites))}
        for qq, op in _spin_projectors(len(sites)):
            observables[str(qq)]['exact'] = float(np.trace(expected@op).real)
        calibrations[cut] = dict(exhaustive_max_element_error=error, sampled_projector_weights=observables,
            sampled_rdm_frobenius_error=float(np.linalg.norm(sampled['rho']-expected)),
            diagonal_only_frobenius_error=float(np.linalg.norm(np.diag(np.diag(expected))-expected)),
            records=sampled['records'], independent_streams=sampled['independent_chains'])
    restricted = restricted_span(basis, subspace, spec['calibration_regions']['strip_6'], splits)
    vap = {q:dict(separation=abs(difference('triangular_vap_2026092901_terminal','triangular_vap_2026092902_terminal')[q]['high']),
        refinement_sum=sum(abs(difference(f'triangular_vap_{seed}_terminal', f'triangular_vap_{seed}_parent')[q]['high']) for seed in (2026092901,2026092902))) for q in splits}
    result = dict(config=spec, seconds=perf_counter()-start, operators={q:s.diagnostics for q,s in splits.items()},
        states=records, coupling_j0_to_j0125=difference('triangular_j0.125','triangular_j0'),
        primary_quality_flows=screen['primary_quality_flows'], primary_pairs=pairs, VAP_quality=vap,
        errors_to_reference={name:difference(name,'saved_reference') for name in screen['states']},
        restricted_sector_check=restricted, sampling_calibration=calibrations, new_ED=0, new_states=0,
        new_training=0, phase_claim=False, size_evolution_released=False,
        warning='High is a signed six-body observable, not a probability or an entanglement fraction. Quality-flow differences are empirical, not certified state-error bounds.')
    atomic_json(out/'results.json', result)
    np.savez_compressed(out/'operators.npz', **{q+'_'+k:getattr(s,k) for q,s in splits.items() for k in ('full','low','high')})
    print(json.dumps(dict(seconds=result['seconds'], coupling=result['coupling_j0_to_j0125'],
        restricted=restricted, resolved_pairs=sum(any(c['exceeds_refinement_sum'] for c in p['channels'].values()) for p in pairs))), flush=True)
    return result


def reuse_optimized_carriers(config: Path, optimized_source: Path) -> dict:
    """Replay OLD optimized pair parameters, never repeat their optimization.

    Use the saved complete14-dimensional S0/A1 operator and a3-dimensional
    Ritz reconstruction of the ALREADY-defined K2 state. This is not a new
    many-body ED, deeper filtering bank, or fresh variational run.
    """
    spec=json.loads(config.read_text());out=ROOT/spec['output']/'optimized_carrier_reuse'
    if out.exists():raise FileExistsError('optimized-state reuse output is immutable')
    archive=ROOT/optimized_source if not optimized_source.is_absolute() else optimized_source
    old=json.loads(archive.read_text());body=dict(old);digest=body.pop('result_sha256')
    if canonical_sha256(body)!=digest or old['status']!='complete':raise ValueError('old optimizer seal fails')
    saved=next(row for row in old['rows'] if row['j2_over_j1']==.125)['physical_adaptive_carrier']
    screen=json.loads((ROOT/spec['candidate_archive']/'results.json').read_text());bundle=ROOT/screen['state_bundle']['path']
    if file_sha256(bundle)!=screen['state_bundle']['sha256']:raise ValueError('cached physical basis changed')
    sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    out.mkdir(parents=True)
    for p in sources:
        dst=out/'source_snapshots'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dst)
    atomic_json(out/'registration.json',dict(sources=sources,
        inputs={str(p.relative_to(ROOT)):file_sha256(p) for p in (config,archive,bundle)},
        protocol='Replay fixed existing j=.125 optimized pair coordinates, bare and same-degree K2; no new parameter selection, optimization, ED or sampling.',
        new_training=0,new_many_body_ED=0,new_optimizer_evaluations=0,old_optimizer_repeated=False))
    start=perf_counter()
    with np.load(bundle) as z:
        bits,u,h=z['basis'],z['physical_a1_basis'],z['h_a1']
        vectors={name:z[name+'_state'] for name in ('pi_flux_k0','pi_flux_k2','saved_reference')}
    # 'square4' denotes the HNF shape; this public family is TRIANGULAR J1/J2.
    family=triangular_physical_adaptive_pair_carrier(bits,SMALL_CASES['square4'])
    zero=family.amplitudes(np.zeros(3));zero/=np.linalg.norm(zero)
    if 1-abs(np.vdot(zero,vectors['pi_flux_k0']))**2>1e-12:raise ValueError('fixed carrier definition is not shared')
    max_energy_error=0.;max_variance_error=0.;max_sector_error=0.
    replay={}
    for degree in (0,2):
        raw=family.amplitudes(saved[str(degree)]['parameters']);raw/=np.linalg.norm(raw)
        c=u.conj().T@raw;defect=float(np.linalg.norm(raw-u@c));max_sector_error=max(max_sector_error,defect)
        if defect>1e-10:raise ValueError('AGP state is outside saved complete S0/A1 sector; do not repair')
        if degree:
            columns=np.column_stack((c,h@c,h@(h@c)));v,s,_=np.linalg.svd(columns,full_matrices=False)
            v=v[:,s>1e-12*s[0]];_,w=np.linalg.eigh(v.conj().T@h@v);c=v@w[:,0]
        c/=np.linalg.norm(c);vector=u@c
        e=float(np.vdot(c,h@c).real)/16;var=float(np.linalg.norm(h@c-16*e*c)**2)/16
        de=abs(e-saved[str(degree)]['energy_per_site']);dv=abs(var-saved[str(degree)]['variance_per_site'])
        max_energy_error=max(max_energy_error,de);max_variance_error=max(max_variance_error,dv)
        if max(de,dv)>2e-10:raise ValueError('old optimized energy/variance cannot be replayed')
        name=f'optimized_pair_k{degree}';vectors[name]=vector
        replay[name]=dict(energy_per_site=e,variance_per_site=var,energy_replay_error=de,
            variance_replay_error=dv,parameters=saved[str(degree)]['parameters'],sector_residual=defect)
    splits={str(q):joint_spin_operator_split(6,(0,2,4),(q,q,0)) for q in (1,3)}
    records={};rhos={}
    for name,vector in vectors.items():
        rho=reduced_density_matrix(vector,bits,(0,1,4,5,8,9));rhos[name]=rho
        c=u.conj().T@vector;e=float(np.vdot(c,h@c).real)/16
        records[name]=dict(energy_per_site=e,variance_per_site=float(np.linalg.norm(h@c-16*e*c)**2)/16,
            signals={q:s.measure(rho) for q,s in splits.items()})
    def difference(a,b):
        return dict(physical_trace_distance=regional_readout_distance(rhos[a],rhos[b])['physical_trace_distance'],
            high={q:records[a]['signals'][q]['high']-records[b]['signals'][q]['high'] for q in splits})
    result=dict(states=records,old_parameter_replay=replay,
        final_family_separation=difference('optimized_pair_k2','pi_flux_k2'),
        fixed_own_quality_change=difference('pi_flux_k2','pi_flux_k0'),
        optimized_own_quality_change=difference('optimized_pair_k2','optimized_pair_k0'),
        errors_to_reference={n:difference(n,'saved_reference') for n in vectors},
        maximum_energy_replay_error=max_energy_error,maximum_variance_replay_error=max_variance_error,
        maximum_raw_sector_residual=max_sector_error,seconds=perf_counter()-start,
        new_training=0,new_optimizer_evaluations=0,new_many_body_ED=0,new_samples=0,
        new_saved_state_reconstructions=2,new_variational_Ritz_replays=1,old_optimizer_repeated=False,
        distinct_QSL_claim=False,size_evolution_released=False,
        warning='An optimized pi-flux-derived AGP family is not automatically a distinct paired Z2 liquid. Replaying old parameter states improves the quality control, not the evidence for two phases.')
    atomic_json(out/'results.json',result)
    np.savez_compressed(out/'states.npz',basis=bits,**vectors,**{n+'_rho':rho for n,rho in rhos.items()})
    print(json.dumps(dict(seconds=result['seconds'],states={k:{a:v[a] for a in ('energy_per_site','variance_per_site')} for k,v in records.items()},
        replay=[max_energy_error,max_variance_error,max_sector_error])),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--reuse-optimized-carriers',type=Path,help='Existing sealed optimizer results; replay only, no retraining.')
    args=parser.parse_args()
    if args.reuse_optimized_carriers:reuse_optimized_carriers(args.config,args.reuse_optimized_carriers)
    else:run(args.config)
