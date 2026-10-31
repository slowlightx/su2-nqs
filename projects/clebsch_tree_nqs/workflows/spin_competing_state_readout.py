"""Nested-region physics readout of saved states; no training or new ED.

Separates regional spin weights from multiplicity spectra and checks their
ansatz/refinement sensitivity. A 4x4 triangular torus lacks K: none of these
states is identified as the LE/HE phase of the recent cylinder PRL.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from math import comb
from pathlib import Path
import signal
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
from su2_nqs.benchmarks.su2_entanglement_spectrum import reduced_density_matrix
from su2_nqs.benchmarks.regional_su2_readout import (
    su2_multiplicity_readout, su2_cyclic_entanglement_readout,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, array_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from projects.clebsch_tree_nqs.workflows import spin_symmetry_readout as previous

ROOT = previous.ROOT
CONFIG = ROOT / 'projects/clebsch_tree_nqs/configs/validation/spin_competing_state_readout.json'


def align_basis_vector(vector: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Reindex the SAME physical amplitudes; reject different/duplicate supports."""
    source,target=np.asarray(source),np.asarray(target)
    vector=np.asarray(vector)
    if source.shape!=target.shape or source.ndim!=2 or vector.shape!=(len(source),):
        raise ValueError('basis/vector shapes disagree')
    if np.any((source!=0)&(source!=1)) or np.any((target!=0)&(target!=1)):
        raise ValueError('basis must contain bits')
    lookup={tuple(row):i for i,row in enumerate(source)}
    if len(lookup)!=len(source) or set(lookup)!=set(map(tuple,target)) or len(set(map(tuple,target)))!=len(target):
        raise ValueError('basis supports must be identical and unique')
    return vector[[lookup[tuple(row)] for row in target]]


def collective_sublattice_singlet(basis: np.ndarray, blocks: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """Analytic maximal-sublattice-spin singlet with two or three blocks.

    Doubled sublattice spins are their site counts. This is a collective-order
    control, NOT a lattice ground state or a QSL classifier. Three unequal
    blocks need not describe precisely 120-degree order or preserve A1.
    """
    basis = np.asarray(basis, dtype=np.int8)
    if basis.ndim != 2 or len(blocks) not in (2, 3) or np.any((basis != 0) & (basis != 1)):
        raise ValueError('requires binary basis and two or three blocks')
    if any(not b for b in blocks) or sorted(i for b in blocks for i in b) != list(range(basis.shape[1])):
        raise ValueError('blocks must partition all physical sites')
    if np.any(2 * basis.sum(axis=1) != basis.shape[1]):
        raise ValueError('analytic singlet requires fixed M=0 basis')
    q = tuple(len(b) for b in blocks)
    counts = np.column_stack([basis[:,b].sum(axis=1) for b in blocks])
    @lru_cache(None)
    def amplitude(ups: tuple[int, ...]) -> float:
        m = tuple(2*u-n for u,n in zip(ups,q))
        if len(q)==2:
            c=clebsch_gordan(q[0],m[0],q[1],m[1],0,0)
        else:
            c=clebsch_gordan(q[1],m[1],q[2],m[2],q[0],-m[0])*clebsch_gordan(q[0],m[0],q[0],-m[0],0,0)
        return c/np.sqrt(np.prod([comb(n,u) for n,u in zip(q,ups)],dtype=float))
    vector=np.array([amplitude(tuple(int(x) for x in row)) for row in counts],complex)
    norm=np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError('sublattice spins cannot couple to a singlet')
    return vector/norm


def _distance(a: np.ndarray, b: np.ndarray, ra: dict, rb: dict) -> dict:
    """Compare same-region physical RDMs and gauge-invariant sector spectra."""
    trace=.5*float(np.abs(np.linalg.eigvalsh(a-b)).sum())
    p_tv=.5*sum(abs(ra['sectors'][q]['p']-rb['sectors'][q]['p']) for q in ra['sectors'])
    spectral=.5*sum(float(np.abs(
        np.asarray(ra['sectors'][q]['unnormalized_multiplicity_spectrum'])-
        np.asarray(rb['sectors'][q]['unnormalized_multiplicity_spectrum'])
    ).sum()) for q in ra['sectors'])
    return dict(physical_trace_distance=trace,p_q_total_variation=p_tv,
                multiplicity_spectrum_total_variation=spectral,
                multiplicity_entropy_difference=ra['multiplicity_entropy']-rb['multiplicity_entropy'])


def _load(spec: dict) -> tuple[np.ndarray, dict, dict, dict]:
    old=json.loads((ROOT/spec['previous_readout_config']).read_text())
    hashes={**old['source_files'],**spec['additional_source_files']}
    for path,sha in hashes.items():
        if file_sha256(ROOT/path)!=sha:
            raise ValueError('source seal mismatch: '+path)
    basis,states,metadata=previous.load_states(old)
    base=ROOT/'projects/clebsch_tree_nqs/results'
    with np.load(base/'spin_carrier_cell_scalar_circuit_v1/states.npz',allow_pickle=False) as z:
        phi=z['physical_a1_basis']
        for j in (0.,.125):
            for kind,suffix in [('carrier','carrier_a1'),('two_lanczos','two_lanczos_a1')]:
                name=f'triangular_j{j:g}_{kind}'
                states[name]=phi@z[f'coupling_{j:g}_{suffix}']
                metadata[name]=dict(geometry='triangular_4x4_PBC',coupling=j,kind='saved '+kind,
                    source='spin_carrier_cell_scalar_circuit_v1/states.npz',key=f'coupling_{j:g}_{suffix}')
    for geometry in ('triangular','square'):
        for seed in (2026092901,2026092902):
            path=base/f'spin_full_vit_vap_v1/{geometry}_d16_l2_s{seed}_vap'
            source=previous._sealed_result(path/'results.json')
            with np.load(path/'states.npz',allow_pickle=False) as z:
                if not np.array_equal(z['basis'],basis):
                    raise ValueError('VAP basis order mismatch')
                for stage in ('parent','terminal'):
                    name=f'{geometry}_vap_{seed}_{stage}'
                    states[name]=z['projected_'+stage]
                    metadata[name]=dict(geometry=geometry+'_4x4_PBC',coupling=.125 if geometry=='triangular' else .5,
                        kind='saved full ViT VAP '+stage,source=str(path.relative_to(ROOT)/'states.npz'),
                        key='projected_'+stage,expected_energy_per_site=source['frozen_projection' if stage=='parent' else 'final']['energy_per_site'])
    with np.load(base/'spin_qsl_symmetry_diagnostics_v1/states.npz',allow_pickle=False) as z:
        ref=states['triangular_j0.125']
        exact=align_basis_vector(z['exact_ground'],z['basis'],basis)
        if abs(np.vdot(ref,exact))**2 < 1-1e-10:
            raise ValueError('two saved exact references disagree')
        for key in ('bare_pi_flux','optimized_pi_flux_jastrow_two_lanczos',
                    'pi_flux_fixed_sz_gamma_a1_spatial_vit','pi_flux_exact_su2_a1_operator_vit'):
            name='triangular_'+key
            states[name]=align_basis_vector(z[key],z['basis'],basis)
            metadata[name]=dict(geometry='triangular_4x4_PBC',coupling=.125,kind='saved '+key,
                                source='spin_qsl_symmetry_diagnostics_v1/states.npz',key=key,
                                source_basis_sha256=array_sha256(z['basis']),
                                basis_reindexed_not_reprojected=not np.array_equal(z['basis'],basis))
    # Duplicate square terminals were already in the old calibration; retain
    # old files but do not duplicate the same state as independent evidence.
    for name in list(states):
        if name.startswith('square_j0.5_vap_seed'):
            del states[name]; del metadata[name]
    coords=[divmod(i,4) for i in range(16)]
    for name,labels in [('collective_two_sublattice',[(r+c)%2 for r,c in coords]),
                        ('collective_three_sublattice',[(r+c)%3 for r,c in coords])]:
        blocks=tuple(tuple(i for i,l in enumerate(labels) if l==a) for a in sorted(set(labels)))
        states[name]=collective_sublattice_singlet(basis,blocks)
        metadata[name]=dict(kind='analytic maximal-sublattice-spin singlet control',blocks=blocks,
                           warning='not a lattice ground state, not necessarily A1; not a thermodynamic order diagnosis')
    return basis,states,metadata,hashes


def _observables(vector: np.ndarray, actions: FixedSzPairActions) -> tuple[dict,np.ndarray]:
    n=actions.n_sites
    corr=np.eye(n)*.75
    for i in range(n):
        for j in range(i+1,n):
            corr[i,j]=corr[j,i]=float(np.vdot(vector,actions.pair(vector,i,j)).real)
    bonds=tuple((i,(i//4)*4+(i+1)%4) for i in range(n))
    images=np.column_stack([actions.pair(vector,*b) for b in bonds])
    mean=vector.conj()@images
    dimer=(images.conj().T@images-np.outer(mean.conj(),mean)).real
    coords=np.array([divmod(i,4) for i in range(n)])
    grid=[]
    for k0 in range(4):
        for k1 in range(4):
            phase=np.exp(2j*np.pi*(coords[:,0]*k0+coords[:,1]*k1)/4)
            grid.append(float((phase.conj()@corr@phase).real/n))
    dimer_profile=[float(np.mean([dimer[i,(i//4)*4+(i+r)%4] for i in range(n)])) for r in range(4)]
    # q weights first moment must exactly reproduce the sum of physical pairs.
    return dict(total_spin_squared=float(corr.sum()),spin_correlation_matrix=corr.tolist(),
                spin_structure_factor_grid=np.array(grid).reshape(4,4).tolist(),
                dimer_connected_matrix=dimer.tolist(),horizontal_dimer_profile=dimer_profile,
                dimer_bonds=bonds,max_pair_correlation=float(np.max(corr-np.eye(n)*.75))),corr


def run(config: Path=CONFIG) -> dict:
    """Bounded saved-state analysis; sources and protocol are frozen first."""
    spec=json.loads(config.read_text())
    if spec['task']!='spin_competing_state_readout' or any(spec[k] for k in ('new_ed','new_training','new_samples')):
        raise ValueError('readout may not train, sample or solve a Hamiltonian')
    if spec['n_sites']!=16 or spec['wall_seconds_cap']>600:
        raise ValueError('bounded N16 protocol changed')
    output=ROOT/spec['output']
    if output.exists() and any(p.name!='registration.json' for p in output.iterdir()):
        raise FileExistsError('never overwrite numerical data or a completed readout')
    basis,states,metadata,hashes=_load(spec)
    output.mkdir(parents=True,exist_ok=True)
    registration=dict(config=spec,config_sha256=file_sha256(config),inputs=hashes,
                      state_hashes={name:array_sha256(v) for name,v in states.items()})
    if (output/'registration.json').exists():
        if json.loads((output/'registration.json').read_text())!=json.loads(json.dumps(registration)):
            raise ValueError('cannot resume a different frozen physical-state protocol')
    atomic_json(output/'registration.json',registration)
    start=perf_counter()
    def timeout(*_):
        raise TimeoutError('bounded readout wall budget exhausted')
    prior=signal.signal(signal.SIGALRM,timeout); signal.alarm(spec['wall_seconds_cap'])
    try:
        actions=FixedSzPairActions(basis)
        records,rdms,arrays={}, {}, {'basis':basis}
        h_calls=0
        for name,vector in states.items():
            vector=np.asarray(vector,complex)/np.linalg.norm(vector)
            physical,corr=_observables(vector,actions)
            record=dict(provenance=metadata[name],physical=physical,regions={},
                        strict_su2_eligible=abs(physical['total_spin_squared'])<spec['tolerance'])
            if 'geometry' in metadata[name]:
                factory=triangular_j1j2_bonds if metadata[name]['geometry'].startswith('triangular') else square_j1j2_bonds
                image=actions.weighted_bonds(vector,factory(4,4,j2=metadata[name]['coupling'],periodic=True)); h_calls+=1
                energy=float(np.vdot(vector,image).real)
                record['quality']=dict(energy_per_site=energy/16,variance_per_site=float(np.linalg.norm(image-energy*vector)**2)/16)
                anchor=metadata[name].get('expected_energy_per_site',metadata[name].get('energy_per_site'))
                if anchor is not None and abs(energy/16-anchor)>2e-10:
                    raise ValueError('saved energy replay mismatch: '+name)
            for region,sites in spec['regions'].items():
                rho=reduced_density_matrix(vector,basis,sites)
                rdms[name,region]=rho
                try:
                    readout=su2_multiplicity_readout(rho,tolerance=spec['tolerance'])
                except ValueError as error:
                    if physical['total_spin_squared'] < spec['tolerance']:
                        raise
                    record['strict_su2_eligible']=False
                    record['regions'][region]=dict(status='ineligible_no_projection_repair',reason=str(error))
                    continue
                closure=float(abs(readout['spin_squared_mean']-corr[np.ix_(sites,sites)].sum()))
                if closure > spec['tolerance']:
                    raise ValueError('spin weights fail independent pair-correlation closure')
                readout['spin_pair_closure_error']=closure
                record['regions'][region]=readout
                arrays[name+'_'+region+'_rho']=rho
            arrays[name+'_state']=vector
            records[name]=record
            print(name, 'eligible' if record['strict_su2_eligible'] else 'non-SU2',flush=True)
        comparisons={}
        refs={j:f'triangular_j{j:g}' for j in spec['couplings']}
        for name,record in records.items():
            meta=record['provenance']
            if not record['strict_su2_eligible'] or not meta.get('geometry','').startswith('triangular'):
                continue
            ref=refs[meta['coupling']]
            comparisons[name]={region:_distance(rdms[name,region],rdms[ref,region],record['regions'][region],records[ref]['regions'][region]) for region in spec['regions']}
            record['quality']['energy_error_per_site']=record['quality']['energy_per_site']-records[ref]['quality']['energy_per_site']
            record['quality']['fidelity_to_saved_reference']=float(abs(np.vdot(arrays[name+'_state'],arrays[ref+'_state']))**2)
        coupling_distances={}
        for j in (.075,.125,.175):
            a,b=refs[0.],refs[j]
            coupling_distances[str(j)]={region:_distance(rdms[b,region],rdms[a,region],records[b]['regions'][region],records[a]['regions'][region]) for region in spec['regions']}
        scales=coupling_distances['0.125']
        reliability={name:{region:comparison[region]['physical_trace_distance']/max(scales[region]['physical_trace_distance'],1e-15) for region in spec['regions']}
                     for name,comparison in comparisons.items() if records[name]['provenance']['coupling']==.125}
        tmp=output/'readout.tmp'
        with tmp.open('wb') as handle:
            np.savez_compressed(handle,**arrays)
        tmp.replace(output/'readout.npz')
        result=dict(task=spec['task'],status='complete',config=spec,registration=registration,states=records,
                    errors_to_same_coupling_reference=comparisons,coupling_distances_from_j0=coupling_distances,
                    ansatz_error_over_j0_to_j0125_signal=reliability,
                    seconds=perf_counter()-start,new_ED=0,new_training=0,new_samples=0,new_full_H_actions=h_calls,
                    readout_sha256=file_sha256(output/'readout.npz'),
                    sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')),
                    disposition='Calibration and N16 ansatz/quality sensitivity only. No LE/HE pair or size extrapolation; do not open N36 production on this alone.')
        result['result_sha256']=canonical_sha256(result)
        atomic_json(output/'results.json',result)
        return result
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,prior)


def derive_cyclic_readout(directory: Path) -> dict:
    """Resolve the saved eight-site winding strip by doubled spin and C4.

    Uses only previously computed physical density matrices. The parent
    artifact remains unchanged; a separate derived seal records all inputs
    and the executed implementation. Nonsymmetric controls are not twirled.
    """
    directory = Path(directory)
    target = directory / 'cyclic_readout.json'
    if target.exists():
        raise FileExistsError('do not overwrite a completed cyclic readout')
    parent = json.loads((directory/'results.json').read_text())
    if canonical_sha256({k:v for k,v in parent.items() if k!='result_sha256'}) != parent['result_sha256']:
        raise ValueError('parent numerical seal does not match')
    if file_sha256(directory/'readout.npz') != parent['readout_sha256']:
        raise ValueError('parent density-matrix archive does not match')
    region = 'strip_8'
    if parent['config']['regions'][region] != [0,1,4,5,8,9,12,13]:
        raise ValueError('cyclic analysis requires this fixed winding-strip geometry')
    permutation = (2,3,4,5,6,7,0,1)
    start = perf_counter()
    records = {}
    with np.load(directory/'readout.npz') as archive:
        for name, state in parent['states'].items():
            key = name+'_'+region+'_rho'
            if not state['strict_su2_eligible'] or key not in archive:
                records[name] = dict(status='ineligible',reason='not strict SU2; no projection repair')
                continue
            try:
                readout = su2_cyclic_entanglement_readout(archive[key],permutation,order=4)
            except ValueError as error:
                records[name] = dict(status='ineligible',reason=str(error))
            else:
                records[name] = dict(status='eligible',readout=readout)
    def weights(name: str) -> np.ndarray:
        return np.array([[records[name]['readout']['sectors'][str(q)][str(k)]['p']
                          for k in range(4)] for q in range(0,9,2)])
    def distance(a: str, b: str) -> float:
        return .5*float(np.abs(weights(a)-weights(b)).sum())
    changes = {str(j): dict(joint_q_k_total_variation=distance('triangular_j0',f'triangular_j{j:g}'),
                            delta_p_q_k=(weights(f'triangular_j{j:g}')-weights('triangular_j0')).tolist())
               for j in (.075,.125,.175)}
    errors = {name:dict(joint_q_k_total_variation=distance(name,'triangular_j0.125'),
                       error_over_coupling_signal=distance(name,'triangular_j0.125')/
                       changes['0.125']['joint_q_k_total_variation'])
              for name,state in parent['states'].items()
              if records[name]['status']=='eligible' and
              state['provenance'].get('geometry','').startswith('triangular') and
              state['provenance'].get('coupling')==.125}
    result = dict(task='derived_regional_spin_cyclic_readout',status='complete',
                  parent_result_sha256=file_sha256(directory/'results.json'),
                  parent_readout_sha256=parent['readout_sha256'],region=region,
                  active_permutation=list(permutation),cyclic_order=4,
                  protocol='All saved states; reject broken SU2 or regional translation. No spatial twirl, state selection or energy optimization.',
                  states=records,coupling_distances_from_j0=changes,
                  ansatz_errors_at_j0125=errors,seconds=perf_counter()-start,
                  new_ED=0,new_training=0,new_samples=0,new_full_H_actions=0,
                  sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
                            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')),
                  disposition='Regional entanglement momentum, not bulk K/M. Finite-size calibration, not a competing-phase discriminator or PRL claim.')
    result['result_sha256'] = canonical_sha256(result)
    atomic_json(target,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--cyclic-from',type=Path,help='Derive joint q/C4 readout from an existing immutable result directory')
    args=parser.parse_args()
    result=derive_cyclic_readout(args.cyclic_from) if args.cyclic_from else run(args.config)
    print(json.dumps({k:result[k] for k in ('status','seconds','new_ED','new_samples','new_full_H_actions')}))
