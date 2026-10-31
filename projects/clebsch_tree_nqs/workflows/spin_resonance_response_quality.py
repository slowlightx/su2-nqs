"""Fixed-state six-spin response of one frozen, geometry-labelled operator.

No new ansatz, samples, ED, H queries or physical-state repair.  The source
eta below is a LOCAL FILTER, not a change in the many-body Hamiltonian.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from functools import lru_cache
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.spin_operator_information import joint_spin_operator_split, spin_dot_matrix
from su2_nqs.benchmarks.spin_partition_readout import _three_plus_three_singlet_columns
from su2_nqs.benchmarks.su2_maximum_entropy import invariant_constraint_space, fit_invariant_maximum_entropy
from su2_nqs.vmc.regional_density import regional_density_summary
from su2_nqs.frontier.artifacts import atomic_json


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "projects/clebsch_tree_nqs/results/spin_resonance_response_quality_20261011"
REGION = [20, 21, 15, 8, 7, 13]


def filter_response(rho: np.ndarray, operator: np.ndarray, observables: dict[str, np.ndarray],
                    eta: float = 0.0) -> dict[str, float]:
    """Normalized exp(eta O/2) rho exp(eta O/2); tangent is symmetrized covariance.

    Eta and O are dimensionless.  No clipping, twirling or positivity repair
    occurs.  In particular the response of O at eta=0 is Var(O), not 2Var(O).
    """
    rho, operator = np.asarray(rho, complex), np.asarray(operator, complex)
    if (rho.shape != operator.shape or rho.ndim != 2 or rho.shape[0] != rho.shape[1]
            or not np.isfinite(rho).all() or not np.isfinite(operator).all()
            or np.linalg.norm(rho-rho.conj().T) > 1e-9
            or np.linalg.norm(operator-operator.conj().T) > 1e-9
            or abs(np.trace(rho)-1) > 1e-9 or np.linalg.eigvalsh(rho).min() < -1e-9):
        raise ValueError("physical rho and matching Hermitian operator required")
    eigenvalues, basis = np.linalg.eigh(operator)
    filtering = (basis * np.exp(float(eta)*eigenvalues/2)) @ basis.conj().T
    filtered = filtering @ rho @ filtering
    filtered /= np.trace(filtered)
    mean = float(np.trace(filtered@operator).real)
    result = {"source_mean": mean,
              "source_variance": float(np.trace(filtered@operator@operator).real-mean**2)}
    for name, observable in observables.items():
        value = float(np.trace(filtered@observable).real)
        slope = float(np.trace(filtered@(observable@operator+operator@observable)/2).real-value*mean)
        result[name] = value
        result[name+"__filter_slope"] = slope
    return result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jackknife_errors(values: list[dict]) -> dict[str, float]:
    matrix = np.array([[r[k] for k in values[0]] for r in values])
    error = np.sqrt((len(matrix)-1)/len(matrix)*np.sum((matrix-matrix.mean(0))**2, axis=0))
    return dict(zip(values[0], map(float, error)))


def full_h_response_from_cache(endpoint: int = 9600) -> dict:
    """Pair center R_local reconstructed from conditional Gram rows with H_local.

    Tr(M R) is Rao--Blackwellized over the region and MUST NOT be multiplied
    by the original center's local energy. M[b,a]/M[a,a]=psi(b)/psi(a)
    recovers the correct center ratio, including its complex phase.
    """
    directory=OUTPUT/f'inputs/tri_{endpoint}'
    registration=json.loads((directory/'regional/registration.json').read_text())
    audit=json.loads((directory/'energy.json').read_text())
    regional=json.loads((directory/'regional/results.json').read_text())
    if any(audit[k]!=regional[k] for k in ('step','params_sha256','checkpoint_sha256')):
        raise ValueError('independent full-H and regional parents differ')
    inputs={}
    indices=registration['fixed_snapshot_indices']
    selected_states, selected_energy, full_energy, full_groups=[],[],[],[]
    for stream_id,stream in enumerate(audit['streams']):
        path=directory/f"stream{stream['seed']}.npz"
        digest=_sha(path)
        if digest!=stream['raw_sha256']:
            raise ValueError('audit cache hash mismatch')
        inputs[str(path.relative_to(ROOT))]=digest
        with np.load(path) as data:
            states,energy=data['states'],data['energies']
            if energy.shape!=states.shape[:2]:
                raise ValueError('state/energy record shape mismatch')
            selected_states.append(states[indices])
            selected_energy.append(energy[indices])
            full_energy.append(energy.reshape(-1))
            full_groups.append(np.tile(np.arange(states.shape[1])+stream_id*states.shape[1],states.shape[0]))
    full_energy=np.concatenate(full_energy);full_groups=np.concatenate(full_groups)
    selected_states=np.asarray(selected_states).reshape((-1,36))
    energies=np.asarray(selected_energy).reshape(-1)
    raw_path=directory/'regional/physics.npz'
    if _sha(raw_path)!=regional['raw_sha256']:
        raise ValueError('regional cache hash mismatch')
    frozen_path=OUTPUT/'singlet_ring_analysis_v2/operators.npz'
    if _sha(frozen_path)!='c2696b91c08e17321d1276ddcc61ee06532a35e0d5aba525fa2c3b17c565c4cd':
        raise ValueError('frozen ring operator archive changed')
    with np.load(frozen_path) as z:
        operator=z['O']
    with np.load(raw_path) as data:
        np.testing.assert_array_equal(selected_states,data['states'])
        groups=data['chain_ids']
        matrices=data['local_hexagon6__per_record_matrices']
        local=selected_states[:,REGION]@(2**np.arange(5,-1,-1))
        denominator=matrices[np.arange(len(matrices)),local,local]
        if np.any(denominator.real<=0) or np.max(abs(denominator.imag))>1e-12:
            raise ValueError('sampled center has zero/nonreal conditional Born weight; do not discard')
        column=matrices[np.arange(len(matrices)),:,local]
        ratios=np.einsum('nb,nb->n',operator[local],column)/denominator
        if not np.isfinite(ratios).all():
            raise ValueError('nonfinite operator ratio; do not clip')
    def estimate(keep,keep_full):
        e=full_energy[keep_full].mean();r=ratios[keep].mean()
        slope=np.mean((energies[keep]-e).conj()*ratios[keep]).real/36
        # A secondary same-subset covariance exposes sensitivity to the E center.
        subset_e=energies[keep].mean()
        secondary=np.mean((energies[keep]-subset_e).conj()*ratios[keep]).real/36
        return dict(full_H_filter_slope_per_site=float(slope),
                    paired_subset_center_slope_per_site=float(secondary),
                    direct_R_mean_real=float(r.real),direct_R_mean_imag=float(r.imag),
                    audit_energy_per_site=float(e.real/36),subset_energy_per_site=float(subset_e.real/36))
    point=estimate(np.ones(len(groups),bool),np.ones(len(full_groups),bool))
    deleted=[estimate(groups!=g,full_groups!=g) for g in np.unique(groups)]
    error=_jackknife_errors(deleted)
    mean_e=full_energy.mean()
    contribution=((energies-mean_e).conj()*ratios).real/36
    chain=np.array([contribution[groups==g].mean() for g in np.unique(groups)])
    centered=chain-chain.mean()
    result=dict(endpoint=endpoint,params_sha256=audit['params_sha256'],checkpoint_sha256=audit['checkpoint_sha256'],
        operator_archive_sha256=_sha(frozen_path),raw_inputs_sha256=inputs,
        regional_cache_sha256=_sha(raw_path),audit_json_sha256=_sha(directory/'energy.json'),
        identical_actual_configuration_order=True,selected_snapshot_indices=indices,records=len(ratios),
        independent_chains=len(np.unique(groups)),point=point,whole_chain_SE=error,
        max_abs_R_local=float(np.max(abs(ratios))),minimum_center_conditional_born_weight=float(denominator.real.min()),
        top_four_chains_fraction_squared_influence=float(np.sort(centered**2)[-4:].sum()/np.sum(centered**2)),
        stream_slopes=[estimate(groups<64,full_groups<64),estimate(groups>=64,full_groups>=64)],
        definition='For exp(eta R/2)|psi>, d(E/N)/deta=Re[<(E_local-E)*conjugate(R_local)>]/N. Center ratios, not conditional Tr(MR), paired with same stored configurations.',
        new_samples=0,new_H_calls=0,new_amplitude_queries=0,
        warning='Exploratory full-H tangent evaluated after regional-channel freezing. Not a ground-state susceptibility or finite-step global energy win; tails and finite sampling remain.',workflow_sha256=_sha(Path(__file__)))
    target=OUTPUT/f'full_H_response_step{endpoint:05d}_v1'
    target.mkdir(exist_ok=False)
    np.savez_compressed(target/'paired_records.npz',states=selected_states,chain_ids=groups,energies=energies,R_local=ratios,
                        full_energies=full_energy,full_chain_ids=full_groups)
    result['paired_records_sha256']=_sha(target/'paired_records.npz')
    (target/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def complement_response_from_cache(endpoint: int = 9600) -> dict:
    """Joint-chain uncertainty for d<H-H_patch>/deta, in TOTAL J1 units.

    H-H_patch includes every boundary AND remote bond. The physical local
    response uses the coherent regional estimator; the full-H response uses
    the actual center-ratio estimator. Both delete the SAME full chains.
    """
    full_dir=OUTPUT/f'full_H_response_step{endpoint:05d}_v1'
    full_result=json.loads((full_dir/'results.json').read_text())
    paired_path=full_dir/'paired_records.npz'
    if _sha(paired_path)!=full_result['paired_records_sha256']:
        raise ValueError('paired full-H cache changed')
    with np.load(paired_path) as data:
        energies,ratios,groups=data['energies'],data['R_local'],data['chain_ids']
        all_energy,all_groups=data['full_energies'],data['full_chain_ids']
        states=data['states']
    regional_path=OUTPUT/f'inputs/tri_{endpoint}/regional/physics.npz'
    if _sha(regional_path)!=full_result['regional_cache_sha256']:
        raise ValueError('regional cache changed')
    with np.load(regional_path) as data:
        np.testing.assert_array_equal(groups,data['chain_ids'])
        np.testing.assert_array_equal(states,data['states'])
        matrices=data['local_hexagon6__per_record_matrices']
    operators_path=OUTPUT/'singlet_ring_analysis_v2/operators.npz'
    if _sha(operators_path)!=full_result['operator_archive_sha256']:
        raise ValueError('frozen operator archive changed')
    with np.load(operators_path) as data:
        r,h=data['O'],data['internal_H_patch']
    summary=regional_density_summary(matrices,groups)
    def estimates(keep,all_keep,rho):
        full=float(np.mean((energies[keep]-all_energy[all_keep].mean()).conj()*ratios[keep]).real)
        patch=filter_response(rho,r,{'H_patch':h})['H_patch__filter_slope']
        return np.array([full,patch,full-patch])
    point=estimates(np.ones(len(groups),bool),np.ones(len(all_groups),bool),summary['rho'])
    deleted=np.stack([estimates(groups!=g,all_groups!=g,rho)
        for g,rho in zip(np.unique(groups),summary['jackknife_rhos'])])
    center=deleted-deleted.mean(0)
    covariance=(len(deleted)-1)/len(deleted)*center.T@center
    se=np.sqrt(np.diag(covariance))
    stream=[]
    for selected,full_selected in ((groups<64,all_groups<64),(groups>=64,all_groups>=64)):
        rho=matrices[selected].sum(0);rho/=np.trace(rho)
        stream.append(estimates(selected,full_selected,rho).tolist())
    tail=center[:,2]**2
    result=dict(endpoint=endpoint,columns=['full_H_total','internal_H_patch','H_minus_H_patch'],
        units='TOTAL energy in J1 units per dimensionless eta; full_H_total is 36 times the per-site report',
        point=point.tolist(),whole_chain_SE=se.tolist(),joint_delete_chain_covariance=covariance.tolist(),
        nominal_z=(point/se).tolist(),independent_stream_estimates=stream,
        squared_influence_top_four_chain_fraction_complement=float(np.sort(tail)[-4:].sum()/tail.sum()),
        same_128_chains_deleted=True,errors_combined_independently=False,
        prior_full_H_results_sha256=_sha(full_dir/'results.json'),paired_records_sha256=_sha(paired_path),
        regional_cache_sha256=_sha(regional_path),operator_archive_sha256=_sha(operators_path),
        full_A1_linear_response=dict(
            condition='U_g|psi>=|psi> (or one 1D spatial character) and U_g H U_g^dagger=H',
            identity='For Rbar=mean_g U_g R U_g^dagger, Re<deltaH Rbar>=Re<deltaH R>; therefore first energy derivatives under exp(eta Rbar/2) and exp(eta R/2) agree.',
            finite_filter_equivalent=False,
            reason='The second and higher derivatives of the averaged source contain products of translated/rotated region operators; a single-region RDM cannot determine them. The averaged source preserves A1, a one-region source generally does not.'),
        interpretation='A nominally resolved opposite response of H-H_patch compensates the internal patch response. This complement includes remote bonds, not only boundary bonds. Full-H response remains unresolved; no finite-step energy win, Hamiltonian susceptibility or phase distinction follows.',
        new_samples=0,new_H_calls=0,new_amplitude_queries=0,workflow_sha256=_sha(Path(__file__)))
    target=OUTPUT/f'complement_response_step{endpoint:05d}_v1'
    target.mkdir(exist_ok=False)
    np.savez_compressed(target/'joint_jackknife.npz',point=point,jackknife=deleted,covariance=covariance)
    result['joint_jackknife_sha256']=_sha(target/'joint_jackknife.npz')
    (target/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def singlet_ring_operator() -> tuple[np.ndarray, dict]:
    """SU(2) scalar six-ring exchange, traceless on the five regional singlets.

    The ordered sites form a contractible hexagon. A one-site cyclic shift
    interchanges its two alternating three-dimer coverings. No data select
    coefficients or an eigenvector. The full P0 block is retained.
    """
    bits = (np.arange(64)[:,None] >> np.arange(5,-1,-1)) & 1
    image = np.roll(bits, 1, axis=1) @ (2**np.arange(5,-1,-1))
    cycle = np.zeros((64,64));cycle[image,np.arange(64)]=1
    exchange = (cycle+cycle.T)/2
    columns = _three_plus_three_singlet_columns((0,1,2))
    p0 = columns@columns.conj().T
    scalar = float(np.trace(p0@exchange).real/5)
    operator = p0@exchange@p0-scalar*p0
    return operator, dict(singlet_scalar_subtracted=scalar,
        cyclic_order_error=float(np.linalg.norm(np.linalg.matrix_power(cycle,6)-np.eye(64))),
        singlet_trace_error=float(abs(np.trace(operator))),
        singlet_eigenvalues=np.linalg.eigvalsh(columns.conj().T@operator@columns).tolist())


@lru_cache(maxsize=1)
def ring_channel_projectors() -> dict[str, np.ndarray]:
    """Fixed regional S=0 projectors for cyclic-exchange eigenvalues -1, 1/2, 1.

    These are eigenvalues of R, not angular-momentum or bulk-momentum labels.
    Rank is 2, 2, 1. The sum is the complete five-dimensional singlet projector.
    """
    columns = _three_plus_three_singlet_columns((0,1,2))
    operator,_ = singlet_ring_operator()
    ev,u = np.linalg.eigh(columns.conj().T@operator@columns)
    projectors = {}
    for label,value,rank in (('minus_one',-1.,2),('half',.5,2),('one',1.,1)):
        selected=np.isclose(ev,value,atol=1e-12,rtol=0.)
        if selected.sum()!=rank:
            raise RuntimeError('frozen singlet-ring channel rank changed')
        vectors=columns@u[:,selected]
        p=vectors@vectors.conj().T;p.setflags(write=False)
        projectors[label]=p
    return projectors


def ring_channel_readout(rho: np.ndarray) -> dict[str, float | None]:
    """Channel occupation and sequential block-spin screening on the same RDM.

    R and the three-plus-three quartet projector need not commute. Therefore
    Tr(rho P_r P33 P_r) describes screening AFTER a channel measurement, not
    a joint distribution of two commuting quantum numbers. No SU2 repair.
    """
    operator,_=singlet_ring_operator()
    filter_response(rho,operator,{})  # independent physical-RDM validation
    columns=_three_plus_three_singlet_columns((0,1,2))
    p0=columns@columns.conj().T;p33=np.outer(columns[:,4],columns[:,4].conj())
    singlet=float(np.trace(rho@p0).real)
    out={'p_regional_q0':singlet,'p33_singlet':float(np.trace(rho@p33).real)}
    for label,p in ring_channel_projectors().items():
        weight=float(np.trace(rho@p).real)
        screened=float(np.trace(rho@p@p33@p).real)
        out[label+'_weight']=weight
        out[label+'_given_q0']=weight/singlet if singlet>1e-14 else None
        out[label+'_quartet_sequential_weight']=screened
        out[label+'_quartet_given_channel']=screened/weight if weight>1e-14 else None
    out['quartet_ring_symmetrized_covariance']=filter_response(
        rho,operator,{'quartet':p33})['quartet__filter_slope']
    return out


def ring_maxent_refits(rho: np.ndarray, jackknife_rhos: np.ndarray | None,
                       cache_path: Path | None = None) -> dict:
    """Propagate fitting and measured-RDM error together through whole-chain refits.

    The signed mean residual is relative to fixed p(q), every pair scalar and
    every symmetrized two-bond product. It is a fitted comparator, not a heldout
    prediction test or entanglement measure. Failed fits are retained as errors.
    """
    space=invariant_constraint_space(6);operator,_=singlet_ring_operator()
    reduced=np.zeros((sum(space.dimensions),)*2,complex)
    for sl,us in zip(space.slices,space.magnetic_columns):
        reduced[sl,sl]=us[0].conj().T@operator@us[0]
    input_digest=hashlib.sha256(np.ascontiguousarray(rho).tobytes())
    if jackknife_rhos is not None:
        input_digest.update(np.ascontiguousarray(jackknife_rhos).tobytes())
    digest=input_digest.hexdigest();started=perf_counter()
    cache={'input_sha256':digest,'fits':[]}
    if cache_path is not None and cache_path.exists():
        cache=json.loads(cache_path.read_text())
        if cache['input_sha256']!=digest:
            raise ValueError('refit cache belongs to another physical RDM')
    def evaluate(matrix):
        omega=space.data(matrix)
        report,model,_=fit_invariant_maximum_entropy(omega,space)
        if not report['certified_moment_fit']:
            raise RuntimeError('ring MaxEnt moment fit failed; no result promoted: '+json.dumps({
                k:report[k] for k in ('iterations','optimizer_message','moment_residual_max',
                                      'relative_entropy_gap_closure')}))
        actual=float(np.trace(matrix@operator).real)
        predicted=float(np.trace(model@reduced).real)
        return dict(actual_mean=actual,predicted_mean=predicted,excess_mean=actual-predicted,
                    moment_residual=report['moment_residual_max'],
                    moment_refinement_iterations=report.get('moment_refinement_iterations',0),
                    information_TR_even=report['information_TR_even'])
    targets=[rho]+([] if jackknife_rhos is None else list(jackknife_rhos))
    for index in range(len(cache['fits']),len(targets)):
        cache['fits'].append(evaluate(targets[index]))
        if cache_path is not None:atomic_json(cache_path,cache)
    errors=None if jackknife_rhos is None else _jackknife_errors(cache['fits'][1:])
    return dict(point=cache['fits'][0],whole_chain_refit_SE=errors,
        full_refit_count=len(cache['fits']),maximum_moment_residual=max(x['moment_residual'] for x in cache['fits']),
        constraints='p(q), all pair scalars, all symmetrized two-bond products',
        physical_state_replaced=False,finite_sample_fit_bias_corrected=False,
        heldout_prediction_test=False,seconds_this_invocation=perf_counter()-started)


def registered_quality_series(config: Path) -> dict:
    """Zero-query readout of explicitly hashed NQS/MPS quality sequences.

    Absolute inputs are allowed for a separate Kim reader workspace, but every
    input must be declared by byte hash. NQS caches also require their original
    canonical seals and matching audit/parameter/checkpoint identities. Ordered
    cuts and the labelled physical Hamiltonian are checked before comparison.
    q labels are doubled; reduced matrices are invariant-operator data, never
    spectra of a repaired physical state. No producer or source state is edited.
    """
    from .spin_su2_analysis import _problem_signature, _regional_scalar_data, _singlet_channel_scalars
    from su2_nqs.frontier.artifacts import canonical_sha256

    spec=json.loads(config.read_text()); destination=Path(spec['output'])
    if not destination.is_absolute():destination=ROOT/destination
    config_hash=_sha(config)
    if (destination/'completion.json').exists():
        seal=json.loads((destination/'completion.json').read_text())
        if seal['config_sha256']!=config_hash:
            raise ValueError('completed series belongs to another manifest')
        for name in ('results.json','readouts.npz'):
            if _sha(destination/name)!=seal[name]:raise ValueError('completed series changed')
        return json.loads((destination/'results.json').read_text())
    destination.mkdir(parents=True,exist_ok=True)
    started=perf_counter(); inputs={}; records=[]; arrays={}

    def checked(name: str) -> Path:
        path=Path(name)
        if not path.is_absolute():path=ROOT/path
        expected=spec['inputs_sha256'].get(name)
        if expected is None or _sha(path)!=expected:
            raise ValueError('missing or changed manifest input: '+name)
        inputs[str(path)]=expected
        return path

    def read(name: str, *, sealed: bool=False) -> dict:
        row=json.loads(checked(name).read_text())
        if sealed and row.get('result_sha256')!=canonical_sha256(
                {k:v for k,v in row.items() if k!='result_sha256'}):
            raise ValueError('original numerical seal changed: '+name)
        return row

    labels=[s['label'] for s in spec['sources']]
    if len(labels)!=len(set(labels)):raise ValueError('unique physical-state labels required')
    for source in spec['sources']:
        if perf_counter()-started>spec['seconds_cap']:
            raise RuntimeError('bounded series analysis expired; no completion promoted')
        problem=source['problem']; signature,bonds=_problem_signature(problem)
        definition=dict(label=source['label'],method=source['kind'],problem=problem,
                        physical_problem_signature=signature)
        if source['kind']=='NQS':
            audit=read(source['audit'],sealed=True)
            saved=read(source['regional'],sealed=True)
            registration=read(source['registration'],sealed=True)
            if (saved['physical_problem_signature']!=signature
                    or registration['config']['problem']!=problem
                    or saved['registration_sha256']!=registration['result_sha256']
                    or any(saved[k]!=audit[k] or saved[k]!=registration[k]
                           for k in ('params_sha256','checkpoint_sha256'))
                    or audit['step']!=source['step'] or saved['step']!=source['step']):
                raise ValueError('NQS cache/audit/physical parent mismatch')
            raw=checked(source['raw'])
            if _sha(raw)!=saved['raw_sha256']:raise ValueError('original regional raw hash changed')
            definition.update(step=source['step'],energy=audit['energy'],
                params_sha256=audit['params_sha256'],checkpoint_sha256=audit['checkpoint_sha256'],
                total_spin_squared=audit.get('total_spin_squared'),
                structure_factors=audit.get('structure_factors'),strict_SU2=False)
            with np.load(raw) as z:
                if len(np.unique(z['chain_ids']))!=saved['independent_chains']:
                    raise ValueError('saved whole-chain layout changed')
                matrices={name:(z[name+'__rho'],z[name+'__jackknife_rhos'])
                          for name in problem['regions']}
        elif source['kind']=='DMRG':
            registration=read(source['registration']); rc=registration['config']
            if any(rc[k]!=problem[k] for k in ('rows','columns','j2')):
                raise ValueError('MPS Hamiltonian metadata changed')
            from su2_nqs.benchmarks.readout_quality import physical_problem_signature
            if physical_problem_signature(problem['rows']*problem['columns'],registration['bonds'],
                    geometry=f"{problem['geometry']}_{problem['rows']}x{problem['columns']}_row_major",
                    boundary=problem['boundary'])!=signature:
                raise ValueError('MPS labelled physical Hamiltonian changed')
            status=read(source['status']); raw=checked(source['raw'])
            if source.get('fixed_environment',False):
                if status['status']!='complete':raise ValueError('incomplete finite-MPS reader')
                matching=[r for r in status['regions'] if r['region']['name']==source['cut']]
                if len(matching)!=1:raise ValueError('ambiguous MPS region')
                row=matching[0]
                if (row['region']['sites']!=problem['regions'][source['cut']]
                        or row['rdm_sha256']!=_sha(raw) or not row['checkpoint_unchanged']
                        or not row['mps_unchanged'] or row['checkpoint_sha256']!=source['checkpoint_sha256']):
                    raise ValueError('finite-MPS state/order/normalization provenance changed')
            elif (status['global_sweep']!=source['sweep'] or status['chi']!=source['chi']
                    or status['rdms_sha256']!=_sha(raw)):
                raise ValueError('completed sweep RDM identity changed')
            definition.update(chi=source['chi'],sweep=source['sweep'],strict_SU2=False,
                energy=dict(energy_per_site=source['energy_per_site'],variance_measured=False))
            with np.load(raw) as z:
                matrices={name:(z[key],None) for name,key in source['array_keys'].items()}
                if 'sites' in z and list(z['sites'])!=problem['regions'][source['cut']]:
                    raise ValueError('raw finite-MPS ordered cut changed')
            saved=None
        else:raise ValueError('only frozen NQS and completed DMRG inputs are accepted')
        for name,(rho,jack) in matrices.items():
            cut=dict(sites=problem['regions'][name],left=problem['left_sites'][name],
                cyclic=name in problem.get('cyclic_regions',[]),
                compact_triangular_2x3=name in problem.get('compact_triangular_2x3',[]))
            report,blocks,scalars=_regional_scalar_data(rho,cut)
            extra,internal=_singlet_channel_scalars(rho,cut);scalars.update(extra)
            if saved is not None:
                cached=saved['regions'][name]
                if cached['sites']!=cut['sites'] or cached['left']!=cut['left']:
                    raise ValueError('cached cut/partition changed')
                for key,value in cached['scalars'].items():
                    if key in scalars and not np.isclose(value,scalars[key],atol=2e-12,rtol=0):
                        raise ValueError('cached physical readout changed: '+key)
                deleted=[_singlet_channel_scalars(a,cut)[0] for a in jack]
                valid=[k for k in extra if all(k in r for r in deleted)]
                errors=dict(cached['scalar_whole_chain_SE'],**_jackknife_errors(
                    [{k:r[k] for k in valid} for r in deleted]))
            else:errors=None
            row=dict(**definition,cut=name,sites=cut['sites'],left=cut['left'],
                scalars=scalars,whole_chain_SE=errors,
                symmetry_defect=report['multiplicity']['symmetry_defect_trace_distance'],
                strict_entanglement_spectrum=False,coherence_norm_bias_removed=False)
            if name=='local_hexagon6':
                if problem['geometry']!='triangular' or cut['sites']!=REGION:
                    raise ValueError('frozen ring requires the declared contractible hexagon')
                point=dict(filter_response(rho,singlet_ring_operator()[0],{}),**ring_channel_readout(rho))
                deleted=[] if jack is None else [dict(filter_response(a,singlet_ring_operator()[0],{}),
                        **ring_channel_readout(a)) for a in jack]
                valid=[k for k,v in point.items() if v is not None and all(r[k] is not None for r in deleted)]
                row['ring']=dict(point=point,whole_chain_SE=None if jack is None else
                    _jackknife_errors([{k:r[k] for k in valid} for r in deleted]),
                    maxent=ring_maxent_refits(rho,jack,destination/(source['label']+'_refit.json')))
            records.append(row)
            prefix=source['label']+'__'+name+'__'
            arrays.update({prefix+'rho':rho,**{prefix+'sigma_q'+str(q):a for q,a in blocks.items()},
                           **{prefix+k:a for k,a in internal.items()}})
        print(json.dumps(dict(completed=source['label'],regions=list(matrices))),flush=True)
    result=dict(records=records,inputs_sha256=inputs,config_sha256=config_hash,seconds=perf_counter()-started,
        new_samples=0,new_H_calls=0,new_amplitude_queries=0,new_MPS_contractions=0,new_training=0,
        phase_claim=False,strict_entanglement_spectrum=False,
        warning='Independent endpoint whole-chain errors; deterministic chi/sweep drifts are sensitivity, not certified bias bounds. Methods are not competing phases. Conditional ratios/norms and MaxEnt excess retain finite-sample bias. No physical state is projected, twirled or repaired.',
        workflow_sha256=_sha(Path(__file__)))
    np.savez_compressed(destination/'readouts.npz',**arrays)
    atomic_json(destination/'results.json',result)
    atomic_json(destination/'completion.json',dict(config_sha256=config_hash,
        **{name:_sha(destination/name) for name in ('results.json','readouts.npz')}))
    return result


def advance_quality() -> dict:
    """Finish the same frozen-channel late-ViT/DMRG quality comparison from caches."""
    destination=OUTPUT/'quality_refit_v1';destination.mkdir(exist_ok=True)
    if (destination/'completion.json').exists():
        seals=json.loads((destination/'completion.json').read_text())
        if _sha(destination/'results.json')!=seals['results.json']:
            raise ValueError('completed quality artifact changed')
        return json.loads((destination/'results.json').read_text())
    inputs={};sources=[]
    def checked(path,expected=None):
        digest=_sha(path)
        if expected is not None and digest!=expected:raise ValueError('quality input hash mismatch')
        inputs[str(path.relative_to(ROOT))]=digest
    frozen=OUTPUT/'singlet_ring_analysis_v2/operators.npz'
    checked(frozen,'c2696b91c08e17321d1276ddcc61ee06532a35e0d5aba525fa2c3b17c565c4cd')
    with np.load(frozen) as z:operator,h=z['O'],z['internal_H_patch']
    baseline=OUTPUT/'singlet_ring_analysis_v3';checked(baseline/'completion.json')
    sealed=json.loads((baseline/'completion.json').read_text())
    checked(baseline/'results.json',sealed['results.json']);checked(baseline/'readouts.npz',sealed['readouts.npz'])
    previous=json.loads((baseline/'results.json').read_text())
    with np.load(baseline/'readouts.npz') as z:
        row=next(x for x in previous['records'] if x['label']=='ViT_7600')
        sources.append(dict(label='ViT_7600',method='ViT',step=7600,
            energy_per_site=row['energy_per_site'],rho=z['ViT_7600__rho'],jack=z['ViT_7600__jackknife_rhos']))
    for step in (8600,9600,10600):
        folder=OUTPUT/f'inputs/tri_{step}'
        report=json.loads((folder/'regional/results.json').read_text())
        checked(folder/'regional/results.json');checked(folder/'regional/physics.npz',report['raw_sha256'])
        audit=json.loads((folder/'energy.json').read_text());checked(folder/'energy.json')
        if any(audit[k]!=report[k] for k in ('step','params_sha256','checkpoint_sha256')):
            raise ValueError('quality audit and regional cache differ')
        with np.load(folder/'regional/physics.npz') as z:
            summary=regional_density_summary(z['local_hexagon6__per_record_matrices'],z['chain_ids'])
        sources.append(dict(label=f'ViT_{step}',method='ViT',step=step,
            energy_per_site=audit['energy']['energy_per_site'],energy_SE=audit['energy']['chain_mean_standard_error'],
            rho=summary['rho'],jack=summary['jackknife_rhos']))
    for sweep,folder in ((28,'chi8192_sweep28_hexagon6_env_v3'),(30,'chi8192_sweep30_hexagon6_env_v1')):
        path=OUTPUT/f'inputs/dmrg/{folder}'
        report=json.loads((path/'results.json').read_text());checked(path/'results.json')
        checked(path/'regions/local_hexagon6/rdm.npz',report['regions'][0]['rdm_sha256'])
        with np.load(path/'regions/local_hexagon6/rdm.npz') as z:
            if list(z['sites'])!=REGION:raise ValueError('DMRG ordered hexagon changed')
            rho=z['rho']
        sources.append(dict(label=f'DMRG_8192_{sweep}',method='DMRG',sweep=sweep,chi=8192,
            energy_per_site=(-.515084738960105 if sweep==28 else -.5150870027306547),rho=rho,jack=None))
    protocol=dict(physical_region=REGION,operator_sha256=_sha(frozen),
        endpoints=[x['label'] for x in sources],post_9600_quality_update=True,
        statistics='Whole-chain delete-one RDMs; every deletion fully refits the fixed MaxEnt comparator. Endpoint errors are unpaired; DMRG drift is deterministic quality sensitivity.',
        channel_screening='Tr(rho P_r P33 P_r): quartet occupation after ring-channel projection. Noncommuting channels are not presented as a joint charge distribution.',
        new_samples=0,new_H_calls=0,new_training=0)
    if (destination/'protocol.json').exists() and json.loads((destination/'protocol.json').read_text())!=protocol:
        raise ValueError('registered quality protocol changed')
    atomic_json(destination/'protocol.json',protocol)
    rows=[];arrays={}
    for source in sources:
        label=source['label'];rho=source['rho'];jack=source['jack']
        point=dict(filter_response(rho,operator,{'H_patch':h}),**ring_channel_readout(rho))
        deleted=[] if jack is None else [dict(filter_response(a,operator,{'H_patch':h}),**ring_channel_readout(a)) for a in jack]
        keys=[k for k,v in point.items() if v is not None and all(x[k] is not None for x in deleted)]
        se=None if jack is None else _jackknife_errors([{k:x[k] for k in keys} for x in deleted])
        refits=ring_maxent_refits(rho,jack,destination/f'{label}_refit_cache.json')
        row={k:v for k,v in source.items() if k not in ('rho','jack')}
        row.update(point=point,whole_chain_SE=se,maxent=refits)
        rows.append(row);arrays[label+'__rho']=rho
        if jack is not None:arrays[label+'__jackknife_rhos']=jack
        print(json.dumps(dict(completed=label,excess=refits['point']['excess_mean'],
                             refit_SE=None if se is None else refits['whole_chain_refit_SE']['excess_mean'])),flush=True)
    dmrg_old,dmrg_new=rows[-2:];vit_old,vit_new=rows[2:4]
    comparisons={}
    for field in ('source_mean','source_variance','H_patch__filter_slope','p_regional_q0',
                  'minus_one_given_q0','half_given_q0','one_given_q0'):
        delta=vit_new['point'][field]-dmrg_new['point'][field]
        se=vit_new['whole_chain_SE'][field]
        vdrift=abs(vit_new['point'][field]-vit_old['point'][field])
        ddrift=abs(dmrg_new['point'][field]-dmrg_old['point'][field])
        comparisons[field]=dict(difference=delta,ViT_SE=se,ViT_last_quality_drift=vdrift,
            DMRG_last_quality_drift=ddrift,exceeds_2SE_and_both_drifts=bool(abs(delta)>2*se+vdrift+ddrift))
    trace_drift=float(np.abs(np.linalg.eigvalsh(sources[-1]['rho']-sources[-2]['rho'])).sum()/2)
    result=dict(protocol=protocol,records=rows,comparisons=comparisons,
        dmrg_raw_RDM_trace_distance_28_to_30=trace_drift,inputs_sha256=inputs,
        physical_phase_claim=False,strict_entanglement_spectrum=False,
        response_type='equal-time local filtering, not a reoptimized Hamiltonian susceptibility',
        workflow_sha256=_sha(Path(__file__)))
    np.savez_compressed(destination/'readouts.npz',**arrays)
    atomic_json(destination/'results.json',result)
    atomic_json(destination/'completion.json',{name:_sha(destination/name) for name in ('results.json','protocol.json','readouts.npz')})
    return result


def magnetic_late_quality() -> dict:
    """Collect sealed S.S quality data, retaining all three unaveraged M points.

    An M-orbit average is an observable with paired chain covariance, not a
    symmetrized replacement for either physical state. No new sampling occurs.
    """
    from .spin_spatial_vit_quality import read_sealed
    destination=OUTPUT/'magnetic_late_quality_v1'
    if (destination/'completion.json').exists():
        seal=json.loads((destination/'completion.json').read_text())
        if _sha(destination/'results.json')!=seal['results.json']:
            raise ValueError('magnetic quality seal changed')
        return json.loads((destination/'results.json').read_text())
    destination.mkdir(exist_ok=True)
    rows=[];inputs={};labels=('Gamma','K','minus_K','M1','M2','M3')
    def checked(path,expected=None):
        digest=_sha(path)
        if expected is not None and digest!=expected:raise ValueError('magnetic input seal mismatch')
        inputs[str(path.relative_to(ROOT))]=digest
    for step in (8600,9600,10600):
        folder=OUTPUT/f'inputs/tri_{step}'
        audit=read_sealed(folder/'energy.json');report=read_sealed(folder/'magnetic/results.json')
        checked(folder/'energy.json');checked(folder/'magnetic/results.json')
        checked(folder/'magnetic/physics.npz',report['raw_sha256'])
        if any(audit[k]!=report[k] for k in ('step','checkpoint_sha256','params_sha256')):
            raise ValueError('magnetic and full-H physical parents differ')
        with np.load(folder/'magnetic/physics.npz') as data:
            ids=np.unique(data['chain_ids'])
            counts=[np.count_nonzero(data['chain_ids']==g) for g in ids]
            if len(ids)!=128 or len(set(counts))!=1:
                raise ValueError('registered equal-record whole-chain layout differs')
            means=np.stack([data['scalar'][data['chain_ids']==g].real.mean(0) for g in ids])
        covariance=np.cov(means,rowvar=False)/len(ids)
        np.testing.assert_allclose(covariance,report['scalar_whole_chain_covariance_of_means'],atol=2e-12,rtol=0)
        point=means.mean(0)
        np.testing.assert_allclose(point,[report['structure_factors'][k]['scalar']['value'] for k in labels],atol=2e-12,rtol=0)
        weights=np.array([0,0,0,1,1,1])/3
        row=dict(method='ViT',step=step,label=f'ViT_{step}',**audit['energy'],
            structure_factors=dict(zip(labels,map(float,point))),
            whole_chain_SE=dict(zip(labels,map(float,np.sqrt(np.diag(covariance))))),
            M_orbit_mean=float(weights@point),M_orbit_mean_SE=float(np.sqrt(weights@covariance@weights)),
            total_spin_squared=report['total_spin_squared'],strict_SU2=False,full_A1=True)
        rows.append(row)
    for sweep in (28,30):
        folder=OUTPUT/f'inputs/dmrg/chi8192_sweep{sweep}_spin_env_v3'
        report=json.loads((folder/'results.json').read_text());checked(folder/'results.json')
        region=report['regions'][0]
        checked(folder/'regions/spin/rdm.npz',region['rdm_sha256'])
        if (region['actual_chi']!=8192 or not region['mps_unchanged']
                or 'full finite left/right environments' not in region['diagnostics']['contraction']):
            raise ValueError('normalized finite-MPS spin contraction required')
        sf=region['diagnostics']['structure_factors']
        rows.append(dict(method='DMRG',sweep=sweep,chi=8192,label=f'DMRG_8192_{sweep}',
            energy_per_site=(-.515084738960105 if sweep==28 else -.5150870027306547),
            structure_factors=sf,whole_chain_SE=None,M_orbit_mean=float(np.mean([sf[k] for k in labels[3:]])),
            total_spin_squared=region['diagnostics']['spin_squared'],full_A1=False,
            variance_measured=False,checkpoint_sha256=region['checkpoint_sha256']))
    result=dict(problem=dict(geometry='triangular',rows=6,columns=6,j2=.125,boundary='PBC_both_physical'),
        records=rows,inputs_sha256=inputs,new_samples=0,new_H_calls=0,new_training=0,
        definition='S(k)=sum_ij exp[2pi i k.(r_i-r_j)] <S_i.S_j>/N. Scalar, not 3SzSz.',
        warning='No thermodynamic magnetic moment or phase assignment. All M values retained; their orbit mean does not project a finite-chi MPS into A1. Endpoint errors are independent; deterministic chi drift is not a certified error bound.',
        workflow_sha256=_sha(Path(__file__)))
    atomic_json(destination/'results.json',result)
    atomic_json(destination/'completion.json',{'results.json':_sha(destination/'results.json')})
    return result


def render_late_quality() -> dict:
    """Plot sealed late-quality data; no bars, phase labels or strict spectra."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUTPUT/'quality_refit_v1'
    seal=json.loads((folder/'completion.json').read_text())
    if _sha(folder/'results.json')!=seal['results.json']:
        raise ValueError('sealed ring-quality result changed')
    result=json.loads((folder/'results.json').read_text())
    vit=[r for r in result['records'] if r['method']=='ViT']
    reference=[r for r in result['records'] if r['method']=='DMRG']
    target=ROOT/'projects/clebsch_tree_nqs/figures/exploration';target.mkdir(exist_ok=True)
    paths={}
    def save(fig,name):
        for ext in ('png','pdf'):
            path=target/(name+'.'+ext);fig.savefig(path,dpi=180,bbox_inches='tight')
            paths[name+'.'+ext]=str(path)
        plt.close(fig)
    def reference_band(ax,values,label='DMRG 8192, sweeps 28--30'):
        low,high=min(values),max(values)
        ax.axhline(values[-1],color='C3',ls='--',lw=1,label=label)
        ax.axhspan(low,high,color='C3',alpha=.18)
    steps=[r['step'] for r in vit]
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,field,title in zip(axes,('source_mean','H_patch__filter_slope','excess_mean'),
        (r'$\langle R\rangle$',r'$\partial_\eta\langle H_A\rangle_{\mathrm{filter}}$',r'$\langle R\rangle-\langle R\rangle_{\mathrm{MaxEnt}}$')):
        if field=='excess_mean':
            values=[r['maxent']['point'][field] for r in vit]
            errors=[r['maxent']['whole_chain_refit_SE'][field] for r in vit]
            ref=[r['maxent']['point'][field] for r in reference]
        else:
            values=[r['point'][field] for r in vit];errors=[r['whole_chain_SE'][field] for r in vit]
            ref=[r['point'][field] for r in reference]
        ax.errorbar(steps,values,yerr=errors,fmt='o-',capsize=3,label='ViT, whole-chain 1 SE')
        reference_band(ax,ref);ax.axhline(0,color='gray',lw=.6)
        ax.set(xlabel='Frozen training endpoint',title=title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Triangular 6x6 PBC, J2/J1=0.125: contractible-hexagon quality validation',fontsize=11)
    save(fig,'singlet_ring_late_quality_refitted')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for key,label,color in (('minus_one',r'$r=-1$','C0'),('half',r'$r=1/2$','C1'),('one',r'$r=1$','C2')):
        for ax,field in zip(axes,(key+'_given_q0',key+'_quartet_given_channel')):
            ax.errorbar(steps,[r['point'][field] for r in vit],
                yerr=[r['whole_chain_SE'][field] for r in vit],fmt='o-',color=color,capsize=3,label=label+' ViT')
            ref=[r['point'][field] for r in reference]
            ax.axhline(ref[-1],color=color,ls='--',lw=1)
            ax.axhspan(min(ref),max(ref),color=color,alpha=.13)
    axes[0].set(title='Ring channel given regional singlet',ylabel=r'$p(r\mid q_A=0)$')
    axes[1].set(title='Quartet screening after channel projection',ylabel=r'$\mathrm{Tr}(\rho P_rP_{33}P_r)/p(r)$')
    for ax in axes:ax.set(xlabel='Frozen training endpoint');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Dashed: DMRG 8192; screening is sequential, not a joint charge probability',fontsize=10)
    save(fig,'singlet_ring_channels_and_screening')
    region_folder=OUTPUT/'regional_late_quality_v1'
    if (region_folder/'completion.json').exists():
        regional=json.loads((region_folder/'results.json').read_text())
        seals=json.loads((region_folder/'completion.json').read_text())
        if _sha(region_folder/'results.json')!=seals['results.json']:
            raise ValueError('regional plotting input changed')
        fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained')
        columns=('compact6','local_hexagon6','winding6')
        fields=('rows_p33_singlet','rows_p33_high','rows_doublet_conditional_operator_purity')
        labels=(r'$p(S_L=S_R=3/2,S_A=0)$',r'$p_{33}$: orthogonal high-order part',r'Doublet-block operator purity')
        for col,cut in enumerate(columns):
            neural=sorted((r for r in regional['records'] if r['method']=='ViT' and r['cut']==cut),key=lambda r:r['step'])
            ref=sorted((r for r in regional['records'] if r['method']=='DMRG' and r['cut']==cut),key=lambda r:r['sweep'])
            for row,field in enumerate(fields):
                ax=axes[row,col]
                ax.errorbar([r['step'] for r in neural],[r['scalars'][field] for r in neural],
                    yerr=[r['whole_chain_SE'][field] for r in neural],fmt='o-',capsize=3,label='ViT')
                reference_band(ax,[r['scalars'][field] for r in ref],f"DMRG {ref[0]['sweep']}--{ref[-1]['sweep']}")
                if row==0:ax.set_title(cut)
                if col==0:ax.set_ylabel(labels[row])
                if row==2:ax.set_xlabel('Frozen training endpoint')
                ax.grid(alpha=.2)
            axes[0,col].legend(fontsize=7)
        fig.suptitle('Same triangular torus/cuts: screening and resonance quality sensitivity\nPurity is invariant operator data, not a strict entanglement spectrum',fontsize=11)
        save(fig,'three_region_late_quality')
        fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
        field='connected_dimer_0_1_3_4'
        for ax,cut in zip(axes,columns):
            neural=sorted((r for r in regional['records'] if r['method']=='ViT' and r['cut']==cut),key=lambda r:r['step'])
            ref=sorted((r for r in regional['records'] if r['method']=='DMRG' and r['cut']==cut),key=lambda r:r['sweep'])
            ax.errorbar([r['step'] for r in neural],[r['scalars'][field] for r in neural],
                yerr=[r['whole_chain_SE'][field] for r in neural],fmt='o-',capsize=3,label='ViT, chain 1 SE')
            reference_band(ax,[r['scalars'][field] for r in ref],f"DMRG {ref[0]['sweep']}--{ref[-1]['sweep']}")
            ax.set(title=cut,xlabel='Frozen training endpoint');ax.grid(alpha=.2)
        axes[0].set_ylabel(r'$\langle B_{01}B_{34}\rangle-\langle B_{01}\rangle\langle B_{34}\rangle$')
        axes[0].legend(fontsize=7)
        fig.suptitle('Connected dimer quality control: ordered sites are cut-local labels, not identical geometries',fontsize=10)
        save(fig,'three_region_connected_dimer_quality')
    magnetic_folder=OUTPUT/'magnetic_late_quality_v1'
    if (magnetic_folder/'completion.json').exists():
        seals=json.loads((magnetic_folder/'completion.json').read_text())
        if _sha(magnetic_folder/'results.json')!=seals['results.json']:
            raise ValueError('magnetic plotting input changed')
        magnetic=json.loads((magnetic_folder/'results.json').read_text())
        neural=[r for r in magnetic['records'] if r['method']=='ViT']
        ref=[r for r in magnetic['records'] if r['method']=='DMRG']
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        x=[r['step'] for r in neural]
        for ax,key,err,title in ((axes[0,0],'energy_per_site','chain_mean_standard_error',r'$E/N$'),
                                (axes[0,1],'variance_per_site','variance_chain_standard_error',r'$\mathrm{Var}(H)/N$')):
            ax.errorbar(x,[r[key] for r in neural],yerr=[r[err] for r in neural],fmt='o-',capsize=3,label='ViT, independent chain 1 SE')
            if key=='energy_per_site':reference_band(ax,[r[key] for r in ref])
            else:ax.text(.02,.9,'DMRG variance not measured',transform=ax.transAxes,fontsize=9)
            ax.set_title(title)
        axes[1,0].errorbar(x,[r['structure_factors']['K'] for r in neural],
            yerr=[r['whole_chain_SE']['K'] for r in neural],fmt='o-',capsize=3,label='ViT, K = -K')
        reference_band(axes[1,0],[r['structure_factors']['K'] for r in ref]);axes[1,0].set_title(r'$S(K)$')
        for key,color in zip(('M1','M2','M3'),('C0','C1','C2')):
            axes[1,1].errorbar(x,[r['structure_factors'][key] for r in neural],
                yerr=[r['whole_chain_SE'][key] for r in neural],fmt='o-',capsize=3,color=color,label=key+' ViT')
            values=[r['structure_factors'][key] for r in ref]
            axes[1,1].axhline(values[-1],color=color,ls='--',lw=1)
            axes[1,1].axhspan(min(values),max(values),color=color,alpha=.13)
        axes[1,1].set_title('All M points (dashed: DMRG)')
        for ax in axes.flat:ax.set_xlabel('Frozen training endpoint');ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Triangular 6x6 PBC, J2/J1=0.125: late energy and scalar magnetic quality',fontsize=11)
        save(fig,'vit_late_energy_and_magnetic_quality')
    response_paths=[OUTPUT/f'complement_response_step{step:05d}_v1/results.json' for step in (9600,10600)]
    if all(p.exists() for p in response_paths):
        response=[json.loads(p.read_text()) for p in response_paths]
        for row,path in zip(response,response_paths):
            if (row['columns']!=['full_H_total','internal_H_patch','H_minus_H_patch']
                    or not row['same_128_chains_deleted']
                    or _sha(path.parent/'joint_jackknife.npz')!=row['joint_jackknife_sha256']):
                raise ValueError('paired environment-response artifact differs')
        fig,ax=plt.subplots(figsize=(7.4,4),layout='constrained')
        for i,label,color in ((0,'Full H','C0'),(1,'Internal hexagon bonds','C1'),
                              (2,'H - H_patch (boundary + remote)','C2')):
            ax.errorbar([r['endpoint'] for r in response],[r['point'][i] for r in response],
                yerr=[r['whole_chain_SE'][i] for r in response],fmt='o-',capsize=4,color=color,label=label)
        ax.axhline(0,color='gray',lw=.7);ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.set(xlabel='Frozen training endpoint',ylabel=r'$\partial_\eta\langle H_X\rangle$ (TOTAL $J_1$ units)',
               title='Frozen singlet-ring filter: local and environment compensation')
        fig.suptitle('Equal-time tangent, not a reoptimized Hamiltonian susceptibility',fontsize=10)
        save(fig,'singlet_ring_full_H_environment_quality')
    atomic_json(folder/'figures.json',paths)
    return paths


def regional_late_quality() -> dict:
    """Reuse the common scalar protocol on three cuts and late quality pairs.

    Source RDMs and energy audits are sealed; these are independent endpoint
    estimates, not paired Monte Carlo estimates or competing phase labels.
    All magnetic indices are traced only for invariant operator data. No
    noninvariant empirical RDM is assigned a strict entanglement spectrum.
    """
    from .spin_su2_analysis import _regional_scalar_data,_singlet_channel_scalars
    from .spin_spatial_vit_quality import read_sealed
    destination=OUTPUT/'regional_late_quality_v1'
    if (destination/'completion.json').exists():
        seal=json.loads((destination/'completion.json').read_text())
        if _sha(destination/'results.json')!=seal['results.json']:
            raise ValueError('regional quality seal changed')
        return json.loads((destination/'results.json').read_text())
    cuts={'compact6':dict(sites=[12,13,14,18,19,20],left=[0,1,2],compact_triangular_2x3=True),
          'winding6':dict(sites=[12,13,14,15,16,17],left=[0,1,2],cyclic=True),
          'local_hexagon6':dict(sites=REGION,left=[0,1,2])}
    records=[];inputs={};arrays={};started=perf_counter()
    def checked(path,expected=None):
        value=_sha(path)
        if expected is not None and value!=expected:raise ValueError('regional input seal mismatch')
        inputs[str(path.relative_to(ROOT))]=value
    for step in (8600,9600,10600):
        folder=OUTPUT/f'inputs/tri_{step}';audit=read_sealed(folder/'energy.json')
        saved=read_sealed(folder/'regional/results.json');registration=read_sealed(folder/'regional/registration.json')
        for name in ('energy.json','regional/results.json','regional/registration.json'):checked(folder/name)
        checked(folder/'regional/physics.npz',saved['raw_sha256'])
        if (saved['physical_problem_signature']!='32b5fdc2bf743c63986f30d3742112a81fe0067f3270dc026b2957bd7c328fe6'
                or any(saved[k]!=audit[k] for k in ('step','checkpoint_sha256','params_sha256'))
                or saved['registration_sha256']!=registration['result_sha256']):
            raise ValueError('late regional/H audit physical parent mismatch')
        with np.load(folder/'regional/physics.npz') as z:
            for name,cut in cuts.items():
                cached=saved['regions'][name]
                if cached['sites']!=cut['sites'] or cached['left']!=cut['left']:
                    raise ValueError('ordered cut/partition changed')
                summary=regional_density_summary(z[name+'__per_record_matrices'],z['chain_ids'])
                rho,jack=summary['rho'],summary['jackknife_rhos']
                np.testing.assert_allclose(rho,z[name+'__rho'],atol=1e-12,rtol=0)
                extra,blocks=_singlet_channel_scalars(rho,cut)
                deleted=[_singlet_channel_scalars(a,cut)[0] for a in jack]
                keys=[k for k in extra if all(k in row for row in deleted)]
                errors=_jackknife_errors([{k:row[k] for k in keys} for row in deleted])
                records.append(dict(method='ViT',step=step,label=f'ViT_{step}',cut=name,
                    energy_per_site=audit['energy']['energy_per_site'],energy_SE=audit['energy']['chain_mean_standard_error'],
                    variance_per_site=audit['energy']['variance_per_site'],variance_SE=audit['energy']['variance_chain_standard_error'],
                    scalars=dict(cached['scalars'],**extra),whole_chain_SE=dict(cached['scalar_whole_chain_SE'],**errors),
                    symmetry_defect=cached['protocol']['multiplicity']['symmetry_defect_trace_distance'],
                    strict_entanglement_spectrum=False,independent_chains=128))
                arrays.update({f'ViT_{step}__{name}__'+k:v for k,v in dict(rho=rho,**blocks).items()})
    archive=OUTPUT/'inputs/dmrg/chi8192_sweep30'
    registration=json.loads((archive/'registration.json').read_text());checked(archive/'registration.json')
    if any(registration['config'][k]!=v for k,v in dict(rows=6,columns=6,j2=.125).items()):
        raise ValueError('late MPS physical problem changed')
    references=[]
    for sweep in (29,30):
        path=archive/f'sweep_readouts/sweep{sweep:04d}.json';row=json.loads(path.read_text());checked(path)
        raw=path.with_suffix('.npz');checked(raw,row['rdms_sha256'])
        with np.load(raw) as z:
            for name in ('compact6','winding6'):
                if registration['config']['regions'][name]!=cuts[name]['sites']:
                    raise ValueError('late MPS ordered region changed')
                references.append(dict(label=f'DMRG_8192_{sweep}',cut=name,sweep=sweep,
                    energy_per_site=row['energy_per_site'],rho=z['chi8192_'+name]))
    for sweep,folder in ((28,'chi8192_sweep28_hexagon6_env_v3'),(30,'chi8192_sweep30_hexagon6_env_v1')):
        path=OUTPUT/'inputs/dmrg'/folder;row=json.loads((path/'results.json').read_text());checked(path/'results.json')
        raw=path/'regions/local_hexagon6/rdm.npz';checked(raw,row['regions'][0]['rdm_sha256'])
        with np.load(raw) as z:
            if list(z['sites'])!=REGION:raise ValueError('late hexagon site order changed')
            references.append(dict(label=f'DMRG_8192_{sweep}',cut='local_hexagon6',sweep=sweep,
                energy_per_site=(-.515084738960105 if sweep==28 else -.5150870027306547),rho=z['rho']))
    for source in references:
        rho=source.pop('rho');report,data,scalars=_regional_scalar_data(rho,cuts[source['cut']])
        extra,blocks=_singlet_channel_scalars(rho,cuts[source['cut']]);scalars.update(extra)
        records.append(dict(**source,method='DMRG',chi=8192,scalars=scalars,whole_chain_SE=None,
            variance_per_site=None,variance_measured=False,
            symmetry_defect=report['multiplicity']['symmetry_defect_trace_distance'],strict_entanglement_spectrum=False))
        arrays.update({source['label']+'__'+source['cut']+'__'+k:v for k,v in dict(rho=rho,**blocks).items()})
    fields=('p_q0','p_q2','spin_squared','rows_p33_singlet','rows_p33_high',
            'rows_doublet_conditional_operator_purity','rows_cross_q1_q3_coherence_norm','connected_dimer_0_1_3_4')
    comparisons={}
    for name in cuts:
        vit=sorted((r for r in records if r['method']=='ViT' and r['cut']==name),key=lambda r:r['step'])
        ref=sorted((r for r in records if r['method']=='DMRG' and r['cut']==name),key=lambda r:r['sweep'])
        out={}
        for field in fields:
            if any(field not in r['scalars'] for r in (*vit[-2:],*ref)):continue
            latest,prior=vit[-1],vit[-2];new,old=ref[-1],ref[-2]
            delta=latest['scalars'][field]-new['scalars'][field]
            own=latest['scalars'][field]-prior['scalars'][field]
            drift=abs(new['scalars'][field]-old['scalars'][field]);se=latest['whole_chain_SE'][field]
            drift_SE=float(np.hypot(se,prior['whole_chain_SE'][field]))
            out[field]=dict(ViT_minus_DMRG=delta,ViT_SE=se,ViT_quality_drift=own,ViT_drift_SE=drift_SE,
                DMRG_quality_drift=drift,DMRG_quality_sweeps=[old['sweep'],new['sweep']],
                exceeds_quality_and_two_SE=bool(abs(delta)>abs(own)+drift+2*(se+drift_SE)))
        comparisons[name]=out
    result=dict(problem=dict(geometry='triangular',rows=6,columns=6,j2=.125,boundary='PBC_both_physical'),
        cuts=cuts,records=records,comparisons=comparisons,inputs_sha256=inputs,
        new_samples=0,new_amplitude_queries=0,new_H_calls=0,new_training=0,
        physical_phase_claim=False,strict_entanglement_spectrum=False,seconds=perf_counter()-started,
        warning='Independent endpoint errors; finite-chi drift is not a certified error bound. Conditional operator purity/coherence norms retain sampling bias. Methods are not distinct candidate phases.')
    destination.mkdir(exist_ok=True);np.savez_compressed(destination/'readouts.npz',**arrays)
    atomic_json(destination/'results.json',result)
    atomic_json(destination/'completion.json',{name:_sha(destination/name) for name in ('results.json','readouts.npz')})
    return result


def run(endpoint: int = 8600) -> dict:
    """Analyze the frozen local-hexagon channel and preserve source provenance."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if endpoint not in (8600, 9600):
        raise ValueError("only the exploratory endpoint and frozen validation endpoint are admitted")
    destination = OUTPUT / ("singlet_ring_analysis_v3" if endpoint == 8600 else "prospective_9600_validation_v1")
    if destination.exists():
        raise FileExistsError("fresh immutable analysis output required")
    destination.mkdir()
    inputs: dict[str, str] = {}

    def checked(path: Path, expected: str | None = None) -> None:
        digest = _sha(path)
        if expected is not None and digest != expected:
            raise ValueError("input hash mismatch: "+str(path))
        inputs[str(path.relative_to(ROOT))] = digest

    split = joint_spin_operator_split(6, [0, 1, 2], [1, 1, 0])
    operator, ring_diagnostics = singlet_ring_operator()
    quartet = joint_spin_operator_split(6, [0, 1, 2], [3, 3, 0]).full
    columns = _three_plus_three_singlet_columns((0, 1, 2))
    observables = {"p11_singlet": split.full, "p33_singlet": quartet,
                   "joint_spin_high_O": split.high}
    observables.update({f"doublet_a{a}_b{b}": np.outer(columns[:, 2*a+b], columns[:, 2*a+b].conj())
                        for a in range(2) for b in range(2)})
    current = OUTPUT / f"inputs/tri_{endpoint}/regional"
    saved = json.loads((current/"results.json").read_text())
    h_local = np.zeros((64, 64), complex)
    local_bonds = []
    for i, j, weight in saved["bonds"]:
        if i in REGION and j in REGION:
            a, b = REGION.index(i), REGION.index(j)
            h_local += weight*spin_dot_matrix(6, a, b)
            local_bonds.append([i, j, weight])
    observables["internal_H_patch"] = h_local
    # This is the energy of Hamiltonian bonds fully inside the region, not E/N.
    space = invariant_constraint_space(6)
    op_reduced = np.zeros((sum(space.dimensions),)*2, complex)
    for sl, us in zip(space.slices, space.magnetic_columns):
        op_reduced[sl, sl] = us[0].conj().T @ operator @ us[0]

    protocol = dict(region=REGION, left_local=[0,1,2], right_local=[3,4,5],
        operator="R=P0[(C6+C6_dagger)/2-Tr(P0(C6+C6_dagger)/2)/5]P0; P0 is the complete five-dimensional regional singlet projector",
        operator_identity="C6 cyclically shifts the ordered physical hexagon; it interchanges the alternating three-dimer coverings. The same fixed geometric operator is measured on every state.",
        operator_norm=float(np.linalg.norm(operator, 2)), normalization="unscaled Hermitian cyclic exchange restricted to total S=0; scalar singlet average removed",
        joint_spin_calibration="O=P11 minus I/pair/symmetrized-two-bond span; excludes p(q), is entirely a function of joint Casimirs, and does not resolve internal multiplicity. Not the response source R.",
        maxent_constraints="fixed p(q), all pair scalars, all symmetrized two-bond products; separate nonlinear comparator",
        filter="rho(eta)=exp(eta O/2) rho exp(eta O/2)/Tr; d<O>/deta=Var(O)",
        source_units="dimensionless eta; H_patch in J1 units", local_bonds=local_bonds,
        Hamiltonian_static_susceptibility=False, global_A1_filter=False,
        selection="exploratory choice after 7600/8600 quality readouts; no untouched discovery claim",
        prospective_endpoint=9600,
        prospective_screen="freeze O and geometry; report mean, Var(O), H_patch and p11/p33 cross-responses on 9600. Compare unpaired 2SE and previous quality drift; no state/phase separation without DMRG hexagon quality pair.",
        exact_diagnostics=ring_diagnostics)
    (destination/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    np.savez_compressed(destination/"operators.npz", O=operator, **observables)
    rows, arrays = [], {}

    def measure(label: str, rho: np.ndarray, *, jack: np.ndarray | None = None,
                method: str, energy: float | None = None, **metadata) -> None:
        point = filter_response(rho, operator, observables)
        errors = _jackknife_errors([filter_response(r, operator, observables) for r in jack]) if jack is not None else None
        h = 1e-4
        plus, minus = [filter_response(rho, operator, observables, eta) for eta in (h, -h)]
        derivative_errors = {name:abs((plus[name]-minus[name])/(2*h)-point[name+"__filter_slope"])
                             for name in observables}
        derivative_errors["source"] = abs((plus["source_mean"]-minus["source_mean"])/(2*h)-point["source_variance"])
        eta_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        response = [dict(eta=x, **filter_response(rho, operator, observables, x)) for x in eta_values]
        # MaxEnt describes measured invariant data; its physical lift is only
        # a comparator and is NEVER substituted for the actual RDM above.
        maxent = None
        omega = space.data(rho)
        if np.linalg.eigvalsh(omega).min() > 1e-12:
            report, omega_me, _ = fit_invariant_maximum_entropy(omega, space)
            if not report["certified_moment_fit"]:
                raise RuntimeError("MaxEnt moment closure failed")
            rho_me = np.zeros_like(rho, complex)
            for q, sl, us in zip(space.qs, space.slices, space.magnetic_columns):
                rho_me += sum(u @ omega_me[sl,sl] @ u.conj().T for u in us)/(q+1)
            maxent = dict(fit=report, predicted=filter_response(rho_me,operator,observables),
                          excess_mean=point["source_mean"]-float(np.trace(op_reduced@omega_me).real),
                          plug_in_no_fit_uncertainty=True)
        row = dict(label=label,method=method,energy_per_site=energy,point=point,SE=errors,
                   response=response,finite_difference_max_error=max(derivative_errors.values()),
                   maxent=maxent,**metadata)
        rows.append(row)
        arrays[label+"__rho"] = rho
        if jack is not None:
            arrays[label+"__jackknife_rhos"] = jack

    old_spec_path = ROOT/"projects/clebsch_tree_nqs/configs/validation/spin_su2_unified_quality.json"
    checked(old_spec_path)
    old_spec = json.loads(old_spec_path.read_text())
    for candidate in old_spec["candidates"]:
        for rel in candidate["region_caches"]:
            directory = ROOT/rel
            result = json.loads((directory/"results.json").read_text())
            if "local_hexagon6" not in result["regions"]:
                continue
            checked(directory/"results.json")
            checked(directory/"physics.npz",result["raw_sha256"])
            energy_path = ROOT/candidate["energy_audit"]
            checked(energy_path)
            energy = json.loads(energy_path.read_text())["energy"]["energy_per_site"]
            with np.load(directory/"physics.npz") as data:
                summary=regional_density_summary(data["local_hexagon6__per_record_matrices"],data["chain_ids"])
                measure("ViT_"+str(candidate["step"]),summary["rho"],jack=summary["jackknife_rhos"],
                        method="ViT",energy=energy,step=candidate["step"],independent_chains=128)
    checked(current/"results.json")
    checked(current/"physics.npz",saved["raw_sha256"])
    with np.load(current/"physics.npz") as data:
        summary=regional_density_summary(data["local_hexagon6__per_record_matrices"],data["chain_ids"])
        energy_path = current.parent/"energy.json"
        registration = json.loads((current/"registration.json").read_text())
        energy_candidates = [digest for path,digest in registration["inputs_sha256"].items()
                             if path.endswith(f"audits/step{endpoint:05d}/energy.json")]
        if len(energy_candidates) != 1:
            raise ValueError("regional registration must identify exactly one parent energy audit")
        checked(energy_path,energy_candidates[0])
        energy=json.loads(energy_path.read_text())["energy"]["energy_per_site"]
        measure(f"ViT_{endpoint}",summary["rho"],jack=summary["jackknife_rhos"],method="ViT",step=endpoint,
                energy=energy, independent_chains=128,
                energy_source=str(energy_path.relative_to(ROOT)))
        # A few high leverage chain contributions can defeat nominal SE.
        ids=data["chain_ids"]
        per_record=np.einsum('nij,ji->n',data['local_hexagon6__per_record_matrices'],operator).real
        traces=np.trace(data['local_hexagon6__per_record_matrices'],axis1=1,axis2=2).real
        chain=np.array([per_record[ids==i].mean() for i in np.unique(ids)])
        chain_norm=np.array([traces[ids==i].mean() for i in np.unique(ids)])
        influence=(chain-rows[-1]['point']['source_mean']*chain_norm)/chain_norm.mean()
        rows[-1]["tail_check"] = dict(chain_raw_min=float(chain.min()),chain_raw_max=float(chain.max()),
            top_four_chains_fraction_absolute_influence=float(np.sort(abs(influence))[-4:].sum()/abs(influence).sum()),
            top_four_chains_fraction_squared_influence=float(np.sort(influence**2)[-4:].sum()/np.sum(influence**2)),
            half_stream_normalized_means=[float(chain[:64].sum()/chain_norm[:64].sum()),float(chain[64:].sum()/chain_norm[64:].sum())],
            mixing_certified=False)
    dmrg_path=OUTPUT/"inputs/dmrg/chi8192_sweep28_hexagon6_env_v3/regions/local_hexagon6/rdm.npz"
    dmrg_result=OUTPUT/"inputs/dmrg/chi8192_sweep28_hexagon6_env_v3/results.json"
    checked(dmrg_result)
    result=json.loads(dmrg_result.read_text())
    checked(dmrg_path,result["regions"][0]["rdm_sha256"])
    with np.load(dmrg_path) as data:
        if list(data["sites"])!=REGION:
            raise ValueError("different DMRG physical sites/order")
        measure("DMRG_8192_28",data["rho"],method="DMRG",energy=-0.515084738960105,
                chi=8192,sweep=28,finite_chi_error_unknown=True,hexagon_quality_pair_available=False)
    controls=ROOT/"projects/clebsch_tree_nqs/results/spin_su2_unified_quality_v1/controls"
    complete=json.loads((controls/"completion.json").read_text())
    checked(controls/"readouts.npz",complete["readouts_sha256"])
    with np.load(controls/"readouts.npz") as data:
        for label in ("horizontal_dimers_shift0","horizontal_dimers_shift1","classical_K_plus","classical_M1","classical_M2","classical_M3"):
            measure(label,data["local_hexagon6__"+label+"__rho"],method="synthetic_control",
                    optimized_quantum_ordered_state=False)

    latest,previous,dmrg=rows[3],rows[2],rows[4]
    comparisons={}
    for key in ("source_mean","source_variance","internal_H_patch__filter_slope",
                "p11_singlet__filter_slope","p33_singlet__filter_slope"):
        a,b,c=latest["point"][key],previous["point"][key],dmrg["point"][key]
        se=latest["SE"][key];dse=float(np.hypot(se,previous["SE"][key]))
        comparisons[key]=dict(latest=a,previous=b,dmrg=c,latest_SE=se,
            within_vit_drift=a-b,within_vit_drift_SE=dse,unpaired_drift_z=(a-b)/dse,
            cross_family_difference=a-c,cross_family_statistical_z=(a-c)/se,
            cross_family_claim_allowed=False,dmrg_quality_pair_missing=True)
    result=dict(protocol=protocol,records=rows,comparisons=comparisons,inputs_sha256=inputs,
        new_samples=0,new_training=0,new_H_queries=0,new_ED=0,new_MPS_contractions=0,
        interpretation="One geometric singlet-ring resonance observable with controlled equal-time filter responses. Not a Hamiltonian susceptibility, phase/QSL discriminator or evidence that two candidate phases differ.")
    if endpoint == 9600:
        preregistered=OUTPUT/'prospective_9600_protocol.json'
        expected='56938f695a8ccdc4059b903215799ad5f4089d4d69c2f643b3a8c24c684ec029'
        checked(preregistered,expected)
        previous_result=OUTPUT/'singlet_ring_analysis_v3/results.json'
        checked(previous_result)
        baseline=json.loads(previous_result.read_text())
        old=next(r for r in baseline['records'] if r['label']=='ViT_8600')
        earlier=next(r for r in baseline['records'] if r['label']=='ViT_7600')
        new=next(r for r in rows if r['label']=='ViT_9600')
        screens={}
        for field in ('source_mean','source_variance','internal_H_patch__filter_slope'):
            delta=new['point'][field]-old['point'][field]
            bound=2*np.hypot(new['SE'][field],old['SE'][field])+abs(old['point'][field]-earlier['point'][field])
            screens[field]=dict(change=float(delta),bound=float(bound),passed=bool(abs(delta)<=bound))
        field='internal_H_patch__filter_slope'
        result['prospective_validation']=dict(protocol_sha256=expected,primary=screens,
            all_primary_stable=all(v['passed'] for v in screens.values()),
            positive_patch_energy_response=bool(new['point'][field]>2*new['SE'][field]),
            response_z=float(new['point'][field]/new['SE'][field]),
            selection_using_endpoint_9600=False)
    result['workflow_sha256']=_sha(Path(__file__))
    (destination/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(destination/"readouts.npz",**arrays)
    (destination/"completion.json").write_text(json.dumps({name:_sha(destination/name)
        for name in ("results.json","protocol.json","operators.npz","readouts.npz")},indent=2)+"\n")
    return result


def render_response_quality(endpoint: int = 9600) -> dict[str, str]:
    """Render only sealed local readouts; no result/registration is rewritten."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def read(folder: str) -> dict:
        directory=OUTPUT/folder
        seals=json.loads((directory/'completion.json').read_text())
        if _sha(directory/'results.json')!=seals['results.json']:
            raise ValueError('sealed plotting input changed')
        return json.loads((directory/'results.json').read_text())

    if endpoint not in (8600,9600):
        raise ValueError('only existing sealed 8600/9600 analyses may be plotted')
    baseline=read('singlet_ring_analysis_v3')
    vit=[r for r in baseline['records'] if r['method']=='ViT']
    if endpoint==9600:
        current=read('prospective_9600_validation_v1')
        vit.append(next(r for r in current['records'] if r['label']=='ViT_9600'))
        destination=OUTPUT/'prospective_9600_validation_v1'
        stem='ring_response_quality_9600'
    else:
        current=baseline
        destination=OUTPUT/'singlet_ring_analysis_v3'
        stem='ring_response_quality'
    dmrg=next(r for r in current['records'] if r['method']=='DMRG')
    fields=[('source_mean',r'$\langle R\rangle$'),
            ('source_variance',r'${\rm Var}(R)$'),
            ('internal_H_patch__filter_slope',r'$d\langle H_A\rangle/d\eta$')]
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,(field,label) in zip(axes.flat,fields):
        ax.errorbar([r['step'] for r in vit],[r['point'][field] for r in vit],
            yerr=[r['SE'][field] for r in vit],fmt='o-',capsize=3,label='ViT, whole-chain 1 SE')
        ax.axhline(dmrg['point'][field],ls='--',c='C1',
            label=r'DMRG $\chi=8192$ (finite-$\chi$ error unknown)')
        if endpoint==9600:
            ax.axvline(9100,lw=.7,ls=':',c='grey')
        if field=='source_mean':
            ax.plot([r['step'] for r in vit],[r['maxent']['predicted'][field] for r in vit],
                ls=':',marker='s',c='C2',label='MaxEnt: p(q), pairs, sym. two-bond (plug-in)')
        ax.set(xlabel='Frozen training step'+(' (9600: prospective test)' if endpoint==9600 else ''),
               ylabel=label)
        ax.grid(alpha=.2)
    ax=axes[1,1]
    for row in ([vit[-2],vit[-1],dmrg] if endpoint==9600 else [vit[-1],dmrg]):
        ax.plot([x['eta'] for x in row['response']],
            [x['internal_H_patch']-row['point']['internal_H_patch'] for x in row['response']],
            marker='o',label=row['label'])
    ax.axhline(0,c='grey',lw=.5);ax.axvline(0,c='grey',lw=.5)
    ax.set(xlabel=r'Local filter source $\eta$',ylabel=r'$\langle H_A\rangle_\eta-\langle H_A\rangle_0$')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(('Frozen hexagon singlet-ring channel: prospective 9600 quality validation'
        if endpoint==9600 else 'Contractible hexagon singlet-ring resonance: quality sequence and local filter response')
        +'\nEqual-time response; not Hamiltonian susceptibility or a phase distinction',fontsize=12)
    outputs={suffix:str(destination/f'{stem}.{suffix}') for suffix in ('png','pdf')}
    for suffix,path in outputs.items():
        fig.savefig(path,dpi=180 if suffix=='png' else None)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint',type=int,default=8600,choices=(8600,9600,10600))
    parser.add_argument('--full-h-response',action='store_true')
    parser.add_argument('--complement-response',action='store_true')
    parser.add_argument('--render-response',action='store_true')
    parser.add_argument('--advance-quality',action='store_true')
    parser.add_argument('--render-late-quality',action='store_true')
    parser.add_argument('--regional-late-quality',action='store_true')
    parser.add_argument('--magnetic-late-quality',action='store_true')
    parser.add_argument('--quality-manifest',type=Path)
    args=parser.parse_args()
    result=(registered_quality_series(args.quality_manifest) if args.quality_manifest else magnetic_late_quality() if args.magnetic_late_quality else regional_late_quality() if args.regional_late_quality else render_late_quality() if args.render_late_quality else advance_quality() if args.advance_quality else render_response_quality(args.endpoint) if args.render_response else
            complement_response_from_cache(args.endpoint) if args.complement_response else
            full_h_response_from_cache(args.endpoint) if args.full_h_response else run(args.endpoint))
    print(json.dumps(dict(records=len(result['records']),seconds=result['seconds']) if args.quality_manifest else result if args.full_h_response or args.complement_response or args.render_response or args.render_late_quality or args.magnetic_late_quality else result['comparisons'],indent=2))
