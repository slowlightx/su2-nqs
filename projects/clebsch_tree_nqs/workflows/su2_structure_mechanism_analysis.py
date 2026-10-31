"""Resolve a frozen five-spin GME witness into local SU(2) charge processes.

This offline analysis reuses saved physical RDMs and a previously frozen witness.
It does not optimize a state, repair an RDM, or identify a QSL phase.
All q labels are doubled angular momenta. Site zero is the most significant bit.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.multipartite_geometry import partial_trace_sites
from su2_nqs.benchmarks.regional_su2_readout import regional_schur_columns
from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from .multipartite_geometry_analysis import ROOT, load_quality


BASE = ROOT / 'projects/clebsch_tree_nqs/results/multipartite_geometry_20261013_v2'
OUTPUT = ROOT / 'projects/clebsch_tree_nqs/results/su2_structure_mechanism_20261013'
CASES = {
    'upper': dict(keep=(0, 1, 2, 4, 5), left=(0, 3), center=1, right=(2, 4)),
    'lower': dict(keep=(0, 1, 3, 4, 5), left=(0, 2), center=3, right=(1, 4)),
}


def charge_process_operators(witness: np.ndarray, left: tuple[int, int],
                             right: tuple[int, int]) -> dict[str, np.ndarray]:
    """Decompose score=-W into diagonal, transfer, pair and single-q changes.

    Four blocks are (q_left,q_right)=(0,0),(0,2),(2,0),(2,2). Offdiagonal
    components alone are observables, not individually valid GME witnesses.
    """
    w = np.asarray(witness, np.complex128)
    if w.shape != (32, 32) or np.linalg.norm(w-w.conj().T) > 1e-10:
        raise ValueError('Hermitian five-spin witness required')
    eye = np.eye(32)
    a = eye/4-spin_dot_matrix(5, *left)
    b = eye/4-spin_dot_matrix(5, *right)
    p = np.stack([a@b, a@(eye-b), (eye-a)@b, (eye-a)@(eye-b)])
    score = -w
    diagonal = sum(x@score@x for x in p)
    transfer = p[1]@score@p[2]+p[2]@score@p[1]
    pair = p[0]@score@p[3]+p[3]@score@p[0]
    single = score-diagonal-transfer-pair
    return dict(score_total=score, score_diagonal=diagonal, score_transfer=transfer,
                score_pair_creation=pair, score_single_change=single,
                score_rest=score-transfer, charge_projectors=p)


def validate_operators(operators: dict[str, np.ndarray],
                       left: tuple[int, int], right: tuple[int, int]) -> dict:
    """Independent bitwise swaps, generators, Hermiticity and closure checks."""
    bits = (np.arange(32)[:, None] >> np.arange(4, -1, -1)) & 1
    z = np.diag(bits.sum(1)-2.5)
    plus = np.zeros((32, 32))
    for i in range(5):
        old = np.flatnonzero(bits[:, i] == 0)
        plus[old ^ (1 << (4-i)), old] += 1
    swap_singlets = []
    for a, b in (left, right):
        perm = np.arange(32) ^ np.where(bits[:, a] != bits[:, b],
                                      (1 << (4-a)) | (1 << (4-b)), 0)
        swap = np.eye(32)[:, perm]
        swap_singlets.append((np.eye(32)-swap)/2)
    a, b = swap_singlets; eye = np.eye(32)
    independent = np.stack([a@b, a@(eye-b), (eye-a)@b, (eye-a)@(eye-b)])
    report = dict(projector_swap_error=float(np.max(abs(independent-operators['charge_projectors']))))
    names = [k for k in operators if k.startswith('score_')]
    report.update(hermitian_max=max(float(np.max(abs(operators[k]-operators[k].conj().T))) for k in names),
        total_Sz_commutator_max=max(float(np.max(abs(z@operators[k]-operators[k]@z))) for k in names),
        total_Splus_commutator_max=max(float(np.max(abs(plus@operators[k]-operators[k]@plus))) for k in names),
        sum_closure_max=float(np.max(abs(sum(operators[k] for k in (
            'score_diagonal', 'score_transfer', 'score_pair_creation', 'score_single_change'))-operators['score_total']))))
    if max(report.values()) > 2e-10:
        raise RuntimeError('independent operator validation failed: '+str(report))
    report.update(q_labels=[[0,0],[0,2],[2,0],[2,2]], strict_local_Sz_conserving=True,
                  fixed_q_components_are_not_individual_GME_witnesses=True)
    return report


def multiplicity_state(rho: np.ndarray, left: tuple[int, int, int]) -> np.ndarray:
    """Local q=1 filtering and magnetic trace, with no nonlocal singlet filter.

    Output is an unnormalized 2x2-multiplicity-qubit density matrix. Its trace
    is the probability of the local q_left=q_right=1 outcomes.
    """
    u, labels = regional_schur_columns(3)
    magnetic = [u[:, [i for i, x in enumerate(labels) if x[0] == 1 and x[1] == m]]
                for m in (-1, 1)]
    right = tuple(i for i in range(6) if i not in left)
    bits = (np.arange(64)[:, None] >> np.arange(5, -1, -1)) & 1
    order = bits[:, left+right] @ (2**np.arange(5, -1, -1))
    result = np.zeros((4, 4), complex)
    for a in magnetic:
        for b in magnetic:
            v = np.kron(a, b)[order]
            result += v.conj().T@rho@v
    return result


def center_resolve(operator: np.ndarray, center: int) -> tuple[np.ndarray, np.ndarray]:
    """Hilbert-Schmidt split into I_center and traceless center-spin parts.

    Tr_center(O)/2 tensor I is the exact spectator contribution. The remainder
    contains center-spin correlations. Neither piece alone is a GME witness.
    """
    if operator.shape != (32, 32) or center not in range(5):
        raise ValueError('five-spin operator and site in 0..4 required')
    bits = (np.arange(32)[:, None] >> np.arange(4, -1, -1)) & 1
    keep = tuple(i for i in range(5) if i != center)
    index = bits[:, keep] @ (2**np.arange(3, -1, -1))
    small = partial_trace_sites(operator, keep)/2
    spectator = small[index[:, None], index[None, :]]*(bits[:, center, None] == bits[None, :, center])
    return spectator, operator-spectator


def center_analysis() -> None:
    """Append an independently sealed physical split without changing first results."""
    target = OUTPUT/'center_analysis.json'
    if target.exists():
        raise FileExistsError('completed center decomposition is immutable')
    op_path = OUTPUT/'frozen_operators.npz'
    with np.load(op_path) as arrays:
        bank = {k:np.array(arrays[k]) for k in arrays.files}
    cfg = json.loads((BASE/'registration.json').read_text())['config']
    records, inputs, _ = load_quality(cfg)
    saved = {}; rows = []; controls = []; checks = {}
    for name, case in CASES.items():
        ops = {}
        for process in ('transfer', 'pair_creation', 'single_change'):
            original = bank[name+'__score_'+process]
            spectator, mediated = center_resolve(original, case['center'])
            ops[process+'__spectator'] = spectator
            ops[process+'__center_correlated'] = mediated
            checks[name+'__'+process] = dict(
                sum_closure_max=float(np.max(abs(spectator+mediated-original))),
                center_trace_remainder_max=float(np.max(abs(partial_trace_sites(mediated,
                    tuple(i for i in range(5) if i != case['center']))))),
                HS_orthogonality=float(abs(np.vdot(spectator, mediated))),
                center_correlated_fraction_Frobenius=float(np.linalg.norm(mediated)/np.linalg.norm(original)))
        saved.update({name+'__'+k:v for k,v in ops.items()})
        for row in records:
            if row['parent'] != 'compact6':
                continue
            rho = partial_trace_sites(row['rho'], case['keep'])
            values = {k:float(np.trace(rho@v).real) for k,v in ops.items()}
            se = {}
            if row['chains'] is not None:
                chains = partial_trace_sites(row['chains'], case['keep'])
                for k,v in ops.items():
                    x = np.einsum('bij,ji->b', chains, v).real
                    se[k] = float(x.std(ddof=1)/np.sqrt(len(x)))
            rows.append(dict(case=name, **{k:row[k] for k in ('label','method','j2','step','chi')},
                             values=values, whole_chain_SE=se))
        with np.load(BASE/'triangle_pair5_local_probe_v1/arrays.npz') as cs:
            prefix = 'triangle_pair5_'+name+'__control__'
            for key in cs.files:
                if key.startswith(prefix) and key.endswith('__rho'):
                    controls.append(dict(case=name, label=key[len(prefix):-5],
                        values={k:float(np.trace(cs[key]@v).real) for k,v in ops.items()}, QSL=False))
    for report in checks.values():
        if max(report[k] for k in ('sum_closure_max','center_trace_remainder_max','HS_orthogonality')) > 1e-10:
            raise RuntimeError('center split check failed')
    np.savez_compressed(OUTPUT/'center_resolved_operators.npz', **saved)
    atomic_json(target, dict(records=rows, controls=controls, checks=checks,
        input_sha256=inputs, frozen_operators_sha256=file_sha256(op_path),
        center_operators_sha256=file_sha256(OUTPUT/'center_resolved_operators.npz'),
        source_code_sha256=file_sha256(Path(__file__)),
        interpretation='Algebraic operator decomposition, not a causal dynamical intervention. Center-correlated means traceless on the center site; changing the center state was not simulated.',
        new_samples=0, new_H_calls=0, new_training=0))
    print(json.dumps(dict(center_records=len(rows), controls=len(controls))), flush=True)


def dimer_pauli_expectations(pairs: tuple[tuple[int, int], ...],
                             sites: tuple[int, ...], words: np.ndarray) -> np.ndarray:
    """Exact Pauli-string means in a product of physical singlet dimers.

    Labels 0,1,2,3 denote I,2Sx,2Sy,2Sz in down/up order. A singlet gives
    <sigma_a sigma_b>=-delta_ab; an unpaired nonidentity factor gives zero.
    Sites outside the explicit observable support carry identity.
    """
    labels = np.asarray(words)
    if (labels.ndim != 2 or labels.shape[1] != len(sites) or
            not np.issubdtype(labels.dtype, np.integer) or
            np.any((labels < 0) | (labels > 3)) or len(set(sites)) != len(sites)):
        raise ValueError('distinct sites and integer Pauli words required')
    flat = [s for pair in pairs for s in pair]
    if len(flat) != len(set(flat)) or not set(sites).issubset(flat):
        raise ValueError('disjoint covering must contain all observable sites')
    index = {s:i for i,s in enumerate(sites)}
    zero = np.zeros(len(labels), int); result = np.ones(len(labels))
    for a,b in pairs:
        left = labels[:, index[a]] if a in index else zero
        right = labels[:, index[b]] if b in index else zero
        result *= np.where(left == right, np.where(left == 0, 1., -1.), 0.)
    return result


def dimer_operator_moments(pairs: tuple[tuple[int, int], ...],
                           sites_a: tuple[int, ...], sites_b: tuple[int, ...],
                           labels: np.ndarray, coefficients: np.ndarray) -> dict:
    """Exact same-operator moments on two disjoint regions of a dimer product."""
    if set(sites_a) & set(sites_b):
        raise ValueError('this factorized product routine requires disjoint regions')
    n = len(coefficients)
    a = coefficients @ dimer_pauli_expectations(pairs, sites_a, labels)
    b = coefficients @ dimer_pauli_expectations(pairs, sites_b, labels)
    joint_words = np.concatenate([np.repeat(labels, n, axis=0), np.tile(labels, (n,1))], axis=1)
    joint_coefficients = np.outer(coefficients, coefficients).reshape(-1)
    joint = joint_coefficients @ dimer_pauli_expectations(pairs, sites_a+sites_b, joint_words)
    if max(abs(complex(x).imag) for x in (a,b,joint)) > 1e-12:
        raise RuntimeError('Hermitian dimer moments developed an imaginary part')
    a,b,joint = (float(np.real(x)) for x in (a,b,joint))
    return dict(mean_A=a, mean_B=b, joint=joint, connected=joint-a*b)


def dimer_distance_analysis() -> None:
    """N36 fixed-cover and full-space-group mixed-cover distance controls."""
    from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
    target = OUTPUT/'dimer_distance_controls.json'
    if target.exists():
        raise FileExistsError('completed dimer distance controls are immutable')
    sites_a = (12,13,14,19,20)
    shifts = ((2,0),(0,3),(3,0),(3,3))
    parent = tuple((r*6+c,r*6+c+1) for r in range(6) for c in (0,2,4))
    group = triangular_hnf_gamma_a1_group(((6,0),(0,6)))
    covers = Counter(tuple(sorted(tuple(sorted((p[a],p[b]))) for a,b in parent)) for p in group)
    path = OUTPUT/'pauli_decomposition.npz'
    rows = []; diagnostics = {}; detail = []
    with np.load(path) as arrays:
        all_labels = np.array(arrays['labels'])
        for name,key in [('center_transfer','upper__transfer_center'),
                         ('transfer','upper__transfer'),('single_change','upper__single')]:
            all_coefficients = np.array(arrays[key]); mask = abs(all_coefficients)>1e-14
            labels = all_labels[mask]; coefficients = all_coefficients[mask]
            residual = float(np.sum(abs(all_coefficients[~mask])))
            norm_bound = float(np.sum(abs(all_coefficients)))
            diagnostics[name] = dict(retained_terms=int(mask.sum()),
                discarded_coefficient_L1=residual, mean_absolute_error_bound=residual,
                joint_absolute_error_bound=2*norm_bound*residual+residual**2)
            for dr,dc in shifts:
                sites_b = tuple(((s//6+dr)%6)*6+(s%6+dc)%6 for s in sites_a)
                fixed = dimer_operator_moments(parent, sites_a, sites_b, labels, coefficients)
                moments = []; probabilities = []
                for cover,count in covers.items():
                    report = dimer_operator_moments(cover, sites_a, sites_b, labels, coefficients)
                    probabilities.append(count/len(group)); moments.append(report)
                    detail.append(dict(operator=name, shift=[dr,dc], covering=cover,
                                       group_multiplicity=count, **report))
                probability = np.array(probabilities)
                mean_a,mean_b,joint,within = [float(probability @ [r[k] for r in moments])
                    for k in ('mean_A','mean_B','joint','connected')]
                classical = float(probability @ [r['mean_A']*r['mean_B'] for r in moments]-mean_a*mean_b)
                connected = joint-mean_a*mean_b
                if abs(connected-within-classical) > 1e-13:
                    raise RuntimeError('mixture covariance decomposition failed')
                rows.append(dict(operator=name,shift=[dr,dc],sites_A=sites_a,sites_B=sites_b,
                    fixed_covering=fixed,space_group_mixture=dict(mean_A=mean_a,mean_B=mean_b,
                        joint=joint,connected=connected,mean_within_cover_connected=within,
                        classical_covering_covariance=classical)))
    result = dict(records=rows, per_covering=detail, operator_diagnostics=diagnostics,
        n_sites=36, parent_covering=parent, group_size=len(group),
        unique_coverings=len(covers), group_multiplicities=list(covers.values()),
        operator_source_sha256=file_sha256(path), source_code_sha256=file_sha256(Path(__file__)),
        exact_with_floating_point_operator_coefficients=True,
        physical_state='Product singlet covering or incoherent full-space-group mixture, not coherent projected A1 superposition.',
        controls_energy_matched=False, QSL_claim=False, new_samples=0, new_training=0)
    atomic_json(target,result)
    print(json.dumps(dict(records=len(rows),unique_coverings=len(covers),group_size=len(group))),flush=True)


def main() -> None:
    """Export reusable operators first, then record exact cached-state analyses."""
    if (OUTPUT/'operator_validation.json').exists():
        raise FileExistsError('the frozen operator study is immutable; reuse its saved operators')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    witness_path = BASE/'triangle_pair5_reference_witness_v1/arrays.npz'
    source = np.load(witness_path)
    op_banks = {}; checks = {}; saved = {}
    for name, case in CASES.items():
        ops = charge_process_operators(source[f'triangle_pair5_{name}__DMRG_8192_30__W'],
                                       case['left'], case['right'])
        checks[name] = validate_operators(ops, case['left'], case['right'])
        op_banks[name] = ops
        saved.update({name+'__'+k: v for k, v in ops.items()})
    np.savez_compressed(OUTPUT/'frozen_operators.npz', **saved)
    atomic_json(OUTPUT/'operator_validation.json', dict(cases=CASES, checks=checks,
        source_witness_sha256=file_sha256(witness_path), source_witness_fit='DMRG_8192_sweep30',
        exported_operators_sha256=file_sha256(OUTPUT/'frozen_operators.npz'),
        state_twirl=False, source_code_sha256=file_sha256(Path(__file__)),
        reference_W_was_frozen_before_this_decomposition=True,
        interpretation='score_total=-W is the certified GME lower-bound score. Components are frozen scalar observables; their positivity alone does not certify entanglement.'))
    print('EXPORTED '+str(OUTPUT/'frozen_operators.npz'), flush=True)
    cfg = json.loads((BASE/'registration.json').read_text())['config']
    records, inputs, _ = load_quality(cfg)
    rows = []; controls = []; mult_rows = []
    for name, case in CASES.items():
        ops = op_banks[name]; names = [k for k in ops if k.startswith('score_')]
        for row in records:
            if row['parent'] != 'compact6':
                continue
            rho = partial_trace_sites(row['rho'], case['keep'])
            values = {k: float(np.trace(rho@ops[k]).real) for k in names}
            errors = {}
            if row['chains'] is not None:
                chains = partial_trace_sites(row['chains'], case['keep'])
                for k in names:
                    x = np.einsum('bij,ji->b', chains, ops[k]).real
                    errors[k] = float(x.std(ddof=1)/np.sqrt(len(x)))
            critical = (-values['score_diagonal']/(values['score_total']-values['score_diagonal'])
                        if values['score_total'] > 0 and values['score_diagonal'] < 0 else None)
            rows.append(dict(case=name, **{k:row[k] for k in ('label','method','j2','step','chi')},
                values=values, whole_chain_SE=errors,
                charge_probabilities=[float(np.trace(rho@p).real) for p in ops['charge_projectors']],
                surviving_coherence_fraction_for_positive_frozen_witness=critical,
                phase_claim=False))
        control_path = BASE/'triangle_pair5_local_probe_v1/arrays.npz'
        with np.load(control_path) as cs:
            prefix = 'triangle_pair5_'+name+'__control__'
            for k in cs.files:
                if k.startswith(prefix) and k.endswith('__rho'):
                    rho = cs[k]
                    controls.append(dict(case=name, label=k[len(prefix):-5],
                        values={x:float(np.trace(rho@ops[x]).real) for x in names}, QSL=False,
                        same_Hamiltonian_energy_matched=False))
    for row in records:
        if row['method'] != 'DMRG' or row['chi'] != 10240 or row['step'] != 32:
            continue
        for left in combinations(range(6), 3):
            if left[0] != 0:
                continue
            matrix = multiplicity_state(row['rho'], left)
            pt = matrix.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4)
            eig = np.linalg.eigvalsh(pt)
            mult_rows.append(dict(label=row['label'], region=row['parent'], left=left,
                probability=float(np.trace(matrix).real), min_partial_transpose_eigenvalue=float(eig[0]),
                weighted_negativity=float(-eig[eig<0].sum()),
                two_qubit_PPT_is_separable=bool(eig[0]>1e-10),
                caveat='Only the locally filtered multiplicity state is tested; the full physical RDM can remain entangled.'))
    atomic_json(OUTPUT/'results.json', dict(records=rows, controls=controls,
        local_multiplicity_states=mult_rows, input_sha256=inputs,
        checks=checks, new_samples=0, new_H_calls=0, new_training=0,
        controls_sha256=file_sha256(control_path),
        scientific_scope='Confirmed scalar singlet-triplet resonance mechanism. No QSL uniqueness, coupling contrast, or phase claim.'))
    print(json.dumps(dict(records=len(rows), controls=len(controls), multiplicity_partitions=len(mult_rows))),flush=True)


if __name__ == '__main__':
    dimer_distance_analysis() if '--dimer-distance' in sys.argv else center_analysis() if '--center' in sys.argv else main()
