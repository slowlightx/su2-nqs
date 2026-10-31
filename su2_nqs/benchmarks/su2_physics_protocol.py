"""Quality-aware SU(2) analysis of actual regional physical density matrices.

This is an offline measurement layer, not a new ansatz or fusion-path model.
The magnetic partial trace sigma_q is a state on the invariant observable
algebra for ANY rho. It is NOT a physical entanglement spectrum unless rho
itself is rotationally invariant. Labels q=2S,m=2M are doubled integers;
physical bits are down/up with site zero most significant. No input repair.
"""
from __future__ import annotations

from typing import Sequence
from numbers import Integral
from functools import lru_cache
from itertools import combinations

import numpy as np

from .regional_su2_readout import regional_schur_columns
from .tensor_network_readout import physical_spin_block_readout
from .tensor_network_observables import regional_physical_observables
from .spin_partition_readout import spin_partition_readout
from .spin_operator_information import joint_spin_operator_split, three_bond_readout, spin_dot_matrix
from .singlet_physics_readout import compact6_channel_readout, compact6_channel_operators


@lru_cache(maxsize=5)
def _rotational_audit_operators(n: int) -> tuple[dict, tuple[np.ndarray, ...]]:
    """Offline physical operators; magnetic bits down/up, site zero MSB."""
    single = (np.array([[0, 1], [1, 0]], complex)/2,
              np.array([[0, 1j], [-1j, 0]], complex)/2,
              np.diag([-.5, .5]).astype(complex))
    local = []
    for site in range(n):
        axes = []
        for spin in single:
            op = np.ones((1, 1), complex)
            for index in range(n):
                op = np.kron(op, spin if index == site else np.eye(2))
            axes.append(op)
        local.append(axes)
    totals = tuple(sum(site[axis] for site in local) for axis in range(3))
    pairs = list(combinations(range(n), 2))
    scalar = {f'pair_{i}_{j}': spin_dot_matrix(n, i, j) for i, j in pairs}
    zz = {f'pair_{i}_{j}': local[i][2]@local[j][2] for i, j in pairs}
    if n == 6:
        split = joint_spin_operator_split(6, (0, 1, 2), (3, 3, 0))
        scalar.update(row_partition_q3_full=split.full, row_partition_q3_high=split.high)
        bank = compact6_channel_operators()
        scalar['compact6_E2_dot'] = sum(bank[f'E2_triangle0_component{i}']@bank[f'E2_triangle1_component{i}'] for i in range(2))
        scalar['compact6_chirality_product'] = bank['chi_triangle0']@bank['chi_triangle1']
    for bank in (scalar, zz):
        for matrix in bank.values():
            matrix.setflags(write=False)
    for matrix in totals:
        matrix.setflags(write=False)
    return dict(scalar=scalar, zz=zz), totals


def _rotational_comparator(rho: np.ndarray) -> tuple[np.ndarray, dict]:
    """Diagnostic conditional expectation, NEVER a replacement physical state."""
    n = len(rho).bit_length()-1
    unitary, labels = regional_schur_columns(n)
    x = unitary.conj().T@rho@unitary
    reference = np.zeros_like(x)
    sectors = {}
    for q in range(n % 2, n+1, 2):
        ids = [[i for i, label in enumerate(labels) if label[:2] == (q, m)]
               for m in range(-q, q+1, 2)]
        sigma = sum(x[np.ix_(indices, indices)] for indices in ids)
        for indices in ids:
            reference[np.ix_(indices, indices)] = sigma/(q+1)
        whole = sum(ids, [])
        p = float(np.trace(sigma).real)
        difference = x[np.ix_(whole, whole)]-reference[np.ix_(whole, whole)]
        sectors[str(q)] = dict(p=p, unnormalized_block_trace_distance=.5*float(np.abs(np.linalg.eigvalsh(difference)).sum()),
            conditional_trace_distance=(.5*float(np.abs(np.linalg.eigvalsh(difference)).sum())/p if p > 1e-12 else None))
    return unitary@reference@unitary.conj().T, sectors


def rotational_symmetry_defect(rho: np.ndarray, *, tolerance: float = 1e-9,
                               compact_triangular_2x3: bool = False) -> dict:
    """Audit actual 2..6-site RDMs and distinguish scalar data from strict spectra.

    The SU2 conditional expectation is ONLY a mathematical comparator used
    for distances and continuity bounds. It never replaces rho or supplies
    reported entanglement levels. Scalar operators commute with ALL three
    total-spin generators, and their expectation is unchanged even when rho
    is noninvariant. Nonpositive sampled estimates retain diagnostics but
    receive NO entropy/spectrum continuity bounds and are never repaired.
    """
    matrix = np.asarray(rho, np.complex128)
    d = len(matrix) if matrix.ndim == 2 else 0
    if (not 4 <= d <= 64 or d & (d-1) or matrix.shape != (d, d)
            or not np.isfinite(matrix).all() or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('finite 2..6-site square RDM and positive tolerance required')
    hermitian = float(np.linalg.norm(matrix-matrix.conj().T))
    trace_error = float(abs(np.trace(matrix)-1))
    if max(hermitian, trace_error) > tolerance:
        raise ValueError('Hermitian unit-trace estimate required; no repair is performed')
    eigenvalues = np.linalg.eigvalsh(matrix)
    physical = float(eigenvalues.min()) >= -tolerance
    # Numerical positivity eligibility is not positivity for a mathematical
    # density-matrix continuity bound.  A small negative eigenvalue is kept,
    # never clipped or silently treated as an exactly positive matrix.
    bounds_eligible = float(eigenvalues.min()) >= 0.0
    reference, sectors = _rotational_comparator(matrix)
    delta = matrix-reference
    distance = .5*float(np.abs(np.linalg.eigvalsh(delta)).sum())
    frobenius = float(np.linalg.norm(delta))
    bank, total = _rotational_audit_operators(d.bit_length()-1)
    rows = {}
    for name, operator in bank['scalar'].items():
        if name.startswith('compact6_') and not compact_triangular_2x3:
            continue
        commutator = max(float(np.linalg.norm(operator@spin-spin@operator)) for spin in total)
        actual = np.trace(matrix@operator)
        change = np.trace(delta@operator)
        norm = float(np.max(np.abs(np.linalg.eigvalsh(operator))))
        rows[name] = dict(actual=float(actual.real), comparator_change=float(change.real),
            comparator_change_imaginary=float(abs(change.imag)), total_spin_commutator_frobenius=commutator,
            operator_norm=norm, generic_trace_distance_change_bound=2*distance*norm)
    anisotropy = []
    for name, zz in bank['zz'].items():
        scalar = bank['scalar'][name]
        op = zz-scalar/3
        norm = float(np.max(np.abs(np.linalg.eigvalsh(op))))
        anisotropy.append(dict(pair=list(map(int, name.split('_')[1:])),
            szsz_minus_scalar_over_three=float(np.trace(matrix@op).real),
            comparator_value=float(np.trace(reference@op).real),
            trace_distance_bound=2*distance*norm))
    bounds = None
    if bounds_eligible:
        # Fannes--Audenaert for d-dimensional density matrices; natural logs.
        t = min(distance, 1-1/d)
        binary = -t*np.log(t)-(1-t)*np.log1p(-t) if t > 0 else 0.
        bounds = dict(entropy_difference_upper_bound_nats=float(t*np.log(d-1)+binary),
            eigenvalue_l1_difference_upper_bound=2*distance,
            purity_difference_upper_bound=min(1., 4*distance),
            scope='Bounds compare THIS valid matrix and a diagnostic invariant comparator. Not a bound on a Monte Carlo estimator versus its true RDM, or on ground-state error; no strict spectrum is reported.')
    return dict(region_size=d.bit_length()-1, physical_matrix_valid=physical,
        numerical_positivity_tolerance=tolerance,
        continuity_bounds_eligible=bounds_eligible,
        minimum_eigenvalue=float(eigenvalues.min()), hermiticity_frobenius=hermitian, trace_error=trace_error,
        strict_su2_eligible=physical and frobenius <= tolerance,
        strict_su2_eligibility_scope='Numerical regional invariance/positivity diagnostic at the stated tolerance, not exact positivity or a global-S0 certificate.',
        rotational_defect_trace_distance=distance, rotational_defect_frobenius=frobenius,
        sector_conditional_defects=sectors, invariant_operator_checks=rows,
        pair_axis_anisotropy=anisotropy, maximum_absolute_pair_axis_anisotropy=max(abs(row['szsz_minus_scalar_over_three']) for row in anisotropy),
        continuity_bounds=bounds, strict_entanglement_spectrum_reported=False,
        diagnostic_comparator_only=True, input_repaired=False)


def rotational_defect_chain_audit(matrices: np.ndarray, chain_ids: np.ndarray, *,
                                  compact_triangular_2x3: bool = False) -> dict:
    """De-noise a squared rotational defect using independent WHOLE chains.

    Equal-size independent chain groups are required; time records inside a
    chain are NOT independent observations. The cross-chain numerator is an
    unbiased quadratic estimator under independent identically distributed
    chain means. Division by an estimated normalization retains ratio bias.
    Negative estimates and delete-chain SE are retained, never clipped into
    an exact-S0 certificate. This does not certify Born equilibration.
    """
    a, groups = np.asarray(matrices, complex), np.asarray(chain_ids)
    if (a.ndim != 3 or not 4 <= a.shape[1] <= 64 or a.shape[1] != a.shape[2]
            or a.shape[1] & (a.shape[1]-1) or groups.shape != (len(a),) or not np.isfinite(a).all()):
        raise ValueError('finite per-record 2..6-site matrices with explicit chain IDs required')
    ids, counts = np.unique(groups, return_counts=True)
    if len(ids) < 3 or len(set(counts)) != 1:
        raise ValueError('at least three equal-size independent whole-chain groups required')
    means = np.stack([a[groups == chain].mean(axis=0) for chain in ids])
    norms = np.trace(means, axis1=1, axis2=2)
    if np.max(abs(norms.imag)) > 1e-10 or np.any(norms.real <= 0):
        raise ValueError('positive real chain normalizations required')
    deltas = np.stack([rho-_rotational_comparator(rho)[0] for rho in means])
    def estimate(rows, weights):
        g = len(rows); total = rows.sum(axis=0)
        numerator = float((np.vdot(total, total).real-np.vdot(rows, rows).real)/(g*(g-1)))
        return numerator/float(np.mean(weights))**2, numerator
    value, numerator = estimate(deltas, norms.real)
    deletes = np.array([estimate(np.delete(deltas, i, axis=0), np.delete(norms.real, i))[0] for i in range(len(ids))])
    se = float(np.sqrt((len(ids)-1)*np.mean((deletes-deletes.mean())**2)))
    plug = float(np.vdot(deltas.mean(axis=0), deltas.mean(axis=0)).real/norms.real.mean()**2)
    grand = means.mean(axis=0)/norms.real.mean()
    jack = np.stack([(means.sum(axis=0)-means[i])/(norms.real.sum()-norms[i].real)
                     for i in range(len(ids))])
    bank, _ = _rotational_audit_operators(a.shape[1].bit_length()-1)
    linear = {}
    ops = dict(bank['scalar'])
    ops.update({f'axis_anisotropy_{name}': zz-bank['scalar'][name]/3 for name, zz in bank['zz'].items()})
    for name, op in ops.items():
        if name.startswith('compact6_') and not compact_triangular_2x3:
            continue
        values = np.einsum('cij,ji->c', jack, op)
        linear_value = float(np.trace(grand@op).real)
        error = float(np.sqrt((len(ids)-1)*np.mean(abs(values-values.mean())**2)))
        linear[name] = dict(value=linear_value, whole_chain_jackknife_SE=error,
                            signal_over_SE=linear_value/error if error > 0 else None)
    return dict(independent_chains=len(ids), records_per_chain=int(counts[0]),
        plug_in_squared_frobenius_defect=plug,
        cross_chain_squared_defect=value, cross_chain_numerator=numerator,
        whole_chain_jackknife_SE=se, signal_over_SE=value/se if se > 0 else None,
        negative_estimate_retained=value < 0, normalization_ratio_bias_removed=False,
        linear_observable_chain_estimates=linear,
        exact_SU2_certified=False, mixing_certified=False,
        warning='A detection/noise diagnostic on a fixed cached state. No per-record independence, clipping, exact-singlet certification or unseen-tail guarantee.')


def regional_reference_chain_audit(
    matrices: np.ndarray, chain_ids: np.ndarray, reference: np.ndarray,
) -> dict:
    """Distinct-chain squared discrepancies to ONE fixed physical RDM.

    Only unit-trace Born conditional records are accepted, so no estimated
    importance normalization enters the quadratic U statistic. Its numerator
    is unbiased under genuinely independent identically distributed chain
    means; within-chain records need not be independent. A finite-quality
    deterministic reference is NOT an exact ground-state/error certificate.

    The invariant-algebra feature is the LINEAR SU2 conditional expectation
    used only for an operator discrepancy, never a replacement physical RDM
    or manufactured entanglement spectrum. Negative estimates are retained.
    """
    records = np.asarray(matrices, np.complex128)
    groups = np.asarray(chain_ids)
    fixed = np.asarray(reference, np.complex128)
    if (records.ndim != 3 or not 4 <= records.shape[1] <= 64
            or records.shape[1] != records.shape[2]
            or records.shape[1] & (records.shape[1]-1)
            or groups.shape != (len(records),) or fixed.shape != records.shape[1:]
            or not np.isfinite(records).all() or not np.isfinite(fixed).all()):
        raise ValueError('finite 2..6-site Born records, chain IDs and matching fixed RDM required')
    ids, counts = np.unique(groups, return_counts=True)
    if len(ids) < 3 or len(set(counts)) != 1:
        raise ValueError('at least three equal-size independent whole chains required')
    traces = np.trace(records, axis1=1, axis2=2)
    if (np.max(abs(traces-1)) > 1e-10
            or np.max(abs(records-records.conj().transpose(0, 2, 1))) > 1e-10
            or abs(np.trace(fixed)-1) > 1e-10
            or np.max(abs(fixed-fixed.conj().T)) > 1e-10):
        raise ValueError('unit-trace Hermitian Born records/reference required; no normalization repair')
    means = np.stack([records[groups == c].mean(axis=0) for c in ids])
    differences = means-fixed
    invariant = np.stack([_rotational_comparator(matrix)[0] for matrix in differences])
    features = {'raw':differences, 'invariant_operator':invariant,
                'rotation_noninvariant':differences-invariant}
    def quadratic(rows):
        size = len(rows)
        return float((np.vdot(rows.sum(axis=0), rows.sum(axis=0)).real
                      -np.vdot(rows, rows).real)/(size*(size-1)))
    outputs, deletions = {}, []
    for name, feature in features.items():
        value = quadratic(feature)
        delete = np.array([quadratic(np.delete(feature, i, axis=0)) for i in range(len(ids))])
        error = float(np.sqrt((len(ids)-1)*np.mean((delete-delete.mean())**2)))
        outputs[name] = dict(cross_chain_squared_frobenius_discrepancy=value,
            whole_chain_delete_jackknife_SE=error,
            signal_over_SE=value/error if error > 0 else None,
            plug_in_squared_frobenius_discrepancy=float(np.vdot(feature.mean(axis=0),feature.mean(axis=0)).real),
            negative_estimate_retained=value < 0)
        deletions.append(delete)
    delete_values = np.stack(deletions, axis=1)
    centered = delete_values-delete_values.mean(axis=0)
    return dict(independent_chains=len(ids), records_per_chain=int(counts[0]),
        features=outputs,
        decomposition_closure_error=abs(outputs['raw']['cross_chain_squared_frobenius_discrepancy']
            -sum(outputs[name]['cross_chain_squared_frobenius_discrepancy']
                 for name in ('invariant_operator','rotation_noninvariant'))),
        whole_chain_delete_covariance=((len(ids)-1)*(centered.T@centered/len(ids))).tolist(),
        physical_state_replaced=False, strict_entanglement_spectrum_reported=False,
        normalization_ratio_bias_present=False, mixing_certified=False, phase_claim=False,
        warning='Same cut/operator basis only. Independent-chain U statistic removes diagonal norm-noise bias, not Born nonequilibration, rare tails, deterministic-reference quality error, or candidate-family refinement dependence.')


def compact6_resonance_readout(rho: np.ndarray) -> dict:
    """Separate doublet occupation from conditional multiplicity correlations.

    The ordered 2x3 patch has disjoint triangles (0,3,4) and (1,5,2).
    Each triangle's doubled q=1 sector has two multiplicities. The two E2
    bond components and A2 chirality, normalized to square to P(q=1),
    are three resonance-pseudospin axes, NOT magnetic spin components.
    P11=P(q_L=1)P(q_R=1) includes BOTH regional total q=0 and q=2.
    Conditional means use P11 on BOTH triangles before subtracting them;
    dividing an unconditioned connected correlator by p11 is not equivalent.

    These are legal scalar-operator measurements for any actual rho, not
    SU2 twirling, a manufactured entanglement spectrum, or a phase label.
    The raw chirality product is six-spin information; p11 is at most
    four-spin information. Conditional ratios are nonlinear diagnostics.
    """
    matrix = np.asarray(rho, np.complex128)
    channels = compact6_channel_readout(matrix)  # physical RDM validation
    bank = compact6_channel_operators()
    axes = [[bank[f'E2_triangle{side}_component{i}']/np.sqrt(3/8)
             for i in range(2)] + [bank[f'chi_triangle{side}']/(np.sqrt(3)/4)]
            for side in range(2)]
    supports = [a[2]@a[2] for a in axes]
    both = supports[0]@supports[1]
    mass = float(np.trace(matrix@both).real)
    joint = spin_partition_readout(matrix, (0, 3, 4))
    joint_mass = sum(row['probability'] for row in joint['weights']
                     if row['q_left'] == row['q_right'] == 1)
    closure = abs(mass-joint_mass)
    if closure > 1e-11:
        raise RuntimeError('doublet support and independent CG joint weights disagree')
    result = dict(
        triangles=[[0, 3, 4], [1, 5, 2]], doubled_q_left=1, doubled_q_right=1,
        axis_labels=['E2_component0', 'E2_component1', 'A2_chirality'],
        p_doublet_doublet=mass, joint_probability_closure_error=closure,
        raw_chirality_product=channels['chi_intertriangle_product'],
        conditional=None, strict_entanglement_spectrum=False,
        interpretation='Sector-conditioned invariant resonance observables, not magnetic order or a QSL diagnostic.',
    )
    if mass <= 1e-12:
        return result
    means = np.array([[np.trace(matrix@both@a).real/mass for a in side]
                      for side in axes])
    correlations = np.array([[np.trace(matrix@a@b).real/mass for b in axes[1]]
                             for a in axes[0]])
    if max(np.max(abs(means)), np.max(abs(correlations))) > 1+1e-10:
        raise RuntimeError('conditional pseudospin measurement violates its operator bound')
    connected = correlations-np.outer(means[0], means[1])
    result['conditional'] = dict(
        means=means.tolist(), correlations=correlations.tolist(), connected=connected.tolist(),
        E2_axis_average=float(np.trace(correlations[:2, :2])/2),
        E2_axis_average_connected=float(np.trace(connected[:2, :2])/2),
        A2_axis_correlation=float(correlations[2, 2]),
        A2_axis_correlation_connected=float(connected[2, 2]),
    )
    return result


def saved_amplitude_region_density(
    state: np.ndarray, basis: np.ndarray, sites: Sequence[int],
) -> np.ndarray:
    """Partial trace of a saved coherent state, not configuration frequencies.

    Validation-scale physical support only (N <= 20). Rows of ``basis`` are
    distinct down/up bit strings, site zero MSB. Missing rows mean EXACT zero
    amplitudes, never unobserved Monte-Carlo configurations. The input must
    already be normalized; no state fitting, projection or repair occurs.
    Output site order is exactly ``sites``, not their sorted order.
    """
    bits = np.asarray(basis)
    psi = np.asarray(state, np.complex128)
    cut = tuple(sites)
    if (bits.ndim != 2 or not 2 <= bits.shape[1] <= 20
            or psi.shape != (len(bits),) or not np.isfinite(psi).all()
            or not np.all((bits == 0) | (bits == 1))):
        raise ValueError('distinct physical bit rows and finite coherent amplitudes required')
    n = bits.shape[1]
    if (not 1 <= len(cut) <= min(8, n) or len(set(cut)) != len(cut)
            or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                   or not 0 <= i < n for i in cut)):
        raise ValueError('select 1..8 distinct integer physical sites')
    packed = bits.astype(np.int64) @ (1 << np.arange(n-1, -1, -1))
    if len(np.unique(packed)) != len(bits):
        raise ValueError('duplicate physical configurations are not allowed')
    if abs(np.vdot(psi, psi)-1) > 1e-10:
        raise ValueError('saved physical state must already be normalized')
    rest = tuple(i for i in range(n) if i not in cut)
    rows = bits[:, cut].astype(np.int64) @ (1 << np.arange(len(cut)-1, -1, -1))
    cols = bits[:, rest].astype(np.int64) @ (1 << np.arange(len(rest)-1, -1, -1))
    amplitudes = np.zeros((2**len(cut), 2**len(rest)), np.complex128)
    amplitudes[rows, cols] = psi
    return amplitudes @ amplitudes.conj().T


def multiplicity_operator_data(
    rho: np.ndarray, *, tolerance: float = 1e-9,
) -> tuple[dict, dict[int, np.ndarray]]:
    """Return legal invariant-operator data sigma_q=Tr_m(P_q rho P_q).

    For 1..8 sites, sigma_q is positive with trace p(q). Its off-diagonal
    entries depend on the declared balanced CG basis. Conditional purity is
    basis independent, but is NOT a physical entanglement entropy/purity of
    rho. Only verified invariant inputs receive strict entanglement levels.
    The invariant reconstruction is used ONLY to quantify a symmetry defect;
    it never replaces the input or becomes a newly manufactured state.
    """
    matrix = np.asarray(rho, dtype=np.complex128)
    block = physical_spin_block_readout(matrix, tolerance=tolerance)
    n = matrix.shape[0].bit_length()-1
    unitary, labels = regional_schur_columns(n)
    x = unitary.conj().T @ matrix @ unitary
    invariant = np.zeros_like(x)
    data, sectors = {}, {}
    for q in range(n % 2, n+1, 2):
        indices = [[i for i, label in enumerate(labels) if label[:2] == (q, m)]
                   for m in range(-q, q+1, 2)]
        sigma = sum(x[np.ix_(ids, ids)] for ids in indices)
        p = float(np.trace(sigma).real)
        purity = float(np.trace(sigma @ sigma).real / p**2) if p > 1e-12 else None
        offdiag = sigma-np.diag(np.diag(sigma))
        sectors[str(q)] = dict(
            q=q, multiplicity=sigma.shape[0], p=p,
            conditional_operator_purity=purity,
            offdiagonal_operator_frobenius=float(np.linalg.norm(offdiag)),
            minimum_operator_eigenvalue=float(np.linalg.eigvalsh(sigma).min()),
        )
        sigma.setflags(write=False)
        data[q] = sigma
        for ids in indices:
            invariant[np.ix_(ids, ids)] = sigma/(q+1)
    trace_distance = .5*float(np.abs(np.linalg.eigvalsh(x-invariant)).sum())
    eigenvalues = np.linalg.eigvalsh(matrix)
    positive = eigenvalues[eigenvalues > 0]
    result = dict(
        region_size=n, physical=block, sectors=sectors,
        physical_region_entropy_nats=float(-positive @ np.log(positive)),
        symmetry_defect_trace_distance=trace_distance,
        matrix_basis='regional_schur_columns balanced tree; doubled (q,m,a), down/up, MSB',
        interpretation=('sigma_q stores expectations of SU2-invariant multiplicity operators for ANY rho. '
                        'Noninvariant inputs have NO strict SU2 entanglement spectrum. '
                        'Conditional operator purity is not an entanglement witness.'),
    )
    return result, data


def multiplicity_operator_distance(a: np.ndarray, b: np.ndarray) -> dict:
    """Trace distances of the SAME region's invariant algebra and spin weights.

    D_alg=1/2 sum_q ||sigma_q(a)-sigma_q(b)||_1 is contractive and measurable
    without invariant states. D_weights<=D_alg<=D_raw. Their difference is
    information beyond TOTAL-spin probabilities, not beyond all pair/dimer
    correlations. Conditional distances need occupied sectors on both sides.
    Refinement differences remain empirical drift, not certified errors.
    """
    a, b = np.asarray(a, complex), np.asarray(b, complex)
    if a.shape != b.shape:
        raise ValueError('the same physical region dimension is required')
    ra, sa = multiplicity_operator_data(a)
    rb, sb = multiplicity_operator_data(b)
    raw = .5*float(np.abs(np.linalg.eigvalsh(a-b)).sum())
    rows, algebra, population, conditional_weighted = {}, 0., 0., 0.
    for q in sa:
        pa, pb = ra['sectors'][str(q)]['p'], rb['sectors'][str(q)]['p']
        distance = .5*float(np.abs(np.linalg.eigvalsh(sa[q]-sb[q])).sum())
        weight = .5*abs(pa-pb)
        conditional = (.5*float(np.abs(np.linalg.eigvalsh(sa[q]/pa-sb[q]/pb)).sum())
                       if min(pa, pb) > 1e-12 else None)
        rows[str(q)] = dict(operator_trace_distance=distance, weight_distance=weight,
                            conditional_trace_distance=conditional)
        algebra += distance
        population += weight
        conditional_weighted += min(pa, pb)*conditional if conditional is not None else 0.
    error = max(0., population-algebra, algebra-raw)
    if error > 1e-10:
        raise RuntimeError('invariant-algebra measurement violates trace-distance contractivity')
    return dict(
        physical_trace_distance=raw, invariant_operator_trace_distance=algebra,
        spin_weight_total_variation=population,
        excess_over_spin_weights=algebra-population,
        conditional_trace_distance_weighted_by_minimum_sector_mass=conditional_weighted,
        by_q=rows, contractivity_error=error,
        strict_entanglement_spectrum_comparison=(ra['physical']['strict_su2_eligible']
                                                and rb['physical']['strict_su2_eligible']),
        warning='Operator-algebra distance, NOT entanglement-spectrum distance or a phase criterion.',
    )


def regional_su2_protocol(
    rho: np.ndarray, left_sites: Sequence[int], *,
    cyclic: bool = False, compact_triangular_2x3: bool = False,
) -> tuple[dict, dict[int, np.ndarray]]:
    """Run a coherent 2..6-site readout with an EXPLICIT physical bipartition.

    compact_triangular_2x3=True requires the exact row-major geometric patch
    used by compact6_channel_readout; do not apply it to a winding ring.
    Six-body remainders use ALL pair and symmetrized pair-product scalars,
    not just Hamiltonian bonds. Cumulants are distinct nonlinear readouts.
    """
    matrix = np.asarray(rho, complex)
    mult, data = multiplicity_operator_data(matrix)
    n = mult['region_size']
    if not 2 <= n <= 6 or (compact_triangular_2x3 and n != 6):
        raise ValueError('protocol requires 2..6 sites; compact2x3 requires six')
    partition = spin_partition_readout(matrix, left_sites)
    observables = regional_physical_observables(matrix, cyclic=cyclic)
    result = dict(multiplicity=mult, partition=partition, conventional=observables)
    if n == 6 and len(tuple(left_sites)) == 3:
        triple = three_bond_readout(matrix, left_sites)
        splits = {str(q): joint_spin_operator_split(6, left_sites, (q, q, 0))
                  for q in (1, 3)}
        measures = {q: split.measure(matrix) for q, split in splits.items()}
        closure = max(abs(measures[str(q)]['high']-triple[f'high_q{q}']) for q in (1, 3))
        if closure > 1e-11:
            raise RuntimeError('three-bond and independent low-order projector decompositions disagree')
        result['six_body'] = dict(
            projector_splits=measures, three_bond=triple,
            diagnostics={q: split.diagnostics for q, split in splits.items()},
            independent_three_bond_closure_error=float(closure),
        )
    if compact_triangular_2x3:
        result['local_spatial_channels'] = compact6_channel_readout(matrix)
        result['triangle_resonance'] = compact6_resonance_readout(matrix)
    return result, data


__all__ = ['saved_amplitude_region_density', 'multiplicity_operator_data',
           'multiplicity_operator_distance', 'regional_su2_protocol',
           'compact6_resonance_readout', 'rotational_symmetry_defect',
           'rotational_defect_chain_audit', 'regional_reference_chain_audit']
