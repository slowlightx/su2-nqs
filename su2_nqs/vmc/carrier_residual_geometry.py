"""Post-hoc geometry of saved physical amplitudes around a carrier ray.

All inner products use the SAME positive, possibly unnormalized proposal Q.
No wavefunction division at individual configurations, state evaluation,
sampling, step-size optimization, or variational gate is performed here.
Finite-Q inner products need not make the physical H self-adjoint. Therefore
both interference terms are retained, including after reference splitting.

Outputs are JSON-safe. Concentrations describe observed quadrature rows, not
certified full-Hilbert-space support or independent effective sample sizes.
"""
from __future__ import annotations

import numpy as np


def _complex(value: complex) -> list[float]:
    return [float(np.real(value)), float(np.imag(value))]


def _vector(value: np.ndarray, count: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.complex128)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite vector of length {count}')
    return result


def _matrix(value: np.ndarray, count: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.complex128)
    if result.ndim != 2 or result.shape[0] != count or result.shape[1] < 1 or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite (sample, nonempty column) matrix')
    return result


def _quadrature(q: np.ndarray, count: int) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (count,) or not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError('q must contain one finite positive density per sample')
    # The common factor min(Q) cancels. This keeps sqrt weights <= 1.
    return np.sqrt(np.min(q) / q)


def _groups(groups: np.ndarray | None, count: int) -> np.ndarray | None:
    if groups is None:
        return None
    groups = np.asarray(groups)
    if groups.shape != (count,) or not np.issubdtype(groups.dtype, np.integer):
        raise ValueError('groups must contain one integer group label per sample')
    return groups


def _unit(value: np.ndarray, weight: np.ndarray,
          action: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None] | None:
    scale = float(np.max(abs(value)))
    if scale == 0:
        return None
    weighted = value / scale * weight
    norm = np.linalg.norm(weighted)
    if norm == 0 or not np.isfinite(norm):
        raise FloatingPointError('nonzero state has unrepresentable quadrature norm')
    h = None if action is None else action / scale * weight / norm
    if h is not None and not np.isfinite(h).all():
        raise FloatingPointError('H action has unrepresentable quadrature scale')
    return weighted / norm, h


def _concentration(vector: np.ndarray, groups: np.ndarray | None,
                   *, defined: bool = True) -> dict:
    mass = abs(vector) ** 2
    norm = float(mass.sum())
    if not defined or norm == 0:
        return dict(defined=False, row_fractions=None, max_sample_fraction=None,
                    top_four_fraction=None, effective_row_count=None,
                    group_fractions=None, max_group_fraction=None)
    fraction = mass / norm
    grouped = None if groups is None else {
        str(int(label)): float(fraction[groups == label].sum()) for label in np.unique(groups)}
    return dict(defined=True, row_fractions=fraction.tolist(),
                max_sample_fraction=float(fraction.max()),
                top_four_fraction=float(np.sort(fraction)[-4:].sum()),
                effective_row_count=float(1 / np.sum(fraction ** 2)),
                group_fractions=grouped,
                max_group_fraction=None if grouped is None else max(grouped.values()))


def _moments(v: np.ndarray, h: np.ndarray, n_sites: int) -> dict:
    energy = np.vdot(v, h)
    residual = h - energy * v
    return dict(energy_real_per_site=float(energy.real / n_sites),
                energy_imag_per_site=float(energy.imag / n_sites),
                variance_per_site=float(np.vdot(residual, residual).real / n_sites))


def _tolerance(value: float) -> float:
    if not np.isfinite(value) or value <= 0 or value >= 1:
        raise ValueError('relative_tolerance must lie strictly between zero and one')
    return float(value)


def _ray(v: np.ndarray, c: np.ndarray, tolerance: float) -> dict:
    overlap = np.vdot(c, v)
    perpendicular = v - overlap * c
    sine = float(np.linalg.norm(perpendicular))
    cosine = float(abs(overlap))
    fs = float(np.arctan2(sine, cosine))
    if cosine <= tolerance:
        status = 'near_orthogonal_to_carrier'
        canonical = None
    elif sine <= tolerance:
        status = 'zero_or_unresolved_residual'
        canonical = None
    else:
        status = 'defined'
        canonical = perpendicular * (overlap.conjugate() / abs(overlap)) / cosine
    return dict(status=status, fs=fs, overlap=overlap, perpendicular=perpendicular,
                canonical=canonical, cosine=cosine, sine=sine)


def analyze_residual_geometry(
    carrier: np.ndarray, candidates: np.ndarray, q: np.ndarray, *,
    reference: np.ndarray | None = None, hcarrier: np.ndarray | None = None,
    hcandidates: np.ndarray | None = None, href: np.ndarray | None = None,
    n_sites: int = 1, groups: np.ndarray | None = None,
    relative_tolerance: float = 1e-12,
) -> dict:
    """Analyze carrier-orthogonal directions, support, and saved H actions.

    ``candidates`` has shape (sample, arm). ``reference`` is a physical
    amplitude (e.g. the initial decoder D0); only its component perpendicular
    to the carrier defines the reference ray. Candidate directions use
    R=(psi-b*C)/b, b=<C,psi>Q/<C,C>Q, then share the carrier normalization.
    Thus positive scaling, negative direction, and arbitrary endpoint phase
    cannot be confused. A near-zero b makes this chart explicitly undefined.

    H arrays must all use the SAME units, normally total energy; output is
    divided by n_sites. Href is optional: without it, only the right-hand
    reference/innovation interference is identifiable, not the left or
    quadratic split. No assumption of finite-Q Hermiticity is made.

    ``unit_fs_slope_per_site`` is dE/dt at t=0 on the normalized path
    cos(t)*c + sin(t)*u. ``unit_fs_curvature_per_site`` is its sin(t)^2
    coefficient, i.e. HALF the second derivative at zero. Force alignment
    is <u,(H-Ec)c>/||(H-Ec)c||, not its negative; negative real part is the
    single-sided descent orientation. Finite-Q physical energy needs BOTH
    sides, so this alignment alone does not determine the energy slope.
    """
    c0 = np.asarray(carrier, dtype=np.complex128)
    if c0.ndim != 1 or len(c0) < 1:
        raise ValueError('carrier must be a nonempty vector')
    count = len(c0)
    c0 = _vector(c0, count, 'carrier')
    values = _matrix(candidates, count, 'candidates')
    weight = _quadrature(q, count)
    groups = _groups(groups, count)
    tolerance = _tolerance(relative_tolerance)
    if isinstance(n_sites, (bool, np.bool_)) or not isinstance(n_sites, (int, np.integer)) or n_sites < 1:
        raise ValueError('n_sites must be a positive integer')
    if (hcarrier is None) != (hcandidates is None):
        raise ValueError('hcarrier and hcandidates must be supplied together')
    hc0 = None if hcarrier is None else _vector(hcarrier, count, 'hcarrier')
    actions = None if hcandidates is None else _matrix(hcandidates, count, 'hcandidates')
    if actions is not None and actions.shape != values.shape:
        raise ValueError('hcandidates must have the same shape as candidates')
    if href is not None and (reference is None or hc0 is None):
        raise ValueError('href requires reference and both carrier/candidate H actions')
    normalized = _unit(c0, weight, hc0)
    if normalized is None:
        raise ValueError('carrier must have positive norm')
    c, hc = normalized
    ec = None if hc is None else np.vdot(c, hc)
    reference_status = 'not_supplied'
    s = hs = None
    if reference is not None:
        ref = _vector(reference, count, 'reference')
        hr = None if href is None else _vector(href, count, 'href')
        ref_unit = _unit(ref, weight, hr)
        if ref_unit is None:
            reference_status = 'zero_reference'
        else:
            vref, hvref = ref_unit
            overlap = np.vdot(c, vref)
            perpendicular = vref - overlap * c
            length = np.linalg.norm(perpendicular)
            if length <= tolerance:
                reference_status = 'zero_or_unresolved_reference_residual'
            else:
                reference_status = 'defined'
                s = perpendicular / length
                if hvref is not None:
                    hs = (hvref - overlap * hc) / length
    arms, frames = [], []
    for index in range(values.shape[1]):
        vunit = _unit(values[:, index], weight, None if actions is None else actions[:, index])
        if vunit is None:
            arms.append(dict(status='zero_candidate_norm', fs=None, carrier_overlap_abs=None,
                             canonical_residual_norm_squared=None, residual_concentration=None,
                             reference=dict(status='undefined_candidate'), hamiltonian=None))
            frames.append(None)
            continue
        v, hv = vunit
        ray = _ray(v, c, tolerance)
        r = ray['canonical']
        report = dict(status=ray['status'], fs=ray['fs'], carrier_overlap_abs=ray['cosine'],
                      canonical_residual_norm_squared=None if r is None else float(np.vdot(r, r).real),
                      residual_concentration=_concentration(ray['perpendicular'], groups,
                          defined=ray['sine'] > tolerance),
                      reference=dict(status=reference_status), hamiltonian=None)
        if hv is not None:
            report['hamiltonian'] = _moments(v, hv, n_sites)
            report['hamiltonian']['delta_energy_per_site'] = float((np.vdot(v, hv).real - ec.real) / n_sites)
            report['hamiltonian']['ray_decomposition_status'] = ray['status']
        if r is None:
            arms.append(report)
            frames.append(None)
            continue
        zr = float(np.vdot(r, r).real)
        unit_r = r / np.sqrt(zr)
        projected = innovation = None
        coefficient = None
        if s is not None:
            coefficient = np.vdot(s, r)
            projected = coefficient * s
            innovation = r - projected
            zi = float(np.vdot(innovation, innovation).real)
            report['reference'].update(
                projection_fraction=float(abs(coefficient) ** 2 / zr),
                innovation_fraction=float(zi / zr),
                reference_concentration=_concentration(projected, groups,
                    defined=abs(coefficient) > tolerance * np.sqrt(zr)),
                innovation_concentration=_concentration(innovation, groups,
                    defined=np.sqrt(zi) > tolerance * np.sqrt(zr)))
        if hv is not None:
            hr = (hv / ray['overlap'] - hc)
            hu = hr / np.sqrt(zr)
            denominator = 1 + zr
            left = np.vdot(c, hr)
            right = np.vdot(r, hc)
            quadratic = (np.vdot(r, hr).real - ec.real * zr) / denominator / n_sites
            defect = np.vdot(c, hu) - np.vdot(unit_r, hc).conjugate()
            force = hc - ec * c
            force_norm = float(np.linalg.norm(force))
            alignment = None if force_norm <= tolerance * max(1., np.linalg.norm(hc)) else _complex(np.vdot(unit_r, force) / force_norm)
            report['hamiltonian'].update(
                interference_left_per_site=float(left.real / denominator / n_sites),
                interference_right_per_site=float(right.real / denominator / n_sites),
                interference_sum_per_site=float((left.real + right.real) / denominator / n_sites),
                quadratic_per_site=float(quadratic),
                unit_fs_slope_per_site=float((np.vdot(c, hu).real + np.vdot(unit_r, hc).real) / n_sites),
                unit_fs_curvature_per_site=float((np.vdot(unit_r, hu).real - ec.real) / n_sites),
                hermiticity_cross_defect=_complex(defect / n_sites),
                hermiticity_cross_defect_abs_per_site=float(abs(defect) / n_sites),
                carrier_force_alignment=alignment,
                reference_split=dict(status=reference_status))
            if s is not None:
                split = report['hamiltonian']['reference_split']
                split.update(status='right_only_without_href' if hs is None else 'complete',
                    right_reference_per_site=float(np.vdot(projected, hc).real / denominator / n_sites),
                    right_innovation_per_site=float(np.vdot(innovation, hc).real / denominator / n_sites))
                if hs is not None:
                    hp = coefficient * hs
                    hi = hr - hp
                    scale = denominator * n_sites
                    split.update(
                        left_reference_per_site=float(np.vdot(c, hp).real / scale),
                        left_innovation_per_site=float(np.vdot(c, hi).real / scale),
                        quadratic_reference_per_site=float((np.vdot(projected, hp).real - ec.real * np.vdot(projected, projected).real) / scale),
                        quadratic_innovation_per_site=float((np.vdot(innovation, hi).real - ec.real * np.vdot(innovation, innovation).real) / scale),
                        quadratic_reference_innovation_cross_per_site=float(np.vdot(projected, hi).real / scale),
                        quadratic_innovation_reference_cross_per_site=float(np.vdot(innovation, hp).real / scale),
                        quadratic_cross_norm_correction_per_site=float(-2 * ec.real * np.vdot(projected, innovation).real / scale),
                        quadratic_mixed_per_site=float((np.vdot(projected, hi).real + np.vdot(innovation, hp).real - 2 * ec.real * np.vdot(projected, innovation).real) / scale))
        arms.append(report)
        frames.append(r)
    pairs = []
    for first in range(len(frames)):
        for second in range(first + 1, len(frames)):
            a, b = frames[first], frames[second]
            pair = dict(first=first, second=second, status='undefined_residual',
                        normalized_gram=None, real_overlap=None, abs_overlap_squared=None,
                        best_real_rho=None, real_fit_defect=None, best_nonnegative_rho=None,
                        positive_scale_fit_defect=None)
            if a is not None and b is not None:
                za, zb = np.vdot(a, a).real, np.vdot(b, b).real
                gram = np.vdot(a, b) / np.sqrt(za * zb)
                rho = float(np.vdot(a, b).real / za)
                positive = max(0., rho)
                pair.update(status='defined', normalized_gram=_complex(gram),
                            real_overlap=float(gram.real), abs_overlap_squared=float(abs(gram) ** 2),
                            best_real_rho=rho,
                            real_fit_defect=float(np.vdot(b - rho * a, b - rho * a).real / zb),
                            best_nonnegative_rho=positive,
                            positive_scale_fit_defect=float(np.vdot(b - positive * a, b - positive * a).real / zb))
            pairs.append(pair)
    return dict(carrier=None if hc is None else _moments(c, hc, n_sites), arms=arms, pairs=pairs,
                reference_status=reference_status, sample_count=count,
                quadrature_is_not_full_support_certification=True,
                concentration_is_not_autocorrelation_ess=True,
                finite_Q_Hermiticity_assumed=False, post_hoc_no_gate_or_step_selection=True)


def leave_one_group_geometry(
    carrier: np.ndarray, candidates: np.ndarray, q: np.ndarray, *,
    groups: np.ndarray, **kwargs,
) -> dict:
    """Recompute EVERY carrier/reference projection after deleting a group.

    Groups normally denote independent chains, not configurations or orbits.
    No IID standard error or significance test is inferred from these ranges.
    """
    carrier = np.asarray(carrier)
    candidates, q = np.asarray(candidates), np.asarray(q)
    labels = _groups(groups, len(carrier))
    unique = np.unique(labels)
    if len(unique) < 2:
        raise ValueError('leave-one-group needs at least two groups')
    pooled = analyze_residual_geometry(carrier, candidates, q, groups=labels, **kwargs)
    results = []
    for group in unique:
        keep = labels != group
        selected = dict(kwargs)
        for key in ('reference', 'hcarrier', 'hcandidates', 'href'):
            if selected.get(key) is not None:
                selected[key] = np.asarray(selected[key])[keep]
        result = analyze_residual_geometry(carrier[keep], candidates[keep], q[keep],
                                           groups=labels[keep], **selected)
        results.append(dict(omitted_group=int(group), geometry=result))
    return dict(pooled=pooled, leave_one_group_out=results,
                all_ray_coefficients_recomputed=True)


def real_tangent_span_fraction(
    carrier: np.ndarray, candidates: np.ndarray, tangents: np.ndarray, q: np.ndarray, *,
    relative_tolerance: float = 1e-12,
) -> dict:
    """Fraction in a FINITE set of real-parameter tangent probe directions.

    Project probes off the complex carrier ray. Solve a real least-squares
    problem by SVD of stacked real/imaginary components, equivalent to
    Re(T†WT) and Re(T†Wr), without squaring the numerical condition number.
    Each nonzero probe is column-normalized first; coefficients therefore
    refer to normalized probes. The complex span is only an upper bound.
    Saved random JVPs must never be described as the full parameter tangent.
    """
    c0 = np.asarray(carrier, dtype=np.complex128)
    if c0.ndim != 1 or len(c0) < 1:
        raise ValueError('carrier must be a nonempty vector')
    c0 = _vector(c0, len(c0), 'carrier')
    values = _matrix(candidates, len(c0), 'candidates')
    probes = _matrix(tangents, len(c0), 'tangents')
    weight = _quadrature(q, len(c0))
    tolerance = _tolerance(relative_tolerance)
    unit = _unit(c0, weight)
    if unit is None:
        raise ValueError('carrier must have positive norm')
    c = unit[0]
    columns, retained = [], []
    for index, probe in enumerate(probes.T):
        normalized = _unit(probe, weight)
        if normalized is None:
            continue
        v = normalized[0]
        perpendicular = v - c * np.vdot(c, v)
        length = np.linalg.norm(perpendicular)
        if length > tolerance:
            columns.append(perpendicular / length)
            retained.append(index)
    matrix = np.column_stack(columns) if columns else np.empty((len(c), 0), dtype=np.complex128)
    real = np.concatenate((matrix.real, matrix.imag), axis=0)
    ur, sr, vhr = np.linalg.svd(real, full_matrices=False)
    uc, sc, _ = np.linalg.svd(matrix, full_matrices=False)
    nr = int(np.sum(sr > tolerance * sr[0])) if len(sr) else 0
    nc = int(np.sum(sc > tolerance * sc[0])) if len(sc) else 0
    reports = []
    for candidate in values.T:
        normalized = _unit(candidate, weight)
        ray = None if normalized is None else _ray(normalized[0], c, tolerance)
        r = None if ray is None else ray['canonical']
        entry = dict(status='zero_candidate_norm' if ray is None else ray['status'],
                     real_projection_fraction=None, complex_span_upper_bound=None,
                     real_coefficients_on_normalized_probes=None)
        if r is not None:
            target = r / np.linalg.norm(r)
            target_real = np.concatenate((target.real, target.imag))
            coordinates = ur[:, :nr].T @ target_real
            coefficients = vhr[:nr].T @ (coordinates / sr[:nr])
            entry.update(real_projection_fraction=float(np.sum(coordinates ** 2)),
                         complex_span_upper_bound=float(np.sum(abs(uc[:, :nc].conj().T @ target) ** 2)),
                         real_coefficients_on_normalized_probes=coefficients.tolist())
        reports.append(entry)
    return dict(probe_count=probes.shape[1], retained_probe_indices=retained,
                real_rank=nr, complex_rank=nc,
                real_condition_number=None if nr == 0 else float(sr[0] / sr[nr - 1]),
                real_singular_values=sr.tolist(), complex_singular_values=sc.tolist(),
                arms=reports, finite_probe_span_not_full_parameter_tangent=True,
                complex_span_is_only_upper_bound=True)
