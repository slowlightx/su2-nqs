"""Small exact linear-span tools for physical-orbit refinement and calibration."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np


def make_orbit_amplitude_products(raw_amplitude: Callable, images: np.ndarray, *,
                                  projection_chunk: int = 4) -> Callable:
    """Return coherent Phi and incoherent raw-orbit Q for physical bit batches.

    No Hamiltonian, exchange connections, or wavefunction division is used.
    The caller divides Q by the known raw norm when forming a normalized law.
    """
    maps = np.asarray(images)
    if (maps.ndim != 2 or not len(maps) or maps.shape[1] < 1
            or not np.issubdtype(maps.dtype, np.integer)
            or not np.all(np.sort(maps, axis=1) == np.arange(maps.shape[1]))
            or isinstance(projection_chunk, bool) or not isinstance(projection_chunk, Integral)
            or projection_chunk < 1 or len(maps) % projection_chunk):
        raise ValueError('physical permutations and a positive dividing chunk required')
    groups = jnp.asarray(maps.reshape(-1, projection_chunk, maps.shape[1]))

    def one(bank, bits):
        def part(group):
            values = jax.vmap(lambda image: raw_amplitude(bank, bits[image]))(group)
            return jnp.sum(values), jnp.sum(abs(values)**2)
        coherent, incoherent = jax.lax.map(part, groups)
        return jnp.sum(coherent)/len(maps), jnp.sum(incoherent)/len(maps)
    return jax.jit(jax.vmap(one, in_axes=(None, 0)))


def protected_center_directions(parent: np.ndarray, hparent: np.ndarray,
                                protected_rows: np.ndarray, *, tolerance: float = 2e-11
                                ) -> tuple[np.ndarray, dict]:
    """Build normalized parent/old-column/added-column raw center directions.

    The row mask is an orthogonal physical retained-multiplicity projector.
    Project the residual BEFORE Gram--Schmidt. The old tangent after removing
    the parent need not remain row-supported on old columns: it is an old-row
    direction ABOUT THE SAME possibly enriched parent, not an old-width state.
    These raw-H directions only specify a trial span, not a projected update.
    """
    c, h, mask = np.asarray(parent), np.asarray(hparent), np.asarray(protected_rows)
    if (c.ndim != 1 or h.shape != c.shape or mask.shape != c.shape or mask.dtype != np.bool_
            or not np.isfinite(c).all() or not np.isfinite(h).all()
            or not np.isfinite(tolerance) or not 0 < tolerance < 1e-6
            or not np.any(mask) or np.all(mask)):
        raise ValueError('finite center vectors and nontrivial Boolean protected-row mask required')
    dtype = np.result_type(c.dtype, h.dtype, np.float64)
    c, h = c.astype(dtype), h.astype(dtype)
    norm = float(np.linalg.norm(c))
    if abs(norm-1) > tolerance:
        raise ValueError('parent must have exact raw unit norm')
    energy = np.vdot(c, h)
    if abs(energy.imag) > tolerance*max(1., abs(energy)):
        raise ValueError('raw H expectation must be real')
    residual = h-energy.real*c
    projected = [-np.where(mask, residual, 0), -np.where(~mask, residual, 0)]
    basis = [c.copy()]; projected_norms = []; orthogonal_norms = []
    parent_components = []
    for value in projected:
        projected_norms.append(float(np.linalg.norm(value)))
        parent_components.append([float(np.vdot(c, value).real), float(np.vdot(c, value).imag)])
        tangent = value.copy()
        for _ in range(2):
            for old in basis:
                tangent -= old*np.vdot(old, tangent)
        size = float(np.linalg.norm(tangent)); orthogonal_norms.append(size)
        if size <= tolerance*max(1., float(np.linalg.norm(h))):
            raise ValueError('a requested protected/additional tangent is unresolved; no replacement invented')
        basis.append(tangent/size)
    matrix = np.column_stack(basis)
    defect = float(np.max(abs(matrix.conj().T@matrix-np.eye(3))))
    if defect > tolerance:
        raise ArithmeticError('raw center basis lost orthonormality')
    return matrix, dict(raw_energy=float(energy.real), raw_residual_norm=float(np.linalg.norm(residual)),
        projected_residual_norms=projected_norms, orthogonalized_tangent_norms=orthogonal_norms,
        projected_parent_components=parent_components, raw_basis_Gram_error=defect,
        protected_row_count=int(mask.sum()), added_row_count=int((~mask).sum()),
        old_control_is_direction_about_same_enriched_parent=True,
        row_projection_precedes_orthogonalization=True, projected_A1_optimization_not_performed=True)


def raw_normalized_line_coefficients(coefficients: np.ndarray, fraction: float = 1., *,
                                    raw_angle_cap: float | None = None) -> np.ndarray:
    """Normalize real coefficients in an independently verified raw-orthonormal basis.

    Align the parent coefficient positively, optionally cap the raw FS angle,
    then shrink its raw geodesic. This changes neither basis support nor q.
    """
    c = np.asarray(coefficients)
    if (c.ndim != 1 or not len(c) or np.iscomplexobj(c) or not np.isfinite(c).all()
            or not np.isfinite(fraction) or not 0 <= fraction <= 1 or np.linalg.norm(c) == 0):
        raise ValueError('nonzero finite real coefficients and fraction in[0,1] required')
    if raw_angle_cap is not None and (not np.isfinite(raw_angle_cap) or not 0 < raw_angle_cap < np.pi/2):
        raise ValueError('raw angle cap must be finite and inside(0,pi/2)')
    c = np.asarray(c/np.linalg.norm(c), float)
    if c[0] < 0: c = -c
    size = np.linalg.norm(c[1:])
    if size < 1e-15:
        return np.r_[1., np.zeros(len(c)-1)]
    angle = float(np.arctan2(size, c[0]))
    if raw_angle_cap is not None: angle = min(angle, raw_angle_cap)
    angle *= fraction
    return np.r_[np.cos(angle), np.sin(angle)*c[1:]/size]


def calibrate_shared_projected_radius(phi: np.ndarray, proposal: np.ndarray,
        groups: np.ndarray, fitted_coefficients: np.ndarray, *, seed: int,
        maximum_angle: float = .03, fractions: Sequence[float] = (1., .5, .25, .125, .0625, .03125, 0.)
        ) -> tuple[np.ndarray, dict]:
    """Amplitude-only shared shrink using a nominal joint-bootstrap FS guard.

    All arms share each resample and shrink fraction. A bootstrap maximum
    deviation across the ENTIRE fraction grid and all arms supplies a common
    upper margin. Invalid positive-fraction resamples fail closed; none are
    discarded. This is a nominal finite-sample diagnostic, not a rigorous
    confidence certificate. Energies/H actions are deliberately not accepted.
    """
    p, q, g, fitted = map(np.asarray, (phi, proposal, groups, fitted_coefficients))
    grid = np.asarray(fractions, float)
    if (p.ndim != 2 or q.shape != (len(p),) or g.shape != q.shape
            or fitted.ndim != 2 or fitted.shape[1] != p.shape[1] or np.iscomplexobj(fitted)
            or np.iscomplexobj(q) or not all(np.isfinite(v).all() for v in (p, q, fitted, grid))
            or np.any(q <= 0) or not np.isfinite(maximum_angle) or not 0 < maximum_angle < np.pi/2
            or grid.ndim != 1 or len(grid) < 2 or grid[0] != 1 or grid[-1] != 0
            or not np.all(np.diff(grid) < 0)):
        raise ValueError('finite common-Q amplitudes, real arms and descending[1,...,0] fractions required')
    labels, counts = np.unique(g, return_counts=True)
    if len(labels) < 4 or len(set(counts)) != 1:
        raise ValueError('at least four equal independent calibration groups required')
    normalized = np.asarray([raw_normalized_line_coefficients(c) for c in fitted])
    coefficients = np.asarray([[raw_normalized_line_coefficients(c, float(f))
                               for c in normalized] for f in grid])
    amplitude_scale = float(np.max(abs(p)))
    if amplitude_scale == 0:
        raise ValueError('calibration amplitude span cannot be identically zero')
    p = p/amplitude_scale
    q = q/np.max(q)
    row_grams = p.conj()[:, :, None]*p[:, None, :]/q[:, None, None]

    def angles(gram):
        result = np.full((len(grid), len(fitted)), np.inf)
        if not np.isfinite(gram).all() or gram[0, 0].real <= 0:
            return result
        for k, arms in enumerate(coefficients):
            for a, c in enumerate(arms):
                norm = float(np.vdot(c, gram@c).real)
                if not np.isfinite(norm) or norm <= 0: continue
                overlap = abs(gram[0]@c)/np.sqrt(gram[0, 0].real*norm)
                if not np.isfinite(overlap) or overlap > 1+2e-10: continue
                result[k, a] = np.arccos(min(1., overlap))
        return result

    point = angles(row_grams.mean(0))
    group_grams = np.asarray([row_grams[g == label].mean(0) for label in labels])
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([angles(group_grams[rng.integers(len(labels), size=len(labels))].mean(0))
                            for _ in range(512)])
    finite = bool(np.isfinite(point).all() and np.isfinite(bootstrap).all())
    margin = (max(0., float(np.quantile(np.max(bootstrap-point[None], axis=(1, 2)), .95)))
              if finite else np.inf)
    upper = point.max(1)+margin
    eligible = np.flatnonzero((upper <= maximum_angle) & (grid > 0))
    selected = int(eligible[0]) if len(eligible) else len(grid)-1
    selected_arms = coefficients[selected]
    leave = np.asarray([angles((group_grams.sum(0)-row)/(len(labels)-1))[selected]
                        for row in group_grams])
    se = (np.sqrt((len(labels)-1)/len(labels)*np.sum((leave-leave.mean(0))**2, axis=0))
          if np.isfinite(leave).all() else np.full(len(fitted), np.nan))
    interval = (np.quantile(bootstrap[:, selected], [.025, .975], axis=0).T
                if np.isfinite(bootstrap[:, selected]).all() else np.full((len(fitted), 2), np.nan))
    def finite_list(value):
        array = np.asarray(value)
        if array.ndim == 0: return float(array) if np.isfinite(array) else None
        return [finite_list(row) for row in array]
    report = dict(shared_fraction=float(grid[selected]), selected_grid_index=selected,
        grid=grid.tolist(), point_FS_radians=finite_list(point),
        simultaneous_bootstrap_upper_margin=None if not np.isfinite(margin) else margin,
        joint_arm_grid_upper=[None if not np.isfinite(x) else float(x) for x in upper],
        selected_FS_radians=finite_list(point[selected]), selected_FS_block_jackknife_SE=finite_list(se),
        selected_FS_bootstrap_95=finite_list(interval),
        invalid_bootstrap_values=int(np.count_nonzero(~np.isfinite(bootstrap))),
        maximum_angle_radians=float(maximum_angle), records=len(p), groups=len(labels),
        shared_fraction_and_common_radius_cap=True, equal_physical_radius_not_claimed=True,
        simultaneous_grid_bootstrap_guard_is_not_rigorous_certificate=True,
        amplitude_only=True, energy_or_H_used=False, invalid_resamples_never_dropped=True,
        null_parent_fallback=not len(eligible), bootstrap_seed=int(seed))
    return selected_arms, report
