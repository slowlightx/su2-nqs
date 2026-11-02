"""Node-safe full-A1/full-H action for a linear reduced-tensor subspace."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from ..models.spatial_projector import PhysicalSpatialProjector
from .projected_spin_products import make_projected_heisenberg_products
from .sampled_multistate import SampledSubspaceMoments, sampled_subspace_moments


def make_projected_linear_products(features: Callable, projector: PhysicalSpatialProjector,
    bonds: Sequence[tuple[int, int, float]], *, chunk_size: int = 16) -> tuple[Callable, Callable, Callable]:
    """Return plan, action(bank, coefficient_matrix, plan), projected_features.

    Each column of coefficients defines one physical wavefunction. Identical
    queries and features are shared between columns, not dropped. This is a
    forward-only kernel, retaining full image multiplicities and all bonds.
    """
    plan, _, _ = make_projected_heisenberg_products(
        lambda bank, bits: jnp.sum(features(bank, bits)), projector, bonds, chunk_size=chunk_size)
    shifts = jnp.arange(projector.n_sites-1, -1, -1, dtype=jnp.uint64)
    images = jnp.asarray(projector.site_permutations, dtype=jnp.int32)
    batch_features = jax.vmap(features, in_axes=(None, 0))

    def action(bank, coefficients, prepared):
        if coefficients.ndim != 2:
            raise ValueError('coefficient matrix must have shape (features, columns)')
        keys, inverse, count, diagonal, offdiagonal = prepared
        chunks = (count+chunk_size-1)//chunk_size
        output = jnp.zeros((keys.shape[0], coefficients.shape[1]), dtype=jnp.complex128)
        def step(index, values):
            locations = index*chunk_size+jnp.arange(chunk_size)
            selected = jax.lax.dynamic_slice(keys, (index*chunk_size,), (chunk_size,))
            selected = jnp.where(locations < count, selected, keys[0])
            bits = ((selected[:, None] >> shifts) & 1).astype(jnp.int32)
            value = batch_features(bank, bits)@coefficients
            return jax.lax.dynamic_update_slice(values,
                jnp.where((locations < count)[:, None], value, 0),
                (index*chunk_size, jnp.asarray(0, dtype=index.dtype)))
        output = jax.lax.fori_loop(0, chunks, step, output)
        projected = jnp.mean(output[inverse], axis=2)
        psi = projected[:, 0]
        hpsi = diagonal[:, None]*psi+jnp.sum(offdiagonal[:, :, None]*projected[:, 1:], axis=1)
        return hpsi, psi, count, chunks*chunk_size

    def projected_features(bank, states):
        if states.ndim != 2 or states.shape[1] != projector.n_sites:
            raise ValueError('expected physical bit batch')
        # Single-image accumulation bounds memory independently of group order.
        def one(bits):
            initial = jnp.zeros_like(features(bank, bits))
            return jax.lax.fori_loop(0, projector.group_order,
                lambda i, total: total+features(bank, bits[images[i]]), initial)/projector.group_order
        return jax.vmap(one)(states)
    return plan, action, projected_features


def make_terminal_tensor_products(
    action: Callable, template_bank: dict,
) -> tuple[jax.Array, Callable, Callable]:
    """Expose an EXISTING linear terminal tensor, without new state columns.

    ``action(bank, bit_batch)`` must return ``(psi, Hpsi)`` and be homogeneous
    linear in the real float64 pytree ``bank['top'][-1]``. This holds for a
    singlet ClebschTree root, including its complete coherent spatial sum.
    The caller must establish this structural precondition; differentiation
    is not a proof that an arbitrary supplied action is linear.

    Return current flat coordinates, ``replace(bank, coordinates)``, and
    ``columns(bank, bits) -> (Phi, HPhi)``. Each output matrix has shape
    ``(sample, root_coordinate)``. AD at the ZERO tensor extracts linear
    coefficients without dividing by psi, so exact image/physical nodes and
    Hpsi at those nodes remain legal. Real/imaginary outputs are differentiated
    separately; complex wavefunctions are NOT replaced by their real parts.

    This reuses the caller's original full-H kernel, including ancestor caches.
    It has no uint64 state packing and therefore no 64-site restriction.
    Only the existing terminal tensor is exposed: no head, enlarged virtual
    width, altered magnetic structure, or new variational state is introduced.
    Lower tensors are frozen during this linear refinement. Rebuild their
    prepared bank after any subsequent update to them.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError('terminal tensor products require x64')
    if (not isinstance(template_bank, dict) or 'top' not in template_bank
            or not isinstance(template_bank['top'], (tuple, list))
            or not template_bank['top']):
        raise ValueError('a nonempty top tensor bank is required')
    leaves = jax.tree.leaves(template_bank['top'][-1])
    if (not leaves or any(np.asarray(x).dtype != np.dtype('float64')
                         or not np.isfinite(np.asarray(x)).all() for x in leaves)):
        raise ValueError('terminal coordinates must be finite real float64 arrays')
    coordinates, unravel = ravel_pytree(template_bank['top'][-1])
    if not coordinates.size:
        raise ValueError('terminal tensor must have at least one coordinate')
    zeros = jnp.zeros_like(coordinates)

    def replace(bank: dict, values: jax.Array) -> dict:
        values = jnp.asarray(values)
        if values.shape != coordinates.shape or values.dtype != jnp.dtype('float64'):
            raise ValueError('flat terminal coordinate shape/dtype mismatch')
        return dict(bank, top=tuple(bank['top'][:-1])+(unravel(values),))

    def columns(bank: dict, bits: jax.Array) -> tuple[jax.Array, jax.Array]:
        if bits.ndim != 2 or not jnp.issubdtype(bits.dtype, jnp.integer):
            raise ValueError('an integer physical-bit batch is required')

        def one(row):
            def components(values):
                psi, hpsi = action(replace(bank, values), row[None])
                return jnp.stack((psi[0].real, psi[0].imag,
                                  hpsi[0].real, hpsi[0].imag))
            coefficients = jax.jacrev(components)(zeros)
            return (coefficients[0]+1j*coefficients[1],
                    coefficients[2]+1j*coefficients[3])

        return jax.vmap(one)(bits)

    return coordinates, replace, jax.jit(columns)


def _linear_observations(
    phi: np.ndarray, hphi: np.ndarray, proposal: np.ndarray,
    observation_weights: np.ndarray | None,
) -> tuple[SampledSubspaceMoments, np.ndarray, np.ndarray, dict]:
    """Common scaling protects tiny physical amplitudes without changing ratios."""
    p, hp, q = np.asarray(phi), np.asarray(hphi), np.asarray(proposal)
    if (p.ndim != 2 or min(p.shape, default=0) < 1 or hp.shape != p.shape
            or q.shape != (len(p),) or np.iscomplexobj(q)
            or any(not np.isfinite(x).all() for x in (p, hp, q)) or np.any(q <= 0)):
        raise ValueError('finite matching Phi/HPhi and strictly positive real Q required')
    if observation_weights is None:
        w = np.full(len(p), 1. / len(p))
    else:
        w = np.asarray(observation_weights)
        if (w.shape != q.shape or np.iscomplexobj(w) or not np.isfinite(w).all()
                or np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0):
            raise ValueError('finite nonnegative observation weights with positive sum required')
        w = np.asarray(w / w.sum(), float)
    amplitude_scale = float(np.max(np.abs(p)))
    if amplitude_scale <= 0:
        raise ValueError('the sampled physical amplitude span cannot be identically zero')
    proposal_scale = float(np.max(q))
    p = np.asarray(p / amplitude_scale, complex)
    hp = np.asarray(hp / amplitude_scale, complex)
    q = np.asarray(q / proposal_scale, float)
    if not np.isfinite(hp).all() or np.any(q <= 0):
        raise FloatingPointError('input dynamic range exceeds common linear scaling')
    moments = sampled_subspace_moments(p, hp, q, observation_weights=w)
    root_weight = np.sqrt(w) / np.sqrt(q)
    x, y = p * root_weight[:, None], hp * root_weight[:, None]
    if any(not np.isfinite(a).all() for a in
           (x, y, moments.overlap, moments.hamiltonian_raw, moments.action_gram)):
        raise FloatingPointError('nonfinite common-Q linear moments')
    return moments, x, y, dict(
        amplitude_scale=amplitude_scale, proposal_scale=proposal_scale,
        log_absolute_norm_multiplier=2 * float(np.log(amplitude_scale)) - float(np.log(proposal_scale)),
        sample_count=len(p), coefficient_count=p.shape[1],
        hamiltonian_relative_antihermiticity=moments.hamiltonian_hermiticity.relative_frobenius_error,
        hamiltonian_maximum_antihermiticity=moments.hamiltonian_hermiticity.maximum_absolute_error,
        common_proposal=True, no_wavefunction_division=True,
        finite_sample_not_population_certificate=True,
    )


def _real_linear_coefficients(coefficients: np.ndarray, dimension: int) -> np.ndarray:
    c = np.asarray(coefficients)
    if (c.shape != (dimension,) or np.iscomplexobj(c) or not np.isfinite(c).all()):
        raise ValueError('finite real coefficients matching the column dimension required')
    return np.asarray(c, dtype=np.float64)


def _linear_state_moments(
    coefficients: np.ndarray, x: np.ndarray, y: np.ndarray,
    reference_coefficients: np.ndarray | None = None,
) -> dict:
    """Direct residual norms retain Hpsi at exact amplitude nodes."""
    psi, hpsi = x @ coefficients, y @ coefficients
    norm = float(np.vdot(psi, psi).real)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('a nonzero finite sampled physical state is required')
    energy = complex(np.vdot(psi, hpsi) / norm)
    residual = hpsi - energy * psi
    variance = float(np.vdot(residual, residual).real / norm)
    h2 = float(np.vdot(hpsi, hpsi).real / norm)
    masses = np.abs(psi) ** 2
    result = dict(
        norm_scaled=norm, energy=float(energy.real), energy_imaginary=float(energy.imag),
        h2=h2, variance=variance,
        hermitian_energy_residual_moment=float(np.vdot(hpsi - energy.real * psi,
                                                       hpsi - energy.real * psi).real / norm),
        weight_ess=float(1. / np.sum((masses / norm) ** 2)),
        exact_amplitude_node_rows=int(np.count_nonzero(psi == 0)),
        node_h2_contribution=float(np.sum(np.abs(hpsi[psi == 0]) ** 2) / norm),
        variance_definition='sum_w |Hpsi-E_complex psi|^2 / sum_w |psi|^2; no clipping',
    )
    if not np.isfinite([norm, energy.real, energy.imag, variance, h2]).all():
        raise FloatingPointError('nonfinite physical linear-state moments')
    if reference_coefficients is not None:
        old = x @ reference_coefficients
        old_norm = float(np.vdot(old, old).real)
        if not np.isfinite(old_norm) or old_norm <= 0:
            raise ValueError('a nonzero reference physical state is required')
        overlap = complex(np.vdot(old / np.sqrt(old_norm), psi / np.sqrt(norm)))
        fidelity = float(abs(overlap) ** 2)
        if fidelity > 1 + 1e-10:
            raise FloatingPointError('sampled overlap violates Cauchy-Schwarz')
        result.update(
            fidelity_squared=min(1., fidelity),
            fs_sine=float(np.sqrt(max(0., 1. - fidelity))),
            normalized_overlap_real=float(overlap.real),
            normalized_overlap_imaginary=float(overlap.imag),
            norm_ratio=norm / old_norm,
        )
    return result


def evaluate_linear_state(
    phi: np.ndarray, hphi: np.ndarray, proposal: np.ndarray,
    coefficients: np.ndarray, *, observation_weights: np.ndarray | None = None,
    reference_coefficients: np.ndarray | None = None,
) -> dict:
    """Evaluate frozen real coordinates on a supplied common-Q sample.

    Full complex physical amplitudes and full H actions are retained, including
    Hpsi where psi=0. Energy has an explicitly reported imaginary sampling
    defect. ``variance`` is the direct nonnegative residual moment about that
    complex Rayleigh quotient; it equals physical variance in exact quadrature.
    The Hermitized real-energy residual is also reported, not silently confused
    with it. No PSD projection or clipping of a noisy second moment is used.

    Ordinary IID Q draws use uniform observation weights. Exhaustive reference
    quadrature uses ``observation_weights=Q``. Unknown common normalization of
    Q cancels. Positive Q at observed rows is necessary, NOT proof of tail
    coverage, stationarity or effective independent sample size. ``weight_ess``
    diagnoses concentration only. This routine never selects or changes a state.
    """
    moments, x, y, info = _linear_observations(phi, hphi, proposal, observation_weights)
    c = _real_linear_coefficients(coefficients, moments.state_count)
    reference = (None if reference_coefficients is None else
                 _real_linear_coefficients(reference_coefficients, moments.state_count))
    result = _linear_state_moments(c, x, y, reference)
    result.update(info)
    result['log_norm'] = float(np.log(result['norm_scaled'])) + info['log_absolute_norm_multiplier']
    return result


def fit_parent_anchored_linear_update(
    phi: np.ndarray, hphi: np.ndarray, proposal: np.ndarray,
    parent_coefficients: np.ndarray, *, observation_weights: np.ndarray | None = None,
    damping: float = .1, metric_rcond: float = 1e-10,
    maximum_tangent_rank: int | None = None, maximum_fs_sine: float = .03,
    initial_step: float = 1., maximum_backtracks: int = 10,
    variance_weight: float = 0., maximum_relative_variance_increase: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit ONE parent-anchored natural direction and a finite physical trust step.

    Inputs are existing linear-coordinate Phi and FULL HPhi, with real trainable
    coefficients and a common strictly positive proposal Q. This is not a Ritz
    solve, a new head, an H approximation, or a fit to audit observations. The
    caller must pass FIT rows only and freeze all returned coordinates before
    calibration/audit. A subsequent independent calibration may only shrink a
    predeclared line for its physical-distance guard, not select by audit energy.

    S=Phi^dagger Phi, K=Phi^dagger HPhi, T=HPhi^dagger HPhi are built with the
    same importance weights using ``sampled_subspace_moments``. Both energy
    coherences enter H=(K+K^dagger)/2. H2 uses T, never H S^-1 H. For s=S c,
    the real projective metric is Re(S/n-s s^dagger/n^2). Its scale/global-phase
    and sampled projection nulls are removed BEFORE optional leading-eigenmode
    truncation. A relative ridge regularizes only retained physical directions.
    The finite line is tested by exact normalized overlap on these fit rows.

    The original parent is never projected into the retained metric span. The
    update alone is restricted, and omitted gradient fractions are reported:
    truncated/sampled-null directions cannot imply full-network stationarity.
    The fit objective is E_real + a fixed nonnegative variance_weight times the
    direct complex-energy residual variance. An optional relative variance
    guard is also FIT-only. The first acceptable predeclared halving is chosen;
    if none passes, return the identical parent and ``accepted=False``.

    A returned update has the same FIT norm and real-sign alignment as its
    parent. General complex phase alignment is deliberately NOT applied to real
    coordinates. All values are total-H units, not per site. No new H queries.
    """
    controls = (damping, metric_rcond, maximum_fs_sine, initial_step, variance_weight)
    if (any(isinstance(v, bool) for v in controls) or not np.isfinite(controls).all()
            or damping <= 0 or not 0 < metric_rcond < 1
            or not 0 < maximum_fs_sine < 1 or initial_step <= 0 or variance_weight < 0):
        raise ValueError('finite positive damping/step, fractional metric/trust controls required')
    if (isinstance(maximum_backtracks, bool) or not isinstance(maximum_backtracks, Integral)
            or maximum_backtracks < 0):
        raise ValueError('maximum_backtracks must be a nonnegative integer')
    if (maximum_tangent_rank is not None
            and (isinstance(maximum_tangent_rank, bool)
                 or not isinstance(maximum_tangent_rank, Integral) or maximum_tangent_rank < 1)):
        raise ValueError('maximum_tangent_rank must be a positive integer or None')
    if (maximum_relative_variance_increase is not None
            and (isinstance(maximum_relative_variance_increase, bool)
                 or not np.isfinite(maximum_relative_variance_increase)
                 or maximum_relative_variance_increase < 0)):
        raise ValueError('maximum_relative_variance_increase must be nonnegative or None')
    moments, x, y, diagnostics = _linear_observations(phi, hphi, proposal, observation_weights)
    c = _real_linear_coefficients(parent_coefficients, moments.state_count).copy()
    before = _linear_state_moments(c, x, y, c)
    norm, energy, imaginary = before['norm_scaled'], before['energy'], before['energy_imaginary']
    s, h, t = moments.overlap, moments.hamiltonian, moments.action_gram
    anti = (moments.hamiltonian_raw - moments.hamiltonian_raw.conj().T) / (2j)
    sc = s @ c
    gradient_energy = 2 * np.real((h @ c - energy * sc) / norm)
    gradient_imaginary = 2 * np.real((anti @ c - imaginary * sc) / norm)
    gradient_h2 = 2 * np.real((t @ c - before['h2'] * sc) / norm)
    gradient_variance = gradient_h2 - 2 * energy * gradient_energy - 2 * imaginary * gradient_imaginary
    gradient = gradient_energy + variance_weight * gradient_variance
    if any(not np.isfinite(g).all() for g in (gradient, gradient_energy, gradient_variance)):
        raise FloatingPointError('nonfinite finite-Q linear objective gradient')
    # Center weighted physical columns directly to avoid cancellation in S/n-ss†/n².
    parent = x @ c
    centered = (x - parent[:, None] * (parent.conj() @ x / norm)[None, :]) / np.sqrt(norm)
    metric = np.real(centered.conj().T @ centered)
    metric = (metric + metric.T) / 2
    values, vectors = np.linalg.eigh(metric)
    largest = max(0., float(values[-1]))
    if np.min(values) < -1e-10 * max(largest, np.finfo(float).tiny):
        raise FloatingPointError('projective Gram is not positive semidefinite')
    roundoff_floor = (32 * np.finfo(float).eps * len(c)
                     * float(np.max(np.diag(s).real)) / norm)
    cutoff = max(largest * metric_rcond, roundoff_floor)
    resolved = np.flatnonzero(values > cutoff)
    kept = resolved if maximum_tangent_rank is None else resolved[-maximum_tangent_rank:]
    basis = vectors[:, kept]
    components = basis.T @ gradient
    direction = -(basis @ (components / (values[kept] + damping * largest)))
    projected_gradient = basis @ components
    gradient_norm = float(np.linalg.norm(gradient))
    scale_null = float(np.linalg.norm(metric @ c) / max(np.linalg.norm(c), np.finfo(float).tiny))
    speed = float(np.sqrt(max(0., direction @ metric @ direction)))
    report = dict(diagnostics, accepted=False, before=before, after=before,
        metric_eigenvalues=values.tolist(), metric_resolved_rank=len(resolved),
        metric_retained_rank=len(kept), maximum_tangent_rank=maximum_tangent_rank,
        metric_rcond=float(metric_rcond), damping=float(damping),
        metric_cutoff=float(cutoff), metric_roundoff_floor=float(roundoff_floor),
        ridge=float(damping * largest), scale_gauge_metric_residual=scale_null,
        gradient_norm=gradient_norm, energy_gradient_norm=float(np.linalg.norm(gradient_energy)),
        variance_gradient_norm=float(np.linalg.norm(gradient_variance)),
        discarded_gradient_fraction=float(np.linalg.norm(gradient-projected_gradient)
                                          / max(gradient_norm, np.finfo(float).tiny)),
        predicted_energy_slope=float(gradient_energy @ direction),
        predicted_variance_slope=float(gradient_variance @ direction),
        predicted_objective_slope=float(gradient @ direction),
        tangent_fs_speed=speed, direction=direction.tolist(),
        variance_weight=float(variance_weight),
        maximum_relative_variance_increase=maximum_relative_variance_increase,
        maximum_fs_sine=float(maximum_fs_sine), initial_step=float(initial_step),
        maximum_backtracks=int(maximum_backtracks), selected_step=0., trials=[],
        fit_only=True, audit_used_for_selection=False, no_new_H_calls=True,
        full_network_stationarity_certified=False, same_fit_norm=True,
        phase_alignment='real sign only; complex overlap retained',
        regularization_coordinates='leading real projective physical-Gram eigenmodes',
    )
    if not len(kept) or not np.isfinite(direction).all() or speed == 0:
        report['status'] = 'no resolved nonzero projective direction'
        return c, report
    old_objective = energy + variance_weight * before['variance']
    for i in range(maximum_backtracks + 1):
        step = initial_step * 2. ** (-i)
        candidate = c + step * direction
        after = _linear_state_moments(candidate, x, y, c)
        objective = after['energy'] + variance_weight * after['variance']
        safe_variance = (maximum_relative_variance_increase is None
                         or after['variance'] <= before['variance'] * (1 + maximum_relative_variance_increase))
        flags = dict(trust=after['fs_sine'] <= maximum_fs_sine,
                     objective_decreases=objective < old_objective,
                     variance_guard=bool(safe_variance))
        report['trials'].append(dict(step=step, **after, **flags))
        if all(flags.values()):
            candidate *= np.sqrt(norm / after['norm_scaled'])
            if np.vdot(parent, x @ candidate).real < 0:
                candidate *= -1
            report.update(accepted=True, selected_step=step,
                          after=_linear_state_moments(candidate, x, y, c),
                          status='one fit-only parent-anchored physical-trust update')
            return candidate, report
    report['status'] = 'no predeclared finite fit step passed; parent unchanged'
    return c, report


__all__ = ['make_projected_linear_products', 'make_terminal_tensor_products',
           'evaluate_linear_state', 'fit_parent_anchored_linear_update']
