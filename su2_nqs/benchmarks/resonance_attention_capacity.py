"""Necessary table-capacity bounds for a narrow scalar-attention output head.

The current real attention raw table has the form F=H W+1 b, with hidden
width d and ten scalar generator outputs. Across its36 contexts, a centered
F has rank at most d. For an anchored generator G(theta)=F(theta)-F0, the
correct target matrix is F0+G_teacher, *not* G_teacher alone.

An SVD tail is an exact necessary lower bound on table-fitting error for this
output head. It is not an achievability certificate for the nonlinear trunk,
not a lower bound on physical state error, and not a variational-energy bound:
different operator tables can act identically on a fixed projected carrier.
The module only analyzes supplied arrays; it never samples, trains, or calls H.
"""
from __future__ import annotations

from numbers import Integral
from collections.abc import Sequence

import numpy as np

from ..models.conditional_carrier_resonance import local_scalar_components


def _real_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f'{name} must be real numeric data')
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite')
    return array


def _weights(value: np.ndarray | None, count: int, name: str) -> np.ndarray:
    weights = np.ones(count, dtype=np.float64) if value is None else _real_array(value, name)
    if weights.shape != (count,) or np.any(weights <= 0):
        raise ValueError(f'{name} must contain one strictly positive weight per coordinate')
    return weights


def scalar_generator_hs_weights() -> tuple[np.ndarray, np.ndarray]:
    """Hilbert-Schmidt weights for the two-environment real generator table.

    A complete (q,a) projector has trace/norm-squared q+1. The canonical
    symmetric active generator units have squared norm q_active+1, including
    their normalized off-diagonal units. Thus row(z,w) has weight
    (q_z+1)(q_w+1), column r weight q_active(r)+1. All q are doubled integers.
    These weights concern the local GENERATOR, not exp(G) or its physical state.
    """
    parts = local_scalar_components()
    dimensions = np.asarray([q+1 for q, _ in parts.channels], dtype=np.float64)
    rows = np.outer(dimensions, dimensions).reshape(-1)
    columns = np.asarray([q+1 for q, _, _ in parts.generator_labels], dtype=np.float64)
    rows.setflags(write=False); columns.setflags(write=False)
    return rows, columns


def head_rank_certificate(
    reference_raw: np.ndarray,
    teacher_generator: np.ndarray,
    *,
    head_width: int = 8,
    row_weights: np.ndarray | None = None,
    column_weights: np.ndarray | None = None,
    include_best_table: bool = False,
) -> dict:
    """Necessary weighted table-error bound, preserving the original anchor.

    Inputs share shape (...,output_count); every leading axis is a context.
    The freely adjustable output bias fits the weighted context mean, so the
    rank test uses sqrt(Wrow) [F0+G_teacher-weighted_mean] sqrt(Wcolumn).
    A common scalar-identity gauge shift, or any common output-vector shift,
    disappears on centering. The absolute tail bound therefore survives the
    physical global-ray gauge. Relative-to-uncentered-teacher normalization
    need not be gauge invariant and is labeled separately.

    The optional best table is the optimal rank-limited *matrix* relaxation;
    it need not be realizable by the attention trunk. No fitting is performed.
    """
    reference = _real_array(reference_raw, 'reference_raw')
    target = _real_array(teacher_generator, 'teacher_generator')
    if reference.shape != target.shape or target.ndim < 2 or target.size == 0:
        raise ValueError('reference and teacher need the same nonempty context/output shape')
    if isinstance(head_width, bool) or not isinstance(head_width, Integral) or head_width < 1:
        raise ValueError('head_width must be a positive integer')
    outputs = target.shape[-1]
    if outputs < 1:
        raise ValueError('at least one output coordinate is required')
    t = target.reshape(-1, outputs); f0 = reference.reshape(t.shape)
    contexts = len(t)
    wr = _weights(row_weights, contexts, 'row_weights')
    wc = _weights(column_weights, outputs, 'column_weights')
    sr, sc = np.sqrt(wr), np.sqrt(wc)
    raw = f0+t
    mean = np.sum(wr[:, None]*raw, axis=0)/wr.sum()
    centered = raw-mean
    weighted = sr[:, None]*centered*sc[None, :]
    left, singular, right = np.linalg.svd(weighted, full_matrices=False)
    rank_limit = min(int(head_width), contexts-1, outputs)
    tail = float(np.linalg.norm(singular[rank_limit:]))
    target_norm = float(np.linalg.norm(sr[:, None]*t*sc[None, :]))
    target_mean = np.sum(wr[:, None]*t, axis=0)/wr.sum()
    centered_target_norm = float(np.linalg.norm(sr[:, None]*(t-target_mean)*sc[None, :]))
    # The real rank-r matrix variety after removing the context mean, plus
    # an unconstrained output bias. This is an upper bound, not a proof the
    # nonlinear trunk reaches the entire variety or that a fit will converge.
    tangent_bound = rank_limit*((contexts-1)+outputs-rank_limit)+outputs
    result = dict(
        context_count=contexts, output_count=outputs, head_width=int(head_width),
        context_centered_raw_rank_limit=rank_limit,
        table_jacobian_rank_upper_bound=tangent_bound,
        weighted_singular_values=singular.tolist(),
        weighted_frobenius_error_lower_bound=tail,
        weighted_mean_square_error_lower_bound=tail**2/float(wr.sum()*wc.sum()),
        weighted_rms_error_lower_bound=tail/np.sqrt(float(wr.sum()*wc.sum())),
        teacher_weighted_frobenius_norm=target_norm,
        centered_teacher_weighted_frobenius_norm=centered_target_norm,
        relative_to_teacher_norm=tail/target_norm if target_norm > 0 else None,
        relative_to_centered_teacher_norm=tail/centered_target_norm if centered_target_norm > 0 else None,
        weighted_raw_mean=mean.tolist(), row_weights=wr.tolist(), column_weights=wc.tolist(),
        original_anchor_preserved=True, common_output_shift_cannot_remove_tail=True,
        bound_scope='necessary raw-table output-head bound; not physical state, energy, or optimization convergence',
        nonlinear_trunk_achievability_certified=False,
        physical_state_replay_excluded=False,
        new_samples=0, new_H_calls=0, training_steps=0,
    )
    if include_best_table:
        relaxation = ((left[:, :rank_limit]*singular[:rank_limit])@right[:rank_limit]
                      /sr[:, None]/sc[None, :]+mean)
        result['best_rank_limited_generator_table'] = (relaxation-f0).reshape(target.shape).tolist()
        result['best_table_scope'] = 'matrix relaxation only; not asserted to be a neural output'
    return result


def jacobian_rank_report(
    jacobian: np.ndarray,
    *,
    relative_tolerances: Sequence[float] = (1e-4, 1e-6, 1e-8, 1e-10, 1e-12),
    target_deltas: dict[str, np.ndarray] | None = None,
    projection_relative_tolerance: float = 1e-10,
) -> dict:
    """Diagnose a supplied real table Jacobian, with explicit SVD thresholds.

    The Jacobian is (flattened table coordinates, real parameters). Optional
    target increments are projected onto its finite-threshold column space.
    A missing local tangent direction may become reachable after nonlinear
    motion; neither a low numerical rank nor a residual is a global no-go.
    This helper does not differentiate or evaluate a neural model itself.
    """
    matrix = _real_array(jacobian, 'jacobian')
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError('jacobian must be a nonempty matrix')
    tolerances = tuple(float(value) for value in relative_tolerances)
    if (not tolerances or any(not np.isfinite(t) or not 0 < t < 1 for t in tolerances)
            or not np.isfinite(projection_relative_tolerance)
            or not 0 < projection_relative_tolerance < 1):
        raise ValueError('relative SVD thresholds must lie strictly between zero and one')
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    largest = float(singular[0])
    rank = int(np.count_nonzero(singular > largest*projection_relative_tolerance))
    targets = {}
    for name, delta in (target_deltas or {}).items():
        vector = _real_array(delta, f'target delta {name}').reshape(-1)
        if vector.shape != (matrix.shape[0],):
            raise ValueError('each target delta must match the table coordinate count')
        remainder = vector-left[:, :rank]@(left[:, :rank].T@vector)
        norm = float(np.linalg.norm(vector)); error = float(np.linalg.norm(remainder))
        targets[name] = dict(target_norm=norm, linearized_unreachable_norm=error,
                             linearized_unreachable_fraction=error/norm if norm > 0 else None)
    return dict(shape=list(matrix.shape), singular_values=singular.tolist(),
        largest_singular_value=largest,
        ranks_by_relative_tolerance={str(t):int(np.count_nonzero(singular > largest*t)) for t in tolerances},
        projection_relative_tolerance=projection_relative_tolerance,
        projection_rank=rank, target_projections=targets,
        interpretation='local, coordinate-dependent conditioning/tangent diagnostic; not a global expressivity certificate',
        new_samples=0, new_H_calls=0, training_steps=0)


__all__ = ['scalar_generator_hs_weights', 'head_rank_certificate', 'jacobian_rank_report']
