"""Local diagnostics of an affine coupling-input path, without training.

The current wrapper only supplies ``b_eff(j) = b + (j-center) w`` to a
pre-LayerNorm residual ViT with final LayerNorm. A common scalar shift of b
or w along the all-ones hidden vector is exactly invisible. Thus at width d
the new w has at most d-1 physical directions; this is an input-coordinate
bound, not the numerical rank of the full neural state family.

The nonlinear backbone may turn that affine input line into a curved state
curve. A scalar j always has one derivative at a point; that observation
alone is NOT evidence of a conditioning bottleneck. Supplied state Jacobians
can distinguish the learned j direction, all patch-bias directions, and the
full-backbone tangent. These are local first-order tests, not convergence or
global representability certificates. No H or model is evaluated here.
"""
from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iufc" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric data")
    return np.asarray(array, dtype=np.complex128 if np.iscomplexobj(array) else np.float64)


def affine_bias_rank_bound(width: int, couplings: Sequence[float], *, center: float = .5) -> dict:
    """Rank of (b,w) -> {b+(j-center)w} after the exact uniform-bias gauge.

    This algebraic bound assumes the documented pre-LN/final-LN backbone.
    It does not assert that every gauge-free bias direction changes its
    physical output. Actual rank can be lower, notably after projection.
    """
    if isinstance(width, bool) or not isinstance(width, Integral) or width < 2:
        raise ValueError("width must be an integer >= 2")
    points = _finite(np.asarray(couplings), "couplings")
    if points.ndim != 1 or not len(points) or np.iscomplexobj(points):
        raise ValueError("couplings must be a nonempty real vector")
    if np.ndim(center) != 0 or np.iscomplexobj(center) or not np.isfinite(center):
        raise ValueError("center must be a finite real scalar")
    design = np.column_stack((np.ones(len(points)), points-float(center)))
    singular = np.linalg.svd(design, compute_uv=False)
    # Distinct represented real inputs determine exact affine identifiability;
    # the singular spectrum separately reports near-degeneracy/conditioning.
    affine_rank = 1 if np.all(points == points[0]) else 2
    physical_bound = affine_rank*(int(width)-1)
    return dict(width=int(width), coupling_count=len(points), affine_design_rank=affine_rank,
                design_singular_values=singular.tolist(), stored_bias_and_slope_count=2*int(width),
                coordinate_rank_before_uniform_gauge=affine_rank*int(width),
                gauge_free_bias_input_rank_upper_bound=physical_bound,
                bias_and_slope_nullity_lower_bound=2*int(width)-physical_bound,
                new_slope_physical_direction_upper_bound=int(width)-1,
                slope_derivative_identically_zero_at_center=bool(np.all(points == center)),
                scope="affine input map after uniform LayerNorm gauge, not full model/state rank")


def ray_projected_jacobian(state: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    """Remove normalization/phase along state, returning normalized ray tangents.

    The rows must use one fixed physical inner product (normally complete
    support). For a sampled metric, premultiply state AND derivatives by the
    same fixed square-root weights before calling; this does not certify
    coverage. Columns are derivatives with respect to REAL parameters.
    """
    psi = _finite(state, "state").astype(np.complex128)
    matrix = _finite(jacobian, "jacobian").astype(np.complex128)
    if psi.ndim != 1 or not len(psi) or matrix.ndim != 2 or matrix.shape[0] != len(psi):
        raise ValueError("state must be a vector and jacobian have one row per amplitude")
    norm = float(np.linalg.norm(psi))
    if norm == 0:
        raise ValueError("state must have positive norm")
    unit = psi/norm
    return (matrix-unit[:, None]*(unit.conj()@matrix)[None, :])/norm


def _realify(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix.real, matrix.imag), axis=0)


def _span(matrix: np.ndarray, tolerance: float, absolute_tolerance: float,
          target: np.ndarray | None) -> tuple[dict, np.ndarray]:
    real = _realify(matrix)
    left, singular, _ = np.linalg.svd(real, full_matrices=False)
    threshold = max(absolute_tolerance, tolerance*float(singular[0])) if len(singular) else absolute_tolerance
    rank = int(np.count_nonzero(singular > threshold))
    basis = left[:, :rank]
    report = dict(parameter_columns=matrix.shape[1], real_tangent_rank=rank,
                  singular_values=singular.tolist(), relative_rank_tolerance=tolerance,
                  absolute_rank_tolerance=absolute_tolerance, rank_threshold=threshold)
    if target is not None:
        vector = _realify(target)
        norm_squared = float(np.vdot(vector, vector).real)
        captured = float(np.linalg.norm(basis.T@vector)**2)
        fraction = min(max(captured/norm_squared, 0.), 1.) if norm_squared > 0 else None
        report.update(target_squared_norm=norm_squared,
                      captured_squared_fraction=fraction,
                      uncaptured_squared_fraction=None if fraction is None else 1-fraction)
    return report, basis


def conditioning_tangent_report(
    state: np.ndarray, bias_jacobian: np.ndarray, coupling_bias: np.ndarray, *,
    backbone_jacobian: np.ndarray | None = None,
    target_derivative: np.ndarray | None = None,
    actual_coupling_derivative: np.ndarray | None = None,
    relative_rank_tolerance: float = 1e-10,
    absolute_rank_tolerance: float = 1e-12,
) -> dict:
    """Compare a learned coupling tangent with available input/body tangents.

    ``bias_jacobian`` differentiates the effective patch bias at the SAME
    physical state and coupling. Optional ``backbone_jacobian`` differentiates
    every old backbone parameter there. Optional target/actual derivatives
    use the same unnormalized amplitude gauge as state. A target might be an
    independently validated response or an energy-descent direction; its
    physical meaning must be supplied by the caller, not inferred here.

    Project real and imaginary rows together: complex-valued wavefunctions
    do not authorize complex neural parameter displacements. This distinction
    matters for claiming that a desired response lies in a neural tangent.
    """
    if not np.isfinite(relative_rank_tolerance) or not 0 < relative_rank_tolerance < 1:
        raise ValueError("relative_rank_tolerance must lie strictly between zero and one")
    if not np.isfinite(absolute_rank_tolerance) or absolute_rank_tolerance < 0:
        raise ValueError("absolute_rank_tolerance must be finite and nonnegative")
    raw = _finite(bias_jacobian, "bias_jacobian")
    bias = _finite(coupling_bias, "coupling_bias")
    if raw.ndim != 2 or bias.shape != (raw.shape[1],) or np.iscomplexobj(bias):
        raise ValueError("coupling_bias must be a real vector matching bias_jacobian columns")
    if len(bias) < 1:
        raise ValueError("bias_jacobian must have at least one column")
    tangents = ray_projected_jacobian(state, raw)
    learned = tangents@bias

    def project_vector(vector, name):
        array = _finite(vector, name)
        if array.shape != (raw.shape[0],):
            raise ValueError(f"{name} must match the state vector")
        return ray_projected_jacobian(state, array[:, None])[:, 0]

    target = None if target_derivative is None else project_vector(target_derivative, "target_derivative")
    single, _ = _span(learned[:, None], relative_rank_tolerance, absolute_rank_tolerance, target)
    inputs, _ = _span(tangents, relative_rank_tolerance, absolute_rank_tolerance, target)
    result = dict(
        learned_coupling_tangent=single, all_patch_bias_tangents=inputs,
        learned_fidelity_susceptibility=float(np.linalg.norm(learned)**2),
        uniform_bias_tangent_norm=float(np.linalg.norm(tangents@np.ones(len(bias)))),
        coupling_bias_uniform_component=float(np.mean(bias)),
        coupling_bias_centered_norm=float(np.linalg.norm(bias-np.mean(bias))),
        new_samples=0, new_H_calls=0, training_steps=0,
        global_conditioning_failure_certified=False, global_backbone_failure_certified=False,
        scope="same-point real-parameter ray tangent diagnostic; not a global expressivity or response accuracy certificate")
    if actual_coupling_derivative is not None:
        actual = project_vector(actual_coupling_derivative, "actual_coupling_derivative")
        error = float(np.linalg.norm(actual-learned))
        denominator = max(float(np.linalg.norm(actual)), float(np.linalg.norm(learned)))
        result["coupling_chain_rule_absolute_error"] = error
        result["coupling_chain_rule_relative_error"] = error/denominator if denominator > 0 else None
    if backbone_jacobian is not None:
        body = ray_projected_jacobian(state, backbone_jacobian)
        report, span = _span(body, relative_rank_tolerance, absolute_rank_tolerance, target)
        real_bias = _realify(tangents)
        remainder = real_bias-span@(span.T@real_bias)
        denominator = float(np.linalg.norm(real_bias))
        report["bias_tangent_outside_backbone_fraction"] = (
            float(np.linalg.norm(remainder))/denominator if denominator > 0 else None)
        result["full_backbone_tangents"] = report
    return result


__all__ = ["affine_bias_rank_bound", "ray_projected_jacobian", "conditioning_tangent_report"]
