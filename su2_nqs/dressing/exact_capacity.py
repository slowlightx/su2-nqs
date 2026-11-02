"""Exact-small local-intertwiner fidelity and tangent capacity oracles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from .local_schur import FOUR_SPIN_MULTIPLICITIES, FOUR_SPIN_TOTAL_Q, FourSpinSchurBasis, four_spin_schur_basis


def schur_transition_blocks(
    transition: np.ndarray,
    *,
    basis: FourSpinSchurBasis | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """Reduce an SU(2)-scalar four-spin transition matrix by magnetic index.

    The returned block for doubled spin ``q`` is the sum of its identical
    magnetic slices.  Consequently a multiplicity operator ``O_q``
    contributes ``trace(O_q @ block_q)`` to the global transition overlap.
    """

    value = np.asarray(transition, dtype=np.complex128)
    if value.shape != (16, 16) or np.any(~np.isfinite(value)):
        raise ValueError("transition must be a finite 16x16 matrix")
    selected = four_spin_schur_basis() if basis is None else basis
    schur = selected.columns.conj().T @ value @ selected.columns
    reconstructed = np.zeros_like(schur)
    blocks: dict[int, np.ndarray] = {}
    maximum_m_slice_error = 0.0
    for q in FOUR_SPIN_TOTAL_Q:
        slices = []
        for m in range(-q, q + 1, 2):
            indices = selected.indices(q, m=m)
            block = schur[np.ix_(indices, indices)]
            slices.append(block)
            reconstructed[np.ix_(indices, indices)] = block
        mean = sum(slices) / len(slices)
        maximum_m_slice_error = max(
            maximum_m_slice_error,
            max(float(np.max(np.abs(block - mean), initial=0.0)) for block in slices),
        )
        blocks[q] = sum(slices)
    denominator = max(float(np.linalg.norm(schur)), 1.0e-300)
    return blocks, {
        "relative_non_scalar_block_norm": float(
            np.linalg.norm(schur - reconstructed) / denominator
        ),
        "maximum_m_slice_error": maximum_m_slice_error,
    }


def _orthogonal_support(matrix: np.ndarray, determinant: int) -> float:
    """Return ``max_{det(O)=determinant} trace(O @ matrix)`` for real O."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("orthogonal Procrustes matrix must be square")
    if determinant not in (-1, 1):
        raise ValueError("orthogonal determinant must be -1 or +1")
    if value.shape == (1, 1):
        return float(determinant * value[0, 0])
    left, singular, right_h = np.linalg.svd(value.T)
    orientation = float(np.linalg.det(left @ right_h))
    return float(np.sum(singular[:-1]) + determinant * orientation * singular[-1])


def real_multiplicity_overlap_ceiling(
    transition: np.ndarray,
    *,
    active_q: Sequence[int] = (0, 2),
    determinants: Mapping[int, int] | None = None,
    q4_sign: int = 1,
    phase_grid_size: int = 192,
    basis: FourSpinSchurBasis | None = None,
) -> dict[str, object]:
    """Maximize overlap over selected real SU(2) multiplicity blocks.

    ``active_q=(0,2)`` with determinants ``+1,+1`` is the continuous
    ``SO(2) x SO(3)`` PlaquetteDress family.  Inactive sectors remain the
    identity, while ``q4_sign`` optionally enumerates the discrete O(1)
    component.  A one-dimensional phase maximization converts the absolute
    complex overlap into real special-orthogonal Procrustes problems.
    """

    active = tuple(int(q) for q in active_q)
    if len(set(active)) != len(active) or any(q not in FOUR_SPIN_TOTAL_Q for q in active):
        raise ValueError("active_q must contain distinct q=0,2,4 sectors")
    if q4_sign not in (-1, 1):
        raise ValueError("q4_sign must be -1 or +1")
    if phase_grid_size < 32:
        raise ValueError("phase_grid_size must be at least 32")
    determinant_map = {q: 1 for q in active}
    if determinants is not None:
        determinant_map.update({int(q): int(value) for q, value in determinants.items()})
    if any(determinant_map[q] not in (-1, 1) for q in active):
        raise ValueError("every active determinant must be -1 or +1")
    blocks, scalar_audit = schur_transition_blocks(transition, basis=basis)

    def support(phase: float) -> float:
        rotation = np.exp(-1.0j * phase)
        result = 0.0
        for q in FOUR_SPIN_TOTAL_Q:
            sign = q4_sign if q == 4 else 1
            real_block = np.real(rotation * sign * blocks[q])
            if q in active:
                result += _orthogonal_support(real_block, determinant_map[q])
            else:
                result += float(np.trace(real_block))
        return result

    try:
        from scipy.optimize import minimize_scalar
    except ImportError as error:
        raise RuntimeError("exact capacity phase optimization requires scipy") from error

    phases = np.linspace(0.0, 2.0 * math.pi, phase_grid_size, endpoint=False)
    samples = np.asarray([support(float(phase)) for phase in phases])
    spacing = 2.0 * math.pi / phase_grid_size
    candidate_indices = np.argsort(samples)[-min(8, phase_grid_size) :]
    best_value = float(np.max(samples))
    best_phase = float(phases[int(np.argmax(samples))])
    for index in candidate_indices:
        center = float(phases[int(index)])
        optimized = minimize_scalar(
            lambda phase: -support(float(phase) % (2.0 * math.pi)),
            bounds=(center - spacing, center + spacing),
            method="bounded",
            options={"xatol": 1.0e-13},
        )
        value = -float(optimized.fun)
        if value > best_value:
            best_value = value
            best_phase = float(optimized.x % (2.0 * math.pi))
    overlap = min(max(best_value, 0.0), 1.0 + 1.0e-10)
    return {
        "maximum_overlap": overlap,
        "maximum_fidelity": float(min(overlap**2, 1.0 + 2.0e-10)),
        "optimizing_phase": best_phase,
        "active_q": list(active),
        "determinants": {str(q): determinant_map[q] for q in active},
        "q4_sign": q4_sign,
        "phase_grid_size": int(phase_grid_size),
        **scalar_audit,
    }


def multiplicity_component_atlas(
    transition: np.ndarray,
    *,
    phase_grid_size: int = 192,
    basis: FourSpinSchurBasis | None = None,
) -> dict[str, object]:
    """Return connected, sector-ablated, and all real O-component ceilings."""

    selected = four_spin_schur_basis() if basis is None else basis
    identity = real_multiplicity_overlap_ceiling(
        transition,
        active_q=(),
        phase_grid_size=phase_grid_size,
        basis=selected,
    )
    q0 = real_multiplicity_overlap_ceiling(
        transition,
        active_q=(0,),
        phase_grid_size=phase_grid_size,
        basis=selected,
    )
    q2 = real_multiplicity_overlap_ceiling(
        transition,
        active_q=(2,),
        phase_grid_size=phase_grid_size,
        basis=selected,
    )
    components = {}
    for determinant_q0 in (-1, 1):
        for determinant_q2 in (-1, 1):
            name = f"det_q0_{determinant_q0:+d}_det_q2_{determinant_q2:+d}"
            components[name] = real_multiplicity_overlap_ceiling(
                transition,
                active_q=(0, 2),
                determinants={0: determinant_q0, 2: determinant_q2},
                phase_grid_size=phase_grid_size,
                basis=selected,
            )
    connected = components["det_q0_+1_det_q2_+1"]
    best_name = max(
        components, key=lambda name: float(components[name]["maximum_fidelity"])
    )
    return {
        "identity": identity,
        "q0_only": q0,
        "q2_only": q2,
        "connected_SO2xSO3": connected,
        "orthogonal_components": components,
        "best_orthogonal_component": best_name,
        "best_orthogonal_fidelity": components[best_name]["maximum_fidelity"],
    }


__all__ = [
    "multiplicity_component_atlas",
    "real_multiplicity_overlap_ceiling",
    "schur_transition_blocks",
]
