"""Fixed-q response-enriched multiplicity frames, without a neural head.

Every column belongs to the SAME doubled spin q. Only multiplicity axes are
changed; the associated magnetic multiplet is always retained whole. This
is distinct from the six-site resolvent/interface carrier initializer.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import numpy as np

from ..su2.quantum_numbers import validate_q


@dataclass(frozen=True)
class ResponseFrame:
    """Orthonormal fixed-q frame with protected low-energy columns."""

    q: int
    frame: np.ndarray
    protected_columns: int
    response_columns: int
    fallback_columns: int
    singular_values: np.ndarray
    isometry_error: float


def density_protection_count(probability: float, eigenvalues: np.ndarray,
                             *, maximum_lost_norm: float) -> int:
    """Smallest protected count with ``p(q)*tail <= maximum_lost_norm``.

    Input eigenvalues are the descending, normalized conditional physical
    multiplicity density spectrum of ONE fixed doubled-q sector. Keeping
    these directions bounds lost GLOBAL RAW squared norm, including the
    sector probability. This is not a full-A1 normalized fidelity bound.
    Counts0 and the entire width are meaningful: no protection needed, or
    no remaining capacity for response, respectively.
    """
    values = np.asarray(eigenvalues)
    if (np.iscomplexobj(probability) or not np.isfinite(probability) or not 0 <= probability <= 1
            or np.iscomplexobj(maximum_lost_norm) or not np.isfinite(maximum_lost_norm)
            or not 0 <= maximum_lost_norm < 1):
        raise ValueError("finite sector probability in[0,1] and loss budget in[0,1) required")
    if (values.ndim != 1 or not len(values) or np.iscomplexobj(values)
            or not np.isfinite(values).all() or values.min() < -1e-12
            or np.any(np.diff(values) > 1e-12) or abs(float(values.sum())-1) > 1e-10):
        raise ValueError("descending normalized positive physical density spectrum required")
    # Remove at most allowed exact-PSD contraction/eigh roundoff, not data error.
    values = np.maximum(values, 0.); values /= values.sum()
    for k in range(len(values)+1):
        if float(probability)*float(values[k:].sum()) <= float(maximum_lost_norm):
            return k
    raise ArithmeticError("protecting the entire spectrum should have zero loss")


def _checked_frame(frame: np.ndarray) -> np.ndarray:
    value = np.asarray(frame)
    if value.ndim != 2 or not 0 < value.shape[1] <= value.shape[0] or not np.isfinite(value).all():
        raise ValueError("finite tall fixed-q frame required")
    if np.max(abs(value.conj().T @ value-np.eye(value.shape[1]))) > 1e-10:
        raise ValueError("fixed-q frame must be isometric")
    return value


def interpolate_fixed_q_frame(q: int, old: np.ndarray, proposed: np.ndarray, *,
                              step: float) -> tuple[np.ndarray, dict[str, object]]:
    """Procrustes-align and polar-retract two SAME-q multiplicity subspaces.

    Both frames must use the same physical child-space row coordinates and
    have the same retained width. Independent column gauges are aligned by
    the polar unitary of ``proposed.H @ old``. Nonsingular principal overlap
    is required so that this alignment is unambiguous. This changes only
    multiplicity directions, retaining the complete doubled-q multiplet.

    The interpolation is NOT a statement about a normalized full-A1 state:
    the old physical state must be projected and ancestor gauges absorbed
    separately, and normalized A1 overlaps must be checked independently.
    """
    validate_q(q)
    a = _checked_frame(old); b = _checked_frame(proposed)
    if a.shape != b.shape:
        raise ValueError("same physical rows and same multiplicity width required")
    if (isinstance(step, (bool, np.bool_)) or np.iscomplexobj(step)
            or not np.isfinite(step) or not 0 <= step <= 1):
        raise ValueError("finite interpolation step in[0,1] required")
    u, principal, vh = np.linalg.svd(b.conj().T @ a, full_matrices=False)
    if principal.min() <= 1e-10:
        raise ValueError("singular principal overlap makes Procrustes alignment ambiguous")
    aligned = b @ (u @ vh)
    chord = (1-float(step))*a+float(step)*aligned
    gram = chord.conj().T @ chord
    eigenvalues, axes = np.linalg.eigh((gram+gram.conj().T)/2)
    lower_bound = (1-float(step))**2+float(step)**2
    if eigenvalues.min() < lower_bound-2e-12:
        raise ArithmeticError("aligned chord lost its positive polar Gram bound")
    frame = chord @ ((axes/eigenvalues**.5) @ axes.conj().T)
    if step == 0:
        frame = a.copy()
    error = float(np.max(abs(frame.conj().T @ frame-np.eye(a.shape[1]))))
    if error > 1e-10:
        raise ArithmeticError("polar retraction lost fixed-q isometry")
    def distance(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(max(0., 2*a.shape[1]-2*np.linalg.norm(x.conj().T @ y)**2)))
    return frame, dict(q=int(q), step=float(step), multiplicity_width=a.shape[1],
        physical_rows=a.shape[0], principal_overlaps=principal.tolist(),
        chord_Gram_eigenvalues=eigenvalues.tolist(), chord_Gram_lower_bound=lower_bound,
        frame_Gram_error=error, old_proposed_projector_Frobenius=distance(a, b),
        old_retracted_projector_Frobenius=distance(a, frame),
        column_gauges_aligned_in_common_child_basis=True,
        full_A1_fidelity_bound=False)


def fixed_q_response_frame(q: int, energy_frame: np.ndarray, responses: np.ndarray,
                           *, protect: int, seed: int | None = None) -> ResponseFrame:
    """Protect leading reference columns; fill the SAME width from responses.

    The supplied frame may be energy ordered or physical-density ordered.
    ``responses[:,i]`` must already be a reduced vector in this fixed-q
    source space. It may include square-root density weights. ``seed=None``
    chooses leading response singular directions; an integer instead gives
    a registered random orthogonal-completion control. No H is evaluated.
    Unresolved response-null slots fall back to original energy directions.
    """
    validate_q(q)
    energy = _checked_frame(energy_frame)
    r = np.asarray(responses)
    if r.ndim != 2 or r.shape[0] != len(energy) or not np.isfinite(r).all():
        raise ValueError("finite response columns in the same source space required")
    if isinstance(protect, bool) or not isinstance(protect, Integral) or not 0 <= protect <= energy.shape[1]:
        raise ValueError("protect must lie between zero and the full retained width")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0):
        raise ValueError("random-control seed must be a nonnegative integer")
    if protect == energy.shape[1]:
        return ResponseFrame(q, energy.copy(), int(protect), 0, 0, np.empty(0),
                             float(np.max(abs(energy.conj().T @ energy-np.eye(protect)))))
    dtype = np.result_type(energy, r, np.float64)
    protected = np.asarray(energy[:, :protect], dtype=dtype)
    residual = np.asarray(r, dtype=dtype)-protected @ (protected.conj().T @ r)
    if seed is None:
        candidates, singular, _ = np.linalg.svd(residual, full_matrices=False)
        rank = int(np.sum(singular > max(1e-13, float(singular.max(initial=0))*1e-11)))
        candidates = candidates[:, :rank]
    else:
        rng = np.random.default_rng(seed)
        candidates = rng.normal(size=(len(energy), energy.shape[1]-protect))
        if np.issubdtype(dtype, np.complexfloating):
            candidates = candidates+1j*rng.normal(size=candidates.shape)
        singular = np.linalg.svd(residual, compute_uv=False)
    columns = [protected[:, i].copy() for i in range(protect)]
    response_count = 0
    for source, learned in ((candidates.T, True), (energy[:, protect:].T, False)):
        for candidate in source:
            if len(columns) == energy.shape[1]:
                break
            value = np.asarray(candidate, dtype=dtype).copy()
            for _ in range(2):
                for previous in columns:
                    value -= previous*np.vdot(previous, value)
            norm = np.linalg.norm(value)
            if norm <= 1e-11:
                continue
            value /= norm
            pivot = int(np.argmax(abs(value)))
            value *= np.exp(-1j*np.angle(value[pivot])) if np.iscomplexobj(value) else np.sign(value[pivot])
            columns.append(value)
            response_count += int(learned)
    if len(columns) != energy.shape[1]:
        raise ArithmeticError("response/fallback directions cannot complete the requested width")
    frame = np.column_stack(columns)
    error = float(np.max(abs(frame.conj().T @ frame-np.eye(frame.shape[1]))))
    if error > 1e-10:
        raise ArithmeticError("response enrichment lost isometry")
    return ResponseFrame(q, frame, int(protect), response_count,
                         frame.shape[1]-protect-response_count, singular, error)


def response_capture(frame: np.ndarray, responses: np.ndarray) -> dict[str, float]:
    """Retained squared response norm in a fixed, declared source space."""
    q = _checked_frame(frame); r = np.asarray(responses)
    if r.ndim != 2 or r.shape[0] != len(q) or not np.isfinite(r).all():
        raise ValueError("response shape must agree with the fixed-q source")
    denominator = float(np.vdot(r, r).real)
    numerator = float(np.linalg.norm(q.conj().T @ r)**2)
    if denominator <= 0 or numerator > denominator*(1+1e-10)+1e-13:
        raise ArithmeticError("response denominator is zero or projector is not contractive")
    return dict(numerator=numerator, denominator=denominator, fraction=numerator/denominator)


def correlated_bath_response_factor(
    q: int, responses: np.ndarray, couplings: np.ndarray, bath_gram: np.ndarray,
    *, tolerance: float = 1e-12,
) -> tuple[np.ndarray, dict[str, object]]:
    """Thin SAME-q response factor for a declared PSD correlated bath model.

    ``responses[i,a,s]`` is boundary-site i's response in target multiplicity
    a and source column s. ``couplings[i,j]`` is REAL, and the standard bath
    Gram convention is ``bath_gram[j,k]=<w_j|w_k>``. Tracing the joint columns
    ``sum_ij J[i,j] responses[i,:,s] tensor w_j`` therefore gives

    ``sum_ik (J @ bath_gram.T @ J.T)[i,k] R_i @ R_k.H``.

    The ordinary transpose is essential for complex bath vectors. Return F
    with this covariance F F.H, WITHOUT constructing a target-by-target
    density matrix. Source columns are traced with identity; their possible
    correlations with bath indices are NOT modelled by this scalar bath
    Gram. This is a specified product-bath approximation, not the exact
    conditional environment of an entangled parent, nor an energy guarantee.

    Every target magnetic multiplet stays whole. No additional q+1 or spin
    spherical-to-Cartesian factor is inserted here. The caller must supply
    and independently validate any such physical conversion. Only declared
    relative Hermitian/PSD roundoff is repaired, never a noisy bath RDM.
    """
    validate_q(q)
    if (isinstance(tolerance, (bool, complex)) or not np.isscalar(tolerance)
            or not np.isfinite(tolerance) or not 0 < tolerance <= 1e-8):
        raise ValueError("finite roundoff tolerance in(0,1e-8] required")
    r = np.asarray(responses); j = np.asarray(couplings); g = np.asarray(bath_gram)
    if (r.ndim != 3 or min(r.shape) < 1 or not np.isfinite(r).all()):
        raise ValueError("finite responses[site,target_multiplicity,source] required")
    if (j.ndim != 2 or j.shape[0] != r.shape[0] or j.shape[1] < 1
            or np.iscomplexobj(j) or not np.isfinite(j).all()):
        raise ValueError("finite REAL couplings[site,bathsite] required")
    if g.shape != (j.shape[1], j.shape[1]) or not np.isfinite(g).all():
        raise ValueError("finite square Gram on the declared bath sites required")
    r = np.asarray(r, dtype=np.result_type(r, np.float64))
    j = np.asarray(j, dtype=np.float64)
    g = np.asarray(g, dtype=np.result_type(g, np.float64))
    scale = max(float(np.max(abs(g))), np.finfo(float).tiny)
    hermitian_error = float(np.max(abs(g-g.conj().T)))
    if hermitian_error > tolerance*scale:
        raise ValueError("bath Gram must be Hermitian within declared relative roundoff")
    g = (g+g.conj().T)/2
    values, axes = np.linalg.eigh(g)
    spectral_scale = max(float(np.max(abs(values))), np.finfo(float).tiny)
    if values.min() < -tolerance*spectral_scale:
        raise ValueError("bath Gram has a genuine negative eigenvalue")
    clipped_negative = float(-min(float(values.min()), 0.))
    values = np.maximum(values, 0.)
    # Preserve EVERY strictly positive eigenvalue; numerical rank is merely
    # reported below. No hidden low-rank/capacity selection happens here.
    positive = values > 0
    l = j @ (axes[:, positive].conj()*np.sqrt(values[positive]))
    factor = np.einsum("ias,ir->ars", r, l, optimize=True).reshape(r.shape[1], -1)
    site_kernel = l @ l.conj().T  # Small site matrix, never target covariance.
    kernel_values = np.linalg.eigvalsh((site_kernel+site_kernel.conj().T)/2)
    kernel_scale = max(float(np.max(abs(kernel_values))), np.finfo(float).tiny)
    diagnostics = dict(q=int(q), boundary_sites=r.shape[0], target_multiplicity=r.shape[1],
        source_columns=r.shape[2], bath_sites=j.shape[1], factor_columns=factor.shape[1],
        bath_Gram_trace=float(values.sum()), site_kernel_trace=float(np.linalg.norm(l)**2),
        response_covariance_trace=float(np.linalg.norm(factor)**2),
        bath_Gram_numerical_rank=int(np.sum(values > tolerance*spectral_scale)),
        site_kernel_numerical_rank=int(np.sum(kernel_values > tolerance*kernel_scale)),
        bath_Gram_minimum_input_eigenvalue=float(values.min()-clipped_negative),
        bath_Gram_clipped_negative_roundoff=clipped_negative,
        bath_Gram_Hermitian_absolute_error=hermitian_error,
        Gram_convention="G[j,k]=<w_j|w_k>; K=J @ G.T @ J.T",
        covariance_factor_convention="F[a,(r,s)]=sum_i R[i,a,s] (J U.conj() sqrt(lambda))[i,r]",
        target_covariance_constructed=False, extra_q_plus_one_factor=False,
        spherical_Cartesian_conversion_inserted=False,
        product_bath_model_not_exact_entangled_conditional_environment=True,
        energy_improvement_guaranteed=False)
    return factor, diagnostics


def change_child_coordinates(matrix: np.ndarray, channels: tuple, *,
                             left: dict[int, np.ndarray] | None = None,
                             right: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Apply square per-q child overlaps to a reduced merge coefficient map.

    ``left[q][new,old]`` and its right analogue multiply coefficient axes
    WITHOUT conjugation. All doubled-q labels and channel ordering remain
    unchanged; no magnetic CG convention is modified.
    """
    dtype = np.result_type(matrix, *(left or {}).values(), *(right or {}).values(), np.float64)
    value = np.asarray(matrix, dtype=dtype); result = value.copy()
    if value.ndim != 2 or len(channels) != len(value):
        raise ValueError("one row per declared (q_left,a,q_right,b) channel required")
    for ql, qr in sorted({(x[0], x[2]) for x in channels}):
        ids = [i for i, x in enumerate(channels) if x[0] == ql and x[2] == qr]
        dl = 1+max(channels[i][1] for i in ids); dr = 1+max(channels[i][3] for i in ids)
        a = np.eye(dl) if left is None or ql not in left else np.asarray(left[ql])
        b = np.eye(dr) if right is None or qr not in right else np.asarray(right[qr])
        if a.shape != (dl, dl) or b.shape != (dr, dr):
            raise ValueError("child overlaps must preserve the declared multiplicity width")
        block = value[ids].reshape(dl, dr, -1)
        result[ids] = np.einsum("ai,bj,ijk->abk", a, b, block).reshape(len(ids), -1)
    return result
