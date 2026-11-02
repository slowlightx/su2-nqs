"""Offline covariant operator filters of inherited multiplicity columns.

The learned objects are scalar filter rules, never magnetic-index tensors or
configuration-dependent wavefunction heads.  A caller supplies a Hermitian
operator and a transported isometric seed.  Changing the physical/reduced
gauge rotates the output subspace covariantly.  Seed selection at a degenerate
cutoff is a separate, explicitly caller-owned choice.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


def occupied_seed_rotations(coefficients: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Leading left/right ket Schmidt frames of C_ab, with whole q copies.

    C multiplies a normalized singlet-channel basis. Its sector weight is
    sum|C|², with NO extra (q+1) factor. The right ket density is C.T C.conj(),
    not C†C. An exactly empty sector retains deterministic legal columns.
    """
    c = np.asarray(coefficients)
    if (c.ndim != 2 or not np.isfinite(c).all() or isinstance(width,bool)
            or not isinstance(width,Integral) or not 1 <= width <= min(c.shape)):
        raise ValueError('finite root coefficient matrix and legal positive retained width required')
    weight = float(np.vdot(c,c).real); frames = []; gaps = []; masses = []
    for density in (c @ c.conj().T, c.T @ c.conj()):
        values, vectors = np.linalg.eigh((density+density.conj().T)/2)
        order = np.argsort(-values,kind='stable'); values = values[order]; vectors = vectors[:,order]
        frames.append(np.eye(len(density),dtype=c.dtype)[:,:width] if weight == 0 else vectors[:,:width])
        masses.append(float(values[:width].sum()))
        gaps.append(float(values[width-1]-values[width]) if width < len(values) else None)
    return frames[0],frames[1],dict(sector_weight=weight, retained_weight_left=masses[0],retained_weight_right=masses[1],
        cutoff_gaps=gaps, exactly_zero_sector=weight == 0,
        cutoff_degenerate=[gap is not None and gap <= 1e-11*max(weight,1e-300) for gap in gaps],
        right_density_convention='C.T @ C.conj()', no_extra_magnetic_factor=True)


@dataclass(frozen=True)
class OperatorFilterSketch:
    """Four covariant operator-word blocks in one legal doubled-q sector."""

    basis: np.ndarray
    gram: np.ndarray
    energy: np.ndarray
    seed: np.ndarray
    center: float
    radius: float
    H_columns: int
    coefficient_bound: float = .3
    minimum_gram_bound: float = .01


def build_boundary_residual_sketch(action: Callable[[np.ndarray], np.ndarray],
                                  internal_action: Callable[[np.ndarray], np.ndarray],
                                  seed: np.ndarray, internal_lower: float,
                                  internal_upper: float, cross_bound: float) -> OperatorFilterSketch:
    """Boundary/gap/two-exchange response directions, orthogonal to the seed.

    With norm-bounded A=normalized internal H and B=H_cross/cross_bound, use
    D=(Q_perp B V, Q_perp[A,B]V/2, Q_perp B²V). Each block has norm <=1 and
    V†D=0. Consequently (V+D a)†(V+D a)>=I for ANY scalar coefficients;
    |a_i|<=2 additionally bounds its condition number by37. These are legal
    multiplicity-space vectors, not magnetic parameters. Five shared full-H
    block actions suffice; internal child-H applications are separate work.
    """
    v = np.asarray(seed, dtype=np.result_type(seed, np.float64))
    if (v.ndim != 2 or not 0 < v.shape[1] <= v.shape[0] or not np.isfinite(v).all()
            or not np.isfinite([internal_lower, internal_upper, cross_bound]).all()
            or internal_lower >= internal_upper or cross_bound <= 0):
        raise ValueError('finite isometric seed and certified nonzero spectral widths required')
    if np.max(abs(v.conj().T @ v-np.eye(v.shape[1]))) > 2e-10:
        raise ValueError('seed columns must be isometric')
    c, r = (internal_upper+internal_lower)/2, (internal_upper-internal_lower)/2
    def columns(fn, matrix):
        out = np.column_stack([fn(x) for x in matrix.T])
        if out.shape != matrix.shape or not np.isfinite(out).all():
            raise ValueError('operator returned invalid columns')
        return out
    h0v, hv = columns(internal_action, v), columns(action, v)
    av = (h0v-c*v)/r
    bv = (hv-h0v)/cross_bound
    h0av, hav = columns(internal_action, av), columns(action, av)
    h0bv, hbv = columns(internal_action, bv), columns(action, bv)
    commutator = ((h0bv-c*bv)/r-(hav-h0av)/cross_bound)/2
    twice = (hbv-h0bv)/cross_bound
    directions = []
    for raw in (bv, commutator, twice):
        d = raw-v @ (v.conj().T @ raw)
        d = d-v @ (v.conj().T @ d)
        directions.append(d)
    p = np.column_stack((v, *directions))
    hd1 = hbv-hv @ (v.conj().T @ bv)
    hp = np.column_stack((hv, hd1, columns(action, directions[1]), columns(action, directions[2])))
    radius = r+cross_bound
    gram, energy = p.conj().T @ p, p.conj().T @ ((hp-c*p)/radius)
    if (np.max(abs(energy-energy.conj().T)) > 3e-10*max(1., float(np.max(abs(energy))))
            or np.max(abs(gram[:v.shape[1], v.shape[1]:])) > 2e-10):
        raise ValueError('residual sketch violates Hermiticity/seed orthogonality')
    return OperatorFilterSketch(p, (gram+gram.conj().T)/2,
        (energy+energy.conj().T)/2, v, float(c), float(radius), 5*v.shape[1], 2., 1.)


def build_filter_sketch(action: Callable[[np.ndarray], np.ndarray], seed: np.ndarray,
                        lower: float, upper: float) -> OperatorFilterSketch:
    """Cache [V,hV,h²V,h³V] and its exact Gram/energy moments.

    ``lower,upper`` must bound the supplied Hermitian operator. With
    h=(H-center)/radius its spectrum is in [-1,1]. Computing the complete
    energy matrix needs h⁴V, hence exactly four times seed-width H columns.
    No dense Hamiltonian, symbolic work or magnetic tensors are constructed.
    """
    v = np.asarray(seed, dtype=np.result_type(seed, np.float64))
    if (v.ndim != 2 or not 0 < v.shape[1] <= v.shape[0] or not np.isfinite(v).all()
            or not np.isfinite([lower, upper]).all() or lower >= upper):
        raise ValueError('finite isometric seed and strictly ordered spectral bounds required')
    if np.max(abs(v.conj().T @ v-np.eye(v.shape[1]))) > 2e-10:
        raise ValueError('seed columns must be isometric')
    center, radius = (upper+lower)/2, (upper-lower)/2
    powers = [v]
    for _ in range(4):
        acted = np.column_stack([action(x) for x in powers[-1].T])
        if acted.shape != v.shape or not np.isfinite(acted).all():
            raise ValueError('operator action returned invalid columns')
        powers.append((acted-center*powers[-1])/radius)
    p, hp = np.column_stack(powers[:4]), np.column_stack(powers[1:])
    gram, energy = p.conj().T @ p, p.conj().T @ hp
    defect = float(np.max(abs(energy-energy.conj().T)))
    if defect > 3e-10*max(1., float(np.max(abs(energy)))):
        raise ValueError('cached operator moments are not Hermitian')
    return OperatorFilterSketch(p, (gram+gram.conj().T)/2,
        (energy+energy.conj().T)/2, v, float(center), float(radius), 4*v.shape[1])


def filter_coefficients_matrix(coefficients: jax.Array, width: int) -> jax.Array:
    """Block column [I,a1 I,a2 I,a3 I], with no learned magnetic entries."""
    a = jnp.concatenate((jnp.ones(1, dtype=coefficients.dtype), coefficients))
    return (a[:, None, None]*jnp.eye(width, dtype=a.dtype)).reshape(4*width, width)


def filter_trace_energy(gram: jax.Array, energy: jax.Array,
                        coefficients: jax.Array) -> jax.Array:
    """Gauge-invariant mean Ritz trace using the TRUE unregularized Gram."""
    width = gram.shape[-1]//4
    c = filter_coefficients_matrix(coefficients, width)
    overlap, projected = c.conj().T @ gram @ c, c.conj().T @ energy @ c
    return jnp.trace(jnp.linalg.solve(overlap, projected)).real/width


def filtered_columns(sketch: OperatorFilterSketch,
                     coefficients: np.ndarray) -> tuple[np.ndarray, dict]:
    """Apply a certified nonsingular scalar filter and return whole multiplets.

    The certificate belongs to the sketch. Polynomial/Hermitian-word sketches
    use |a_i|<=.3 and Gram>=.01. Orthogonal residual-word sketches instead use
    |a_i|<=2 and Gram>=I; they need not be Hermitian filters. Invalid or
    rank-deficient input is rejected, never completed with invented directions.
    """
    a = np.asarray(coefficients)
    if a.shape != (3,) or np.iscomplexobj(a) or not np.isfinite(a).all() or np.max(abs(a)) > sketch.coefficient_bound+1e-12:
        raise ValueError('three finite real coefficients within the sketch certificate required')
    k = sketch.seed.shape[1]
    c = np.concatenate([x*np.eye(k) for x in np.r_[1., a]])
    raw = sketch.basis @ c
    eigen = np.linalg.eigvalsh(raw.conj().T @ raw)
    if eigen[0] < sketch.minimum_gram_bound-2e-9:
        raise ArithmeticError('spectral-bound/rank certificate failed')
    w = sketch.seed.copy() if not np.any(a) else np.linalg.qr(raw, mode='reduced')[0]
    value = float(np.trace(np.linalg.solve(c.conj().T @ sketch.gram @ c,
                                         c.conj().T @ sketch.energy @ c)).real/k)
    return w, dict(minimum_filter_Gram_eigenvalue=float(eigen[0]),
        filter_Gram_condition=float(eigen[-1]/eigen[0]), normalized_trace_energy=value,
        maximum_isometry_error=float(np.max(abs(w.conj().T @ w-np.eye(k)))),
        raw_trace_energy=sketch.center+sketch.radius*value)


def sketch_ritz_columns(sketch: OperatorFilterSketch) -> tuple[np.ndarray, dict]:
    """Strong nonattention control: exact rank-k Ritz in the identical sketch."""
    u, s, vh = np.linalg.svd(sketch.basis, full_matrices=False)
    keep = s > max(s[0]*2e-11, 1e-13)
    k = sketch.seed.shape[1]
    if keep.sum() < k:
        raise ArithmeticError('operator sketch lost its inherited seed rank')
    transform = vh[keep].conj().T/s[keep]
    h = transform.conj().T @ sketch.energy @ transform
    values, vectors = np.linalg.eigh((h+h.conj().T)/2)
    w = u[:, keep] @ vectors[:, :k]
    return w, dict(sketch_rank=int(keep.sum()), normalized_trace_energy=float(values[:k].mean()),
        raw_trace_energy=float(sketch.center+sketch.radius*values[:k].mean()),
        maximum_isometry_error=float(np.max(abs(w.conj().T @ w-np.eye(k)))))


def init_filter_controller(key: jax.Array, features: int, kind: str) -> dict:
    """One shared offline controller; all three arms start at zero filter."""
    if kind not in ('constant', 'linear', 'mlp', 'attention') or features < 1:
        raise ValueError('known controller and positive feature count required')
    if kind == 'constant':
        return dict(bias=jnp.zeros(3))
    keys = jax.random.split(key, 5)
    if kind == 'linear':
        return dict(out=jnp.zeros((features, 3)), bias=jnp.zeros(3))
    # At seventeen input features this gives 1095 MLP versus 1107 attention
    # parameters, while their physical output rank/sketch are identical.
    width = 52 if kind == 'mlp' else 16
    p = dict(embed=jax.random.normal(keys[0], (features, width))/np.sqrt(features),
             eb=jnp.zeros(width), out=jnp.zeros((width, 3)), bias=jnp.zeros(3))
    if kind == 'attention':
        p.update({name: jax.random.normal(k, (width, width))/np.sqrt(width)
                  for name, k in zip(('query', 'key', 'value'), keys[1:4], strict=True)})
    return p


def apply_filter_controller(params: dict, tokens: jax.Array, kind: str, *, coefficient_bound: float = .3) -> jax.Array:
    """Map scalar q/environment tokens to three rank-safe filter coefficients."""
    if kind == 'constant':
        return jnp.broadcast_to(coefficient_bound*jnp.tanh(params['bias']), (tokens.shape[0], 3))
    if kind == 'linear':
        hidden = tokens
    elif kind in ('mlp', 'attention'):
        hidden = jnp.tanh(tokens @ params['embed']+params['eb'])
        if kind == 'attention':
            score = (hidden @ params['query']) @ (hidden @ params['key']).T/4
            weights = jax.nn.softmax(score, axis=-1)
            hidden = jnp.tanh(hidden+weights @ (hidden @ params['value']))
    else:
        raise ValueError('unknown controller')
    return coefficient_bound*jnp.tanh(hidden @ params['out']+params['bias'])
