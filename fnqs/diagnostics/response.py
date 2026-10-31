"""Separate input differentiation from the physical ground-state response.

At fixed parameters, s = d_gamma log(psi), tau = (s-<s>) psi. The
projected response defect is Q[(H-E) tau + (d_gamma H) psi], with
Q = I-|psi><psi|. A small defect is a diagnostic, not an accuracy bound
without control of the state error and spectral gap. No optimization occurs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from fnqs.basis import validate_size
from fnqs.hamiltonians import IsingCouplings, dense_hamiltonian

LogAmplitude = Callable[[dict, jax.Array, jax.Array], jax.Array]


def response_records(params: dict, bits: jax.Array, couplings: jax.Array,
                     log_amplitude: LogAmplitude,
                     direction: Sequence[float] = (0., 0., 0., 1.),
                     fd_steps: tuple[float, ...] = (1e-3, 5e-4)) -> dict[str, jax.Array]:
    """Born-row estimands for Pauli PBC H and a fixed-theta input JVP.

    Coupling/direction order is (j1,j2,jx,h); bit 0 is Z=-1, site 0 MSB.
    Bits are cast before spin arithmetic, including for uint8 archives.
    Shape validation is JIT-compatible; nonfinite rows are rejected by the
    host summarizer. FD evaluations reuse these rows, without fresh draws.
    """
    bits = jnp.asarray(bits)
    g = jnp.asarray(couplings, dtype=jnp.float64)
    d = np.asarray(direction, dtype=np.float64)
    if bits.ndim != 2 or g.shape != (4,) or d.shape != (4,):
        raise ValueError("expected (batch,n) bits and four coupling/direction entries")
    validate_size(bits.shape[1])
    if not np.isfinite(d).all() or not np.any(d):
        raise ValueError("direction must be finite and nonzero")
    if not fd_steps or any(not np.isfinite(x) or x <= 0 for x in fd_steps):
        raise ValueError("finite-difference steps must be positive and finite")
    d_jax = jnp.asarray(d)
    valid = jnp.all((bits == 0) | (bits == 1), axis=1)
    bits = bits.astype(jnp.int32)
    batch, n = bits.shape
    masks = jnp.eye(n, dtype=jnp.int32)
    masks = jnp.concatenate((masks, masks ^ jnp.roll(masks, 1, axis=1)))
    flipped = (bits[:, None, :] ^ masks[None, :, :]).reshape(-1, n)

    def evaluate(configurations):
        return jax.jvp(lambda x: log_amplitude(params, configurations, x), (g,), (d_jax,))

    logpsi, score = evaluate(bits)
    flip_logpsi, flip_score = evaluate(flipped)
    ratio = jnp.exp(flip_logpsi.reshape(batch, 2*n) - logpsi[:, None])
    flip_score = flip_score.reshape(batch, 2*n)
    spin = 2 * bits - 1
    zz1 = jnp.sum(spin * jnp.roll(spin, -1, axis=1), axis=1)
    zz2 = jnp.sum(spin * jnp.roll(spin, -2, axis=1), axis=1)
    diagonal = -g[0] * zz1 - g[1] * zz2
    coeff = jnp.concatenate((jnp.full(n, -g[3]), jnp.full(n, -g[2])))
    energy = diagonal + jnp.sum(ratio * coeff, axis=1)
    h_score = diagonal * score + jnp.sum(ratio * coeff * flip_score, axis=1)
    source = (-d_jax[0] * zz1 - d_jax[1] * zz2
              - d_jax[3] * ratio[:, :n].sum(axis=1) - d_jax[2] * ratio[:, n:].sum(axis=1))
    fd = jnp.stack([(log_amplitude(params, bits, g+x*d_jax)
                     - log_amplitude(params, bits, g-x*d_jax))/(2*x) for x in fd_steps], axis=1)
    return {"score": jnp.where(valid, score, jnp.nan), "energy": energy,
            "h_score": h_score, "source": source, "fd_scores": fd}


def summarize_response(records: Mapping[str, np.ndarray],
                       weights: np.ndarray | None = None) -> dict:
    """Recenter all moments on the supplied Born rows (or exact weights).

    HF defect = d<E>/d_gamma - <dH/d_gamma>; the extra covariance is retained
    for an inexact neural state. All norms and chi are TOTAL, not per-site.
    The complex formula also removes the unphysical phase/gauge component.
    """
    a = {k: np.asarray(records[k], dtype=np.complex128)
         for k in ("score", "energy", "h_score", "source", "fd_scores")}
    count = a["score"].size
    if count < 2 or any(a[k].shape != (count,) for k in ("score", "energy", "h_score", "source")):
        raise ValueError("at least two scalar Born rows required")
    if a["fd_scores"].ndim != 2 or a["fd_scores"].shape[0] != count or a["fd_scores"].shape[1] < 1:
        raise ValueError("finite-difference scores must be (rows,steps)")
    if not all(np.isfinite(x).all() for x in a.values()):
        raise ValueError("nonfinite diagnostic rows")
    p = np.ones(count)/count if weights is None else np.asarray(weights, dtype=np.float64)
    if p.shape != (count,) or not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("invalid Born weights")
    p = p/p.sum()

    def mean(x):
        return np.einsum("i,i...->...", p, x)

    def norm2(x):
        return float(np.real(mean(abs(x)**2)))

    mu = mean(a["score"])
    s = a["score"]-mu
    energy = float(mean(a["energy"]).real)
    e = a["energy"]-energy
    v = a["source"]-mean(a["source"])
    # Q is essential when psi is not an eigenstate. Recenter before norm.
    defect = a["h_score"]-mu*a["energy"]-energy*s+a["source"]
    defect -= mean(defect)
    chi = norm2(s)
    source_norm = np.sqrt(norm2(v))
    absolute = np.sqrt(norm2(defect))
    hf_defect = float(2*mean(s.conj()*e).real)
    fd = a["fd_scores"]-mean(a["fd_scores"])
    fd_chi = np.real(mean(abs(fd)**2))
    errors = np.sqrt(np.real(mean(abs(fd-s[:, None])**2)))
    return {"chi_total": chi, "energy_total": energy, "energy_variance": norm2(e),
            "state_residual_norm": float(np.sqrt(norm2(e))),
            "source_norm": float(source_norm), "response_residual_norm": float(absolute),
            "response_relative_residual": float(absolute/source_norm) if source_norm > 0 else None,
            "hf_expectation": float(mean(a["source"]).real), "hf_derivative_defect": hf_defect,
            "variational_energy_derivative": float(mean(a["source"]).real)+hf_defect,
            "fd_chi_total": fd_chi.tolist(), "fd_score_rms_error": errors.tolist(),
            "fd_score_relative_rms_error": (errors/np.sqrt(chi)).tolist() if chi > 0 else None}


def bootstrap_response(records: Mapping[str, np.ndarray], seed: int, replicates: int = 200) -> dict:
    """Paired iid-row bootstrap SE; every replica recomputes E, Q and means."""
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("at least two bootstrap replicas required")
    point = summarize_response(records)
    rng = np.random.default_rng(seed)
    metrics = ("chi_total", "response_relative_residual", "hf_derivative_defect")
    if point["response_relative_residual"] is None:
        raise ValueError("response source vanishes; relative bootstrap undefined")
    rows = len(records["score"])
    boot = []
    for _ in range(replicates):
        indices = rng.integers(rows, size=rows)
        summary = summarize_response({k: np.asarray(v)[indices] for k, v in records.items()})
        boot.append([summary[k] for k in metrics])
    point["bootstrap_se"] = dict(zip(metrics, np.std(boot, axis=0, ddof=1).tolist()))
    point["bootstrap_replicates"] = replicates
    return point


@dataclass(frozen=True)
class DenseResponse:
    """Independent full-spectrum response, restricted to even n<=10."""
    energy: float
    state: np.ndarray
    tangent: np.ndarray
    chi_total: float
    residual_norm: float


def dense_response(n: int, couplings: IsingCouplings,
                   direction: Sequence[float] = (0., 0., 0., 1.)) -> DenseResponse:
    """Solve by spectral perturbation, without neural AD or response rows."""
    d = np.asarray(direction, dtype=np.float64)
    if d.shape != (4,) or not np.isfinite(d).all() or not np.any(d):
        raise ValueError("direction must be finite, nonzero and have four entries")
    h = dense_hamiltonian(n, couplings)
    v = dense_hamiltonian(n, IsingCouplings(*d))
    values, vectors = np.linalg.eigh(h)
    if values[1]-values[0] <= 1e-12:
        raise ValueError("nondegenerate ground state required")
    psi = vectors[:, 0].copy()
    psi *= 1 if psi[np.argmax(abs(psi))] >= 0 else -1
    tangent = -vectors[:, 1:] @ ((vectors[:, 1:].T @ (v @ psi))/(values[1:]-values[0]))
    source = v@psi-psi*np.vdot(psi, v@psi)
    residual = (h-values[0]*np.eye(len(psi)))@tangent+source
    return DenseResponse(float(values[0]), psi, tangent, float(np.vdot(tangent, tangent).real),
                         float(np.linalg.norm(residual)))


def tangent_comparison(state: np.ndarray, scores: np.ndarray, reference: DenseResponse) -> dict:
    """Compare normalized state and horizontal tangent with an exact witness."""
    psi = np.asarray(state, dtype=np.complex128)
    s = np.asarray(scores, dtype=np.complex128)
    if psi.shape != reference.state.shape or s.shape != psi.shape or not np.isfinite(psi).all() or not np.isfinite(s).all():
        raise ValueError("state and scores must match the exact basis")
    if np.linalg.norm(psi) <= 0:
        raise ValueError("zero state")
    psi = psi/np.linalg.norm(psi)
    overlap = np.vdot(reference.state, psi)
    psi *= np.exp(-1j*np.angle(overlap))
    tangent = psi*(s-np.sum(abs(psi)**2*s))
    norm = np.linalg.norm(tangent)
    refnorm = np.linalg.norm(reference.tangent)
    return {"state_infidelity": float(max(0., 1-abs(overlap)**2)),
            "tangent_norm_ratio": float(norm/refnorm),
            "tangent_relative_error": float(np.linalg.norm(tangent-reference.tangent)/refnorm),
            "tangent_cosine_real": float(np.vdot(reference.tangent, tangent).real/(refnorm*norm)) if norm > 0 else None}
