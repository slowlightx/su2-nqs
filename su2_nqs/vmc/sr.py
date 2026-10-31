"""Matrix-free stochastic-reconfiguration linear algebra."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np


def importance_amplitude_metric(
    amplitudes: np.ndarray,
    derivatives: np.ndarray,
    proposal_density: np.ndarray,
) -> np.ndarray:
    """Real-parameter ``2*FS`` metric under a positive importance proposal.

    ``derivatives[sample, parameter]`` differentiates the *amplitude*, not its
    logarithm. Thus exact wavefunction nodes remain legitimate samples, with
    their generally nonzero tangent contribution. The proposal is held fixed
    during differentiation; its arbitrary overall scale cancels. No Born or
    Markov-chain effective sample size is implied by this finite-sample metric.
    """
    psi = np.asarray(amplitudes, dtype=np.complex128)
    jac = np.asarray(derivatives, dtype=np.complex128)
    q = np.asarray(proposal_density, dtype=np.float64)
    if (psi.ndim != 1 or not len(psi) or q.shape != psi.shape
            or jac.ndim != 2 or jac.shape[0] != len(psi)
            or not jac.shape[1] or np.any(q <= 0)
            or any(not np.all(np.isfinite(a)) for a in (psi, jac, q))):
        raise ValueError('finite amplitudes/derivatives and positive density required')
    norm = float(np.sum(abs(psi)**2 / q))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError('positive resolved wavefunction norm required')
    scaled = jac / np.sqrt(q * norm)[:, None]
    overlap = np.sum(psi.conj()[:, None] * jac / q[:, None], axis=0) / norm
    metric = 2 * np.real(scaled.conj().T @ scaled - np.outer(overlap.conj(), overlap))
    return .5 * (metric + metric.T)


def importance_hermitian_force(
    amplitudes: np.ndarray,
    derivatives: np.ndarray,
    hamiltonian_actions: np.ndarray,
    proposal_density: np.ndarray,
    *,
    n_sites: int,
) -> np.ndarray:
    """Canonical Hermitian VMC force per site, retaining nodal contributions.

    For Hermitian H the TRUE energy derivative is ``2 Re <dpsi|H-E|psi>``.
    A finite quadrature of ``<psi|Hpsi>`` need not be Hermitian in its sampled
    inner product: differentiating both factors in that finite ratio gives a
    different estimator, not necessarily compatible with the sampled FS
    metric. This function uses the standard centered Hermitian identity.

    The complex empirical mean is used to remove normalization/phase gauges.
    At every finite sample size this force obeys the Cauchy bound associated
    with :func:`importance_amplitude_metric`. Exact nodes require neither
    log derivatives nor a wavefunction floor. H is supplied, not owned here.
    """
    psi=np.asarray(amplitudes,dtype=np.complex128)
    jac=np.asarray(derivatives,dtype=np.complex128)
    b=np.asarray(hamiltonian_actions,dtype=np.complex128)
    q=np.asarray(proposal_density,dtype=np.float64)
    if (isinstance(n_sites,(bool,np.bool_)) or not isinstance(n_sites,(int,np.integer))
            or n_sites<1 or psi.ndim!=1 or not len(psi) or b.shape!=psi.shape
            or q.shape!=psi.shape or jac.ndim!=2 or jac.shape[0]!=len(psi)
            or not jac.shape[1] or np.any(q<=0)
            or any(not np.isfinite(a).all() for a in (psi,jac,b,q))):
        raise ValueError('valid site count, amplitudes, derivatives, Hpsi and positive density required')
    norm=float(np.sum(abs(psi)**2/q))
    if norm<=0 or not np.isfinite(norm):
        raise ValueError('positive resolved wavefunction norm required')
    mean=np.sum(psi.conj()*b/q)/norm
    residual=b-mean*psi
    return 2*np.real(jac.conj().T@(residual/q))/norm/int(n_sites)


def sr_matvec(log_derivative_matrix: jnp.ndarray, vector: jnp.ndarray, damping: float) -> jnp.ndarray:
    """Apply the centered real-parameter quantum Fisher matrix to ``vector``."""

    centered = log_derivative_matrix - jnp.mean(log_derivative_matrix, axis=0, keepdims=True)
    projected = centered @ vector
    fisher_vector = 2.0 * jnp.real(jnp.conj(centered).T @ projected) / centered.shape[0]
    return fisher_vector + damping * vector


def conjugate_gradient(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    rhs: jnp.ndarray,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> tuple[jnp.ndarray, int]:
    """Solve a positive-definite matrix-free system by conjugate gradients."""

    solution = jnp.zeros_like(rhs)
    residual = rhs - matvec(solution)
    direction = residual
    squared_norm = jnp.vdot(residual, residual).real
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        product = matvec(direction)
        alpha = squared_norm / jnp.vdot(direction, product).real
        solution = solution + alpha * direction
        residual = residual - alpha * product
        next_squared_norm = jnp.vdot(residual, residual).real
        if float(jnp.sqrt(next_squared_norm)) <= tolerance:
            break
        direction = residual + (next_squared_norm / squared_norm) * direction
        squared_norm = next_squared_norm
    return solution, iterations
