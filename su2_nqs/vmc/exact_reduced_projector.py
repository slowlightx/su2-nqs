"""Exact-small coherent space-group projection and its actual Born sampler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..benchmarks.clebsch_tree_exact import physical_permutation_index_table
from ..lattices.spatial_symmetry import validated_permutation_group


@dataclass(frozen=True)
class ExactA1OrbitProjector:
    """Differentiable exact-small ``A1`` projector ``P = B† B``.

    ``B`` maps reduced coefficients to normalized physical orbit sums. It
    replaces repeated Reynolds gathers, NOT the complete reduced basis.
    The supplied isometry determines doubled ``q,m``; this object adds only
    the trivial spatial irrep. It never turns native AR samples into samples
    of the projected state. Construction enumerates physical support and is
    deliberately refused above 16 sites; it is NOT an N36 implementation.
    """

    labels: np.ndarray
    factor: np.ndarray
    orbit_index: np.ndarray
    orbit_sizes: np.ndarray
    group_order: int
    gram_error: float
    closure_error: float

    def apply(self, coefficients: jax.Array) -> jax.Array:
        """Project complex amplitudes without normalization or probability averaging."""
        vector = jnp.asarray(coefficients)
        if vector.shape != (len(self.labels),):
            raise ValueError("one amplitude per reduced label required")
        b = jnp.asarray(self.factor)
        return b.conj().T @ (b @ vector)

    def probabilities(self, coefficients: jax.Array) -> jax.Array:
        """Actual normalized coherent Born law; zero projection is invalid.

        This pure JAX map leaves a zero-norm projection nonfinite rather than
        substituting the native AR law. Callers must reject that endpoint.
        """
        weights = jnp.abs(self.apply(coefficients)) ** 2
        return weights / jnp.sum(weights)

    def sample_indices(self, key: jax.Array, coefficients: jax.Array,
                       count: int) -> jax.Array:
        """Independent projected-Born draws, never native-AR relabeling."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("sample count must be a positive integer")
        probability = self.probabilities(coefficients)
        return jax.random.categorical(
            key, jnp.where(probability > 0, jnp.log(probability), -jnp.inf),
            shape=(count,),
        )


def compile_exact_a1_orbit_projector(
    labels: np.ndarray, isometry: np.ndarray, physical_bits: np.ndarray,
    group_images: Sequence[Sequence[int]], *, tolerance: float = 1e-10,
) -> ExactA1OrbitProjector:
    """Factor a full trivial-irrep projector on exact-small physical support.

    For physical orbit ``o``, ``B[o,a] = sum(U[s,a], s in o)/sqrt(|o|)``.
    Full Gram and subspace-closure checks are performed, not Gram probes.
    There is no CG truncation, symbolic runtime work, or eigensolver. This
    reference mechanism is useful for VAP gradients and sampling validation;
    its physical-support enumeration cannot scale to a 36-spin singlet.
    """
    rows = np.asarray(labels)
    bits = np.asarray(physical_bits)
    u = np.asarray(isometry)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if (rows.ndim != 2 or not np.issubdtype(rows.dtype, np.integer)
            or len(rows) < 1 or bits.ndim != 2 or len(bits) < 1
            or not np.issubdtype(bits.dtype, np.integer)
            or np.any((bits != 0) & (bits != 1))):
        raise ValueError("nonempty integer reduced labels and binary physical rows required")
    # Refuse exponential workloads before allocating group gathers or Grams.
    if bits.shape[1] > 16 or len(bits) > 65536 or len(rows) > 4096:
        raise ValueError("exact-small orbit plan cap exceeded; no N36 projection implied")
    if u.shape != (len(bits), len(rows)) or not np.isfinite(u).all():
        raise ValueError("finite isometry must match complete supplied supports")
    gram_error = float(np.max(np.abs(u.conj().T @ u - np.eye(len(rows)))))
    if gram_error > tolerance:
        raise ValueError("reduced basis is not orthonormal on supplied physical support")
    group = validated_permutation_group(group_images)
    indices = physical_permutation_index_table(bits, group)
    representatives = np.min(indices, axis=0)
    _, orbit_index, sizes = np.unique(representatives, return_inverse=True, return_counts=True)
    b = np.zeros((len(sizes), len(rows)), dtype=u.dtype)
    np.add.at(b, orbit_index, u)
    b /= np.sqrt(sizes[:, None])
    # Singular values of W†U must be 0 or 1 for a closed reduced sector.
    # BB†B=B checks that without an expensive reduced-space square matrix.
    closure_error = float(np.max(np.abs((b @ b.conj().T) @ b - b)))
    if closure_error > tolerance:
        raise ValueError("physical A1 projector does not close in the supplied reduced space")
    copied = [rows.copy(), b, orbit_index, sizes]
    for array in copied:
        array.setflags(write=False)
    return ExactA1OrbitProjector(*copied, len(group), gram_error, closure_error)


@dataclass(frozen=True)
class ExactProjectedReducedState:
    """Normalized projected coefficients; samples are NOT the native AR law."""
    labels: np.ndarray
    coefficients: np.ndarray
    projection_norm_squared: float
    group_order: int
    physical_projection_closure_error: float

    @property
    def probabilities(self) -> np.ndarray:
        return np.abs(self.coefficients)**2

    def sample(self,key: jax.Array,count: int) -> jax.Array:
        """Independent categorical draws from the coherent projected Born law."""
        if isinstance(count,bool) or not isinstance(count,int) or count<1:
            raise ValueError("sample count must be a positive integer")
        probability = jnp.asarray(self.probabilities)
        index = jax.random.categorical(key,jnp.where(probability>0,jnp.log(probability),-jnp.inf),shape=(count,))
        return jnp.asarray(self.labels)[index]


def exact_projected_reduced_state(labels: np.ndarray, coefficients: np.ndarray,
                                 isometry: np.ndarray, physical_bits: np.ndarray,
                                 group_images: Sequence[Sequence[int]], *,
                                 tolerance: float = 1e-10,
                                 gram_probe_indices: Sequence[int] | None = None) -> ExactProjectedReducedState:
    """Project AMPLITUDES through a complete physical group, then normalize.

    ``isometry`` must contain a closed exact reduced sector on ``physical_bits``.
    By default it is checked fully for orthonormality and physical projection
    closure. No positive-probability orbit average or native AR relabeling is
    used. The exhaustive physical reference is intentionally exact-small.
    An explicitly supplied Gram probe is only a spot check, combined with ALL
    column norms; it requires an externally validated exact-CG isometry and
    is not advertised as a general orthonormality certificate.
    """
    rows = np.asarray(labels)
    if not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError("tolerance must be positive and finite")
    c = np.asarray(coefficients,dtype=np.complex128)
    u = np.asarray(isometry)
    bits = np.asarray(physical_bits)
    if rows.ndim!=2 or not np.issubdtype(rows.dtype,np.integer) or c.shape!=(len(rows),):
        raise ValueError("integer reduced labels and coefficients must match")
    if u.shape!=(len(bits),len(rows)) or bits.ndim!=2 or not np.isfinite(c).all() or not np.isfinite(u).all():
        raise ValueError("finite isometry and physical bits must match")
    probe = np.arange(len(rows)) if gram_probe_indices is None else np.asarray(gram_probe_indices,dtype=np.int64)
    if probe.ndim!=1 or len(probe)<1 or np.any(probe<0) or np.any(probe>=len(rows)) or len(set(probe.tolist()))!=len(probe):
        raise ValueError("Gram probes must be distinct in-range column indices")
    selected=u[:,probe]
    gram_error = max(float(np.max(np.abs(selected.conj().T@selected-np.eye(len(probe))))),
        float(np.max(np.abs(np.sum(abs(u)**2,axis=0)-1))))
    if gram_error>tolerance:
        raise ValueError("reduced basis is not orthonormal on supplied physical support")
    group = validated_permutation_group(group_images)
    indices = physical_permutation_index_table(bits,group)
    physical = u@c
    projected = np.mean(physical[indices],axis=0)
    reduced = u.conj().T@projected
    closure = float(np.max(np.abs(u@reduced-projected)))
    if closure>tolerance:
        raise ValueError("physical projector leaves the supplied reduced subspace")
    norm_squared = float(np.vdot(reduced,reduced).real)
    if not np.isfinite(norm_squared) or norm_squared<=1e-24:
        raise ValueError("projected state has zero or numerically unresolved norm")
    return ExactProjectedReducedState(rows.copy(),reduced/np.sqrt(norm_squared),norm_squared,len(group),closure)


__all__ = ["ExactProjectedReducedState", "exact_projected_reduced_state",
           "ExactA1OrbitProjector", "compile_exact_a1_orbit_projector"]
