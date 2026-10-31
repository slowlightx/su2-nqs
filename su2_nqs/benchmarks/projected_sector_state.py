"""Exact-small projected-sector states with proposal-independent row access.

This module is an integration reference for a future spatial-sector VMC state,
not a scalable projector or sampler.  It first applies an audited
:class:`~su2_nqs.benchmarks.common_basis_gamma_a1.FixedBasisGammaA1Circuit` to
full vectors in one fixed fusion basis, then evaluates the physical
Hamiltonian action on the projected vectors.  A separate positive proposal
may subsequently select rows for importance-weighted estimators.

The separation is deliberate: exact ``Gamma.A1`` amplitudes define the state,
whereas the proposal only determines how configurations are observed.  The
proposal therefore need not equal the projected Born distribution.  Since the
projector still enumerates the complete fixed-spin multiplicity basis, every
public constructor fails closed above an explicit exact-small dimension cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from .common_basis_gamma_a1 import FixedBasisGammaA1Circuit


def _read_only(array: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(reference)), float(np.finfo(np.float64).tiny)
    )
    return float(np.linalg.norm(candidate - reference) / denominator)


def validated_full_support_proposal(
    proposal_probabilities: Sequence[float],
    dimension: int,
    *,
    normalization_tolerance: float = 5.0e-13,
) -> np.ndarray:
    """Return a normalized, strictly positive exact-small proposal.

    The entries are probabilities on the *complete* fixed-spin fusion basis,
    rather than unnormalized weights or probabilities conditional on a sampled
    subset.  Strict positivity ensures that every nonzero projected amplitude
    and Hamiltonian-action row is covered by importance sampling.
    """

    if isinstance(dimension, bool) or not isinstance(dimension, Integral):
        raise TypeError("dimension must be an integer")
    if int(dimension) < 1:
        raise ValueError("dimension must be positive")
    if (
        not np.isfinite(normalization_tolerance)
        or normalization_tolerance <= 0.0
    ):
        raise ValueError("normalization_tolerance must be finite and positive")
    proposal = np.asarray(proposal_probabilities, dtype=np.float64)
    if proposal.shape != (int(dimension),):
        raise ValueError("proposal must have one probability per fusion-basis state")
    if np.any(~np.isfinite(proposal)) or np.any(proposal <= 0.0):
        raise ValueError("proposal probabilities must be finite and strictly positive")
    normalization_error = abs(float(np.sum(proposal)) - 1.0)
    if normalization_error > normalization_tolerance:
        raise ValueError(
            "proposal probabilities must sum to one on the complete basis: "
            f"error={normalization_error:.6e}"
        )
    return _read_only(proposal, dtype=np.float64)


@dataclass(frozen=True)
class ProjectedSectorProposalRows:
    """Amplitude and action rows evaluated under an external proposal.

    ``observation_weights`` is the exact quadrature measure when
    ``is_exhaustive`` is true.  It is ``None`` for IID samples, for which the
    generic sampled estimator must use uniform empirical observation weights.
    In both cases ``proposal_probabilities`` contains the actual global
    proposal probability of each returned basis index.
    """

    basis_indices: np.ndarray
    amplitudes: np.ndarray
    hamiltonian_actions: np.ndarray
    proposal_probabilities: np.ndarray
    observation_weights: np.ndarray | None
    is_exhaustive: bool

    def __post_init__(self) -> None:
        indices = np.asarray(self.basis_indices)
        amplitudes = np.asarray(self.amplitudes)
        actions = np.asarray(self.hamiltonian_actions)
        proposal = np.asarray(self.proposal_probabilities)
        if indices.ndim != 1 or len(indices) < 1:
            raise ValueError("basis_indices must be a nonempty vector")
        if amplitudes.ndim != 2 or amplitudes.shape[0] != len(indices):
            raise ValueError("amplitudes must have one row per basis index")
        if actions.shape != amplitudes.shape:
            raise ValueError("hamiltonian_actions must match amplitudes")
        if proposal.shape != (len(indices),) or np.any(proposal <= 0.0):
            raise ValueError("sampled proposal probabilities must be positive")
        if self.observation_weights is not None:
            weights = np.asarray(self.observation_weights)
            if weights.shape != (len(indices),) or np.any(weights < 0.0):
                raise ValueError("observation_weights must match the returned rows")

    @property
    def sample_count(self) -> int:
        return int(len(self.basis_indices))


@dataclass(frozen=True)
class ExactSmallProjectedSectorState:
    """Normalized projected vectors and their exact Hamiltonian actions.

    Instances should be created with :meth:`from_base_amplitudes`.  The class
    stores complete vectors and is therefore intentionally unsuitable for
    N=36.  ``projector_residual`` and ``action_projector_residual`` certify,
    within the requested numerical tolerance, that both ``psi`` and ``H psi``
    remain in the circuit's trivial-character ``Gamma.A1`` image.
    """

    circuit: FixedBasisGammaA1Circuit
    amplitudes: np.ndarray
    hamiltonian_actions: np.ndarray
    projector_residual: float
    action_projector_residual: float
    hamiltonian_hermiticity_error: float

    @classmethod
    def from_base_amplitudes(
        cls,
        circuit: FixedBasisGammaA1Circuit,
        hamiltonian: np.ndarray,
        base_amplitudes: np.ndarray,
        *,
        maximum_dimension: int = 4096,
        zero_tolerance: float = 1.0e-13,
        relative_rank_tolerance: float = 1.0e-12,
        invariance_tolerance: float = 1.0e-10,
    ) -> "ExactSmallProjectedSectorState":
        """Project independent base vectors and evaluate their physical action.

        The supplied Hamiltonian must be a finite Hermitian matrix on the same
        fixed fusion basis.  This constructor does not silently replace it by
        ``P H P``: it computes ``H psi`` and fails if that action leaves the
        projected sector.  Consequently a passing state audits the original
        Hamiltonian action rather than a modified projected Hamiltonian.
        """

        if not isinstance(circuit, FixedBasisGammaA1Circuit):
            raise TypeError("circuit must be a FixedBasisGammaA1Circuit")
        if (
            isinstance(maximum_dimension, bool)
            or not isinstance(maximum_dimension, Integral)
            or int(maximum_dimension) < 1
        ):
            raise ValueError("maximum_dimension must be a positive integer")
        if circuit.dimension > int(maximum_dimension):
            raise ValueError(
                "fusion dimension exceeds the exact-small projected-state cap"
            )
        for value, name in (
            (zero_tolerance, "zero_tolerance"),
            (relative_rank_tolerance, "relative_rank_tolerance"),
            (invariance_tolerance, "invariance_tolerance"),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        matrix = np.asarray(hamiltonian, dtype=np.complex128)
        expected_matrix_shape = (circuit.dimension, circuit.dimension)
        if matrix.shape != expected_matrix_shape:
            raise ValueError("Hamiltonian shape disagrees with the fusion sector")
        if np.any(~np.isfinite(matrix)):
            raise ValueError("Hamiltonian entries must be finite")
        hermiticity_error = _relative_error(matrix, matrix.conj().T)
        if hermiticity_error > invariance_tolerance:
            raise ValueError(
                "Hamiltonian is not Hermitian within tolerance: "
                f"error={hermiticity_error:.6e}"
            )

        base = np.asarray(base_amplitudes, dtype=np.complex128)
        if (
            base.ndim != 2
            or base.shape[0] != circuit.dimension
            or base.shape[1] < 1
        ):
            raise ValueError(
                "base_amplitudes must have shape (fusion dimension, state count)"
            )
        if np.any(~np.isfinite(base)):
            raise ValueError("base amplitudes must be finite")

        projected = np.asarray(circuit.apply(base), dtype=np.complex128)
        norms = np.linalg.norm(projected, axis=0)
        if np.any(~np.isfinite(norms)) or np.any(norms <= zero_tolerance):
            raise ValueError("a base vector has zero projection into Gamma.A1")
        amplitudes = projected / norms[None, :]
        gram_eigenvalues = np.linalg.eigvalsh(amplitudes.conj().T @ amplitudes)
        largest = float(gram_eigenvalues[-1])
        if (
            largest <= 0.0
            or float(gram_eigenvalues[0]) <= relative_rank_tolerance * largest
        ):
            raise ValueError("projected state columns are not numerically independent")

        projected_again = np.asarray(circuit.apply(amplitudes), dtype=np.complex128)
        projector_residual = _relative_error(amplitudes, projected_again)
        if projector_residual > invariance_tolerance:
            raise FloatingPointError(
                "projected amplitudes fail the Gamma.A1 invariance gate: "
                f"error={projector_residual:.6e}"
            )

        actions = matrix @ amplitudes
        projected_actions = np.asarray(circuit.apply(actions), dtype=np.complex128)
        action_residual = _relative_error(actions, projected_actions)
        if action_residual > invariance_tolerance:
            raise ValueError(
                "Hamiltonian action leaves the Gamma.A1 sector: "
                f"error={action_residual:.6e}"
            )
        return cls(
            circuit=circuit,
            amplitudes=_read_only(amplitudes, dtype=np.complex128),
            hamiltonian_actions=_read_only(actions, dtype=np.complex128),
            projector_residual=float(projector_residual),
            action_projector_residual=float(action_residual),
            hamiltonian_hermiticity_error=float(hermiticity_error),
        )

    @property
    def dimension(self) -> int:
        return int(self.amplitudes.shape[0])

    @property
    def state_count(self) -> int:
        return int(self.amplitudes.shape[1])

    @property
    def born_mixture_probabilities(self) -> np.ndarray:
        """Return the equal-root Born mixture for comparison only."""

        probabilities = np.mean(np.abs(self.amplitudes) ** 2, axis=1)
        return _read_only(probabilities, dtype=np.float64)

    def exhaustive_proposal_rows(
        self, proposal_probabilities: Sequence[float]
    ) -> ProjectedSectorProposalRows:
        """Return every row with exact proposal quadrature weights."""

        proposal = validated_full_support_proposal(
            proposal_probabilities, self.dimension
        )
        indices = np.arange(self.dimension, dtype=np.int64)
        return ProjectedSectorProposalRows(
            basis_indices=_read_only(indices, dtype=np.int64),
            amplitudes=self.amplitudes,
            hamiltonian_actions=self.hamiltonian_actions,
            proposal_probabilities=proposal,
            observation_weights=proposal,
            is_exhaustive=True,
        )

    def sample_proposal_rows(
        self,
        proposal_probabilities: Sequence[float],
        rng: np.random.Generator,
        n_samples: int,
    ) -> ProjectedSectorProposalRows:
        """Draw IID rows from an arbitrary positive full-support proposal."""

        proposal = validated_full_support_proposal(
            proposal_probabilities, self.dimension
        )
        if (
            isinstance(n_samples, bool)
            or not isinstance(n_samples, Integral)
            or int(n_samples) < 1
        ):
            raise ValueError("n_samples must be a positive integer")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        indices = np.asarray(
            rng.choice(self.dimension, size=int(n_samples), p=proposal),
            dtype=np.int64,
        )
        return ProjectedSectorProposalRows(
            basis_indices=_read_only(indices, dtype=np.int64),
            amplitudes=_read_only(self.amplitudes[indices], dtype=np.complex128),
            hamiltonian_actions=_read_only(
                self.hamiltonian_actions[indices], dtype=np.complex128
            ),
            proposal_probabilities=_read_only(proposal[indices], dtype=np.float64),
            observation_weights=None,
            is_exhaustive=False,
        )


__all__ = [
    "ExactSmallProjectedSectorState",
    "ProjectedSectorProposalRows",
    "validated_full_support_proposal",
]
