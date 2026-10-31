"""Exact references and result contracts for total-spin spectroscopy.

The public quantum-number label is always the doubled total spin
``target_q = 2S``.  These routines deliberately construct an independent
fixed-``M=S`` computational-basis reference before any neural optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..models.fusion_gru import FusionGRUConfig
from ..operators.permutation import (
    heisenberg_graph_fusion_matrix,
    site_permutation_fusion_matrix,
)
from ..operators.sz_graph_sparse import heisenberg_sz_graph_sparse
from ..su2.fusion_path import FusionPath, count_paths, enumerate_paths
from ..su2.sz_basis import enumerate_sz_configs
from .coupled_basis import coupled_basis_matrix
from .spin_sector import spin_squared_sz_sparse

SpinBond = tuple[int, int, float]


def _configuration_integer(configuration: Sequence[int]) -> int:
    """Return the MSB-first computational index of one bit configuration."""

    n_sites = len(configuration)
    return sum(
        int(bit) << (n_sites - site - 1)
        for site, bit in enumerate(configuration)
    )


def _normalize_bonds(
    n_sites: int,
    bonds: Iterable[tuple[int, int] | tuple[int, int, float]],
) -> tuple[SpinBond, ...]:
    combined: dict[tuple[int, int], float] = {}
    for raw_bond in bonds:
        if len(raw_bond) == 2:
            first, second = raw_bond
            coupling = 1.0
        elif len(raw_bond) == 3:
            first, second, coupling = raw_bond
        else:
            raise ValueError(f"invalid spin bond: {raw_bond!r}")
        if (
            isinstance(first, bool)
            or isinstance(second, bool)
            or not isinstance(first, Integral)
            or not isinstance(second, Integral)
        ):
            raise TypeError("bond sites must be integers")
        first, second = int(first), int(second)
        if first > second:
            first, second = second, first
        if not 0 <= first < second < n_sites:
            raise ValueError(
                f"bond sites must satisfy 0 <= i < j < N: {raw_bond!r}"
            )
        if isinstance(coupling, bool) or not isinstance(coupling, Real):
            raise TypeError("bond coupling must be a real scalar")
        value = float(coupling)
        if not np.isfinite(value):
            raise ValueError("bond coupling must be finite")
        combined[(first, second)] = combined.get((first, second), 0.0) + value
    return tuple(
        (first, second, coupling)
        for (first, second), coupling in sorted(combined.items())
        if abs(coupling) > 0.0
    )


@dataclass(frozen=True)
class SpinSectorSpec:
    """One deterministic graph Hamiltonian and exact total-spin sector."""

    system_id: str
    n_sites: int
    target_q: int
    bonds: tuple[SpinBond, ...]
    lattice: str = "custom"
    boundary: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.system_id.strip():
            raise ValueError("system_id must be nonempty")
        if isinstance(self.n_sites, bool) or not isinstance(self.n_sites, Integral):
            raise TypeError("n_sites must be an integer")
        if isinstance(self.target_q, bool) or not isinstance(self.target_q, Integral):
            raise TypeError("target_q must be a doubled-integer label")
        n_sites, target_q = int(self.n_sites), int(self.target_q)
        if n_sites < 1:
            raise ValueError("n_sites must be positive")
        if target_q < 0 or count_paths(n_sites, target_q) == 0:
            raise ValueError(
                f"target_q={target_q} is unreachable for n_sites={n_sites}"
            )
        normalized = _normalize_bonds(n_sites, self.bonds)
        if not normalized:
            raise ValueError("at least one nonzero bond is required")
        object.__setattr__(self, "n_sites", n_sites)
        object.__setattr__(self, "target_q", target_q)
        object.__setattr__(self, "bonds", normalized)

    @property
    def n_up_at_highest_weight(self) -> int:
        """Number of up spins in the independent fixed-``M=S`` reference."""

        return (self.n_sites + self.target_q) // 2

    @property
    def spin_squared_eigenvalue(self) -> float:
        """Return ``S(S+1)`` as an observable value, not a Hilbert label."""

        return self.target_q * (self.target_q + 2) / 4.0


@dataclass(frozen=True)
class ExactSpinSectorReference:
    """Independent small-system reference for one exact total-spin sector."""

    spec: SpinSectorSpec
    fusion_basis: tuple[FusionPath, ...]
    sz_basis: tuple[tuple[int, ...], ...]
    fusion_to_sz: np.ndarray
    fusion_hamiltonian: Any
    sz_hamiltonian: Any
    ground_energy: float
    ground_state: np.ndarray
    audit: Mapping[str, float]


def chain_translation_fusion_matrix(
    reference: ExactSpinSectorReference,
) -> np.ndarray:
    """Return one-site cyclic translation in the reference's fusion basis.

    The active convention sends the spin at site ``i`` to ``(i + 1) mod N``.
    The implementation reuses the audited local adjacent-swap/F-move compiler.
    Its agreement with the independently ordered fixed-``M=S`` basis is tested
    separately.
    """

    if reference.spec.lattice != "chain" or reference.spec.boundary != "periodic":
        raise ValueError("cyclic translation requires a periodic chain reference")
    image = tuple((site + 1) % reference.spec.n_sites for site in range(reference.spec.n_sites))
    return np.asarray(
        site_permutation_fusion_matrix(
            image, target_q=reference.spec.target_q
        ).toarray(),
        dtype=np.complex128,
    )


def cyclic_momentum_projectors(
    translation: np.ndarray,
    period: int,
) -> tuple[np.ndarray, ...]:
    """Return all integer-momentum projectors of one cyclic translation.

    Projector index ``k`` denotes momentum ``2*pi*k/N`` under the active
    convention ``T|k> = exp(-2*pi*i*k/N)|k>``.
    """

    operator = np.asarray(translation, dtype=np.complex128)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("translation must be a square matrix")
    dimension = operator.shape[0]
    if dimension < 1:
        raise ValueError("translation matrix must be nonempty")
    if isinstance(period, bool) or not isinstance(period, Integral) or period < 1:
        raise ValueError("period must be a positive integer")
    period = int(period)
    identity = np.eye(dimension, dtype=np.complex128)
    closure_error = np.linalg.norm(
        np.linalg.matrix_power(operator, period) - identity, ord="fro"
    )
    if closure_error > 1.0e-10 * max(np.sqrt(dimension), 1.0):
        raise ValueError("translation does not close at the requested period")

    powers = [identity]
    for _ in range(1, period):
        powers.append(powers[-1] @ operator)
    projectors = []
    for momentum in range(period):
        projector = sum(
            (
                np.exp(2.0j * np.pi * momentum * shift / period)
                * powers[shift]
                for shift in range(period)
            ),
            start=np.zeros_like(operator),
        ) / period
        projectors.append(projector)
    return tuple(projectors)


def momentum_weights(
    state: np.ndarray,
    projectors: Sequence[np.ndarray],
) -> tuple[float, ...]:
    """Return normalized weights in a complete cyclic-momentum resolution."""

    vector = np.asarray(state, dtype=np.complex128)
    norm = float(np.vdot(vector, vector).real)
    if vector.ndim != 1 or not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("state must be a finite nonzero vector")
    weights = tuple(
        float(np.real(np.vdot(vector, projector @ vector)) / norm)
        for projector in projectors
    )
    return tuple(0.0 if abs(value) < 1.0e-14 else value for value in weights)


def normalized_projected_state(
    state: np.ndarray,
    projector: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Project and normalize a state, returning its pre-normalization weight."""

    vector = np.asarray(state, dtype=np.complex128)
    projected = np.asarray(projector, dtype=np.complex128) @ vector
    weight = float(np.vdot(projected, projected).real / np.vdot(vector, vector).real)
    if not np.isfinite(weight) or weight <= 1.0e-14:
        raise ValueError("state has zero weight in the requested momentum sector")
    projected /= np.linalg.norm(projected)
    return projected, weight


def momentum_orbit_sampling_metrics(
    state: np.ndarray,
    translation: np.ndarray,
    momentum: int,
    period: int,
) -> dict[str, float | int]:
    """Audit a shared-parameter character projection and its orbit proposal.

    The coherent target is ``P_k |state>``.  Its proposal is the incoherent
    equal mixture of the translated component Born laws.  The returned ESS is
    an exact finite-basis population quantity; it diagnoses, but does not by
    itself implement, scalable sampling in different fusion-tree frames.
    """

    vector = np.asarray(state, dtype=np.complex128)
    operator = np.asarray(translation, dtype=np.complex128)
    if operator.shape != (vector.size, vector.size):
        raise ValueError("translation shape does not match the state")
    projectors = cyclic_momentum_projectors(operator, period)
    momentum = int(momentum) % int(period)
    normalized = vector / np.linalg.norm(vector)
    components = [normalized]
    for _ in range(1, period):
        components.append(operator @ components[-1])
    component_matrix = np.column_stack(components)
    component_norm_error = float(
        np.max(
            np.abs(np.sum(np.abs(component_matrix) ** 2, axis=0) - 1.0),
            initial=0.0,
        )
    )
    coherent, projected_weight = normalized_projected_state(
        normalized, projectors[momentum]
    )
    proposal = np.mean(np.abs(component_matrix) ** 2, axis=1)
    target = np.abs(coherent) ** 2
    support = proposal > 0.0
    outside_support = float(np.sum(target[~support]))
    importance = np.zeros_like(proposal)
    importance[support] = target[support] / proposal[support]
    second_moment = float(np.sum(proposal * importance**2))
    eigenvalue = np.exp(-2.0j * np.pi * momentum / period)
    return {
        "period": int(period),
        "momentum_integer": momentum,
        "shared_parameter_component_count": int(period),
        "projected_weight": projected_weight,
        "projected_normalization_error": abs(
            float(np.vdot(coherent, coherent).real) - 1.0
        ),
        "translation_eigen_residual": float(
            np.linalg.norm(operator @ coherent - eigenvalue * coherent)
        ),
        "maximum_component_normalization_error": component_norm_error,
        "proposal_population_ess_fraction": (
            1.0 / second_moment if second_moment > 0.0 else 0.0
        ),
        "proposal_target_total_variation": float(
            0.5 * np.sum(np.abs(target - proposal))
        ),
        "proposal_maximum_importance_weight": float(
            np.max(importance, initial=0.0)
        ),
        "target_weight_outside_proposal_support": outside_support,
    }


def fixed_m_coupled_basis(
    n_sites: int,
    target_q: int,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Return the fusion isometry in fixed ``M=S`` configuration order."""

    paths = enumerate_paths(n_sites, target_q)
    if not paths:
        raise ValueError(
            f"target_q={target_q} is unreachable for n_sites={n_sites}"
        )
    n_up = (n_sites + target_q) // 2
    sz_basis = enumerate_sz_configs(n_sites, n_up)
    full = coupled_basis_matrix(paths, target_q, target_q)
    indices = tuple(_configuration_integer(configuration) for configuration in sz_basis)
    return np.asarray(full[np.asarray(indices), :], dtype=np.complex128), sz_basis


def _lowest_eigenpair(matrix: Any) -> tuple[float, np.ndarray, float]:
    dimension = int(matrix.shape[0])
    if dimension == 1:
        energy = float(matrix[0, 0])
        state = np.ones(1, dtype=np.float64)
        return energy, state, 0.0
    try:
        from scipy.sparse.linalg import eigsh
    except ImportError as error:
        raise RuntimeError("sector references require scipy") from error
    initial = np.linspace(1.0, 2.0, dimension, dtype=np.float64)
    initial /= np.linalg.norm(initial)
    values, vectors = eigsh(
        matrix,
        k=1,
        which="SA",
        tol=1.0e-13,
        maxiter=max(1000, 20 * dimension),
        v0=initial,
    )
    energy = float(values[0])
    state = np.asarray(vectors[:, 0], dtype=np.float64)
    state /= np.linalg.norm(state)
    residual = float(np.linalg.norm(matrix @ state - energy * state))
    return energy, state, residual


def build_exact_sector_reference(
    spec: SpinSectorSpec,
) -> ExactSpinSectorReference:
    """Build and cross-audit fusion and fixed-``M=S`` graph Hamiltonians."""

    paths = enumerate_paths(spec.n_sites, spec.target_q)
    transform, sz_basis = fixed_m_coupled_basis(spec.n_sites, spec.target_q)
    fusion = heisenberg_graph_fusion_matrix(
        spec.n_sites,
        spec.bonds,
        target_q=spec.target_q,
    ).tocsr()
    sz_hamiltonian, reference_basis = heisenberg_sz_graph_sparse(
        spec.n_sites,
        spec.bonds,
        n_up=spec.n_up_at_highest_weight,
    )
    if tuple(reference_basis) != tuple(sz_basis):
        raise RuntimeError("fixed-M basis order changed between exact references")
    spin_squared, spin_basis = spin_squared_sz_sparse(
        spec.n_sites,
        n_up=spec.n_up_at_highest_weight,
    )
    if tuple(spin_basis) != tuple(sz_basis):
        raise RuntimeError("fixed-M basis order changed in S^2 reference")

    identity = np.eye(len(paths), dtype=np.complex128)
    projected = transform.conj().T @ (sz_hamiltonian @ transform)
    fusion_dense = np.asarray(fusion.toarray(), dtype=np.complex128)
    spin_action = spin_squared @ transform
    expected_spin_action = spec.spin_squared_eigenvalue * transform
    energy, state, eigen_residual = _lowest_eigenpair(fusion)
    spectrum_error = float(
        np.max(
            np.abs(
                np.linalg.eigvalsh(fusion_dense)
                - np.linalg.eigvalsh(projected)
            ),
            initial=0.0,
        )
    )
    audit = {
        "isometry_error": float(
            np.max(np.abs(transform.conj().T @ transform - identity), initial=0.0)
        ),
        "spin_squared_error": float(
            np.max(np.abs(spin_action - expected_spin_action), initial=0.0)
        ),
        "hamiltonian_projection_error": float(
            np.max(np.abs(fusion_dense - projected), initial=0.0)
        ),
        "spectrum_error": spectrum_error,
        "hermiticity_error": float(
            np.max(np.abs(fusion_dense - fusion_dense.conj().T), initial=0.0)
        ),
        "eigen_residual": eigen_residual,
    }
    return ExactSpinSectorReference(
        spec=spec,
        fusion_basis=paths,
        sz_basis=sz_basis,
        fusion_to_sz=transform,
        fusion_hamiltonian=fusion,
        sz_hamiltonian=sz_hamiltonian.tocsr(),
        ground_energy=energy,
        ground_state=state,
        audit=audit,
    )


def sector_state_metrics(
    state: np.ndarray,
    reference: ExactSpinSectorReference,
) -> dict[str, float]:
    """Evaluate normalized energy, variance and fidelity in one sector."""

    vector = np.asarray(state, dtype=np.complex128)
    dimension = len(reference.fusion_basis)
    if vector.shape != (dimension,):
        raise ValueError(f"state must have shape ({dimension},)")
    norm = float(np.vdot(vector, vector).real)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("state must have finite positive norm")
    normalized = vector / np.sqrt(norm)
    action = np.asarray(reference.fusion_hamiltonian @ normalized)
    energy_value = complex(np.vdot(normalized, action))
    if abs(energy_value.imag) > 1.0e-10:
        raise RuntimeError("Hermitian energy acquired an imaginary part")
    energy = float(energy_value.real)
    variance = float(max(np.vdot(action, action).real - energy**2, 0.0))
    fidelity = float(abs(np.vdot(reference.ground_state, normalized)) ** 2)
    return {
        "normalization_before_rescale": norm,
        "energy": energy,
        "energy_error": energy - reference.ground_energy,
        "energy_error_per_site": (energy - reference.ground_energy)
        / reference.spec.n_sites,
        "variance": variance,
        "fidelity": fidelity,
        "infidelity": 1.0 - fidelity,
        "spin_squared_expectation": reference.spec.spin_squared_eigenvalue,
        "sector_leakage": 0.0,
    }


def build_exact_spin_problem(
    reference: ExactSpinSectorReference,
    *,
    embedding_dim: int,
    hidden_dim: int,
):
    """Build an :class:`ExactGRUProblem` for the audited sector reference."""

    return build_exact_fusion_problem(
        reference.spec.n_sites,
        reference.spec.target_q,
        reference.fusion_hamiltonian,
        exact_energy=reference.ground_energy,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )


def build_exact_fusion_problem(
    n_sites: int,
    target_q: int,
    hamiltonian: Any,
    *,
    exact_energy: float,
    embedding_dim: int,
    hidden_dim: int,
):
    """Build an exhaustive fusion-GRU problem from an audited Hamiltonian.

    Unlike :func:`build_exact_spin_problem`, this constructor does not build a
    computational-basis cross-reference.  It is intended for a measured local
    feasibility profile only after the same operator family has passed the
    required small-system projection ladder.
    """

    import jax.numpy as jnp

    from .optimization import ExactGRUProblem

    paths = enumerate_paths(n_sites, target_q)
    dimension = len(paths)
    if getattr(hamiltonian, "shape", None) != (dimension, dimension):
        raise ValueError("hamiltonian shape does not match the fusion sector")
    dense = (
        hamiltonian.toarray()
        if hasattr(hamiltonian, "toarray")
        else np.asarray(hamiltonian)
    )
    if not np.isfinite(float(exact_energy)):
        raise ValueError("exact_energy must be finite")
    config = FusionGRUConfig(
        n=n_sites,
        target_q=target_q,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    spin_eigenvalue = target_q * (target_q + 2) / 4.0
    spin_squared = spin_eigenvalue * np.eye(dimension)
    return ExactGRUProblem(
        n=n_sites,
        model="fusion",
        config=config,
        basis=jnp.asarray(paths, dtype=jnp.int32),
        hamiltonian=jnp.asarray(dense, dtype=jnp.float64),
        spin_squared=jnp.asarray(spin_squared, dtype=jnp.float64),
        exact_energy=float(exact_energy),
    )


SECTOR_RESULT_REQUIRED_KEYS = frozenset(
    {
        "system_id",
        "n_sites",
        "target_q",
        "basis_dimension",
        "exact_energy",
        "exact_eigen_residual",
        "model",
        "metrics",
        "cost",
        "gate_pass",
    }
)


def validate_sector_result_row(row: Mapping[str, Any]) -> None:
    """Validate the minimal machine-readable T0--T3 result contract."""

    missing = SECTOR_RESULT_REQUIRED_KEYS - set(row)
    if missing:
        raise ValueError(f"sector result row is missing keys: {sorted(missing)}")
    n_sites, target_q = row["n_sites"], row["target_q"]
    if (
        isinstance(n_sites, bool)
        or not isinstance(n_sites, Integral)
        or isinstance(target_q, bool)
        or not isinstance(target_q, Integral)
    ):
        raise TypeError("n_sites and target_q must be integer labels")
    expected_dimension = count_paths(int(n_sites), int(target_q))
    if expected_dimension == 0:
        raise ValueError("result row uses an unreachable target_q")
    if int(row["basis_dimension"]) != expected_dimension:
        raise ValueError("basis_dimension does not match the exact path count")
    metrics = row["metrics"]
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    for key in ("normalization", "energy", "variance", "fidelity", "sector_leakage"):
        if key not in metrics:
            raise ValueError(f"metrics is missing {key!r}")
        if not np.isfinite(float(metrics[key])):
            raise ValueError(f"metric {key!r} must be finite")
    if not 0.0 <= float(metrics["fidelity"]) <= 1.0 + 1.0e-10:
        raise ValueError("fidelity lies outside [0,1]")
    if float(metrics["sector_leakage"]) < -1.0e-14:
        raise ValueError("sector_leakage must be nonnegative")
