"""Carrier-preserving SU(2)-scalar circuits from overlapping Schur cells.

This exact-small benchmark asks a deliberately physical question.  Starting
from the same projected pi-flux carrier, can a shallow product of local
cell-spin/multiplicity operators improve the state because those operators
respect the lattice geometry?  All states stay in the complete ``S=0``,
``Gamma-A1`` sector.  A Haar-conjugated operator bank with identical spectra
is the matched locality falsifier.

The construction is not the scalable many-cell implementation: its dense
``Gamma-A1`` representation is used only to validate the proposed state
manifold on triangular ``4x4`` before matrix-free GPU work is licensed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..lattices.spatial_symmetry import (
    triangular_hnf_gamma_a1_group,
    triangular_hnf_translation_group,
)
from ..lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from ..su2.local_schur_fusion import (
    four_cell_sector_action_table,
    local_schur_channels,
)
from ..vmc.adam import train_exact_adam
from .clebsch_tree_exact import (
    graph_hamiltonian_coo,
    physical_permutation_index_table,
)
from .four_cell_schur import four_cell_schur_sector_basis_matrix
from .overlapping_schur_recoupling import (
    OverlappingSchurA1Representation,
    _reduced_feature_table,
    _tagged_modified_gram_schmidt,
    gamma_a1_coefficient_basis,
)
from .qsl_three_arm import _ritz_state
from .triangular_state_manifold import (
    TRIANGULAR_N16_RHOMBUS_SUPPORTS,
    TRIANGULAR_N16_SUPERCELL,
)
from .xdiag_physical_seed_krylov import SMALL_CASES, physical_seed_amplitudes


jax.config.update("jax_enable_x64", True)


TRIANGULAR_N16_EXACT_ENERGY_PER_SITE = {
    0.0: -0.5347196823447661,
    0.05: -0.5334708185348914,
    0.075: -0.5334387431652716,
    0.1: -0.533808440287404,
    0.125: -0.5345819429505472,
    0.15: -0.5357589120174036,
    0.175: -0.5373368107120877,
}
TRIANGULAR_N16_JASTROW_TWO_LANCZOS_PER_SITE = {
    0.0: -0.5344793244926445,
    0.05: -0.5333453445612246,
    0.075: -0.5333219669574663,
    0.1: -0.5336709651817819,
    0.125: -0.5344026258646033,
    0.15: -0.5355289798484821,
    0.175: -0.5370565508794579,
}
TRIANGULAR_N16_OPERATOR_VIT_PER_SITE = {
    0.0: -0.5347073457829501,
    0.05: -0.5334621291686428,
    0.075: -0.5334302241037252,
    0.1: -0.5338018702307502,
    0.125: -0.5345723598040336,
    0.15: -0.535749083217054,
    0.175: -0.5373274734054345,
}


@dataclass(frozen=True)
class CarrierCellGeometry:
    """Coupling-independent exact-small carrier and Schur representation."""

    configurations: tuple[tuple[int, ...], ...]
    singlet_basis: np.ndarray
    representation: OverlappingSchurA1Representation
    carrier_physical: np.ndarray
    carrier_a1: np.ndarray
    h1: csr_matrix
    h2: csr_matrix
    h1_a1: np.ndarray
    h2_a1: np.ndarray


def _normalized(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("state has zero or nonfinite norm")
    return values / norm


def _sparse_hamiltonian(
    configurations: tuple[tuple[int, ...], ...],
    *,
    j1: float,
    j2: float,
) -> csr_matrix:
    bonds = triangular_hnf_j1j2_bonds(
        TRIANGULAR_N16_SUPERCELL, j1=j1, j2=j2
    )
    rows, columns, values, checked = graph_hamiltonian_coo(16, 0, bonds)
    if checked != configurations:
        raise RuntimeError("Schur and Hamiltonian fixed-M bases disagree")
    return coo_matrix(
        (values, (rows, columns)),
        shape=(len(configurations), len(configurations)),
    ).tocsr()


def _canonical_tiling(
    supports: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(tuple(sorted(int(site) for site in support)) for support in supports)
    )


def _triangular_n16_tiling_orbit() -> tuple[tuple[tuple[int, ...], ...], ...]:
    primary = TRIANGULAR_N16_RHOMBUS_SUPPORTS
    translations = triangular_hnf_translation_group(TRIANGULAR_N16_SUPERCELL)
    result = [tuple(tuple(support) for support in primary)]
    seen = {_canonical_tiling(primary)}
    for image in translations:
        mapped = tuple(
            tuple(int(image[site]) for site in support) for support in primary
        )
        signature = _canonical_tiling(mapped)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(mapped)
    return tuple(result)


def _build_triangular_n16_overlapping_representation(
    singlet_basis: np.ndarray,
    configurations: tuple[tuple[int, ...], ...],
    permutation_indices: np.ndarray,
    *,
    maximum_operator_count: int,
) -> OverlappingSchurA1Representation:
    """Specialized wrapper kept separate from immutable sealed builders."""

    primary = np.asarray(singlet_basis, dtype=np.float64)
    coefficient_a1, physical_a1, spatial_residual = gamma_a1_coefficient_basis(
        primary, permutation_indices
    )
    features, feature_labels = _reduced_feature_table("square_plaquette")
    tilings = _triangular_n16_tiling_orbit()
    accumulated = np.zeros(
        (features.shape[1], coefficient_a1.shape[1], coefficient_a1.shape[1]),
        dtype=np.float64,
    )
    maximum_view_gram_error = 0.0
    for tiling in tilings:
        if _canonical_tiling(tiling) == _canonical_tiling(tilings[0]):
            view = primary
            checked = configurations
        else:
            view, checked = four_cell_schur_sector_basis_matrix(
                "square_plaquette", 0, target_m2=0, supports=tiling
            )
        if checked != configurations:
            raise RuntimeError("overlapping Schur views use different physical bases")
        reduced_a1 = np.asarray(view, dtype=np.float64).T @ physical_a1
        maximum_view_gram_error = max(
            maximum_view_gram_error,
            float(
                np.linalg.norm(
                    reduced_a1.T @ reduced_a1 - np.eye(reduced_a1.shape[1])
                )
            ),
        )
        accumulated += np.einsum("ni,nk,nj->kij", reduced_a1, features, reduced_a1)
    accumulated /= float(len(tilings))
    identity = np.eye(coefficient_a1.shape[1])
    candidates = []
    for operator in accumulated:
        hermitian = 0.5 * (operator + operator.T)
        hermitian -= np.trace(hermitian) / hermitian.shape[0] * identity
        candidates.append(hermitian.reshape(-1))
    orthogonalized = _tagged_modified_gram_schmidt(
        candidates,
        tolerance=2.0e-10,
        maximum_count=maximum_operator_count,
    )
    if not orthogonalized:
        raise RuntimeError("overlapping Schur features generated no A1 operators")
    physical_operators = np.stack(
        [vector.reshape(identity.shape) for _index, vector in orthogonalized]
    )
    labels = tuple(
        f"orthogonalized_from_{feature_labels[index]}"
        for index, _vector in orthogonalized
    )
    rotation = _haar_orthogonal(coefficient_a1.shape[1], 2026092403)
    scrambled_operators = np.einsum(
        "ai,kab,bj->kij", rotation, physical_operators, rotation
    )
    spectrum_error = max(
        float(
            np.max(
                np.abs(np.linalg.eigvalsh(physical) - np.linalg.eigvalsh(scrambled))
            )
        )
        for physical, scrambled in zip(
            physical_operators, scrambled_operators, strict=True
        )
    )
    hermiticity_error = float(
        max(
            np.max(
                np.abs(
                    physical_operators - physical_operators.transpose(0, 2, 1)
                )
            ),
            np.max(
                np.abs(
                    scrambled_operators - scrambled_operators.transpose(0, 2, 1)
                )
            ),
        )
    )
    if max(maximum_view_gram_error, spectrum_error, hermiticity_error) > 5.0e-10:
        raise RuntimeError("triangular N16 overlapping representation invariant failed")
    for array in (physical_operators, scrambled_operators):
        array.setflags(write=False)
    return OverlappingSchurA1Representation(
        kind="square_plaquette",
        tilings=tilings,
        coefficient_a1_basis=coefficient_a1,
        physical_a1_basis=physical_a1,
        physical_operators=physical_operators,
        scrambled_operators=scrambled_operators,
        operator_labels=labels,
        diagnostics={
            "tiling_count": len(tilings),
            "singlet_dimension": int(primary.shape[1]),
            "gamma_a1_dimension": int(coefficient_a1.shape[1]),
            "operator_count": int(len(physical_operators)),
            "spatial_residual": spatial_residual,
            "maximum_view_gram_error": maximum_view_gram_error,
            "maximum_operator_hermiticity_error": hermiticity_error,
            "maximum_scrambled_spectrum_error": spectrum_error,
        },
    )


@lru_cache(maxsize=None)
def triangular_n16_carrier_cell_geometry(
    maximum_operator_count: int = 8,
) -> CarrierCellGeometry:
    """Build and cache the complete ``S=0``, ``Gamma-A1`` validation space."""

    if isinstance(maximum_operator_count, bool) or maximum_operator_count < 1:
        raise ValueError("maximum_operator_count must be a positive integer")
    singlet_basis, configurations = four_cell_schur_sector_basis_matrix(
        "square_plaquette",
        0,
        target_m2=0,
        supports=TRIANGULAR_N16_RHOMBUS_SUPPORTS,
    )
    group = triangular_hnf_gamma_a1_group(TRIANGULAR_N16_SUPERCELL)
    indices = physical_permutation_index_table(configurations, group)
    representation = _build_triangular_n16_overlapping_representation(
        singlet_basis,
        configurations,
        indices,
        maximum_operator_count=maximum_operator_count,
    )
    carrier = physical_seed_amplitudes(
        np.asarray(configurations, dtype=np.int8), SMALL_CASES["square4"]
    )[:, 0]
    carrier = _normalized(carrier)
    carrier_a1 = representation.physical_a1_basis.T @ carrier
    reconstruction_error = float(
        np.linalg.norm(representation.physical_a1_basis @ carrier_a1 - carrier)
    )
    if reconstruction_error > 5.0e-11:
        raise RuntimeError("projected pi-flux carrier is not exact S=0/Gamma-A1")
    h1 = _sparse_hamiltonian(configurations, j1=1.0, j2=0.0)
    h2 = _sparse_hamiltonian(configurations, j1=0.0, j2=1.0)
    phi = representation.physical_a1_basis
    h1_a1 = phi.T @ (h1 @ phi)
    h2_a1 = phi.T @ (h2 @ phi)
    for array in (carrier, carrier_a1, h1_a1, h2_a1):
        array.setflags(write=False)
    return CarrierCellGeometry(
        configurations=configurations,
        singlet_basis=singlet_basis,
        representation=representation,
        carrier_physical=carrier,
        carrier_a1=carrier_a1,
        h1=h1,
        h2=h2,
        h1_a1=h1_a1,
        h2_a1=h2_a1,
    )


def serial_scalar_circuit_state(
    weights: jax.Array,
    carrier: jax.Array,
    operators: jax.Array,
) -> jax.Array:
    """Apply normalized residual scalar layers to one reduced carrier.

    ``weights`` has shape ``(depth, operator_count)``.  Zero weights reproduce
    the carrier exactly.  Since every operator acts inside the exact reduced
    sector, no magnetic index or post-hoc symmetry projection is involved.
    """

    if weights.ndim != 2 or operators.ndim != 3:
        raise ValueError("weights and operators must have ranks two and three")
    if weights.shape[1] != operators.shape[0]:
        raise ValueError("weight and operator counts disagree")
    if operators.shape[1:] != (carrier.shape[0], carrier.shape[0]):
        raise ValueError("operator and carrier dimensions disagree")
    state = carrier
    for layer in range(weights.shape[0]):
        responses = jnp.einsum("kij,j->ki", operators, state)
        state = state + jnp.einsum("k,ki->i", weights[layer], responses)
        state = state / jnp.sqrt(jnp.vdot(state, state).real)
    return state


def _haar_orthogonal(dimension: int, seed: int) -> np.ndarray:
    raw = np.random.default_rng(seed).normal(size=(dimension, dimension))
    orthogonal, triangular = np.linalg.qr(raw)
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    return orthogonal * signs[None, :]


def _energy_variance_reduced(
    state: np.ndarray, hamiltonian: np.ndarray
) -> tuple[float, float]:
    vector = _normalized(state)
    action = hamiltonian @ vector
    energy = float(np.vdot(vector, action).real)
    residual = action - energy * vector
    return energy, float(np.vdot(residual, residual).real)


def local_spin_channel_weights(
    reduced_a1_state: np.ndarray,
    geometry: CarrierCellGeometry,
) -> dict[str, object]:
    """Resolve a state into local-cell and inter-cell doubled-spin weights."""

    reduced = _normalized(reduced_a1_state)
    coefficients = geometry.representation.coefficient_a1_basis @ reduced
    probabilities = np.abs(coefficients) ** 2
    probabilities /= np.sum(probabilities)
    actions = four_cell_sector_action_table("square_plaquette", 0)
    channels = local_schur_channels("square_plaquette")
    local_indices = actions[:, (0, 1, 3, 4)]
    local_q = np.asarray(
        [[channels[int(index)][0] for index in row] for row in local_indices],
        dtype=np.int32,
    )
    pair_q = actions[:, (2, 5)]
    local_weights = {
        f"q{q}": float(
            np.sum(probabilities[:, None] * (local_q == q)) / local_q.shape[1]
        )
        for q in sorted({int(value) for value in local_q.flat})
    }
    pair_weights = {
        f"q{q}": float(
            np.sum(probabilities[:, None] * (pair_q == q)) / pair_q.shape[1]
        )
        for q in sorted({int(value) for value in pair_q.flat})
    }
    return {
        "local_cell_q_weights": local_weights,
        "inter_cell_pair_q_weights": pair_weights,
        "multiplicity_inverse_participation": float(np.sum(probabilities**2)),
        "mean_local_spin_j": float(np.sum(probabilities[:, None] * local_q) / 8.0),
    }


def _state_row(
    state: np.ndarray,
    geometry: CarrierCellGeometry,
    coupling: float,
    exact_energy_total: float,
) -> dict[str, object]:
    reduced = _normalized(state)
    e1 = float(np.vdot(reduced, geometry.h1_a1 @ reduced).real)
    e2 = float(np.vdot(reduced, geometry.h2_a1 @ reduced).real)
    energy, variance = _energy_variance_reduced(
        reduced, geometry.h1_a1 + coupling * geometry.h2_a1
    )
    return {
        "energy_total": energy,
        "energy_per_site": energy / 16.0,
        "energy_error_total": energy - exact_energy_total,
        "energy_error_per_site": (energy - exact_energy_total) / 16.0,
        "variance_total": variance,
        "j1_energy_total": e1,
        "j2_operator_expectation_total": e2,
        "carrier_fidelity": float(abs(np.vdot(geometry.carrier_a1, reduced)) ** 2),
        "channel_diagnostics": local_spin_channel_weights(reduced, geometry),
    }


def _train_arm(
    *,
    operators: np.ndarray,
    hamiltonian: np.ndarray,
    carrier: np.ndarray,
    depth: int,
    seed: int,
    steps: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], float]:
    count = operators.shape[0]
    mixing = _haar_orthogonal(count, seed)
    mixed_operators = np.einsum("ak,kij->aij", mixing, operators)
    jax_operators = jnp.asarray(mixed_operators)
    jax_hamiltonian = jnp.asarray(hamiltonian)
    jax_carrier = jnp.asarray(carrier)
    initial = jnp.zeros((depth, count), dtype=jnp.float64)

    def state_fn(weights: jax.Array) -> jax.Array:
        return serial_scalar_circuit_state(weights, jax_carrier, jax_operators)

    def loss_fn(weights: jax.Array) -> jax.Array:
        state = state_fn(weights)
        return jnp.vdot(state, jax_hamiltonian @ state).real

    started = time.perf_counter()
    trained, optimizer = train_exact_adam(
        initial,
        loss_fn=loss_fn,
        steps=steps,
        learning_rate=learning_rate,
        record_every=max(1, steps // 20),
    )
    state = np.asarray(state_fn(trained))
    canonical_weights = np.asarray(trained) @ mixing
    return state, canonical_weights, optimizer, time.perf_counter() - started


def carrier_cell_scalar_circuit_frontier(
    *,
    couplings: Sequence[float] = (0.0, 0.125),
    seeds: Sequence[int] = (20260927, 20260928, 20260929),
    depth: int = 2,
    steps: int = 600,
    learning_rate: float = 2.0e-2,
    maximum_operator_count: int = 8,
    exact_energy_per_site: Mapping[float, float] = TRIANGULAR_N16_EXACT_ENERGY_PER_SITE,
    jastrow_two_lanczos_per_site: Mapping[float, float] = TRIANGULAR_N16_JASTROW_TWO_LANCZOS_PER_SITE,
    operator_vit_per_site: Mapping[float, float] = TRIANGULAR_N16_OPERATOR_VIT_PER_SITE,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run the preregistered physical-vs-scrambled carrier circuit test."""

    checked_couplings = tuple(float(value) for value in couplings)
    checked_seeds = tuple(int(seed) for seed in seeds)
    if not checked_couplings or any(not np.isfinite(value) or value < 0.0 for value in checked_couplings):
        raise ValueError("couplings must be a nonempty finite nonnegative sequence")
    if len(checked_seeds) < 3 or len(set(checked_seeds)) != len(checked_seeds):
        raise ValueError("at least three distinct seeds are required")
    if depth < 1 or steps < 1 or learning_rate <= 0.0:
        raise ValueError("depth, steps, and learning_rate must be positive")
    if any(value not in exact_energy_per_site for value in checked_couplings):
        raise ValueError("every coupling needs a preregistered exact-energy anchor")
    geometry_started = time.perf_counter()
    geometry = triangular_n16_carrier_cell_geometry(maximum_operator_count)
    geometry_seconds = time.perf_counter() - geometry_started
    representation = geometry.representation
    physical_operators = np.asarray(representation.physical_operators)
    scrambled_operators = np.asarray(representation.scrambled_operators)
    rows = []
    state_arrays: dict[str, np.ndarray] = {
        "carrier_a1": np.asarray(geometry.carrier_a1),
        "physical_a1_basis": np.asarray(representation.physical_a1_basis),
        "coefficient_a1_basis": np.asarray(representation.coefficient_a1_basis),
        "physical_operators": physical_operators,
        "scrambled_operators": scrambled_operators,
    }
    canonical_by_coupling: dict[float, list[np.ndarray]] = {}
    best_states: dict[tuple[float, str], tuple[float, np.ndarray]] = {}
    two_lanczos_ranks: dict[float, int] = {}
    for coupling in checked_couplings:
        exact_total = 16.0 * float(exact_energy_per_site[coupling])
        hamiltonian = geometry.h1_a1 + coupling * geometry.h2_a1
        carrier_row = _state_row(
            geometry.carrier_a1, geometry, coupling, exact_total
        )
        carrier_physical = geometry.carrier_physical
        sparse_hamiltonian = geometry.h1 + coupling * geometry.h2
        two_lanczos_physical, _, _, rank = _ritz_state(
            carrier_physical, sparse_hamiltonian, 2
        )
        two_lanczos_a1 = representation.physical_a1_basis.T @ two_lanczos_physical
        two_lanczos_ranks[coupling] = rank
        for arm_id, operators in (
            ("physical_overlapping_cell_circuit", physical_operators),
            ("isospectral_scrambled_cell_circuit", scrambled_operators),
        ):
            for seed in checked_seeds:
                state, canonical, optimizer, elapsed = _train_arm(
                    operators=operators,
                    hamiltonian=hamiltonian,
                    carrier=geometry.carrier_a1,
                    depth=depth,
                    seed=seed,
                    steps=steps,
                    learning_rate=learning_rate,
                )
                metrics = _state_row(state, geometry, coupling, exact_total)
                row = {
                    "j2_over_j1": coupling,
                    "arm_id": arm_id,
                    "seed": seed,
                    "depth": depth,
                    "parameter_count": int(depth * len(operators)),
                    "optimization_steps": steps,
                    "outer_full_h_calls": int(steps + 2),
                    "learning_rate": learning_rate,
                    "wall_seconds": elapsed,
                    "metrics": metrics,
                    "canonical_weights": canonical.tolist(),
                    "optimizer": optimizer,
                }
                rows.append(row)
                key = (coupling, arm_id)
                previous = best_states.get(key)
                if previous is None or metrics["energy_total"] < previous[0]:
                    best_states[key] = (float(metrics["energy_total"]), state)
                if arm_id == "physical_overlapping_cell_circuit":
                    canonical_by_coupling.setdefault(coupling, []).append(canonical)
        state_arrays[f"coupling_{coupling:g}_two_lanczos_a1"] = two_lanczos_a1
        state_arrays[f"coupling_{coupling:g}_carrier_a1"] = geometry.carrier_a1
        state_arrays[f"coupling_{coupling:g}_h_a1"] = hamiltonian
        state_arrays[f"coupling_{coupling:g}_reference_marker"] = np.asarray([coupling])

    summaries = {}
    for coupling in checked_couplings:
        exact_total = 16.0 * float(exact_energy_per_site[coupling])
        coupling_rows = [row for row in rows if row["j2_over_j1"] == coupling]
        arm_summaries = {}
        for arm_id in (
            "physical_overlapping_cell_circuit",
            "isospectral_scrambled_cell_circuit",
        ):
            arm_rows = [row for row in coupling_rows if row["arm_id"] == arm_id]
            energies = np.asarray([row["metrics"]["energy_per_site"] for row in arm_rows])
            variances = np.asarray([row["metrics"]["variance_total"] for row in arm_rows])
            arm_summaries[arm_id] = {
                "mean_energy_per_site": float(np.mean(energies)),
                "standard_error_energy_per_site": float(np.std(energies, ddof=1) / np.sqrt(len(energies))),
                "median_energy_per_site": float(np.median(energies)),
                "best_energy_per_site": float(np.min(energies)),
                "median_variance_total": float(np.median(variances)),
                "every_seed_below_jastrow_two_lanczos": bool(
                    np.all(energies < float(jastrow_two_lanczos_per_site[coupling]))
                ),
                "every_seed_below_sealed_operator_vit": bool(
                    np.all(energies < float(operator_vit_per_site[coupling]))
                ),
            }
        physical = [
            row for row in coupling_rows
            if row["arm_id"] == "physical_overlapping_cell_circuit"
        ]
        scrambled = [
            row for row in coupling_rows
            if row["arm_id"] == "isospectral_scrambled_cell_circuit"
        ]
        carrier_metrics = _state_row(
            geometry.carrier_a1, geometry, coupling, exact_total
        )
        two_state = state_arrays[f"coupling_{coupling:g}_two_lanczos_a1"]
        summaries[str(coupling)] = {
            "exact_anchor_energy_per_site": float(exact_energy_per_site[coupling]),
            "references": {
                "bare_projected_pi_flux": carrier_metrics,
                "bare_carrier_two_lanczos": _state_row(
                    two_state, geometry, coupling, exact_total
                ),
                "bare_carrier_two_lanczos_krylov_rank": two_lanczos_ranks[coupling],
                "optimized_jastrow_two_lanczos_energy_per_site": float(
                    jastrow_two_lanczos_per_site[coupling]
                ),
                "sealed_operator_vit_energy_per_site": float(operator_vit_per_site[coupling]),
            },
            "arms": arm_summaries,
            "paired_physical_minus_scrambled_energy_per_site": [
                float(left["metrics"]["energy_per_site"] - right["metrics"]["energy_per_site"])
                for left, right in zip(physical, scrambled, strict=True)
            ],
        }
    transfer_rows = []
    for trained_coupling, weights_by_seed in canonical_by_coupling.items():
        for seed, weights in zip(checked_seeds, weights_by_seed, strict=True):
            for target_coupling in checked_couplings:
                state = np.asarray(
                    serial_scalar_circuit_state(
                        jnp.asarray(weights),
                        jnp.asarray(geometry.carrier_a1),
                        jnp.asarray(physical_operators),
                    )
                )
                exact_total = 16.0 * float(exact_energy_per_site[target_coupling])
                transfer_rows.append(
                    {
                        "trained_j2_over_j1": trained_coupling,
                        "target_j2_over_j1": target_coupling,
                        "seed": seed,
                        "metrics": _state_row(
                            state, geometry, target_coupling, exact_total
                        ),
                    }
                )
    for (coupling, arm_id), (_energy, state) in best_states.items():
        state_arrays[f"coupling_{coupling:g}_{arm_id}_best_a1"] = state
    physical_all = [
        row for row in rows if row["arm_id"] == "physical_overlapping_cell_circuit"
    ]
    scrambled_all = [
        row for row in rows if row["arm_id"] == "isospectral_scrambled_cell_circuit"
    ]
    gates = {
        "physical_beats_paired_scrambled_every_seed_and_coupling": bool(
            all(
                left["metrics"]["energy_per_site"]
                < right["metrics"]["energy_per_site"]
                for left, right in zip(physical_all, scrambled_all, strict=True)
            )
        ),
        "physical_beats_optimized_jastrow_two_lanczos_every_seed_and_coupling": bool(
            all(
                row["metrics"]["energy_per_site"]
                < float(jastrow_two_lanczos_per_site[row["j2_over_j1"]])
                for row in physical_all
            )
        ),
        "physical_beats_sealed_operator_vit_every_seed_and_coupling": bool(
            all(
                row["metrics"]["energy_per_site"]
                < float(operator_vit_per_site[row["j2_over_j1"]])
                for row in physical_all
            )
        ),
    }
    gates["license_n36_matrix_free_pilot"] = bool(all(gates.values()))
    result = {
        "benchmark": "carrier_preserving_overlapping_cell_scalar_circuit",
        "system": "triangular 4x4 periodic J1-J2 Heisenberg",
        "state_manifold": "projected pi-flux carrier times a serial exact-SU2/Gamma-A1 local-Schur scalar circuit",
        "geometry_build_seconds": geometry_seconds,
        "representation": {
            **representation.diagnostics,
            "operator_labels": list(representation.operator_labels),
            "carrier_a1_reconstruction_error": float(
                np.linalg.norm(
                    representation.physical_a1_basis @ geometry.carrier_a1
                    - geometry.carrier_physical
                )
            ),
            "circuit_depth": depth,
            "parameter_count": depth * len(physical_operators),
        },
        "preregistration": {
            "couplings": list(checked_couplings),
            "seeds": list(checked_seeds),
            "steps": steps,
            "learning_rate": learning_rate,
            "primary_gate": "physical locality must beat the paired isospectral scramble, optimized Jastrow+two-Lanczos, and the sealed Operator-ViT for every seed at both couplings",
        },
        "runs": rows,
        "coupling_summaries": summaries,
        "cross_coupling_transfer": transfer_rows,
        "gates": gates,
        "interpretation_contract": (
            "A pass supports local SU(2) multiplicity organization as a useful "
            "carrier dressing rather than merely extra parameters. A failure "
            "blocks triangular 6x6 production for this mechanism."
        ),
    }
    return result, state_arrays


__all__ = [
    "CarrierCellGeometry",
    "TRIANGULAR_N16_EXACT_ENERGY_PER_SITE",
    "carrier_cell_scalar_circuit_frontier",
    "local_spin_channel_weights",
    "serial_scalar_circuit_state",
    "triangular_n16_carrier_cell_geometry",
]
