"""Exact-small diagnostics for resonating, shared-parameter fusion charts.

The same abstract fusion-path amplitude vector may be assigned to several
geometry-motivated physical-site orders.  Transporting every assignment to a
common reference frame produces a small variational subspace without adding a
separate neural network per chart.  A Hamiltonian Ritz solve in that subspace
is target independent and tests whether coherent recoupling charts carry a
useful physical correction to one frozen neural state.

This module deliberately contains no optimizer or exact target construction.
It is an exact-small diagnostic; scalable mixtures must evaluate and sample
the coherent sum without enumerating the fusion sector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
import numpy as np

from ..lattices.spatial_routes import rectangular_support_route_orders
from .fusion_atlas_oracle import fusion_atlas_oracle, transport_tree_chart_state
from ..su2.fusion_tree import FusionTree


SiteOrder = tuple[int, ...]


def rectangular_fusion_atlas_orders(
    lattice: str,
    rows: int,
    columns: int,
) -> dict[str, SiteOrder]:
    """Return eight target-independent order-adapted fusion charts.

    Coordinates match the row-major square and oblique triangular lattice
    builders.  The square charts resolve local plaquettes, bipartite order,
    and the two stripe orientations.  The triangular charts resolve local
    triangles, both orientations of the three-sublattice order, and all three
    stripe directions.
    """

    return rectangular_support_route_orders(lattice, rows, columns)


def globally_scrambled_atlas_orders(
    orders: Mapping[str, Sequence[int]],
    permutation: Sequence[int],
    *,
    preserve_first: bool = True,
) -> dict[str, SiteOrder]:
    """Apply one global site relabeling to a chart family.

    A single relabeling preserves all relations internal to the chart family.
    Keeping the first chart unchanged makes its frozen-state baseline exactly
    matched between physical and control atlases.
    """

    names = tuple(orders)
    if not names:
        raise ValueError("orders must be nonempty")
    checked = {name: tuple(int(site) for site in order) for name, order in orders.items()}
    n_sites = len(checked[names[0]])
    if any(len(order) != n_sites for order in checked.values()):
        raise ValueError("all chart orders must have the same size")
    permutation = tuple(int(site) for site in permutation)
    if tuple(sorted(permutation)) != tuple(range(n_sites)):
        raise ValueError("permutation must contain every site exactly once")
    return {
        name: (
            order
            if preserve_first and index == 0
            else tuple(permutation[site] for site in order)
        )
        for index, (name, order) in enumerate(checked.items())
    }


def fusion_chart_components(
    base_state: Sequence[complex],
    orders: Mapping[str, Sequence[int]],
    *,
    target_q: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Transport one shared fusion-path state into every physical chart."""

    names = tuple(orders)
    if not names:
        raise ValueError("orders must be nonempty")
    reference = tuple(int(site) for site in orders[names[0]])
    tree = FusionTree.left_associated(len(reference))
    columns = []
    audits = []
    for name in names:
        order = tuple(int(site) for site in orders[name])
        transported, audit = transport_tree_chart_state(
            np.asarray(base_state, dtype=np.complex128),
            tree,
            int(target_q),
            reference,
            order,
        )
        columns.append(transported)
        audits.append({"name": name, "physical_order": list(order), **audit})
    return np.column_stack(columns), audits


def hamiltonian_ritz_state(
    components: np.ndarray, hamiltonian: np.ndarray
) -> tuple[float, np.ndarray, int]:
    """Return the lowest normalized state in a possibly redundant chart span."""

    matrix = np.asarray(components, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("components must contain at least one state column")
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    if hamiltonian.shape != (matrix.shape[0], matrix.shape[0]):
        raise ValueError("hamiltonian shape does not match the chart components")
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise ValueError("components have zero numerical rank")
    retained = singular_values > 1.0e-12 * singular_values[0]
    basis = left[:, retained]
    reduced = basis.conj().T @ hamiltonian @ basis
    reduced = 0.5 * (reduced + reduced.conj().T)
    values, vectors = np.linalg.eigh(reduced)
    state = basis @ vectors[:, 0]
    state /= np.linalg.norm(state)
    return float(values[0]), state, int(np.count_nonzero(retained))


def _ritz_energy(components: np.ndarray, hamiltonian: np.ndarray) -> float:
    return hamiltonian_ritz_state(components, hamiltonian)[0]


def atlas_subset_complexity(
    components: np.ndarray,
    hamiltonian: np.ndarray,
    *,
    recovery_fraction: float = 0.9,
) -> dict[str, object]:
    """Exhaustively resolve how many charts recover the coherent Ritz gain.

    The reference is the best one-chart energy, so the reported gain cannot be
    explained by merely selecting a better ordering.  All subsets are allowed;
    no exact target enters this calculation.
    """

    matrix = np.asarray(components, dtype=np.complex128)
    if matrix.ndim != 2 or not 1 <= matrix.shape[1] <= 12:
        raise ValueError("subset audit requires between one and twelve charts")
    if not 0.0 < recovery_fraction <= 1.0:
        raise ValueError("recovery_fraction must lie in (0, 1]")
    single_energies = [
        _ritz_energy(matrix[:, [index]], hamiltonian)
        for index in range(matrix.shape[1])
    ]
    best_single = min(single_energies)
    full_energy = _ritz_energy(matrix, hamiltonian)
    coherent_gain = max(best_single - full_energy, 0.0)
    threshold = best_single - recovery_fraction * coherent_gain
    minimum_size = 1
    best_by_size: dict[str, float] = {"1": best_single}
    witness: tuple[int, ...] = (int(np.argmin(single_energies)),)
    if coherent_gain > 1.0e-14:
        for size in range(2, matrix.shape[1] + 1):
            rows = [
                (_ritz_energy(matrix[:, subset], hamiltonian), subset)
                for subset in combinations(range(matrix.shape[1]), size)
            ]
            energy, subset = min(rows, key=lambda row: row[0])
            best_by_size[str(size)] = energy
            if minimum_size == 1 and energy <= threshold + 1.0e-12:
                minimum_size = size
                witness = subset
        if minimum_size == 1:
            minimum_size = matrix.shape[1]
            witness = tuple(range(matrix.shape[1]))
    return {
        "best_single_energy": best_single,
        "full_ritz_energy": full_energy,
        "coherent_gain_below_best_single": coherent_gain,
        "recovery_fraction": float(recovery_fraction),
        "minimum_chart_count_for_recovery": minimum_size,
        "witness_indices": list(witness),
        "best_energy_by_chart_count": best_by_size,
    }


def resonating_atlas_analysis(
    components: np.ndarray,
    hamiltonian: np.ndarray,
    exact_state: np.ndarray,
    *,
    component_names: Sequence[str],
    exact_energy: float,
    recovery_fraction: float = 0.9,
) -> dict[str, object]:
    """Combine target-independent Ritz and retrospective exact metrics."""

    analysis = fusion_atlas_oracle(
        components,
        hamiltonian,
        exact_state,
        component_names=component_names,
    )
    for key in (
        "base_component",
        "best_component",
        "hamiltonian_ritz",
        "target_projection_ceiling",
    ):
        analysis[key]["energy_error"] = float(analysis[key]["energy"]) - exact_energy
    for row in analysis["component_metrics"]:
        row["energy_error"] = float(row["energy"]) - exact_energy
    base_error = float(analysis["base_component"]["energy_error"])
    ritz_error = float(analysis["hamiltonian_ritz"]["energy_error"])
    analysis["fractional_energy_error_reduction_from_base"] = (
        (base_error - ritz_error) / base_error if base_error > 0.0 else 0.0
    )
    analysis["ritz_fidelity_gain_from_base"] = float(
        analysis["hamiltonian_ritz"]["fidelity"]
        - analysis["base_component"]["fidelity"]
    )
    analysis["ritz_variance_reduction_from_base"] = float(
        analysis["base_component"]["variance"]
        - analysis["hamiltonian_ritz"]["variance"]
    )
    analysis["subset_complexity"] = atlas_subset_complexity(
        components,
        hamiltonian,
        recovery_fraction=recovery_fraction,
    )
    return analysis


__all__ = [
    "atlas_subset_complexity",
    "fusion_chart_components",
    "globally_scrambled_atlas_orders",
    "hamiltonian_ritz_state",
    "rectangular_fusion_atlas_orders",
    "resonating_atlas_analysis",
]
