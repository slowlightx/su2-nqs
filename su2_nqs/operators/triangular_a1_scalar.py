"""On-demand triangular ``A1`` scalar-operator connectivity.

The five registered tokens match the exact-small Operator-ViT benchmark but
act on one physical bit configuration at a time.  This is the scalable bridge
to sampled VMC: no fixed-``Sz`` basis or full operator matrix is constructed.
Connectivity follows the repository convention ``O[new, old]`` and repeated
images are combined before returning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np

from ..lattices.triangular_hnf import (
    IntMatrix,
    J1_POSITIVE,
    J2_POSITIVE,
    hnf_sites,
    reduce_hnf,
    validate_hnf,
)
from .spin_config import heisenberg_spin_connectivity


TRIANGULAR_A1_SCALAR_LABELS = (
    "J1_shell",
    "J2_shell",
    "rhombus_ring_exchange",
    "rhombus_dimer_resonance",
    "six_site_loop_rearrangement",
)


def _checked_configuration(
    spins: Sequence[int], supercell: IntMatrix
) -> tuple[int, ...]:
    matrix = validate_hnf(supercell)
    checked = tuple(int(value) for value in spins)
    if len(checked) != len(hnf_sites(matrix)) or any(
        value not in (0, 1) for value in checked
    ):
        raise ValueError("spins must be one binary value per triangular site")
    return checked


@lru_cache(maxsize=None)
def _bonds(
    supercell: IntMatrix,
    displacements: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    sites = hnf_sites(supercell)
    positions = {site: index for index, site in enumerate(sites)}
    result = set()
    for first, site in enumerate(sites):
        for displacement in displacements:
            target = reduce_hnf(
                (site[0] + displacement[0], site[1] + displacement[1]),
                supercell,
            )
            second = positions[target]
            if first == second:
                raise ValueError("scalar shell aliases a site to itself")
            result.add(tuple(sorted((first, second))))
    return tuple(sorted(result))


def _cycle_configuration(
    spins: tuple[int, ...], cycle: tuple[int, ...]
) -> tuple[int, ...]:
    result = list(spins)
    for source, target in zip(cycle, cycle[1:] + cycle[:1], strict=True):
        result[target] = spins[source]
    return tuple(result)


def _swap_configuration(
    spins: tuple[int, ...], first: int, second: int
) -> tuple[int, ...]:
    result = list(spins)
    result[first], result[second] = result[second], result[first]
    return tuple(result)


def _add(
    elements: dict[tuple[int, ...], float],
    configuration: tuple[int, ...],
    value: float,
) -> None:
    elements[configuration] = elements.get(configuration, 0.0) + float(value)


def _finalize(
    elements: Mapping[tuple[int, ...], float],
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    items = tuple(
        sorted(
            (
                (configuration, value)
                for configuration, value in elements.items()
                if abs(value) > 1.0e-14
            ),
            key=lambda item: item[0],
        )
    )
    return (
        tuple(item[0] for item in items),
        np.asarray([item[1] for item in items], dtype=np.float64),
    )


@lru_cache(maxsize=None)
def _rhombi_and_hexagons(
    supercell: IntMatrix,
) -> tuple[
    tuple[tuple[int, int, int, int], ...],
    tuple[tuple[int, int, int, int, int, int], ...],
]:
    sites = hnf_sites(supercell)
    positions = {site: index for index, site in enumerate(sites)}
    rhombi = []
    for site in sites:
        for first_axis in range(3):
            for second_axis in range(first_axis + 1, 3):
                first_step = J1_POSITIVE[first_axis]
                second_step = J1_POSITIVE[second_axis]
                coordinates = (
                    site,
                    reduce_hnf(
                        (site[0] + first_step[0], site[1] + first_step[1]),
                        supercell,
                    ),
                    reduce_hnf(
                        (
                            site[0] + first_step[0] + second_step[0],
                            site[1] + first_step[1] + second_step[1],
                        ),
                        supercell,
                    ),
                    reduce_hnf(
                        (site[0] + second_step[0], site[1] + second_step[1]),
                        supercell,
                    ),
                )
                cycle = tuple(positions[value] for value in coordinates)
                if len(set(cycle)) != 4:
                    raise ValueError("rhombus aliases on this triangular torus")
                rhombi.append(cycle)
    neighbour_steps = ((1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1))
    hexagons = []
    for site in sites:
        cycle = tuple(
            positions[
                reduce_hnf(
                    (site[0] + step[0], site[1] + step[1]), supercell
                )
            ]
            for step in neighbour_steps
        )
        if len(set(cycle)) != 6:
            raise ValueError("six-site loop aliases on this triangular torus")
        hexagons.append(cycle)
    return tuple(rhombi), tuple(hexagons)


def triangular_a1_scalar_token_connectivity(
    spins: Sequence[int],
    supercell: IntMatrix,
) -> dict[str, tuple[tuple[tuple[int, ...], ...], np.ndarray]]:
    """Return separate exact connectivity for all five scalar tokens."""

    matrix = validate_hnf(supercell)
    old = _checked_configuration(spins, matrix)
    j1 = heisenberg_spin_connectivity(old, _bonds(matrix, J1_POSITIVE))
    j2 = heisenberg_spin_connectivity(old, _bonds(matrix, J2_POSITIVE))
    rhombi, hexagons = _rhombi_and_hexagons(matrix)

    ring: dict[tuple[int, ...], float] = {}
    dimer: dict[tuple[int, ...], float] = {}
    for cycle in rhombi:
        _add(ring, _cycle_configuration(old, cycle), 1.0)
        _add(ring, _cycle_configuration(old, tuple(reversed(cycle))), 1.0)
        for first_pair, second_pair in (
            ((cycle[0], cycle[1]), (cycle[2], cycle[3])),
            ((cycle[1], cycle[2]), (cycle[3], cycle[0])),
        ):
            first_swapped = _swap_configuration(old, *first_pair)
            second_swapped = _swap_configuration(old, *second_pair)
            both_swapped = _swap_configuration(first_swapped, *second_pair)
            _add(dimer, old, 0.25)
            _add(dimer, first_swapped, -0.25)
            _add(dimer, second_swapped, -0.25)
            _add(dimer, both_swapped, 0.25)

    hexagon: dict[tuple[int, ...], float] = {}
    for cycle in hexagons:
        _add(hexagon, _cycle_configuration(old, cycle), 1.0)
        _add(hexagon, _cycle_configuration(old, tuple(reversed(cycle))), 1.0)
    return {
        "J1_shell": j1,
        "J2_shell": j2,
        "rhombus_ring_exchange": _finalize(ring),
        "rhombus_dimer_resonance": _finalize(dimer),
        "six_site_loop_rearrangement": _finalize(hexagon),
    }


def triangular_a1_scalar_connectivity(
    spins: Sequence[int],
    supercell: IntMatrix,
    weights: Sequence[float],
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    """Combine token connectivity with one real coefficient per label."""

    coefficients = np.asarray(weights, dtype=np.float64)
    if coefficients.shape != (len(TRIANGULAR_A1_SCALAR_LABELS),) or np.any(
        ~np.isfinite(coefficients)
    ):
        raise ValueError("weights must contain five finite real coefficients")
    tokens = triangular_a1_scalar_token_connectivity(spins, supercell)
    combined: dict[tuple[int, ...], float] = {}
    for coefficient, label in zip(
        coefficients, TRIANGULAR_A1_SCALAR_LABELS, strict=True
    ):
        if coefficient == 0.0:
            continue
        states, values = tokens[label]
        for state, value in zip(states, values, strict=True):
            _add(combined, state, coefficient * float(value))
    return _finalize(combined)


__all__ = [
    "TRIANGULAR_A1_SCALAR_LABELS",
    "triangular_a1_scalar_connectivity",
    "triangular_a1_scalar_token_connectivity",
]
