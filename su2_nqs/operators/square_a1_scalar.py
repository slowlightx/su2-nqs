"""On-demand square-lattice ``A1`` scalar-operator connectivity.

The registered bank mirrors the triangular production bank while respecting
square geometry: J1 and J2 exchange shells, four-site plaquette ring exchange,
opposite-edge singlet resonance, and six-site 2x1 rectangular loops.
Connectivity follows ``O[new_configuration, old_configuration]`` and combines
all repeated images before returning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np

from ..lattices.j1j2 import square_j1j2_bonds
from .spin_config import heisenberg_spin_connectivity


SQUARE_A1_SCALAR_LABELS = (
    "J1_shell",
    "J2_shell",
    "plaquette_ring_exchange",
    "plaquette_dimer_resonance",
    "six_site_rectangle_rearrangement",
)


def _checked_configuration(
    spins: Sequence[int], linear_size: int
) -> tuple[int, ...]:
    if linear_size < 3:
        raise ValueError("linear_size must be at least three")
    checked = tuple(int(value) for value in spins)
    if len(checked) != linear_size**2 or any(value not in (0, 1) for value in checked):
        raise ValueError("spins must be one binary value per square-lattice site")
    return checked


def _site(x: int, y: int, linear_size: int) -> int:
    return (y % linear_size) * linear_size + (x % linear_size)


@lru_cache(maxsize=None)
def _shell_bonds(
    linear_size: int, shell: int
) -> tuple[tuple[int, int], ...]:
    if shell not in (1, 2):
        raise ValueError("shell must be one or two")
    weighted = square_j1j2_bonds(
        linear_size,
        linear_size,
        j1=1.0 if shell == 1 else 0.0,
        j2=1.0 if shell == 2 else 0.0,
        periodic=True,
    )
    return tuple((int(first), int(second)) for first, second, value in weighted if value)


@lru_cache(maxsize=None)
def _plaquettes_and_rectangles(
    linear_size: int,
) -> tuple[
    tuple[tuple[int, int, int, int], ...],
    tuple[tuple[int, int, int, int, int, int], ...],
]:
    plaquettes = []
    rectangles = []
    for y in range(linear_size):
        for x in range(linear_size):
            plaquettes.append(
                (
                    _site(x, y, linear_size),
                    _site(x + 1, y, linear_size),
                    _site(x + 1, y + 1, linear_size),
                    _site(x, y + 1, linear_size),
                )
            )
            rectangles.extend(
                (
                    (
                        _site(x, y, linear_size),
                        _site(x + 1, y, linear_size),
                        _site(x + 2, y, linear_size),
                        _site(x + 2, y + 1, linear_size),
                        _site(x + 1, y + 1, linear_size),
                        _site(x, y + 1, linear_size),
                    ),
                    (
                        _site(x, y, linear_size),
                        _site(x, y + 1, linear_size),
                        _site(x, y + 2, linear_size),
                        _site(x + 1, y + 2, linear_size),
                        _site(x + 1, y + 1, linear_size),
                        _site(x + 1, y, linear_size),
                    ),
                )
            )
    if any(len(set(cycle)) != len(cycle) for cycle in (*plaquettes, *rectangles)):
        raise ValueError("registered square loops alias on this torus")
    return tuple(plaquettes), tuple(rectangles)


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


def square_a1_scalar_token_connectivity(
    spins: Sequence[int], linear_size: int
) -> dict[str, tuple[tuple[tuple[int, ...], ...], np.ndarray]]:
    """Return exact connectivity for the five square scalar tokens."""

    old = _checked_configuration(spins, linear_size)
    j1 = heisenberg_spin_connectivity(old, _shell_bonds(linear_size, 1))
    j2 = heisenberg_spin_connectivity(old, _shell_bonds(linear_size, 2))
    plaquettes, rectangles = _plaquettes_and_rectangles(linear_size)

    ring: dict[tuple[int, ...], float] = {}
    dimer: dict[tuple[int, ...], float] = {}
    for cycle in plaquettes:
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

    rectangle: dict[tuple[int, ...], float] = {}
    for cycle in rectangles:
        _add(rectangle, _cycle_configuration(old, cycle), 1.0)
        _add(rectangle, _cycle_configuration(old, tuple(reversed(cycle))), 1.0)
    return {
        "J1_shell": j1,
        "J2_shell": j2,
        "plaquette_ring_exchange": _finalize(ring),
        "plaquette_dimer_resonance": _finalize(dimer),
        "six_site_rectangle_rearrangement": _finalize(rectangle),
    }


def square_a1_scalar_connectivity(
    spins: Sequence[int], linear_size: int, weights: Sequence[float]
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    """Combine token connectivity with one finite real weight per token."""

    coefficients = np.asarray(weights, dtype=np.float64)
    if coefficients.shape != (len(SQUARE_A1_SCALAR_LABELS),) or np.any(
        ~np.isfinite(coefficients)
    ):
        raise ValueError("weights must contain five finite real coefficients")
    tokens = square_a1_scalar_token_connectivity(spins, linear_size)
    combined: dict[tuple[int, ...], float] = {}
    for coefficient, label in zip(coefficients, SQUARE_A1_SCALAR_LABELS, strict=True):
        states, values = tokens[label]
        for state, value in zip(states, values, strict=True):
            _add(combined, state, coefficient * float(value))
    return _finalize(combined)


__all__ = [
    "SQUARE_A1_SCALAR_LABELS",
    "square_a1_scalar_connectivity",
    "square_a1_scalar_token_connectivity",
]
