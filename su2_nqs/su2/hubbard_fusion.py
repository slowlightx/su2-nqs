"""Spin-SU(2) fusion labels for the four-state Hubbard site space.

The conserved charge is represented explicitly by integer local occupations.
Spin labels and projections continue to use doubled integers.  This module is
kept separate from :mod:`fusion_path`, whose public contract assumes one
spin-1/2 irrep on every site and therefore cannot represent holes or doublons.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

from .quantum_numbers import coupling_range, validate_q


LOCAL_OCCUPATIONS = (0, 1, 2)


def local_spin_q(occupation: int) -> int:
    """Return the doubled spin of one Hubbard-site charge state."""

    if isinstance(occupation, bool) or not isinstance(occupation, Integral):
        raise TypeError("occupation must be an integer")
    value = int(occupation)
    if value not in LOCAL_OCCUPATIONS:
        raise ValueError("occupation must be 0, 1, or 2")
    return 1 if value == 1 else 0


@dataclass(frozen=True, order=True)
class HubbardFusionState:
    """Local charges and left-associated doubled-spin labels."""

    occupations: tuple[int, ...]
    path: tuple[int, ...]

    @property
    def n_sites(self) -> int:
        return len(self.occupations)

    @property
    def n_electrons(self) -> int:
        return sum(self.occupations)

    @property
    def target_q(self) -> int:
        return self.path[-1]


def is_valid_hubbard_fusion_state(
    state: HubbardFusionState,
    *,
    n_electrons: int | None = None,
    target_q: int | None = None,
) -> bool:
    """Check charge labels and every left-associated SU(2) fusion step."""

    if not isinstance(state, HubbardFusionState):
        return False
    occupations = tuple(state.occupations)
    path = tuple(state.path)
    if not occupations or len(path) != len(occupations):
        return False
    if any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) not in LOCAL_OCCUPATIONS
        for value in occupations
    ):
        return False
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in path):
        return False
    running_q = 0
    for occupation, next_q in zip(occupations, path):
        next_q = int(next_q)
        if next_q not in coupling_range(running_q, local_spin_q(int(occupation))):
            return False
        running_q = next_q
    if n_electrons is not None and sum(int(value) for value in occupations) != n_electrons:
        return False
    if target_q is not None and running_q != target_q:
        return False
    return True


@lru_cache(maxsize=None)
def enumerate_hubbard_fusion_states(
    n_sites: int,
    n_electrons: int,
    target_q: int,
) -> tuple[HubbardFusionState, ...]:
    """Enumerate a deterministic fixed-charge, fixed-total-spin basis."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    if (
        isinstance(n_electrons, bool)
        or not isinstance(n_electrons, Integral)
        or not 0 <= n_electrons <= 2 * n_sites
    ):
        raise ValueError("n_electrons must lie between 0 and 2*n_sites")
    validate_q(target_q, name="target_q")
    n_sites = int(n_sites)
    n_electrons = int(n_electrons)
    target_q = int(target_q)
    states: list[HubbardFusionState] = []

    def visit(
        occupations: tuple[int, ...],
        path: tuple[int, ...],
        charge: int,
        running_q: int,
    ) -> None:
        sites_used = len(occupations)
        if sites_used == n_sites:
            if charge == n_electrons and running_q == target_q:
                states.append(HubbardFusionState(occupations, path))
            return
        remaining_after = n_sites - sites_used - 1
        for occupation in LOCAL_OCCUPATIONS:
            next_charge = charge + occupation
            charge_left = n_electrons - next_charge
            if not 0 <= charge_left <= 2 * remaining_after:
                continue
            for next_q in coupling_range(running_q, local_spin_q(occupation)):
                # Each remaining site can change doubled spin by at most one.
                if abs(next_q - target_q) > remaining_after:
                    continue
                visit(
                    occupations + (occupation,),
                    path + (next_q,),
                    next_charge,
                    next_q,
                )

    visit((), (), 0, 0)
    return tuple(states)


def count_hubbard_fusion_states(n_sites: int, n_electrons: int, target_q: int) -> int:
    """Return the multiplicity dimension of one charge/spin sector."""

    return len(enumerate_hubbard_fusion_states(n_sites, n_electrons, target_q))


def occupations_from_fock_mask(mask: int, n_sites: int) -> tuple[int, ...]:
    """Return site occupations for interleaved ``(up,down)`` orbitals."""

    if isinstance(mask, bool) or not isinstance(mask, Integral) or mask < 0:
        raise ValueError("mask must be a non-negative integer")
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    if int(mask) >= 1 << (2 * int(n_sites)):
        raise ValueError("mask contains orbitals outside the lattice")
    return tuple(
        ((int(mask) >> (2 * site)) & 1) + ((int(mask) >> (2 * site + 1)) & 1)
        for site in range(int(n_sites))
    )


def encode_hubbard_fusion_state(state: HubbardFusionState) -> tuple[int, ...]:
    """Interleave ``(occupation, cumulative_q)`` labels for neural/VMC APIs."""

    if not is_valid_hubbard_fusion_state(state):
        raise ValueError(f"invalid Hubbard fusion state: {state!r}")
    return tuple(
        value
        for pair in zip(state.occupations, state.path)
        for value in pair
    )


def decode_hubbard_fusion_state(labels: Sequence[int]) -> HubbardFusionState:
    """Decode interleaved neural labels into a validated fusion state."""

    raw = tuple(labels)
    if not raw or len(raw) % 2:
        raise ValueError("Hubbard fusion labels must have positive even length")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
        raise TypeError("Hubbard fusion labels must be integers")
    state = HubbardFusionState(
        tuple(int(value) for value in raw[0::2]),
        tuple(int(value) for value in raw[1::2]),
    )
    if not is_valid_hubbard_fusion_state(state):
        raise ValueError(f"invalid Hubbard fusion labels: {raw!r}")
    return state


__all__ = [
    "HubbardFusionState",
    "LOCAL_OCCUPATIONS",
    "count_hubbard_fusion_states",
    "decode_hubbard_fusion_state",
    "encode_hubbard_fusion_state",
    "enumerate_hubbard_fusion_states",
    "is_valid_hubbard_fusion_state",
    "local_spin_q",
    "occupations_from_fock_mask",
]
