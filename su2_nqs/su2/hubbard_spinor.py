"""Reduced ``q_op=1`` Hubbard hopping with explicit fermion parity.

The local annihilation and creation operators transform as doubled-rank-one
spinors.  A directed spin-preserving hop couples the two spinors to a total
scalar.  This module evaluates that scalar directly between left-associated
charge/spin fusion paths.  It propagates only block magnetizations appearing
in finite Clebsch--Gordan sums; it never enumerates Fock masks or magnetic
configurations of the full lattice.

Fermionic statistics are kept separate from SU(2) recoupling.  The canonical
interleaved orbital order is ``(0 up, 0 down, 1 up, 1 down, ...)``.  A parity
sign is evaluated from the fixed local charges and the endpoint spinor
component before the reduced transfer is contracted.
"""

from __future__ import annotations

from numbers import Integral

from .clebsch_gordan import clebsch_gordan
from .hubbard_fusion import (
    HubbardFusionState,
    is_valid_hubbard_fusion_state,
    local_spin_q,
)


Q_OP_FERMION_SPINOR = 1
SPINOR_COMPONENTS = (1, -1)


def _local_mask(occupation: int, magnetic_q: int) -> int:
    """Encode one local Hubbard ket in canonical ``(up, down)`` order."""

    if occupation == 0:
        if magnetic_q != 0:
            raise ValueError("an empty site has only magnetic label zero")
        return 0
    if occupation == 1:
        if magnetic_q == 1:
            return 1
        if magnetic_q == -1:
            return 2
        raise ValueError("a singly occupied site has magnetic label +/-1")
    if occupation == 2:
        if magnetic_q != 0:
            raise ValueError("a doublon has only magnetic label zero")
        return 3
    raise ValueError("occupation must be 0, 1, or 2")


def _local_magnetic_q(mask: int) -> int:
    if mask == 0 or mask == 3:
        return 0
    if mask == 1:
        return 1
    if mask == 2:
        return -1
    raise ValueError("local mask must lie between zero and three")


def _identity_local_pairs(occupation: int) -> tuple[tuple[int, int], ...]:
    if occupation == 1:
        return ((1, 1), (-1, -1))
    if occupation in (0, 2):
        return ((0, 0),)
    raise ValueError("occupation must be 0, 1, or 2")


def _endpoint_local_pair(
    old_occupation: int,
    *,
    spinor_component: int,
    creation: bool,
) -> tuple[int, int, int, int] | None:
    """Return ``(m_new,m_old,new_mask,old_mask)`` for one odd local action."""

    if spinor_component not in SPINOR_COMPONENTS:
        raise ValueError("spinor_component must be doubled spin +/-1")
    orbital = 0 if spinor_component == 1 else 1
    bit = 1 << orbital
    if creation:
        if old_occupation == 0:
            old_mask = 0
        elif old_occupation == 1:
            old_mask = 1 << (1 - orbital)
        else:
            return None
        new_mask = old_mask | bit
    else:
        if old_occupation == 1:
            old_mask = bit
        elif old_occupation == 2:
            old_mask = 3
        else:
            return None
        new_mask = old_mask & ~bit
    return (
        _local_magnetic_q(new_mask),
        _local_magnetic_q(old_mask),
        new_mask,
        old_mask,
    )


def canonical_hopping_parity(
    occupations: tuple[int, ...],
    destination: int,
    source: int,
    spinor_component: int,
) -> int:
    """Return the canonical Fock sign for one allowed directed component.

    Non-endpoint singly occupied sites may use either local spin orientation:
    both orbitals of such a site lie on the same side of each endpoint orbital,
    so only their charge enters the parity.  The implementation chooses ``up``
    as a deterministic representative and never enumerates a Fock basis.
    """

    raw = tuple(occupations)
    if not raw or any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) not in (0, 1, 2)
        for value in raw
    ):
        raise ValueError("occupations must be a nonempty sequence of 0, 1, 2")
    n_sites = len(raw)
    if (
        isinstance(destination, bool)
        or isinstance(source, bool)
        or not isinstance(destination, Integral)
        or not isinstance(source, Integral)
        or destination == source
        or not 0 <= int(destination) < n_sites
        or not 0 <= int(source) < n_sites
    ):
        raise ValueError("destination and source must be distinct valid sites")
    if spinor_component not in SPINOR_COMPONENTS:
        raise ValueError("spinor_component must be doubled spin +/-1")
    destination = int(destination)
    source = int(source)
    created = _endpoint_local_pair(
        int(raw[destination]),
        spinor_component=spinor_component,
        creation=True,
    )
    annihilated = _endpoint_local_pair(
        int(raw[source]),
        spinor_component=spinor_component,
        creation=False,
    )
    if created is None or annihilated is None:
        raise ValueError("the requested hopping component is Pauli forbidden")

    mask = 0
    for site, occupation in enumerate(raw):
        if site == destination:
            local = created[3]
        elif site == source:
            local = annihilated[3]
        else:
            local = (0, 1, 3)[int(occupation)]
        mask |= local << (2 * site)
    source_orbital = 2 * source + (0 if spinor_component == 1 else 1)
    destination_orbital = 2 * destination + (
        0 if spinor_component == 1 else 1
    )
    annihilation_parity = (mask & ((1 << source_orbital) - 1)).bit_count()
    removed = mask & ~(1 << source_orbital)
    creation_parity = (
        removed & ((1 << destination_orbital) - 1)
    ).bit_count()
    return -1 if (annihilation_parity + creation_parity) % 2 else 1


def _component_transfer(
    new_state: HubbardFusionState,
    old_state: HubbardFusionState,
    destination: int,
    source: int,
    spinor_component: int,
) -> float:
    created = _endpoint_local_pair(
        old_state.occupations[destination],
        spinor_component=spinor_component,
        creation=True,
    )
    annihilated = _endpoint_local_pair(
        old_state.occupations[source],
        spinor_component=spinor_component,
        creation=False,
    )
    if created is None or annihilated is None:
        return 0.0
    if (
        created[2].bit_count() != new_state.occupations[destination]
        or annihilated[2].bit_count() != new_state.occupations[source]
    ):
        return 0.0

    amplitudes: dict[tuple[int, int], float] = {(0, 0): 1.0}
    q_prefix_new = 0
    q_prefix_old = 0
    for site, (new_occupation, old_occupation, q_new, q_old) in enumerate(
        zip(
            new_state.occupations,
            old_state.occupations,
            new_state.path,
            old_state.path,
            strict=True,
        )
    ):
        if site == destination:
            local_pairs = ((created[0], created[1]),)
        elif site == source:
            local_pairs = ((annihilated[0], annihilated[1]),)
        else:
            if new_occupation != old_occupation:
                return 0.0
            local_pairs = _identity_local_pairs(old_occupation)
        q_local_new = local_spin_q(new_occupation)
        q_local_old = local_spin_q(old_occupation)
        updated: dict[tuple[int, int], float] = {}
        for (m_prefix_new, m_prefix_old), amplitude in amplitudes.items():
            for m_local_new, m_local_old in local_pairs:
                m_new = m_prefix_new + m_local_new
                m_old = m_prefix_old + m_local_old
                bra = clebsch_gordan(
                    q_prefix_new,
                    m_prefix_new,
                    q_local_new,
                    m_local_new,
                    q_new,
                    m_new,
                )
                ket = clebsch_gordan(
                    q_prefix_old,
                    m_prefix_old,
                    q_local_old,
                    m_local_old,
                    q_old,
                    m_old,
                )
                value = amplitude * bra * ket
                if value:
                    key = (m_new, m_old)
                    updated[key] = updated.get(key, 0.0) + value
        amplitudes = updated
        q_prefix_new = q_new
        q_prefix_old = q_old
        if not amplitudes:
            return 0.0
    highest_weight = (new_state.target_q, old_state.target_q)
    parity = canonical_hopping_parity(
        old_state.occupations,
        destination,
        source,
        spinor_component,
    )
    return parity * amplitudes.get(highest_weight, 0.0)


def directed_hopping_reduced_element(
    new_state: HubbardFusionState,
    old_state: HubbardFusionState,
    destination: int,
    source: int,
) -> float:
    """Return ``<new|sum_sigma c^dagger_dst,sigma c_src,sigma|old>``.

    Rows are new fusion paths and columns are old fusion paths.  The two local
    ``q_op=1`` components are summed only after their explicit canonical
    fermion parities and reduced CG transfers have been evaluated.
    """

    if not is_valid_hubbard_fusion_state(
        new_state
    ) or not is_valid_hubbard_fusion_state(old_state):
        raise ValueError("new_state and old_state must be valid fusion states")
    if (
        new_state.n_sites != old_state.n_sites
        or new_state.n_electrons != old_state.n_electrons
        or new_state.target_q != old_state.target_q
    ):
        raise ValueError("states must belong to one charge/spin sector")
    n_sites = old_state.n_sites
    if (
        isinstance(destination, bool)
        or isinstance(source, bool)
        or not isinstance(destination, Integral)
        or not isinstance(source, Integral)
        or destination == source
        or not 0 <= int(destination) < n_sites
        or not 0 <= int(source) < n_sites
    ):
        raise ValueError("destination and source must be distinct valid sites")
    destination = int(destination)
    source = int(source)
    expected = list(old_state.occupations)
    expected[destination] += 1
    expected[source] -= 1
    if tuple(expected) != new_state.occupations:
        return 0.0
    if not 0 <= expected[destination] <= 2 or not 0 <= expected[source] <= 2:
        return 0.0
    return sum(
        _component_transfer(
            new_state,
            old_state,
            destination,
            source,
            component,
        )
        for component in SPINOR_COMPONENTS
    )


__all__ = [
    "Q_OP_FERMION_SPINOR",
    "SPINOR_COMPONENTS",
    "canonical_hopping_parity",
    "directed_hopping_reduced_element",
]
