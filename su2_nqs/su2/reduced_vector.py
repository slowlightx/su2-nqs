"""Pure-numeric reduced matrix elements of weighted spin vectors.

This module compiles

``B^(1) = sum_i c_i S_i``

on a left-associated block of spin-1/2 sites.  Every angular-momentum label
is doubled: the tensor rank is ``q_op=2`` and the only possible total-spin
changes are ``q_new-q_old in (-2, 0, 2)``.  Sparse entries always follow the
project convention ``B[new_path, old_path]``.

The spherical convention is Condon--Shortley,

``T_0 = S_z, T_+1 = -S_+ / sqrt(2), T_-1 = S_- / sqrt(2)``,

and reduced matrix elements are defined by

``<new,q',m'|T_mu|old,q,m> = CG(q,m;2,mu|q',m') R/sqrt(q'+1)``.

The main compiler advances one spin at a time using cached finite CG sums;
it never materializes a ``2**N`` Hilbert space.  A deliberately exponential
compiler lives in ``benchmarks.reduced_vector_reference`` as an independent
offline small-system reference.  No path in this module calls SymPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Integral, Number
from typing import Mapping, Sequence

import numpy as np

from .clebsch_gordan import clebsch_gordan, spin_half_cg
from .fusion_path import FusionPath, enumerate_paths, is_valid_path
from .quantum_numbers import (
    is_admissible_triangle,
    magnetic_numbers,
    validate_m,
    validate_q,
)


Q_OP_VECTOR = 2
VECTOR_DELTA_Q: tuple[int, int, int] = (-2, 0, 2)
SPHERICAL_MU_Q: tuple[int, int, int] = (-2, 0, 2)


def _coerce_weight(value: complex, *, name: str) -> complex:
    if isinstance(value, bool) or not isinstance(value, Number):
        raise TypeError(f"{name} must be a finite real or complex number")
    answer = complex(value)
    if not math.isfinite(answer.real) or not math.isfinite(answer.imag):
        raise ValueError(f"{name} must be finite")
    return answer


def _coerce_weights(weights: Sequence[complex]) -> np.ndarray:
    try:
        raw = tuple(weights)
    except TypeError as error:
        raise TypeError("endpoint_weights must be a finite numeric sequence") from error
    if not raw:
        raise ValueError("endpoint_weights must contain at least one site")
    return np.asarray(
        [_coerce_weight(value, name=f"endpoint_weights[{index}]") for index, value in enumerate(raw)],
        dtype=np.complex128,
    )


@lru_cache(maxsize=None)
def _all_paths(n_sites: int) -> tuple[FusionPath, ...]:
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be an integer")
    if n_sites < 1:
        raise ValueError("n_sites must be positive")
    parity = int(n_sites) % 2
    return tuple(
        path
        for q in range(parity, int(n_sites) + 1, 2)
        for path in enumerate_paths(int(n_sites), q)
    )


def _extended_paths(path: FusionPath) -> tuple[FusionPath, ...]:
    q = path[-1]
    lower = (path + (q - 1,),) if q > 0 else ()
    return lower + (path + (q + 1,),)


def _spherical_spin_half_element(mu_q: int, m_new: int, m_old: int) -> float:
    """Return ``<m_new|S_mu|m_old>`` in doubled labels."""

    if mu_q not in SPHERICAL_MU_Q:
        raise ValueError("a vector component must have mu_q=-2, 0, or 2")
    if m_new not in (-1, 1) or m_old not in (-1, 1):
        raise ValueError("spin-1/2 projections must be doubled labels -1 or +1")
    if mu_q == 0 and m_new == m_old:
        return 0.5 * m_old
    if mu_q == 2 and (m_new, m_old) == (1, -1):
        return -1.0 / math.sqrt(2.0)
    if mu_q == -2 and (m_new, m_old) == (-1, 1):
        return 1.0 / math.sqrt(2.0)
    return 0.0


def _project_reduced_element(
    q_new: int,
    q_old: int,
    magnetic_element,
) -> complex:
    """Project magnetic matrix elements onto the chosen Wigner--Eckart gauge."""

    if not is_admissible_triangle(q_old, Q_OP_VECTOR, q_new):
        return 0.0j
    total = 0.0j
    for m_old in magnetic_numbers(q_old):
        for mu_q in SPHERICAL_MU_Q:
            m_new = m_old + mu_q
            if abs(m_new) > q_new or (q_new - m_new) % 2:
                continue
            coefficient = clebsch_gordan(
                q_old, m_old, Q_OP_VECTOR, mu_q, q_new, m_new
            )
            if coefficient:
                total += coefficient * magnetic_element(
                    m_new, m_old, mu_q
                )
    return total / math.sqrt(q_new + 1.0)


@lru_cache(maxsize=None)
def _old_block_advance_coefficient(
    q_prefix_new: int,
    q_prefix_old: int,
    q_new: int,
    q_old: int,
) -> float:
    """Recouple a unit reduced vector on the old block through one spin."""

    if not (
        is_admissible_triangle(q_prefix_new, 1, q_new)
        and is_admissible_triangle(q_prefix_old, 1, q_old)
        and is_admissible_triangle(q_prefix_old, Q_OP_VECTOR, q_prefix_new)
        and is_admissible_triangle(q_old, Q_OP_VECTOR, q_new)
    ):
        return 0.0

    def element(m_new: int, m_old: int, mu_q: int) -> complex:
        answer = 0.0
        for m_spin in (-1, 1):
            m_prefix_new = m_new - m_spin
            m_prefix_old = m_old - m_spin
            if (
                abs(m_prefix_new) > q_prefix_new
                or (q_prefix_new - m_prefix_new) % 2
                or abs(m_prefix_old) > q_prefix_old
                or (q_prefix_old - m_prefix_old) % 2
            ):
                continue
            bra = spin_half_cg(
                q_prefix_new, m_prefix_new, m_spin, q_new
            )
            ket = spin_half_cg(
                q_prefix_old, m_prefix_old, m_spin, q_old
            )
            reduced = clebsch_gordan(
                q_prefix_old,
                m_prefix_old,
                Q_OP_VECTOR,
                mu_q,
                q_prefix_new,
                m_prefix_new,
            ) / math.sqrt(q_prefix_new + 1.0)
            answer += bra * reduced * ket
        return answer

    value = _project_reduced_element(q_new, q_old, element)
    if abs(value.imag) > 2.0e-14:
        raise RuntimeError("real CG recoupling unexpectedly produced a complex value")
    return float(value.real)


@lru_cache(maxsize=None)
def _endpoint_advance_coefficient(
    q_prefix: int,
    q_new: int,
    q_old: int,
) -> float:
    """Reduced coefficient of the newly appended physical spin vector."""

    if not (
        is_admissible_triangle(q_prefix, 1, q_new)
        and is_admissible_triangle(q_prefix, 1, q_old)
        and is_admissible_triangle(q_old, Q_OP_VECTOR, q_new)
    ):
        return 0.0

    def element(m_new: int, m_old: int, mu_q: int) -> complex:
        answer = 0.0
        for m_prefix in magnetic_numbers(q_prefix):
            for m_spin_old in (-1, 1):
                if m_prefix + m_spin_old != m_old:
                    continue
                for m_spin_new in (-1, 1):
                    if m_prefix + m_spin_new != m_new:
                        continue
                    spin_value = _spherical_spin_half_element(
                        mu_q, m_spin_new, m_spin_old
                    )
                    if spin_value:
                        answer += (
                            spin_half_cg(
                                q_prefix, m_prefix, m_spin_new, q_new
                            )
                            * spin_value
                            * spin_half_cg(
                                q_prefix, m_prefix, m_spin_old, q_old
                            )
                        )
        return answer

    value = _project_reduced_element(q_new, q_old, element)
    if abs(value.imag) > 2.0e-14:
        raise RuntimeError("real CG recoupling unexpectedly produced a complex value")
    return float(value.real)


@dataclass(frozen=True)
class ReducedVectorTable:
    """Sparse reduced matrix elements indexed by left-fusion paths.

    ``new_path_indices`` and ``old_path_indices`` index ``paths``.  The
    parallel ``q_new`` and ``q_old`` arrays are included deliberately so a
    numeric runtime kernel does not need to inspect Python path tuples.
    """

    n_sites: int
    endpoint_weights: np.ndarray
    paths: tuple[FusionPath, ...]
    path_q: np.ndarray
    new_path_indices: np.ndarray
    old_path_indices: np.ndarray
    q_new: np.ndarray
    q_old: np.ndarray
    reduced_values: np.ndarray
    compiler: str
    q_op: int = Q_OP_VECTOR
    delta_q: tuple[int, int, int] = VECTOR_DELTA_Q

    def __post_init__(self) -> None:
        if self.n_sites < 1 or len(self.endpoint_weights) != self.n_sites:
            raise ValueError("table size and endpoint weights disagree")
        if self.q_op != Q_OP_VECTOR or self.delta_q != VECTOR_DELTA_Q:
            raise ValueError("this table represents only a q_op=2 vector")
        expected_paths = _all_paths(self.n_sites)
        if tuple(self.paths) != expected_paths:
            raise ValueError("paths are not in canonical all-irrep order")
        if np.asarray(self.path_q).shape != (len(self.paths),):
            raise ValueError("path_q must have one entry per path")
        count = len(self.reduced_values)
        arrays = (
            self.new_path_indices,
            self.old_path_indices,
            self.q_new,
            self.q_old,
        )
        if any(np.asarray(value).shape != (count,) for value in arrays):
            raise ValueError("all sparse entry arrays must have equal length")
        if count:
            if np.any(self.new_path_indices < 0) or np.any(
                self.new_path_indices >= len(self.paths)
            ):
                raise ValueError("new path index lies outside the basis")
            if np.any(self.old_path_indices < 0) or np.any(
                self.old_path_indices >= len(self.paths)
            ):
                raise ValueError("old path index lies outside the basis")
            if any(int(delta) not in self.delta_q for delta in self.q_new - self.q_old):
                raise ValueError("a sparse entry violates the q_op=2 selection rule")
            if np.any(self.q_new != self.path_q[self.new_path_indices]) or np.any(
                self.q_old != self.path_q[self.old_path_indices]
            ):
                raise ValueError("entry q labels disagree with their paths")
        if np.any(~np.isfinite(self.endpoint_weights)) or np.any(
            ~np.isfinite(self.reduced_values)
        ):
            raise ValueError("numeric table values must be finite")

    @property
    def entry_count(self) -> int:
        return int(len(self.reduced_values))

    def entry_dict(self) -> dict[tuple[FusionPath, FusionPath], complex]:
        """Return sparse entries in explicit ``(new_path, old_path)`` order."""

        return {
            (self.paths[int(new)], self.paths[int(old)]): complex(value)
            for new, old, value in zip(
                self.new_path_indices,
                self.old_path_indices,
                self.reduced_values,
                strict=True,
            )
        }

    def reduced_element(
        self, new_path: Sequence[int], old_path: Sequence[int]
    ) -> complex:
        """Look up ``<new||B^(1)||old>``; absent sparse entries are zero."""

        new = tuple(new_path)
        old = tuple(old_path)
        if not is_valid_path(new) or len(new) != self.n_sites:
            raise ValueError("new_path is not a valid path for this block")
        if not is_valid_path(old) or len(old) != self.n_sites:
            raise ValueError("old_path is not a valid path for this block")
        return self.entry_dict().get((new, old), 0.0j)

    def dense_block(
        self, q_new: int, q_old: int
    ) -> tuple[tuple[FusionPath, ...], tuple[FusionPath, ...], np.ndarray]:
        """Return one multiplicity block with rows ``new`` and columns ``old``."""

        validate_q(q_new, name="q_new")
        validate_q(q_old, name="q_old")
        new_paths = tuple(path for path in self.paths if path[-1] == q_new)
        old_paths = tuple(path for path in self.paths if path[-1] == q_old)
        matrix = np.zeros((len(new_paths), len(old_paths)), dtype=np.complex128)
        new_position = {path: index for index, path in enumerate(new_paths)}
        old_position = {path: index for index, path in enumerate(old_paths)}
        for (new, old), value in self.entry_dict().items():
            if new[-1] == q_new and old[-1] == q_old:
                matrix[new_position[new], old_position[old]] = value
        return new_paths, old_paths, matrix

    def numeric_payload(self) -> dict[str, np.ndarray]:
        """Return only plain arrays suitable for ``np.savez`` and JAX loading."""

        return {
            "n_sites": np.asarray(self.n_sites, dtype=np.int32),
            "q_op": np.asarray(self.q_op, dtype=np.int32),
            "delta_q": np.asarray(self.delta_q, dtype=np.int32),
            "endpoint_weights": np.asarray(
                self.endpoint_weights, dtype=np.complex128
            ),
            "paths": np.asarray(self.paths, dtype=np.int32),
            "path_q": np.asarray(self.path_q, dtype=np.int32),
            "new_path_indices": np.asarray(
                self.new_path_indices, dtype=np.int32
            ),
            "old_path_indices": np.asarray(
                self.old_path_indices, dtype=np.int32
            ),
            "q_new": np.asarray(self.q_new, dtype=np.int32),
            "q_old": np.asarray(self.q_old, dtype=np.int32),
            "reduced_values": np.asarray(
                self.reduced_values, dtype=np.complex128
            ),
        }


def _pack_table(
    n_sites: int,
    endpoint_weights: np.ndarray,
    entries: Mapping[tuple[FusionPath, FusionPath], complex],
    *,
    compiler: str,
    zero_tolerance: float,
) -> ReducedVectorTable:
    if zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be non-negative")
    paths = _all_paths(n_sites)
    positions = {path: index for index, path in enumerate(paths)}
    retained = []
    for key, raw_value in entries.items():
        try:
            raw_new, raw_old = key
        except (TypeError, ValueError) as error:
            raise ValueError(
                "reduced entry keys must be (new_path, old_path) pairs"
            ) from error
        new, old = tuple(raw_new), tuple(raw_old)
        if new not in positions or old not in positions:
            raise ValueError("a reduced entry path is not in this block basis")
        value = _coerce_weight(raw_value, name="reduced entry")
        if abs(value) > zero_tolerance:
            retained.append((positions[new], positions[old], value))
    retained.sort(key=lambda item: (item[0], item[1]))
    new_indices = np.asarray([item[0] for item in retained], dtype=np.int32)
    old_indices = np.asarray([item[1] for item in retained], dtype=np.int32)
    values = np.asarray([item[2] for item in retained], dtype=np.complex128)
    path_q = np.asarray([path[-1] for path in paths], dtype=np.int32)
    return ReducedVectorTable(
        n_sites=n_sites,
        endpoint_weights=np.asarray(endpoint_weights, dtype=np.complex128),
        paths=paths,
        path_q=path_q,
        new_path_indices=new_indices,
        old_path_indices=old_indices,
        q_new=path_q[new_indices],
        q_old=path_q[old_indices],
        reduced_values=values,
        compiler=compiler,
    )


def advance_weighted_spin_vector(
    table: ReducedVectorTable,
    endpoint_weight: complex,
    *,
    zero_tolerance: float = 2.0e-14,
) -> ReducedVectorTable:
    """Append one spin-1/2 and update the reduced table by cached CG sums.

    This is the production-facing advance API.  Its cost follows the sparse
    multiplicity entries rather than ``2**N`` magnetic states.  The returned
    arrays are still an exact multiplicity-space operator and can therefore
    grow exponentially with block size; a neural forward pass should cache
    only the active endpoint/channel slices it needs.
    """

    weight = _coerce_weight(endpoint_weight, name="endpoint_weight")
    accumulated: dict[tuple[FusionPath, FusionPath], complex] = {}

    for (prefix_new, prefix_old), old_value in table.entry_dict().items():
        for new in _extended_paths(prefix_new):
            for old in _extended_paths(prefix_old):
                coefficient = _old_block_advance_coefficient(
                    prefix_new[-1], prefix_old[-1], new[-1], old[-1]
                )
                if coefficient:
                    key = (new, old)
                    accumulated[key] = accumulated.get(key, 0.0j) + coefficient * old_value

    if weight:
        for prefix in table.paths:
            for new in _extended_paths(prefix):
                for old in _extended_paths(prefix):
                    coefficient = _endpoint_advance_coefficient(
                        prefix[-1], new[-1], old[-1]
                    )
                    if coefficient:
                        key = (new, old)
                        accumulated[key] = accumulated.get(key, 0.0j) + coefficient * weight

    return _pack_table(
        table.n_sites + 1,
        np.concatenate(
            [table.endpoint_weights, np.asarray([weight], dtype=np.complex128)]
        ),
        accumulated,
        compiler="recursive_cg",
        zero_tolerance=zero_tolerance,
    )


def compile_weighted_spin_vector(
    endpoint_weights: Sequence[complex],
    *,
    zero_tolerance: float = 2.0e-14,
) -> ReducedVectorTable:
    """Compile ``sum_i c_i S_i`` by exact one-site reduced recurrences."""

    weights = _coerce_weights(endpoint_weights)
    initial_path: FusionPath = (1,)
    table = _pack_table(
        1,
        weights[:1],
        {(initial_path, initial_path): math.sqrt(1.5) * weights[0]},
        compiler="recursive_cg",
        zero_tolerance=zero_tolerance,
    )
    for weight in weights[1:]:
        table = advance_weighted_spin_vector(
            table, weight, zero_tolerance=zero_tolerance
        )
    return table


def make_reduced_vector_table(
    endpoint_weights: Sequence[complex],
    entries: Mapping[tuple[FusionPath, FusionPath], complex],
    *,
    compiler: str = "numeric_entries",
    zero_tolerance: float = 2.0e-14,
) -> ReducedVectorTable:
    """Pack externally compiled reduced entries into the canonical payload.

    This boundary lets offline reference compilers live in the benchmark
    layer without introducing a reverse ``su2 -> benchmarks`` dependency.
    Entry keys use the mandatory ``(new_path, old_path)`` orientation.
    """

    weights = _coerce_weights(endpoint_weights)
    return _pack_table(
        len(weights),
        weights,
        entries,
        compiler=compiler,
        zero_tolerance=zero_tolerance,
    )


def coupled_multiplet_basis(
    table: ReducedVectorTable,
) -> tuple[tuple[FusionPath, int], ...]:
    """Return ``(path,m)`` labels for the complete coupled block basis."""

    return tuple(
        (path, m) for path in table.paths for m in magnetic_numbers(path[-1])
    )


def spherical_component_coupled_matrix(
    table: ReducedVectorTable, mu_q: int
) -> tuple[np.ndarray, tuple[tuple[FusionPath, int], ...]]:
    """Expand one component as a coupled-basis matrix ``T[new,old]``."""

    if mu_q not in SPHERICAL_MU_Q:
        raise ValueError("mu_q must be -2, 0, or 2")
    basis = coupled_multiplet_basis(table)
    positions = {state: index for index, state in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    for (new_path, old_path), reduced in table.entry_dict().items():
        q_new, q_old = new_path[-1], old_path[-1]
        for m_old in magnetic_numbers(q_old):
            m_new = m_old + mu_q
            if abs(m_new) > q_new or (q_new - m_new) % 2:
                continue
            coefficient = clebsch_gordan(
                q_old, m_old, Q_OP_VECTOR, mu_q, q_new, m_new
            )
            matrix[
                positions[(new_path, m_new)], positions[(old_path, m_old)]
            ] = coefficient * reduced / math.sqrt(q_new + 1.0)
    return matrix, basis


def coupled_block_scalar_basis(
    left: ReducedVectorTable,
    right: ReducedVectorTable,
    target_q: int,
) -> tuple[tuple[FusionPath, FusionPath], ...]:
    """Multiplicity basis obtained by coupling two blocks to ``target_q``."""

    validate_q(target_q, name="target_q")
    return tuple(
        (left_path, right_path)
        for left_path in left.paths
        for right_path in right.paths
        if is_admissible_triangle(left_path[-1], right_path[-1], target_q)
    )


@lru_cache(maxsize=None)
def _scalar_reduced_coefficient(
    q_left_new: int,
    q_right_new: int,
    q_left_old: int,
    q_right_old: int,
    target_q: int,
    total_m: int,
) -> float:
    """Magnetic contraction multiplying two unit reduced matrix elements."""

    if not (
        is_admissible_triangle(q_left_new, q_right_new, target_q)
        and is_admissible_triangle(q_left_old, q_right_old, target_q)
        and is_admissible_triangle(q_left_old, Q_OP_VECTOR, q_left_new)
        and is_admissible_triangle(q_right_old, Q_OP_VECTOR, q_right_new)
    ):
        return 0.0
    answer = 0.0
    for m_left_old in magnetic_numbers(q_left_old):
        m_right_old = total_m - m_left_old
        if (
            abs(m_right_old) > q_right_old
            or (q_right_old - m_right_old) % 2
        ):
            continue
        ket = clebsch_gordan(
            q_left_old,
            m_left_old,
            q_right_old,
            m_right_old,
            target_q,
            total_m,
        )
        if not ket:
            continue
        for mu_q in SPHERICAL_MU_Q:
            m_left_new = m_left_old + mu_q
            m_right_new = m_right_old - mu_q
            if (
                abs(m_left_new) > q_left_new
                or (q_left_new - m_left_new) % 2
                or abs(m_right_new) > q_right_new
                or (q_right_new - m_right_new) % 2
            ):
                continue
            bra = clebsch_gordan(
                q_left_new,
                m_left_new,
                q_right_new,
                m_right_new,
                target_q,
                total_m,
            )
            left_component = clebsch_gordan(
                q_left_old,
                m_left_old,
                Q_OP_VECTOR,
                mu_q,
                q_left_new,
                m_left_new,
            ) / math.sqrt(q_left_new + 1.0)
            right_component = clebsch_gordan(
                q_right_old,
                m_right_old,
                Q_OP_VECTOR,
                -mu_q,
                q_right_new,
                m_right_new,
            ) / math.sqrt(q_right_new + 1.0)
            phase = 1.0 if mu_q == 0 else -1.0
            answer += bra * phase * left_component * right_component * ket
    return float(answer)


def old_block_vector_advance_coefficient(
    q_prefix_new: int,
    q_prefix_old: int,
    q_new: int,
    q_old: int,
) -> float:
    """Return the cached coefficient for transporting an old-block vector.

    All arguments are doubled angular momenta.  The coefficient multiplies
    ``<prefix_new||B^(1)||prefix_old>`` after appending one spin-half, with
    the mandatory matrix orientation ``[new, old]``.  Invalid triangles
    return zero.  This small public boundary lets compiled recurrence kernels
    reuse exactly the same Condon--Shortley table as
    :func:`advance_weighted_spin_vector` without materializing path tables.
    """

    for name, value in (
        ("q_prefix_new", q_prefix_new),
        ("q_prefix_old", q_prefix_old),
        ("q_new", q_new),
        ("q_old", q_old),
    ):
        validate_q(value, name=name)
    return _old_block_advance_coefficient(
        q_prefix_new, q_prefix_old, q_new, q_old
    )


def appended_spin_vector_source_coefficient(
    q_prefix: int,
    q_new: int,
    q_old: int,
) -> float:
    """Return the cached coefficient of the newly appended spin vector.

    The result multiplies the physical endpoint weight in
    ``sum_i c_i S_i``.  Labels are doubled and rows/columns are respectively
    the new/old coupled states.  Invalid triangles return zero.
    """

    validate_q(q_prefix, name="q_prefix")
    validate_q(q_new, name="q_new")
    validate_q(q_old, name="q_old")
    return _endpoint_advance_coefficient(q_prefix, q_new, q_old)


def appended_spin_scalar_contraction_coefficient(
    q_prefix_new: int,
    q_prefix_old: int,
    q_total: int,
) -> float:
    """Contract an old-block vector with the appended spin into a scalar.

    This returns the coefficient multiplying
    ``<prefix_new||B^(1)||prefix_old>`` in the scalar matrix element of
    ``B^(1).S`` between states where the old block is coupled with the new
    spin-half to the same doubled total spin ``q_total``.  The single-spin
    reduced matrix element ``sqrt(3/2)`` is included.  Hence for two spins
    the coefficient times ``<1||S||1>=sqrt(3/2)`` gives ``-3/4`` in the
    singlet and ``+1/4`` in the triplet.
    """

    validate_q(q_prefix_new, name="q_prefix_new")
    validate_q(q_prefix_old, name="q_prefix_old")
    validate_q(q_total, name="q_total")
    return math.sqrt(1.5) * _scalar_reduced_coefficient(
        q_prefix_new,
        1,
        q_prefix_old,
        1,
        q_total,
        q_total,
    )


def contract_reduced_vector_scalar(
    left: ReducedVectorTable,
    right: ReducedVectorTable,
    target_q: int,
    *,
    total_m: int | None = None,
) -> tuple[np.ndarray, tuple[tuple[FusionPath, FusionPath], ...]]:
    """Contract ``A^(1).B^(1)`` in a two-block coupled multiplicity basis.

    Rows are new block-path pairs and columns are old pairs.  The contraction
    is bilinear in endpoint weights (there is intentionally no conjugation).
    """

    validate_q(target_q, name="target_q")
    if total_m is None:
        total_m = target_q
    validate_m(target_q, total_m, name="total_m")
    basis = coupled_block_scalar_basis(left, right, target_q)
    positions = {state: index for index, state in enumerate(basis)}
    result = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    left_entries = left.entry_dict()
    right_entries = right.entry_dict()
    for (left_new, left_old), left_value in left_entries.items():
        for (right_new, right_old), right_value in right_entries.items():
            new_state = (left_new, right_new)
            old_state = (left_old, right_old)
            row = positions.get(new_state)
            column = positions.get(old_state)
            if row is None or column is None:
                continue
            coefficient = _scalar_reduced_coefficient(
                left_new[-1],
                right_new[-1],
                left_old[-1],
                right_old[-1],
                target_q,
                total_m,
            )
            if coefficient:
                result[row, column] += coefficient * left_value * right_value
    return result, basis


__all__ = [
    "Q_OP_VECTOR",
    "ReducedVectorTable",
    "SPHERICAL_MU_Q",
    "VECTOR_DELTA_Q",
    "advance_weighted_spin_vector",
    "appended_spin_scalar_contraction_coefficient",
    "appended_spin_vector_source_coefficient",
    "compile_weighted_spin_vector",
    "contract_reduced_vector_scalar",
    "coupled_block_scalar_basis",
    "coupled_multiplet_basis",
    "make_reduced_vector_table",
    "old_block_vector_advance_coefficient",
    "spherical_component_coupled_matrix",
]
