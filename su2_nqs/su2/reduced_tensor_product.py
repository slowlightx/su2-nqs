"""Reduced products of two rank-one SU(2) operators.

This module couples two ``q_op=2`` :class:`ReducedVectorTable` objects into
the complete doubled-rank product channels ``q_op=0,2,4``.  It works only in
multiplicity space: the universal recoupling coefficient is evaluated from
finite Condon--Shortley CG sums and no computational-basis Hilbert space is
materialized.

For vector operators ``A`` and ``B`` the ordered convention is

``[A x B]^Q_M = sum_(mu,nu) CG(2,mu;2,nu|Q,M) A_mu B_nu``.

Sparse entries retain the project orientation ``T[new_path, old_path]``.
The Jordan and commutator helpers include the exchange phase of the coupled
components, so their second term is respectively ``B_nu A_mu`` rather than
``B_mu A_nu``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Number, Real
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
from .reduced_vector import (
    Q_OP_VECTOR,
    ReducedVectorTable,
    advance_weighted_spin_vector,
    compile_weighted_spin_vector,
)


VECTOR_PRODUCT_RANKS: tuple[int, int, int] = (0, 2, 4)
VECTOR_COMPONENTS: tuple[int, int, int] = (-2, 0, 2)


def tensor_delta_q(q_op: int) -> tuple[int, ...]:
    """Return allowed doubled total-spin changes for tensor rank ``q_op``."""

    rank = validate_q(q_op, name="q_op")
    if rank not in VECTOR_PRODUCT_RANKS:
        raise ValueError("a vector product must have q_op=0, 2, or 4")
    return tuple(range(-rank, rank + 1, 2))


@dataclass(frozen=True)
class ReducedTensorTable:
    """Sparse reduced matrix elements of one fixed-rank tensor operator."""

    n_sites: int
    q_op: int
    paths: tuple[FusionPath, ...]
    path_q: np.ndarray
    new_path_indices: np.ndarray
    old_path_indices: np.ndarray
    q_new: np.ndarray
    q_old: np.ndarray
    reduced_values: np.ndarray
    compiler: str
    delta_q: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.n_sites < 1:
            raise ValueError("n_sites must be positive")
        if self.q_op not in VECTOR_PRODUCT_RANKS:
            raise ValueError("a vector product table must have q_op=0, 2, or 4")
        if self.delta_q != tensor_delta_q(self.q_op):
            raise ValueError("delta_q disagrees with q_op")
        if not self.paths or any(
            not is_valid_path(path) or len(path) != self.n_sites
            for path in self.paths
        ):
            raise ValueError("paths must be valid paths for n_sites")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("paths must be unique")
        if np.asarray(self.path_q).shape != (len(self.paths),):
            raise ValueError("path_q must have one entry per path")
        if np.any(self.path_q != np.asarray([path[-1] for path in self.paths])):
            raise ValueError("path_q disagrees with paths")
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
            if np.any(self.q_new != self.path_q[self.new_path_indices]) or np.any(
                self.q_old != self.path_q[self.old_path_indices]
            ):
                raise ValueError("entry q labels disagree with their paths")
            if any(
                int(delta) not in self.delta_q
                for delta in self.q_new - self.q_old
            ):
                raise ValueError("a sparse entry violates the tensor selection rule")
            if any(
                not is_admissible_triangle(int(old), self.q_op, int(new))
                for new, old in zip(self.q_new, self.q_old, strict=True)
            ):
                raise ValueError("a sparse entry violates the SU(2) triangle rule")
        if np.any(~np.isfinite(self.reduced_values)):
            raise ValueError("reduced_values must be finite")

    @property
    def entry_count(self) -> int:
        return int(len(self.reduced_values))

    @property
    def component_frobenius_norm_squared(self) -> float:
        """Sum of Frobenius norms over all magnetic tensor components.

        In the Wigner--Eckart convention used here, CG orthogonality makes
        this exactly the squared Euclidean norm of the reduced entries.
        """

        return float(np.vdot(self.reduced_values, self.reduced_values).real)

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
        """Look up one reduced matrix element; absent entries are zero."""

        new = tuple(new_path)
        old = tuple(old_path)
        if new not in self.paths or old not in self.paths:
            raise ValueError("new_path and old_path must belong to this table")
        return self.entry_dict().get((new, old), 0.0j)

    def numeric_payload(self) -> dict[str, np.ndarray]:
        """Return only plain arrays suitable for offline caching/JAX loading."""

        return {
            "n_sites": np.asarray(self.n_sites, dtype=np.int32),
            "q_op": np.asarray(self.q_op, dtype=np.int32),
            "delta_q": np.asarray(self.delta_q, dtype=np.int32),
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


def _pack_tensor_table(
    paths: tuple[FusionPath, ...],
    q_op: int,
    entries: Mapping[tuple[FusionPath, FusionPath], complex],
    *,
    compiler: str,
    zero_tolerance: float,
) -> ReducedTensorTable:
    rank = validate_q(q_op, name="q_op")
    delta_q = tensor_delta_q(rank)
    if not math.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative")
    positions = {path: index for index, path in enumerate(paths)}
    retained: list[tuple[int, int, complex]] = []
    for (new, old), raw_value in entries.items():
        if new not in positions or old not in positions:
            raise ValueError("a reduced entry path is not in this block basis")
        value = complex(raw_value)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("reduced entries must be finite")
        if abs(value) > zero_tolerance:
            retained.append((positions[new], positions[old], value))
    retained.sort(key=lambda item: (item[0], item[1]))
    new_indices = np.asarray([item[0] for item in retained], dtype=np.int32)
    old_indices = np.asarray([item[1] for item in retained], dtype=np.int32)
    values = np.asarray([item[2] for item in retained], dtype=np.complex128)
    path_q = np.asarray([path[-1] for path in paths], dtype=np.int32)
    return ReducedTensorTable(
        n_sites=len(paths[0]),
        q_op=rank,
        paths=paths,
        path_q=path_q,
        new_path_indices=new_indices,
        old_path_indices=old_indices,
        q_new=path_q[new_indices],
        q_old=path_q[old_indices],
        reduced_values=values,
        compiler=compiler,
        delta_q=delta_q,
    )


@lru_cache(maxsize=None)
def _all_paths(n_sites: int) -> tuple[FusionPath, ...]:
    parity = n_sites % 2
    return tuple(
        path
        for q in range(parity, n_sites + 1, 2)
        for path in enumerate_paths(n_sites, q)
    )


def _extended_paths(path: FusionPath) -> tuple[FusionPath, ...]:
    lower = (path + (path[-1] - 1,),) if path[-1] > 0 else ()
    return lower + (path + (path[-1] + 1,),)


def _spin_half_component(
    component_q: int, m_new: int, m_old: int
) -> float:
    if component_q not in VECTOR_COMPONENTS:
        return 0.0
    return (
        clebsch_gordan(1, m_old, 2, component_q, 1, m_new)
        * math.sqrt(1.5)
        / math.sqrt(2.0)
    )


def _project_rank_reduced_element(
    q_new: int, q_old: int, q_op: int, magnetic_element
) -> complex:
    if not is_admissible_triangle(q_old, q_op, q_new):
        return 0.0j
    total = 0.0j
    for m_old in magnetic_numbers(q_old):
        for component_q in magnetic_numbers(q_op):
            m_new = m_old + component_q
            if abs(m_new) > q_new or (q_new - m_new) % 2:
                continue
            coefficient = clebsch_gordan(
                q_old, m_old, q_op, component_q, q_new, m_new
            )
            if coefficient:
                total += coefficient * magnetic_element(
                    m_new, m_old, component_q
                )
    return total / math.sqrt(q_new + 1.0)


@lru_cache(maxsize=None)
def _old_block_tensor_advance_coefficient(
    q_prefix_new: int,
    q_prefix_old: int,
    q_new: int,
    q_old: int,
    q_op: int,
) -> float:
    if not (
        q_op in VECTOR_PRODUCT_RANKS
        and is_admissible_triangle(q_prefix_new, 1, q_new)
        and is_admissible_triangle(q_prefix_old, 1, q_old)
        and is_admissible_triangle(q_prefix_old, q_op, q_prefix_new)
        and is_admissible_triangle(q_old, q_op, q_new)
    ):
        return 0.0

    def element(m_new: int, m_old: int, component_q: int) -> complex:
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
            answer += (
                spin_half_cg(q_prefix_new, m_prefix_new, m_spin, q_new)
                * clebsch_gordan(
                    q_prefix_old,
                    m_prefix_old,
                    q_op,
                    component_q,
                    q_prefix_new,
                    m_prefix_new,
                )
                / math.sqrt(q_prefix_new + 1.0)
                * spin_half_cg(q_prefix_old, m_prefix_old, m_spin, q_old)
            )
        return answer

    value = _project_rank_reduced_element(q_new, q_old, q_op, element)
    if abs(value.imag) > 2.0e-14:
        raise RuntimeError("real tensor transport produced a complex coefficient")
    return float(value.real)


@lru_cache(maxsize=None)
def _old_vector_appended_spin_product_coefficient(
    q_prefix_new: int,
    q_prefix_old: int,
    q_new: int,
    q_old: int,
    q_op: int,
) -> float:
    """Coefficient of ``[B_old x S_appended]^q_op``."""

    if not (
        q_op in VECTOR_PRODUCT_RANKS
        and is_admissible_triangle(q_prefix_new, 1, q_new)
        and is_admissible_triangle(q_prefix_old, 1, q_old)
        and is_admissible_triangle(q_prefix_old, Q_OP_VECTOR, q_prefix_new)
        and is_admissible_triangle(q_old, q_op, q_new)
    ):
        return 0.0

    def element(m_new: int, m_old: int, component_q: int) -> complex:
        answer = 0.0
        for m_prefix_old in magnetic_numbers(q_prefix_old):
            m_spin_old = m_old - m_prefix_old
            if m_spin_old not in (-1, 1):
                continue
            ket = spin_half_cg(
                q_prefix_old, m_prefix_old, m_spin_old, q_old
            )
            if not ket:
                continue
            for mu_q in VECTOR_COMPONENTS:
                nu_q = component_q - mu_q
                if nu_q not in VECTOR_COMPONENTS:
                    continue
                m_prefix_new = m_prefix_old + mu_q
                m_spin_new = m_spin_old + nu_q
                if (
                    abs(m_prefix_new) > q_prefix_new
                    or (q_prefix_new - m_prefix_new) % 2
                    or m_spin_new not in (-1, 1)
                    or m_prefix_new + m_spin_new != m_new
                ):
                    continue
                product = clebsch_gordan(
                    Q_OP_VECTOR,
                    mu_q,
                    Q_OP_VECTOR,
                    nu_q,
                    q_op,
                    component_q,
                )
                prefix = clebsch_gordan(
                    q_prefix_old,
                    m_prefix_old,
                    Q_OP_VECTOR,
                    mu_q,
                    q_prefix_new,
                    m_prefix_new,
                ) / math.sqrt(q_prefix_new + 1.0)
                spin = _spin_half_component(
                    nu_q, m_spin_new, m_spin_old
                )
                bra = spin_half_cg(
                    q_prefix_new, m_prefix_new, m_spin_new, q_new
                )
                answer += bra * product * prefix * spin * ket
        return answer

    value = _project_rank_reduced_element(q_new, q_old, q_op, element)
    if abs(value.imag) > 2.0e-14:
        raise RuntimeError("real mixed source produced a complex coefficient")
    return float(value.real)


@lru_cache(maxsize=None)
def _appended_spin_product_source_coefficient(
    q_prefix: int, q_new: int, q_old: int, q_op: int
) -> float:
    """Reduced coefficient of ``[S_appended x S_appended]^q_op``."""

    if not (
        q_op in VECTOR_PRODUCT_RANKS
        and is_admissible_triangle(q_prefix, 1, q_new)
        and is_admissible_triangle(q_prefix, 1, q_old)
        and is_admissible_triangle(q_old, q_op, q_new)
    ):
        return 0.0

    def local_element(
        m_spin_new: int, m_spin_old: int, component_q: int
    ) -> complex:
        answer = 0.0
        for mu_q in VECTOR_COMPONENTS:
            nu_q = component_q - mu_q
            if nu_q not in VECTOR_COMPONENTS:
                continue
            product = clebsch_gordan(
                Q_OP_VECTOR,
                mu_q,
                Q_OP_VECTOR,
                nu_q,
                q_op,
                component_q,
            )
            for m_middle in (-1, 1):
                answer += (
                    product
                    * _spin_half_component(mu_q, m_spin_new, m_middle)
                    * _spin_half_component(nu_q, m_middle, m_spin_old)
                )
        return answer

    def element(m_new: int, m_old: int, component_q: int) -> complex:
        answer = 0.0
        for m_prefix in magnetic_numbers(q_prefix):
            m_spin_old = m_old - m_prefix
            m_spin_new = m_new - m_prefix
            if m_spin_old not in (-1, 1) or m_spin_new not in (-1, 1):
                continue
            answer += (
                spin_half_cg(q_prefix, m_prefix, m_spin_new, q_new)
                * local_element(m_spin_new, m_spin_old, component_q)
                * spin_half_cg(q_prefix, m_prefix, m_spin_old, q_old)
            )
        return answer

    value = _project_rank_reduced_element(q_new, q_old, q_op, element)
    if abs(value.imag) > 2.0e-14:
        raise RuntimeError("real endpoint source produced a complex coefficient")
    return float(value.real)


def old_block_tensor_advance_coefficient(
    q_prefix_new: int,
    q_prefix_old: int,
    q_new: int,
    q_old: int,
    q_op: int,
) -> float:
    """Transport an old-block rank-``q_op`` tensor through one spin-half."""

    for name, value in (
        ("q_prefix_new", q_prefix_new),
        ("q_prefix_old", q_prefix_old),
        ("q_new", q_new),
        ("q_old", q_old),
        ("q_op", q_op),
    ):
        validate_q(value, name=name)
    tensor_delta_q(q_op)
    return _old_block_tensor_advance_coefficient(
        q_prefix_new, q_prefix_old, q_new, q_old, q_op
    )


def old_vector_appended_spin_product_coefficient(
    q_prefix_new: int,
    q_prefix_old: int,
    q_new: int,
    q_old: int,
    q_op: int,
) -> float:
    """Return the local coefficient of ``[B_old x S]^q_op``."""

    for name, value in (
        ("q_prefix_new", q_prefix_new),
        ("q_prefix_old", q_prefix_old),
        ("q_new", q_new),
        ("q_old", q_old),
        ("q_op", q_op),
    ):
        validate_q(value, name=name)
    tensor_delta_q(q_op)
    return _old_vector_appended_spin_product_coefficient(
        q_prefix_new, q_prefix_old, q_new, q_old, q_op
    )


def appended_spin_product_source_coefficient(
    q_prefix: int, q_new: int, q_old: int, q_op: int
) -> float:
    """Return the local coefficient of ``[S x S]^q_op``."""

    for name, value in (
        ("q_prefix", q_prefix),
        ("q_new", q_new),
        ("q_old", q_old),
        ("q_op", q_op),
    ):
        validate_q(value, name=name)
    tensor_delta_q(q_op)
    return _appended_spin_product_source_coefficient(
        q_prefix, q_new, q_old, q_op
    )


@lru_cache(maxsize=None)
def _ordered_product_recoupling_coefficient(
    q_new: int, q_middle: int, q_old: int, q_op: int
) -> float:
    """Coefficient multiplying ``A[new,middle] B[middle,old]``."""

    if not (
        q_op in VECTOR_PRODUCT_RANKS
        and is_admissible_triangle(q_middle, Q_OP_VECTOR, q_new)
        and is_admissible_triangle(q_old, Q_OP_VECTOR, q_middle)
        and is_admissible_triangle(q_old, q_op, q_new)
    ):
        return 0.0
    total = 0.0
    for m_old in magnetic_numbers(q_old):
        for nu_q in magnetic_numbers(Q_OP_VECTOR):
            m_middle = m_old + nu_q
            if abs(m_middle) > q_middle or (q_middle - m_middle) % 2:
                continue
            right = clebsch_gordan(
                q_old,
                m_old,
                Q_OP_VECTOR,
                nu_q,
                q_middle,
                m_middle,
            )
            if not right:
                continue
            for mu_q in magnetic_numbers(Q_OP_VECTOR):
                component_q = mu_q + nu_q
                if abs(component_q) > q_op or (q_op - component_q) % 2:
                    continue
                m_new = m_middle + mu_q
                if abs(m_new) > q_new or (q_new - m_new) % 2:
                    continue
                product_cg = clebsch_gordan(
                    Q_OP_VECTOR,
                    mu_q,
                    Q_OP_VECTOR,
                    nu_q,
                    q_op,
                    component_q,
                )
                if not product_cg:
                    continue
                left = clebsch_gordan(
                    q_middle,
                    m_middle,
                    Q_OP_VECTOR,
                    mu_q,
                    q_new,
                    m_new,
                )
                projected = clebsch_gordan(
                    q_old,
                    m_old,
                    q_op,
                    component_q,
                    q_new,
                    m_new,
                )
                total += projected * product_cg * left * right
    return float(total / ((q_new + 1.0) * math.sqrt(q_middle + 1.0)))


def couple_reduced_vectors(
    left: ReducedVectorTable,
    right: ReducedVectorTable,
    q_op: int,
    *,
    zero_tolerance: Real = 2.0e-14,
) -> ReducedTensorTable:
    """Compile the ordered product ``[left x right]^q_op``.

    The right vector acts first.  The algorithm joins sparse reduced entries
    on their intermediate fusion path and therefore never expands magnetic
    multiplets or a ``2**N`` computational basis.
    """

    rank = validate_q(q_op, name="q_op")
    tensor_delta_q(rank)
    tolerance = float(zero_tolerance)
    if left.n_sites != right.n_sites or left.paths != right.paths:
        raise ValueError("left and right vector tables must use the same block basis")
    right_by_new: dict[FusionPath, list[tuple[FusionPath, complex]]] = {}
    for (middle, old), value in right.entry_dict().items():
        right_by_new.setdefault(middle, []).append((old, value))
    accumulated: dict[tuple[FusionPath, FusionPath], complex] = {}
    for (new, middle), left_value in left.entry_dict().items():
        for old, right_value in right_by_new.get(middle, ()):
            coefficient = _ordered_product_recoupling_coefficient(
                new[-1], middle[-1], old[-1], rank
            )
            if coefficient:
                key = (new, old)
                accumulated[key] = (
                    accumulated.get(key, 0.0j)
                    + coefficient * left_value * right_value
                )
    return _pack_tensor_table(
        left.paths,
        rank,
        accumulated,
        compiler="sparse_cg_vector_product",
        zero_tolerance=tolerance,
    )


def advance_reduced_vector_product(
    product: ReducedTensorTable,
    left: ReducedVectorTable,
    right: ReducedVectorTable,
    left_endpoint_weight: complex,
    right_endpoint_weight: complex,
    *,
    zero_tolerance: Real = 2.0e-14,
) -> ReducedTensorTable:
    """Append a spin-half and advance ``[left x right]^q_op`` exactly.

    The update has four locality-fixed pieces: transported old product,
    ``right_endpoint_weight * [left_old x S]``,
    ``left_endpoint_weight * [S x right_old]``, and the same-site source
    ``left_endpoint_weight * right_endpoint_weight * [S x S]``.
    """

    if (
        product.n_sites != left.n_sites
        or product.n_sites != right.n_sites
        or product.paths != left.paths
        or product.paths != right.paths
    ):
        raise ValueError("product and vector tables must share their block basis")
    for name, raw in (
        ("left_endpoint_weight", left_endpoint_weight),
        ("right_endpoint_weight", right_endpoint_weight),
    ):
        if isinstance(raw, bool) or not isinstance(raw, Number):
            raise TypeError(f"{name} must be a finite number")
        value = complex(raw)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError(f"{name} must be finite")
    left_weight = complex(left_endpoint_weight)
    right_weight = complex(right_endpoint_weight)
    tolerance = float(zero_tolerance)
    rank = product.q_op
    accumulated: dict[tuple[FusionPath, FusionPath], complex] = {}

    def add(key: tuple[FusionPath, FusionPath], value: complex) -> None:
        accumulated[key] = accumulated.get(key, 0.0j) + value

    for (prefix_new, prefix_old), old_value in product.entry_dict().items():
        for new in _extended_paths(prefix_new):
            for old in _extended_paths(prefix_old):
                coefficient = _old_block_tensor_advance_coefficient(
                    prefix_new[-1], prefix_old[-1], new[-1], old[-1], rank
                )
                if coefficient:
                    add((new, old), coefficient * old_value)

    if right_weight:
        for (prefix_new, prefix_old), old_value in left.entry_dict().items():
            for new in _extended_paths(prefix_new):
                for old in _extended_paths(prefix_old):
                    coefficient = _old_vector_appended_spin_product_coefficient(
                        prefix_new[-1], prefix_old[-1], new[-1], old[-1], rank
                    )
                    if coefficient:
                        add((new, old), coefficient * old_value * right_weight)

    if left_weight:
        exchange_phase = -1.0 if (rank // 2) % 2 else 1.0
        for (prefix_new, prefix_old), old_value in right.entry_dict().items():
            for new in _extended_paths(prefix_new):
                for old in _extended_paths(prefix_old):
                    coefficient = _old_vector_appended_spin_product_coefficient(
                        prefix_new[-1], prefix_old[-1], new[-1], old[-1], rank
                    )
                    if coefficient:
                        add(
                            (new, old),
                            exchange_phase * coefficient * old_value * left_weight,
                        )

    if left_weight and right_weight:
        for prefix in left.paths:
            for new in _extended_paths(prefix):
                for old in _extended_paths(prefix):
                    coefficient = _appended_spin_product_source_coefficient(
                        prefix[-1], new[-1], old[-1], rank
                    )
                    if coefficient:
                        add(
                            (new, old),
                            coefficient * left_weight * right_weight,
                        )

    return _pack_tensor_table(
        _all_paths(product.n_sites + 1),
        rank,
        accumulated,
        compiler="recursive_cg_vector_product",
        zero_tolerance=tolerance,
    )


def compile_weighted_spin_vector_product(
    left_endpoint_weights: Sequence[complex],
    right_endpoint_weights: Sequence[complex],
    q_op: int,
    *,
    zero_tolerance: Real = 2.0e-14,
) -> ReducedTensorTable:
    """Compile a vector product entirely by one-site product recurrences."""

    left_weights = tuple(left_endpoint_weights)
    right_weights = tuple(right_endpoint_weights)
    if not left_weights or len(left_weights) != len(right_weights):
        raise ValueError("endpoint-weight sequences must have equal positive length")
    tensor_delta_q(q_op)
    left = compile_weighted_spin_vector(left_weights[:1])
    right = compile_weighted_spin_vector(right_weights[:1])
    product = couple_reduced_vectors(
        left, right, q_op, zero_tolerance=zero_tolerance
    )
    for left_weight, right_weight in zip(
        left_weights[1:], right_weights[1:], strict=True
    ):
        product = advance_reduced_vector_product(
            product,
            left,
            right,
            left_weight,
            right_weight,
            zero_tolerance=zero_tolerance,
        )
        left = advance_weighted_spin_vector(
            left, left_weight, zero_tolerance=zero_tolerance
        )
        right = advance_weighted_spin_vector(
            right, right_weight, zero_tolerance=zero_tolerance
        )
    return product


def compile_weighted_spin_vector_jordan(
    left_endpoint_weights: Sequence[complex],
    right_endpoint_weights: Sequence[complex],
    q_op: int,
    *,
    zero_tolerance: Real = 2.0e-14,
) -> ReducedTensorTable:
    """Compile the coupled Jordan product from two ordered recurrences."""

    tolerance = float(zero_tolerance)
    ordered = compile_weighted_spin_vector_product(
        left_endpoint_weights,
        right_endpoint_weights,
        q_op,
        zero_tolerance=0.0,
    )
    reversed_order = compile_weighted_spin_vector_product(
        right_endpoint_weights,
        left_endpoint_weights,
        q_op,
        zero_tolerance=0.0,
    )
    exchange_phase = -1.0 if (q_op // 2) % 2 else 1.0
    return _combine_tensor_tables(
        ordered,
        reversed_order,
        0.5,
        0.5 * exchange_phase,
        compiler="recursive_cg_vector_jordan",
        zero_tolerance=tolerance,
    )


def _combine_tensor_tables(
    first: ReducedTensorTable,
    second: ReducedTensorTable,
    first_scale: complex,
    second_scale: complex,
    *,
    compiler: str,
    zero_tolerance: float,
) -> ReducedTensorTable:
    if (
        first.n_sites != second.n_sites
        or first.q_op != second.q_op
        or first.paths != second.paths
    ):
        raise ValueError("tensor tables must share their block basis and q_op")
    entries: dict[tuple[FusionPath, FusionPath], complex] = {}
    for key, value in first.entry_dict().items():
        entries[key] = entries.get(key, 0.0j) + first_scale * value
    for key, value in second.entry_dict().items():
        entries[key] = entries.get(key, 0.0j) + second_scale * value
    return _pack_tensor_table(
        first.paths,
        first.q_op,
        entries,
        compiler=compiler,
        zero_tolerance=zero_tolerance,
    )


def couple_reduced_vectors_jordan(
    left: ReducedVectorTable,
    right: ReducedVectorTable,
    q_op: int,
    *,
    zero_tolerance: Real = 2.0e-14,
) -> ReducedTensorTable:
    """Compile the coupled anticommutator of vector components.

    This represents ``sum CG * (A_mu B_nu + B_nu A_mu) / 2``.  The exchange
    phase is ``(-1)**(q_op/2)`` for two doubled-rank-two factors.
    """

    rank = validate_q(q_op, name="q_op")
    tensor_delta_q(rank)
    tolerance = float(zero_tolerance)
    ordered = couple_reduced_vectors(
        left, right, rank, zero_tolerance=0.0
    )
    reversed_order = couple_reduced_vectors(
        right, left, rank, zero_tolerance=0.0
    )
    exchange_phase = -1.0 if (rank // 2) % 2 else 1.0
    return _combine_tensor_tables(
        ordered,
        reversed_order,
        0.5,
        0.5 * exchange_phase,
        compiler="sparse_cg_vector_jordan",
        zero_tolerance=tolerance,
    )


def couple_reduced_vectors_commutator(
    left: ReducedVectorTable,
    right: ReducedVectorTable,
    q_op: int,
    *,
    zero_tolerance: Real = 2.0e-14,
) -> ReducedTensorTable:
    """Compile the coupled commutator of vector components."""

    rank = validate_q(q_op, name="q_op")
    tensor_delta_q(rank)
    tolerance = float(zero_tolerance)
    ordered = couple_reduced_vectors(
        left, right, rank, zero_tolerance=0.0
    )
    reversed_order = couple_reduced_vectors(
        right, left, rank, zero_tolerance=0.0
    )
    exchange_phase = -1.0 if (rank // 2) % 2 else 1.0
    return _combine_tensor_tables(
        ordered,
        reversed_order,
        0.5,
        -0.5 * exchange_phase,
        compiler="sparse_cg_vector_commutator",
        zero_tolerance=tolerance,
    )


def coupled_tensor_block_scalar_basis(
    left: ReducedTensorTable,
    right: ReducedTensorTable,
    target_q: int,
) -> tuple[tuple[FusionPath, FusionPath], ...]:
    """Return the two-block multiplicity basis coupled to ``target_q``."""

    validate_q(target_q, name="target_q")
    return tuple(
        (left_path, right_path)
        for left_path in left.paths
        for right_path in right.paths
        if is_admissible_triangle(left_path[-1], right_path[-1], target_q)
    )


@lru_cache(maxsize=None)
def _tensor_scalar_recoupling_coefficient(
    q_left_new: int,
    q_right_new: int,
    q_left_old: int,
    q_right_old: int,
    q_op: int,
    target_q: int,
    total_m: int,
) -> float:
    """Magnetic contraction for two unit rank-``q_op`` reduced elements."""

    if not (
        q_op in VECTOR_PRODUCT_RANKS
        and is_admissible_triangle(q_left_new, q_right_new, target_q)
        and is_admissible_triangle(q_left_old, q_right_old, target_q)
        and is_admissible_triangle(q_left_old, q_op, q_left_new)
        and is_admissible_triangle(q_right_old, q_op, q_right_new)
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
        for component_q in magnetic_numbers(q_op):
            m_left_new = m_left_old + component_q
            m_right_new = m_right_old - component_q
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
                q_op,
                component_q,
                q_left_new,
                m_left_new,
            ) / math.sqrt(q_left_new + 1.0)
            right_component = clebsch_gordan(
                q_right_old,
                m_right_old,
                q_op,
                -component_q,
                q_right_new,
                m_right_new,
            ) / math.sqrt(q_right_new + 1.0)
            phase = -1.0 if (component_q // 2) % 2 else 1.0
            answer += (
                bra * phase * left_component * right_component * ket
            )
    return float(answer)


def contract_reduced_tensors_scalar(
    left: ReducedTensorTable,
    right: ReducedTensorTable,
    target_q: int,
    *,
    total_m: int | None = None,
) -> tuple[np.ndarray, tuple[tuple[FusionPath, FusionPath], ...]]:
    """Contract equal-rank tensors into a scalar between two blocks.

    The contraction is ``sum_M (-1)^M A_M B_-M`` with ordinary integer
    projection ``M=component_q/2``.  It is bilinear: reduced entries are not
    conjugated.  Rows and columns follow ``(new_path, old_path)`` orientation.
    """

    validate_q(target_q, name="target_q")
    if left.q_op != right.q_op:
        raise ValueError("left and right tensors must have the same q_op")
    if total_m is None:
        total_m = target_q
    validate_m(target_q, total_m, name="total_m")
    basis = coupled_tensor_block_scalar_basis(left, right, target_q)
    positions = {state: index for index, state in enumerate(basis)}
    result = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    for (left_new, left_old), left_value in left.entry_dict().items():
        for (right_new, right_old), right_value in right.entry_dict().items():
            row = positions.get((left_new, right_new))
            column = positions.get((left_old, right_old))
            if row is None or column is None:
                continue
            coefficient = _tensor_scalar_recoupling_coefficient(
                left_new[-1],
                right_new[-1],
                left_old[-1],
                right_old[-1],
                left.q_op,
                target_q,
                total_m,
            )
            if coefficient:
                result[row, column] += coefficient * left_value * right_value
    return result, basis


def tensor_spherical_component_coupled_matrix(
    table: ReducedTensorTable, component_q: int
) -> tuple[np.ndarray, tuple[tuple[FusionPath, int], ...]]:
    """Expand one tensor component in the complete coupled block basis."""

    if component_q not in magnetic_numbers(table.q_op):
        raise ValueError("component_q is not a projection of q_op")
    basis = tuple(
        (path, m) for path in table.paths for m in magnetic_numbers(path[-1])
    )
    positions = {state: index for index, state in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    for (new_path, old_path), reduced in table.entry_dict().items():
        q_new, q_old = new_path[-1], old_path[-1]
        for m_old in magnetic_numbers(q_old):
            m_new = m_old + component_q
            if abs(m_new) > q_new or (q_new - m_new) % 2:
                continue
            coefficient = clebsch_gordan(
                q_old,
                m_old,
                table.q_op,
                component_q,
                q_new,
                m_new,
            )
            matrix[
                positions[(new_path, m_new)], positions[(old_path, m_old)]
            ] = coefficient * reduced / math.sqrt(q_new + 1.0)
    return matrix, basis


__all__ = [
    "ReducedTensorTable",
    "VECTOR_PRODUCT_RANKS",
    "advance_reduced_vector_product",
    "appended_spin_product_source_coefficient",
    "compile_weighted_spin_vector_product",
    "compile_weighted_spin_vector_jordan",
    "couple_reduced_vectors",
    "couple_reduced_vectors_commutator",
    "couple_reduced_vectors_jordan",
    "contract_reduced_tensors_scalar",
    "coupled_tensor_block_scalar_basis",
    "old_block_tensor_advance_coefficient",
    "old_vector_appended_spin_product_coefficient",
    "tensor_delta_q",
    "tensor_spherical_component_coupled_matrix",
]
