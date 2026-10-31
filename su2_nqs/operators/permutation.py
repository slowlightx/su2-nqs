"""Sparse spin permutations and graph Heisenberg operators in fusion space."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from functools import cached_property
from typing import Callable, Iterable, Sequence

import numpy as np

from ..su2.fusion_path import FusionPath, enumerate_paths, is_valid_path
from ..su2.wigner6j import pair_heisenberg_block
from .heisenberg_chain import get_conn
from .pair_spin import pair_spin_energy


SitePermutation = tuple[int, ...]
SwapWord = tuple[int, ...]


def adjacent_swap_path_connectivity(
    path: Sequence[int],
    bond: int,
    *,
    target_q: int | None = None,
) -> tuple[tuple[FusionPath, ...], np.ndarray]:
    """Apply one adjacent transposition using only its local F-move block.

    This is the path-local form of ``P=2 S_i.S_j + I/2``.  It neither
    enumerates the fusion Hilbert space nor materializes a sparse matrix.
    Angular-momentum labels remain doubled integers throughout.
    """

    candidate = tuple(int(value) for value in path)
    if not is_valid_path(candidate):
        raise ValueError(f"invalid fusion path: {candidate!r}")
    n = len(candidate)
    if (
        isinstance(bond, bool)
        or not isinstance(bond, (int, np.integer))
        or not 0 <= int(bond) < n - 1
    ):
        raise ValueError("bond must index an adjacent pair in the path")
    if target_q is not None and candidate[-1] != int(target_q):
        raise ValueError("path does not end in the requested doubled-spin sector")
    bond = int(bond)

    if bond == 0:
        value = 2.0 * pair_spin_energy(candidate[1]) + 0.5
        return (candidate,), np.asarray((value,), dtype=np.float64)

    q_left = candidate[bond - 1]
    q_mid = candidate[bond]
    q_total = candidate[bond + 1]
    block, channels = pair_heisenberg_block(q_left, q_total)
    try:
        column = channels.index(q_mid)
    except ValueError as error:
        raise ValueError(
            f"path contains an inadmissible local channel: {candidate!r}"
        ) from error
    connected: list[FusionPath] = []
    values: list[float] = []
    for row, q_new in enumerate(channels):
        value = 2.0 * float(block[row, column])
        if q_new == q_mid:
            value += 0.5
        if abs(value) > 1.0e-15:
            connected.append(candidate[:bond] + (q_new,) + candidate[bond + 1 :])
            values.append(value)
    return tuple(connected), np.asarray(values, dtype=np.float64)


@lru_cache(maxsize=None)
def _site_permutation_fusion_row_cached(
    output_path: FusionPath,
    image: SitePermutation,
    target_q: int,
) -> tuple[tuple[FusionPath, ...], tuple[float, ...]]:
    """Cached row support of one physical permutation in fusion space."""

    if len(output_path) != len(image) or not is_valid_path(output_path, target_q):
        raise ValueError("output path does not belong to the requested sector")
    coefficients: dict[FusionPath, float] = {output_path: 1.0}

    # If U=P_last...P_first, then row <a|U is obtained by applying
    # U.T=P_first...P_last to |a>; sequential left multiplication therefore
    # consumes the compiled chronological word in reverse.
    word = site_permutation_to_adjacent_swap_word(image)
    for bond in reversed(word):
        updated: dict[FusionPath, float] = {}
        for path, outer in coefficients.items():
            connected, values = adjacent_swap_path_connectivity(
                path, bond, target_q=target_q
            )
            for new_path, value in zip(connected, values, strict=True):
                updated[new_path] = updated.get(new_path, 0.0) + outer * float(value)
        coefficients = {
            path: value for path, value in updated.items() if abs(value) > 1.0e-15
        }

    items = tuple(sorted(coefficients.items()))
    return (
        tuple(path for path, _ in items),
        tuple(float(value) for _, value in items),
    )


def site_permutation_fusion_row(
    output_path: Sequence[int],
    image: Sequence[int],
    *,
    target_q: int,
) -> tuple[tuple[FusionPath, ...], np.ndarray]:
    """Return one row of a site permutation without Hilbert-space arrays.

    The result satisfies ``(U psi)[output_path] = sum_s coeff[s] psi[s]``.
    Its work is controlled by the recoupling support generated along the
    adjacent-swap word, not by the full fusion-space dimension.
    """

    checked_path = tuple(int(value) for value in output_path)
    checked_image = _validated_site_permutation(image)
    paths, values = _site_permutation_fusion_row_cached(
        checked_path, checked_image, int(target_q)
    )
    return paths, np.asarray(values, dtype=np.float64)


def clear_permutation_row_cache() -> None:
    """Release cached matrix-free permutation rows between benchmark stages."""

    _site_permutation_fusion_row_cached.cache_clear()


def weighted_site_permutation_fusion_row(
    output_path: Sequence[int],
    weighted_images: Sequence[tuple[complex, Sequence[int]]],
    *,
    target_q: int,
) -> tuple[tuple[FusionPath, ...], np.ndarray]:
    """Return a weighted symmetry-orbit row from local recoupling supports."""

    terms = tuple(weighted_images)
    if not terms:
        raise ValueError("weighted_images must be nonempty")
    combined: dict[FusionPath, complex] = {}
    for raw_weight, image in terms:
        weight = complex(raw_weight)
        if not np.isfinite(weight.real) or not np.isfinite(weight.imag):
            raise ValueError("orbit weights must be finite")
        paths, values = site_permutation_fusion_row(
            output_path, image, target_q=target_q
        )
        for path, value in zip(paths, values, strict=True):
            combined[path] = combined.get(path, 0.0j) + weight * float(value)
    items = tuple(
        sorted(
            (path, value)
            for path, value in combined.items()
            if abs(value) > 1.0e-15
        )
    )
    return (
        tuple(path for path, _ in items),
        np.asarray([value for _, value in items], dtype=np.complex128),
    )


def evaluate_site_permutation_fusion_amplitude(
    output_path: Sequence[int],
    image: Sequence[int],
    amplitude: Callable[[FusionPath], complex],
    *,
    target_q: int,
) -> complex:
    """Evaluate one permuted amplitude from only its local row support."""

    paths, coefficients = site_permutation_fusion_row(
        output_path, image, target_q=target_q
    )
    return complex(
        sum(
            coefficient * complex(amplitude(path))
            for path, coefficient in zip(paths, coefficients, strict=True)
        )
    )


def _validated_site_permutation(image: Sequence[int]) -> SitePermutation:
    """Return a checked active ``old_site -> new_site`` permutation."""

    raw = tuple(image)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise TypeError("site permutation entries must be integers")
    result = tuple(int(value) for value in raw)
    if tuple(sorted(result)) != tuple(range(len(result))):
        raise ValueError("site permutation must contain every site exactly once")
    return result


def site_permutation_to_adjacent_swap_word(
    image: Sequence[int],
) -> SwapWord:
    """Compile an active physical-site permutation into adjacent swaps.

    If ``word=(b0,b1,...)``, its fusion action is evaluated chronologically
    as ``... @ P[b1,b1+1] @ P[b0,b0+1] @ vector``.
    """

    image = _validated_site_permutation(image)
    n = len(image)
    target = [0] * n
    for old, new in enumerate(image):
        target[new] = old
    current = list(range(n))
    word: list[int] = []
    for target_position, wanted_old_site in enumerate(target):
        position = current.index(wanted_old_site)
        while position > target_position:
            bond = position - 1
            current[bond], current[bond + 1] = current[bond + 1], current[bond]
            word.append(bond)
            position -= 1
    if current != target:
        raise RuntimeError("adjacent-swap compiler did not realize permutation")
    return tuple(word)


def apply_adjacent_swap_word(
    vectors: np.ndarray,
    *,
    n: int,
    target_q: int,
    word: Sequence[int],
    inverse: bool = False,
) -> np.ndarray:
    """Apply an adjacent-swap word to one or many fusion-basis columns."""

    array = np.asarray(vectors)
    if array.ndim not in (1, 2):
        raise ValueError("vectors must have shape (dimension,) or (dimension,batch)")
    dimension = len(enumerate_paths(n, target_q))
    if array.shape[0] != dimension:
        raise ValueError(
            f"leading dimension {array.shape[0]} does not match fusion dimension "
            f"{dimension}"
        )
    checked_word = tuple(word)
    if any(
        isinstance(bond, bool)
        or not isinstance(bond, (int, np.integer))
        or not 0 <= int(bond) < n - 1
        for bond in checked_word
    ):
        raise ValueError("swap word contains an invalid adjacent bond")
    order = reversed(checked_word) if inverse else checked_word
    result = np.array(
        array, dtype=np.result_type(array.dtype, np.float64), copy=True
    )
    for raw_bond in order:
        result = adjacent_swap_matrix(n, target_q, int(raw_bond)) @ result
    return np.asarray(result)


def apply_site_permutation_fusion(
    vectors: np.ndarray,
    image: Sequence[int],
    *,
    target_q: int,
) -> np.ndarray:
    """Apply a physical-site permutation in a fixed-total-spin fusion basis."""

    image = _validated_site_permutation(image)
    return apply_adjacent_swap_word(
        vectors,
        n=len(image),
        target_q=target_q,
        word=site_permutation_to_adjacent_swap_word(image),
    )


def site_permutation_fusion_matrix(
    image: Sequence[int],
    *,
    target_q: int,
):
    """Return the sparse fixed-spin representation of a site permutation."""

    sparse = _sparse()
    image = _validated_site_permutation(image)
    dimension = len(enumerate_paths(len(image), target_q))
    result = sparse.identity(dimension, format="csr", dtype=np.float64)
    for bond in site_permutation_to_adjacent_swap_word(image):
        result = adjacent_swap_matrix(len(image), target_q, bond) @ result
    result.eliminate_zeros()
    return result.tocsr()


def _sparse():
    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("sparse permutation operators require scipy") from error
    return sparse


@lru_cache(maxsize=None)
def adjacent_swap_matrix(n: int, target_q: int, bond: int):
    """Representation of adjacent physical transposition ``(bond,bond+1)``."""

    sparse = _sparse()
    if not 0 <= bond < n - 1:
        raise ValueError("adjacent bond is outside the chain")
    basis = enumerate_paths(n, target_q)
    positions = {path: index for index, path in enumerate(basis)}
    couplings = [0.0] * (n - 1)
    couplings[bond] = 1.0
    rows, columns, values = [], [], []
    for column, path in enumerate(basis):
        connected, elements = get_conn(path, couplings)
        rows.append(column)
        columns.append(column)
        values.append(0.5)
        for candidate, element in zip(connected, elements):
            rows.append(positions[candidate])
            columns.append(column)
            values.append(2.0 * float(element))
    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(len(basis), len(basis))
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


@lru_cache(maxsize=None)
def transposition_matrix(n: int, target_q: int, first: int, second: int):
    """Representation of the arbitrary physical transposition ``(first,second)``."""

    sparse = _sparse()
    if first == second:
        dimension = len(enumerate_paths(n, target_q))
        return sparse.identity(dimension, format="csr", dtype=np.float64)
    if first > second:
        first, second = second, first
    if not 0 <= first < second < n:
        raise ValueError("transposition sites lie outside the system")
    sequence = tuple(range(first, second)) + tuple(range(second - 2, first - 1, -1))
    dimension = len(enumerate_paths(n, target_q))
    result = sparse.identity(dimension, format="csr", dtype=np.float64)
    for bond in sequence:
        result = adjacent_swap_matrix(n, target_q, bond) @ result
    result.eliminate_zeros()
    return result.tocsr()


def heisenberg_graph_fusion_matrix(
    n: int,
    bonds: Iterable[tuple[int, int] | tuple[int, int, float]],
    *,
    target_q: int = 0,
):
    """Sparse graph Hamiltonian using ``S_i.S_j = P_ij/2 - I/4``."""

    sparse = _sparse()
    dimension = len(enumerate_paths(n, target_q))
    identity = sparse.identity(dimension, format="csr", dtype=np.float64)
    hamiltonian = sparse.csr_matrix((dimension, dimension), dtype=np.float64)
    for bond in bonds:
        if len(bond) == 2:
            first, second = bond
            coupling = 1.0
        elif len(bond) == 3:
            first, second, coupling = bond
        else:
            raise ValueError(f"invalid bond: {bond!r}")
        permutation = transposition_matrix(n, target_q, int(first), int(second))
        hamiltonian = hamiltonian + float(coupling) * (0.5 * permutation - 0.25 * identity)
    hamiltonian.eliminate_zeros()
    return hamiltonian.tocsr()


def heisenberg_graph_path_connectivity(
    path: Sequence[int],
    bonds: Iterable[tuple[int, int] | tuple[int, int, float]],
    *,
    target_q: int = 0,
) -> tuple[tuple[FusionPath, ...], np.ndarray]:
    """Return one graph-Hamiltonian column without a full fusion basis.

    The returned entries follow ``H[new_path, old_path]``.  A transposition is
    real, symmetric, and involutory, so its matrix-free row support is also its
    column support.  Contributions from different bonds are combined before
    returning.  Runtime depends on the generated adjacent-swap support and can
    still become exponential for a geometry poorly aligned with the left
    fusion path; this function makes that scaling measurable without first
    allocating the exponentially large sector matrix.
    """

    old_path = tuple(int(value) for value in path)
    if not is_valid_path(old_path, target_q):
        raise ValueError("path does not belong to the requested sector")
    n = len(old_path)
    combined: dict[FusionPath, float] = {}
    for bond in bonds:
        if len(bond) == 2:
            first, second = bond
            coupling = 1.0
        elif len(bond) == 3:
            first, second, coupling = bond
        else:
            raise ValueError(f"invalid bond: {bond!r}")
        if any(
            isinstance(site, bool) or not isinstance(site, (int, np.integer))
            for site in (first, second)
        ):
            raise TypeError("bond sites must be integers")
        first, second = int(first), int(second)
        coupling = float(coupling)
        if first == second or not 0 <= first < n or not 0 <= second < n:
            raise ValueError(f"invalid graph bond: {bond!r}")
        if not np.isfinite(coupling):
            raise ValueError("bond coupling must be finite")
        image = list(range(n))
        image[first], image[second] = image[second], image[first]
        connected, values = site_permutation_fusion_row(
            old_path, image, target_q=target_q
        )
        combined[old_path] = combined.get(old_path, 0.0) - 0.25 * coupling
        for new_path, value in zip(connected, values, strict=True):
            combined[new_path] = (
                combined.get(new_path, 0.0) + 0.5 * coupling * float(value)
            )
    items = tuple(
        sorted(
            (new_path, value)
            for new_path, value in combined.items()
            if abs(value) > 1.0e-15
        )
    )
    return (
        tuple(new_path for new_path, _ in items),
        np.asarray([value for _, value in items], dtype=np.float64),
    )


@dataclass(frozen=True)
class HeisenbergGraphFusion:
    """Arbitrary graph Heisenberg connectivity in one total-spin sector."""

    n: int
    bonds: tuple[tuple[int, int, float], ...]
    target_q: int = 0

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("N must be positive")
        normalized = []
        for bond in self.bonds:
            if len(bond) != 3:
                raise ValueError("graph bonds must contain two sites and a coupling")
            first, second, coupling = bond
            if first == second or not 0 <= first < self.n or not 0 <= second < self.n:
                raise ValueError(f"invalid graph bond: {bond!r}")
            normalized.append((int(first), int(second), float(coupling)))
        object.__setattr__(self, "bonds", tuple(normalized))

    @cached_property
    def basis(self) -> tuple[tuple[int, ...], ...]:
        return enumerate_paths(self.n, self.target_q)

    @cached_property
    def _positions(self) -> dict[tuple[int, ...], int]:
        return {path: position for position, path in enumerate(self.basis)}

    @cached_property
    def _matrix(self):
        return heisenberg_graph_fusion_matrix(
            self.n, self.bonds, target_q=self.target_q
        )

    def get_conn(
        self, path: Sequence[int]
    ) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
        """Return combined nonzero entries of one Hamiltonian column."""

        candidate = tuple(int(value) for value in path)
        try:
            column = self._positions[candidate]
        except KeyError as error:
            raise ValueError("path does not belong to this graph sector") from error
        vector = self._matrix.getcol(column).tocoo()
        order = np.argsort(vector.row)
        rows = vector.row[order]
        values = vector.data[order]
        return tuple(self.basis[int(row)] for row in rows), np.asarray(
            values, dtype=np.float64
        )

    def to_dense(self) -> np.ndarray:
        """Return the graph Hamiltonian in deterministic fusion-path order."""

        return self._matrix.toarray()
