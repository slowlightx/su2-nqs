"""State-blind exchange-noncommutativity diagnostics for fusion-tree cuts.

For a physical support ``A`` split the Heisenberg Hamiltonian into bonds
inside ``A``, bonds inside its complement, and bonds crossing the cut.  The
commutators of either internal Hamiltonian with the crossing Hamiltonian are
sums of scalar-spin-chirality operators.  For spin one half those operators
are Hilbert--Schmidt orthogonal on distinct three-site supports, so their
Gram matrices can be evaluated from the weighted graph in polynomial time.

The resulting signature is deliberately state blind: it has no wavefunction,
energy, optimizer, or filesystem input.  It is intended only as a secondary
lexicographic diagnostic after an independently defined hierarchy boundary
score is tied.  It must not be combined with that score using an energy-fit
coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Integral, Real
from typing import Sequence

from ..su2.fusion_tree import FusionTree, Leaf, TreeNode


WeightedEdge = tuple[int, int, float]
ChiralityTriple = tuple[int, int, int]
CoefficientRow = tuple[tuple[ChiralityTriple, float], ...]

EDGE_AGGREGATION_CONTRACT = (
    "repeated undirected Hamiltonian edges are summed algebraically before "
    "forming boundary or chirality coefficients"
)

LEXICOGRAPHIC_USE_CONTRACT = (
    "compare (incoherent_load, coherence_ratio) only inside a pre-existing "
    "boundary-score tie; do not fit or scalarize it with boundary score"
)


@dataclass(frozen=True)
class CutChiralityCoefficients:
    """Chirality coefficients for one oriented subtree support.

    ``left`` comes from ``[H_A, V_A]`` and ``right`` from
    ``[H_complement, V_A]``.  A row ``(triple, kappa)`` means that the
    anti-Hermitian commutator contains ``1j * kappa * chi_triple`` under the
    sorted-site chirality convention.
    """

    support: tuple[int, ...]
    left: CoefficientRow
    right: CoefficientRow


@dataclass(frozen=True)
class CutChiralityGram:
    """Normalized left/right Hilbert--Schmidt Gram matrices for many cuts."""

    supports: tuple[tuple[int, ...], ...]
    left: tuple[tuple[float, ...], ...]
    right: tuple[tuple[float, ...], ...]
    j_scale: float
    chirality_triple_count: int
    canonical_edges: tuple[WeightedEdge, ...]
    edge_aggregation_contract: str = EDGE_AGGREGATION_CONTRACT


@dataclass(frozen=True)
class HierarchyChiralitySignature:
    """Two-stage noncommutativity signature for a rooted fusion hierarchy.

    ``incoherent_load`` is the mean individual-cut current norm.  The
    ``coherent_load`` is the squared norm of the mean current across cuts,
    keeping the inside and outside currents separate.  Their ratio lies in
    ``[0, 1]`` and measures repeated chirality directions across the hierarchy.
    """

    n_sites: int
    internal_cut_count: int
    j_scale: float
    incoherent_load: float
    coherent_load: float
    coherence_ratio: float
    supports: tuple[tuple[int, ...], ...]
    left_gram: tuple[tuple[float, ...], ...]
    right_gram: tuple[tuple[float, ...], ...]
    chirality_triple_count: int
    construction: str = "state_blind_cut_chirality_gram_v1"
    lexicographic_use_contract: str = LEXICOGRAPHIC_USE_CONTRACT


def _validate_n_sites(n_sites: int) -> int:
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be an integer")
    checked = int(n_sites)
    if checked < 2:
        raise ValueError("cut chirality requires at least two sites")
    return checked


def _canonical_edges(
    n_sites: int, edges: Sequence[Sequence[Real]],
) -> tuple[WeightedEdge, ...]:
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in edges:
        if len(row) != 3:
            raise ValueError("each edge must contain first, second, and coupling")
        first_raw, second_raw, coupling_raw = row
        if (
            isinstance(first_raw, bool)
            or isinstance(second_raw, bool)
            or not isinstance(first_raw, Integral)
            or not isinstance(second_raw, Integral)
        ):
            raise TypeError("edge endpoints must be integers")
        if isinstance(coupling_raw, bool) or not isinstance(coupling_raw, Real):
            raise TypeError("edge couplings must be real numbers")
        first, second = int(first_raw), int(second_raw)
        if not 0 <= first < n_sites or not 0 <= second < n_sites:
            raise ValueError("edge endpoint lies outside the graph")
        if first == second:
            raise ValueError("self edges are not supported")
        coupling = float(coupling_raw)
        if not isfinite(coupling):
            raise ValueError("edge couplings must be finite")
        pair = tuple(sorted((first, second)))
        grouped.setdefault(pair, []).append(coupling)

    canonical = []
    for (first, second), values in sorted(grouped.items()):
        coupling = fsum(sorted(values))
        if coupling != 0.0:
            canonical.append((first, second, coupling))
    return tuple(canonical)


def _validate_support(
    n_sites: int, support: Sequence[int], *, name: str,
) -> tuple[int, ...]:
    checked = []
    for site in support:
        if isinstance(site, bool) or not isinstance(site, Integral):
            raise TypeError(f"{name} sites must be integers")
        value = int(site)
        if not 0 <= value < n_sites:
            raise ValueError(f"{name} contains a site outside the graph")
        checked.append(value)
    row = tuple(sorted(checked))
    if len(set(row)) != len(row):
        raise ValueError(f"{name} contains a repeated site")
    if not 0 < len(row) < n_sites:
        raise ValueError(f"{name} must define a nontrivial cut")
    return row


def _orientation_parity(first: int, second: int, third: int) -> int:
    values = (first, second, third)
    inversions = sum(
        values[left] > values[right]
        for left in range(3)
        for right in range(left + 1, 3)
    )
    return -1 if inversions % 2 else 1


def _freeze_coefficients(
    terms: dict[ChiralityTriple, list[float]],
) -> CoefficientRow:
    output = []
    for triple, values in sorted(terms.items()):
        coefficient = fsum(sorted(values))
        if coefficient != 0.0:
            output.append((triple, coefficient))
    return tuple(output)


def _coefficient_rows(
    n_sites: int,
    support: tuple[int, ...],
    edges: tuple[WeightedEdge, ...],
) -> tuple[CoefficientRow, CoefficientRow]:
    inside = frozenset(support)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n_sites)]
    for first, second, coupling in edges:
        adjacency[first].append((second, coupling))
        adjacency[second].append((first, coupling))

    rows: list[CoefficientRow] = []
    for center_inside in (True, False):
        terms: dict[ChiralityTriple, list[float]] = {}
        for center in range(n_sites):
            if (center in inside) is not center_inside:
                continue
            internal = []
            boundary = []
            for neighbor, coupling in adjacency[center]:
                if (neighbor in inside) is center_inside:
                    internal.append((neighbor, coupling))
                else:
                    boundary.append((neighbor, coupling))
            for internal_site, internal_coupling in internal:
                for boundary_site, boundary_coupling in boundary:
                    triple = tuple(sorted((center, internal_site, boundary_site)))
                    term = (
                        _orientation_parity(
                            center, internal_site, boundary_site
                        )
                        * internal_coupling
                        * boundary_coupling
                    )
                    terms.setdefault(triple, []).append(term)
        rows.append(_freeze_coefficients(terms))
    return rows[0], rows[1]


def cut_chirality_coefficients(
    n_sites: int,
    support: Sequence[int],
    edges: Sequence[Sequence[Real]],
) -> CutChiralityCoefficients:
    """Return exact graph coefficients for one Heisenberg cut.

    Repeated undirected edges are summed algebraically.  The result contains
    no factor from the spin-half chirality norm; that universal factor is
    removed by the normalization used in :func:`cut_chirality_gram`.
    """

    checked_n = _validate_n_sites(n_sites)
    checked_support = _validate_support(checked_n, support, name="support")
    canonical = _canonical_edges(checked_n, edges)
    left, right = _coefficient_rows(checked_n, checked_support, canonical)
    return CutChiralityCoefficients(checked_support, left, right)


def _dot(left: CoefficientRow, right: CoefficientRow) -> float:
    right_map = dict(right)
    return fsum(
        sorted(coefficient * right_map.get(triple, 0.0) for triple, coefficient in left)
    )


def cut_chirality_gram(
    n_sites: int,
    supports: Sequence[Sequence[int]],
    edges: Sequence[Sequence[Real]],
) -> CutChiralityGram:
    """Build normalized chirality Gram matrices for an ordered cut family.

    For ``K_A = [H_A, V_A]`` the returned entries obey

    ``G[A,B] = (32 / (3 J_*^4)) 2^-N Tr(K_A^dagger K_B)``.

    ``left`` uses ``H_A`` and ``right`` the complement Hamiltonian.  Distinct
    cut supports are required, while complementary supports are allowed and
    exchange the two Gram matrices.
    """

    checked_n = _validate_n_sites(n_sites)
    checked_supports = tuple(
        _validate_support(checked_n, support, name=f"supports[{index}]")
        for index, support in enumerate(supports)
    )
    if len(set(checked_supports)) != len(checked_supports):
        raise ValueError("supports must not contain duplicate cuts")
    canonical = _canonical_edges(checked_n, edges)
    scale = max((abs(coupling) for _, _, coupling in canonical), default=0.0)
    coefficients = tuple(
        CutChiralityCoefficients(
            support,
            *_coefficient_rows(checked_n, support, canonical),
        )
        for support in checked_supports
    )
    normalization = scale**4 if scale != 0.0 else 1.0
    size = len(coefficients)
    left_rows = [[0.0] * size for _ in range(size)]
    right_rows = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row, size):
            left_value = _dot(
                coefficients[row].left, coefficients[column].left
            ) / normalization
            right_value = _dot(
                coefficients[row].right, coefficients[column].right
            ) / normalization
            left_rows[row][column] = left_rows[column][row] = left_value
            right_rows[row][column] = right_rows[column][row] = right_value

    triples = {
        triple
        for coefficients_for_cut in coefficients
        for side in (coefficients_for_cut.left, coefficients_for_cut.right)
        for triple, _ in side
    }
    return CutChiralityGram(
        supports=checked_supports,
        left=tuple(tuple(row) for row in left_rows),
        right=tuple(tuple(row) for row in right_rows),
        j_scale=scale,
        chirality_triple_count=len(triples),
        canonical_edges=canonical,
    )


def fusion_tree_cut_supports(
    tree: FusionTree, site_order: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """Return non-root internal subtree supports in tree postorder."""

    if len(site_order) != tree.n_leaves:
        raise ValueError("site_order length must match the tree")
    if any(isinstance(site, bool) or not isinstance(site, Integral) for site in site_order):
        raise TypeError("site_order entries must be integers")
    checked_order = tuple(int(site) for site in site_order)
    if tuple(sorted(checked_order)) != tuple(range(tree.n_leaves)):
        raise ValueError("site_order must be a permutation of all sites")

    def support(node: TreeNode) -> tuple[int, ...]:
        if isinstance(node, Leaf):
            return (checked_order[node.index],)
        return tuple(sorted((*support(node.left), *support(node.right))))

    return tuple(support(node) for node in tree.internal_nodes[:-1])


def _signature_from_gram(
    n_sites: int, gram: CutChiralityGram,
) -> HierarchyChiralitySignature:
    count = len(gram.supports)
    if count == 0:
        return HierarchyChiralitySignature(
            n_sites=n_sites,
            internal_cut_count=0,
            j_scale=gram.j_scale,
            incoherent_load=0.0,
            coherent_load=0.0,
            coherence_ratio=0.0,
            supports=gram.supports,
            left_gram=gram.left,
            right_gram=gram.right,
            chirality_triple_count=gram.chirality_triple_count,
        )

    trace = fsum(
        gram.left[index][index] + gram.right[index][index]
        for index in range(count)
    )
    coherent_numerator = fsum(
        value
        for matrix in (gram.left, gram.right)
        for row in matrix
        for value in row
    )
    tolerance = 2.0e-13 * max(1.0, trace * count)
    if trace < -tolerance or coherent_numerator < -tolerance:
        raise RuntimeError("a chirality Gram quadratic form became negative")
    trace = max(0.0, trace)
    coherent_numerator = max(0.0, coherent_numerator)
    incoherent = trace / count
    coherent = coherent_numerator / count**2
    ratio = coherent / incoherent if incoherent != 0.0 else 0.0
    if not -2.0e-13 <= ratio <= 1.0 + 2.0e-13:
        raise RuntimeError("chirality coherence ratio lies outside [0, 1]")
    ratio = min(1.0, max(0.0, ratio))
    return HierarchyChiralitySignature(
        n_sites=n_sites,
        internal_cut_count=count,
        j_scale=gram.j_scale,
        incoherent_load=incoherent,
        coherent_load=coherent,
        coherence_ratio=ratio,
        supports=gram.supports,
        left_gram=gram.left,
        right_gram=gram.right,
        chirality_triple_count=gram.chirality_triple_count,
    )


def hierarchy_chirality_signature(
    tree: FusionTree,
    site_order: Sequence[int],
    edges: Sequence[Sequence[Real]],
) -> HierarchyChiralitySignature:
    """Evaluate the state-blind cut-chirality signature of a fusion tree."""

    supports = fusion_tree_cut_supports(tree, site_order)
    gram = cut_chirality_gram(tree.n_leaves, supports, edges)
    return _signature_from_gram(tree.n_leaves, gram)


__all__ = [
    "ChiralityTriple",
    "CoefficientRow",
    "CutChiralityCoefficients",
    "CutChiralityGram",
    "EDGE_AGGREGATION_CONTRACT",
    "HierarchyChiralitySignature",
    "LEXICOGRAPHIC_USE_CONTRACT",
    "WeightedEdge",
    "cut_chirality_coefficients",
    "cut_chirality_gram",
    "fusion_tree_cut_supports",
    "hierarchy_chirality_signature",
]
