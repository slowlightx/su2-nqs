"""Physical diagnostics for short-range cluster cores in spin NQS.

The main use case is a periodic lattice tiled by translated ``2 x 3``
clusters.  The routines in this module keep three logically distinct
questions separate:

* how much Hamiltonian energy lies on bonds inside or between clusters;
* which total-spin sectors occur on a six-site cluster; and
* whether the state develops the expected square-Neel or triangular-120
  correlations.

Every estimator acts on physical computational-basis bits, ``0=down`` and
``1=up``.  The pair actions contain the full ``S_i . S_j`` operator, not only
its diagonal ``S_i^z S_j^z`` part.  Cluster-spin labels are doubled integers
``q=2S`` in accordance with the repository-wide convention.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .spin_observables import (
    SpinStructureFactorSpec,
    prepare_spin_structure_factor,
    reciprocal_phase_weights,
)


SpinAmplitude = Callable[[object, jax.Array], jax.Array]
WeightedBond = tuple[int, int, float]


@dataclass(frozen=True)
class TranslatedClusterCover:
    """One distinct translated tiling and its group-orbit multiplicity.

    ``clusters`` are disjoint sorted supports that cover every site exactly
    once.  On a ``6 x 6`` torus, translating a ``2 x 3`` tiling by all 36
    lattice translations produces six distinct covers, each with
    ``translation_multiplicity=6``.
    """

    clusters: tuple[tuple[int, ...], ...]
    translation_multiplicity: int


@dataclass(frozen=True)
class SpinPairObservableBatchSpec:
    """Several operators sharing one set of unordered spin pairs.

    Mode ``a`` represents

    ``O_a = constants[a] I + sum_p coefficients[a,p] S_i[p] . S_j[p]``.

    The common pair layout lets all modes share connected-amplitude work.
    """

    n_sites: int
    labels: tuple[str, ...]
    constants: tuple[float, ...]
    pair_first: tuple[int, ...]
    pair_second: tuple[int, ...]
    coefficients: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class ClusterBondDecompositionSpec:
    """Translated-cover bond partition and compiled pair observables.

    For every cover there are consecutive ``intra`` and ``inter`` modes.
    The final three modes are the translation-weighted ``average_intra``,
    ``average_inter``, and ``total`` operators.  Every per-cover pair, and
    the averaged pair, sums exactly to ``total`` coefficient by coefficient.
    """

    covers: tuple[TranslatedClusterCover, ...]
    observables: SpinPairObservableBatchSpec
    cover_mode_indices: tuple[tuple[int, int], ...]
    average_intra_index: int
    average_inter_index: int
    total_index: int


@dataclass(frozen=True)
class ClusterSpinSectorSpec:
    """Sparse row representation of total-spin projectors on equal clusters.

    ``projector_rows[q_index, local_state, slot]`` contains matrix elements
    ``<local_state|P_q|connected_local_state>``.  Only configurations with
    the same cluster magnetization are stored because ``P_q`` commutes with
    ``S_C^z``.  This reduces a six-spin action from 64 to at most 20 model
    evaluations per cluster.
    """

    n_sites: int
    clusters: tuple[tuple[int, ...], ...]
    q_values: tuple[int, ...]
    local_basis_bits: tuple[tuple[int, ...], ...]
    connected_local_indices: tuple[tuple[int, ...], ...]
    connected_valid: tuple[tuple[bool, ...], ...]
    projector_rows: tuple[tuple[tuple[float, ...], ...], ...]

    @property
    def cluster_size(self) -> int:
        return len(self.clusters[0])


@dataclass(frozen=True)
class ClusterStateProjectorSpec:
    """Rank-one projector onto one normalized local cluster state.

    ``local_state`` follows the repository computational-basis ordering with
    the first cluster site as the most-significant bit.  Consequently each
    cluster tuple must preserve the physical orientation used to define
    ``local_state``; a sorted support that wraps a periodic boundary is not
    generally an oriented translate.  Keeping this
    projector separate from total-spin sectors matters physically: six
    spin-halves contain five orthogonal singlets, so ``P(S_C=0)`` alone does
    not measure fidelity to the chosen short-range cluster ground state.
    """

    n_sites: int
    clusters: tuple[tuple[int, ...], ...]
    local_state: tuple[complex, ...]
    connected_local_indices: tuple[int, ...]

    @property
    def cluster_size(self) -> int:
        return len(self.clusters[0])


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _canonical_bonds(
    n_sites: int, bonds: Iterable[Sequence[float | int]]
) -> tuple[WeightedBond, ...]:
    combined: dict[tuple[int, int], float] = {}
    for raw in bonds:
        if len(raw) == 2:
            first, second = raw
            coupling = 1.0
        elif len(raw) == 3:
            first, second, coupling = raw
        else:
            raise ValueError(f"invalid spin bond: {raw!r}")
        if (
            isinstance(first, bool)
            or isinstance(second, bool)
            or not isinstance(first, Integral)
            or not isinstance(second, Integral)
        ):
            raise TypeError("spin-bond sites must be integers")
        first, second = int(first), int(second)
        value = float(coupling)
        if first == second or not (0 <= first < n_sites and 0 <= second < n_sites):
            raise ValueError(f"invalid spin bond: {raw!r}")
        if not math.isfinite(value):
            raise ValueError("spin-bond couplings must be finite")
        edge = tuple(sorted((first, second)))
        combined[edge] = combined.get(edge, 0.0) + value
    return tuple(
        (first, second, coupling)
        for (first, second), coupling in sorted(combined.items())
        if coupling != 0.0
    )


def translated_rectangular_cluster_covers(
    rows: int,
    columns: int,
    *,
    cluster_rows: int = 2,
    cluster_columns: int = 3,
) -> tuple[TranslatedClusterCover, ...]:
    """Return distinct translations of a periodic rectangular cluster tiling.

    All ``rows * columns`` translations are enumerated.  Translations giving
    the same unordered set of cluster supports are combined, and their exact
    multiplicity is retained for orbit averages.  The construction uses only
    integer lattice coordinates and is equally applicable to the repository's
    square and 120-degree triangular coordinate conventions.
    """

    rows = _positive_integer("rows", rows)
    columns = _positive_integer("columns", columns)
    cluster_rows = _positive_integer("cluster_rows", cluster_rows)
    cluster_columns = _positive_integer("cluster_columns", cluster_columns)
    if rows % cluster_rows or columns % cluster_columns:
        raise ValueError("cluster shape must tile the rectangular torus exactly")

    base = []
    for row0 in range(0, rows, cluster_rows):
        for column0 in range(0, columns, cluster_columns):
            base.append(
                tuple(
                    (row0 + dr) * columns + column0 + dc
                    for dr in range(cluster_rows)
                    for dc in range(cluster_columns)
                )
            )

    counts: dict[tuple[tuple[int, ...], ...], int] = {}
    for dy in range(rows):
        for dx in range(columns):
            translated = []
            for cluster in base:
                support = []
                for site in cluster:
                    row, column = divmod(site, columns)
                    support.append(
                        ((row + dy) % rows) * columns + (column + dx) % columns
                    )
                translated.append(tuple(sorted(support)))
            cover = tuple(sorted(translated))
            counts[cover] = counts.get(cover, 0) + 1

    result = tuple(
        TranslatedClusterCover(cover, counts[cover]) for cover in sorted(counts)
    )
    expected_sites = tuple(range(rows * columns))
    for item in result:
        flattened = tuple(sorted(site for cluster in item.clusters for site in cluster))
        if flattened != expected_sites:
            raise RuntimeError("a translated cluster cover lost or repeated a site")
    if sum(item.translation_multiplicity for item in result) != rows * columns:
        raise RuntimeError("translated-cover multiplicities do not exhaust the group")
    return result


def prepare_spin_pair_observables(
    n_sites: int,
    modes: Sequence[
        tuple[str, float, Iterable[Sequence[float | int]]]
    ],
) -> SpinPairObservableBatchSpec:
    """Compile named constant-plus-Heisenberg-pair observables.

    Repeated bonds within a mode are combined.  The union of all nonzero
    pairs is stored once, making a multi-mode local estimator substantially
    cheaper than evaluating every operator independently.
    """

    n_sites = _positive_integer("n_sites", n_sites)
    if not modes:
        raise ValueError("modes must contain at least one observable")
    labels: list[str] = []
    constants: list[float] = []
    compiled: list[dict[tuple[int, int], float]] = []
    for label, constant, bonds in modes:
        name = str(label)
        if not name or name in labels:
            raise ValueError("observable labels must be nonempty and unique")
        offset = float(constant)
        if not math.isfinite(offset):
            raise ValueError("observable constants must be finite")
        canonical = _canonical_bonds(n_sites, bonds)
        labels.append(name)
        constants.append(offset)
        compiled.append({(first, second): value for first, second, value in canonical})
    pairs = tuple(sorted(set().union(*(set(item) for item in compiled))))
    return SpinPairObservableBatchSpec(
        n_sites=n_sites,
        labels=tuple(labels),
        constants=tuple(constants),
        pair_first=tuple(pair[0] for pair in pairs),
        pair_second=tuple(pair[1] for pair in pairs),
        coefficients=tuple(
            tuple(item.get(pair, 0.0) for pair in pairs) for item in compiled
        ),
    )


def prepare_translated_cluster_bond_decomposition(
    rows: int,
    columns: int,
    bonds: Iterable[Sequence[float | int]],
    *,
    cluster_rows: int = 2,
    cluster_columns: int = 3,
) -> ClusterBondDecompositionSpec:
    """Partition a Hamiltonian into intra/inter-cluster bonds for all covers."""

    rows = _positive_integer("rows", rows)
    columns = _positive_integer("columns", columns)
    n_sites = rows * columns
    canonical = _canonical_bonds(n_sites, bonds)
    covers = translated_rectangular_cluster_covers(
        rows,
        columns,
        cluster_rows=cluster_rows,
        cluster_columns=cluster_columns,
    )
    modes: list[tuple[str, float, tuple[WeightedBond, ...]]] = []
    cover_indices = []
    average_intra: dict[tuple[int, int], float] = {}
    average_inter: dict[tuple[int, int], float] = {}
    group_order = sum(item.translation_multiplicity for item in covers)
    for index, cover in enumerate(covers):
        owner = {
            site: cluster_index
            for cluster_index, cluster in enumerate(cover.clusters)
            for site in cluster
        }
        intra = tuple(
            bond for bond in canonical if owner[bond[0]] == owner[bond[1]]
        )
        inter = tuple(
            bond for bond in canonical if owner[bond[0]] != owner[bond[1]]
        )
        intra_index = len(modes)
        modes.append((f"cover_{index}_intra", 0.0, intra))
        inter_index = len(modes)
        modes.append((f"cover_{index}_inter", 0.0, inter))
        cover_indices.append((intra_index, inter_index))
        weight = cover.translation_multiplicity / group_order
        for first, second, coupling in intra:
            edge = (first, second)
            average_intra[edge] = average_intra.get(edge, 0.0) + weight * coupling
        for first, second, coupling in inter:
            edge = (first, second)
            average_inter[edge] = average_inter.get(edge, 0.0) + weight * coupling

    def as_bonds(values: dict[tuple[int, int], float]) -> tuple[WeightedBond, ...]:
        return tuple((first, second, value) for (first, second), value in sorted(values.items()))

    average_intra_index = len(modes)
    modes.append(("translation_average_intra", 0.0, as_bonds(average_intra)))
    average_inter_index = len(modes)
    modes.append(("translation_average_inter", 0.0, as_bonds(average_inter)))
    total_index = len(modes)
    modes.append(("total", 0.0, canonical))
    observable_spec = prepare_spin_pair_observables(n_sites, modes)

    coefficients = np.asarray(observable_spec.coefficients, dtype=np.float64)
    for intra_index, inter_index in cover_indices:
        if not np.array_equal(
            coefficients[intra_index] + coefficients[inter_index],
            coefficients[total_index],
        ):
            raise RuntimeError("a cluster cover does not partition the Hamiltonian")
    average_error = np.max(
        np.abs(
            coefficients[average_intra_index]
            + coefficients[average_inter_index]
            - coefficients[total_index]
        ),
        initial=0.0,
    )
    if average_error > 2.0e-15:
        raise RuntimeError("the translated cover average does not partition the Hamiltonian")
    return ClusterBondDecompositionSpec(
        covers=covers,
        observables=observable_spec,
        cover_mode_indices=tuple(cover_indices),
        average_intra_index=average_intra_index,
        average_inter_index=average_inter_index,
        total_index=total_index,
    )


def spin_pair_observable_hpsi(
    params: object,
    samples: jax.Array,
    spec: SpinPairObservableBatchSpec,
    *,
    amplitude_fn: SpinAmplitude,
) -> tuple[jax.Array, jax.Array]:
    """Return centers and all compiled pair-observable actions.

    The action array has shape ``(batch, modes)`` and is differentiable with
    respect to ``params``.  ``spec`` is intended to be closed over by JIT.
    """

    spins = jnp.asarray(samples, dtype=jnp.int32)
    if spins.ndim != 2 or spins.shape[0] < 1 or spins.shape[1] != spec.n_sites:
        raise ValueError(f"samples must have shape (batch, {spec.n_sites})")
    centers = jax.vmap(lambda row: amplitude_fn(params, row))(spins)
    if centers.ndim != 1:
        raise ValueError("amplitude_fn must return one scalar per configuration")
    dtype = jnp.result_type(centers.dtype, jnp.float64)
    constants = jnp.asarray(spec.constants, dtype=dtype)
    coefficients = jnp.asarray(spec.coefficients, dtype=dtype)
    actions = centers[:, None] * constants[None, :]
    if not spec.pair_first:
        return centers, actions

    first = jnp.asarray(spec.pair_first, dtype=jnp.int32)
    second = jnp.asarray(spec.pair_second, dtype=jnp.int32)
    slots = jnp.arange(len(spec.pair_first), dtype=jnp.int32)
    same = spins[:, first] == spins[:, second]
    diagonal = jnp.where(same, 0.25, -0.25)
    actions = actions + centers[:, None] * (diagonal @ coefficients.T)

    def swapped_pairs(row: jax.Array) -> jax.Array:
        connected = jnp.broadcast_to(row, (len(spec.pair_first), spec.n_sites))
        connected = connected.at[slots, first].set(row[second])
        return connected.at[slots, second].set(row[first])

    connected = jax.vmap(swapped_pairs)(spins)
    amplitudes = jax.vmap(lambda row: amplitude_fn(params, row))(
        connected.reshape((-1, spec.n_sites))
    ).reshape(same.shape)
    amplitudes = jnp.where(same, jnp.zeros((), dtype=amplitudes.dtype), amplitudes)
    return centers, actions + 0.5 * (amplitudes @ coefficients.T)


def spin_pair_observable_local_values(
    params: object,
    samples: jax.Array,
    spec: SpinPairObservableBatchSpec,
    *,
    amplitude_fn: SpinAmplitude,
    zero_tolerance: float = 0.0,
) -> jax.Array:
    """Return local values with shape ``(batch, modes)``.

    Zero-amplitude centers fail closed to ``NaN`` instead of silently entering
    a Monte Carlo mean.
    """

    tolerance = float(zero_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    centers, actions = spin_pair_observable_hpsi(
        params, samples, spec, amplitude_fn=amplitude_fn
    )
    local = actions / centers[:, None]
    return jnp.where(
        jnp.abs(centers[:, None]) > tolerance,
        local,
        jnp.full_like(local, jnp.nan),
    )


@lru_cache(maxsize=None)
def _cluster_spin_projectors(
    cluster_size: int,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """Return doubled-spin labels, local bits, and dense exact projectors."""

    size = _positive_integer("cluster_size", cluster_size)
    dimension = 1 << size
    bits = np.asarray(
        [
            [(state >> (size - 1 - site)) & 1 for site in range(size)]
            for state in range(dimension)
        ],
        dtype=np.int32,
    )
    s2 = 0.75 * size * np.eye(dimension, dtype=np.float64)
    for first in range(size):
        for second in range(first + 1, size):
            pair = np.zeros((dimension, dimension), dtype=np.float64)
            for state, row in enumerate(bits):
                if row[first] == row[second]:
                    pair[state, state] += 0.25
                else:
                    pair[state, state] -= 0.25
                    flipped = np.array(row, copy=True)
                    flipped[first], flipped[second] = flipped[second], flipped[first]
                    target = int(
                        sum(
                            int(bit) << (size - 1 - site)
                            for site, bit in enumerate(flipped)
                        )
                    )
                    pair[state, target] += 0.5
            s2 += 2.0 * pair
    eigenvalues, eigenvectors = np.linalg.eigh(s2)
    q_values = tuple(range(size % 2, size + 1, 2))
    projectors = []
    for q in q_values:
        target = q * (q + 2) / 4.0
        selected = np.abs(eigenvalues - target) < 2.0e-10
        if not np.any(selected):
            raise RuntimeError(f"cluster total-spin sector q={q} is absent")
        projector = eigenvectors[:, selected] @ eigenvectors[:, selected].T
        projector = 0.5 * (projector + projector.T)
        projector[np.abs(projector) < 2.0e-14] = 0.0
        projectors.append(projector)
    stacked = np.asarray(projectors, dtype=np.float64)
    if np.max(np.abs(np.sum(stacked, axis=0) - np.eye(dimension))) > 2.0e-12:
        raise RuntimeError("cluster-spin projectors do not resolve the identity")
    bits.setflags(write=False)
    stacked.setflags(write=False)
    return q_values, bits, stacked


def prepare_cluster_spin_sectors(
    n_sites: int,
    clusters: Sequence[Sequence[int]],
) -> ClusterSpinSectorSpec:
    """Compile exact total-spin projectors for equal-size physical clusters."""

    n_sites = _positive_integer("n_sites", n_sites)
    if not clusters:
        raise ValueError("clusters must be nonempty")
    checked = []
    size = None
    for raw in clusters:
        if any(isinstance(site, bool) or not isinstance(site, Integral) for site in raw):
            raise TypeError("cluster sites must be integers")
        cluster = tuple(int(site) for site in raw)
        if not cluster or len(set(cluster)) != len(cluster):
            raise ValueError("each cluster must contain distinct sites")
        if any(site < 0 or site >= n_sites for site in cluster):
            raise ValueError("cluster site lies outside n_sites")
        if size is None:
            size = len(cluster)
        elif len(cluster) != size:
            raise ValueError("all clusters must have the same size")
        checked.append(cluster)
    if len(set(checked)) != len(checked):
        raise ValueError("clusters must be duplicate free")
    assert size is not None
    q_values, bits, projectors = _cluster_spin_projectors(size)
    dimension = 1 << size
    sectors = [np.flatnonzero(np.sum(bits, axis=1) == n_up) for n_up in range(size + 1)]
    width = max(len(indices) for indices in sectors)
    connected = np.zeros((dimension, width), dtype=np.int32)
    valid = np.zeros((dimension, width), dtype=bool)
    rows = np.zeros((len(q_values), dimension, width), dtype=np.float64)
    for local_state in range(dimension):
        indices = sectors[int(np.sum(bits[local_state]))]
        connected[local_state, : len(indices)] = indices
        valid[local_state, : len(indices)] = True
        rows[:, local_state, : len(indices)] = projectors[:, local_state, indices]
    return ClusterSpinSectorSpec(
        n_sites=n_sites,
        clusters=tuple(checked),
        q_values=q_values,
        local_basis_bits=tuple(tuple(int(bit) for bit in row) for row in bits),
        connected_local_indices=tuple(tuple(int(value) for value in row) for row in connected),
        connected_valid=tuple(tuple(bool(value) for value in row) for row in valid),
        projector_rows=tuple(
            tuple(tuple(float(value) for value in row) for row in sector)
            for sector in rows
        ),
    )


def prepare_translated_cluster_spin_sectors(
    rows: int,
    columns: int,
    *,
    cluster_rows: int = 2,
    cluster_columns: int = 3,
) -> ClusterSpinSectorSpec:
    """Compile sector projectors on every translated cluster support.

    Each distinct support is retained once.  For a ``6 x 6`` lattice with
    ``2 x 3`` clusters this gives 36 supports.  Their simple arithmetic mean
    is exactly the translation-orbit mean of the per-cover cluster average:
    every distinct cover has the same multiplicity and every site occurs in
    the same number of supports.
    """

    rows = _positive_integer("rows", rows)
    columns = _positive_integer("columns", columns)
    covers = translated_rectangular_cluster_covers(
        rows,
        columns,
        cluster_rows=cluster_rows,
        cluster_columns=cluster_columns,
    )
    clusters = tuple(
        sorted({cluster for cover in covers for cluster in cover.clusters})
    )
    return prepare_cluster_spin_sectors(rows * columns, clusters)


def prepare_cluster_state_projector(
    n_sites: int,
    clusters: Sequence[Sequence[int]],
    local_state: Sequence[complex],
    *,
    support_tolerance: float = 2.0e-14,
) -> ClusterStateProjectorSpec:
    """Compile ``|phi><phi|`` on one or more equal physical supports.

    Site order is semantically significant: ``clusters[k][i]`` receives
    local leg ``i``.  In particular, callers must not sort a translated
    periodic support unless ``local_state`` is invariant under that
    permutation.
    """

    n_sites = _positive_integer("n_sites", n_sites)
    if not clusters:
        raise ValueError("clusters must be nonempty")
    checked: list[tuple[int, ...]] = []
    size = None
    for raw in clusters:
        if any(isinstance(site, bool) or not isinstance(site, Integral) for site in raw):
            raise TypeError("cluster sites must be integers")
        cluster = tuple(int(site) for site in raw)
        if not cluster or len(set(cluster)) != len(cluster):
            raise ValueError("each cluster must contain distinct sites")
        if any(site < 0 or site >= n_sites for site in cluster):
            raise ValueError("cluster site lies outside n_sites")
        if size is None:
            size = len(cluster)
        elif len(cluster) != size:
            raise ValueError("all clusters must have the same size")
        checked.append(cluster)
    if len(set(checked)) != len(checked):
        raise ValueError("clusters must be duplicate free")
    assert size is not None
    state = np.asarray(tuple(local_state), dtype=np.complex128)
    if state.shape != (1 << size,):
        raise ValueError("local_state dimension must equal 2**cluster_size")
    if np.any(~np.isfinite(state)):
        raise FloatingPointError("local_state contains non-finite values")
    norm = float(np.vdot(state, state).real)
    if abs(norm - 1.0) > 2.0e-12:
        raise ValueError("local_state must be normalized")
    tolerance = float(support_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("support_tolerance must be finite and nonnegative")
    support = tuple(int(index) for index in np.flatnonzero(np.abs(state) > tolerance))
    if not support:
        raise ValueError("local_state has empty retained support")
    return ClusterStateProjectorSpec(
        n_sites=n_sites,
        clusters=tuple(checked),
        local_state=tuple(complex(value) for value in state),
        connected_local_indices=support,
    )


def cluster_spin_sector_local_values(
    params: object,
    samples: jax.Array,
    spec: ClusterSpinSectorSpec,
    *,
    amplitude_fn: SpinAmplitude,
    zero_tolerance: float = 0.0,
) -> jax.Array:
    """Return local cluster-sector projectors.

    The result has shape ``(batch, clusters, sectors)``.  After a Born-sample
    mean, the entries at ``q=0`` and ``q=2`` are respectively the singlet and
    triplet probabilities, while summing entries with ``q>=4`` gives
    ``P(S_C>=2)``.  Individual local values may be complex or lie outside
    ``[0,1]`` and must not themselves be called probabilities.  Contracting
    the last axis with ``S(S+1)=q(q+2)/4`` gives the local estimator of
    ``S_C^2``.
    """

    tolerance = float(zero_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    spins = jnp.asarray(samples, dtype=jnp.int32)
    if spins.ndim != 2 or spins.shape[0] < 1 or spins.shape[1] != spec.n_sites:
        raise ValueError(f"samples must have shape (batch, {spec.n_sites})")
    centers = jax.vmap(lambda row: amplitude_fn(params, row))(spins)
    basis = jnp.asarray(spec.local_basis_bits, dtype=jnp.int32)
    indices_by_state = jnp.asarray(spec.connected_local_indices, dtype=jnp.int32)
    valid_by_state = jnp.asarray(spec.connected_valid)
    projector_rows = jnp.asarray(spec.projector_rows, dtype=jnp.float64)
    binary_weights = 1 << jnp.arange(spec.cluster_size - 1, -1, -1, dtype=jnp.int32)

    def one_cluster(row: jax.Array, cluster_sites: tuple[int, ...]) -> jax.Array:
        sites = jnp.asarray(cluster_sites, dtype=jnp.int32)
        local_index = jnp.sum(row[sites] * binary_weights)
        connected_indices = indices_by_state[local_index]
        connected_bits = basis[connected_indices]
        connected = jnp.broadcast_to(row, (connected_indices.shape[0], spec.n_sites))
        connected = connected.at[:, sites].set(connected_bits)
        amplitudes = jax.vmap(lambda state: amplitude_fn(params, state))(connected)
        amplitudes = jnp.where(
            valid_by_state[local_index], amplitudes, jnp.zeros_like(amplitudes)
        )
        return projector_rows[:, local_index, :] @ amplitudes

    actions = jax.vmap(
        lambda row: jnp.stack(
            tuple(one_cluster(row, cluster) for cluster in spec.clusters), axis=0
        )
    )(spins)
    local = actions / centers[:, None, None]
    return jnp.where(
        jnp.abs(centers[:, None, None]) > tolerance,
        local,
        jnp.full_like(local, jnp.nan),
    )


def cluster_state_projector_local_values(
    params: object,
    samples: jax.Array,
    spec: ClusterStateProjectorSpec,
    *,
    amplitude_fn: SpinAmplitude,
    zero_tolerance: float = 0.0,
) -> jax.Array:
    """Return local values of the rank-one cluster-state projectors.

    The result has shape ``(batch, clusters)``.  For a translation-projected
    ``k=0`` state one representative translated support has the same exact
    expectation as the full support orbit, while retaining all supports can
    reduce Monte Carlo variance in a final audit.
    """

    tolerance = float(zero_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    spins = jnp.asarray(samples, dtype=jnp.int32)
    if spins.ndim != 2 or spins.shape[0] < 1 or spins.shape[1] != spec.n_sites:
        raise ValueError(f"samples must have shape (batch, {spec.n_sites})")
    centers = jax.vmap(lambda row: amplitude_fn(params, row))(spins)
    local_state = jnp.asarray(spec.local_state, dtype=jnp.complex128)
    retained = jnp.asarray(spec.connected_local_indices, dtype=jnp.int32)
    cluster_size = spec.cluster_size
    basis_bits = (
        (retained[:, None] >> jnp.arange(cluster_size - 1, -1, -1, dtype=jnp.int32))
        & 1
    ).astype(jnp.int32)
    binary_weights = 1 << jnp.arange(cluster_size - 1, -1, -1, dtype=jnp.int32)

    def one_cluster(row: jax.Array, cluster_sites: tuple[int, ...]) -> jax.Array:
        sites = jnp.asarray(cluster_sites, dtype=jnp.int32)
        local_index = jnp.sum(row[sites] * binary_weights)
        connected = jnp.broadcast_to(row, (retained.shape[0], spec.n_sites))
        connected = connected.at[:, sites].set(basis_bits)
        amplitudes = jax.vmap(lambda state: amplitude_fn(params, state))(connected)
        overlap_action = jnp.sum(jnp.conj(local_state[retained]) * amplitudes)
        return local_state[local_index] * overlap_action

    actions = jax.vmap(
        lambda row: jnp.stack(
            tuple(one_cluster(row, cluster) for cluster in spec.clusters), axis=0
        )
    )(spins)
    local = actions / centers[:, None]
    return jnp.where(
        jnp.abs(centers[:, None]) > tolerance,
        local,
        jnp.full_like(local, jnp.nan),
    )


def cluster_spin_squared_from_sector_values(
    sector_values: jax.Array,
    q_values: Sequence[int],
) -> jax.Array:
    """Contract sector local values with exact ``S(S+1)`` eigenvalues."""

    values = jnp.asarray(sector_values)
    checked = tuple(int(q) for q in q_values)
    if values.ndim < 1 or values.shape[-1] != len(checked):
        raise ValueError("the last sector_values axis must match q_values")
    eigenvalues = jnp.asarray(
        tuple(q * (q + 2) / 4.0 for q in checked), dtype=jnp.float64
    )
    return values @ eigenvalues


def square_neel_structure_factor_spec(
    rows: int, columns: int
) -> SpinStructureFactorSpec:
    """Return standard ``N^-1`` square-lattice ``S(pi,pi)``."""

    rows = _positive_integer("rows", rows)
    columns = _positive_integer("columns", columns)
    coordinates = tuple(
        (row, column) for row in range(rows) for column in range(columns)
    )
    return prepare_spin_structure_factor(
        reciprocal_phase_weights(coordinates, (math.pi, math.pi))
    )


def triangular_120_structure_factor_spec(
    rows: int, columns: int
) -> SpinStructureFactorSpec:
    """Return standard ``N^-1`` triangular-lattice ``S(K)``.

    Coordinates use the repository's 120-degree primitive vectors.  Thus
    ``K=(2*pi/3,2*pi/3)`` gives phase differences of ``+/- 2*pi/3`` on all
    three nearest-neighbour directions ``(1,0)``, ``(0,1)``, and ``(1,1)``.
    """

    rows = _positive_integer("rows", rows)
    columns = _positive_integer("columns", columns)
    coordinates = tuple(
        (row, column) for row in range(rows) for column in range(columns)
    )
    momentum = 2.0 * math.pi / 3.0
    return prepare_spin_structure_factor(
        reciprocal_phase_weights(coordinates, (momentum, momentum))
    )


__all__ = [
    "ClusterBondDecompositionSpec",
    "ClusterStateProjectorSpec",
    "ClusterSpinSectorSpec",
    "SpinPairObservableBatchSpec",
    "TranslatedClusterCover",
    "cluster_spin_sector_local_values",
    "cluster_spin_squared_from_sector_values",
    "cluster_state_projector_local_values",
    "prepare_cluster_state_projector",
    "prepare_cluster_spin_sectors",
    "prepare_spin_pair_observables",
    "prepare_translated_cluster_bond_decomposition",
    "prepare_translated_cluster_spin_sectors",
    "spin_pair_observable_hpsi",
    "spin_pair_observable_local_values",
    "square_neel_structure_factor_spec",
    "translated_rectangular_cluster_covers",
    "triangular_120_structure_factor_spec",
]
