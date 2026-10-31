"""Matrix-free Heisenberg bonds in an arbitrary ordered fusion-tree basis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..su2.fusion_tree import FusionTree
from ..su2.tree_rotations import leaf_signature, recoupling_matrix_sparse


def _linear_operator():
    try:
        from scipy.sparse.linalg import LinearOperator
    except ImportError as error:
        raise RuntimeError("tree operator plans require scipy") from error
    return LinearOperator


def sibling_pair_tree(n: int, bond: int) -> FusionTree:
    """Return an ordered tree in which leaves ``bond,bond+1`` are siblings."""

    if n < 2 or not 0 <= bond < n - 1:
        raise ValueError("adjacent bond is outside the tree")
    items: list[object] = list(range(bond))
    items.append((bond, bond + 1))
    items.extend(range(bond + 2, n))
    nested = items[0]
    for item in items[1:]:
        nested = (nested, item)
    return FusionTree.from_nested(nested)  # type: ignore[arg-type]


@dataclass(frozen=True)
class AdjacentSwapPlan:
    """F-move plan for one adjacent physical transposition."""

    bond: int
    recoupling: object
    eigenvalues: np.ndarray
    f_move_count: int

    def apply(self, vector: np.ndarray) -> np.ndarray:
        """Apply the swap without materializing it in the source basis."""

        transformed = self.recoupling @ np.asarray(vector)
        return np.asarray(self.recoupling.T @ (self.eigenvalues * transformed))


def build_adjacent_swap_plan(
    tree: FusionTree,
    target_q: int,
    bond: int,
) -> AdjacentSwapPlan:
    """Recouple an adjacent leaf pair to siblings and cache its swap parity."""

    pair_tree = sibling_pair_tree(tree.n_leaves, bond)
    recoupling, moves = recoupling_matrix_sparse(tree, pair_tree, target_q)
    pair_index = next(
        index
        for index, node in enumerate(pair_tree.internal_nodes)
        if leaf_signature(node) == (bond, bond + 1)
    )
    eigenvalues = np.asarray(
        [
            -1.0 if state.labels[pair_index] == 0 else 1.0
            for state in pair_tree.enumerate_states(target_q)
        ],
        dtype=np.float64,
    )
    return AdjacentSwapPlan(bond, recoupling, eigenvalues, len(moves))


def transposition_sequence(first: int, second: int, n: int) -> tuple[int, ...]:
    """Adjacent-swap word for physical transposition ``(first,second)``."""

    if first > second:
        first, second = second, first
    if not 0 <= first <= second < n:
        raise ValueError("transposition sites lie outside the system")
    if first == second:
        return ()
    return tuple(range(first, second)) + tuple(range(second - 2, first - 1, -1))


@dataclass(frozen=True)
class TreeHeisenbergPlan:
    """Reusable matrix-free plan for a graph Hamiltonian in one tree basis."""

    tree: FusionTree
    target_q: int
    bonds: tuple[tuple[int, int, float], ...]
    adjacent_plans: tuple[AdjacentSwapPlan, ...]

    @property
    def dimension(self) -> int:
        return len(self.tree.enumerate_states(self.target_q))

    @property
    def stored_nnz(self) -> int:
        return int(sum(plan.recoupling.nnz for plan in self.adjacent_plans))

    def apply_transposition(
        self, vector: np.ndarray, first: int, second: int
    ) -> np.ndarray:
        value = np.asarray(vector)
        for adjacent in transposition_sequence(first, second, self.tree.n_leaves):
            value = self.adjacent_plans[adjacent].apply(value)
        return value

    def apply(self, vector: np.ndarray) -> np.ndarray:
        """Apply ``sum J_ij S_i.S_j`` through cached local recouplings."""

        source = np.asarray(vector)
        if source.shape != (self.dimension,):
            raise ValueError(f"expected vector shape ({self.dimension},), got {source.shape}")
        result = np.zeros_like(source, dtype=np.result_type(source, np.float64))
        for first, second, coupling in self.bonds:
            permuted = self.apply_transposition(source, first, second)
            result += coupling * (0.5 * permuted - 0.25 * source)
        return result

    def as_linear_operator(self):
        """Expose the plan to SciPy iterative eigensolvers."""

        LinearOperator = _linear_operator()
        dimension = self.dimension
        return LinearOperator(
            (dimension, dimension),
            matvec=self.apply,
            rmatvec=self.apply,
            dtype=np.dtype(np.float64),
        )

    def to_jax(self) -> "JaxTreeHeisenbergPlan":
        """Convert cached recouplings to a JIT-compatible JAX sparse plan."""

        try:
            import jax.numpy as jnp
            from jax.experimental import sparse as jsparse
        except ImportError as error:
            raise RuntimeError("compiled tree operator plans require jax") from error

        adjacent = []
        for plan in self.adjacent_plans:
            coo = plan.recoupling.tocoo()
            indices = np.column_stack((coo.row, coo.col)).astype(np.int32)
            matrix = jsparse.BCOO(
                (jnp.asarray(coo.data), jnp.asarray(indices)),
                shape=coo.shape,
                indices_sorted=False,
                unique_indices=True,
            )
            adjacent.append(
                JaxAdjacentSwapPlan(
                    matrix,
                    jnp.asarray(plan.eigenvalues),
                    plan.f_move_count,
                )
            )
        return JaxTreeHeisenbergPlan(
            self.tree, self.target_q, self.bonds, tuple(adjacent), self.dimension
        )


@dataclass(frozen=True)
class JaxAdjacentSwapPlan:
    """One adjacent transposition represented by a JAX ``BCOO`` F-map."""

    recoupling: object
    eigenvalues: object
    f_move_count: int

    def apply(self, vector):
        transformed = self.recoupling @ vector
        return self.recoupling.T @ (self.eigenvalues * transformed)


@dataclass(frozen=True)
class JaxTreeHeisenbergPlan:
    """Static sparse tree plan whose Python loops are unrolled by ``jax.jit``."""

    tree: FusionTree
    target_q: int
    bonds: tuple[tuple[int, int, float], ...]
    adjacent_plans: tuple[JaxAdjacentSwapPlan, ...]
    dimension: int

    def apply_transposition(self, vector, first: int, second: int):
        value = vector
        for adjacent in transposition_sequence(first, second, self.tree.n_leaves):
            value = self.adjacent_plans[adjacent].apply(value)
        return value

    def apply(self, vector):
        """Apply the Hamiltonian to one vector with JAX-traceable operations."""

        import jax.numpy as jnp

        source = jnp.asarray(vector)
        if source.ndim != 1 or source.shape[0] != self.dimension:
            raise ValueError(f"expected vector shape ({self.dimension},), got {source.shape}")
        result = jnp.zeros_like(source)
        for first, second, coupling in self.bonds:
            permuted = self.apply_transposition(source, first, second)
            result = result + coupling * (0.5 * permuted - 0.25 * source)
        return result

    def apply_batch(self, vectors):
        """Apply the plan to a leading batch of state vectors."""

        import jax
        import jax.numpy as jnp

        values = jnp.asarray(vectors)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(
                f"expected vectors shape (batch, {self.dimension}), got {values.shape}"
            )
        return jax.vmap(self.apply)(values)

    def compiled_apply(self):
        """Return a compiled single-vector operator callable."""

        import jax

        return jax.jit(self.apply)

    def compiled_apply_batch(self):
        """Return a compiled batched operator callable."""

        import jax

        return jax.jit(self.apply_batch)

    def compiled_local_apply_batch(self):
        """Compile each adjacent F-map locally and keep plan traversal in Python.

        This avoids constructing one enormous XLA graph for all bonds and
        transposition words.  Calls remain asynchronous until the returned
        batch is synchronized by the caller.
        """

        import jax
        import jax.numpy as jnp

        kernels = tuple(
            jax.jit(jax.vmap(adjacent.apply)) for adjacent in self.adjacent_plans
        )

        def apply(vectors):
            source = jnp.asarray(vectors)
            if source.ndim != 2 or source.shape[1] != self.dimension:
                raise ValueError(
                    f"expected vectors shape (batch, {self.dimension}), got {source.shape}"
                )
            result = jnp.zeros_like(source)
            for first, second, coupling in self.bonds:
                value = source
                for adjacent in transposition_sequence(first, second, self.tree.n_leaves):
                    value = kernels[adjacent](value)
                result = result + coupling * (0.5 * value - 0.25 * source)
            return result

        return apply


def build_tree_heisenberg_plan(
    tree: FusionTree,
    bonds: Iterable[tuple[int, int] | tuple[int, int, float]],
    *,
    target_q: int = 0,
) -> TreeHeisenbergPlan:
    """Build the adjacent F-move plans needed by the graph bonds."""

    normalized: list[tuple[int, int, float]] = []
    for bond in bonds:
        if len(bond) == 2:
            first, second = bond
            coupling = 1.0
        elif len(bond) == 3:
            first, second, coupling = bond
        else:
            raise ValueError(f"invalid bond: {bond!r}")
        normalized.append((int(first), int(second), float(coupling)))
    plans = tuple(
        build_adjacent_swap_plan(tree, target_q, adjacent)
        for adjacent in range(tree.n_leaves - 1)
    )
    return TreeHeisenbergPlan(tree, target_q, tuple(normalized), plans)
