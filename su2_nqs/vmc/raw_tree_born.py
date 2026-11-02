"""Exact independent physical Born draws from a canonical SU(2) raw tree.

Only physical leaf outcomes are measured. Internal doubled-q and multiplicity
labels remain coherent. Conditional virtual densities are blocked by doubled
magnetization m, NOT q; coherences between different q at the same m survive.
No Hamiltonian, rejection chain, probability floor, or hidden-path sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.fusion_tree import Leaf, TreeNode
from ..su2.reduced_tree_environment import ReducedSingletTree


@dataclass(frozen=True)
class PhysicalBornDraw:
    """One physical spin-half outcome and its exact normalized raw Born law."""

    bits: np.ndarray
    log_probability: float
    amplitude: complex
    log_probability_replay_error: float


class RawTreeBornSampler:
    """Sequential whole-leaf measurement of an isometric raw singlet tree.

    ``leaf_isometries[i][q]`` has shape ``(2**len(sites),d_q,q+1)`` with
    increasing doubled m and bit zero meaning spin down. ``leaf_sites`` maps
    those ordered physical bits to global sites. Every local/upper map must
    be isometric; arbitrary noncanonical banks are deliberately rejected.
    The root vector may carry a coherent multiplicity superposition.
    """

    def __init__(self, state: ReducedSingletTree,
                 leaf_isometries: Sequence[Mapping[int, np.ndarray]],
                 leaf_sites: Sequence[Sequence[int]], *,
                 memory_limit_bytes: int = 768*1024**2,
                 tolerance: float = 2e-10,
                 shared_tensor_cache: dict | None = None):
        if not isinstance(state, ReducedSingletTree):
            raise TypeError("validated reduced singlet tree required")
        if (isinstance(memory_limit_bytes, bool) or not isinstance(memory_limit_bytes, Integral)
                or memory_limit_bytes < 1 or not np.isfinite(tolerance) or not 0 < tolerance < 1e-5):
            raise ValueError("positive memory budget and finite small tolerance required")
        if len(leaf_isometries) != state.tree.n_leaves or len(leaf_sites) != state.tree.n_leaves:
            raise ValueError("one physical isometry and ordered site list per leaf required")
        flat = [s for sites in leaf_sites for s in sites]
        if (any(isinstance(s, bool) or not isinstance(s, Integral) for s in flat)
                or sorted(flat) != list(range(len(flat)))):
            raise ValueError("leaf sites must partition physical sites 0..N-1")
        self.state = state; self.n_sites = len(flat); self.tolerance = tolerance
        self.widths = {}; self.positions = {}; self.leaves = {}; self.tensors = {}
        cache = {} if shared_tensor_cache is None else shared_tensor_cache
        maximum_isometry_error = 0.
        for leaf, iso, sites, grams in zip(state.tree.leaves, leaf_isometries, leaf_sites,
                                          state.leaf_grams, strict=True):
            if not sites or len(sites) > 12 or set(iso) != set(grams):
                raise ValueError("small physical leaves and matching doubled-q sectors required")
            widths = {q: len(g) for q, g in grams.items()}
            self.widths[leaf] = widths
            positions = self._positions(widths); self.positions[leaf] = positions
            rows = 2**len(sites)
            physical_m = np.asarray([2*i.bit_count()-len(sites) for i in range(rows)])
            features = {}
            for q, d in widths.items():
                value = np.asarray(iso[q])
                if value.shape != (rows, d, q+1) or not np.isfinite(value).all():
                    raise ValueError("finite physical leaf arrays with declared q/m widths required")
                if np.max(abs(np.asarray(grams[q])-np.eye(d))) > tolerance:
                    raise ValueError("canonical isometric leaf metrics required")
                for index, m in enumerate(range(-q, q+1, 2)):
                    if np.max(abs(value[physical_m != m, :, index]), initial=0.) > tolerance:
                        raise ValueError("physical leaf amplitudes violate doubled magnetization")
            for m, (slices, d) in positions.items():
                selected = np.flatnonzero(physical_m == m)
                matrix = np.concatenate([np.asarray(iso[q])[selected, :, (m+q)//2]
                                         for q in slices], axis=1)
                error = float(np.max(abs(matrix.conj().T@matrix-np.eye(d))))
                maximum_isometry_error = max(maximum_isometry_error, error)
                if error > tolerance:
                    raise ValueError("physical leaf columns must be mutually orthonormal at fixed m")
                features[m] = (selected, matrix)
            self.leaves[leaf] = (tuple(map(int, sites)), features)
        dtype = np.result_type(*[np.asarray(t).dtype for kernels in state.kernels for t in kernels.values()],
                               np.asarray(state.root_vector).dtype, np.float64)
        used_cache_keys = set()
        for node, kernels in zip(state.tree.internal_nodes, state.kernels, strict=True):
            widths = {qt: np.asarray(t).shape[0] for (_, _, qt), t in kernels.items()}
            self.widths[node] = widths; self.positions[node] = self._positions(widths)
            for q, d in widths.items():
                matrix = np.concatenate([np.asarray(t).reshape(d, -1).T
                                         for (_, _, qt), t in kernels.items() if qt == q])
                error = float(np.max(abs(matrix.conj().T@matrix-np.eye(d))))
                maximum_isometry_error = max(maximum_isometry_error, error)
                if error > tolerance:
                    raise ValueError("canonical isometric upper maps required")
            digest = sha256(str((self.widths[node.left], self.widths[node.right], widths, dtype)).encode())
            for key, value in sorted(kernels.items()):
                digest.update(str(key).encode()); digest.update(np.ascontiguousarray(value).tobytes())
            key = digest.hexdigest(); used_cache_keys.add(key)
            if key not in cache:
                lp, rp, pp = (self.positions[n] for n in (node.left, node.right, node))
                estimated = sum(dl*dr*pp[ml+mr][1]*dtype.itemsize
                    for ml, (_, dl) in lp.items() for mr, (_, dr) in rp.items() if ml+mr in pp)
                resident = sum(t.nbytes for value in cache.values() for t in value.values())
                if resident+estimated > memory_limit_bytes:
                    raise MemoryError("magnetic-block tensor cache exceeds explicit memory budget")
                tensors = {(ml, mr): np.zeros((dl, dr, pp[ml+mr][1]), dtype=dtype)
                    for ml, (_, dl) in lp.items() for mr, (_, dr) in rp.items() if ml+mr in pp}
                for (ql, qr, qp), kernel in kernels.items():
                    reduced = np.asarray(kernel).transpose(1, 2, 0)
                    for ml in range(-ql, ql+1, 2):
                        for mr in range(-qr, qr+1, 2):
                            mp = ml+mr
                            if abs(mp) <= qp:
                                coefficient = clebsch_gordan(ql, ml, qr, mr, qp, mp)
                                if coefficient:
                                    tensors[ml, mr][lp[ml][0][ql], rp[mr][0][qr], pp[mp][0][qp]] += coefficient*reduced
                cache[key] = tensors
            self.tensors[node] = cache[key]
        self.root_vector = np.asarray(state.root_vector)
        root_norm = float(np.vdot(self.root_vector, self.root_vector).real)
        if abs(root_norm-1.) > tolerance:
            raise ValueError("normalized coherent root vector required")
        self.diagnostics = dict(physical_sites=self.n_sites,
            maximum_isometry_error=maximum_isometry_error,
            unique_magnetic_tensor_bytes=sum(t.nbytes for key in used_cache_keys for t in cache[key].values()),
            all_shared_tensor_cache_bytes=sum(t.nbytes for value in cache.values() for t in value.values()),
            memory_budget_excludes_physical_leaf_tables_and_transient_density_workspaces=True,
            internal_q_and_multiplicity_never_measured=True)

    @staticmethod
    def _positions(widths: Mapping[int, int]) -> dict:
        result = {}
        for m in sorted({m for q in widths for m in range(-q, q+1, 2)}):
            slices = {}; offset = 0
            for q, d in sorted(widths.items()):
                if abs(m) <= q:
                    slices[q] = slice(offset, offset+d); offset += d
            result[m] = (slices, offset)
        return result

    def draw(self, rng: np.random.Generator) -> PhysicalBornDraw:
        """Draw one independent physical outcome using caller-owned RNG state."""
        if not isinstance(rng, np.random.Generator):
            raise TypeError("explicit NumPy Generator required")
        bits = np.zeros(self.n_sites, np.int32)
        root_rho = {0: np.outer(self.root_vector, self.root_vector.conj())}

        def visit(node: TreeNode, rho: dict) -> tuple[int, np.ndarray, float]:
            trace = float(sum(np.trace(value).real for value in rho.values()))
            if not np.isfinite(trace) or abs(trace-1.) > 50*self.tolerance:
                raise ArithmeticError("conditional physical density lost unit trace")
            if isinstance(node, Leaf):
                sites, features = self.leaves[node]
                probabilities = np.zeros(2**len(sites))
                for m, (rows, matrix) in features.items():
                    if m in rho:
                        probabilities[rows] = np.einsum("ia,ab,ib->i", matrix, rho[m], matrix.conj(), optimize=True).real
                if not np.isfinite(probabilities).all() or probabilities.min() < -self.tolerance*trace:
                    raise ArithmeticError("materially negative or nonfinite physical conditional probability")
                # Only negative floating-point roundoff, bounded above, is
                # removed. No missing q sector or physical outcome is added.
                probabilities = np.maximum(probabilities, 0.)
                normalization = probabilities.sum()
                if abs(normalization-1.) > 50*self.tolerance:
                    raise ArithmeticError("physical leaf conditional failed normalization")
                outcome = int(rng.choice(len(probabilities), p=probabilities/normalization))
                local = (outcome >> np.arange(len(sites)-1, -1, -1)) & 1
                bits[list(sites)] = local
                m = int(2*local.sum()-len(sites)); rows, matrix = features[m]
                position = int(np.searchsorted(rows, outcome))
                return m, matrix[position], float(np.log(probabilities[outcome]/normalization))
            tensors = self.tensors[node]
            left_rho = {}
            for (ml, mr), tensor in tensors.items():
                if ml+mr not in rho:
                    continue
                dl, dr, dp = tensor.shape
                weighted = (tensor.reshape(dl*dr, dp)@rho[ml+mr]).reshape(dl, dr*dp)
                value = weighted@tensor.reshape(dl, dr*dp).conj().T
                left_rho[ml] = left_rho.get(ml, np.zeros_like(value))+value
            ml, feature_left, log_left = visit(node.left, left_rho)
            probability_left = float(np.real(feature_left@left_rho[ml]@feature_left.conj()))
            if not np.isfinite(probability_left) or probability_left <= 0:
                raise ArithmeticError("sampled left physical outcome has zero conditional probability")
            if abs(np.log(probability_left)-log_left) > 100*self.tolerance:
                raise ArithmeticError("left conditional probability disagrees with physical-feature contraction")
            right_rho = {}; conditioned = {}
            for (mleft, mr), tensor in tensors.items():
                if mleft != ml or ml+mr not in rho:
                    continue
                matrix = np.einsum("a,abp->bp", feature_left, tensor, optimize=True)
                conditioned[mr] = matrix
                right_rho[mr] = (matrix@rho[ml+mr]@matrix.conj().T)/probability_left
            mr, feature_right, log_right = visit(node.right, right_rho)
            # Ket coefficients require an ordinary transpose, not adjoint.
            return ml+mr, conditioned[mr].T@feature_right, log_left+log_right

        m, feature, log_probability = visit(self.state.tree.root, root_rho)
        amplitude = complex(feature@self.root_vector)
        if m != 0 or int(bits.sum())*2 != self.n_sites or not np.isfinite(amplitude):
            raise ArithmeticError("physical draw violated exact singlet support or finiteness")
        error = abs(log_probability-2*np.log(abs(amplitude)))
        if not np.isfinite(error) or error > 100*self.tolerance:
            raise ArithmeticError("telescoping sampling probability disagrees with raw physical Born amplitude")
        return PhysicalBornDraw(bits, log_probability, amplitude, float(error))
