from math import log2

import numpy as np

from su2_nqs.benchmarks.tree_physics import (
    hamiltonian_basis_metrics,
    tree_wavefunction_metrics,
)
from su2_nqs.su2.fusion_tree import FusionTree


def test_tree_wavefunction_metrics_obey_born_entropy_chain_rule():
    tree = FusionTree.balanced(8)
    dimension = len(tree.enumerate_states(0))
    uniform = np.ones(dimension, dtype=np.complex128)
    metrics = tree_wavefunction_metrics(tree, 3.0j * uniform)
    np.testing.assert_allclose(metrics["participation_entropy"], log2(dimension), atol=1e-14)
    np.testing.assert_allclose(
        metrics["born_conditional_entropy_sum"], log2(dimension), atol=1e-14
    )
    np.testing.assert_allclose(metrics["effective_support"], dimension, atol=1e-13)


def test_tree_wavefunction_metrics_detect_a_basis_state():
    tree = FusionTree.left_associated(6)
    amplitudes = np.zeros(len(tree.enumerate_states(0)))
    amplitudes[1] = -2.0
    metrics = tree_wavefunction_metrics(tree, amplitudes)
    assert metrics["participation_entropy"] == 0.0
    assert metrics["born_conditional_entropy_sum"] == 0.0
    assert metrics["inverse_participation_support"] == 1.0


def test_hamiltonian_basis_metrics_count_effective_connectivity():
    matrix = np.asarray([[2.0, -1.0, 1.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 3.0]])
    metrics = hamiltonian_basis_metrics(matrix)
    assert metrics["offdiagonal_nnz"] == 4.0
    np.testing.assert_allclose(metrics["offdiagonal_nnz_per_row"], 4.0 / 3.0)
    np.testing.assert_allclose(metrics["positive_offdiagonal_fraction"], 0.5)
