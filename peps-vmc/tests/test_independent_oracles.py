"""Closed-network oracles independent of the PEPS contraction implementation.

The six-site triangular-prism graph contains loops and odd cycles.  It is
small enough to leave every physical index open and contract its *complete*
wavefunction with NumPy, without an approximate boundary or sampler.
"""

from itertools import product

import numpy as np

from peps_vmc.symmetry import local_intertwiner_basis, reduced_tensor, virtual_bond_metric


EDGES = ((0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3), (1, 4), (2, 5))


def closed_prism(tensors, *, reverse_edge=None):
    """Explicitly contract oriented singlet bonds, keeping physical site zero first."""
    operands = []
    for site, tensor in enumerate(tensors):
        indices = [site]
        for edge, (first, second) in enumerate(EDGES):
            if site == first:
                indices.append(6 + 2 * edge)
            elif site == second:
                indices.append(6 + 2 * edge + 1)
        operands.extend((tensor, indices))
    bond = virtual_bond_metric()
    for edge in range(len(EDGES)):
        operands.extend((bond.T if edge == reverse_edge else bond, [6 + 2 * edge, 7 + 2 * edge]))
    operands.append(list(range(6)))
    return np.einsum(*operands, optimize="greedy").reshape(-1)


def pauli_total_spin():
    """Dense physical generators in bit-zero-down / site-zero-MSB order."""
    local = (
        np.array([[0, 0.5], [0.5, 0]], dtype=complex),
        np.array([[0, 0.5j], [-0.5j, 0]], dtype=complex),
        np.diag([-0.5, 0.5]).astype(complex),
    )
    total = []
    for generator in local:
        matrix = np.zeros((64, 64), dtype=complex)
        for site in range(6):
            term = np.ones((1, 1), dtype=complex)
            for other in range(6):
                term = np.kron(term, generator if other == site else np.eye(2))
            matrix += term
        total.append(matrix)
    return total


def test_arbitrary_complex_closed_loop_peps_is_exact_physical_singlet():
    rng = np.random.default_rng(2026103001)
    basis = local_intertwiner_basis(3)
    tensors = [
        reduced_tensor(rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis)), basis)
        for _ in range(6)
    ]
    psi = closed_prism(tensors)
    assert np.linalg.norm(psi) > 1e-8
    psi /= np.linalg.norm(psi)
    generators = pauli_total_spin()
    for generator in generators:
        np.testing.assert_allclose(generator @ psi, 0, atol=3e-14, rtol=0)
    spin_squared = sum(generator @ generator for generator in generators)
    np.testing.assert_allclose(spin_squared @ psi, 0, atol=4e-14, rtol=0)
    for bits, amplitude in zip(product((0, 1), repeat=6), psi):
        if sum(bits) != 3:
            assert abs(amplitude) < 3e-14


def test_mixed_virtual_sectors_require_edge_orientation_or_compensating_parity():
    rng = np.random.default_rng(2026103002)
    basis = local_intertwiner_basis(3)
    tensors = [reduced_tensor(rng.normal(size=len(basis)), basis) for _ in range(6)]
    original = closed_prism(tensors)
    edge = 0
    reversed_only = closed_prism(tensors, reverse_edge=edge)
    # Changing a bond orientation is not one harmless global sign when both
    # q=0 and q=1 travel across the edge.
    overlap = np.vdot(original, reversed_only)
    residual = reversed_only - original * overlap / np.vdot(original, original)
    assert np.linalg.norm(residual) / np.linalg.norm(original) > 1e-3
    first = EDGES[edge][0]
    incident = [number for number, endpoints in enumerate(EDGES) if first in endpoints]
    leg = 1 + incident.index(edge)
    parity_shape = [1] * tensors[first].ndim
    parity_shape[leg] = 3
    compensated = list(tensors)
    compensated[first] = tensors[first] * np.array([1.0, -1.0, -1.0]).reshape(parity_shape)
    np.testing.assert_allclose(closed_prism(compensated, reverse_edge=edge), original, atol=2e-14, rtol=2e-13)
