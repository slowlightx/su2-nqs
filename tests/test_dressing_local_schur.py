import numpy as np

from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.benchmarks.lattice_observables import (
    scalar_chirality_computational_matrix,
)
from su2_nqs.dressing import (
    apply_local_gate,
    apply_local_gate_to_columns,
    four_spin_schur_basis,
    multiplicity_block_operator,
    plaquette_generators,
    plaquette_multiplicity_blocks,
    plaquette_unitary,
    recouple_multiplicity_blocks,
)
from su2_nqs.su2.fusion_tree import FusionTree


def _maximum(value):
    return float(np.max(np.abs(value)))


def test_four_spin_schur_basis_is_complete_and_has_2_3_1_multiplicities():
    basis = four_spin_schur_basis()
    assert _maximum(basis.columns.conj().T @ basis.columns - np.eye(16)) < 1e-13
    multiplicities = {
        q: len({label.multiplicity for label in basis.labels if label.q == q})
        for q in (0, 2, 4)
    }
    assert multiplicities == {0: 2, 2: 3, 4: 1}
    assert [sum(label.q == q for label in basis.labels) for q in (0, 2, 4)] == [2, 9, 5]


def test_plaquette_gate_is_unitary_and_commutes_with_spin_and_sz():
    angles = (0.37, -0.29, 0.41, 0.17)
    gate = plaquette_unitary(angles)
    spin_squared, _ = spin_squared_matrix(4)
    diagonal_sz = np.asarray(
        [sum(0.5 if (state >> (3 - site)) & 1 else -0.5 for site in range(4))
         for state in range(16)],
        dtype=np.float64,
    )
    sz = np.diag(diagonal_sz)
    assert _maximum(gate.conj().T @ gate - np.eye(16)) < 1e-13
    assert _maximum(gate @ spin_squared - spin_squared @ gate) < 1e-13
    assert _maximum(gate @ sz - sz @ gate) < 1e-13
    assert _maximum(gate.imag) < 1e-14


def test_gate_and_generators_commute_with_all_collective_spin_components():
    identity = np.eye(2)
    pauli = (
        np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.complex128),
        np.asarray(((0.0, -1.0j), (1.0j, 0.0)), dtype=np.complex128),
        np.asarray(((1.0, 0.0), (0.0, -1.0)), dtype=np.complex128),
    )
    collective = []
    for local in pauli:
        total = np.zeros((16, 16), dtype=np.complex128)
        for site in range(4):
            factors = [identity] * 4
            factors[site] = 0.5 * local
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            total += term
        collective.append(total)

    gate = plaquette_unitary((0.19, -0.31, 0.27, 0.43))
    generators = plaquette_generators()
    for spin in collective:
        assert _maximum(gate @ spin - spin @ gate) < 1e-13
        for generator in generators:
            assert _maximum(generator @ spin - spin @ generator) < 1e-13
    for generator in generators:
        assert _maximum(generator.conj().T + generator) < 1e-13

    epsilon = 1.0e-7
    for index, generator in enumerate(generators):
        angles = np.zeros(4)
        angles[index] = epsilon
        finite_difference = (
            plaquette_unitary(angles) - plaquette_unitary(-angles)
        ) / (2.0 * epsilon)
        assert _maximum(finite_difference - generator) < 2e-9


def test_continuous_gate_algebra_equals_four_spin_chirality_span():
    generators = np.column_stack(
        [generator.reshape(-1) for generator in plaquette_generators()]
    )
    triples = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    chiralities = np.column_stack(
        [
            (-1.0j * scalar_chirality_computational_matrix(4, *triple)).reshape(-1)
            for triple in triples
        ]
    )
    assert np.linalg.matrix_rank(generators, tol=1.0e-12) == 4
    assert np.linalg.matrix_rank(chiralities, tol=1.0e-12) == 4
    gate_basis, _ = np.linalg.qr(generators)
    chirality_basis, _ = np.linalg.qr(chiralities)
    gate_residual = generators - chirality_basis @ (
        chirality_basis.conj().T @ generators
    )
    chirality_residual = chiralities - gate_basis @ (
        gate_basis.conj().T @ chiralities
    )
    assert np.linalg.norm(gate_residual) / np.linalg.norm(generators) < 1e-12
    assert np.linalg.norm(chirality_residual) / np.linalg.norm(chiralities) < 1e-12


def test_computational_gate_agrees_with_reduced_block_action():
    basis = four_spin_schur_basis()
    blocks = plaquette_multiplicity_blocks((0.13, 0.21, -0.34, 0.55))
    reduced = multiplicity_block_operator(basis, blocks)
    gate = plaquette_unitary((0.13, 0.21, -0.34, 0.55), basis=basis)
    assert _maximum(basis.columns.conj().T @ gate @ basis.columns - reduced) < 1e-13


def test_same_physical_gate_is_covariant_under_fusion_tree_change():
    source = four_spin_schur_basis(FusionTree.left_associated(4))
    target = four_spin_schur_basis(FusionTree.from_nested(((0, 1), (2, 3))))
    blocks = plaquette_multiplicity_blocks((0.71, -0.11, 0.23, -0.47))
    target_blocks = recouple_multiplicity_blocks(
        blocks, source=source, target=target
    )
    source_gate = (
        source.columns
        @ multiplicity_block_operator(source, blocks)
        @ source.columns.conj().T
    )
    target_gate = (
        target.columns
        @ multiplicity_block_operator(target, target_blocks)
        @ target.columns.conj().T
    )
    assert _maximum(source_gate - target_gate) < 1e-13


def test_local_gate_action_respects_site_zero_most_significant_bit():
    # X on the first local tensor factor maps |00000> to |10000> when the
    # ordered gate sites begin with physical site zero.
    x = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    gate = np.kron(np.kron(np.kron(x, np.eye(2)), np.eye(2)), np.eye(2))
    state = np.zeros(32, dtype=np.complex128)
    state[0] = 1.0
    transformed = apply_local_gate(state, gate, (0, 1, 2, 3))
    expected = np.zeros_like(state)
    expected[16] = 1.0
    assert np.array_equal(transformed, expected)


def test_batched_local_gate_action_matches_individual_columns():
    rng = np.random.default_rng(91027)
    vectors = rng.normal(size=(64, 5)) + 1.0j * rng.normal(size=(64, 5))
    gate = plaquette_unitary((0.17, -0.25, 0.33, 0.09))
    sites = (0, 2, 4, 5)
    expected = np.column_stack(
        [apply_local_gate(vectors[:, index], gate, sites) for index in range(5)]
    )
    actual = apply_local_gate_to_columns(vectors, gate, sites)
    np.testing.assert_allclose(actual, expected, atol=2.0e-14)
