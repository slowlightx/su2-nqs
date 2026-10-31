"""Independent physical-space checks for finite Schur-cutset attention.

Dense operators here are test oracles only (N=8); no Hamiltonian or exact
ground-state calculation is needed.  The production model must contract its
fixed Clebsch tensors, not call these dense projectors.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.su2.local_schur import cell_schur_spec, cell_schur_tensor
from su2_nqs.lattices.spatial_symmetry import rectangular_gamma_a1_group
from su2_nqs.models.cell_multiplicity_clebsch_tree import (
    SpinCellMultiplicityConfig, init_spin_cell_multiplicity_tree,
    spin_cell_multiplicity_tree_raw_amplitude,
)
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector, spatial_projected_amplitude,
)
from su2_nqs.models.schur_cutset_attention import (
    CutsetAttentionConfig, init_cutset_attention, conditional_coefficients,
    make_branch_bank, make_free_parameters, make_free_branch_bank,
    branch_raw_amplitude, cutset_raw_amplitude, make_centered_additive, spatial_features,
)

jax.config.update('jax_enable_x64', True)


def _states(n_sites):
    labels = np.arange(1 << n_sites, dtype=np.int64)
    return ((labels[:, None] >> np.arange(n_sites - 1, -1, -1)) & 1).astype(np.int32)


@lru_cache(maxsize=2)
def _collective_spin(n_sites):
    """Independent Pauli construction, ordered |down>, |up> at each site."""
    local = (
        np.array([[0., .5], [.5, 0.]], dtype=np.complex128),
        np.array([[0., .5j], [-.5j, 0.]], dtype=np.complex128),
        np.diag([-.5, .5]).astype(np.complex128),
    )
    answer = []
    for spin in local:
        total = np.zeros((1 << n_sites, 1 << n_sites), dtype=np.complex128)
        for site in range(n_sites):
            product = np.ones((1, 1), dtype=np.complex128)
            for index in range(n_sites):
                product = np.kron(product, spin if index == site else np.eye(2))
            total += product
        answer.append(total)
    return tuple(answer)


def _local_projectors():
    """Project onto a FULL magnetic multiplet, independently of leaf masking."""
    spec = cell_schur_spec('square_plaquette')
    schur = cell_schur_tensor('square_plaquette')
    labels, projectors = [], []
    for q in spec.q_values:
        for a in range(spec.multiplicity(q)):
            vectors = np.column_stack([
                schur[:, q, m2 + spec.cell_size, a]
                for m2 in range(-q, q + 1, 2)
            ])
            labels.append((q, a))
            projectors.append(vectors @ vectors.conj().T)
    return tuple(labels), np.asarray(projectors)


def _apply_local(matrix, vector, sites, n_sites):
    """Dense local oracle with explicit physical-axis movement; MSB site zero."""
    order = tuple(sites) + tuple(site for site in range(n_sites) if site not in sites)
    tensor = np.asarray(vector).reshape((2,) * n_sites).transpose(order)
    transformed = matrix @ tensor.reshape(1 << len(sites), -1)
    inverse = np.argsort(order)
    return transformed.reshape((2,) * n_sites).transpose(inverse).reshape(-1)


def test_six_local_labels_are_complete_orthogonal_su2_scalar_projectors():
    labels, projectors = _local_projectors()
    assert labels == ((0, 0), (0, 1), (2, 0), (2, 1), (2, 2), (4, 0))
    np.testing.assert_allclose(projectors.sum(axis=0), np.eye(16), atol=2e-14)
    for left, projector in enumerate(projectors):
        np.testing.assert_allclose(projector, projector.conj().T, atol=1e-15)
        assert np.trace(projector).real == pytest.approx(labels[left][0] + 1, abs=2e-14)
        for right, other in enumerate(projectors):
            np.testing.assert_allclose(projector @ other,
                                       projector if left == right else 0., atol=2e-14)
        for spin in _collective_spin(4):
            np.testing.assert_allclose(projector @ spin, spin @ projector, atol=2e-14)


def _configuration(n_cells=2):
    if n_cells == 2:
        cells = ((0, 1, 4, 5), (2, 3, 6, 7))
        coordinates = ((0., 0.), (0., 2.))
    else:
        assert n_cells == 4
        cells = ((0, 1, 4, 5), (2, 3, 6, 7),
                 (8, 9, 12, 13), (10, 11, 14, 15))
        coordinates = ((0., 0.), (0., 2.), (2., 0.), (2., 2.))
    tree = SpinCellMultiplicityConfig('square_plaquette', cells,
                                      feature_dim=2, initialization_scale=.4)
    return CutsetAttentionConfig(tree=tree, cell_coordinates=coordinates,
                                 cutset_cell=0, embedding_dim=4, rank=2)


def _parameters(config, seed=8431):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    base = {part: init_spin_cell_multiplicity_tree(key, config.tree)
            for part, key in zip(('real', 'imag'), keys[:2])}
    return init_cutset_attention(keys[2], base, config), base


def _select(bank, branch):
    return jax.tree_util.tree_map(lambda value: value[branch], bank)


def _projected(bank, bits, config):
    projector = PhysicalSpatialProjector(rectangular_gamma_a1_group(2, 4))
    return spatial_projected_amplitude(
        bank, bits, config, projector,
        amplitude_fn=lambda p, x, cfg: cutset_raw_amplitude(p, x, cfg),
    )


def _random_tangent(params, seed):
    leaves, structure = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    values = [jax.random.normal(key, leaf.shape, dtype=leaf.dtype)
              for key, leaf in zip(keys, leaves)]
    norm = jnp.sqrt(sum(jnp.vdot(value, value).real for value in values))
    return jax.tree_util.tree_unflatten(structure, [value / norm for value in values])


def test_branch_contraction_matches_independent_dense_full_multiplet_projection():
    config = _configuration()
    params, _ = _parameters(config)
    bank = make_branch_bank(params, config)
    states = _states(8)
    labels, projectors = _local_projectors()
    branch_vectors = []
    projection_errors, spin_errors = [], []
    for branch_index, (label, projector) in enumerate(zip(labels, projectors)):
        branch = _select(bank, branch_index)
        unmasked = np.asarray(jax.jit(jax.vmap(
            lambda bits: branch_raw_amplitude(branch, bits, config)
        ))(jnp.asarray(states)))
        actual = np.asarray(jax.jit(jax.vmap(
            lambda bits: branch_raw_amplitude(branch, bits, config, label=label)
        ))(jnp.asarray(states)))
        expected = _apply_local(projector, unmasked, config.tree.cells[0], 8)
        assert np.linalg.norm(actual) > 1e-12
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=3e-12)
        projection_errors.append(float(np.max(abs(actual - expected))))
        for spin in _collective_spin(8):
            error = float(np.linalg.norm(spin @ actual) / np.linalg.norm(actual))
            assert error < 2e-12
            spin_errors.append(error)
        branch_vectors.append(actual)
    coherent = np.asarray(jax.jit(jax.vmap(
        lambda bits: cutset_raw_amplitude(bank, bits, config)
    ))(jnp.asarray(states)))
    np.testing.assert_allclose(coherent, np.sum(branch_vectors, axis=0),
                               atol=2e-14, rtol=3e-12)
    print({'dense_projection_max': max(projection_errors),
           'branch_collective_spin_relative_max': max(spin_errors)})


def test_complete_spatial_projection_is_coherent_and_matches_exhaustive_pullbacks():
    config = _configuration()
    params, _ = _parameters(config, seed=8461)
    bank = make_branch_bank(params, config)
    states = _states(8)
    raw = np.asarray(jax.jit(jax.vmap(
        lambda bits: cutset_raw_amplitude(bank, bits, config)
    ))(jnp.asarray(states)))
    groups = np.asarray(rectangular_gamma_a1_group(2, 4), dtype=np.int32)
    indices = np.sum(states[:, groups] * (1 << np.arange(7, -1, -1)), axis=-1)
    expected = np.mean(raw[indices], axis=1)
    actual = np.asarray(jax.jit(jax.vmap(
        lambda bits: _projected(bank, bits, config)
    ))(jnp.asarray(states)))
    assert np.linalg.norm(actual) > 1e-12
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=3e-12)
    np.testing.assert_allclose(actual[indices], np.broadcast_to(actual[:, None], indices.shape),
                               atol=2e-14, rtol=3e-12)
    for spin in _collective_spin(8):
        assert np.linalg.norm(spin @ actual) / np.linalg.norm(actual) < 2e-12
    print({'complete_A1_max': float(np.max(abs(actual - expected)))})


def test_free_coefficients_nest_attention_and_all_branches_have_real_parameters():
    config = _configuration(4)
    params, _ = _parameters(config)
    for scrambled in (False, True):
        bank = make_branch_bank(params, config, scrambled=scrambled)
        free = make_free_parameters(params, config, scrambled=scrambled)
        other = make_free_branch_bank(free, config)
        left, structure = jax.tree_util.tree_flatten(bank)
        right, other_structure = jax.tree_util.tree_flatten(other)
        assert structure == other_structure
        for first, second in zip(left, right):
            assert first.shape[0] == 6
            assert first.dtype == jnp.float64
            np.testing.assert_allclose(first, second, atol=2e-15, rtol=2e-14)


def test_schur_label_changes_a_remote_kernel_not_only_a_scalar_branch_weight():
    config = _configuration(4)
    params, _ = _parameters(config)
    coefficients = np.asarray(conditional_coefficients(params['attention'], config))
    assert coefficients.shape == (6, 3, 2)
    bank = make_branch_bank(params, config)
    # Postorder node 1 joins cells 2 and 3: neither contains the cutset cell 0.
    remote = bank['real']['merges'][1]
    channels = jax.tree_util.tree_leaves(remote)
    flattened = np.concatenate([np.asarray(value).reshape(6, -1) for value in channels], axis=1)
    singular = np.linalg.svd(flattened, compute_uv=False)
    assert singular[1] > 1e-8 * singular[0]
    assert np.max(np.ptp(coefficients[:, 1, :], axis=0)) > 1e-8
    print({'remote_kernel_second_over_first_singular': float(singular[1] / singular[0]),
           'remote_coefficient_label_span': float(np.max(np.ptp(coefficients[:, 1, :], axis=0)))})


def test_spatial_scramble_is_a_nontrivial_edge_control_only_when_edges_allow_it():
    two_cells = _configuration()
    np.testing.assert_array_equal(spatial_features(two_cells),
                                  spatial_features(two_cells, scrambled=True))
    config = _configuration(4)
    params, _ = _parameters(config)
    original, scrambled = (spatial_features(config, scrambled=value) for value in (False, True))
    assert not np.array_equal(original, scrambled)
    upper = np.triu_indices(4, 1)
    np.testing.assert_array_equal(np.sort(original[upper], axis=0),
                                  np.sort(scrambled[upper], axis=0))
    np.testing.assert_array_equal(scrambled.transpose(1, 0, 2),
                                  scrambled * np.array([-1., -1., 1.]))
    first = np.asarray(conditional_coefficients(params['attention'], config))
    second = np.asarray(conditional_coefficients(params['attention'], config, scrambled=True))
    assert np.max(abs(first - second)) > 1e-8


def test_centered_carrier_replay_has_nonzero_attention_tangent_and_exact_symmetry():
    config = _configuration()
    params, _ = _parameters(config, seed=8442)
    reference = make_branch_bank(params, config)
    carrier_params, _ = _parameters(config, seed=8443)
    carrier_bank = make_branch_bank(carrier_params, config)
    states = jnp.asarray(_states(8))
    carrier = lambda bits: _projected(carrier_bank, bits, config)
    residual = lambda bank, bits: _projected(bank, bits, config)
    additive = make_centered_additive(carrier, residual, reference, eta=1.)
    zero = make_centered_additive(carrier, residual, reference, eta=0.)

    def state(candidate):
        bank = make_branch_bank(candidate, config)
        return jax.vmap(lambda bits: additive(bank, bits))(states)

    def normalized(candidate):
        vector = state(candidate)
        return vector / jnp.linalg.norm(vector)

    carrier_vector = np.asarray(jax.jit(jax.vmap(carrier))(states))
    actual = np.asarray(jax.jit(state)(params))
    # The generated live bank is fused inside JIT while the frozen reference
    # was materialized outside; their algebraic equality allows roundoff only.
    np.testing.assert_allclose(actual, carrier_vector, atol=2e-15, rtol=0.)
    direction = jax.tree_util.tree_map(jnp.zeros_like, params)
    direction['attention'] = _random_tangent(params['attention'], seed=8444)
    value, tangent = jax.jit(lambda p, d: jax.jvp(normalized, (p,), (d,)))(params, direction)
    value, tangent = np.asarray(value), np.asarray(tangent)
    projective_tangent = tangent - value * np.vdot(value, tangent)
    assert np.linalg.norm(projective_tangent) > 1e-8
    for spin in _collective_spin(8):
        assert np.linalg.norm(spin @ tangent) / np.linalg.norm(tangent) < 3e-12
    groups = np.asarray(rectangular_gamma_a1_group(2, 4))
    indices = np.sum(np.asarray(states)[:, groups] * (1 << np.arange(7, -1, -1)), axis=-1)
    np.testing.assert_allclose(tangent[indices], np.broadcast_to(tangent[:, None], indices.shape),
                               atol=2e-13, rtol=3e-12)
    epsilon = 2e-6
    plus = jax.tree_util.tree_map(lambda p, d: p + epsilon * d, params, direction)
    minus = jax.tree_util.tree_map(lambda p, d: p - epsilon * d, params, direction)
    evaluate = jax.jit(normalized)
    finite_difference = (np.asarray(evaluate(plus)) - np.asarray(evaluate(minus))) / (2 * epsilon)
    np.testing.assert_allclose(tangent, finite_difference, atol=2e-9, rtol=5e-7)
    print({'attention_only_projective_tangent_norm': float(np.linalg.norm(projective_tangent)),
           'attention_directional_gradient_max': float(np.max(abs(tangent - finite_difference))),
           'centered_carrier_replay_max': float(np.max(abs(actual - carrier_vector)))})
    # eta=0 preserves the carrier but is unsuitable as a training initialization.
    def zero_state(candidate):
        bank = make_branch_bank(candidate, config)
        return jax.vmap(lambda bits: zero(bank, bits))(states)
    _, zero_tangent = jax.jvp(zero_state, (params,), (direction,))
    np.testing.assert_array_equal(zero_tangent, np.zeros_like(carrier_vector))


def test_config_rejects_noninteger_cutset_labels_and_spin_truncation():
    from dataclasses import replace
    config = _configuration()
    with pytest.raises(TypeError):
        replace(config, cutset_cell=.0)
    with pytest.raises(ValueError):
        replace(config, cutset_cell=2)
    with pytest.raises(ValueError):
        replace(config, tree=replace(config.tree, maximum_intermediate_q=4))
    with pytest.raises(ValueError):
        replace(config, tree=replace(config.tree, target_q=2))


@pytest.mark.parametrize('n_cells', [2, 4])
def test_zero_conditional_directions_exactly_reproduce_original_complex_decoder(n_cells):
    config = _configuration(n_cells)
    params, base = _parameters(config, seed=8451)
    params = {**params, 'directions': jax.tree_util.tree_map(jnp.zeros_like, params['directions'])}
    bank = make_branch_bank(params, config)
    n_sites = config.tree.n_sites
    if n_sites == 8:
        states = _states(n_sites)
    else:
        rng = np.random.default_rng(8452)
        states = np.asarray([rng.permutation([0] * 8 + [1] * 8) for _ in range(8)], dtype=np.int32)
    def old(bits):
        return (spin_cell_multiplicity_tree_raw_amplitude(base['real'], bits, config.tree)
                + 1j * spin_cell_multiplicity_tree_raw_amplitude(base['imag'], bits, config.tree))
    expected = np.asarray(jax.jit(jax.vmap(old))(jnp.asarray(states)))
    actual = np.asarray(jax.jit(jax.vmap(
        lambda bits: cutset_raw_amplitude(bank, bits, config)
    ))(jnp.asarray(states)))
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=3e-12)
    print({'nesting_N': n_sites, 'old_decoder_amplitude_max': float(np.max(abs(actual - expected)))})


def _dense_pauli_hamiltonian(n_sites, bonds):
    """Independent dense H and its physical diagonal H_z; no diagonalization."""
    pauli = (np.array([[0., .5], [.5, 0.]], dtype=np.complex128),
             np.array([[0., .5j], [-.5j, 0.]], dtype=np.complex128),
             np.diag([-.5, .5]).astype(np.complex128))
    hamiltonian = np.zeros((1 << n_sites, 1 << n_sites), dtype=np.complex128)
    # Operators on distinct sites are tensor products. Repeated and negative
    # bonds deliberately exercise accumulation and signs independently of the
    # production exchange-connection formula.
    for first, second, coupling in bonds:
        for spin in pauli:
            operator = np.ones((1, 1), dtype=np.complex128)
            for site in range(n_sites):
                operator = np.kron(operator, spin if site in (first, second) else np.eye(2))
            hamiltonian += coupling * operator
    return hamiltonian, np.diag(hamiltonian).real


def test_node_safe_H_action_matches_independent_dense_pauli_oracle():
    from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
    config = _configuration()
    params, _ = _parameters(config, seed=8455)
    bank = make_branch_bank(params, config)
    states = _states(8)
    amplitude = lambda candidate, bits: _projected(candidate, bits, config)
    vector = np.asarray(jax.jit(jax.vmap(amplitude, in_axes=(None, 0)))(bank, jnp.asarray(states)))
    bonds = ((0, 1, 1.), (1, 4, -.4), (3, 7, .125), (0, 1, .07))
    hamiltonian, _ = _dense_pauli_hamiltonian(8, bonds)
    selected = np.argsort(abs(vector))[-4:]
    psi, hpsi = make_amplitude_heisenberg_action(amplitude, bonds, n_sites=8)(bank, jnp.asarray(states[selected]))
    expected = (hamiltonian @ vector)[selected]
    np.testing.assert_allclose(psi, vector[selected], atol=2e-14, rtol=3e-12)
    np.testing.assert_allclose(hpsi, expected, atol=2e-14, rtol=3e-12)
    print({'dense_H_action_max': float(np.max(abs(np.asarray(hpsi) - expected)))})


def test_strict_singlet_diagonal_energy_and_attention_gradient_equal_full_H_not_variance():
    """SU(2) permits cheap diagonal energy training, NOT a variance shortcut.

    For normalized strict singlets, P0 H_z P0 = H P0 / 3.  Thus the
    population energy and every singlet tangent derivative agree.  H_z also
    creates a spin-two component, whose norm contributes only to its diagonal
    fluctuation; 9 Var(H_z) must not be reported as Var(H).
    """
    config = _configuration()
    params, _ = _parameters(config, seed=8471)
    states = jnp.asarray(_states(8))
    bonds = ((0, 1, 1.), (1, 4, -.4), (3, 7, .125), (0, 1, .07))
    hamiltonian, hz_diagonal = _dense_pauli_hamiltonian(8, bonds)
    h, hz = jnp.asarray(hamiltonian), jnp.asarray(hz_diagonal)

    def normalized(candidate):
        bank = make_branch_bank(candidate, config)
        vector = jax.vmap(lambda bits: _projected(bank, bits, config))(states)
        return vector / jnp.linalg.norm(vector)

    def energies(candidate):
        vector = normalized(candidate)
        return jnp.stack((jnp.vdot(vector, h @ vector).real,
                          3 * jnp.vdot(vector, hz * vector).real))

    direction = jax.tree_util.tree_map(jnp.zeros_like, params)
    direction['attention'] = _random_tangent(params['attention'], seed=8472)
    energy, derivative = jax.jit(lambda p, d: jax.jvp(energies, (p,), (d,)))(params, direction)
    energy, derivative = np.asarray(energy), np.asarray(derivative)
    np.testing.assert_allclose(energy[0], energy[1], atol=3e-14, rtol=0.)
    np.testing.assert_allclose(derivative[0], derivative[1], atol=3e-13, rtol=0.)
    assert abs(derivative[0]) > 1e-10
    vector = np.asarray(jax.jit(normalized)(params))
    h_vector, hz_vector = hamiltonian @ vector, hz_diagonal * vector
    full_variance = float(np.vdot(h_vector, h_vector).real - energy[0] ** 2)
    diagonal_variance = float(9 * np.vdot(hz_vector, hz_vector).real - energy[1] ** 2)
    spin_two = hz_vector - h_vector / 3
    extra = float(9 * np.vdot(spin_two, spin_two).real)
    assert extra > 1e-8  # This fixture has a genuinely nonzero rank-two response.
    np.testing.assert_allclose(diagonal_variance - full_variance, extra, atol=3e-14, rtol=0.)
    spin_squared_response = sum(spin @ (spin @ spin_two) for spin in _collective_spin(8))
    np.testing.assert_allclose(spin_squared_response, 6 * spin_two, atol=3e-14, rtol=0.)
    print({'singlet_diagonal_energy_error': float(abs(energy[0] - energy[1])),
           'singlet_diagonal_attention_gradient_error': float(abs(derivative[0] - derivative[1])),
           'full_quantum_variance': full_variance,
           'nine_times_Hz_variance': diagonal_variance,
           'spin_two_extra_variance': extra})
