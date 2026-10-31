import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
)
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.su2.tree_rotations import recoupling_matrix_sparse
from su2_nqs.vmc.many_cell_tree import (
    apply_many_cell_tree_connectivity,
    build_many_cell_tree_connectivity,
    compile_many_cell_device_hamiltonian,
    evaluate_external_carrier_additive_local_energy,
    evaluate_many_cell_device_local_energy,
    many_cell_device_hpsi,
    evaluate_many_cell_tree_local_energy,
    sample_many_cell_additive_proposal,
)


def test_many_cell_matrix_free_ratios_match_independent_dense_reference():
    layout = rectangular_four_spin_layout(2, 2, triangular_coordinates=False)
    config = ManyCellSchurViTConfig(
        layout, embedding_dim=8, head_count=2, depth=1
    )
    params = init_many_cell_schur_vit(jax.random.PRNGKey(13), config)
    states = layout.tree.enumerate_states(0)
    labels = np.asarray([state.labels for state in states], dtype=np.int32)
    bonds = ((0, 1, 0.7), (1, 3, -0.2), (3, 2, 1.1), (2, 0, 0.4))
    workload = build_many_cell_tree_connectivity(labels, config, bonds)
    result = evaluate_many_cell_tree_local_energy(params, workload, config)

    left = FusionTree.left_associated(4)
    recoupling, _ = recoupling_matrix_sparse(left, layout.tree, 0)
    left_h = heisenberg_graph_fusion_matrix(4, bonds, target_q=0)
    dense = np.asarray((recoupling @ left_h @ recoupling.T).toarray())
    amplitudes = np.exp(
        np.asarray(many_cell_schur_log_amplitudes(params, labels, config))
    )
    expected = dense @ amplitudes / amplitudes
    np.testing.assert_allclose(result.local_energies, expected, atol=2.0e-12)
    np.testing.assert_allclose(
        apply_many_cell_tree_connectivity(workload, amplitudes),
        dense @ amplitudes,
        atol=2.0e-12,
    )

    lookup = {tuple(row): amplitudes[index] for index, row in enumerate(labels)}
    connected = np.zeros(workload.mask.shape, dtype=np.complex128)
    for row in range(len(labels)):
        for slot in range(int(workload.widths[row])):
            connected[row, slot] = lookup[
                tuple(workload.connected_labels[row, slot])
            ]
    additive = evaluate_external_carrier_additive_local_energy(
        amplitudes,
        connected,
        params,
        0.0,
        workload,
        config,
    )
    np.testing.assert_allclose(additive.local_energies, expected, atol=2.0e-12)


def test_additive_state_uses_a_normalized_two_arm_proposal():
    layout = rectangular_four_spin_layout(2, 2)
    config = ManyCellSchurViTConfig(
        layout, embedding_dim=8, head_count=2, depth=1
    )
    carrier = init_many_cell_schur_vit(jax.random.PRNGKey(20), config)
    residual = init_many_cell_schur_vit(jax.random.PRNGKey(21), config)
    batch = sample_many_cell_additive_proposal(
        carrier,
        residual,
        0.2,
        jax.random.PRNGKey(22),
        64,
        config,
        carrier_probability=0.4,
    )
    assert batch.labels.shape == (64, 3)
    assert np.all(np.isfinite(np.asarray(batch.log_proposal_probability)))
    assert np.all(np.isfinite(np.asarray(batch.log_importance_weight)))
    support = {
        state.labels for state in layout.tree.enumerate_states(config.target_q)
    }
    assert all(tuple(row) in support for row in np.asarray(batch.labels))


def test_device_full_h_local_energy_matches_host_connectivity_reference():
    layout = rectangular_four_spin_layout(2, 2, triangular_coordinates=True)
    config = ManyCellSchurViTConfig(
        layout, embedding_dim=8, head_count=2, depth=1
    )
    params = init_many_cell_schur_vit(jax.random.PRNGKey(31), config)
    labels = np.asarray(
        [state.labels for state in layout.tree.enumerate_states(0)],
        dtype=np.int32,
    )
    bonds = (
        (0, 1, 1.0),
        (1, 3, 0.7),
        (3, 2, -0.2),
        (2, 0, 0.5),
        (0, 3, 0.125),
    )
    host = build_many_cell_tree_connectivity(labels, config, bonds)
    expected = evaluate_many_cell_tree_local_energy(params, host, config)
    compiled = compile_many_cell_device_hamiltonian(
        config, bonds, candidate_capacity=8
    )
    actual = evaluate_many_cell_device_local_energy(compiled, params, labels)
    np.testing.assert_allclose(
        actual.local_energies, expected.local_energies, atol=3.0e-12
    )
    assert not np.any(actual.overflow)
    assert actual.full_h_term_calls == len(labels) * len(bonds)
    assert np.all(actual.unique_widths <= compiled.connected_capacity)
    centers, hpsi, overflow, unique = many_cell_device_hpsi(
        compiled, params, jnp.asarray(labels)
    )
    np.testing.assert_allclose(
        np.asarray(hpsi / centers), actual.local_energies, atol=3.0e-12
    )
    assert not np.any(np.asarray(overflow))
    np.testing.assert_array_equal(np.asarray(unique), actual.unique_widths)


def test_device_full_h_matches_nontrivial_n16_columns_without_host_rebuild():
    layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=True)
    config = ManyCellSchurViTConfig(
        layout, embedding_dim=8, head_count=2, depth=1
    )
    params = init_many_cell_schur_vit(jax.random.PRNGKey(41), config)
    from su2_nqs.models.many_cell_schur_vit import sample_many_cell_schur_vit

    labels = np.asarray(
        sample_many_cell_schur_vit(
            params, jax.random.PRNGKey(42), 2, config
        )
    )
    bonds = tuple(
        (site, (site + 1) % 16, 1.0) for site in range(16)
    ) + ((0, 10, 0.125), (5, 15, -0.2))
    host = build_many_cell_tree_connectivity(labels, config, bonds)
    expected = evaluate_many_cell_tree_local_energy(params, host, config)
    compiled = compile_many_cell_device_hamiltonian(
        config, bonds, candidate_capacity=32
    )
    actual = evaluate_many_cell_device_local_energy(compiled, params, labels)
    np.testing.assert_allclose(
        actual.local_energies, expected.local_energies, atol=8.0e-12
    )
    assert not np.any(actual.overflow)
    np.testing.assert_array_equal(actual.unique_widths, host.widths)


def test_tiled_device_amplitudes_preserve_exact_n16_action():
    layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=True)
    config = ManyCellSchurViTConfig(
        layout, embedding_dim=8, head_count=2, depth=1
    )
    params = init_many_cell_schur_vit(jax.random.PRNGKey(43), config)
    from su2_nqs.models.many_cell_schur_vit import sample_many_cell_schur_vit

    labels = np.asarray(
        sample_many_cell_schur_vit(
            params, jax.random.PRNGKey(44), 2, config
        )
    )
    bonds = tuple((site, (site + 1) % 16, 1.0) for site in range(16))
    untiled = compile_many_cell_device_hamiltonian(
        config, bonds, candidate_capacity=32, connected_capacity=64
    )
    tiled = compile_many_cell_device_hamiltonian(
        config,
        bonds,
        candidate_capacity=32,
        connected_capacity=64,
        amplitude_tile_size=8,
    )
    expected = evaluate_many_cell_device_local_energy(untiled, params, labels)
    actual = evaluate_many_cell_device_local_energy(tiled, params, labels)
    np.testing.assert_allclose(
        actual.local_energies, expected.local_energies, atol=8.0e-12
    )
    np.testing.assert_array_equal(actual.unique_widths, expected.unique_widths)
    assert tiled.amplitude_tile_size == 8
    assert not np.any(actual.overflow)


def test_device_amplitude_tile_must_divide_connected_capacity():
    layout = rectangular_four_spin_layout(2, 2)
    config = ManyCellSchurViTConfig(
        layout, embedding_dim=8, head_count=2, depth=1
    )
    with np.testing.assert_raises_regex(ValueError, "must divide"):
        compile_many_cell_device_hamiltonian(
            config,
            ((0, 1, 1.0),),
            connected_capacity=16,
            amplitude_tile_size=3,
        )


def test_active_tiles_preserve_complex_action_jvp_and_reverse_gradient():
    layout=rectangular_four_spin_layout(2,2,triangular_coordinates=True)
    config=ManyCellSchurViTConfig(layout,embedding_dim=8,head_count=2,depth=1,
        direct_phase_initialization_scale=.7)
    params=init_many_cell_schur_vit(jax.random.PRNGKey(871),config)
    labels=jnp.asarray([s.labels for s in layout.tree.enumerate_states(0)])
    bonds=((0,1,1.),(1,3,.4),(0,3,.125))
    padded=compile_many_cell_device_hamiltonian(config,bonds,
        connected_capacity=64,amplitude_tile_size=4)
    active=compile_many_cell_device_hamiltonian(config,bonds,
        connected_capacity=64,amplitude_tile_size=4,skip_padded_amplitudes=True)
    reference=evaluate_many_cell_device_local_energy(padded,params,labels)
    actual=evaluate_many_cell_device_local_energy(active,params,labels)
    np.testing.assert_allclose(actual.local_energies,reference.local_energies,atol=3e-12,rtol=0)
    assert actual.evaluated_connected_amplitudes<reference.evaluated_connected_amplitudes
    np.testing.assert_array_equal(actual.unique_widths,reference.unique_widths)
    tangent=jax.tree_util.tree_map(lambda x:jnp.ones_like(x)*.001,params)
    def h(kernel,p): return many_cell_device_hpsi(kernel,p,labels)[1]
    for derivative in ('jvp','reverse'):
        if derivative=='jvp':
            expected=jax.jvp(lambda p:h(padded,p),(params,),(tangent,))[1]
            observed=jax.jvp(lambda p:h(active,p),(params,),(tangent,))[1]
            np.testing.assert_allclose(observed,expected,atol=3e-12,rtol=0)
        else:
            loss=lambda kernel,p:jnp.sum(abs(h(kernel,p))**2)
            expected=jax.grad(lambda p:loss(padded,p))(params)
            observed=jax.grad(lambda p:loss(active,p))(params)
            for a,b in zip(jax.tree_util.tree_leaves(observed),jax.tree_util.tree_leaves(expected)):
                np.testing.assert_allclose(a,b,atol=3e-11,rtol=0)


def test_active_tiles_do_not_hide_connectivity_overflow():
    layout=rectangular_four_spin_layout(4,4)
    config=ManyCellSchurViTConfig(layout,embedding_dim=4,head_count=1,depth=1)
    params=init_many_cell_schur_vit(jax.random.PRNGKey(872),config)
    labels=np.array([layout.tree.enumerate_states(0)[-1].labels])
    kernel=compile_many_cell_device_hamiltonian(config,((0,10,1.),),
        connected_capacity=1,amplitude_tile_size=1,skip_padded_amplitudes=True)
    with np.testing.assert_raises_regex(RuntimeError,'overflow'):
        evaluate_many_cell_device_local_energy(kernel,params,labels)


def test_bucketed_neural_batch_matches_complex_action_and_fixed_bucket_gradient():
    layout=rectangular_four_spin_layout(2,2)
    config=ManyCellSchurViTConfig(layout,embedding_dim=8,head_count=2,depth=1,
        direct_phase_initialization_scale=.7)
    params=init_many_cell_schur_vit(jax.random.PRNGKey(873),config)
    labels=jnp.asarray([s.labels for s in layout.tree.enumerate_states(0)])
    bonds=((0,1,1.),(1,3,.4),(0,3,.125))
    reference=compile_many_cell_device_hamiltonian(config,bonds,connected_capacity=64,amplitude_tile_size=4)
    bucketed=compile_many_cell_device_hamiltonian(config,bonds,connected_capacity=64,
        amplitude_tile_size=4,bucket_padded_amplitudes=True)
    expected=evaluate_many_cell_device_local_energy(reference,params,labels)
    actual=evaluate_many_cell_device_local_energy(bucketed,params,labels)
    np.testing.assert_allclose(actual.local_energies,expected.local_energies,atol=3e-12,rtol=0)
    assert actual.evaluated_connected_amplitudes==8
    conn,coef,mask,overflow,width=bucketed._connectivity_kernel(labels)
    assert not np.any(overflow)
    def loss(p,size):return jnp.sum(abs(bucketed._ratio_kernel(p,labels,conn[:,:size],coef[:,:size],mask[:,:size]))**2)
    small=jax.grad(lambda p:loss(p,4))(params);full=jax.grad(lambda p:loss(p,64))(params)
    for a,b in zip(jax.tree_util.tree_leaves(small),jax.tree_util.tree_leaves(full)):
        np.testing.assert_allclose(a,b,atol=4e-11,rtol=0)


def test_bucketed_evaluator_rejects_overflow_and_conflicting_execution_modes():
    layout=rectangular_four_spin_layout(4,4)
    config=ManyCellSchurViTConfig(layout,embedding_dim=4,head_count=1,depth=1)
    params=init_many_cell_schur_vit(jax.random.PRNGKey(874),config)
    labels=np.array([layout.tree.enumerate_states(0)[-1].labels])
    kernel=compile_many_cell_device_hamiltonian(config,((0,10,1.),),connected_capacity=1,
        amplitude_tile_size=1,bucket_padded_amplitudes=True)
    with np.testing.assert_raises_regex(RuntimeError,'overflow'):
        evaluate_many_cell_device_local_energy(kernel,params,labels)
    with np.testing.assert_raises(ValueError):
        compile_many_cell_device_hamiltonian(config,((0,1,1.),),skip_padded_amplitudes=True,
            bucket_padded_amplitudes=True)


def test_exact_capacity_buckets_reduce_padding_and_preserve_n16_action():
    layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=True)
    config = ManyCellSchurViTConfig(
        layout, embedding_dim=8, head_count=2, depth=1
    )
    params = init_many_cell_schur_vit(jax.random.PRNGKey(51), config)
    from su2_nqs.models.many_cell_schur_vit import sample_many_cell_schur_vit

    labels = np.asarray(
        sample_many_cell_schur_vit(
            params, jax.random.PRNGKey(52), 1, config
        )
    )
    bonds = tuple(
        (site, (site + 1) % 16, 1.0) for site in range(16)
    ) + ((0, 10, 0.125), (5, 15, -0.2))
    host = build_many_cell_tree_connectivity(labels, config, bonds)
    expected = evaluate_many_cell_tree_local_energy(params, host, config)
    compiled = compile_many_cell_device_hamiltonian(config, bonds)
    actual = evaluate_many_cell_device_local_energy(compiled, params, labels)
    np.testing.assert_allclose(
        actual.local_energies, expected.local_energies, atol=8.0e-12
    )
    largest = max(
        int(np.max(compiled.plan.left_candidate_capacity_bounds)),
        int(np.max(compiled.plan.right_candidate_capacity_bounds)),
    )
    assert actual.raw_transition_slots < len(bonds) * largest**2
    assert actual.compact_transition_slots <= actual.raw_transition_slots
    assert len(compiled.bucket_capacities) > 1
