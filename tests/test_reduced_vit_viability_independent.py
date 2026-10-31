"""Independent reduced-ViT viability oracles, without energy optimization.

These checks are deliberately independent of the new viability workflow.
They distinguish exact native singlet Born sampling from a coherently
projected lattice-sector state and check the two-sided energy derivative.
"""

from itertools import product

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin import _bonds, _groups
from su2_nqs.models.compact_tree_gru import split_tables
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
    many_cell_schur_spatial_distances,
    sample_many_cell_schur_vit,
)
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.fusion_tree import Leaf
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.models.tree_gru import tree_decisions
from su2_nqs.su2.tree_rotations import leaf_signature
from su2_nqs.vmc.many_cell_tree import (
    apply_many_cell_tree_connectivity, build_many_cell_tree_connectivity,
)


def _independent_cg_isometry(config, labels, physical_leaf_order=None):
    """Explicit fixed-CG product, independently ordered physical bits."""
    bits = np.asarray(list(product((0, 1), repeat=config.n)), dtype=np.int32)
    u = np.empty((len(bits), len(labels)), dtype=np.float64)
    order = (np.asarray(config.layout.site_order) if physical_leaf_order is None
             else np.asarray(physical_leaf_order))
    for column, row in enumerate(labels):
        assignments = dict(zip(config.layout.tree.internal_nodes, row, strict=True))
        for index, physical in enumerate(bits):
            model = physical[order]

            def visit(node):
                if isinstance(node, Leaf):
                    return 1, int(2*model[node.index]-1), 1.0
                ql, ml, al = visit(node.left)
                qr, mr, ar = visit(node.right)
                q = int(assignments[node])
                m = ml+mr
                if al == 0.0 or ar == 0.0 or abs(m) > q:
                    return q, m, 0.0
                return q, m, al*ar*clebsch_gordan(ql, ml, qr, mr, q, m)

            q, m, value = visit(config.layout.tree.root)
            u[index, column] = value if m == 0 else 0.0
    np.testing.assert_allclose(u.T@u, np.eye(len(labels)), atol=2e-14, rtol=0)
    all_pairs = [(first, second, 1.0) for first in range(config.n)
                 for second in range(first+1, config.n)]
    s2 = .75*config.n*np.eye(len(bits))+2*_physical_h(bits, all_pairs)
    np.testing.assert_allclose(s2@u, 0.0, atol=2e-14, rtol=0)
    return bits, u


def _small_problem():
    config = ManyCellSchurViTConfig(
        rectangular_four_spin_layout(2, 2), embedding_dim=8, head_count=2,
        depth=1, geometry="square", direct_phase_initialization_scale=1.0,
    )
    params = init_many_cell_schur_vit(jax.random.PRNGKey(202610091), config)
    labels = np.asarray([state.labels for state in config.layout.tree.enumerate_states(0)],
                        dtype=np.int32)
    bits, u = _independent_cg_isometry(config, labels)
    return config, params, labels, bits, u


def _physical_h(bits, bonds):
    """Explicit down/up computational-basis columns, H[new,old]."""
    lookup = {tuple(row): index for index, row in enumerate(bits)}
    h = np.zeros((len(bits), len(bits)), dtype=np.float64)
    for old, row in enumerate(bits):
        for first, second, coupling in bonds:
            same = row[first] == row[second]
            h[old, old] += coupling*(0.25 if same else -0.25)
            if not same:
                new = row.copy()
                new[first], new[second] = row[second], row[first]
                h[lookup[tuple(new)], old] += 0.5*coupling
    return h


def test_native_ar_probability_is_not_the_coherently_projected_a1_born_law():
    config, params, labels, bits, u = _small_problem()
    logs = np.asarray(many_cell_schur_log_amplitudes(params, labels, config))
    c = np.exp(logs)
    native = abs(c)**2
    np.testing.assert_allclose(native.sum(), 1.0, atol=3e-14, rtol=0)
    sampled = np.asarray(sample_many_cell_schur_vit(
        params, jax.random.PRNGKey(202610092), 4096, config))
    frequency = np.asarray([np.mean(np.all(sampled == row, axis=1)) for row in labels])
    assert np.max(abs(frequency-native)) < 6*np.sqrt(np.max(native*(1-native))/4096)

    # Physical group action uses only site permutation, independent of any
    # reduced-basis permutation or recoupling implementation.
    physical_lookup = {tuple(row): index for index, row in enumerate(bits)}
    projector = np.zeros((16, 16), dtype=np.float64)
    group = _groups("square", 2, 2)["gamma_a1"]
    for image in group:
        for old, row in enumerate(bits):
            projector[physical_lookup[tuple(row[np.asarray(image)])], old] += 1/len(group)
    reduced = u.T@projector@u
    np.testing.assert_allclose(reduced@reduced, reduced, atol=3e-14, rtol=0)
    projected = reduced@c
    projected /= np.linalg.norm(projected)
    probability = abs(projected)**2
    assert np.max(abs(probability-native)) > 1e-3
    np.testing.assert_allclose(projector@(u@projected), u@projected, atol=3e-14, rtol=0)
    # Therefore native iid AR draws cannot simply be relabelled as projected
    # iid draws. A reweighted proposal or a new sampler is required.


def test_complex_phase_covariance_force_equals_the_two_sided_rayleigh_gradient():
    config, params, labels, bits, u = _small_problem()
    bonds = [(0, 1, 1.0), (1, 3, 0.7), (3, 2, -0.2), (2, 0, 0.5),
             (0, 3, 0.125)]
    h = jnp.asarray(u.T@_physical_h(bits, bonds)@u)
    flat, unravel = ravel_pytree(params)

    def log_fn(vector):
        return many_cell_schur_log_amplitudes(unravel(vector), labels, config)

    def energy(vector):
        c = jnp.exp(log_fn(vector))
        return jnp.real(jnp.vdot(c, h@c)/jnp.vdot(c, c))

    logs = log_fn(flat)
    c = jnp.exp(logs)
    p = jnp.abs(c)**2
    score = jax.jacrev(lambda v: jnp.real(log_fn(v)))(flat)+1j*jax.jacrev(
        lambda v: jnp.imag(log_fn(v)))(flat)
    local = h@c/c
    mean = jnp.sum(p*local)/jnp.sum(p)
    force = 2*jnp.real(jnp.sum(p[:, None]*jnp.conj(score)*(local-mean)[:, None], axis=0))
    exact = jax.grad(energy)(flat)
    np.testing.assert_allclose(force, exact, atol=3e-13, rtol=2e-10)

    # Explicit phase-only variation must have a physical slope; dropping
    # imaginary score terms would incorrectly omit this part of the force.
    tangent = jax.tree_util.tree_map(jnp.zeros_like, params)
    tangent["direct_phase_bilinear"] = jnp.ones_like(params["direct_phase_bilinear"])
    direction, _ = ravel_pytree(tangent)
    slope = jnp.dot(exact, direction)
    epsilon = 1e-5
    finite_difference = (energy(flat+epsilon*direction)-energy(flat-epsilon*direction))/(2*epsilon)
    assert abs(float(slope)) > 1e-6
    np.testing.assert_allclose(slope, finite_difference, atol=2e-9, rtol=2e-6)


def test_n36_native_samples_obey_every_complete_reachability_mask_and_full_graph():
    config = ManyCellSchurViTConfig(
        rectangular_four_spin_layout(6, 6), embedding_dim=8, head_count=2, depth=1,
    )
    params = init_many_cell_schur_vit(jax.random.PRNGKey(202610093), config)
    labels = np.asarray(sample_many_cell_schur_vit(
        params, jax.random.PRNGKey(202610094), 16, config))
    assert labels.shape == (16, 35)
    assert np.all(labels[:, -1] == 0)
    left, right, masks = split_tables(config.layout.tree)
    for decision in tree_decisions(config.layout.tree):
        for row in labels:
            q = row[decision.node_index]
            ql = 1 if decision.left_index is None else row[decision.left_index]
            qr = 1 if decision.right_index is None else row[decision.right_index]
            assert np.any(masks[decision.order, q]
                          & (left[decision.order, q] == ql)
                          & (right[decision.order, q] == qr))
    # A large valid root split is allowed: no finite virtual-q truncation.
    root = tree_decisions(config.layout.tree)[0]
    allowed = left[root.order, 0][masks[root.order, 0]]
    assert int(max(allowed)) == min(root.left_leaves, root.right_leaves)
    graph = _bonds("triangular", 6, 6, 0.125)
    assert len(graph) == 216
    assert sum(float(bond[2]) == 1.0 for bond in graph) == 108
    assert sum(float(bond[2]) == 0.125 for bond in graph) == 108
    assert len({tuple(sorted(bond[:2])) for bond in graph}) == 216
    # Independently classify shells in the physical 120-degree convention,
    # rather than trusting the graph generator's direction list.
    for first, second, coupling in graph:
        row0, col0 = divmod(first, 6)
        row1, col1 = divmod(second, 6)
        squared = min(
            (row1-row0+6*a)**2+(col1-col0+6*b)**2
            -(row1-row0+6*a)*(col1-col0+6*b)
            for a in (-1, 0, 1) for b in (-1, 0, 1)
        )
        assert squared == (1 if float(coupling) == 1.0 else 3)
    inverse = np.asarray(config.layout.site_order)
    remapped = config.layout.remap_bonds(graph)
    restored = {(*sorted((int(inverse[a]), int(inverse[b]))), float(w)) for a, b, w in remapped}
    assert restored == {(*sorted((int(a), int(b))), float(w)) for a, b, w in graph}


def test_nontrivial_n8_reflection_preserves_the_full_physical_hamiltonian():
    """No diagonalization: explicit physical U†H U versus recoupling action."""
    config = ManyCellSchurViTConfig(
        rectangular_four_spin_layout(4, 2), embedding_dim=8, head_count=2, depth=1,
    )
    labels = np.asarray([state.labels for state in config.layout.tree.enumerate_states(0)],
                        dtype=np.int32)
    reflection = np.arange(8).reshape(2, 4)[:, (-np.arange(4)) % 4].ravel()
    assert np.any(reflection != np.arange(8))
    np.testing.assert_array_equal(reflection[reflection], np.arange(8))
    physical_order = reflection[np.asarray(config.layout.site_order)]
    bits, u = _independent_cg_isometry(config, labels, physical_order)
    physical_bonds = _bonds("triangular", 2, 4, .125)
    reflected_bonds = [(int(reflection[a]), int(reflection[b]), w)
                       for a, b, w in physical_bonds]
    physical_h = _physical_h(bits, physical_bonds)
    reduced_h = u.T@physical_h@u
    host = build_many_cell_tree_connectivity(labels, config, reflected_bonds)
    for column in range(len(labels)):
        unit = np.eye(len(labels), dtype=np.complex128)[:, column]
        actual = apply_many_cell_tree_connectivity(host, unit)
        np.testing.assert_allclose(actual, reduced_h[:, column], atol=3e-13, rtol=0)
    # The nontrivial reflection changes coordinates, not physical energies.
    rng = np.random.default_rng(202610095)
    c = rng.normal(size=len(labels))+1j*rng.normal(size=len(labels))
    c /= np.linalg.norm(c)
    psi = u@c
    np.testing.assert_allclose(np.vdot(psi, physical_h@psi),
                               np.vdot(c, apply_many_cell_tree_connectivity(host, c)),
                               atol=3e-13, rtol=0)


def test_n36_reflection_aligns_all_spatial_distances_and_group_gathers():
    """Physical 120-degree metric is exactly neural 60-degree metric via R."""
    config = ManyCellSchurViTConfig(
        rectangular_four_spin_layout(6, 6), embedding_dim=8, head_count=2, depth=1,
    )
    reflection = np.arange(36).reshape(6, 6)[:, (-np.arange(6)) % 6].ravel()
    distance = np.zeros((36, 36))
    for a in range(36):
        row0, col0 = divmod(a, 6)
        for b in range(36):
            row1, col1 = divmod(b, 6)
            distance[a, b] = np.sqrt(min(
                (row1-row0+6*r)**2+(col1-col0+6*c)**2
                -(row1-row0+6*r)*(col1-col0+6*c)
                for r in (-1, 0, 1) for c in (-1, 0, 1)))
    supports = []
    for decision in tree_decisions(config.layout.tree):
        node = config.layout.tree.internal_nodes[decision.node_index]
        supports.append([reflection[config.layout.site_order[index]]
                         for index in leaf_signature(node)])
    physical = np.asarray([[distance[np.ix_(a, b)].min() for b in supports]
                           for a in supports])
    np.testing.assert_allclose(many_cell_schur_spatial_distances(config), physical,
                               atol=2e-14, rtol=0)
    row = np.random.default_rng(202610096).permutation([0]*18+[1]*18)
    for physical_image in _groups("triangular", 6, 6)["gamma_a1"]:
        image = np.asarray(physical_image)
        model_image = reflection[image[reflection]]
        np.testing.assert_array_equal(row[image][reflection], row[reflection][model_image])
        np.testing.assert_allclose(distance[np.ix_(image, image)], distance, atol=2e-14)
