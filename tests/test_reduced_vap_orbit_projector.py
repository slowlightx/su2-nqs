"""Physical-oracle tests of coherent reduced-space A1 VAP, not orbit probabilities."""
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.clebsch_tree_exact import graph_hamiltonian_coo
from su2_nqs.lattices.spatial_symmetry import rectangular_gamma_a1_group
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig, init_many_cell_schur_vit, many_cell_schur_log_amplitudes,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.su2.reduced_physical_overlap import tree_overlaps_on_bits
from su2_nqs.vmc.exact_reduced_projector import compile_exact_a1_orbit_projector
from projects.clebsch_tree_nqs.workflows.spin_reduced_vit_revision import hermitian_rayleigh_surrogate


@pytest.fixture(scope='module')
def oracle():
    layout = rectangular_four_spin_layout(4, 2)
    labels = np.asarray([s.labels for s in layout.tree.enumerate_states(0)], np.int32)
    bits = np.asarray([b for b in product((0, 1), repeat=8) if sum(b) == 4], np.int32)
    u = tree_overlaps_on_bits(layout.tree, labels, bits, physical_leaf_order=layout.site_order)
    group = rectangular_gamma_a1_group(2, 4)
    plan = compile_exact_a1_orbit_projector(labels, u, bits, group)
    # Independent explicit physical gather: no orbit factor or index-table oracle.
    lookup = {tuple(b): i for i, b in enumerate(bits)}
    p = np.zeros((len(bits), len(bits)))
    for g in group:
        for row, b in enumerate(bits):
            p[row, lookup[tuple(b[np.asarray(g)])]] += 1 / len(group)
    return layout, labels, bits, u, group, plan, p


def test_factored_projector_and_actual_born_law(oracle):
    _, labels, _, u, _, plan, physical_p = oracle
    p = plan.factor.conj().T @ plan.factor
    np.testing.assert_allclose(u @ p, physical_p @ u, atol=2e-14, rtol=0)
    np.testing.assert_allclose(p @ p, p, atol=2e-14, rtol=0)
    np.testing.assert_allclose(p.conj().T, p, atol=2e-14, rtol=0)
    assert 1 < np.trace(p).real < len(labels)
    rng = np.random.default_rng(33)
    c = rng.normal(size=len(labels)) + 1j*rng.normal(size=len(labels))
    c /= np.linalg.norm(c)
    projected = np.asarray(jax.jit(plan.apply)(jnp.asarray(c)))
    np.testing.assert_allclose(u @ projected, physical_p @ (u @ c), atol=2e-14, rtol=0)
    probabilities = np.asarray(plan.probabilities(c))
    teacher=p@(rng.normal(size=len(labels))+1j*rng.normal(size=len(labels)))
    teacher/=np.linalg.norm(teacher)
    np.testing.assert_allclose(abs(np.vdot(teacher,projected/np.linalg.norm(projected)))**2,
        abs(np.vdot(teacher,c))**2/np.vdot(projected,projected).real,atol=2e-14,rtol=0)
    assert np.max(abs(probabilities - abs(c)**2)) > .01
    draws = np.asarray(plan.sample_indices(jax.random.PRNGKey(31), c, 32768))
    frequencies = np.bincount(draws, minlength=len(labels))/len(draws)
    assert np.max(abs(frequencies-probabilities)) < .012


def test_projected_gradient_is_full_physical_rayleigh_gradient(oracle):
    layout, labels, bits, u, _, plan, physical_p = oracle
    # Independent physical Heisenberg graph, invariant under the rectangle.
    bonds = set()
    for row in range(2):
        for col in range(4):
            site = 4*row+col
            for neighbor in (4*row+(col+1)%4, 4*((row+1)%2)+col):
                bonds.add((*sorted((site, neighbor)), 1.))
    i, j, v, basis = graph_hamiltonian_coo(8, 0, sorted(bonds))
    # The independent fixed-M Hamiltonian enumerates ones-first, while this
    # physical oracle uses lexicographic zeros-first bit strings.
    lookup = {tuple(b): index for index,b in enumerate(bits)}
    reorder = np.asarray([lookup[tuple(b)] for b in basis])
    h = np.zeros((len(bits), len(bits))); np.add.at(h, (reorder[i],reorder[j]), v)
    np.testing.assert_allclose(h@physical_p, physical_p@h, atol=2e-14, rtol=0)
    config = ManyCellSchurViTConfig(layout, embedding_dim=8, head_count=2, depth=1,
                                  conditional_mode='aligned_joint', geometry='square',
                                  direct_phase_initialization_scale=.1)
    params = init_many_cell_schur_vit(jax.random.PRNGKey(9), config)
    hj = jnp.asarray(u.T@h@u); uj = jnp.asarray(u); pp = jnp.asarray(physical_p)
    def reduced(p):
        c = plan.apply(jnp.exp(many_cell_schur_log_amplitudes(p, jnp.asarray(labels), config)))
        return jnp.vdot(c,hj@c).real / jnp.vdot(c,c).real
    def physical(p):
        state = pp@uj@jnp.exp(many_cell_schur_log_amplitudes(p,jnp.asarray(labels),config))
        return jnp.vdot(state,jnp.asarray(h)@state).real/jnp.vdot(state,state).real
    a, ga = jax.value_and_grad(reduced)(params)
    b, gb = jax.value_and_grad(physical)(params)
    np.testing.assert_allclose(a,b,atol=2e-13,rtol=0)
    for first, second in zip(jax.tree_util.tree_leaves(ga),jax.tree_util.tree_leaves(gb),strict=True):
        np.testing.assert_allclose(first,second,atol=2e-12,rtol=0)
    # A directional finite difference checks complex amplitude/phase jointly.
    direction = jax.tree_util.tree_map(lambda x:jnp.ones_like(x)*.01,params)
    shifted = lambda t:jax.tree_util.tree_map(lambda x,d:x+t*d,params,direction)
    finite = (float(reduced(shifted(1e-4)))-float(reduced(shifted(-1e-4))))/2e-4
    analytic = sum(float(jnp.vdot(g,d).real) for g,d in zip(
        jax.tree_util.tree_leaves(ga),jax.tree_util.tree_leaves(direction),strict=True))
    np.testing.assert_allclose(analytic,finite,atol=2e-8,rtol=0)
    for updated in (params,shifted(.2)):
        c = np.asarray(plan.apply(jnp.exp(many_cell_schur_log_amplitudes(updated,jnp.asarray(labels),config))))
        np.testing.assert_allclose(physical_p@(u@c),u@c,atol=3e-14,rtol=0)


def test_plan_rejects_invalid_basis_and_exponential_request(oracle):
    _, labels, bits, u, group, _, _ = oracle
    with pytest.raises(ValueError,match='orthonormal'):
        compile_exact_a1_orbit_projector(labels,2*u,bits,group)
    with pytest.raises(ValueError,match='cap'):
        compile_exact_a1_orbit_projector(labels,u,np.zeros((1,36),np.int32),[tuple(range(36))])
    # An orthonormal one-dimensional subspace is not in general group closed.
    with pytest.raises(ValueError,match='does not close'):
        compile_exact_a1_orbit_projector(labels[:1],u[:,:1],bits,group)


def test_two_sided_hermitian_first_gradient_with_complex_coefficients():
    # A complex Hermitian matrix makes a missing coherent term or conjugation
    # observable. Scaling/norm direction is included, not just normalized c.
    h=jnp.asarray([[.7, .3+.8j],[.3-.8j,-.2]])
    def state(x):return x[:2]+1j*x[2:]
    def direct(x):
        c=state(x);return jnp.vdot(c,h@c).real/jnp.vdot(c,c).real
    def analytic(x):
        c=state(x);return hermitian_rayleigh_surrogate(c,h@c)
    x=jnp.asarray([.8,-.1,.2,.5])
    first,ga=jax.value_and_grad(direct)(x);second,gb=jax.value_and_grad(analytic)(x)
    np.testing.assert_allclose(first,second,atol=2e-15,rtol=0)
    np.testing.assert_allclose(ga,gb,atol=2e-15,rtol=0)
