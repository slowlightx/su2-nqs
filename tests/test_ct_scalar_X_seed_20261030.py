"""Independent dense scalar-source, spin, character and actual-H checks."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.explorations.ct_scalar_X_seed_20261030 import (
    projected_scalar_bonds, make_coherent_scalar_source, source_covariance_residual,
)
from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.recoupling import tree_basis_matrix
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action


def test_scalar_seed_matches_explicit_projection_exact_singlet_and_characters():
    n = 6
    rows = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    powers = 1 << np.arange(n-1, -1, -1)
    images = tuple(tuple(np.roll(np.arange(n), -t)) for t in range(n))
    characters = tuple(np.exp(2j*np.pi*t/n) for t in range(n))
    projector = PhysicalSpatialProjector(images, characters)
    seed = ((0, 1, np.sqrt(n)),)
    bonds = projected_scalar_bonds(projector, seed)
    u, _ = tree_basis_matrix(FusionTree.left_associated(n), 0, 0)
    rng = np.random.default_rng(30301)
    raw = u@rng.normal(size=u.shape[1])
    gamma = np.mean([raw[rows[:, p]@powers] for p in images], axis=0)
    source_matrix = sum(c*spin_dot_matrix(n, a, b) for a, b, c in bonds)
    source = source_matrix@gamma
    seed_state = np.sqrt(n)*spin_dot_matrix(n, 0, 1)@gamma
    explicit = sum(np.conj(c)*seed_state[rows[:, p]@powers]
                   for p, c in zip(images, characters))/n
    assert np.linalg.norm(source) > .01
    np.testing.assert_allclose(source, explicit, atol=2e-15, rtol=0)
    s2 = .75*n*np.eye(2**n)+2*sum(spin_dot_matrix(n, a, b)
        for a in range(n) for b in range(a+1, n))
    np.testing.assert_allclose(s2@source, 0., atol=2e-15, rtol=0)
    for p, c in zip(images, characters):
        np.testing.assert_allclose(source[rows[:, p]@powers], c*source, atol=2e-15, rtol=0)

    def factory(edges):
        ed = jnp.asarray(edges)
        def query(bank, row):
            def changed(edge):
                a, b = edge
                y = row.at[a].set(row[b]).at[b].set(row[a])
                return bank[jnp.dot(y, jnp.asarray(powers))]
            return bank[jnp.dot(row, jnp.asarray(powers))], jax.vmap(changed)(ed)
        return query
    evaluate, _ = make_coherent_scalar_source(factory, np.asarray(images), bonds, projection_chunk=2)
    actual = np.asarray(evaluate(jnp.asarray(raw), jnp.asarray(rows, jnp.uint8)))
    np.testing.assert_allclose(actual[:, 0], gamma, atol=2e-15, rtol=0)
    np.testing.assert_allclose(actual[:, 1], source, atol=2e-15, rtol=0)
    # Nonuniform H and an imperfect carrier: full H Bpsi, not E0 Bpsi.
    h_bonds = tuple((i, (i+1)%n, 1.+.1*i) for i in range(n))
    h = sum(c*spin_dot_matrix(n, a, b) for a, b, c in h_bonds)
    amplitude = lambda bank, row: evaluate(bank, row[None, :])[0, 1]
    action = make_amplitude_heisenberg_action(amplitude, h_bonds, n_sites=n)
    actual_psi, actual_hpsi = map(np.asarray, action(jnp.asarray(raw), jnp.asarray(rows)))
    np.testing.assert_allclose(actual_psi, source, atol=2e-15, rtol=0)
    np.testing.assert_allclose(actual_hpsi, h@source, atol=3e-15, rtol=0)


def test_scalar_source_invalid_inputs_not_repaired():
    projector = PhysicalSpatialProjector(((0, 1, 2, 3),))
    for bonds in ((), ((0, 0, 1.),), ((False, 1, 1.),), ((0, 1, np.nan),)):
        with pytest.raises(ValueError):
            projected_scalar_bonds(projector, bonds)
    with pytest.raises(ValueError):
        make_coherent_scalar_source(lambda edges: None, np.asarray([[0, 1, 2, 3]]),
                                    ((0, 1, 1.),), projection_chunk=2)


def test_character_gate_does_not_accept_null_or_nonfinite_sources():
    for original, changed in ((np.zeros(2), np.zeros(2)),
                              (np.ones(2), np.array([np.nan, 1.])),
                              (np.array([np.inf, 1.]), np.ones(2))):
        with pytest.raises(ArithmeticError):
            source_covariance_residual(original, changed, 1.)
    assert source_covariance_residual(np.array([1j, 2.]), np.array([1j, 2.]), 1.) == 0.


def test_noncommuting_translation_reflection_group_scalar_seed():
    n = 4
    rows = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    powers = 1 << np.arange(n-1, -1, -1)
    images = tuple(tuple((sign*np.arange(n)+t)%n) for sign in (1, -1) for t in range(n))
    characters = tuple(complex((-1)**t * sign) for sign in (1, -1) for t in range(n))
    projector = PhysicalSpatialProjector(images, characters)
    u, _ = tree_basis_matrix(FusionTree.left_associated(n), 0, 0)
    raw = u@np.array([.3, .7])
    gamma = np.mean([raw[rows[:, p]@powers] for p in images], axis=0)
    bonds = projected_scalar_bonds(projector, ((0, 1, 1.),))
    source = sum(c*spin_dot_matrix(n, a, b) for a, b, c in bonds)@gamma
    explicit_seed = spin_dot_matrix(n, 0, 1)@gamma
    explicit = sum(np.conj(c)*explicit_seed[rows[:, p]@powers]
                   for p, c in zip(images, characters))/len(images)
    assert np.linalg.norm(source) > .01
    np.testing.assert_allclose(source, explicit, atol=2e-15, rtol=0)
    for p, c in zip(images, characters):
        np.testing.assert_allclose(source[rows[:, p]@powers], c*source, atol=2e-15, rtol=0)
