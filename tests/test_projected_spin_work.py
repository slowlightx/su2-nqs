"""Independent dense oracles for exact image/connection reuse and derivatives."""
import itertools

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin import _prepare_case
from projects.clebsch_tree_nqs.workflows.spin_decoder_cost import physical_config, amplitude_functions
from projects.clebsch_tree_nqs.workflows.spin_vap_decoder import init_decoder
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.vmc.projected_spin_work import (
    make_chunked_a1_amplitudes, make_deduplicated_a1_heisenberg, make_frozen_vmc_force,
)


@pytest.mark.parametrize('geometry', ['triangular', 'square'])
def test_deduplicated_H_replays_dense_complex_state_with_stabilizers_and_duplicates(geometry):
    data = _prepare_case(dict(geometry=geometry, rows=2, columns=2,
        j2_over_j1=.3, exact_energy_anchor=0), 'fixed_m0')
    projector = PhysicalSpatialProjector(data['groups']['gamma_a1'])
    basis = jnp.asarray(data['basis']); codes = jnp.array([8, 4, 2, 1])
    table = jnp.arange(1, 17, dtype=jnp.float64) + 1j * jnp.arange(16)**2 / 23
    raw = lambda p, x: p[jnp.dot(x, codes)]
    full = jnp.mean(table[jnp.sum(basis[:, jnp.asarray(projector.site_permutations)] * codes, axis=-1)], axis=1)
    duplicate_bonds = [(a, b, v/3) for a, b, v in data['bonds']] + [
        (b, a, 2*v/3) for a, b, v in data['bonds']]
    plan, evaluate, action = make_deduplicated_a1_heisenberg(raw, projector, duplicate_bonds, chunk_size=5)
    states = jnp.concatenate((basis, basis[:2]))
    ratios, psi, count, evaluations = jax.jit(action)(table, states)
    exact = (data['hamiltonian'] @ np.asarray(full))/np.asarray(full)
    np.testing.assert_allclose(ratios, np.concatenate((exact, exact[:2])), atol=2e-13)
    np.testing.assert_allclose(psi, np.concatenate((full, full[:2])), atol=2e-13)
    assert int(count) == 6 and int(evaluations) == 10
    np.testing.assert_allclose(jax.jit(evaluate)(table, jax.jit(plan)(states))[0], ratios, atol=2e-13)
    chunked = make_chunked_a1_amplitudes(raw, projector, chunk_size=5)
    np.testing.assert_allclose(jax.jit(chunked)(table, states), psi, atol=2e-13)
    # Fully polarized: only diagonal bonds, a single unique configuration.
    polarized = jnp.ones((2, 4), dtype=jnp.int32)
    result = jax.jit(action)(table, polarized)
    np.testing.assert_allclose(result[0], sum(v for _, _, v in data['bonds'])/4, atol=1e-14)
    assert int(result[2]) == 1


def test_chunked_schur_amplitude_gradient_and_full_H_are_unchanged():
    data = _prepare_case(dict(geometry='square', rows=2, columns=4,
        j2_over_j1=.5, exact_energy_anchor=0), 'fixed_m0')
    config = physical_config(2, 4, 2); params = init_decoder(991, config)
    old = amplitude_functions(config, data['groups']['gamma_a1'])
    full = jax.jit(jax.vmap(old['a1'], in_axes=(None, 0)))
    basis = jnp.asarray(data['basis']); vector = np.asarray(full(params, basis))
    selected = np.argsort(abs(vector))[-8:]; states = basis[selected]
    chunked = make_chunked_a1_amplitudes(old['raw'], old['projector'], chunk_size=7)
    np.testing.assert_allclose(jax.jit(chunked)(params, basis), vector, atol=2e-17, rtol=2e-12)
    _, _, action = make_deduplicated_a1_heisenberg(old['raw'], old['projector'], data['bonds'], chunk_size=7)
    result = jax.jit(action)(params, states)
    expected = (data['hamiltonian'] @ vector)[selected] / vector[selected]
    np.testing.assert_allclose(result[0], expected, atol=2e-12, rtol=2e-12)
    energies = jnp.asarray(expected)
    force = jax.jit(make_frozen_vmc_force(chunked, energies, 8))(params, states)
    reference = jax.jit(make_frozen_vmc_force(full, energies, 8))(params, states)
    flat, unravel = ravel_pytree(params)
    gradient = ravel_pytree(force)[0]
    np.testing.assert_allclose(gradient, ravel_pytree(reference)[0], atol=3e-12, rtol=3e-12)
    direction = jnp.sin(jnp.arange(len(flat)) + .3); direction /= jnp.linalg.norm(direction)
    def objective(theta):
        logs = jnp.log(chunked(unravel(theta), states))
        return 2*jnp.real(jnp.mean(jnp.conj(logs)*(energies-jnp.mean(energies))))/8
    objective = jax.jit(objective); epsilon = 1e-6
    finite_difference = (objective(flat+epsilon*direction)-objective(flat-epsilon*direction))/(2*epsilon)
    np.testing.assert_allclose(jnp.dot(gradient, direction), finite_difference, atol=2e-8, rtol=2e-7)


def test_exact_key_encoding_uses_all_36_bits_and_no_H_symmetry_assumption():
    n = 36; identity = tuple(range(n)); reverse = tuple(reversed(identity))
    projector = PhysicalSpatialProjector((identity, reverse))
    params = jnp.linspace(.01, .3, n)
    raw = lambda p, x: jnp.exp(jnp.dot(p, x)) * (1 + .1j*jnp.dot(x, jnp.arange(n)))
    bonds = [(0, 35, 1.), (1, 5, -.4)]  # deliberately not group-invariant
    states = np.zeros((3, n), dtype=np.int32)
    states[0, :18] = 1; states[1, 18:] = 1; states[2, ::2] = 1
    chunked = jax.jit(make_chunked_a1_amplitudes(raw, projector, chunk_size=3))
    _, _, action = make_deduplicated_a1_heisenberg(raw, projector, bonds, chunk_size=3)
    actual = jax.jit(action)(params, jnp.asarray(states))[0]
    expected = []
    for bits in states:
        center = chunked(params, bits[None])[0]; value = 0j
        for a, b, v in bonds:
            swapped = bits.copy(); swapped[a], swapped[b] = swapped[b], swapped[a]
            value += v*(.25 if bits[a] == bits[b] else -.25+.5*chunked(params, swapped[None])[0]/center)
        expected.append(value)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_chunked_projection_preserves_metropolis_detailed_balance_exactly():
    projector = PhysicalSpatialProjector((tuple(range(4)), (3, 2, 1, 0)))
    raw = lambda p, x: jnp.exp(x@p)*(1+.2j*x[0])
    params = jnp.asarray([.2, -.1, .4, -.3])
    states = np.asarray([x for x in itertools.product((0, 1), repeat=4) if sum(x)==2])
    amps = np.asarray(jax.jit(make_chunked_a1_amplitudes(raw, projector, chunk_size=3))(params, states))
    weights = abs(amps)**2; weights /= sum(weights)
    lookup = {tuple(x):i for i,x in enumerate(states)}
    transition = np.zeros((6, 6))
    for i, x in enumerate(states):
        for a in range(4):
            for offset in range(1, 4):
                b = (a+offset)%4; y=x.copy(); y[a], y[b]=y[b], y[a]
                j=lookup[tuple(y)]; acceptance=min(1., weights[j]/weights[i])
                transition[i,j] += acceptance/12
                transition[i,i] += (1-acceptance)/12
    np.testing.assert_allclose(weights[:,None]*transition, (weights[:,None]*transition).T, atol=2e-16)
    np.testing.assert_allclose(transition.sum(axis=1), 1, atol=2e-16)


def test_unsupported_characters_and_key_lengths_fail_closed():
    with pytest.raises(ValueError):
        make_chunked_a1_amplitudes(None, PhysicalSpatialProjector((tuple(range(64)),)))
    projector = PhysicalSpatialProjector(((0,1), (1,0)), characters=(1,-1))
    with pytest.raises(ValueError): make_deduplicated_a1_heisenberg(None, projector, [(0,1,1.)])
