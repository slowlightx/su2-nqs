"""Exact N8 checks for compact-carrier full-parameter training pullbacks.

Dense matrices are independent test oracles, not a training or N36/N64 path.
Every amplitude has the complete SU(2) decoder and coherent spatial A1 sum.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.lattices.spatial_symmetry import rectangular_gamma_a1_group
from su2_nqs.models.cell_multiplicity_clebsch_tree import (
    SpinCellMultiplicityConfig, init_spin_cell_multiplicity_tree,
)
from su2_nqs.models.schur_cutset_attention import (
    CutsetAttentionConfig, init_cutset_attention, make_branch_bank,
)
from su2_nqs.models.schur_cutset_compact import (
    make_compact_cutset_amplitude, make_full_a1_batch,
)
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.singlet_carrier_control import (
    carrier_control_evaluation, carrier_control_loss,
)

jax.config.update('jax_enable_x64', True)


def _pauli_hamiltonian(n, bonds):
    """Independent tensor-product oracle in |down>, |up> site order."""
    spins = (np.asarray([[0, .5], [.5, 0]], dtype=complex),
             np.asarray([[0, .5j], [-.5j, 0]], dtype=complex),
             np.diag([-.5, .5]).astype(complex))
    h = np.zeros((2**n, 2**n), dtype=complex)
    for a, b, coupling in bonds:
        for spin in spins:
            term = np.ones((1, 1), dtype=complex)
            for site in range(n):
                term = np.kron(term, spin if site in (a, b) else np.eye(2))
            h += coupling*term
    return h


@pytest.fixture(scope='module')
def problem():
    n = 8
    tree = SpinCellMultiplicityConfig('square_plaquette',
        ((0, 1, 4, 5), (2, 3, 6, 7)), feature_dim=2, initialization_scale=.4)
    cfg = CutsetAttentionConfig(tree, ((0., 0.), (2., 0.)))
    keys = jax.random.split(jax.random.PRNGKey(2026100511), 3)
    base = {name: init_spin_cell_multiplicity_tree(key, tree)
            for name, key in zip(('real', 'imag'), keys[:2], strict=True)}
    reference = init_cutset_attention(keys[2], base, cfg)
    # Finite displacement, not just the special carrier-replay point.
    leaves, structure = jax.tree_util.tree_flatten(reference)
    rng = np.random.default_rng(2026100512)
    params = jax.tree_util.tree_unflatten(structure, [
        v+jnp.asarray(.03*rng.normal(size=v.shape)) for v in leaves])
    full_states = ((np.arange(256)[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.int32)
    support = np.flatnonzero(full_states.sum(axis=1) == 4)
    states = jnp.asarray(full_states[support])
    group = np.asarray(rectangular_gamma_a1_group(2, 4), dtype=np.int32)
    residual = make_full_a1_batch(make_compact_cutset_amplitude(cfg),
        PhysicalSpatialProjector(group), chunk_size=7)
    bank_fn = lambda p: make_branch_bank(p, cfg)
    reference_bank = bank_fn(reference)
    reference_values = jax.jit(residual)(reference_bank, states)
    scale = float(jnp.linalg.norm(reference_values))
    assert scale > 1e-12
    # Independent exact singlet carrier: coherently A1-project four dimers.
    dimer = np.asarray([0., 1., -1., 0.])/np.sqrt(2.)
    carrier_full = dimer
    for _ in range(3):
        carrier_full = np.kron(carrier_full, dimer)
    image_indices = full_states[:, group] @ (1 << np.arange(7, -1, -1))
    carrier_full = carrier_full[image_indices].mean(axis=1).astype(complex)
    carrier_full /= np.linalg.norm(carrier_full)
    carrier = jnp.asarray(carrier_full[support])
    bonds = ((0, 1, 1.), (1, 4, -.4), (3, 7, .125), (0, 1, .07))
    full_h = _pauli_hamiltonian(n, bonds)
    h = jnp.asarray(full_h[np.ix_(support, support)])
    diagonal = jnp.asarray(3*np.diag(full_h)[support].real)
    hc = h@carrier
    eta = .2
    def vector(p):
        return carrier+eta*(residual(bank_fn(p), states)-reference_values)/scale
    return dict(n=n, cfg=cfg, params=params, states=states, group=group,
        residual=residual, bank_fn=bank_fn, reference=reference,
        reference_bank=reference_bank, reference_values=reference_values,
        scale=scale, eta=eta, vector=vector, carrier=carrier, hc=hc,
        h=h, diagonal=diagonal, bonds=bonds)


def _chunked_full_gradient(data, q, *, chunk=13):
    """Split neural-bank and physical-contraction pullbacks, one global loss."""
    p, residual, bank_fn = data['params'], data['residual'], data['bank_fn']
    bank, bank_pullback = jax.vjp(bank_fn, p)
    psi = data['vector'](p)
    evaluation = carrier_control_evaluation(psi, data['carrier'], data['hc'], q, data['diagonal'])
    source = jax.lax.stop_gradient(evaluation.residual/(q*evaluation.norm_sum))
    def bank_surrogate(b, x, s):
        return 2*jnp.vdot(data['eta']*residual(b, x)/data['scale'], s).real
    partial = jax.jit(jax.grad(bank_surrogate))
    gradient = jax.tree_util.tree_map(jnp.zeros_like, bank)
    for start in range(0, len(data['states']), chunk):
        values = partial(bank, data['states'][start:start+chunk], source[start:start+chunk])
        gradient = jax.tree_util.tree_map(lambda a, b: a+b, gradient, values)
    return evaluation.energy, bank_pullback(gradient)[0]


def _assert_gradients(first, second, *, atol=3e-12, rtol=3e-10):
    left = np.concatenate([np.asarray(v).ravel() for v in jax.tree_util.tree_leaves(first)])
    right = np.concatenate([np.asarray(v).ravel() for v in jax.tree_util.tree_leaves(second)])
    np.testing.assert_allclose(left, right, atol=atol, rtol=rtol)
    relative = np.linalg.norm(left-right)/np.linalg.norm(right)
    assert relative < rtol
    return float(np.max(abs(left-right))), float(relative)


def test_complete_singlet_cv_chunked_full_gradient_equals_dense_H_and_three_Hz(problem):
    data = problem
    q = jnp.ones(len(data['states']))
    def dense_loss(p):
        v = data['vector'](p)
        return jnp.vdot(v, data['h']@v).real/jnp.vdot(v, v).real
    def diagonal_loss(p):
        v = data['vector'](p)
        return jnp.vdot(v, data['diagonal']*v).real/jnp.vdot(v, v).real
    energy, gradient = _chunked_full_gradient(data, q)
    exact, exact_gradient = jax.jit(jax.value_and_grad(dense_loss))(data['params'])
    diag, diagonal_gradient = jax.jit(jax.value_and_grad(diagonal_loss))(data['params'])
    np.testing.assert_allclose([energy, diag], exact, atol=3e-13, rtol=0.)
    errors = [_assert_gradients(gradient, exact_gradient),
              _assert_gradients(diagonal_gradient, exact_gradient)]
    for key in ('base', 'directions', 'attention'):
        norm = np.sqrt(sum(np.sum(abs(np.asarray(v))**2)
                           for v in jax.tree_util.tree_leaves(gradient[key])))
        assert norm > 1e-9, f'{key} gradient must actually be exercised'
    assert np.linalg.norm(np.asarray(data['vector'](data['params'])-data['carrier'])) > 1e-4
    print({'full_parameter_CV_vs_H_and_3Hz_max_errors': errors})


def test_nonuniform_Q_chunked_pullback_equals_explicit_weighted_Hermitian_control(problem):
    data = problem
    q = jnp.asarray(np.exp(np.random.default_rng(2026100513).normal(size=len(data['states']))))
    c, hc, d = map(np.asarray, (data['carrier'], data['hc'], data['diagonal']))
    weight = 1/np.asarray(q); zc = np.sum(weight*abs(c)**2); r = hc-d*c
    a = np.sum(weight*c.conj()*r).real
    heff = (np.diag(d)+np.outer(r, weight*c.conj())/zc
            +np.outer(c, weight*r.conj())/zc-a*np.outer(c, weight*c.conj())/zc**2)
    np.testing.assert_allclose(weight[:, None]*heff, heff.conj().T*weight[None, :], atol=2e-15)
    matrix = jnp.asarray(heff)
    def oracle(p):
        v = data['vector'](p)
        return jnp.sum(v.conj()*(matrix@v)/q).real/jnp.sum(abs(v)**2/q)
    def ordinary_autodiff(p):
        return carrier_control_loss(data['vector'](p), data['carrier'], data['hc'], q, data['diagonal'])
    energy, gradient = _chunked_full_gradient(data, q, chunk=11)
    expected, reference = jax.jit(jax.value_and_grad(oracle))(data['params'])
    direct_energy, direct = jax.jit(jax.value_and_grad(ordinary_autodiff))(data['params'])
    np.testing.assert_allclose([energy, direct_energy], expected, atol=3e-13)
    errors = [_assert_gradients(gradient, reference), _assert_gradients(direct, reference)]
    assert np.count_nonzero(c == 0) > 0  # No C(x) or psi(x) denominator is needed.
    print({'weighted_control_full_parameter_max_errors': errors})


def test_endpoint_H_action_shares_frozen_reference_and_equals_dense_oracle(problem):
    data = problem
    residual = data['residual']; factor = data['eta']/data['scale']
    action = make_amplitude_heisenberg_action(
        lambda bank, x: factor*residual(bank, x[None])[0], data['bonds'], n_sites=8)
    bank = data['bank_fn'](data['params'])
    # Include carrier-node rows as well as nonzero rows; no ratio is formed.
    selected = np.asarray([0, 4, 13, 27, 39, 52, 61, 69])
    x = data['states'][selected]
    reference, href = action(data['reference_bank'], x)
    endpoint, hend = action(bank, x)
    psi = data['carrier'][selected]+endpoint-reference
    hpsi = data['hc'][selected]+hend-href
    full_vector = data['vector'](data['params'])
    np.testing.assert_allclose(psi, full_vector[selected], rtol=3e-12, atol=3e-14)
    np.testing.assert_allclose(hpsi, (data['h']@full_vector)[selected], rtol=3e-12, atol=3e-14)
    print({'shared_endpoint_H_max_error': float(np.max(abs(hpsi-(data['h']@full_vector)[selected])))})
