"""Independent complete N12 tensor-action oracle; no ED or N36 evaluation."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from su2_nqs.benchmarks.multicontext_resonance_queries import (
    multicontext_carrier_amplitudes, multicontext_overlap_filter_rows,
)
from su2_nqs.benchmarks.overlapping_carrier_queries import (
    compile_overlapping_carrier_queries, local_overlap_connections, overlap_filter_rows,
)
from su2_nqs.models.conditional_carrier_resonance import local_scalar_components


def _active_oracle(generators):
    """Independent SciPy reduced exponentials, preserving every m copy."""
    parts = local_scalar_components()
    matrices = np.empty((6, 6, 16, 16))
    for first in range(6):
        for second in range(6):
            coefficients = np.zeros(10)
            for basis in parts.reduced_basis.values():
                generator = np.einsum("r,rij->ij", generators[first, second], basis)
                delta = expm(generator) - np.eye(basis.shape[-1])
                coefficients += np.einsum("rij,ij->r", basis, delta)
            matrices[first, second] = np.eye(16) + np.einsum("r,rij->ij", coefficients, parts.active_basis)
    return matrices


def _actions(first, second):
    p = local_scalar_components().spectator_projectors
    ub, uc = _active_oracle(first), _active_oracle(second)
    def k1(vector):
        return np.einsum("uij,uvkl,vmn,jln->ikm", p, ub, p,
                         vector.reshape(16, 16, 16), optimize=True).reshape(-1)
    def k2(vector):
        return np.einsum("uij,vkl,vumn,jln->ikm", p, p, uc,
                         vector.reshape(16, 16, 16), optimize=True).reshape(-1)
    return k1, k2


def _case():
    rng = np.random.default_rng(10401)
    first = rng.normal(0, .06, (6, 6, 10))
    second = rng.normal(0, .06, (6, 6, 10))
    vector = rng.normal(size=4096) + 1j * rng.normal(size=4096)
    vector[::9] = 0.
    vector /= np.linalg.norm(vector)
    return first, second, vector


def test_bitwise_identity_on_every_packed_local_row():
    rows = np.arange(4096)
    expected = np.zeros((4096, 216))
    for row in rows:
        columns, mask, identity = local_overlap_connections(int(row))
        expected[row, identity] = 1.
        assert columns[identity] == row and mask[identity]
    for mode in ("cascade", "no_cascade"):
        actual = np.asarray(multicontext_overlap_filter_rows(
            np.zeros((6, 6, 10)), np.zeros((6, 6, 10)), rows, mode=mode))
        np.testing.assert_array_equal(actual, expected)


def test_one_environment_limit_reproduces_the_old_sixty_table_exactly():
    rng = np.random.default_rng(10402)
    first, second = rng.normal(0, .06, (2, 6, 10))
    tables1 = np.broadcast_to(first[:, None], (6, 6, 10))
    tables2 = np.broadcast_to(second[:, None], (6, 6, 10))
    rows = np.arange(4096)
    maximum = 0.
    for mode in ("cascade", "no_cascade"):
        new = np.asarray(multicontext_overlap_filter_rows(tables1, tables2, rows, mode=mode))
        old = np.asarray(overlap_filter_rows(first, second, rows, mode=mode))
        maximum = max(maximum, float(np.max(abs(new - old))))
        np.testing.assert_allclose(new, old, atol=3e-14, rtol=3e-14)
    print("multicontext one-environment old-kernel limit max", maximum)


def test_every_n12_row_matches_independent_complete_tensor_product():
    first, second, carrier = _case()
    k1, k2 = _actions(first, second)
    rows = np.arange(4096)
    columns = np.stack([local_overlap_connections(int(row))[0] for row in rows])
    maximum = 0.
    for mode, direct in (("cascade", k2(k1(carrier))),
                         ("no_cascade", k1(carrier) + k2(carrier) - carrier)):
        weights = np.asarray(multicontext_overlap_filter_rows(first, second, rows, mode=mode))
        observed = np.sum(weights * carrier[columns], axis=1)
        maximum = max(maximum, float(np.max(abs(observed - direct))))
        np.testing.assert_allclose(observed, direct, atol=2e-14, rtol=3e-13)
    # These contexts genuinely fail to commute and have a nonzero cascade.
    assert np.linalg.norm(k2(k1(carrier)) - k1(k2(carrier))) > 1e-5
    assert np.linalg.norm(k2(k1(carrier)) - k1(carrier) - k2(carrier) + carrier) > 1e-4
    print("independent complete N12 product max", maximum)


def test_both_gate_gradients_match_independent_scipy_tensor_finite_differences():
    first, second, carrier = _case()
    rows = np.random.default_rng(10403).choice(4096, size=24, replace=False)
    columns = np.stack([local_overlap_connections(int(row))[0] for row in rows])
    weights = np.random.default_rng(10404).normal(size=24)
    def objective(a, b):
        matrix = multicontext_overlap_filter_rows(a, b, jnp.asarray(rows))
        values = jnp.sum(matrix * jnp.asarray(carrier[columns]), axis=1)
        return jnp.real(jnp.vdot(jnp.asarray(weights), values))
    gradients = jax.grad(objective, argnums=(0, 1))(jnp.asarray(first), jnp.asarray(second))
    eps = 1e-6
    maximum = 0.
    for index in (0, 1):
        direction = np.random.default_rng(10405 + index).normal(size=(6, 6, 10))
        direction /= np.linalg.norm(direction)
        plus, minus = [first.copy(), second.copy()], [first.copy(), second.copy()]
        plus[index] += eps * direction
        minus[index] -= eps * direction
        p1, p2 = _actions(*plus)
        m1, m2 = _actions(*minus)
        finite = np.vdot(weights, (p2(p1(carrier))[rows] - m2(m1(carrier))[rows]) / (2 * eps)).real
        exact = float(jnp.sum(gradients[index] * direction))
        maximum = max(maximum, abs(finite - exact))
        assert np.linalg.norm(np.asarray(gradients[index])) > 1e-4
        np.testing.assert_allclose(exact, finite, atol=3e-9, rtol=2e-6)
    print("two dynamic gate gradients independent FD max", maximum)


def test_existing_query_cache_reuse_with_one_final_complete_group_projection():
    first, second, raw = _case()
    all_bits = ((np.arange(4096)[:, None] >> np.arange(11, -1, -1)) & 1).astype(np.int8)
    group = np.stack([np.roll(np.arange(12), shift) for shift in (0, 4, 8)])
    orbit_indices = all_bits[:, group] @ (1 << np.arange(11, -1, -1))
    project = lambda vector: vector[orbit_indices].mean(axis=1)
    carrier = project(raw)
    rows = np.random.default_rng(10407).choice(4096, size=8, replace=False)
    plan = compile_overlapping_carrier_queries(all_bits[rows], group,
                                               max_unique_queries=4096, max_terminal_requests=100_000)
    indices = plan.configurations @ (1 << np.arange(11, -1, -1))
    cached = carrier[indices]
    k1, k2 = _actions(first, second)
    expected = project(k2(k1(carrier)))
    actual = np.asarray(multicontext_carrier_amplitudes(first, second, plan, cached))
    np.testing.assert_allclose(actual, expected[rows], atol=2e-14, rtol=2e-13)
    identity = multicontext_carrier_amplitudes(np.zeros((6, 6, 10)), np.zeros((6, 6, 10)), plan, cached)
    np.testing.assert_allclose(identity, carrier[rows], atol=3e-16)
    assert np.linalg.norm(expected - project(k2(project(k1(carrier))))) > 1e-5
    with pytest.raises(ValueError):
        multicontext_carrier_amplitudes(first, second, plan, cached[:-1])


def test_invalid_local_rows_and_composition_fail_closed():
    zero = np.zeros((6, 6, 10))
    for bad in (np.array([-1]), np.array([4096]), np.array([1.5]), np.array([], dtype=int)):
        with pytest.raises(ValueError):
            multicontext_overlap_filter_rows(zero, zero, bad)
    with pytest.raises(ValueError):
        multicontext_overlap_filter_rows(zero, zero, np.array([1]), mode="intermediate_projection")
