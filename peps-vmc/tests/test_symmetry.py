"""Independent dense-generator checks for the isolated SU(2)-PEPS core."""

from itertools import product

import numpy as np
import pytest

from peps_vmc.symmetry import (
    PHYSICAL_M,
    VIRTUAL_LABELS,
    intertwiner_labels,
    local_intertwiner_basis,
    reduced_tensor,
    rvb_coefficients,
    spin_generators,
    virtual_bond_metric,
    virtual_generators,
)
from su2_nqs.su2.clebsch_gordan import clebsch_gordan


@pytest.mark.parametrize("coordination,dimension", [(1, 1), (2, 2), (3, 5), (4, 12), (5, 30), (6, 76)])
def test_complete_reduced_channel_counts(coordination, dimension):
    labels = intertwiner_labels(coordination)
    assert len(labels) == dimension
    assert len(set(labels)) == dimension
    assert all(channel.fusion_q[-1] == 1 for channel in labels)


@pytest.mark.parametrize("coordination", [4, 6])
def test_intertwiners_are_orthonormal_with_declared_magnetic_norm(coordination):
    basis = local_intertwiner_basis(coordination).reshape(-1, 2, 3**coordination)
    for physical in range(2):
        np.testing.assert_allclose(
            basis[:, physical] @ basis[:, physical].T,
            np.eye(len(basis)),
            atol=3e-14,
            rtol=0,
        )
    np.testing.assert_allclose(
        basis.reshape(len(basis), -1) @ basis.reshape(len(basis), -1).T,
        2 * np.eye(len(basis)),
        atol=3e-14,
        rtol=0,
    )


@pytest.mark.parametrize("coordination", [4, 6])
def test_dense_lie_generator_intertwining_and_virtual_casimir(coordination):
    basis = local_intertwiner_basis(coordination)
    physical_generators = spin_generators(1)
    for physical_generator, virtual_generator in zip(physical_generators, virtual_generators()):
        virtual_action = np.zeros_like(basis, dtype=np.complex128)
        for leg in range(coordination):
            # Right action on a virtual ket: (A S)[...,input,...].
            term = np.tensordot(basis, virtual_generator, axes=((2 + leg,), (0,)))
            virtual_action += np.moveaxis(term, -1, 2 + leg)
        physical_action = np.einsum("ab,kb...->ka...", physical_generator, basis)
        np.testing.assert_allclose(virtual_action, physical_action, atol=2e-14, rtol=0)

    # An independent Casimir action on every mapped virtual state gives 3/4.
    casimir_action = np.zeros_like(basis, dtype=np.complex128)
    for virtual_generator in virtual_generators():
        once = np.zeros_like(basis, dtype=np.complex128)
        for leg in range(coordination):
            term = np.tensordot(basis, virtual_generator, axes=((2 + leg,), (0,)))
            once += np.moveaxis(term, -1, 2 + leg)
        for leg in range(coordination):
            term = np.tensordot(once, virtual_generator, axes=((2 + leg,), (0,)))
            casimir_action += np.moveaxis(term, -1, 2 + leg)
    np.testing.assert_allclose(casimir_action, 0.75 * basis, atol=3e-14, rtol=0)


def test_magnetic_selection_and_one_half_leg_initializer():
    z = 4
    basis = local_intertwiner_basis(z)
    tensor = reduced_tensor(rvb_coefficients(z), basis)
    for indices in product(range(3), repeat=z):
        ms = [VIRTUAL_LABELS[index][1] for index in indices]
        qs = [VIRTUAL_LABELS[index][0] for index in indices]
        for physical, m in enumerate(PHYSICAL_M):
            if sum(ms) != m:
                assert np.all(basis[(slice(None), physical) + indices] == 0)
            expected = float(sum(qs) == 1 and sum(ms) == m)
            assert tensor[(physical,) + indices] == pytest.approx(expected, abs=2e-15)


def test_bond_condon_shortley_oracle_and_dense_singlet_generators():
    bond = virtual_bond_metric()
    for first, (q, m) in enumerate(VIRTUAL_LABELS):
        for second, (other_q, other_m) in enumerate(VIRTUAL_LABELS):
            oracle = clebsch_gordan(q, m, q, other_m, 0, 0) if q == other_q else 0.0
            assert bond[first, second] == pytest.approx(oracle, abs=2e-15)
    assert bond[1, 2] < 0 < bond[2, 1]
    for generator in virtual_generators():
        np.testing.assert_allclose(generator @ bond + bond @ generator.T, 0, atol=2e-15)


def test_two_site_closed_map_is_physical_singlet():
    first = reduced_tensor(rvb_coefficients(1), local_intertwiner_basis(1))
    second = first.copy()
    wavefunction = first @ virtual_bond_metric() @ second.T
    np.testing.assert_allclose(wavefunction[0, 1], -1 / np.sqrt(2), atol=2e-15)
    np.testing.assert_allclose(wavefunction[1, 0], 1 / np.sqrt(2), atol=2e-15)
    for generator in spin_generators(1):
        np.testing.assert_allclose(generator @ wavefunction + wavefunction @ generator.T, 0, atol=2e-15)


def test_complex_reduced_coefficients_remain_intertwiners():
    basis = local_intertwiner_basis(4)
    rng = np.random.default_rng(19)
    coefficients = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    tensor = reduced_tensor(coefficients, basis)
    for physical_generator, virtual_generator in zip(spin_generators(1), virtual_generators()):
        virtual_action = np.zeros_like(tensor)
        for leg in range(4):
            term = np.tensordot(tensor, virtual_generator, axes=((1 + leg,), (0,)))
            virtual_action += np.moveaxis(term, -1, 1 + leg)
        np.testing.assert_allclose(
            np.einsum("ab,b...->a...", physical_generator, tensor),
            virtual_action,
            atol=5e-14,
            rtol=0,
        )


def test_jax_reduced_tensor_and_gradient_match_direct_linear_map():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    basis = local_intertwiner_basis(4)
    weights = np.arange(len(basis), dtype=np.float64) / 11
    tensor = reduced_tensor(jnp.asarray(weights), basis, backend="jax")
    np.testing.assert_allclose(tensor, reduced_tensor(weights, basis), atol=2e-14)
    loss = lambda w: jnp.sum(reduced_tensor(w, basis, backend="jax") ** 2)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(weights), 4 * weights, atol=2e-14)


@pytest.mark.parametrize("value", [True, 1.5, "4"])
def test_noninteger_coordination_rejected(value):
    with pytest.raises(TypeError):
        intertwiner_labels(value)


@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_coordination_rejected(value):
    with pytest.raises(ValueError):
        local_intertwiner_basis(value)


def test_invalid_reduced_api_inputs_rejected():
    basis = local_intertwiner_basis(4)
    with pytest.raises(ValueError):
        reduced_tensor(np.zeros(11), basis)
    with pytest.raises(ValueError):
        reduced_tensor(np.zeros(12), np.zeros((12, 2, 2, 2)))
    with pytest.raises(ValueError):
        reduced_tensor(np.zeros(12), basis, backend="not-a-backend")
    with pytest.raises(ValueError):
        rvb_coefficients(4, [1, 2])
    with pytest.raises(TypeError):
        spin_generators(0.5)
