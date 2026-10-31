from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.lattice_observables import (
    spin_structure_factor_sz_operator,
    square_coordinates,
    triangular_phase_coordinates,
)
from su2_nqs.models.physical_tree import (
    PhysicalSpinTreeConfig,
    init_physical_spin_tree,
    physical_spin_tree_amplitude,
)
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.sz_basis import enumerate_sz_configs
from su2_nqs.vmc.spin_observables import (
    prepare_spin_structure_factor,
    reciprocal_phase_weights,
    spin_structure_factor_hpsi,
    spin_structure_factor_local_values,
    spin_structure_factor_multi_hpsi,
    spin_structure_factor_multi_local_values,
)


def _bit_lookup_amplitude(values: jax.Array, bits: jax.Array) -> jax.Array:
    powers = 2 ** jnp.arange(bits.shape[0] - 1, -1, -1, dtype=jnp.int32)
    return values[jnp.sum(bits.astype(jnp.int32) * powers)]


def _full_sector_state(
    basis: tuple[tuple[int, ...], ...], coefficients: np.ndarray
) -> np.ndarray:
    n_sites = len(basis[0])
    powers = 2 ** np.arange(n_sites - 1, -1, -1, dtype=np.int64)
    result = np.zeros(2**n_sites, dtype=np.complex128)
    for bits, value in zip(basis, coefficients, strict=True):
        result[int(np.dot(bits, powers))] = value
    return result


def _random_complex_state(n_sites: int, seed: int):
    basis = tuple(enumerate_sz_configs(n_sites, n_sites // 2))
    rng = np.random.default_rng(seed)
    values = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))
    values /= np.linalg.norm(values)
    return basis, values, _full_sector_state(basis, values)


def _exact_action(
    basis: tuple[tuple[int, ...], ...],
    coordinates: tuple[tuple[float, float], ...],
    wavevector: tuple[float, float],
    coefficients: np.ndarray,
    *,
    normalize_by_n_squared: bool = False,
) -> np.ndarray:
    operator, operator_basis = spin_structure_factor_sz_operator(
        coordinates,
        wavevector,
        {site: site for site in range(len(coordinates))},
        n_up=len(coordinates) // 2,
        normalize_by_n_squared=normalize_by_n_squared,
    )
    assert tuple(operator_basis) == basis
    return np.asarray(operator @ coefficients)


def test_registered_neel_k_and_m_phase_conventions() -> None:
    square = square_coordinates(2, 2)
    neel = reciprocal_phase_weights(square, (math.pi, math.pi))
    np.testing.assert_allclose(neel, (1.0, -1.0, -1.0, 1.0), atol=8.0e-16)

    triangular = triangular_phase_coordinates(2, 3)
    k = reciprocal_phase_weights(
        triangular, (2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0)
    )
    expected_k = np.asarray(
        [
            np.exp(2.0j * math.pi * (row + column) / 3.0)
            for row in range(2)
            for column in range(3)
        ]
    )
    np.testing.assert_allclose(k, expected_k, atol=8.0e-16)
    for wavevector, parity in (
        ((math.pi, 0.0), lambda row, column: row),
        ((0.0, math.pi), lambda row, column: column),
        ((math.pi, math.pi), lambda row, column: row + column),
    ):
        weights = reciprocal_phase_weights(triangular, wavevector)
        expected = [
            (-1.0) ** parity(row, column)
            for row in range(2)
            for column in range(3)
        ]
        np.testing.assert_allclose(weights, expected, atol=8.0e-16)


def test_arbitrary_complex_fixed_m_state_matches_independent_sparse_operator() -> None:
    n_sites = 8
    basis, coefficients, full_state = _random_complex_state(n_sites, seed=318)
    coordinates = square_coordinates(2, 4)
    wavevector = (0.37, -1.13)
    weights = reciprocal_phase_weights(coordinates, wavevector)
    spec = prepare_spin_structure_factor(weights)
    samples = jnp.asarray(basis, dtype=jnp.int32)

    @jax.jit
    def action(values):
        return spin_structure_factor_hpsi(
            values,
            samples,
            spec,
            amplitude_fn=_bit_lookup_amplitude,
        )

    centers, actual_action = action(jnp.asarray(full_state))
    expected_action = _exact_action(
        basis, coordinates, wavevector, coefficients
    )
    np.testing.assert_allclose(centers, coefficients, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        actual_action, expected_action, atol=3.0e-13, rtol=2.0e-13
    )

    local = spin_structure_factor_local_values(
        jnp.asarray(full_state),
        samples,
        spec,
        amplitude_fn=_bit_lookup_amplitude,
    )
    born_mean = np.vdot(coefficients, coefficients * np.asarray(local))
    exact_mean = np.vdot(coefficients, expected_action)
    np.testing.assert_allclose(born_mean, exact_mean, atol=3.0e-13, rtol=2.0e-13)
    assert abs(float(np.imag(born_mean))) < 2.0e-14


def test_multi_q_action_shares_kernel_and_matches_each_exact_operator() -> None:
    n_sites = 8
    basis, coefficients, full_state = _random_complex_state(n_sites, seed=991)
    coordinates = square_coordinates(2, 4)
    modes = (
        ((math.pi, math.pi), False),
        ((math.pi, 0.0), False),
        ((0.0, math.pi), False),
        ((0.37, -1.13), True),
    )
    specs = tuple(
        prepare_spin_structure_factor(
            reciprocal_phase_weights(coordinates, wavevector),
            normalization=float(n_sites**2) if is_m2 else None,
        )
        for wavevector, is_m2 in modes
    )
    samples = jnp.asarray(basis, dtype=jnp.int32)

    @jax.jit
    def multi_action(values):
        return spin_structure_factor_multi_hpsi(
            values,
            samples,
            specs,
            amplitude_fn=_bit_lookup_amplitude,
        )

    centers, actions = multi_action(jnp.asarray(full_state))
    assert actions.shape == (len(basis), len(specs))
    np.testing.assert_allclose(centers, coefficients, atol=0.0, rtol=0.0)
    for column, ((wavevector, is_m2), spec) in enumerate(zip(modes, specs, strict=True)):
        expected = _exact_action(
            basis,
            coordinates,
            wavevector,
            coefficients,
            normalize_by_n_squared=is_m2,
        )
        np.testing.assert_allclose(
            actions[:, column], expected, atol=2.0e-13, rtol=2.0e-13
        )
        _, single = spin_structure_factor_hpsi(
            jnp.asarray(full_state),
            samples,
            spec,
            amplitude_fn=_bit_lookup_amplitude,
        )
        np.testing.assert_allclose(
            actions[:, column], single, atol=5.0e-16, rtol=5.0e-15
        )

    multi_local = spin_structure_factor_multi_local_values(
        jnp.asarray(full_state),
        samples,
        specs,
        amplitude_fn=_bit_lookup_amplitude,
    )
    np.testing.assert_allclose(
        multi_local, np.asarray(actions) / coefficients[:, None], atol=2.0e-13
    )

    incompatible = prepare_spin_structure_factor(np.ones(6, dtype=np.complex128))
    with pytest.raises(ValueError, match="same site count"):
        spin_structure_factor_multi_hpsi(
            jnp.asarray(full_state),
            samples,
            (specs[0], incompatible),
            amplitude_fn=_bit_lookup_amplitude,
        )


@pytest.mark.parametrize(
    ("geometry", "wavevector"),
    (
        ("square", (math.pi, math.pi)),
        ("triangular", (2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0)),
        ("triangular", (math.pi, 0.0)),
        ("triangular", (0.0, math.pi)),
        ("triangular", (math.pi, math.pi)),
    ),
)
def test_named_2d_momenta_match_exact_actions(
    geometry: str, wavevector: tuple[float, float]
) -> None:
    n_sites = 6
    basis, coefficients, full_state = _random_complex_state(n_sites, seed=44)
    coordinates = (
        square_coordinates(2, 3)
        if geometry == "square"
        else triangular_phase_coordinates(2, 3)
    )
    spec = prepare_spin_structure_factor(
        reciprocal_phase_weights(coordinates, wavevector)
    )
    _, action = spin_structure_factor_hpsi(
        jnp.asarray(full_state),
        jnp.asarray(basis),
        spec,
        amplitude_fn=_bit_lookup_amplitude,
    )
    expected = _exact_action(basis, coordinates, wavevector, coefficients)
    np.testing.assert_allclose(action, expected, atol=2.0e-13, rtol=2.0e-13)


def test_physical_singlet_tree_has_zero_q0_structure_factor() -> None:
    n_sites = 6
    config = PhysicalSpinTreeConfig(
        FusionTree.balanced(n_sites),
        target_q=0,
        target_m2=0,
        feature_dim=3,
        initialization_scale=0.25,
    )
    params = init_physical_spin_tree(jax.random.PRNGKey(307), config)
    samples = jnp.asarray(enumerate_sz_configs(n_sites, n_sites // 2))
    spec = prepare_spin_structure_factor(np.ones(n_sites, dtype=np.complex128))

    @jax.jit
    def action(candidate):
        return spin_structure_factor_hpsi(
            candidate,
            samples,
            spec,
            amplitude_fn=lambda values, bits: physical_spin_tree_amplitude(
                values, bits, config
            ),
        )

    centers, structure_action = action(params)
    assert float(jnp.linalg.norm(centers)) > 1.0e-10
    np.testing.assert_allclose(structure_action, 0.0, atol=3.0e-13, rtol=0.0)


def test_discrete_reciprocal_sum_rules_distinguish_structure_factor_and_m2() -> None:
    n_sites = 4
    basis, coefficients, full_state = _random_complex_state(n_sites, seed=112)
    coordinates = tuple((float(site),) for site in range(n_sites))
    structure_expectations = []
    m2_expectations = []
    for mode in range(n_sites):
        weights = reciprocal_phase_weights(
            coordinates, (2.0 * math.pi * mode / n_sites,)
        )
        spec = prepare_spin_structure_factor(weights)
        assert spec.normalization == n_sites
        assert spec.normalization_kind == "structure_factor"
        centers, action = spin_structure_factor_hpsi(
            jnp.asarray(full_state),
            jnp.asarray(basis),
            spec,
            amplitude_fn=_bit_lookup_amplitude,
        )
        np.testing.assert_allclose(centers, coefficients, atol=0.0, rtol=0.0)
        structure_expectations.append(np.vdot(coefficients, np.asarray(action)))
        m2_spec = prepare_spin_structure_factor(
            weights, normalization=float(n_sites**2)
        )
        assert m2_spec.normalization_kind == "m2"
        _, m2_action = spin_structure_factor_hpsi(
            jnp.asarray(full_state),
            jnp.asarray(basis),
            m2_spec,
            amplitude_fn=_bit_lookup_amplitude,
        )
        m2_expectations.append(np.vdot(coefficients, np.asarray(m2_action)))
    np.testing.assert_allclose(
        np.sum(structure_expectations), 0.75 * n_sites, atol=3.0e-13, rtol=0.0
    )
    np.testing.assert_allclose(
        np.sum(m2_expectations), 0.75, atol=3.0e-13, rtol=0.0
    )


def test_zero_center_is_fail_closed_and_inputs_are_validated() -> None:
    samples = jnp.asarray(((0, 1), (1, 0)), dtype=jnp.int32)
    values = jnp.asarray((0.0, 0.0, 1.0, 0.0), dtype=jnp.complex128)
    spec = prepare_spin_structure_factor((1.0, -1.0))
    local = spin_structure_factor_local_values(
        values,
        samples,
        spec,
        amplitude_fn=_bit_lookup_amplitude,
    )
    assert not np.isfinite(np.asarray(local[0]))
    assert np.isfinite(np.asarray(local[1]))

    with pytest.raises(ValueError, match="nonempty"):
        prepare_spin_structure_factor(())
    with pytest.raises(ValueError, match="positive and finite"):
        prepare_spin_structure_factor((1.0, 1.0), normalization=0.0)
    with pytest.raises(ValueError, match="dimension"):
        reciprocal_phase_weights(((0.0, 0.0),), (0.0,))
    with pytest.raises(ValueError, match="nonnegative"):
        spin_structure_factor_local_values(
            values,
            samples,
            spec,
            amplitude_fn=_bit_lookup_amplitude,
            zero_tolerance=-1.0,
        )
