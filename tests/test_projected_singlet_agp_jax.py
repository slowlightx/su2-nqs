from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.common_basis_gamma_a1 import (
    triangular_hnf_gamma_a1_circuit,
)
from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.exact_diagonalization import (
    computational_basis,
    spin_squared_matrix,
)
from su2_nqs.benchmarks.projected_singlet_agp import (
    ProjectedSingletAGP,
    exact_small_heisenberg_reference,
    pullback_spin_configuration,
    singlet_agp_pfaffian_reference,
    square_gamma_a1_group,
    triangular_hnf_gamma_a1_group,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.models.projected_singlet_agp import (
    ProjectedSingletAGPConfig,
    audit_projected_singlet_agp_batch,
    checked_projected_singlet_agp_log_amplitudes,
    invariant_pairing_fast_amplitudes,
    projected_singlet_agp_amplitudes,
    spatially_invariant_pairing_from_parameters,
    symmetric_pairing_from_parameters,
)
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.su2.sz_basis import enumerate_sz_configs
from su2_nqs.vmc.projected_agp import (
    ProjectedAGPHeisenbergConfig,
    direct_amplitude_rayleigh_estimate,
    projected_agp_heisenberg_action,
    require_valid_direct_amplitude_estimate,
)
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint


def _complex_symmetric_pairing(n_sites: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = (
        rng.normal(size=(n_sites, n_sites))
        + 1.0j * rng.normal(size=(n_sites, n_sites))
    ) / np.sqrt(n_sites)
    pairing = 0.5 * (raw + raw.T)
    np.fill_diagonal(pairing, 0.0)
    return pairing


def _parameters(pairing: np.ndarray, *, complex_mode: bool) -> dict[str, jax.Array]:
    first, second = np.triu_indices(pairing.shape[0], k=1)
    result = {"pairing_real": jnp.asarray(pairing.real[first, second])}
    if complex_mode:
        result["pairing_imag"] = jnp.asarray(pairing.imag[first, second])
    return result


def _integer_bits(state: int, n_sites: int) -> tuple[int, ...]:
    return tuple(
        (int(state) >> (n_sites - site - 1)) & 1
        for site in range(n_sites)
    )


def _relative_norm(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(reference), np.finfo(np.float64).tiny)
    )


def test_complex_jax_reynolds_amplitudes_match_numpy_and_pfaffian_sign() -> None:
    n_sites = 4
    pairing = _complex_symmetric_pairing(n_sites, 2026091350)
    group = square_gamma_a1_group(2)
    config = ProjectedSingletAGPConfig(
        n_sites, group, parameter_mode="complex"
    )
    parameters = _parameters(pairing, complex_mode=True)
    configurations = enumerate_sz_configs(n_sites, n_sites // 2)
    jax_result = projected_singlet_agp_amplitudes(
        parameters, configurations, config, microbatch_size=2
    )
    numpy_carrier = ProjectedSingletAGP(pairing, group)
    numpy_result = numpy_carrier.amplitudes(configurations)
    np.testing.assert_allclose(jax_result.amplitudes, numpy_result, atol=2.0e-14)
    np.testing.assert_allclose(
        jax_result.amplitudes,
        jax_result.phases * np.exp(jax_result.log_absolute),
        atol=2.0e-14,
    )

    identity = ProjectedSingletAGPConfig(
        n_sites, (tuple(range(n_sites)),), parameter_mode="complex"
    )
    base = projected_singlet_agp_amplitudes(
        parameters, configurations, identity
    ).amplitudes
    pfaffian = np.asarray(
        [singlet_agp_pfaffian_reference(state, pairing) for state in configurations]
    )
    np.testing.assert_allclose(base, pfaffian, atol=2.0e-14)


def test_jax_agp_vector_is_an_exact_singlet_for_complex_pairing() -> None:
    n_sites = 8
    pairing = _complex_symmetric_pairing(n_sites, 2026091351)
    config = ProjectedSingletAGPConfig(
        n_sites,
        (tuple(range(n_sites)),),
        parameter_mode="complex",
    )
    integer_basis = computational_basis(n_sites, n_up=n_sites // 2)
    configurations = tuple(_integer_bits(state, n_sites) for state in integer_basis)
    amplitudes = np.asarray(
        projected_singlet_agp_amplitudes(
            _parameters(pairing, complex_mode=True),
            configurations,
            config,
            microbatch_size=17,
        ).amplitudes
    )
    spin_squared, reference_basis = spin_squared_matrix(
        n_sites, n_up=n_sites // 2
    )
    assert reference_basis == integer_basis
    assert np.linalg.norm(amplitudes) > 1.0e-8
    assert np.linalg.norm(spin_squared @ amplitudes) / np.linalg.norm(amplitudes) < 3.0e-13


def test_triangular_jax_projection_matches_fixed_fusion_basis_projector() -> None:
    n_sites = 12
    supercell = ((6, 4), (0, 2))
    pairing = _complex_symmetric_pairing(n_sites, 2026091352)
    group = triangular_hnf_gamma_a1_group(supercell)
    config = ProjectedSingletAGPConfig(
        n_sites, group, parameter_mode="complex"
    )
    parameters = _parameters(pairing, complex_mode=True)
    configurations = enumerate_sz_configs(n_sites, n_sites // 2)
    projected = np.asarray(
        projected_singlet_agp_amplitudes(
            parameters, configurations, config, microbatch_size=128
        ).amplitudes
    )
    base_config = ProjectedSingletAGPConfig(
        n_sites, (tuple(range(n_sites)),), parameter_mode="complex"
    )
    base = np.asarray(
        projected_singlet_agp_amplitudes(
            parameters, configurations, base_config, microbatch_size=128
        ).amplitudes
    )

    full_base = np.zeros(1 << n_sites, dtype=np.complex128)
    full_projected = np.zeros_like(full_base)
    for state, base_value, projected_value in zip(
        computational_basis(n_sites, n_up=n_sites // 2),
        base,
        projected,
        strict=True,
    ):
        full_base[state] = base_value
        full_projected[state] = projected_value
    paths = enumerate_paths(n_sites, 0)
    coupled = coupled_basis_matrix(paths, 0, 0)
    circuit = triangular_hnf_gamma_a1_circuit(supercell, target_q=0)
    reference = circuit.apply(coupled.conj().T @ full_base)
    observed = coupled.conj().T @ full_projected
    assert np.linalg.norm(reference) > 1.0e-8
    assert _relative_norm(observed, reference) < 3.0e-12
    assert _relative_norm(coupled @ reference, full_projected) < 3.0e-12


def test_microbatch_and_complex_autodiff_match_finite_differences() -> None:
    n_sites = 4
    pairing = _complex_symmetric_pairing(n_sites, 2026091353)
    config = ProjectedSingletAGPConfig(
        n_sites, square_gamma_a1_group(2), parameter_mode="complex"
    )
    parameters = _parameters(pairing, complex_mode=True)
    configurations = jnp.asarray(enumerate_sz_configs(n_sites, n_sites // 2))
    weights = jnp.asarray(
        np.random.default_rng(2026091354).normal(size=len(configurations))
        + 1.0j * np.random.default_rng(2026091355).normal(size=len(configurations))
    )

    def loss(candidate, microbatch_size):
        amplitudes = projected_singlet_agp_amplitudes(
            candidate,
            configurations,
            config,
            microbatch_size=microbatch_size,
        ).amplitudes
        return jnp.real(jnp.vdot(weights, amplitudes))

    full_value, full_gradient = jax.value_and_grad(loss)(parameters, None)
    micro_value, micro_gradient = jax.value_and_grad(loss)(parameters, 2)
    np.testing.assert_allclose(micro_value, full_value, atol=2.0e-14)
    for name in parameters:
        np.testing.assert_allclose(
            micro_gradient[name], full_gradient[name], atol=3.0e-13
        )

    epsilon = 2.0e-6
    for name, index in (("pairing_real", 1), ("pairing_imag", 4)):
        plus = {key: np.array(value) for key, value in parameters.items()}
        minus = {key: np.array(value) for key, value in parameters.items()}
        plus[name][index] += epsilon
        minus[name][index] -= epsilon
        finite_difference = (
            float(loss(plus, 2)) - float(loss(minus, 2))
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            full_gradient[name][index], finite_difference, rtol=2.0e-7, atol=2.0e-9
        )


def test_jax_heisenberg_hpsi_matches_independent_dense_reference_and_gradients() -> None:
    n_sites = 4
    pairing = _complex_symmetric_pairing(n_sites, 2026091356)
    group = square_gamma_a1_group(2)
    carrier_config = ProjectedSingletAGPConfig(
        n_sites, group, parameter_mode="complex"
    )
    parameters = _parameters(pairing, complex_mode=True)
    bonds = square_j1j2_bonds(2, 2, j1=1.0, j2=0.31, periodic=True)
    graph_config = ProjectedAGPHeisenbergConfig(n_sites, bonds)
    configurations = enumerate_sz_configs(n_sites, n_sites // 2)
    result = projected_agp_heisenberg_action(
        parameters,
        configurations,
        carrier_config,
        graph_config,
        amplitude_microbatch_size=3,
    )

    numpy_carrier = ProjectedSingletAGP(pairing, group)
    reference = exact_small_heisenberg_reference(
        numpy_carrier, n_sites, bonds
    )
    assert reference.configurations == configurations
    np.testing.assert_allclose(result.amplitudes, reference.amplitudes, atol=3.0e-14)
    np.testing.assert_allclose(
        result.hamiltonian_actions, reference.hamiltonian_actions, atol=5.0e-13
    )

    def action_norm(candidate):
        action = projected_agp_heisenberg_action(
            candidate,
            configurations,
            carrier_config,
            graph_config,
            amplitude_microbatch_size=3,
        ).hamiltonian_actions
        return jnp.real(jnp.vdot(action, action))

    gradient = jax.grad(action_norm)(parameters)
    assert all(np.all(np.isfinite(value)) for value in gradient.values())
    assert sum(float(np.linalg.norm(value)) for value in gradient.values()) > 1.0e-8


def test_direct_amplitude_rayleigh_energy_and_gradient_need_no_logpsi() -> None:
    n_sites = 4
    pairing = _complex_symmetric_pairing(n_sites, 2026091363).real
    group = square_gamma_a1_group(2)
    carrier_config = ProjectedSingletAGPConfig(
        n_sites, group, parameter_mode="real"
    )
    parameters = _parameters(pairing, complex_mode=False)
    bonds = square_j1j2_bonds(2, 2, j1=1.0, j2=0.47, periodic=True)
    graph_config = ProjectedAGPHeisenbergConfig(n_sites, bonds)
    configurations = enumerate_sz_configs(n_sites, n_sites // 2)
    uniform_q = np.full(len(configurations), 1.0 / len(configurations))

    def energy(candidate):
        action = projected_agp_heisenberg_action(
            candidate,
            configurations,
            carrier_config,
            graph_config,
            amplitude_microbatch_size=4,
        )
        return direct_amplitude_rayleigh_estimate(
            action, uniform_q
        ).energy

    observed = float(energy(parameters))
    numpy_carrier = ProjectedSingletAGP(pairing, group)
    reference = exact_small_heisenberg_reference(
        numpy_carrier, n_sites, bonds
    )
    expected = float(
        np.real(
            np.vdot(reference.amplitudes, reference.hamiltonian_actions)
            / np.vdot(reference.amplitudes, reference.amplitudes)
        )
    )
    np.testing.assert_allclose(observed, expected, atol=3.0e-13)

    gradient = jax.grad(energy)(parameters)["pairing_real"]
    epsilon = 2.0e-6
    index = 2
    plus = {"pairing_real": np.array(parameters["pairing_real"])}
    minus = {"pairing_real": np.array(parameters["pairing_real"])}
    plus["pairing_real"][index] += epsilon
    minus["pairing_real"][index] -= epsilon
    finite_difference = (float(energy(plus)) - float(energy(minus))) / (
        2.0 * epsilon
    )
    np.testing.assert_allclose(
        gradient[index], finite_difference, rtol=5.0e-7, atol=3.0e-9
    )

    # A node in one row does not enter a denominator by itself.  The direct
    # estimator remains finite as long as the total carrier norm is nonzero.
    action = projected_agp_heisenberg_action(
        parameters,
        configurations,
        carrier_config,
        graph_config,
        amplitude_microbatch_size=4,
    )
    estimate = direct_amplitude_rayleigh_estimate(action, uniform_q)
    assert np.isfinite(estimate.energy)
    assert estimate.denominator > 0.0
    require_valid_direct_amplitude_estimate(estimate)
    expected_residual = float(
        np.linalg.norm(
            reference.hamiltonian_actions - expected * reference.amplitudes
        )
        / np.linalg.norm(reference.amplitudes)
    )
    np.testing.assert_allclose(estimate.residual_norm, expected_residual, atol=5.0e-13)
    assert estimate.numerator_imaginary_fraction < 3.0e-14
    assert 0.0 < estimate.state_weight_ess_fraction <= 1.0

    # Global AGP scales are a variational gauge.  The estimator removes that
    # scale before products, avoiding overflow even when amplitudes approach
    # the float64 range.
    huge = {"pairing_real": parameters["pairing_real"] * 1.0e150}
    huge_action = projected_agp_heisenberg_action(
        huge,
        configurations,
        carrier_config,
        graph_config,
        amplitude_microbatch_size=4,
    )
    huge_estimate = direct_amplitude_rayleigh_estimate(huge_action, uniform_q)
    require_valid_direct_amplitude_estimate(huge_estimate)
    np.testing.assert_allclose(huge_estimate.energy, expected, atol=5.0e-13)


def test_direct_amplitude_autodiff_stays_finite_at_true_determinant_nodes() -> None:
    n_sites = 4
    pairing = np.zeros((n_sites, n_sites), dtype=np.float64)
    pairing[0, 1] = pairing[1, 0] = 1.0
    pairing[2, 3] = pairing[3, 2] = 0.7
    carrier_config = ProjectedSingletAGPConfig(
        n_sites, (tuple(range(n_sites)),), parameter_mode="real"
    )
    parameters = _parameters(pairing, complex_mode=False)
    bonds = square_j1j2_bonds(2, 2, j1=1.0, j2=0.2, periodic=True)
    graph_config = ProjectedAGPHeisenbergConfig(n_sites, bonds)
    configurations = enumerate_sz_configs(n_sites, n_sites // 2)
    uniform_q = np.full(len(configurations), 1.0 / len(configurations))

    forward = projected_agp_heisenberg_action(
        parameters,
        configurations,
        carrier_config,
        graph_config,
        amplitude_microbatch_size=3,
    )
    node_count = int(np.sum(np.asarray(forward.center_exact_nodes)))
    assert 0 < node_count < len(configurations)
    estimate = direct_amplitude_rayleigh_estimate(forward, uniform_q)
    assert np.isfinite(estimate.energy)
    require_valid_direct_amplitude_estimate(estimate)

    def energy(candidate):
        action = projected_agp_heisenberg_action(
            candidate,
            configurations,
            carrier_config,
            graph_config,
            amplitude_microbatch_size=3,
        )
        return direct_amplitude_rayleigh_estimate(action, uniform_q).energy

    gradient = jax.grad(energy)(parameters)["pairing_real"]
    assert np.all(np.isfinite(gradient))
    audit = audit_projected_singlet_agp_batch(
        parameters, configurations, carrier_config
    )
    assert audit.exact_node_count == node_count
    assert not audit.sampled_log_vmc_safe


def test_zero_carrier_direct_estimate_fails_closed() -> None:
    n_sites = 4
    carrier_config = ProjectedSingletAGPConfig(
        n_sites, (tuple(range(n_sites)),), parameter_mode="real"
    )
    graph_config = ProjectedAGPHeisenbergConfig(
        n_sites,
        square_j1j2_bonds(2, 2, j1=1.0, j2=0.2, periodic=True),
    )
    parameters = {
        "pairing_real": jnp.zeros(
            (carrier_config.parameter_count_per_real_component,)
        )
    }
    configurations = enumerate_sz_configs(n_sites, n_sites // 2)
    action = projected_agp_heisenberg_action(
        parameters, configurations, carrier_config, graph_config
    )
    estimate = direct_amplitude_rayleigh_estimate(
        action, np.full(len(configurations), 1.0 / len(configurations))
    )
    assert not bool(estimate.valid)
    with pytest.raises(FloatingPointError, match="zero norm"):
        require_valid_direct_amplitude_estimate(estimate)


def test_real_pairing_time_reversal_and_node_log_api_fail_closed() -> None:
    n_sites = 8
    pairing = _complex_symmetric_pairing(n_sites, 2026091357).real
    config = ProjectedSingletAGPConfig(
        n_sites, (tuple(range(n_sites)),), parameter_mode="real"
    )
    configurations = enumerate_sz_configs(n_sites, n_sites // 2)
    parameters = _parameters(pairing, complex_mode=False)
    audit = audit_projected_singlet_agp_batch(
        parameters, configurations, config, microbatch_size=19
    )
    assert audit.time_reversal_pass
    assert audit.time_reversal_residual < 3.0e-14
    assert audit.sampled_log_vmc_safe
    logs = checked_projected_singlet_agp_log_amplitudes(
        parameters, configurations, config, microbatch_size=19
    )
    assert np.all(np.isfinite(logs))

    zero_parameters = {
        "pairing_real": jnp.zeros((config.parameter_count_per_real_component,))
    }
    failed = audit_projected_singlet_agp_batch(
        zero_parameters, configurations[:3], config
    )
    assert failed.exact_node_count == 3
    assert failed.severe_cancellation_count == 3
    assert not failed.sampled_log_vmc_safe
    with pytest.raises(FloatingPointError, match="unsafe"):
        checked_projected_singlet_agp_log_amplitudes(
            zero_parameters, configurations[:3], config
        )


def test_complex_pairing_is_explicitly_time_reversal_unconstrained() -> None:
    n_sites = 8
    pairing = _complex_symmetric_pairing(n_sites, 2026091358)
    config = ProjectedSingletAGPConfig(
        n_sites, (tuple(range(n_sites)),), parameter_mode="complex"
    )
    audit = audit_projected_singlet_agp_batch(
        _parameters(pairing, complex_mode=True),
        enumerate_sz_configs(n_sites, n_sites // 2),
        config,
    )
    assert config.checkpoint_metadata()["time_reversal"] == "unconstrained"
    assert audit.time_reversal_residual > 1.0e-3
    # The diagnostic is reported but is not falsely advertised as an enforced
    # gate in the explicitly unconstrained complex parameter mode.
    assert audit.time_reversal_pass


def test_invariant_pairing_fast_path_is_exact_only_for_even_permutation_group() -> None:
    n_sites = 16
    group = square_gamma_a1_group(4)
    config = ProjectedSingletAGPConfig(n_sites, group, parameter_mode="real")
    assert config.all_group_permutations_even
    pairing = _complex_symmetric_pairing(n_sites, 2026091359).real
    parameters = _parameters(pairing, complex_mode=False)
    rng = np.random.default_rng(2026091360)
    configurations = np.asarray(
        [
            enumerate_sz_configs(n_sites, n_sites // 2)[index]
            for index in rng.choice(12_870, size=5, replace=False)
        ]
    )
    invariant = np.asarray(spatially_invariant_pairing_from_parameters(parameters, config))
    projected_parameters = _parameters(invariant, complex_mode=False)
    full = projected_singlet_agp_amplitudes(
        projected_parameters, configurations, config
    ).amplitudes
    fast = invariant_pairing_fast_amplitudes(
        parameters, configurations, config
    ).amplitudes
    np.testing.assert_allclose(fast, full, atol=2.0e-18)

    odd_group_config = ProjectedSingletAGPConfig(
        4, square_gamma_a1_group(2), parameter_mode="real"
    )
    assert not odd_group_config.all_group_permutations_even
    with pytest.raises(ValueError, match="not physical A1"):
        invariant_pairing_fast_amplitudes(
            _parameters(_complex_symmetric_pairing(4, 2026091361).real, complex_mode=False),
            enumerate_sz_configs(4, 2),
            odd_group_config,
        )


def test_value_validation_and_checkpoint_metadata_are_fail_closed() -> None:
    config = ProjectedSingletAGPConfig(
        4, square_gamma_a1_group(2), parameter_mode="real"
    )
    pairing = _complex_symmetric_pairing(4, 2026091362).real
    parameters = _parameters(pairing, complex_mode=False)
    with pytest.raises(ValueError, match="binary"):
        audit_projected_singlet_agp_batch(
            parameters, ((1, 0, 2, -1),), config
        )
    with pytest.raises(ValueError, match="binary"):
        projected_singlet_agp_amplitudes(
            parameters, ((1, 0, 2, -1),), config
        )
    with pytest.raises(ValueError, match="exactly"):
        audit_projected_singlet_agp_batch(
            parameters, ((1, 0, 0, 0),), config
        )
    with pytest.raises(ValueError, match="exactly"):
        projected_singlet_agp_amplitudes(
            parameters, ((1, 0, 0, 0),), config
        )
    with pytest.raises(ValueError, match="parameter keys"):
        symmetric_pairing_from_parameters(
            {
                **parameters,
                "pairing_imag": jnp.zeros(
                    (config.parameter_count_per_real_component,)
                ),
            },
            config,
        )
    metadata = config.checkpoint_metadata()
    graph_metadata = ProjectedAGPHeisenbergConfig(
        4, square_j1j2_bonds(2, 2, j1=1.0, j2=0.2, periodic=True)
    ).checkpoint_metadata()
    assert metadata["carrier_role"] == "alternative_strict_sector_carrier_not_fusion_nqs"
    assert metadata["pairing_parameterization"] == "packed_strict_upper_triangle"
    assert metadata["required_sampling_contract"] == "external_positive_full_support_proposal"
    assert len(metadata["group_images_sha256"]) == 64
    assert len(graph_metadata["bonds_sha256"]) == 64


def test_packed_parameters_and_metadata_checkpoint_round_trip(tmp_path) -> None:
    config = ProjectedSingletAGPConfig(
        4, square_gamma_a1_group(2), parameter_mode="complex"
    )
    pairing = _complex_symmetric_pairing(4, 2026091366)
    parameters = _parameters(pairing, complex_mode=True)
    configurations = enumerate_sz_configs(4, 2)
    reference = projected_singlet_agp_amplitudes(
        parameters, configurations, config
    ).amplitudes
    destination = tmp_path / "agp.pkl"
    save_checkpoint(
        destination,
        parameters,
        optimizer_state={"step": 7},
        metadata=config.checkpoint_metadata(),
    )
    restored = load_checkpoint(destination)
    assert restored["metadata"] == config.checkpoint_metadata()
    assert restored["optimizer_state"] == {"step": 7}
    observed = projected_singlet_agp_amplitudes(
        restored["params"], configurations, config
    ).amplitudes
    np.testing.assert_array_equal(observed, reference)
