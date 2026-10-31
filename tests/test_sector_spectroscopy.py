import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.model_evaluation import model_vector
from su2_nqs.benchmarks.sector_spectroscopy import (
    SpinSectorSpec,
    build_exact_sector_reference,
    build_exact_fusion_problem,
    build_exact_spin_problem,
    chain_translation_fusion_matrix,
    cyclic_momentum_projectors,
    fixed_m_coupled_basis,
    momentum_weights,
    momentum_orbit_sampling_metrics,
    normalized_projected_state,
    sector_state_metrics,
    validate_sector_result_row,
)
from su2_nqs.lattices.chain import chain_bonds
from su2_nqs.models.fusion_gru import (
    FusionGRUConfig,
    init_fusion_gru,
    path_probabilities,
)
from su2_nqs.su2.fusion_path import count_paths, enumerate_paths


@pytest.mark.parametrize("target_q", (0, 2, 4))
def test_q0_q2_q4_reference_matches_fixed_m_projection(target_q: int) -> None:
    spec = SpinSectorSpec(
        system_id=f"chain_pbc_n8_q{target_q}",
        n_sites=8,
        target_q=target_q,
        bonds=chain_bonds(8, periodic=True),
        lattice="chain",
        boundary="periodic",
    )
    reference = build_exact_sector_reference(spec)

    assert len(reference.fusion_basis) == count_paths(8, target_q)
    assert reference.fusion_to_sz.shape == (
        len(reference.sz_basis),
        count_paths(8, target_q),
    )
    assert max(reference.audit.values()) < 3.0e-12
    metrics = sector_state_metrics(reference.ground_state, reference)
    assert abs(metrics["energy_error_per_site"]) < 1.0e-13
    assert metrics["infidelity"] < 1.0e-13
    assert metrics["sector_leakage"] == 0.0


def test_fixed_m_transform_uses_msb_first_sz_order() -> None:
    transform, basis = fixed_m_coupled_basis(6, 4)
    assert transform.shape == (len(basis), count_paths(6, 4))
    assert all(sum(configuration) == 5 for configuration in basis)
    np.testing.assert_allclose(
        transform.conj().T @ transform,
        np.eye(count_paths(6, 4)),
        atol=2.0e-14,
    )


@pytest.mark.parametrize("target_q", (0, 2, 4))
def test_q0_q2_q4_fusion_gru_normalization_and_problem(target_q: int) -> None:
    spec = SpinSectorSpec(
        system_id=f"chain_pbc_n8_q{target_q}",
        n_sites=8,
        target_q=target_q,
        bonds=chain_bonds(8, periodic=True),
    )
    reference = build_exact_sector_reference(spec)
    problem = build_exact_spin_problem(
        reference,
        embedding_dim=3,
        hidden_dim=5,
    )
    params = problem.init(jax.random.PRNGKey(410 + target_q))
    amplitudes = model_vector(
        params,
        np.asarray(problem.basis),
        problem.log_amplitude,
    )
    np.testing.assert_allclose(np.vdot(amplitudes, amplitudes), 1.0, atol=1.0e-12)
    metrics = sector_state_metrics(amplitudes, reference)
    assert metrics["spin_squared_expectation"] == pytest.approx(
        target_q * (target_q + 2) / 4.0
    )

    config = FusionGRUConfig(8, target_q, 3, 5)
    direct = init_fusion_gru(jax.random.PRNGKey(510 + target_q), config)
    probabilities = path_probabilities(
        direct,
        jnp.asarray(enumerate_paths(8, target_q)),
        config,
    )
    np.testing.assert_allclose(float(probabilities.sum()), 1.0, atol=1.0e-12)

    generic = build_exact_fusion_problem(
        spec.n_sites,
        spec.target_q,
        reference.fusion_hamiltonian,
        exact_energy=reference.ground_energy,
        embedding_dim=3,
        hidden_dim=5,
    )
    np.testing.assert_allclose(generic.hamiltonian, problem.hamiltonian)
    np.testing.assert_array_equal(generic.basis, problem.basis)


def test_sector_spec_and_result_schema_fail_closed() -> None:
    with pytest.raises(ValueError, match="unreachable"):
        SpinSectorSpec("bad", 8, 3, chain_bonds(8, periodic=True))

    row = {
        "system_id": "chain_pbc_n8_q4",
        "n_sites": 8,
        "target_q": 4,
        "basis_dimension": count_paths(8, 4),
        "exact_energy": -1.0,
        "exact_eigen_residual": 1.0e-14,
        "model": {"name": "exact_reference"},
        "metrics": {
            "normalization": 1.0,
            "energy": -1.0,
            "variance": 0.0,
            "fidelity": 1.0,
            "sector_leakage": 0.0,
        },
        "cost": {"wall_seconds": 0.0},
        "gate_pass": True,
    }
    validate_sector_result_row(row)
    row["basis_dimension"] += 1
    with pytest.raises(ValueError, match="basis_dimension"):
        validate_sector_result_row(row)


@pytest.mark.parametrize("target_q", (0, 2, 4))
def test_periodic_chain_momentum_projectors_are_exact(target_q: int) -> None:
    spec = SpinSectorSpec(
        system_id=f"chain_pbc_n8_q{target_q}",
        n_sites=8,
        target_q=target_q,
        bonds=chain_bonds(8, periodic=True),
        lattice="chain",
        boundary="periodic",
    )
    reference = build_exact_sector_reference(spec)
    translation = chain_translation_fusion_matrix(reference)
    projectors = cyclic_momentum_projectors(translation, period=8)
    identity = np.eye(len(reference.fusion_basis))

    index = {configuration: position for position, configuration in enumerate(reference.sz_basis)}
    translation_sz = np.zeros((len(index), len(index)))
    for column, configuration in enumerate(reference.sz_basis):
        shifted = (configuration[-1],) + configuration[:-1]
        translation_sz[index[shifted], column] = 1.0
    np.testing.assert_allclose(
        translation,
        reference.fusion_to_sz.conj().T
        @ translation_sz
        @ reference.fusion_to_sz,
        atol=2.0e-13,
    )

    assert len(projectors) == 8
    np.testing.assert_allclose(
        translation.conj().T @ translation, identity, atol=2.0e-13
    )
    np.testing.assert_allclose(
        np.linalg.matrix_power(translation, 8), identity, atol=2.0e-13
    )
    np.testing.assert_allclose(sum(projectors), identity, atol=2.0e-13)
    for first, projector in enumerate(projectors):
        np.testing.assert_allclose(projector, projector.conj().T, atol=2.0e-13)
        np.testing.assert_allclose(projector @ projector, projector, atol=2.0e-13)
        for second, other in enumerate(projectors):
            if first != second:
                np.testing.assert_allclose(projector @ other, 0.0, atol=2.0e-13)

    weights = momentum_weights(reference.ground_state, projectors)
    assert sum(weights) == pytest.approx(1.0, abs=2.0e-13)
    assert max(weights) == pytest.approx(1.0, abs=2.0e-13)
    projected, weight = normalized_projected_state(
        reference.ground_state, projectors[int(np.argmax(weights))]
    )
    assert weight == pytest.approx(1.0, abs=2.0e-13)
    assert sector_state_metrics(projected, reference)["fidelity"] == pytest.approx(
        1.0, abs=2.0e-13
    )
    orbit = momentum_orbit_sampling_metrics(
        reference.ground_state,
        translation,
        int(np.argmax(weights)),
        period=8,
    )
    assert orbit["projected_weight"] == pytest.approx(1.0, abs=2.0e-13)
    assert orbit["translation_eigen_residual"] < 2.0e-13
    assert orbit["proposal_population_ess_fraction"] == pytest.approx(
        1.0, abs=2.0e-13
    )
    assert orbit["target_weight_outside_proposal_support"] == 0.0
