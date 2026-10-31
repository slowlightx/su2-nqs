import pytest
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from su2_nqs.vmc.padded_sparse import (
    make_chunked_paired_proposal_sampled_energy_difference,
    make_chunked_proposal_sampled_energy,
    make_importance_sampled_energy,
    make_paired_proposal_sampled_energy_difference,
    make_proposal_sampled_energy,
    make_proposal_sampled_energy_statistics,
)


def _assert_energy_statistics_equal(left, right, *, tolerance=1.0e-10):
    assert left.energy == pytest.approx(right.energy, abs=tolerance)
    assert left.effective_sample_size == pytest.approx(
        right.effective_sample_size, abs=tolerance
    )
    assert left.effective_sample_size_fraction == pytest.approx(
        right.effective_sample_size_fraction, abs=tolerance
    )
    assert left.delta_method_standard_error == pytest.approx(
        right.delta_method_standard_error, abs=tolerance
    )
    assert left.batch_standard_error == pytest.approx(
        right.batch_standard_error, abs=tolerance
    )
    assert left.standard_error == pytest.approx(right.standard_error, abs=tolerance)
    assert left.standard_error_method == right.standard_error_method
    assert left.sample_count == right.sample_count
    assert left.batch_count == right.batch_count


def test_importance_sampled_energy_matches_complete_two_state_sum():
    matrix = sparse.csr_matrix(np.asarray([[1.0, 0.2], [0.2, -1.0]]))
    basis = ((0,), (1,))
    samples = jnp.asarray(basis, dtype=jnp.int32)

    def log_psi(params, state):
        vector = params / jnp.linalg.norm(params)
        return jnp.log(vector[state[0]] + 0j)

    base = jnp.asarray([1.0, 1.0])
    base_logs = jnp.asarray([log_psi(base, state) for state in samples])
    estimator = make_importance_sampled_energy(
        matrix, basis, samples, base_logs, log_psi
    )
    candidate = jnp.asarray([2.0, 1.0])
    normalized = np.asarray(candidate) / np.linalg.norm(np.asarray(candidate))
    exact = float(normalized @ np.asarray(matrix @ normalized))
    assert estimator(candidate) == pytest.approx(exact, abs=1e-6)


def test_proposal_sampled_energy_matches_stratified_two_state_sum():
    matrix = sparse.csr_matrix(np.asarray([[0.4, -0.7], [-0.7, 1.2]]))
    basis = ((0,), (1,))
    # The deterministic sample multiset has empirical proposal r=(3/4, 1/4).
    samples = jnp.asarray(((0,), (0,), (0,), (1,)), dtype=jnp.int32)
    proposal_logs = jnp.log(jnp.asarray((0.75, 0.75, 0.75, 0.25)))

    def log_psi(params, state):
        vector = params / jnp.linalg.norm(params)
        return jnp.log(vector[state[0]] + 0j)

    params = jnp.asarray([0.6, 0.8])
    estimator = make_proposal_sampled_energy(
        matrix, basis, samples, proposal_logs, log_psi
    )
    vector = np.asarray(params) / np.linalg.norm(np.asarray(params))
    exact = float(np.real(np.vdot(vector, matrix @ vector)))
    assert estimator(params) == pytest.approx(exact, abs=1e-6)


def test_proposal_sampled_energy_statistics_match_ratio_influence_formula():
    matrix = sparse.diags(np.asarray([1.0, 3.0]), format="csr")
    basis = ((0,), (1,))
    samples = jnp.asarray(((0,), (0,), (1,), (1,)), dtype=jnp.int32)
    proposal_logs = jnp.full((4,), jnp.log(0.5))

    def log_psi(params, state):
        vector = params / jnp.linalg.norm(params)
        return jnp.log(vector[state[0]] + 0j)

    estimator = make_proposal_sampled_energy_statistics(
        matrix,
        basis,
        samples,
        proposal_logs,
        log_psi,
        batch_count=2,
    )
    statistics = estimator(jnp.asarray([0.5, np.sqrt(0.75)]))

    assert statistics.energy == pytest.approx(2.5, abs=1e-6)
    assert statistics.effective_sample_size == pytest.approx(3.2, abs=1e-6)
    assert statistics.effective_sample_size_fraction == pytest.approx(0.8, abs=1e-6)
    assert statistics.delta_method_standard_error == pytest.approx(
        np.sqrt(3.0) / 4.0, abs=1e-6
    )
    assert statistics.batch_standard_error == pytest.approx(0.75, abs=1e-6)
    assert statistics.standard_error == pytest.approx(0.75, abs=1e-6)
    assert statistics.standard_error_method == "batch"


def test_paired_energy_difference_uses_shared_sample_covariance():
    matrix = sparse.diags(np.asarray([1.0, 3.0]), format="csr")
    basis = ((0,), (1,))
    samples = jnp.asarray(((0,), (0,), (1,), (1,)), dtype=jnp.int32)
    proposal_logs = jnp.full((4,), jnp.log(0.5))

    def log_psi(params, state):
        vector = params / jnp.linalg.norm(params)
        return jnp.log(vector[state[0]] + 0j)

    compare = make_paired_proposal_sampled_energy_difference(
        matrix,
        basis,
        samples,
        proposal_logs,
        log_psi,
        batch_count=2,
        confidence_level=0.95,
    )
    candidate = jnp.asarray([0.5, np.sqrt(0.75)])
    reference = jnp.asarray([1.0, 1.0])
    result = compare(candidate, reference)

    assert result.energy_difference == pytest.approx(0.5, abs=1e-6)
    assert result.delta_method_standard_error == pytest.approx(
        1.0 / np.sqrt(48.0), abs=1e-6
    )
    assert result.batch_standard_error == pytest.approx(0.25, abs=1e-6)
    assert result.standard_error == pytest.approx(0.25, abs=1e-6)
    assert result.upper_confidence_bound > result.energy_difference
    assert not result.statistically_significant_improvement

    reversed_result = compare(reference, candidate)
    assert reversed_result.energy_difference == pytest.approx(-0.5, abs=1e-6)
    assert reversed_result.upper_confidence_bound < 0.0
    assert reversed_result.statistically_significant_improvement


def test_paired_energy_identical_states_have_exactly_zero_difference_and_error():
    # A complex state with a real Hermitian H has imaginary local energies;
    # the physical estimator deliberately uses their real parts.
    matrix = sparse.csr_matrix(np.asarray([[0.4, -0.7], [-0.7, 1.2]]))
    basis = ((0,), (1,))
    samples = jnp.asarray(((0,), (1,), (0,), (1,)), dtype=jnp.int32)
    proposal_logs = jnp.full((4,), jnp.log(0.5))

    def log_psi(params, state):
        vector = params / jnp.linalg.norm(params)
        return jnp.log(vector[state[0]])

    compare = make_paired_proposal_sampled_energy_difference(
        matrix,
        basis,
        samples,
        proposal_logs,
        log_psi,
        batch_count=2,
    )
    params = jnp.asarray([1.0 + 0.0j, 0.0 + 1.0j])
    result = compare(params, params)

    assert result.candidate.energy == pytest.approx(0.8, abs=1e-6)
    assert result.energy_difference == 0.0
    assert result.delta_method_standard_error == 0.0
    assert result.batch_standard_error == 0.0
    assert result.upper_confidence_bound == 0.0
    assert not result.statistically_significant_improvement


def test_chunked_energy_and_paired_gate_match_monolithic_with_padded_tail():
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [0.8, -0.3 + 0.2j, 0.1],
                [-0.3 - 0.2j, -0.4, 0.25],
                [0.1, 0.25, 1.1],
            ],
            dtype=np.complex128,
        )
    )
    basis = ((0,), (1,), (2,))
    # Six genuine samples do not divide the fixed chunk size four.  The
    # contiguous order also matters for the three two-sample batch means.
    samples = jnp.asarray(((0,), (2,), (1,), (0,), (1,), (2,)), dtype=jnp.int32)
    proposal_logs = jnp.log(jnp.asarray((0.4, 0.25, 0.35, 0.4, 0.35, 0.25)))

    def log_psi(params, state):
        vector = params / jnp.linalg.norm(params)
        return jnp.log(vector[state[0]])

    candidate = jnp.asarray((0.6 + 0.1j, -0.2 + 0.5j, 0.7 - 0.3j))
    reference = jnp.asarray((0.5 - 0.2j, 0.4 + 0.3j, -0.6 + 0.1j))
    monolithic_energy = make_proposal_sampled_energy(
        matrix, basis, samples, proposal_logs, log_psi
    )
    chunked_energy = make_chunked_proposal_sampled_energy(
        matrix,
        basis,
        samples,
        proposal_logs,
        log_psi,
        chunk_size=4,
    )
    assert chunked_energy(candidate) == pytest.approx(
        monolithic_energy(candidate), abs=1.0e-10
    )

    monolithic_compare = make_paired_proposal_sampled_energy_difference(
        matrix,
        basis,
        samples,
        proposal_logs,
        log_psi,
        batch_count=3,
        standard_error_method="max",
        confidence_level=0.975,
    )
    chunked_compare = make_chunked_paired_proposal_sampled_energy_difference(
        matrix,
        basis,
        samples,
        proposal_logs,
        log_psi,
        chunk_size=4,
        batch_count=3,
        standard_error_method="max",
        confidence_level=0.975,
    )
    expected = monolithic_compare(candidate, reference)
    actual = chunked_compare(candidate, reference)
    _assert_energy_statistics_equal(actual.candidate, expected.candidate)
    _assert_energy_statistics_equal(actual.reference, expected.reference)
    assert actual.energy_difference == pytest.approx(
        expected.energy_difference, abs=1.0e-10
    )
    assert actual.delta_method_standard_error == pytest.approx(
        expected.delta_method_standard_error, abs=1.0e-10
    )
    assert actual.batch_standard_error == pytest.approx(
        expected.batch_standard_error, abs=1.0e-10
    )
    assert actual.standard_error == pytest.approx(expected.standard_error, abs=1.0e-10)
    assert actual.standard_error_method == expected.standard_error_method
    assert actual.confidence_level == expected.confidence_level
    assert actual.upper_confidence_bound == pytest.approx(
        expected.upper_confidence_bound, abs=1.0e-10
    )
    assert (
        actual.statistically_significant_improvement
        == expected.statistically_significant_improvement
    )


@pytest.mark.parametrize("chunk_size", (0, -1, 1.5, True))
def test_chunked_energy_rejects_invalid_chunk_size(chunk_size):
    matrix = sparse.eye(2, format="csr")
    basis = ((0,), (1,))
    samples = jnp.asarray(((0,), (1,)), dtype=jnp.int32)
    proposal_logs = jnp.full((2,), jnp.log(0.5))

    def log_psi(params, state):
        return jnp.log(params[state[0]] + 0j)

    with pytest.raises(ValueError, match="positive integer"):
        make_chunked_proposal_sampled_energy(
            matrix,
            basis,
            samples,
            proposal_logs,
            log_psi,
            chunk_size=chunk_size,
        )


def test_chunked_energy_validates_tail_inputs_before_compilation():
    matrix = sparse.eye(2, format="csr")
    basis = ((0,), (1,))

    def log_psi(params, state):
        return jnp.log(params[state[0]] + 0j)

    with pytest.raises(ValueError, match="one value per sample"):
        make_chunked_proposal_sampled_energy(
            matrix,
            basis,
            jnp.asarray(((0,), (1,), (0,)), dtype=jnp.int32),
            jnp.zeros((2,)),
            log_psi,
            chunk_size=2,
        )
    with pytest.raises(ValueError, match="belong to the supplied basis"):
        make_chunked_proposal_sampled_energy(
            matrix,
            basis,
            jnp.asarray(((0,), (2,), (1,)), dtype=jnp.int32),
            jnp.zeros((3,)),
            log_psi,
            chunk_size=2,
        )


def test_statistical_energy_options_reject_ill_defined_batches():
    matrix = sparse.eye(2, format="csr")
    basis = ((0,), (1,))
    samples = jnp.asarray(((0,), (1,), (0,)), dtype=jnp.int32)
    proposal_logs = jnp.full((3,), jnp.log(0.5))

    def log_psi(params, state):
        del state
        return jnp.log(params[0] + 0j)

    with pytest.raises(ValueError, match="divide"):
        make_proposal_sampled_energy_statistics(
            matrix,
            basis,
            samples,
            proposal_logs,
            log_psi,
            batch_count=2,
        )
