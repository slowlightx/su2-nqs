import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from su2_nqs.vmc.fisher_preconditioner import (
    FisherBlockSpec,
    build_chunked_fisher_preconditioner,
    build_diagonal_fisher_preconditioner,
    build_fisher_preconditioner,
    exact_carrier_decoder_block_specs,
    sparse_causal_head_block_specs,
)


def _weighted_center(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalized = weights / np.sum(weights)
    return values - np.sum(normalized.reshape((-1,) + (1,) * (values.ndim - 1)) * values, axis=0)


def _dense_matrix_from_scores(scores: object, weights: np.ndarray) -> np.ndarray:
    leaves = jax.tree_util.tree_leaves(scores)
    sample_count = leaves[0].shape[0]
    matrix = np.concatenate(
        [np.asarray(leaf).reshape(sample_count, -1) for leaf in leaves], axis=1
    )
    normalized = weights / np.sum(weights)
    return 2.0 * np.real(matrix.conj().T @ (normalized[:, None] * matrix))


def _dense_preconditioner(preconditioner, params: object) -> np.ndarray:
    flat, unravel = ravel_pytree(params)
    columns = []
    for index in range(flat.size):
        basis = np.zeros(flat.size, dtype=np.float64)
        basis[index] = 1.0
        applied, _ = ravel_pytree(preconditioner.apply(unravel(jnp.asarray(basis))))
        columns.append(np.asarray(applied))
    return np.column_stack(columns)


def _pcg(
    matrix: np.ndarray,
    rhs: np.ndarray,
    inverse_preconditioner: np.ndarray | None,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, float]:
    solution = np.zeros_like(rhs)
    residual = rhs.copy()
    preconditioned = (
        residual.copy()
        if inverse_preconditioner is None
        else inverse_preconditioner @ residual
    )
    direction = preconditioned.copy()
    residual_product = float(residual @ preconditioned)
    rhs_norm = np.linalg.norm(rhs)
    for iteration in range(1, max_iterations + 1):
        product = matrix @ direction
        step = residual_product / float(direction @ product)
        solution += step * direction
        residual -= step * product
        relative_residual = float(np.linalg.norm(residual) / rhs_norm)
        if relative_residual <= tolerance:
            return solution, iteration, relative_residual
        next_preconditioned = (
            residual.copy()
            if inverse_preconditioner is None
            else inverse_preconditioner @ residual
        )
        next_product = float(residual @ next_preconditioned)
        direction = next_preconditioned + (next_product / residual_product) * direction
        preconditioned = next_preconditioned
        residual_product = next_product
    return solution, max_iterations, float(np.linalg.norm(residual) / rhs_norm)


def test_diagonal_fisher_inverse_matches_weighted_dense_diagonal_and_jit():
    params = {
        "a": jnp.zeros((3,), dtype=jnp.float64),
        "nested": {"b": jnp.zeros((2, 2), dtype=jnp.float64)},
    }
    weights = np.asarray([0.05, 0.15, 0.3, 0.2, 0.3], dtype=np.float64)
    generator = np.random.default_rng(17)
    raw_scores = {
        "a": generator.normal(size=(5, 3))
        + 1j * generator.normal(size=(5, 3)),
        "nested": {
            "b": generator.normal(size=(5, 2, 2))
            + 1j * generator.normal(size=(5, 2, 2))
        },
    }
    scores = jax.tree_util.tree_map(
        lambda value: jnp.asarray(_weighted_center(value, weights)), raw_scores
    )
    damping = 3.0e-3
    preconditioner = build_diagonal_fisher_preconditioner(
        params,
        scores,
        damping=damping,
        sample_weights=jnp.asarray(weights),
    )
    vector = {
        "a": jnp.asarray([0.2, -0.7, 1.1]),
        "nested": {"b": jnp.asarray([[0.4, -0.2], [0.8, 0.3]])},
    }

    flat_vector, _ = ravel_pytree(vector)
    fisher = _dense_matrix_from_scores(scores, weights)
    expected = np.asarray(flat_vector) / (np.diag(fisher) + damping)
    actual, _ = ravel_pytree(preconditioner.apply(vector))
    compiled, _ = ravel_pytree(jax.jit(preconditioner.apply)(vector))

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(np.asarray(compiled), expected, rtol=2e-13, atol=2e-13)
    assert preconditioner.blocks == ()
    assert preconditioner.sample_count == 5


def test_sparse_physical_blocks_match_independent_dense_block_inverses():
    width = 2
    params = {
        "base": jnp.zeros((3,), dtype=jnp.float64),
        "message": {
            "probability_endpoint_head": jnp.zeros((2, 3, width)),
            "phase_endpoint_head": jnp.zeros((2, 3, width)),
            "probability_closed_head": jnp.zeros((width,)),
            "phase_closed_head": jnp.zeros((width,)),
            "probability_hidden_head": jnp.zeros((width,)),
            "phase_hidden_head": jnp.zeros((width,)),
        },
    }
    weights = np.asarray([0.08, 0.12, 0.19, 0.23, 0.17, 0.21])
    generator = np.random.default_rng(401)
    scores = jax.tree_util.tree_map(
        lambda parameter: jnp.asarray(
            _weighted_center(
                generator.normal(size=(6,) + parameter.shape)
                + 1j * generator.normal(size=(6,) + parameter.shape),
                weights,
            )
        ),
        params,
    )
    vector = jax.tree_util.tree_map(
        lambda parameter: jnp.asarray(generator.normal(size=parameter.shape)), params
    )
    damping = 7.0e-3
    specs = sparse_causal_head_block_specs(params)
    preconditioner = build_fisher_preconditioner(
        params,
        scores,
        damping=damping,
        block_specs=specs,
        sample_weights=jnp.asarray(weights),
    )
    actual = preconditioner.apply(vector)
    normalized = weights / np.sum(weights)

    base_score = np.asarray(scores["base"])
    base_diagonal = 2.0 * np.sum(
        normalized[:, None] * np.abs(base_score) ** 2, axis=0
    )
    expected_base = np.asarray(vector["base"]) / (base_diagonal + damping)
    np.testing.assert_allclose(actual["base"], expected_base, rtol=3e-12, atol=3e-12)

    for site in range(2):
        for delta_index in range(3):
            block_score = np.concatenate(
                [
                    np.asarray(scores["message"]["probability_endpoint_head"])[
                        :, site, delta_index
                    ],
                    np.asarray(scores["message"]["phase_endpoint_head"])[
                        :, site, delta_index
                    ],
                ],
                axis=-1,
            )
            block_vector = np.concatenate(
                [
                    np.asarray(vector["message"]["probability_endpoint_head"])[
                        site, delta_index
                    ],
                    np.asarray(vector["message"]["phase_endpoint_head"])[
                        site, delta_index
                    ],
                ]
            )
            fisher = 2.0 * np.real(
                block_score.conj().T @ (normalized[:, None] * block_score)
            )
            expected = np.linalg.solve(fisher + damping * np.eye(2 * width), block_vector)
            joined_actual = np.concatenate(
                [
                    np.asarray(actual["message"]["probability_endpoint_head"])[
                        site, delta_index
                    ],
                    np.asarray(actual["message"]["phase_endpoint_head"])[
                        site, delta_index
                    ],
                ]
            )
            np.testing.assert_allclose(joined_actual, expected, rtol=3e-11, atol=3e-11)

    for scalar_name in ("closed", "hidden"):
        probability_name = f"probability_{scalar_name}_head"
        phase_name = f"phase_{scalar_name}_head"
        block_score = np.concatenate(
            [
                np.asarray(scores["message"][probability_name]),
                np.asarray(scores["message"][phase_name]),
            ],
            axis=-1,
        )
        block_vector = np.concatenate(
            [
                np.asarray(vector["message"][probability_name]),
                np.asarray(vector["message"][phase_name]),
            ]
        )
        fisher = 2.0 * np.real(
            block_score.conj().T @ (normalized[:, None] * block_score)
        )
        expected = np.linalg.solve(fisher + damping * np.eye(2 * width), block_vector)
        joined_actual = np.concatenate(
            [
                np.asarray(actual["message"][probability_name]),
                np.asarray(actual["message"][phase_name]),
            ]
        )
        np.testing.assert_allclose(joined_actual, expected, rtol=3e-11, atol=3e-11)

    assert preconditioner.block_sizes == {
        "qop2_endpoint_site_delta": (6, 4),
        "closed_scalar_heads": (1, 4),
        "hidden_scalar_heads": (1, 4),
    }
    # 3 diagonal base coordinates plus six 4x4 carrier matrices and two 4x4
    # scalar matrices; no global 43x43 matrix is stored.
    assert preconditioner.stored_matrix_elements == 3 + 6 * 16 + 2 * 16


def test_zero_and_near_null_scores_are_finite_and_damping_bounded():
    params = {
        "x": jnp.zeros((4,), dtype=jnp.float64),
        "zero": jnp.zeros((2,), dtype=jnp.float64),
    }
    base = np.asarray([-2.0, -1.0, 1.0, 2.0])[:, None]
    scores = {
        "x": jnp.asarray(
            np.concatenate(
                (
                    base,
                    base * (1.0 + 1.0e-13),
                    1.0e-15 * base,
                    -2.0e-15 * base,
                ),
                axis=1,
            ),
            dtype=jnp.complex128,
        ),
        "zero": jnp.zeros((4, 2), dtype=jnp.complex128),
    }
    damping = 2.0e-5
    block = FisherBlockSpec(name="near_null", paths=(("x",),))
    preconditioner = build_fisher_preconditioner(
        params, scores, damping=damping, block_specs=(block,)
    )
    vector = {"x": jnp.asarray([1.0, -0.5, 0.3, -0.1]), "zero": jnp.ones(2)}
    result = preconditioner.apply(vector)

    assert np.all(np.isfinite(np.asarray(result["x"])))
    assert np.all(np.isfinite(np.asarray(result["zero"])))
    np.testing.assert_allclose(result["zero"], np.ones(2) / damping)
    assert np.linalg.norm(np.asarray(result["x"])) <= (
        np.linalg.norm(np.asarray(vector["x"])) / damping * (1.0 + 1e-10)
    )


def test_physical_block_whitening_improves_condition_and_pcg_residual():
    site_count = 2
    width = 3
    sample_count = 64
    params = {
        "probability_endpoint_head": jnp.zeros((site_count, 3, width)),
        "phase_endpoint_head": jnp.zeros((site_count, 3, width)),
    }
    generator = np.random.default_rng(2211)
    projected = generator.normal(size=(sample_count, site_count * 3 * 2 * width))
    projected -= np.mean(projected, axis=0, keepdims=True)
    orthogonal, _ = np.linalg.qr(projected)
    probability_score = np.zeros((sample_count, site_count, 3, width))
    phase_score = np.zeros_like(probability_score)
    block_dimension = 2 * width
    for block_index in range(site_count * 3):
        rotation, _ = np.linalg.qr(generator.normal(size=(block_dimension, block_dimension)))
        eigenvalues = np.geomspace(1.0 + 0.13 * block_index, 1.0e-6, block_dimension)
        transform = (
            rotation
            @ np.diag(np.sqrt(0.5 * sample_count * eigenvalues))
            @ rotation.T
        )
        start = block_index * block_dimension
        local_score = orthogonal[:, start : start + block_dimension] @ transform
        site, delta_index = divmod(block_index, 3)
        probability_score[:, site, delta_index] = local_score[:, :width]
        phase_score[:, site, delta_index] = local_score[:, width:]
    scores = {
        "probability_endpoint_head": jnp.asarray(probability_score),
        "phase_endpoint_head": jnp.asarray(phase_score),
    }
    damping = 1.0e-8
    physical_spec = FisherBlockSpec(
        name="qop2_endpoint_site_delta",
        paths=(("probability_endpoint_head",), ("phase_endpoint_head",)),
        independent_axes=(0, 1),
    )
    physical = build_fisher_preconditioner(
        params, scores, damping=damping, block_specs=(physical_spec,)
    )
    diagonal = build_diagonal_fisher_preconditioner(
        params, scores, damping=damping
    )
    fisher = _dense_matrix_from_scores(scores, np.ones(sample_count))
    system = fisher + damping * np.eye(fisher.shape[0])
    physical_inverse = _dense_preconditioner(physical, params)
    diagonal_inverse = _dense_preconditioner(diagonal, params)

    physical_spectrum = np.linalg.eigvals(physical_inverse @ system).real
    diagonal_spectrum = np.linalg.eigvals(diagonal_inverse @ system).real
    physical_condition = np.max(physical_spectrum) / np.min(physical_spectrum)
    diagonal_condition = np.max(diagonal_spectrum) / np.min(diagonal_spectrum)
    assert physical_condition < 1.0 + 2.0e-8
    assert diagonal_condition > 1.0e4

    rhs = generator.normal(size=system.shape[0])
    _, plain_iterations, plain_residual = _pcg(
        system,
        rhs,
        None,
        tolerance=1.0e-10,
        max_iterations=200,
    )
    _, physical_iterations, physical_residual = _pcg(
        system,
        rhs,
        physical_inverse,
        tolerance=1.0e-10,
        max_iterations=200,
    )
    _, _, plain_one_step_residual = _pcg(
        system,
        rhs,
        None,
        tolerance=0.0,
        max_iterations=1,
    )
    assert physical_iterations <= 2
    assert physical_iterations < plain_iterations
    assert physical_residual < 1.0e-10
    assert plain_residual < 1.0e-10
    assert physical_residual < plain_one_step_residual * 1.0e-6


def test_block_specs_reject_overlap_and_missing_paths():
    params = {"x": jnp.zeros(2)}
    scores = {"x": jnp.zeros((3, 2), dtype=jnp.complex128)}
    repeated = (
        FisherBlockSpec(name="one", paths=(("x",),)),
        FisherBlockSpec(name="two", paths=(("x",),)),
    )
    with pytest.raises(ValueError, match="overlap"):
        build_fisher_preconditioner(
            params, scores, damping=1.0e-3, block_specs=repeated
        )
    with pytest.raises(ValueError, match="missing parameter path"):
        build_fisher_preconditioner(
            params,
            scores,
            damping=1.0e-3,
            block_specs=(FisherBlockSpec(name="bad", paths=(("y",),)),),
        )


def test_n6_chunked_exact_carrier_accumulator_matches_explicit_centered_scores():
    context_width = 3
    params = {
        "probability_endpoint_head": jnp.zeros((6, 3, context_width)),
        "phase_endpoint_head": jnp.zeros((6, 3, context_width)),
        "probability_closed_head": jnp.zeros((5, context_width)),
        "phase_closed_head": jnp.zeros((5, context_width)),
    }
    generator = np.random.default_rng(7719)
    sample_count = 7
    raw_scores = jax.tree_util.tree_map(
        lambda parameter: jnp.asarray(
            generator.normal(size=(sample_count,) + parameter.shape)
            + 1j * generator.normal(size=(sample_count,) + parameter.shape)
        ),
        params,
    )
    weights = np.asarray([0.03, 0.12, 0.21, 0.09, 0.18, 0.14, 0.23])
    normalized = weights / np.sum(weights)
    centered = jax.tree_util.tree_map(
        lambda score: score
        - jnp.sum(
            jnp.asarray(normalized).reshape(
                (sample_count,) + (1,) * (score.ndim - 1)
            )
            * score,
            axis=0,
            keepdims=True,
        ),
        raw_scores,
    )
    specs = exact_carrier_decoder_block_specs(params)
    damping = 2.5e-4
    explicit = build_fisher_preconditioner(
        params,
        centered,
        damping=damping,
        block_specs=specs,
        sample_weights=jnp.asarray(weights),
    )
    boundaries = ((0, 3), (3, 5), (5, 7))
    chunks = (
        (
            jax.tree_util.tree_map(lambda score: score[start:stop], raw_scores),
            jnp.asarray(weights[start:stop]),
        )
        for start, stop in boundaries
    )
    chunked, diagnostics = build_chunked_fisher_preconditioner(
        params, chunks, damping=damping, block_specs=specs
    )
    vector = jax.tree_util.tree_map(
        lambda parameter: jnp.asarray(generator.normal(size=parameter.shape)), params
    )
    explicit_flat, _ = ravel_pytree(explicit.apply(vector))
    chunked_flat, _ = ravel_pytree(chunked.apply(vector))

    np.testing.assert_allclose(chunked_flat, explicit_flat, rtol=2e-11, atol=2e-11)
    parameter_count = int(ravel_pytree(params)[0].size)
    assert diagnostics.sample_count == sample_count
    assert diagnostics.chunk_count == 3
    assert diagnostics.maximum_score_chunk_shape == (3, parameter_count)
    assert diagnostics.maximum_real_imag_jacobian_shape == (3, 2, parameter_count)
    assert diagnostics.maximum_score_chunk_elements == 3 * parameter_count
    assert diagnostics.maximum_score_chunk_bytes == 3 * parameter_count * 16
    assert diagnostics.estimated_peak_score_workspace_bytes == 6 * parameter_count * 16
    assert diagnostics.largest_block_dimension == 2 * context_width
    assert diagnostics.stored_matrix_elements < parameter_count**2
    assert chunked.block_sizes == {
        "exact_qop2_endpoint_site_delta": (18, 2 * context_width),
        "exact_closed_decision": (5, 2 * context_width),
    }
