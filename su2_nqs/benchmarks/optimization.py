"""Controlled optimization benchmarks in exhaustively enumerable sectors.

These routines use the complete small-system basis to evaluate final metrics.
They support both deterministic exact-objective optimization and statistically
equivalent iid VMC updates drawn from the model distribution.  The exhaustive
path is a benchmark instrument, not the scalable production sampler.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg

from ..models.fusion_gru import FusionGRUConfig, init_fusion_gru, log_psi
from ..models.full_hilbert_gru import (
    FullHilbertGRUConfig,
    full_hilbert_log_psi,
    init_full_hilbert_gru,
)
from ..models.hubbard_fusion_gru import (
    HubbardFusionGRUConfig,
    hubbard_log_psi,
    init_hubbard_fusion_gru,
)
from ..models.sz_gru import SzGRUConfig, init_sz_gru, sz_log_psi
from ..models.symmetry_blind_gru import (
    SymmetryBlindGRUConfig,
    init_symmetry_blind_gru,
    symmetry_blind_log_psi,
)
from ..models.token_position_gru import (
    TokenPositionGRUConfig,
    init_token_position_gru,
    token_position_log_psi,
)
from ..operators.heisenberg_chain import HeisenbergChain
from ..operators.hubbard import FockBond, HubbardHamiltonian
from ..operators.sz_heisenberg import SzHeisenbergChain
from ..su2.fusion_path import enumerate_paths
from ..su2.sz_basis import enumerate_sz_configs
from .exact_diagonalization import computational_basis, heisenberg_sz_matrix, spin_squared_matrix
from .model_evaluation import parameter_count

ModelKind = Literal["blind", "fusion", "full", "hubbard", "sz", "tokenpos"]
ExactFisherMetric = Literal["born", "hamiltonian_flow"]


@dataclass(frozen=True)
class ExactGRUProblem:
    """A small fixed-sector model, basis, Hamiltonian, and spin observable."""

    n: int
    model: ModelKind
    config: (
        FusionGRUConfig
        | FullHilbertGRUConfig
        | HubbardFusionGRUConfig
        | SzGRUConfig
        | SymmetryBlindGRUConfig
        | TokenPositionGRUConfig
    )
    basis: jax.Array
    hamiltonian: jax.Array
    spin_squared: jax.Array
    exact_energy: float

    def init(self, key: jax.Array) -> object:
        if self.model == "fusion":
            return init_fusion_gru(key, self.config)
        if self.model == "full":
            return init_full_hilbert_gru(key, self.config)
        if self.model == "blind":
            return init_symmetry_blind_gru(key, self.config)
        if self.model == "tokenpos":
            return init_token_position_gru(key, self.config)
        if self.model == "hubbard":
            return init_hubbard_fusion_gru(key, self.config)
        if self.model == "sz":
            return init_sz_gru(key, self.config)
        raise ValueError(f"unknown model kind: {self.model!r}")

    def log_amplitude(self, params: object, state: jax.Array) -> jax.Array:
        if self.model == "fusion":
            return log_psi(params, state, self.config)
        if self.model == "full":
            return full_hilbert_log_psi(params, state, self.config)
        if self.model == "blind":
            return symmetry_blind_log_psi(params, state, self.config)
        if self.model == "tokenpos":
            return token_position_log_psi(params, state, self.config)
        if self.model == "hubbard":
            return hubbard_log_psi(params, state, self.config)
        if self.model == "sz":
            return sz_log_psi(params, state, self.config)
        raise ValueError(f"unknown model kind: {self.model!r}")


def _configuration_integer(configuration: tuple[int, ...]) -> int:
    n = len(configuration)
    return sum(bit << (n - site - 1) for site, bit in enumerate(configuration))


def build_exact_problem(
    n: int,
    model: ModelKind,
    *,
    embedding_dim: int,
    hidden_dim: int,
) -> ExactGRUProblem:
    """Build a deterministic benchmark problem with matched GRU dimensions."""

    if n % 2:
        raise ValueError("the S=0 versus Sz=0 benchmark requires even N")
    if model == "fusion":
        config = FusionGRUConfig(n, 0, embedding_dim, hidden_dim)
        basis_values = enumerate_paths(n, 0)
        hamiltonian = HeisenbergChain(n, 0).to_dense()
        spin_operator = np.zeros_like(hamiltonian)
    elif model == "sz":
        config = SzGRUConfig(n, n // 2, embedding_dim, hidden_dim)
        basis_values = enumerate_sz_configs(n, n // 2)
        hamiltonian = SzHeisenbergChain(n, n // 2).to_dense()
        sector_spin, integer_basis = spin_squared_matrix(n, n_up=n // 2)
        positions = {state: index for index, state in enumerate(integer_basis)}
        order = [positions[_configuration_integer(configuration)] for configuration in basis_values]
        spin_operator = sector_spin[np.ix_(order, order)]
    elif model in ("blind", "full", "tokenpos"):
        if model == "blind":
            config = SymmetryBlindGRUConfig(n, embedding_dim, hidden_dim)
        elif model == "tokenpos":
            config = TokenPositionGRUConfig(n, embedding_dim, hidden_dim)
        else:
            config = FullHilbertGRUConfig(n, embedding_dim, hidden_dim)
        integer_basis = computational_basis(n)
        basis_values = tuple(
            tuple((state >> (n - site - 1)) & 1 for site in range(n))
            for state in integer_basis
        )
        hamiltonian, hamiltonian_basis = heisenberg_sz_matrix(n)
        if hamiltonian_basis != integer_basis:
            raise RuntimeError("full-basis Hamiltonian ordering mismatch")
        spin_operator, spin_basis = spin_squared_matrix(n)
        if spin_basis != integer_basis:
            raise RuntimeError("full-basis spin ordering mismatch")
    else:
        raise ValueError(f"unknown model kind: {model!r}")
    return ExactGRUProblem(
        n=n,
        model=model,
        config=config,
        basis=jnp.asarray(basis_values, dtype=jnp.int32),
        hamiltonian=jnp.asarray(hamiltonian, dtype=jnp.float64),
        spin_squared=jnp.asarray(spin_operator, dtype=jnp.float64),
        exact_energy=float(np.linalg.eigvalsh(hamiltonian)[0]),
    )


def build_exact_hubbard_problem(
    n_sites: int,
    n_electrons: int,
    target_q: int,
    bonds: tuple[FockBond, ...],
    *,
    interaction: float,
    embedding_dim: int,
    hidden_dim: int,
    compiler: Literal["direct_cg", "reduced_spinor"] = "direct_cg",
    exact_energy: float | None = None,
) -> ExactGRUProblem:
    """Build an exhaustively enumerable Hubbard fusion-GRU benchmark.

    A trusted externally computed ``exact_energy`` avoids a redundant dense
    diagonalization when a sparse eigensolver has already passed its residual
    gate.  Omitting it retains the independent dense small-system reference.
    """

    config = HubbardFusionGRUConfig(
        n_sites=n_sites,
        n_electrons=n_electrons,
        target_q=target_q,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    operator = HubbardHamiltonian(
        n_sites=n_sites,
        n_electrons=n_electrons,
        target_q=target_q,
        bonds=bonds,
        interaction=interaction,
        compiler=compiler,
    )
    hamiltonian = operator.to_dense()
    return ExactGRUProblem(
        n=n_sites,
        model="hubbard",
        config=config,
        basis=jnp.asarray(operator.basis, dtype=jnp.int32),
        hamiltonian=jnp.asarray(hamiltonian, dtype=jnp.float64),
        spin_squared=jnp.zeros_like(jnp.asarray(hamiltonian, dtype=jnp.float64)),
        exact_energy=(
            float(np.linalg.eigvalsh(hamiltonian)[0])
            if exact_energy is None
            else float(exact_energy)
        ),
    )


def _log_vector(problem: ExactGRUProblem, params: object) -> jax.Array:
    return jax.vmap(lambda state: problem.log_amplitude(params, state))(problem.basis)


def _observables(problem: ExactGRUProblem, params: object) -> tuple[jax.Array, ...]:
    amplitudes = jnp.exp(_log_vector(problem, params))
    norm = jnp.real(jnp.vdot(amplitudes, amplitudes))
    h_amplitudes = problem.hamiltonian @ amplitudes
    energy = jnp.real(jnp.vdot(amplitudes, h_amplitudes)) / norm
    energy_squared = jnp.real(jnp.vdot(h_amplitudes, h_amplitudes)) / norm
    variance = jnp.maximum(energy_squared - energy**2, 0.0)
    spin = jnp.real(jnp.vdot(amplitudes, problem.spin_squared @ amplitudes)) / norm
    return energy, variance, spin, norm


def evaluate_exact(problem: ExactGRUProblem, params: object) -> dict[str, float | int]:
    """Exhaustive normalized energy, variance, and spin contamination."""

    energy, variance, spin, norm = _observables(problem, params)
    return {
        "energy": float(energy),
        "energy_error": float(energy) - problem.exact_energy,
        "variance": float(variance),
        "spin_squared": float(spin),
        "normalization": float(norm),
        "parameter_count": parameter_count(params),
    }


def _first_convergence_step(
    history: list[dict[str, float]], threshold: float
) -> int | None:
    for item in history:
        if item["energy_error"] <= threshold:
            return int(item["step"])
    return None


def _with_convergence(history: list[dict[str, float]]) -> dict[str, int | None]:
    return {
        "1e-2": _first_convergence_step(history, 1e-2),
        "1e-3": _first_convergence_step(history, 1e-3),
        "1e-4": _first_convergence_step(history, 1e-4),
    }


def train_exact_adam(
    problem: ExactGRUProblem,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    record_every: int = 5,
) -> tuple[object, dict[str, object]]:
    """Optimize the exact variational energy with Adam."""

    params = problem.init(jax.random.PRNGKey(seed))
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)

    def objective(candidate: object) -> jax.Array:
        return _observables(problem, candidate)[0]

    value_and_gradient = jax.value_and_grad(objective)

    @jax.jit
    def update(candidate, first_moment, second_moment, step):
        energy, gradient = value_and_gradient(candidate)
        first_moment = jax.tree.map(
            lambda moment, value: 0.9 * moment + 0.1 * value,
            first_moment,
            gradient,
        )
        second_moment = jax.tree.map(
            lambda moment, value: 0.999 * moment + 0.001 * value * value,
            second_moment,
            gradient,
        )
        first_hat = jax.tree.map(
            lambda value: value / (1.0 - 0.9**step), first_moment
        )
        second_hat = jax.tree.map(
            lambda value: value / (1.0 - 0.999**step), second_moment
        )
        candidate = jax.tree.map(
            lambda value, mean, square: value
            - learning_rate * mean / (jnp.sqrt(square) + 1e-8),
            candidate,
            first_hat,
            second_hat,
        )
        gradient_norm = jnp.sqrt(
            sum(jnp.sum(value * value) for value in jax.tree.leaves(gradient))
        )
        return candidate, first_moment, second_moment, energy, gradient_norm

    started = time.perf_counter()
    initial = evaluate_exact(problem, params)
    history = [{"step": 0.0, **{key: float(value) for key, value in initial.items() if key != "parameter_count"}}]
    for step in range(1, steps + 1):
        params, first, second, _, gradient_norm = update(
            params, first, second, jnp.asarray(step, dtype=jnp.float64)
        )
        if step % record_every == 0 or step == steps:
            metrics = evaluate_exact(problem, params)
            history.append(
                {
                    "step": float(step),
                    **{key: float(value) for key, value in metrics.items() if key != "parameter_count"},
                    "gradient_norm": float(gradient_norm),
                }
            )
    elapsed = time.perf_counter() - started
    final = evaluate_exact(problem, params)
    report: dict[str, object] = {
        "optimizer": "exact_adam",
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "elapsed_seconds": elapsed,
        "final": final,
        "convergence_steps": _with_convergence(history),
        "history": history,
    }
    return params, report


def train_sampled_adam(
    problem: ExactGRUProblem,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    batch_size: int,
    record_every: int = 5,
) -> tuple[object, dict[str, object]]:
    """Adam with iid samples and dense-reference local energies.

    Sampling from the exhaustively evaluated probability vector is
    statistically equivalent to exact autoregressive sampling and permits a
    fast, controlled equal-sample-budget study.
    """

    root_key = jax.random.PRNGKey(seed)
    init_key, train_key = jax.random.split(root_key)
    params = problem.init(init_key)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)

    def sampled_loss(candidate: object, key: jax.Array):
        logs = _log_vector(problem, candidate)
        amplitudes = jnp.exp(logs)
        probabilities = jnp.abs(amplitudes) ** 2
        probabilities = probabilities / jnp.sum(probabilities)
        indices = jax.random.choice(
            key, probabilities.size, shape=(batch_size,), p=probabilities
        )
        local = (problem.hamiltonian @ amplitudes) / amplitudes
        sampled_local = local[indices]
        mean = jnp.mean(sampled_local)
        centered = jax.lax.stop_gradient(sampled_local - mean)
        surrogate = 2.0 * jnp.real(jnp.mean(jnp.conj(logs[indices]) * centered))
        loss = (
            jax.lax.stop_gradient(jnp.real(mean))
            + surrogate
            - jax.lax.stop_gradient(surrogate)
        )
        variance = jnp.mean(jnp.abs(sampled_local - mean) ** 2)
        return loss, (jnp.real(mean), jnp.real(variance))

    value_and_gradient = jax.value_and_grad(sampled_loss, has_aux=True)

    @jax.jit
    def update(candidate, first_moment, second_moment, key, step):
        key, sample_key = jax.random.split(key)
        (_, (batch_energy, batch_variance)), gradient = value_and_gradient(
            candidate, sample_key
        )
        first_moment = jax.tree.map(
            lambda moment, value: 0.9 * moment + 0.1 * value,
            first_moment,
            gradient,
        )
        second_moment = jax.tree.map(
            lambda moment, value: 0.999 * moment + 0.001 * value * value,
            second_moment,
            gradient,
        )
        first_hat = jax.tree.map(
            lambda value: value / (1.0 - 0.9**step), first_moment
        )
        second_hat = jax.tree.map(
            lambda value: value / (1.0 - 0.999**step), second_moment
        )
        candidate = jax.tree.map(
            lambda value, mean, square: value
            - learning_rate * mean / (jnp.sqrt(square) + 1e-8),
            candidate,
            first_hat,
            second_hat,
        )
        return (
            candidate,
            first_moment,
            second_moment,
            key,
            batch_energy,
            batch_variance,
        )

    started = time.perf_counter()
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        params, first, second, train_key, batch_energy, batch_variance = update(
            params,
            first,
            second,
            train_key,
            jnp.asarray(step, dtype=jnp.float64),
        )
        if step % record_every == 0 or step == steps:
            metrics = evaluate_exact(problem, params)
            history.append(
                {
                    "step": float(step),
                    **{key: float(value) for key, value in metrics.items() if key != "parameter_count"},
                    "batch_energy": float(batch_energy),
                    "batch_variance": float(batch_variance),
                }
            )
    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "optimizer": "sampled_adam",
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "sample_budget": steps * batch_size,
        "elapsed_seconds": elapsed,
        "final": evaluate_exact(problem, params),
        "convergence_steps": _with_convergence(history),
        "history": history,
    }
    return params, report


def gradient_noise_statistics(
    problem: ExactGRUProblem,
    params: object,
    *,
    batch_size: int,
    batches: int,
    seed: int,
) -> dict[str, float | int]:
    """Trace variance and signal-to-noise ratio of the VMC gradient."""

    flat, unravel = ravel_pytree(params)

    def logs_from_flat(values):
        return _log_vector(problem, unravel(values))

    logs = logs_from_flat(flat)
    amplitudes = jnp.exp(logs)
    probabilities = jnp.abs(amplitudes) ** 2
    probabilities = probabilities / jnp.sum(probabilities)
    local = (problem.hamiltonian @ amplitudes) / amplitudes

    def real_imag_logs(values):
        values_log = logs_from_flat(values)
        return jnp.concatenate((jnp.real(values_log), jnp.imag(values_log)))

    jacobian = jax.jacrev(real_imag_logs)(flat)
    dimension = problem.basis.shape[0]
    log_derivatives = jacobian[:dimension] + 1j * jacobian[dimension:]

    def energy_from_flat(values):
        candidate = jnp.exp(logs_from_flat(values))
        return jnp.real(jnp.vdot(candidate, problem.hamiltonian @ candidate)) / jnp.real(
            jnp.vdot(candidate, candidate)
        )

    exact_gradient = jax.grad(energy_from_flat)(flat)
    indices = jax.random.choice(
        jax.random.PRNGKey(seed),
        probabilities.size,
        shape=(batches, batch_size),
        p=probabilities,
    )
    sampled_local = local[indices]
    centered = sampled_local - jnp.mean(sampled_local, axis=1, keepdims=True)
    sampled_derivatives = log_derivatives[indices]
    gradient_samples = 2.0 * jnp.real(
        jnp.mean(jnp.conj(sampled_derivatives) * centered[..., None], axis=1)
    )
    deviations = gradient_samples - exact_gradient[None, :]
    trace_variance = jnp.mean(jnp.sum(deviations**2, axis=1))
    signal_squared = jnp.sum(exact_gradient**2)
    mean_gradient = jnp.mean(gradient_samples, axis=0)
    bias_squared = jnp.sum((mean_gradient - exact_gradient) ** 2)
    return {
        "batch_size": batch_size,
        "batches": batches,
        "parameter_count": int(flat.size),
        "exact_gradient_norm": float(jnp.sqrt(signal_squared)),
        "trace_variance": float(trace_variance),
        "rms_gradient_noise": float(jnp.sqrt(trace_variance)),
        "signal_to_noise": float(
            jnp.sqrt(signal_squared) / (jnp.sqrt(trace_variance) + 1e-30)
        ),
        "relative_trace_variance": float(trace_variance / (signal_squared + 1e-30)),
        "mean_gradient_bias_norm": float(jnp.sqrt(bias_squared)),
    }


def train_exact_sr(
    problem: ExactGRUProblem,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    damping: float,
    cg_tolerance: float = 1e-6,
    cg_max_iterations: int = 50,
    record_every: int = 5,
    initial_params: object | None = None,
    target_state: np.ndarray | jax.Array | None = None,
    fidelity_threshold: float | None = None,
    fisher_metric: ExactFisherMetric = "born",
) -> tuple[object, dict[str, object]]:
    """Exact stochastic reconfiguration with a matrix-free Fisher product.

    ``fisher_metric="born"`` is conventional SR.  The experimental
    ``"hamiltonian_flow"`` option keeps the exact energy gradient but builds
    the Fisher metric from the equal mixture of the Born distribution and the
    normalized Schrödinger residual ``|(H-E)psi|^2``.  It therefore emphasizes
    configurations that carry the current imaginary-time flow without using
    the exact ground state as a teacher.

    When ``target_state`` and ``fidelity_threshold`` are supplied, training is
    checked every ``record_every`` updates and stops at the first recorded
    fidelity crossing.  The target must use the same deterministic basis order
    as ``problem.basis``.
    """

    if steps < 1 or record_every < 1:
        raise ValueError("steps and record_every must be positive")
    if fisher_metric not in ("born", "hamiltonian_flow"):
        raise ValueError(f"unknown exact Fisher metric: {fisher_metric!r}")
    if fidelity_threshold is not None and target_state is None:
        raise ValueError("fidelity_threshold requires target_state")
    if fidelity_threshold is not None and not 0.0 < fidelity_threshold <= 1.0:
        raise ValueError("fidelity_threshold must lie in (0, 1]")
    target_vector = None
    if target_state is not None:
        target_vector = np.asarray(target_state, dtype=np.complex128)
        if target_vector.shape != (int(problem.basis.shape[0]),):
            raise ValueError("target_state shape does not match the problem basis")
        target_norm = np.linalg.norm(target_vector)
        if not np.isfinite(target_norm) or target_norm == 0.0:
            raise ValueError("target_state must have finite nonzero norm")
        target_vector = target_vector / target_norm

    params = (
        problem.init(jax.random.PRNGKey(seed))
        if initial_params is None
        else initial_params
    )
    flat, unravel = ravel_pytree(params)

    def log_values(values):
        return _log_vector(problem, unravel(values))

    def energy(values):
        amplitudes = jnp.exp(log_values(values))
        return jnp.real(jnp.vdot(amplitudes, problem.hamiltonian @ amplitudes)) / jnp.real(
            jnp.vdot(amplitudes, amplitudes)
        )

    @jax.jit
    def update(values):
        logs = log_values(values)
        amplitudes = jnp.exp(logs)
        norm = jnp.real(jnp.vdot(amplitudes, amplitudes))
        born_probabilities = jnp.abs(amplitudes) ** 2 / norm
        if fisher_metric == "hamiltonian_flow":
            h_amplitudes = problem.hamiltonian @ amplitudes
            current_energy = jnp.real(jnp.vdot(amplitudes, h_amplitudes)) / norm
            residual_weights = jnp.abs(
                h_amplitudes - current_energy * amplitudes
            ) ** 2 / norm
            variance = jnp.sum(residual_weights)
            safe_variance = jnp.maximum(variance, 1.0e-14)
            residual_probabilities = residual_weights / safe_variance
            flow_probabilities = 0.5 * (
                born_probabilities + residual_probabilities
            )
            metric_probabilities = jnp.where(
                variance > 1.0e-14,
                flow_probabilities / jnp.sum(flow_probabilities),
                born_probabilities,
            )
        else:
            metric_probabilities = born_probabilities
        metric_probabilities = jax.lax.stop_gradient(metric_probabilities)
        born_probabilities = jax.lax.stop_gradient(born_probabilities)

        def features(candidate):
            candidate_logs = log_values(candidate)
            centered = candidate_logs - jnp.sum(
                metric_probabilities * candidate_logs
            )
            weights = jnp.sqrt(2.0 * metric_probabilities)
            return jnp.concatenate(
                (weights * jnp.real(centered), weights * jnp.imag(centered))
            )

        _, pullback = jax.vjp(features, values)

        def fisher_product(vector):
            _, tangent = jax.jvp(features, (values,), (vector,))
            return pullback(tangent)[0] + damping * vector

        gradient = jax.grad(energy)(values)
        direction, _ = cg(
            fisher_product,
            gradient,
            tol=cg_tolerance,
            maxiter=cg_max_iterations,
        )
        support_size = metric_probabilities.size
        metric_ess_fraction = 1.0 / (
            support_size * jnp.sum(metric_probabilities**2)
        )
        metric_total_variation = 0.5 * jnp.sum(
            jnp.abs(metric_probabilities - born_probabilities)
        )
        return (
            values - learning_rate * direction,
            energy(values),
            jnp.linalg.norm(gradient),
            jnp.linalg.norm(direction),
            metric_ess_fraction,
            metric_total_variation,
        )

    started = time.perf_counter()
    history: list[dict[str, float]] = []
    executed_steps = 0
    for step in range(1, steps + 1):
        (
            flat,
            _,
            gradient_norm,
            direction_norm,
            metric_ess_fraction,
            metric_total_variation,
        ) = update(flat)
        executed_steps = step
        if step % record_every == 0 or step == steps:
            params = unravel(flat)
            metrics = evaluate_exact(problem, params)
            item = {
                "step": float(step),
                **{
                    key: float(value)
                    for key, value in metrics.items()
                    if key != "parameter_count"
                },
                "gradient_norm": float(gradient_norm),
                "natural_direction_norm": float(direction_norm),
                "fisher_probability_ess_fraction": float(
                    metric_ess_fraction
                ),
                "fisher_born_total_variation": float(
                    metric_total_variation
                ),
            }
            if target_vector is not None:
                amplitudes = np.exp(
                    np.asarray(_log_vector(problem, params), dtype=np.complex128)
                )
                amplitudes /= np.linalg.norm(amplitudes)
                item["target_fidelity"] = float(
                    abs(np.vdot(target_vector, amplitudes)) ** 2
                )
            history.append(item)
            if (
                fidelity_threshold is not None
                and item["target_fidelity"] >= fidelity_threshold
            ):
                break
    params = unravel(flat)
    final = evaluate_exact(problem, params)
    if target_vector is not None:
        amplitudes = np.exp(
            np.asarray(_log_vector(problem, params), dtype=np.complex128)
        )
        amplitudes /= np.linalg.norm(amplitudes)
        final["target_fidelity"] = float(
            abs(np.vdot(target_vector, amplitudes)) ** 2
        )
    report: dict[str, object] = {
        "optimizer": "exact_sr",
        "fisher_metric": fisher_metric,
        "warm_started": initial_params is not None,
        "seed": seed,
        "steps": executed_steps,
        "requested_steps": steps,
        "stopped_early": executed_steps < steps,
        "learning_rate": learning_rate,
        "damping": damping,
        "cg_tolerance": cg_tolerance,
        "cg_max_iterations": cg_max_iterations,
        "elapsed_seconds": time.perf_counter() - started,
        "final": final,
        "convergence_steps": _with_convergence(history),
        "history": history,
    }
    if target_vector is not None:
        report["fidelity_threshold"] = fidelity_threshold
        report["fidelity_threshold_reached"] = bool(
            fidelity_threshold is not None
            and final["target_fidelity"] >= fidelity_threshold
        )
        report["fidelity_convergence_step"] = next(
            (
                int(item["step"])
                for item in history
                if fidelity_threshold is not None
                and item["target_fidelity"] >= fidelity_threshold
            ),
            None,
        )
    return params, report
