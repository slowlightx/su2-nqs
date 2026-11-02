"""Fair small-chain benchmark of exact-SU(2) and fixed-Sz GRUs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..models.fusion_gru import FusionGRUConfig, init_fusion_gru, log_psi, sample
from ..models.sz_gru import SzGRUConfig, init_sz_gru, sample_sz, sz_log_psi
from ..operators.heisenberg_chain import HeisenbergChain
from ..operators.sz_heisenberg import SzHeisenbergChain
from ..su2.fusion_path import enumerate_paths
from ..su2.sz_basis import enumerate_sz_configs
from ..vmc.adam import train_adam
from .comparison import WavefunctionMetrics, wavefunction_metrics
from .coupled_basis import coupled_basis_matrix
from .exact_diagonalization import heisenberg_sz_matrix, spin_squared_matrix
from .model_evaluation import parameter_count


@dataclass(frozen=True)
class TrainedModelResult:
    metrics: WavefunctionMetrics
    history: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class GRUComparison:
    exact_energy: float
    fusion: TrainedModelResult
    sz: TrainedModelResult


def _full_sz_vector(
    amplitudes: np.ndarray,
    configurations: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    n = len(configurations[0])
    vector = np.zeros(2**n, dtype=np.complex128)
    for configuration, amplitude in zip(configurations, amplitudes):
        index = sum(bit << (n - site - 1) for site, bit in enumerate(configuration))
        vector[index] = amplitude
    return vector


def compare_grus(
    n: int,
    *,
    steps: int = 200,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    embedding_dim: int = 16,
    hidden_dim: int = 32,
    seed: int = 0,
) -> GRUComparison:
    """Train parameter-matched models and compute exact post-training metrics."""

    if n % 2:
        raise ValueError("the singlet versus Sz=0 comparison requires even N")
    root_key = jax.random.PRNGKey(seed)
    fusion_init_key, fusion_train_key, sz_init_key, sz_train_key = jax.random.split(root_key, 4)

    fusion_config = FusionGRUConfig(n, 0, embedding_dim, hidden_dim)
    fusion_params = init_fusion_gru(fusion_init_key, fusion_config)
    fusion_operator = HeisenbergChain(n, 0)
    fusion_log = lambda params, path: log_psi(params, path, fusion_config)
    fusion_sample = lambda params, key, count: sample(params, key, count, fusion_config)
    fusion_params, fusion_history = train_adam(
        fusion_params,
        fusion_train_key,
        sample_fn=fusion_sample,
        log_psi_fn=fusion_log,
        get_conn_fn=fusion_operator.get_conn,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    sz_config = SzGRUConfig(n, n // 2, embedding_dim, hidden_dim)
    sz_params = init_sz_gru(sz_init_key, sz_config)
    sz_operator = SzHeisenbergChain(n, n // 2)
    sz_log = lambda params, bits: sz_log_psi(params, bits, sz_config)
    sz_sample_function = lambda params, key, count: sample_sz(params, key, count, sz_config)
    sz_params, sz_history = train_adam(
        sz_params,
        sz_train_key,
        sample_fn=sz_sample_function,
        log_psi_fn=sz_log,
        get_conn_fn=sz_operator.get_conn,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    hamiltonian, _ = heisenberg_sz_matrix(n)
    spin_squared, _ = spin_squared_matrix(n)

    fusion_paths = enumerate_paths(n, 0)
    fusion_logs = jax.vmap(lambda path: fusion_log(fusion_params, path))(
        jnp.asarray(fusion_paths)
    )
    fusion_vector = coupled_basis_matrix(fusion_paths, 0, 0) @ np.asarray(jnp.exp(fusion_logs))
    fusion_metrics = wavefunction_metrics(
        fusion_vector,
        hamiltonian,
        spin_squared,
        parameter_count=parameter_count(fusion_params),
    )

    sz_configurations = enumerate_sz_configs(n, n // 2)
    sz_logs = jax.vmap(lambda bits: sz_log(sz_params, bits))(jnp.asarray(sz_configurations))
    sz_vector = _full_sz_vector(np.asarray(jnp.exp(sz_logs)), sz_configurations)
    sz_metrics = wavefunction_metrics(
        sz_vector,
        hamiltonian,
        spin_squared,
        parameter_count=parameter_count(sz_params),
    )
    exact_energy = float(np.linalg.eigvalsh(fusion_operator.to_dense())[0])
    return GRUComparison(
        exact_energy,
        TrainedModelResult(fusion_metrics, tuple(fusion_history)),
        TrainedModelResult(sz_metrics, tuple(sz_history)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    result = compare_grus(
        arguments.n,
        steps=arguments.steps,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
    )
    print(f"exact energy: {result.exact_energy:.12f}")
    for name, trained in (("fusion", result.fusion), ("Sz", result.sz)):
        metrics = trained.metrics
        print(
            f"{name}: E={metrics.energy:.12f}, var={metrics.variance:.6e}, "
            f"<S^2>={metrics.spin_squared:.6e}, params={metrics.parameter_count}"
        )


if __name__ == "__main__":
    main()

