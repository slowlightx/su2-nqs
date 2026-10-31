"""Matched physical/scrambled gate for invariant RVB covering attention.

The physical arm maps exact space-group bond-relation labels to coefficients
of signed ``Gamma-A1`` covering-orbit columns.  The control reassigns the same
feature rows to different physical orbit columns, preserving parameter count,
optimizer work, exact spin, and exact lattice sector.  Any paired advantage is
therefore attributable to the correspondence between lattice relations and
valence-bond resonance, not to a larger variational space or sector leakage.

Training is performed with exact reduced overlap, Hamiltonian, and ``H^2``
matrices.  This is an exact-small compression test; it is not the future
nonorthogonal covering Monte Carlo estimator.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ..models.rvb_covering_transformer import (
    RVBCoveringTransformerConfig,
    init_rvb_covering_transformer,
    rvb_covering_transformer_coefficients,
    rvb_covering_transformer_parameter_count,
)
from ..vmc.adam import train_exact_adam
from .fusion_spatial_bridge import spin_raising_relative_norm
from .rvb_covering_frontier import (
    RVBFrontierCaseData,
    prepare_rvb_frontier_case,
)
from .rvb_covering_oracle import two_lanczos_state


jax.config.update("jax_enable_x64", True)


ARMS = ("physical_relation_attention", "scrambled_relation_attention")


def _coefficient_tangent_audit(params, nodes, relations, config) -> dict[str, object]:
    """Measure the local real rank of the complex covering coefficients."""

    flat, unravel = ravel_pytree(params)

    def real_outputs(vector):
        values = rvb_covering_transformer_coefficients(
            unravel(vector), nodes, relations, config
        )
        return jnp.concatenate((values.real, values.imag))

    jacobian = np.asarray(jax.jacrev(real_outputs)(flat))
    singular = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = float(singular[0] * 1.0e-10) if singular.size else 0.0
    rank = int(np.count_nonzero(singular > tolerance))
    return {
        "real_rank": rank,
        "maximum_real_rank": int(2 * nodes.shape[0]),
        "relative_rank_tolerance": 1.0e-10,
        "smallest_singular_value": float(singular[-1]),
        "smallest_retained_singular_value": float(singular[rank - 1]),
        "full_rank": rank == 2 * nodes.shape[0],
    }


def _reduced_matrices(data: RVBFrontierCaseData):
    columns = np.column_stack((data.carrier, data.orbit_columns.columns)).astype(
        np.complex128
    )
    actions = np.asarray(data.hamiltonian @ columns)
    overlap = columns.conj().T @ columns
    hamiltonian = columns.conj().T @ actions
    hamiltonian_squared = actions.conj().T @ actions
    return columns, overlap, hamiltonian, hamiltonian_squared


def _metrics(coefficients, overlap, hamiltonian, hamiltonian_squared, n_sites):
    vector = np.asarray(coefficients, dtype=np.complex128)
    norm = float(np.vdot(vector, overlap @ vector).real)
    energy = float(np.vdot(vector, hamiltonian @ vector).real / norm)
    second = float(np.vdot(vector, hamiltonian_squared @ vector).real / norm)
    variance = max(0.0, second - energy * energy)
    return {
        "energy_total": energy,
        "energy_per_site": energy / n_sites,
        "variance_total": variance,
        "variance_per_site": variance / n_sites,
        "coefficient_norm": float(np.linalg.norm(vector[1:])),
    }


def _train_case(
    data: RVBFrontierCaseData,
    *,
    seeds: tuple[int, ...],
    steps: int,
    learning_rate: float,
    embedding_dim: int,
    head_count: int,
    depth: int,
    mlp_ratio: int,
) -> dict[str, object]:
    features = data.relation_features
    config = RVBCoveringTransformerConfig(
        token_count=data.n_sites // 2,
        bond_class_count=features.bond_class_count,
        relation_class_count=features.relation_class_count,
        embedding_dim=embedding_dim,
        head_count=head_count,
        depth=depth,
        mlp_ratio=mlp_ratio,
    )
    nodes = jnp.asarray(features.bond_classes)
    relations = jnp.asarray(features.relation_classes)
    scramble = np.random.default_rng(2026092300 + data.n_sites).permutation(
        nodes.shape[0]
    )
    feature_sets = {
        ARMS[0]: (nodes, relations),
        ARMS[1]: (nodes[scramble], relations[scramble]),
    }
    audit_initial = init_rvb_covering_transformer(
        jax.random.PRNGKey(seeds[0]), config
    )
    tangent_audit = _coefficient_tangent_audit(
        audit_initial, nodes, relations, config
    )
    columns, overlap_np, hamiltonian_np, h2_np = _reduced_matrices(data)
    overlap = jnp.asarray(overlap_np)
    hamiltonian = jnp.asarray(hamiltonian_np)

    def coefficients(params, selected_nodes, selected_relations):
        residual = rvb_covering_transformer_coefficients(
            params, selected_nodes, selected_relations, config
        )
        return jnp.concatenate(
            (jnp.ones((1,), dtype=jnp.complex128), residual)
        )

    runs: dict[str, list[dict[str, object]]] = {arm: [] for arm in ARMS}
    final_states: dict[tuple[str, int], np.ndarray] = {}
    for seed in seeds:
        initial = init_rvb_covering_transformer(
            jax.random.PRNGKey(seed), config
        )
        for arm in ARMS:
            selected_nodes, selected_relations = feature_sets[arm]

            def loss(candidate):
                reduced = coefficients(
                    candidate, selected_nodes, selected_relations
                )
                return (
                    jnp.vdot(reduced, hamiltonian @ reduced).real
                    / jnp.vdot(reduced, overlap @ reduced).real
                )

            initial_energy = float(loss(initial))
            started = time.perf_counter()
            trained, optimizer = train_exact_adam(
                initial,
                loss_fn=loss,
                steps=steps,
                learning_rate=learning_rate,
                record_every=max(1, steps // 5),
            )
            elapsed = time.perf_counter() - started
            reduced = np.asarray(
                coefficients(trained, selected_nodes, selected_relations)
            )
            physical = columns @ reduced
            physical /= np.linalg.norm(physical)
            final_states[(arm, seed)] = physical
            metrics = _metrics(
                reduced,
                overlap_np,
                hamiltonian_np,
                h2_np,
                data.n_sites,
            )
            runs[arm].append(
                {
                    "seed": int(seed),
                    "initial_energy_total": initial_energy,
                    "wall_seconds": elapsed,
                    "steps": steps,
                    "parameter_count": rvb_covering_transformer_parameter_count(
                        trained
                    ),
                    "optimizer": optimizer,
                    "metrics": metrics,
                    "spin_raising_relative_norm": spin_raising_relative_norm(
                        physical, data.configurations
                    ),
                }
            )

    two_lanczos = two_lanczos_state(data.carrier, data.hamiltonian)
    summary: dict[str, object] = {}
    for arm in ARMS:
        energies = np.asarray(
            [row["metrics"]["energy_per_site"] for row in runs[arm]]
        )
        variances = np.asarray(
            [row["metrics"]["variance_total"] for row in runs[arm]]
        )
        summary[arm] = {
            "mean_energy_per_site": float(np.mean(energies)),
            "standard_error_energy_per_site": float(
                np.std(energies, ddof=1) / np.sqrt(len(energies))
                if len(energies) > 1
                else 0.0
            ),
            "median_energy_per_site": float(np.median(energies)),
            "median_variance_total": float(np.median(variances)),
            "seeds_below_two_lanczos": int(
                np.count_nonzero(energies < two_lanczos.energy / data.n_sites)
            ),
        }
    paired = np.asarray(
        [
            physical["metrics"]["energy_per_site"]
            - scrambled["metrics"]["energy_per_site"]
            for physical, scrambled in zip(
                runs[ARMS[0]], runs[ARMS[1]], strict=True
            )
        ]
    )
    paired_variance = np.asarray(
        [
            physical["metrics"]["variance_total"]
            - scrambled["metrics"]["variance_total"]
            for physical, scrambled in zip(
                runs[ARMS[0]], runs[ARMS[1]], strict=True
            )
        ]
    )
    return {
        "case": data.name,
        "n_sites": data.n_sites,
        "covering_orbit_count": int(nodes.shape[0]),
        "bond_class_count": features.bond_class_count,
        "relation_class_count": features.relation_class_count,
        "feature_scramble_permutation": scramble.tolist(),
        "config": {
            "embedding_dim": embedding_dim,
            "head_count": head_count,
            "depth": depth,
            "mlp_ratio": mlp_ratio,
        },
        "coefficient_tangent_audit": tangent_audit,
        "two_lanczos": {
            "energy_per_site": two_lanczos.energy / data.n_sites,
            "variance_total": two_lanczos.variance,
        },
        "runs": runs,
        "summary": summary,
        "paired_physical_minus_scrambled_per_site": paired.tolist(),
        "paired_mean_per_site": float(np.mean(paired)),
        "paired_standard_error_per_site": float(
            np.std(paired, ddof=1) / np.sqrt(len(paired))
            if len(paired) > 1
            else 0.0
        ),
        "paired_variance_physical_minus_scrambled": paired_variance.tolist(),
        "paired_mean_variance": float(np.mean(paired_variance)),
        "paired_standard_error_variance": float(
            np.std(paired_variance, ddof=1) / np.sqrt(len(paired_variance))
            if len(paired_variance) > 1
            else 0.0
        ),
        "physical_wins_every_pair": bool(np.all(paired < 0.0)),
        "physical_wins_variance_every_pair": bool(
            np.all(paired_variance < 0.0)
        ),
    }


def rvb_covering_attention_frontier(
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 500,
    learning_rate: float = 3.0e-3,
    embedding_dim: int = 4,
    head_count: int = 1,
    depth: int = 2,
    mlp_ratio: int = 2,
) -> dict[str, object]:
    """Train matched physical and scrambled exact-small covering models."""

    if not seeds:
        raise ValueError("at least one paired seed is required")

    triangular = _train_case(
        prepare_rvb_frontier_case("triangular_n12_j2_0.125"),
        seeds=seeds,
        steps=steps,
        learning_rate=learning_rate,
        embedding_dim=embedding_dim,
        head_count=head_count,
        depth=depth,
        mlp_ratio=mlp_ratio,
    )
    square = _train_case(
        prepare_rvb_frontier_case("square_n16_j2_0.5"),
        seeds=seeds,
        steps=steps,
        learning_rate=learning_rate,
        embedding_dim=embedding_dim,
        head_count=head_count,
        depth=depth,
        mlp_ratio=mlp_ratio,
    )
    maximum_spin_residual = max(
        row["spin_raising_relative_norm"]
        for case in (triangular, square)
        for arm in ARMS
        for row in case["runs"][arm]
    )
    return {
        "architecture": {
            "name": "space_group_invariant_rvb_covering_transformer",
            "exact_su2": True,
            "exact_gamma_a1": True,
            "magnetic_indices_visible": False,
            "forward_reynolds_sum_required": False,
            "control": "paired orbit-feature reassignment scramble",
        },
        "triangular_n12": triangular,
        "square_n16": square,
        "maximum_spin_raising_relative_norm": float(maximum_spin_residual),
        "exact_spin_gate_passed": maximum_spin_residual < 1.0e-10,
        "full_coefficient_tangent_rank_both_cases": all(
            case["coefficient_tangent_audit"]["full_rank"]
            for case in (triangular, square)
        ),
        "physical_beats_scramble_both_cases": bool(
            triangular["physical_wins_every_pair"]
            and square["physical_wins_every_pair"]
        ),
        "physical_beats_two_lanczos_all_seeds_both_cases": all(
            case["summary"][ARMS[0]]["seeds_below_two_lanczos"] == len(seeds)
            for case in (triangular, square)
        ),
        "claim_boundary": (
            "This exact-small test measures compression and a causal spatial-label "
            "control. It does not validate nonorthogonal covering sampling, N=36 "
            "throughput, or QSL physics."
        ),
    }


__all__ = ["ARMS", "rvb_covering_attention_frontier"]
