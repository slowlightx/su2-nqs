"""Exact-small physics gate for configuration-dependent Schur Spatial-ViT.

The calculations in this file are architecture validation, not production
evidence for a variational advantage.  The local Schur configurations span
the complete singlet space at N=6 and N=8, so reaching ED verifies that the
neural coefficient map and the fixed CG reconstruction have no hidden loss
of capacity.  A larger sampled calculation is still required for a physics
claim.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.j1j2 import square_j1j2_bonds
from ..lattices.spatial_symmetry import rectangular_gamma_a1_group
from ..lattices.triangular_j1j2 import triangular_j1j2_bonds
from ..models.strict_schur_spatial import (
    StrictSchurSpatialPlan,
    two_plaquette_plan,
    two_triangle_plan,
)
from ..models.su2_schur_config_vit import (
    SU2SchurConfigViTConfig,
    init_su2_schur_config_vit,
    projected_su2_schur_config_vit_amplitudes,
    su2_schur_config_vit_amplitudes,
    su2_schur_config_vit_coefficients,
    su2_schur_config_vit_parameter_count,
)
from ..vmc.adam import train_exact_adam
from .exact_diagonalization import heisenberg_sz_matrix, spin_squared_matrix


jax.config.update("jax_enable_x64", True)


def _all_configurations(n_sites: int) -> np.ndarray:
    return np.asarray(
        [
            [
                (state >> (n_sites - site - 1)) & 1
                for site in range(n_sites)
            ]
            for state in range(1 << n_sites)
        ],
        dtype=np.int32,
    )


def _tangent_rank(
    params: object,
    config: SU2SchurConfigViTConfig,
) -> tuple[int, float]:
    jacobian = jax.jacrev(
        lambda candidate: su2_schur_config_vit_coefficients(candidate, config)
    )(params)
    rows = config.plan.reduced_parameter_count
    packed = np.concatenate(
        [np.asarray(leaf).reshape(rows, -1) for leaf in jax.tree_util.tree_leaves(jacobian)],
        axis=1,
    )
    singular = np.linalg.svd(packed, compute_uv=False)
    tolerance = max(float(singular[0]) * 1.0e-10, 1.0e-13)
    return int(np.count_nonzero(singular > tolerance)), float(singular[-1])


def _case(
    name: str,
) -> tuple[StrictSchurSpatialPlan, tuple[tuple[int, int, float], ...], object | None]:
    if name == "triangular_n6_j2_0125":
        # The finite 2x3 ground state is not in Gamma, so imposing Gamma here
        # would test the wrong physical sector.  Exact SU(2) remains enforced.
        return (
            two_triangle_plan(),
            triangular_j1j2_bonds(2, 3, j1=1.0, j2=0.125, periodic=True),
            None,
        )
    if name == "square_n8_j2_05":
        return (
            two_plaquette_plan(),
            square_j1j2_bonds(2, 4, j1=1.0, j2=0.5, periodic=True),
            rectangular_gamma_a1_group(2, 4),
        )
    raise ValueError("unknown Schur-config ViT exact-small case")


def schur_config_vit_exact_case(
    name: str,
    *,
    seeds: tuple[int, ...] = (0, 1),
    steps: int = 1000,
    learning_rate: float = 1.0e-2,
    embedding_dim: int = 8,
    head_count: int = 2,
    depth: int = 1,
) -> dict[str, object]:
    """Train one complete exact-small case and return physics diagnostics."""

    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if steps < 1 or learning_rate <= 0.0:
        raise ValueError("steps and learning_rate must be positive")
    plan, bonds, group = _case(name)
    configurations = jnp.asarray(_all_configurations(plan.n_sites))
    hamiltonian_np, integer_basis = heisenberg_sz_matrix(plan.n_sites, bonds)
    if tuple(integer_basis) != tuple(range(1 << plan.n_sites)):
        raise RuntimeError("full computational basis ordering changed")
    hamiltonian = jnp.asarray(hamiltonian_np)
    values, vectors = np.linalg.eigh(hamiltonian_np)
    exact_energy = float(values[0])
    ground_mask = values <= exact_energy + 2.0e-10
    ground_basis = vectors[:, ground_mask]
    model_config = SU2SchurConfigViTConfig(
        plan,
        embedding_dim=embedding_dim,
        head_count=head_count,
        depth=depth,
    )

    def state(params: object) -> jax.Array:
        if group is None:
            return su2_schur_config_vit_amplitudes(
                params, configurations, model_config
            )
        return projected_su2_schur_config_vit_amplitudes(
            params, configurations, model_config, group
        )

    def energy(params: object) -> jax.Array:
        vector = state(params)
        norm = jnp.vdot(vector, vector).real
        return jnp.vdot(vector, hamiltonian @ vector).real / norm

    spin_squared, spin_basis = spin_squared_matrix(plan.n_sites)
    if tuple(spin_basis) != tuple(integer_basis):
        raise RuntimeError("S2 and Hamiltonian bases disagree")
    runs = []
    for seed in seeds:
        params = init_su2_schur_config_vit(jax.random.PRNGKey(seed), model_config)
        initial_energy = float(energy(params))
        started = time.perf_counter()
        trained, optimizer = train_exact_adam(
            params,
            loss_fn=energy,
            steps=steps,
            learning_rate=learning_rate,
            record_every=max(1, steps // 4),
        )
        wall_seconds = time.perf_counter() - started
        vector = np.asarray(state(trained), dtype=np.complex128)
        vector /= np.linalg.norm(vector)
        action = hamiltonian_np @ vector
        final_energy = float(np.vdot(vector, action).real)
        variance = float(np.vdot(action - final_energy * vector, action - final_energy * vector).real)
        ground_weight = float(np.linalg.norm(ground_basis.conj().T @ vector) ** 2)
        spin_residual = float(np.linalg.norm(spin_squared @ vector))
        runs.append(
            {
                "seed": int(seed),
                "initial_energy_total": initial_energy,
                "final_energy_total": final_energy,
                "final_energy_per_site": final_energy / plan.n_sites,
                "energy_error_total": final_energy - exact_energy,
                "variance_total": variance,
                "ground_subspace_weight": ground_weight,
                "spin_squared_residual_norm": spin_residual,
                "wall_seconds": wall_seconds,
                "optimizer": optimizer,
            }
        )
    best = min(runs, key=lambda row: row["final_energy_total"])
    reference_params = init_su2_schur_config_vit(
        jax.random.PRNGKey(seeds[0]), model_config
    )
    tangent_rank, smallest_singular = _tangent_rank(reference_params, model_config)
    parameter_count = su2_schur_config_vit_parameter_count(reference_params)
    return {
        "case": name,
        "n_sites": plan.n_sites,
        "j2_over_j1": 0.125 if plan.n_sites == 6 else 0.5,
        "spatial_sector": (
            "unprojected_finite_cluster_ground_momentum"
            if group is None
            else "Gamma-A1_exact_Reynolds"
        ),
        "spatial_group_order": 1 if group is None else len(group),
        "complete_singlet_dimension": plan.reduced_parameter_count,
        "model_parameter_count": parameter_count,
        "coefficient_tangent_rank": tangent_rank,
        "coefficient_tangent_smallest_singular_value": smallest_singular,
        "exact_ground_energy_total": exact_energy,
        "exact_ground_energy_per_site": exact_energy / plan.n_sites,
        "exact_ground_degeneracy": int(np.count_nonzero(ground_mask)),
        "best": best,
        "runs": runs,
        "interpretation": (
            "exact-small completeness and trainability gate only; the full singlet "
            "space is available, so this result is not evidence of a scalable "
            "variational advantage"
        ),
    }


def schur_config_vit_exact_suite(**kwargs) -> dict[str, object]:
    """Run the triangular and square exact-small architecture gates."""

    reports = {
        name: schur_config_vit_exact_case(name, **kwargs)
        for name in ("triangular_n6_j2_0125", "square_n8_j2_05")
    }
    passed = all(
        abs(float(report["best"]["energy_error_total"])) < 2.0e-8
        and float(report["best"]["variance_total"]) < 2.0e-8
        and float(report["best"]["spin_squared_residual_norm"]) < 2.0e-10
        and int(report["coefficient_tangent_rank"])
        == int(report["complete_singlet_dimension"])
        for report in reports.values()
    )
    return {
        "passed": passed,
        "reports": reports,
        "next_gate": (
            "four or more local Schur cells with autoregressive reduced-label "
            "sampling, exact lattice-sector action through local F-moves, and "
            "matched-work comparison against fixed-Sz Spatial-ViT"
        ),
    }


__all__ = ["schur_config_vit_exact_case", "schur_config_vit_exact_suite"]
