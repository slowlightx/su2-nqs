"""Exact diagnostics for coherent spatial orbits of one fusion-tree state.

A component ``U_g |psi_theta>`` uses the same variational parameters as the
representative but a symmetry-related physical blocking.  Their equal
coherent sum is therefore a zero-extra-parameter spatial projection.  The
incoherent Born mixture is a directly sampleable proposal for that sum; its
population importance-sampling ESS diagnoses the associated sampling tax.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def rectangular_site_permutations(
    rows: int, columns: int
) -> dict[str, tuple[int, ...]]:
    """Return the geometric point operations of an open rectangle.

    Site indices are row-major and each tuple is an active
    ``old_site -> new_site`` map.  A square additionally receives the four
    operations that interchange its coordinate axes.
    """

    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")

    def image(transform) -> tuple[int, ...]:
        result = []
        for row in range(rows):
            for column in range(columns):
                new_row, new_column = transform(row, column)
                result.append(new_row * columns + new_column)
        return tuple(result)

    result = {
        "identity": image(lambda row, column: (row, column)),
        "reflect_rows": image(lambda row, column: (rows - 1 - row, column)),
        "reflect_columns": image(
            lambda row, column: (row, columns - 1 - column)
        ),
        "rotate_180": image(
            lambda row, column: (rows - 1 - row, columns - 1 - column)
        ),
    }
    if rows == columns:
        result.update(
            {
                "rotate_90": image(
                    lambda row, column: (column, rows - 1 - row)
                ),
                "rotate_270": image(
                    lambda row, column: (columns - 1 - column, row)
                ),
                "reflect_main_diagonal": image(
                    lambda row, column: (column, row)
                ),
                "reflect_anti_diagonal": image(
                    lambda row, column: (
                        rows - 1 - column,
                        columns - 1 - row,
                    )
                ),
            }
        )
    return result


def bond_automorphisms(
    candidates: Mapping[str, Sequence[int]],
    bonds: Sequence[tuple[int, int, float]],
    *,
    tolerance: float = 1.0e-12,
) -> dict[str, tuple[int, ...]]:
    """Keep candidate site maps that preserve a weighted bond Hamiltonian."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    weighted: dict[tuple[int, int], float] = {}
    for first, second, coupling in bonds:
        edge = tuple(sorted((int(first), int(second))))
        weighted[edge] = weighted.get(edge, 0.0) + float(coupling)
    kept = {}
    for name, raw_image in candidates.items():
        image = tuple(int(value) for value in raw_image)
        if tuple(sorted(image)) != tuple(range(len(image))):
            raise ValueError(f"candidate {name!r} is not a site permutation")
        transformed: dict[tuple[int, int], float] = {}
        for (first, second), coupling in weighted.items():
            edge = tuple(sorted((image[first], image[second])))
            transformed[edge] = transformed.get(edge, 0.0) + coupling
        support = set(weighted) | set(transformed)
        discrepancy = max(
            (
                abs(weighted.get(edge, 0.0) - transformed.get(edge, 0.0))
                for edge in support
            ),
            default=0.0,
        )
        if discrepancy <= tolerance:
            kept[str(name)] = image
    if "identity" not in kept:
        raise ValueError("candidate set must contain a bond-preserving identity")
    return kept


def _normalized(state: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have finite nonzero norm")
    return vector / norm


def exact_state_metrics(
    state: np.ndarray,
    hamiltonian: np.ndarray,
    *,
    reference: np.ndarray | None = None,
) -> dict[str, float]:
    """Return normalized energy, variance, and optional reference fidelity."""

    vector = _normalized(state, name="state")
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    if matrix.shape != (vector.size, vector.size):
        raise ValueError("hamiltonian shape does not match state dimension")
    applied = matrix @ vector
    energy = float(np.real(np.vdot(vector, applied)))
    variance = float(max(np.real(np.vdot(applied, applied)) - energy**2, 0.0))
    result = {"energy": energy, "variance": variance}
    if reference is not None:
        target = _normalized(reference, name="reference")
        if target.shape != vector.shape:
            raise ValueError("reference shape does not match state")
        result["reference_fidelity"] = float(abs(np.vdot(target, vector)) ** 2)
    return result


def coherent_tree_orbit_analysis(
    representative: np.ndarray,
    actions: Mapping[str, np.ndarray],
    hamiltonian: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    gram_relative_cutoff: float = 1.0e-12,
) -> dict[str, object]:
    """Analyze equal-coherent, incoherent, and Ritz combinations of an orbit.

    ``actions`` must include ``identity`` and act in one common fusion basis.
    The equal coherent state is the trivial-irrep projection when the actions
    form the full automorphism group.  The returned mixture ESS is exact in
    the finite basis for proposal ``q(s)=mean_g |(U_g psi)(s)|^2``.
    """

    if "identity" not in actions:
        raise ValueError("actions must contain identity")
    if not 0.0 < gram_relative_cutoff < 1.0:
        raise ValueError("gram_relative_cutoff must lie between zero and one")
    state = _normalized(representative, name="representative")
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    if matrix.shape != (state.size, state.size):
        raise ValueError("hamiltonian shape does not match representative")

    names = tuple(actions)
    columns = []
    action_arrays = []
    for name in names:
        action = actions[name]
        if getattr(action, "shape", None) != matrix.shape:
            raise ValueError(f"action {name!r} has incompatible shape")
        component = _normalized(np.asarray(action @ state), name=f"action {name}")
        columns.append(component)
        action_arrays.append(action)
    components = np.column_stack(columns)

    dense_actions = [
        action.toarray() if hasattr(action, "toarray") else np.asarray(action)
        for action in action_arrays
    ]
    projector = np.mean(dense_actions, axis=0)
    projector_scale = max(float(np.linalg.norm(projector, ord="fro")), 1.0e-30)
    projector_hermiticity_error = float(
        np.linalg.norm(projector - projector.conj().T, ord="fro")
        / projector_scale
    )
    projector_idempotency_error = float(
        np.linalg.norm(projector @ projector - projector, ord="fro")
        / projector_scale
    )

    mean_state = np.mean(components, axis=1)
    coherent_norm_squared = float(np.real(np.vdot(mean_state, mean_state)))
    if coherent_norm_squared <= 1.0e-28:
        raise ValueError("equal coherent orbit cancels to numerical zero")
    coherent = mean_state / np.sqrt(coherent_norm_squared)

    component_metrics = [
        {
            "name": name,
            **exact_state_metrics(
                components[:, index], matrix, reference=reference
            ),
        }
        for index, name in enumerate(names)
    ]
    coherent_metrics = exact_state_metrics(coherent, matrix, reference=reference)

    component_energies = np.asarray(
        [row["energy"] for row in component_metrics], dtype=float
    )
    applied_components = matrix @ components
    incoherent_energy = float(np.mean(component_energies))
    incoherent_second = float(
        np.mean(np.sum(np.abs(applied_components) ** 2, axis=0))
    )
    incoherent_metrics = {
        "energy": incoherent_energy,
        "variance": float(max(incoherent_second - incoherent_energy**2, 0.0)),
    }
    if reference is not None:
        target = _normalized(reference, name="reference")
        incoherent_metrics["reference_fidelity"] = float(
            np.mean(np.abs(target.conj() @ components) ** 2)
        )

    gram = components.conj().T @ components
    gram_values, gram_vectors = np.linalg.eigh(gram)
    largest_gram = float(max(gram_values[-1], 0.0))
    retained = gram_values > gram_relative_cutoff * largest_gram
    if not np.any(retained):
        raise RuntimeError("orbit Gram matrix has no retained direction")
    orthonormal = (
        components
        @ gram_vectors[:, retained]
        / np.sqrt(gram_values[retained])[None, :]
    )
    reduced_hamiltonian = orthonormal.conj().T @ matrix @ orthonormal
    ritz_values, ritz_vectors = np.linalg.eigh(reduced_hamiltonian)
    ritz = orthonormal @ ritz_vectors[:, 0]
    ritz_metrics = exact_state_metrics(ritz, matrix, reference=reference)

    proposal = np.mean(np.abs(components) ** 2, axis=1)
    target_probability = np.abs(coherent) ** 2
    supported = proposal > 0.0
    outside_support = float(np.sum(target_probability[~supported]))
    weights = np.zeros_like(proposal)
    weights[supported] = target_probability[supported] / proposal[supported]
    second_weight_moment = float(np.sum(proposal * weights**2))
    ess_fraction = 1.0 / second_weight_moment

    denominator = max(float(np.linalg.norm(matrix, ord="fro")), 1.0e-30)
    covariance_errors = []
    for dense_action in dense_actions:
        covariance_errors.append(
            float(
                np.linalg.norm(matrix @ dense_action - dense_action @ matrix, ord="fro")
                / denominator
            )
        )
    closure_errors = [
        float(np.linalg.norm(np.asarray(action @ coherent) - coherent))
        for action in action_arrays
    ]

    best_index = int(np.argmin(component_energies))
    result: dict[str, object] = {
        "component_names": list(names),
        "component_count": len(names),
        "component_metrics": component_metrics,
        "representative": component_metrics[names.index("identity")],
        "best_component": component_metrics[best_index],
        "incoherent_mixture": incoherent_metrics,
        "equal_coherent": coherent_metrics,
        "orbit_ritz": ritz_metrics,
        "coherent_norm_squared_before_normalization": coherent_norm_squared,
        "gram_eigenvalues": [float(value) for value in gram_values],
        "orbit_effective_rank": int(np.count_nonzero(retained)),
        "maximum_hamiltonian_covariance_error": max(covariance_errors),
        "maximum_equal_orbit_closure_error": max(closure_errors),
        "projector_hermiticity_error": projector_hermiticity_error,
        "projector_idempotency_error": projector_idempotency_error,
        "mixture_proposal": {
            "population_ess_fraction": ess_fraction,
            "chi_squared_divergence": second_weight_moment - 1.0,
            "maximum_importance_weight": float(np.max(weights)),
            "target_weight_outside_proposal_support": outside_support,
            "total_variation_distance": float(
                0.5 * np.sum(np.abs(target_probability - proposal))
            ),
        },
    }
    if reference is not None:
        reference_energy = exact_state_metrics(reference, matrix)["energy"]
        result["exact_reference_energy"] = reference_energy
        for key in (
            "representative",
            "best_component",
            "incoherent_mixture",
            "equal_coherent",
            "orbit_ritz",
        ):
            result[key]["energy_error"] = float(
                result[key]["energy"] - reference_energy
            )
        projected_weight = min(max(coherent_norm_squared, 0.0), 1.0)
        orthogonal_weight = 1.0 - projected_weight
        representative_energy = float(result["representative"]["energy"])
        coherent_energy = float(result["equal_coherent"]["energy"])
        orthogonal_energy = None
        reconstruction_error = None
        if orthogonal_weight > 1.0e-14:
            orthogonal_energy = (
                representative_energy - projected_weight * coherent_energy
            ) / orthogonal_weight
            reconstruction_error = abs(
                representative_energy
                - projected_weight * coherent_energy
                - orthogonal_weight * orthogonal_energy
            )
        raw_error = representative_energy - reference_energy
        removed_fraction = (
            (representative_energy - coherent_energy) / raw_error
            if raw_error > 1.0e-14
            else None
        )
        result["symmetry_sector_decomposition"] = {
            "trivial_sector_weight": projected_weight,
            "nontrivial_sector_weight": orthogonal_weight,
            "trivial_sector_energy": coherent_energy,
            "nontrivial_sector_conditional_energy": orthogonal_energy,
            "energy_reconstruction_error": reconstruction_error,
            "raw_energy_error_removed_fraction": removed_fraction,
        }
    return result
