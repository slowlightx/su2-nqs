"""Bounded amplitude/phase diagnosis of an unchanged reduced-space ViT.

This is an exact-small *oracle* experiment on a previously saved state, not
new ED or a production energy optimization. The current N36 checkpoint is
evaluated on the transferable N16 layout. Physical site reflection matches
the current neural geometry; all CG magnetic structure remains fixed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from su2_nqs.benchmarks.clebsch_tree_exact import (
    fixed_m_spin_basis, graph_hamiltonian_coo, physical_permutation_index_table,
)
from su2_nqs.benchmarks.fidelity_factorization import fidelity_factors, normalize_state
from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.compact_tree_gru import split_tables
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig, _candidate_features, _global_configuration_phase,
    _selected_tree_actions, many_cell_schur_log_amplitudes,
)
from su2_nqs.models.tree_gru import tree_decisions
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.su2.reduced_physical_overlap import tree_overlaps_on_bits
from su2_nqs.vmc.adam import _manual_adam_init, _manual_adam_update
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from .spin_reduced_vit_viability import physical_reflection


ROOT = Path(__file__).resolve().parents[3]
PHASE_KEYS = ("global_layers", "configuration_projection", "direct_phase_bilinear",
              "global_phase_head", "global_phase_bias")


def phase_only_fidelity(reference: np.ndarray, phase: np.ndarray) -> float:
    """Return fidelity with oracle magnitudes in the same doubled-q basis."""
    target = normalize_state(reference)
    angles = np.asarray(phase, dtype=np.float64)
    if angles.shape != target.shape or not np.isfinite(angles).all():
        raise ValueError("one finite phase is required per reduced configuration")
    return float(abs(np.vdot(target, abs(target) * np.exp(1j * angles))) ** 2)


def reference_support_amplitude_oracle(
    reference: np.ndarray, candidate: np.ndarray, *, zero_tolerance: float,
    maximum_ambiguous_mass: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use known reference phases only on its nonzero reduced support.

    Candidate weight outside this support is removed and reported. The
    returned normalized state is a *conditional diagnostic state*, never
    the original candidate or a uniquely defined full phase-repaired state.
    Both arguments use the same fixed-CG, doubled-q basis ordering.
    """
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    if not np.isfinite(maximum_ambiguous_mass) or not 0 <= maximum_ambiguous_mass < 1:
        raise ValueError("maximum_ambiguous_mass must be in [0,1)")
    target = normalize_state(reference)
    model = normalize_state(candidate)
    if target.shape != model.shape:
        raise ValueError("reference and candidate must have the same ordering")
    active = abs(target) > zero_tolerance
    retained = float(np.sum(abs(model[active]) ** 2))
    if retained <= 0:
        raise ValueError("candidate has no weight on known reference support")
    oracle = np.zeros_like(model, dtype=np.complex128)
    oracle[active] = abs(model[active]) * target[active] / abs(target[active])
    oracle /= np.sqrt(retained)
    ambiguity = float(np.sum(abs(model[~active]) ** 2))
    return oracle, dict(
        removed_candidate_mass=ambiguity,
        removed_reference_mass=float(np.sum(abs(target[~active]) ** 2)),
        within_registered_ambiguous_mass_budget=ambiguity <= maximum_ambiguous_mass,
        full_state_reference_phase_energy_reportable=ambiguity == 0.0,
        conditional_state_not_original_candidate=True,
        note="Reference-zero phases are unknown; no full-state oracle energy is assigned.",
    )


def root_channel_diagnosis(config: ManyCellSchurViTConfig,
                           labels: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Measure the output-head rank and its oracle root-q distribution error."""
    left, _, valid = split_tables(config.layout.tree)
    choices = left[0, config.target_q, valid[0, config.target_q]]
    features = _candidate_features(config)[0, config.target_q, valid[0, config.target_q]]
    rank = int(np.linalg.matrix_rank(features, tol=1e-12))
    root = tree_decisions(config.layout.tree)[0]
    weights = abs(normalize_state(reference)) ** 2
    distribution = np.asarray([weights[labels[:, root.left_index] == q].sum() for q in choices])
    basis = np.stack((choices.astype(float), choices.astype(float) ** 2), axis=1)

    def loss(theta: np.ndarray) -> float:
        logits = basis @ theta
        log_z = np.logaddexp.reduce(logits)
        return float(-distribution @ (logits - log_z))

    fit = minimize(loss, np.zeros(2), method="BFGS", options={"gtol": 1e-10})
    logits = basis @ fit.x
    fitted = np.exp(logits - np.logaddexp.reduce(logits))
    active = distribution > 0
    entropy = -float(distribution[active] @ np.log(distribution[active]))
    return dict(q_choices=choices.tolist(), feature_rank=rank,
                physical_probability_rank=rank - 1,
                general_probability_degrees_of_freedom=len(choices) - 1,
                reference_probabilities=distribution.tolist(),
                best_quadratic_probabilities=fitted.tolist(),
                root_KL=loss(fit.x) - entropy,
                root_total_variation=float(abs(distribution - fitted).sum() / 2),
                optimizer_success=bool(fit.success),
                optimizer_gradient_norm=float(np.linalg.norm(fit.jac)))


def state_readout(coefficients: np.ndarray, labels: np.ndarray,
                  config: ManyCellSchurViTConfig, isometry: np.ndarray,
                  hamiltonian: sparse.csr_matrix) -> dict[str, float]:
    """Evaluate physical full-H moments and diagonal reduced spin projectors."""
    c = normalize_state(coefficients)
    physical = isometry @ c
    norm = float(np.vdot(physical, physical).real)
    action = hamiltonian @ physical
    energy = np.vdot(physical, action) / norm
    residual = action - energy * physical
    decoded = config.layout.decode(labels)
    weights = abs(c) ** 2
    phase_gauge = .5 * np.angle(np.sum(c ** 2))
    return dict(physical_norm=norm, energy_per_site=float(energy.real / config.n),
                imaginary_energy_per_site=float(energy.imag / config.n),
                variance_per_site=float(np.vdot(residual, residual).real / norm / config.n),
                mean_cell_q0=float(np.sum(weights[:, None] * (decoded.cell_q == 0)) / config.layout.n_cells),
                all_cells_q0=float(weights[np.all(decoded.cell_q == 0, axis=1)].sum()),
                negative_coefficient_mass_in_best_real_gauge=float(weights[(c * np.exp(-1j * phase_gauge)).real < 0].sum()),
                real_state_overlap=float(abs(np.sum(c ** 2))))


def oracle_fit(parent: dict[str, Any], labels: np.ndarray, reference: np.ndarray,
               config: ManyCellSchurViTConfig, *, arm: str, steps: int,
               learning_rate: float, maximum_seconds: float) -> tuple[dict, dict]:
    """Fit one named factor; never change architecture or use energy in fitting.

The phase arms freeze the shared decision projection as well as every Born
parameter. The amplitude arm includes the shared projection and ignores its
consequences for phase; it is an amplitude-target fit, not a physical
amplitude-only update. A separate reciprocal check uses reference phases
only on the nonzero reference support and reports any removed model mass.
All reference rows are training data: this is not a held-out prediction test.
"""
    if arm not in ("phase_fidelity", "phase_supervision", "amplitude_cross_entropy"):
        raise ValueError("unknown oracle-fit arm")
    if steps < 1 or learning_rate <= 0 or maximum_seconds <= 0:
        raise ValueError("positive bounded training settings are required")
    is_phase = arm.startswith("phase_")
    keys = PHASE_KEYS if is_phase else tuple(k for k in parent if k not in PHASE_KEYS)
    parameters = {key: parent[key] for key in keys}
    state = _manual_adam_init(parameters)
    label_jax = jnp.asarray(labels)
    chosen = _selected_tree_actions(label_jax, config)[2]
    target = normalize_state(reference)
    p = jnp.asarray(abs(target) ** 2)
    unit = jnp.asarray(np.where(abs(target) > 0, target / np.where(abs(target) > 0, abs(target), 1), 1))
    # The reference is a real state up to global phase. This fixed lift is
    # deliberate oracle supervision, not the physical circular loss.
    gauge = np.angle(target[np.argmax(abs(target))])
    aligned = target * np.exp(-1j * gauge)
    lifted = jnp.asarray(np.where(aligned.real < 0, np.pi, 0.0))
    active = abs(target) > 0
    entropy = -float(np.sum(abs(target[active]) ** 2 * np.log(abs(target[active]) ** 2)))

    def loss(part):
        full = {**parent, **part}
        if is_phase:
            phi = _global_configuration_phase(full, chosen, config)
            if arm == "phase_supervision":
                return jnp.sum(p * (phi - lifted) ** 2)
            overlap = jnp.sum(p * jnp.conj(unit) * jnp.exp(1j * phi))
            # Only an optimizer safeguard; reported fidelities have no floor.
            return -jnp.log(jnp.maximum(abs(overlap) ** 2, 1e-15))
        logs = many_cell_schur_log_amplitudes(full, label_jax, config)
        return -2 * jnp.sum(p * logs.real) - entropy

    @jax.jit
    def update(part, optimizer):
        value, gradient = jax.value_and_grad(loss)(part)
        new, optimizer = _manual_adam_update(part, gradient, optimizer, learning_rate)
        return new, optimizer, value

    started = time.perf_counter()
    compiled = update.lower(parameters, state).compile()
    compilation_seconds = time.perf_counter() - started
    history = [dict(step=0, loss=float(loss(parameters)))]
    started = time.perf_counter()
    count = 0
    for count in range(1, steps + 1):
        parameters, state, value = compiled(parameters, state)
        jax.block_until_ready(value)
        if count % 50 == 0 or count == steps:
            item = dict(step=count, loss=float(loss(parameters)))
            history.append(item)
            print(json.dumps(dict(arm=arm, **item)), flush=True)
        if time.perf_counter() - started > maximum_seconds:
            break
    full = {**parent, **parameters}
    return full, dict(arm=arm, completed_steps=count, planned_steps=steps,
                      status="completed" if count == steps else "budget_limited",
                      compilation_seconds=compilation_seconds,
                      optimization_seconds=time.perf_counter() - started,
                      trained_parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(parameters)),
                      frozen_parameter_keys=[key for key in parent if key not in keys],
                      history=history, optimizer_state=state)


def run(config_path: Path) -> dict[str, Any]:
    """Run registered diagnosis without editing any parent checkpoint or seal."""
    spec = json.loads(config_path.read_text())
    output = ROOT / spec["output"]
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    for key in ("source_bundle", "parent_checkpoint"):
        if file_sha256(ROOT / spec[key]) != spec[key + "_sha256"]:
            raise ValueError(f"changed frozen input: {key}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    atomic_json(output / "registration.json", dict(settings=spec,
        configuration_sha256=file_sha256(config_path),
        workflow_sha256=file_sha256(Path(__file__)), runtime=numerical_runtime_identity(),
        no_ED=True, architecture_unchanged=True, reference_is_training_data=True,
        full_A1_native_N36_sampler_implemented=False))
    saved = load_checkpoint(ROOT / spec["parent_checkpoint"])
    parent = jax.tree_util.tree_map(jnp.asarray, saved["params"])
    layout = rectangular_four_spin_layout(4, 4)
    config = ManyCellSchurViTConfig(layout, embedding_dim=spec["embedding_dim"],
        head_count=spec["head_count"], depth=spec["depth"], direct_phase_initialization_scale=.1)
    with np.load(ROOT / spec["source_bundle"]) as data:
        labels = data["labels"].copy()
        physical_reference = data["reconstructed_physical_state"].copy()
        cached_c = data["balanced_coefficients"].copy()
        cached_hc = data["matrix_free_action"].copy()
    expected = np.asarray([s.labels for s in layout.tree.enumerate_states(0)])
    np.testing.assert_array_equal(labels, expected)
    bits = np.asarray(fixed_m_spin_basis(16, 0), dtype=np.int32)
    reflection = physical_reflection(4, 4)
    physical_order = tuple(reflection[site] for site in layout.site_order)
    u = tree_overlaps_on_bits(layout.tree, labels, bits, physical_leaf_order=physical_order)
    reference = u.T @ physical_reference
    np.testing.assert_allclose(np.linalg.norm(reference), 1., atol=2e-12, rtol=0.)
    reconstruction_error = float(np.max(abs(u @ reference - physical_reference)))
    if reconstruction_error > 2e-12:
        raise ValueError("reference does not close in reflected complete singlet basis")
    bonds = triangular_hnf_j1j2_bonds(((4, 0), (0, 4)), j1=1., j2=spec["j2"])
    rows, columns, values, configurations = graph_hamiltonian_coo(16, 0, bonds)
    np.testing.assert_array_equal(np.asarray(configurations), bits)
    h = sparse.coo_matrix((values, (rows, columns)), shape=(len(bits), len(bits))).tocsr()
    ref_readout = state_readout(reference, labels, config, u, h)
    cached_energy = float((np.vdot(cached_c, cached_hc) / np.vdot(cached_c, cached_c)).real / 16)
    if abs(ref_readout["energy_per_site"] - cached_energy) > 2e-12:
        raise ValueError("independent physical-H reference disagrees with frozen replay")
    group = triangular_hnf_gamma_a1_group(((4, 0), (0, 4)))
    indices = physical_permutation_index_table(bits, group)

    def describe(parameters):
        logs = np.asarray(many_cell_schur_log_amplitudes(parameters, labels, config))
        candidate = normalize_state(np.exp(logs))
        phase_hybrid = abs(reference) * np.exp(1j * logs.imag)
        projected_physical = np.mean((u @ candidate)[indices], axis=0)
        projected = u.T @ projected_physical
        projected_weight = float(np.vdot(projected, projected).real)
        projected = normalize_state(projected)
        native_p = abs(candidate) ** 2
        projected_p = abs(projected) ** 2
        return dict(fidelity_factors=asdict(fidelity_factors(reference, candidate)),
            native=state_readout(candidate, labels, config, u, h),
            reference_amplitude_model_phase=state_readout(phase_hybrid, labels, config, u, h),
            reference_amplitude_phase_fidelity=phase_only_fidelity(reference, logs.imag),
            exact_A1=dict(group_order=len(group), projection_weight=projected_weight,
                fidelity_factors=asdict(fidelity_factors(reference, projected)),
                readout=state_readout(projected, labels, config, u, h),
                native_vs_projected_Born_TV=float(abs(native_p-projected_p).sum()/2),
                is_small_exact_reference_not_N36_sampler=True),
            native_mass_on_reference_phase_zeros=float(native_p[abs(reference) <= spec["phase_zero_tolerance"]].sum()))

    baseline = describe(parent)
    print(json.dumps(dict(stage="baseline", metrics=baseline)), flush=True)
    result = dict(settings=spec, parent_params_sha256=pytree_sha256(parent),
        exact_SU2=True, native_full_A1=False, no_ED=True, no_new_N36_H_calls=True,
        reference_is_training_data=True, N16_reflection_order=list(physical_order),
        reference_reconstruction_max_error=reconstruction_error,
        independent_reference_energy_error=abs(ref_readout["energy_per_site"]-cached_energy),
        reference=ref_readout, root_output_head=root_channel_diagnosis(config, labels, reference),
        baseline=baseline, sign_flattened_reference=state_readout(abs(reference), labels, config, u, h), arms={})
    atomic_json(output / "progress.json", result)
    for arm in spec["arms"]:
        trained, metrics = oracle_fit(parent, labels, reference, config,
            arm=arm, steps=spec["steps_per_arm"], learning_rate=spec["learning_rate"],
            maximum_seconds=spec["maximum_training_seconds_per_arm"])
        optimizer = metrics.pop("optimizer_state")
        metrics["endpoint"] = describe(trained)
        checkpoint = output / (arm + "_checkpoint.pkl")
        save_checkpoint(checkpoint, jax.tree_util.tree_map(np.asarray, trained),
            optimizer_state=jax.tree_util.tree_map(np.asarray, optimizer),
            metadata=dict(oracle_diagnosis=True, all_reference_rows_are_training_data=True,
                          arm=arm, steps=metrics["completed_steps"], parent_params_sha256=pytree_sha256(parent)))
        restored = load_checkpoint(checkpoint)
        original = np.asarray(many_cell_schur_log_amplitudes(trained, labels, config))
        replay = np.asarray(many_cell_schur_log_amplitudes(restored["params"], labels, config))
        metrics["checkpoint_log_amplitude_max_error"] = float(np.max(abs(original-replay)))
        metrics["checkpoint_sha256"] = file_sha256(checkpoint)
        result["arms"][arm] = metrics
        atomic_json(output / "progress.json", result)
    old = json.loads((ROOT / spec["parent_results"]).read_text())
    history = old["history"][1:]
    result["N36_previous_cost_only"] = dict(
        samples_per_step=64, steps=len(old["history"]),
        warm_median_step_seconds=float(np.median([r["step_seconds"] for r in history])),
        warm_median_full_H_seconds=float(np.median([r["full_H"]["full_H_seconds"] for r in history])),
        maximum_sample_SR_matrix_dimension=max(r["sr"]["sample_matrix_dimension"] for r in history),
        parameter_count=old["optimized_parameter_count"],
        note="Exact linear-system residual is not an accuracy certificate for the stochastic gradient or Fisher metric.")
    result["elapsed_seconds"] = time.perf_counter() - started
    result["source_inputs_unchanged"] = all(file_sha256(ROOT / spec[key]) == spec[key+"_sha256"]
                                             for key in ("source_bundle", "parent_checkpoint"))
    atomic_json(output / "results.json", result)
    print(json.dumps(dict(stage="completed", elapsed_seconds=result["elapsed_seconds"])), flush=True)
    return result


def reciprocal_check(config_path: Path) -> dict[str, Any]:
    """Check saved endpoints without training, sampling, or new eigensolves.

    This is a post-review supplement, not a new preregistered optimization.
    The original workflow snapshot, registration and results stay immutable.
    """
    spec = json.loads(config_path.read_text())
    output = ROOT / spec["output"]
    target_path = output / "reciprocal_check.json"
    if target_path.exists():
        raise FileExistsError(f"reciprocal diagnostic already exists: {target_path}")
    old = json.loads((output / "results.json").read_text())
    registration = json.loads((output / "registration.json").read_text())
    if file_sha256(output / "workflow_snapshot.py") != registration["workflow_sha256"]:
        raise ValueError("original registered workflow snapshot changed")
    for key in ("source_bundle", "parent_checkpoint"):
        if file_sha256(ROOT / spec[key]) != spec[key + "_sha256"]:
            raise ValueError(f"changed frozen input: {key}")
    started = time.perf_counter()
    layout = rectangular_four_spin_layout(4, 4)
    config = ManyCellSchurViTConfig(layout, embedding_dim=spec["embedding_dim"],
        head_count=spec["head_count"], depth=spec["depth"], direct_phase_initialization_scale=.1)
    with np.load(ROOT / spec["source_bundle"]) as saved:
        labels = saved["labels"].copy()
        physical_reference = saved["reconstructed_physical_state"].copy()
    bits = np.asarray(fixed_m_spin_basis(16, 0), dtype=np.int32)
    reflection = physical_reflection(4, 4)
    u = tree_overlaps_on_bits(layout.tree, labels, bits,
        physical_leaf_order=tuple(reflection[site] for site in layout.site_order))
    reference = u.T @ physical_reference
    bonds = triangular_hnf_j1j2_bonds(((4, 0), (0, 4)), j1=1., j2=spec["j2"])
    rows, columns, values, configurations = graph_hamiltonian_coo(16, 0, bonds)
    np.testing.assert_array_equal(np.asarray(configurations), bits)
    h = sparse.coo_matrix((values, (rows, columns)), shape=(len(bits), len(bits))).tocsr()
    result: dict[str, Any] = dict(
        post_review_supplement=True, no_training=True, no_sampling=True, no_ED=True,
        no_new_N36_H_calls=True, original_results_sha256=file_sha256(output / "results.json"),
        original_workflow_sha256=registration["workflow_sha256"],
        supplement_workflow_sha256=file_sha256(Path(__file__)), states={},
    )
    endpoints = {"baseline": ROOT / spec["parent_checkpoint"]}
    endpoints.update({arm: output / (arm + "_checkpoint.pkl") for arm in spec["arms"]})
    for name, path in endpoints.items():
        if name != "baseline" and file_sha256(path) != old["arms"][name]["checkpoint_sha256"]:
            raise ValueError(f"changed diagnostic checkpoint: {name}")
        params = load_checkpoint(path)["params"]
        candidate = normalize_state(np.exp(np.asarray(many_cell_schur_log_amplitudes(params, labels, config))))
        oracle, metrics = reference_support_amplitude_oracle(reference, candidate,
            zero_tolerance=spec["phase_zero_tolerance"],
            maximum_ambiguous_mass=spec["maximum_ambiguous_mass"])
        metrics["conditional_reference_phase_state"] = state_readout(oracle, labels, config, u, h)
        result["states"][name] = metrics
    result["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(target_path, result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_reduced_vit_diagnosis.json")
    parser.add_argument("--reciprocal", action="store_true", help="Read saved endpoints; no further optimization")
    args = parser.parse_args()
    (reciprocal_check if args.reciprocal else run)(args.config)
