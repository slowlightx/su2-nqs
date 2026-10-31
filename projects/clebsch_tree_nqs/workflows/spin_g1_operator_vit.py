"""G1 exact-SU(2)/A1 Operator-ViT preflight and N=36 canary.

The scalable arm uses two symmetry-labelled tokens: the physical Hamiltonian
``H=J1+J2*J2shell`` and one preregistered A1 resonance.  Two residual layers
therefore contain the full two-Lanczos control and add ``HR``, ``RH``, and
``R2`` directions.  All magnetic structure remains in the projected singlet
carrier; no direct fusion-path model is used.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.operator_tangent_response import triangular_a1_word_actions
from su2_nqs.benchmarks.operator_vit_sampled import (
    carrier_standardization_for_labels,
    centered_scalar_component_matrices,
    effective_word_actions,
    expanded_base_words,
    sampled_operator_vit_energy,
    sampled_operator_vit_summary,
    scalar_component_words,
)
from su2_nqs.benchmarks.qsl_three_arm import (
    _energy_variance,
    triangular_pi_flux_operator_components,
)
from su2_nqs.benchmarks.scalar_operator_frontier import (
    triangular_a1_resonance_operators,
)
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit,
    projected_pi_zero_flux_a1_amplitudes,
)
from su2_nqs.benchmarks.xdiag_physical_seed_krylov import SMALL_CASES
from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    pytree_sha256,
)
from su2_nqs.frontier.parameter_artifact import (
    load_parameter_artifact,
    save_parameter_artifact,
)
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.models.su2_a1_operator_vit import (
    SU2A1OperatorViTConfig,
    init_su2_a1_operator_vit,
    su2_a1_operator_vit_coefficients,
    su2_a1_operator_vit_component_weights,
    su2_a1_operator_vit_parameter_count,
)
from su2_nqs.operators.triangular_a1_scalar import TRIANGULAR_A1_SCALAR_LABELS
from su2_nqs.vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange
from su2_nqs.vmc.physical_sampled_adam import (
    SampledAdamState,
    initialize_sampled_adam,
    sampled_adam_update,
)
from su2_nqs.vmc.resumable_checkpoint import (
    load_latest_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_g1_operator_vit"
EFFECTIVE_LABELS = ("H_J1J2", "A1_resonance")
DESCRIPTORS = ((1.0, 2.0 / 6.0, 0.0), (1.0, 4.0 / 6.0, 1.0))
N36_EXACT_ENERGY_PER_SITE = -0.515564
N36_GCNN_ENERGY_PER_SITE = -0.515386


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid G1 Operator-ViT identity")
    if spec.get("execution_tier") not in {
        "local_exact_connectivity_preflight",
        "remote_n36_canary",
    }:
        raise ValueError("invalid G1 execution tier")
    if not np.isfinite(float(spec.get("j2_over_j1", np.nan))):
        raise ValueError("j2_over_j1 must be finite")
    if str(spec.get("resonance_token")) not in TRIANGULAR_A1_SCALAR_LABELS[2:]:
        raise ValueError("resonance_token must be a registered non-shell scalar")
    model = spec.get("model")
    training = spec.get("training")
    sampling = spec.get("sampling")
    if not all(isinstance(value, dict) for value in (model, training, sampling)):
        raise TypeError("model, training, and sampling must be objects")
    if int(model.get("embedding_dim", 0)) < 2:
        raise ValueError("model.embedding_dim must be at least two")
    if int(training.get("steps", 0)) < 1:
        raise ValueError("training.steps must be positive")
    if float(training.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("training.learning_rate must be positive")
    if int(training.get("checkpoint_every", 0)) < 1:
        raise ValueError("training.checkpoint_every must be positive")
    artifact = spec.get("initial_parameter_artifact")
    if not isinstance(artifact, str) or not artifact.endswith(".json"):
        raise ValueError("initial_parameter_artifact must be a .json path")
    for name in (
        "chain_count",
        "burn_sweeps",
        "measurement_blocks",
        "decorrelation_sweeps",
        "amplitude_batch_size",
    ):
        if int(sampling.get(name, 0)) < 1:
            raise ValueError(f"sampling.{name} must be positive")


def _model_config(spec: dict[str, object]) -> SU2A1OperatorViTConfig:
    model = spec["model"]
    return SU2A1OperatorViTConfig(
        operator_labels=EFFECTIVE_LABELS,
        operator_descriptors=DESCRIPTORS,
        embedding_dim=int(model["embedding_dim"]),
        depth=2,
        coefficient_scale=float(model["coefficient_scale"]),
        initialization_scale=float(model["initialization_scale"]),
    )


def _initial_identity(spec: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "g1_su2_a1_operator_vit_initialization_v1",
        "effective_labels": list(EFFECTIVE_LABELS),
        "descriptors": [list(row) for row in DESCRIPTORS],
        "model": dict(spec["model"]),
        "seed": int(spec["seed"]),
    }


def materialize_g1_operator_vit_initialization(
    config_path: Path, *, overwrite: bool = False
) -> dict[str, object]:
    """Create the common CPU initialization used locally and remotely."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    params = init_su2_a1_operator_vit(
        jax.random.PRNGKey(int(spec["seed"])), _model_config(spec)
    )
    return save_parameter_artifact(
        ROOT / str(spec["initial_parameter_artifact"]),
        params,
        identity=_initial_identity(spec),
        overwrite=overwrite,
    )


def _load_initial(spec: dict[str, object]) -> object:
    template = init_su2_a1_operator_vit(
        jax.random.PRNGKey(int(spec["seed"])), _model_config(spec)
    )
    return load_parameter_artifact(
        ROOT / str(spec["initial_parameter_artifact"]),
        template,
        expected_identity=_initial_identity(spec),
    )


def _effective_contract(spec: dict[str, object]):
    coupling = float(spec["j2_over_j1"])
    resonance = str(spec["resonance_token"])
    expansions = {
        EFFECTIVE_LABELS[0]: {
            TRIANGULAR_A1_SCALAR_LABELS[0]: 1.0,
            TRIANGULAR_A1_SCALAR_LABELS[1]: coupling,
        },
        EFFECTIVE_LABELS[1]: {resonance: 1.0},
    }
    components = scalar_component_words(EFFECTIVE_LABELS)
    required_effective = tuple(
        dict.fromkeys(
            (
                *components,
                *((EFFECTIVE_LABELS[0], *word) for word in components),
            )
        )
    )
    required_base = expanded_base_words(required_effective, expansions)
    return expansions, components, required_effective, required_base


def _generalized_ritz(
    components: np.ndarray, h_components: np.ndarray, indices: tuple[int, ...]
) -> tuple[np.ndarray, dict[str, float]]:
    selected = np.asarray(indices, dtype=np.int64)
    x = np.asarray(components)[:, selected]
    hx = np.asarray(h_components)[:, selected]
    gram = x.conj().T @ x / len(x)
    hamiltonian = x.conj().T @ hx / len(x)
    gram = (gram + gram.conj().T) / 2.0
    hamiltonian = (hamiltonian + hamiltonian.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    cutoff = max(1.0e-13, float(eigenvalues[-1]) * 1.0e-10)
    keep = eigenvalues > cutoff
    whitener = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    projected = whitener.conj().T @ hamiltonian @ whitener
    energies, vectors = np.linalg.eigh((projected + projected.conj().T) / 2.0)
    coefficients = whitener @ vectors[:, 0]
    state, action = x @ coefficients, hx @ coefficients
    norm = float(np.mean(np.abs(state) ** 2))
    energy = complex(np.mean(np.conj(state) * action) / norm)
    variance = float(np.mean(np.abs(action - energy * state) ** 2) / norm)
    return coefficients, {
        "energy_total": float(energy.real),
        "variance_total": variance,
        "gram_rank": int(np.count_nonzero(keep)),
        "gram_condition_number": float(eigenvalues[keep][-1] / eigenvalues[keep][0]),
    }


def _exact_small(spec: dict[str, object], params: object) -> dict[str, object]:
    """Run the 4x4 exact-support gate for the same H+R two-layer family."""

    coupling = float(spec["j2_over_j1"])
    data = triangular_pi_flux_operator_components(coupling)
    resonance = triangular_a1_resonance_operators(
        SMALL_CASES["square4"], data.integer_basis
    )[str(spec["resonance_token"])]
    operators = (data.hamiltonian, resonance)
    means = np.asarray(
        [float(np.vdot(data.carrier, operator @ data.carrier).real) for operator in operators]
    )
    scales = np.asarray(
        [np.linalg.norm(operator @ data.carrier - mean * data.carrier) for operator, mean in zip(operators, means, strict=True)]
    )
    def centered_action(index: int, vector: np.ndarray) -> np.ndarray:
        return np.asarray(
            (operators[index] @ vector - means[index] * vector) / scales[index]
        )

    vectors = [data.carrier]
    first = [centered_action(index, data.carrier) for index in range(2)]
    vectors.extend(first)
    vectors.extend(
        centered_action(outer, first[inner])
        for outer in range(2)
        for inner in range(2)
    )
    components = np.column_stack(vectors)
    h_components = np.column_stack([data.hamiltonian @ vector for vector in vectors])
    config = _model_config(spec)
    initial = sampled_operator_vit_summary(params, components, h_components, config)
    training = dict(spec["training"])
    history = []
    optimizer_state = initialize_sampled_adam(params)
    for step in range(1, int(training["steps"]) + 1):
        value, gradient = jax.value_and_grad(sampled_operator_vit_energy)(
            params,
            jnp.asarray(components),
            jnp.asarray(h_components),
            config,
        )
        norm = float(
            np.sqrt(
                sum(float(np.sum(np.asarray(leaf) ** 2)) for leaf in jax.tree_util.tree_leaves(gradient))
            )
        )
        params, optimizer_state, diagnostics = sampled_adam_update(
            params,
            gradient,
            optimizer_state,
            learning_rate=float(training["learning_rate"]),
            max_relative_update=2.0,
        )
        history.append(
            {
                "step": step,
                "energy_total_before": float(value),
                "gradient_norm": norm,
                "relative_update_norm": diagnostics.relative_update_norm,
            }
        )
    final = sampled_operator_vit_summary(params, components, h_components, config)
    _, lanczos = _generalized_ritz(components, h_components, (0, 1, 3))
    _, oracle = _generalized_ritz(components, h_components, tuple(range(7)))
    state = components @ np.asarray(su2_a1_operator_vit_component_weights(params, config))
    energy, variance = _energy_variance(state, data.hamiltonian)
    return {
        "n_sites": 16,
        "initial": initial,
        "final": {**final, "energy_per_site": energy / 16.0, "variance_per_site": variance / 16.0},
        "two_lanczos_control": {**lanczos, "energy_per_site": lanczos["energy_total"] / 16.0},
        "full_effective_space_oracle": {**oracle, "energy_per_site": oracle["energy_total"] / 16.0},
        "history": history,
        "operator_coefficients": np.asarray(su2_a1_operator_vit_coefficients(params, config)).tolist(),
        "trained_params_sha256": pytree_sha256(params),
    }


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _raw_dataset(
    states: np.ndarray,
    *,
    carrier,
    base_words,
    output: Path,
    prefix: str,
    amplitude_batch_size: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    rows, ledger = [], []
    for index, state in enumerate(states):
        cache = output / "component_cache" / f"{prefix}_{index:04d}.npz"
        state_hash = array_sha256(np.asarray(state, dtype=np.int8))
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as stored:
                cached_state = np.asarray(stored["state"], dtype=np.int8)
                actions = np.asarray(stored["actions"], dtype=np.complex128)
            if array_sha256(cached_state) != state_hash or actions.shape != (1, len(base_words)):
                raise RuntimeError(f"invalid G1 component cache: {cache}")
            row_ledger = {"cache_reused": True}
        else:
            started = time.perf_counter()
            actions, evaluation = triangular_a1_word_actions(
                carrier,
                np.asarray(state, dtype=np.int8)[None, :],
                ((6, 0), (0, 6)),
                base_words,
                amplitude_batch_size=amplitude_batch_size,
            )
            _atomic_npz(cache, state=np.asarray(state, dtype=np.int8), actions=actions)
            row_ledger = {**evaluation, "cache_reused": False, "wall_seconds": time.perf_counter() - started}
        rows.append(actions[0])
        ledger.append({"sample_index": index, "state_sha256": state_hash, **row_ledger})
        print(f"G1 {prefix} sample {index + 1}/{len(states)} cache={row_ledger['cache_reused']}", flush=True)
    return np.asarray(rows), ledger


def _paired_chain_difference(
    first_state, first_action, second_state, second_action, *, chain_count: int
) -> dict[str, object]:
    """Compare two carrier-reweighted ratio estimators by chain jackknife.

    The audit configurations are sampled from the carrier, whereas each trial
    state has its own importance weights ``|state/carrier|**2``.  Averaging
    per-configuration local-energy differences would therefore estimate the
    wrong quantity.  We instead form the two self-normalized energy ratios on
    the full sample and use delete-one-persistent-chain estimates for a paired
    jackknife standard error.
    """

    arrays = tuple(
        np.asarray(value)
        for value in (first_state, first_action, second_state, second_action)
    )
    length = len(arrays[0])
    if chain_count < 1 or length < chain_count or length % chain_count != 0:
        raise ValueError("sample count must be a positive multiple of chain_count")
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("paired state and action arrays must have identical shapes")

    def difference(indices: np.ndarray) -> float:
        energies = []
        for state, action in ((arrays[0], arrays[1]), (arrays[2], arrays[3])):
            weights = np.abs(state[indices]) ** 2
            denominator = float(np.sum(weights))
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise FloatingPointError("invalid importance-weight denominator")
            numerator = np.sum(np.conj(state[indices]) * action[indices])
            energy = complex(numerator / denominator)
            if not np.isfinite(energy):
                raise FloatingPointError("non-finite self-normalized energy")
            energies.append(float(energy.real))
        return (energies[0] - energies[1]) / 36.0

    all_indices = np.arange(length, dtype=np.int64)
    estimate = difference(all_indices)
    leave_one_out = []
    if chain_count > 1:
        for chain in range(chain_count):
            keep = all_indices[all_indices % chain_count != chain]
            leave_one_out.append(difference(keep))
        jackknife_mean = float(np.mean(leave_one_out))
        standard_error = float(
            np.sqrt(
                (chain_count - 1.0)
                / chain_count
                * np.sum((np.asarray(leave_one_out) - jackknife_mean) ** 2)
            )
        )
    else:
        standard_error = 0.0
    return {
        "difference_per_site": estimate,
        "chain_standard_error_per_site": standard_error,
        "leave_one_chain_out_differences_per_site": leave_one_out,
        "uncertainty_method": "paired_delete_one_chain_jackknife_of_ratio_estimators",
    }


def run_spin_g1_operator_vit(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
    maximum_updates_this_invocation: int | None = None,
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    if overwrite and output.exists():
        for name in ("checkpoints", "component_cache"):
            candidate = output / name
            if candidate.is_dir():
                shutil.rmtree(candidate)
        if (output / "results.json").exists():
            (output / "results.json").unlink()
    params = _load_initial(spec)
    exact = _exact_small(spec, params)
    expansions, components_words, effective_words, base_words = _effective_contract(spec)
    profile_state = np.asarray([[index % 2 for index in range(36)]], dtype=np.int8)
    started = time.perf_counter()
    _, connectivity = triangular_a1_word_actions(
        lambda states: np.ones((len(states),), dtype=np.float64),
        profile_state,
        ((6, 0), (0, 6)),
        base_words,
        amplitude_batch_size=10_000_000,
    )
    connectivity["wall_seconds"] = time.perf_counter() - started
    if spec["execution_tier"] == "local_exact_connectivity_preflight":
        payload = {
            "schema_version": 1,
            "domain": "spin",
            "task": TASK,
            "status": "complete",
            "execution_tier": spec["execution_tier"],
            "config": spec,
            "initial_params_sha256": pytree_sha256(params),
            "exact_small": exact,
            "n36_connectivity_profile": connectivity,
            "decision": {
                "remote_g1_canary_licensed": exact["final"]["energy_total"] < exact["initial"]["energy_total"],
                "full_five_token_depth_three_rejected": True,
                "selected_effective_tokens": list(EFFECTIVE_LABELS),
            },
        }
        atomic_json(output / "results.json", payload)
        return payload

    sampling = dict(spec["sampling"])
    orbit = build_n36_pi_zero_flux_orbit()

    def carrier(states):
        return projected_pi_zero_flux_a1_amplitudes(
            states, orbit=orbit, microbatch_size=int(sampling["amplitude_batch_size"])
        ).amplitudes

    train = sample_fixed_wavefunction_exchange(
        carrier,
        n_sites=36,
        n_up=18,
        chain_count=int(sampling["chain_count"]),
        burn_sweeps=int(sampling["burn_sweeps"]),
        measurement_blocks=int(sampling["measurement_blocks"]),
        decorrelation_sweeps=int(sampling["decorrelation_sweeps"]),
        seed=int(spec["seed"]) + 101,
    )
    audit = sample_fixed_wavefunction_exchange(
        carrier,
        n_sites=36,
        n_up=18,
        chain_count=int(sampling["chain_count"]),
        burn_sweeps=int(sampling["burn_sweeps"]),
        measurement_blocks=int(sampling["measurement_blocks"]),
        decorrelation_sweeps=int(sampling["decorrelation_sweeps"]),
        seed=int(spec["seed"]) + 202,
    )
    train_raw, train_ledger = _raw_dataset(
        train.configurations, carrier=carrier, base_words=base_words, output=output, prefix="train", amplitude_batch_size=int(sampling["amplitude_batch_size"])
    )
    audit_raw, audit_ledger = _raw_dataset(
        audit.configurations, carrier=carrier, base_words=base_words, output=output, prefix="audit", amplitude_batch_size=int(sampling["amplitude_batch_size"])
    )
    train_effective = effective_word_actions(train_raw, base_words, effective_words, expansions)
    audit_effective = effective_word_actions(audit_raw, base_words, effective_words, expansions)
    means, scales, standardization = carrier_standardization_for_labels(
        train_effective, effective_words, EFFECTIVE_LABELS
    )
    train_components, train_h = centered_scalar_component_matrices(
        train_effective, effective_words, labels=EFFECTIVE_LABELS, hamiltonian_label=EFFECTIVE_LABELS[0], means=means, scales=scales
    )
    audit_components, audit_h = centered_scalar_component_matrices(
        audit_effective, effective_words, labels=EFFECTIVE_LABELS, hamiltonian_label=EFFECTIVE_LABELS[0], means=means, scales=scales
    )
    train_centers = train_effective[:, effective_words.index(())]
    audit_centers = audit_effective[:, effective_words.index(())]
    train_components, train_h = train_components / train_centers[:, None], train_h / train_centers[:, None]
    audit_components, audit_h = audit_components / audit_centers[:, None], audit_h / audit_centers[:, None]

    identity = {
        "task": TASK,
        "config_sha256": canonical_sha256(spec),
        "source_manifest_sha256": canonical_sha256(transitive_local_source_manifest((Path(__file__).resolve(),), root=ROOT, package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"))),
        "initial_params_sha256": pytree_sha256(params),
        "train_components_sha256": array_sha256(train_components),
        "audit_components_sha256": array_sha256(audit_components),
    }
    checkpoint_dir = output / "checkpoints"
    checkpoint = None
    if checkpoint_dir.is_dir() and not overwrite:
        checkpoint = load_latest_resumable_checkpoint(checkpoint_dir, expected_run_identity=identity)
    history = [] if checkpoint is None else list(checkpoint["history"])
    completed = 0 if checkpoint is None else int(checkpoint["completed_steps"])
    optimizer_state = initialize_sampled_adam(params)
    if checkpoint is not None:
        params = checkpoint["params"]
        optimizer_state = checkpoint["optimizer_state"]
        if not isinstance(optimizer_state, SampledAdamState):
            raise RuntimeError("G1 checkpoint optimizer state is invalid")
    total = int(spec["training"]["steps"])
    stop = total if maximum_updates_this_invocation is None else min(total, completed + int(maximum_updates_this_invocation))
    config = _model_config(spec)
    for step in range(completed + 1, stop + 1):
        value, gradient = jax.value_and_grad(sampled_operator_vit_energy)(params, jnp.asarray(train_components), jnp.asarray(train_h), config)
        norm = float(np.sqrt(sum(float(np.sum(np.asarray(leaf) ** 2)) for leaf in jax.tree_util.tree_leaves(gradient))))
        params, optimizer_state, diagnostics = sampled_adam_update(
            params,
            gradient,
            optimizer_state,
            learning_rate=float(spec["training"]["learning_rate"]),
            max_relative_update=2.0,
        )
        history.append({"step": step, "energy_total_before": float(value), "gradient_norm": norm, "relative_update_norm": diagnostics.relative_update_norm})
        if step % int(spec["training"]["checkpoint_every"]) == 0 or step == stop:
            save_resumable_checkpoint_generation(
                checkpoint_dir,
                params,
                optimizer_state=optimizer_state,
                completed_steps=step,
                next_sample_index=step * len(train_components),
                run_identity=identity,
                history=history,
                metadata={"fixed_train_sample_count": len(train_components), "fixed_audit_sample_count": len(audit_components)},
            )
    if stop < total:
        payload = {"schema_version": 1, "domain": "spin", "task": TASK, "status": "running", "completed_steps": stop, "total_steps": total, "run_identity": identity}
        atomic_json(output / "results.json", payload)
        return payload

    initial_params = _load_initial(spec)
    initial_audit = sampled_operator_vit_summary(initial_params, audit_components, audit_h, config)
    final_audit = sampled_operator_vit_summary(params, audit_components, audit_h, config)
    two_lanczos_coefficients, two_lanczos_train = _generalized_ritz(train_components, train_h, (0, 1, 3))
    control_state = audit_components[:, (0, 1, 3)] @ two_lanczos_coefficients
    control_action = audit_h[:, (0, 1, 3)] @ two_lanczos_coefficients
    control_norm = float(np.mean(np.abs(control_state) ** 2))
    control_energy = complex(np.mean(np.conj(control_state) * control_action) / control_norm)
    control_variance = float(np.mean(np.abs(control_action - control_energy * control_state) ** 2) / control_norm)
    weights = np.asarray(su2_a1_operator_vit_component_weights(params, config))
    operator_state, operator_action = audit_components @ weights, audit_h @ weights
    paired = _paired_chain_difference(
        operator_state, operator_action, control_state, control_action, chain_count=int(sampling["chain_count"])
    )
    _, oracle_train = _generalized_ritz(train_components, train_h, tuple(range(7)))
    payload = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_tier": spec["execution_tier"],
        "config": spec,
        "model_case": {"hamiltonian": "triangular J1-J2 Heisenberg", "boundary": "6x6 periodic", "target_sector": "exact S=0, Gamma-A1", "carrier": "projected pi/0-flux", "effective_tokens": list(EFFECTIVE_LABELS)},
        "initial_params_sha256": identity["initial_params_sha256"],
        "final_params_sha256": pytree_sha256(params),
        "parameter_count": su2_a1_operator_vit_parameter_count(params),
        "exact_small": exact,
        "n36_connectivity_profile": connectivity,
        "standardization": {"means": means.tolist(), "scales": scales.tolist(), **standardization},
        "initial_audit": initial_audit,
        "final_audit": {**final_audit, "energy_per_site": final_audit["energy_total"] / 36.0, "variance_per_site": final_audit["variance_total"] / 36.0},
        "two_lanczos_control": {**two_lanczos_train, "heldout_energy_total": float(control_energy.real), "heldout_energy_per_site": float(control_energy.real / 36.0), "heldout_variance_total": control_variance},
        "full_effective_space_training_oracle": oracle_train,
        "paired_operator_minus_two_lanczos": paired,
        "operator_coefficients": np.asarray(su2_a1_operator_vit_coefficients(params, config)).tolist(),
        "training_history": history,
        "sample_identity": {"train_sha256": array_sha256(train.configurations), "audit_sha256": array_sha256(audit.configurations), "train_count": len(train.configurations), "audit_count": len(audit.configurations)},
        "work_ledger": {
            "train_component_rows": train_ledger,
            "audit_component_rows": audit_ledger,
            "operator_component_precompute_centers_training": len(train_components),
            "operator_component_precompute_centers_audit": len(audit_components),
            "neural_optimizer_component_matrix_rows": total * len(train_components),
            "additional_full_h_calls_per_optimizer_step": 0,
        },
        "literature_targets": {"exact_energy_per_site": N36_EXACT_ENERGY_PER_SITE, "gcnn_energy_per_site": N36_GCNN_ENERGY_PER_SITE},
        "runtime_identity": numerical_runtime_identity(),
        "decision": {"energy_below_two_lanczos": paired["difference_per_site"] < 0.0, "paired_resolved": abs(paired["difference_per_site"]) > 2.0 * paired["chain_standard_error_per_site"], "production_claim_licensed": paired["difference_per_site"] + 2.0 * paired["chain_standard_error_per_site"] < 0.0 and final_audit["variance_total"] < control_variance},
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(output / "results.json", payload)
    return payload


__all__ = [
    "TASK",
    "materialize_g1_operator_vit_initialization",
    "run_spin_g1_operator_vit",
]
