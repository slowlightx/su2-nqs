#!/usr/bin/env python3
"""Re-audit a completed G1 result from its immutable component caches.

The training configurations are sampled from the projected carrier rather
than either optimized trial state.  Energies are consequently self-normalized
importance-sampling ratios.  This analyzer reconstructs the matrices without
new carrier evaluations and reports paired delete-one-chain jackknife errors;
it must be used instead of averaging single-configuration local energies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.workflows.spin_g1_operator_vit import (
    _effective_contract,
    _generalized_ritz,
    _model_config,
    _paired_chain_difference,
)
from su2_nqs.benchmarks.operator_vit_sampled import (
    centered_scalar_component_matrices,
    effective_word_actions,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.models.su2_a1_operator_vit import (
    su2_a1_operator_vit_coefficients,
)
from su2_nqs.vmc.resumable_checkpoint import load_resumable_checkpoint


def _cached_rows(root: Path, prefix: str, count: int) -> np.ndarray:
    rows = []
    for index in range(count):
        path = root / "component_cache" / f"{prefix}_{index:04d}.npz"
        with np.load(path, allow_pickle=False) as stored:
            actions = np.asarray(stored["actions"], dtype=np.complex128)
        if actions.ndim != 2 or actions.shape[0] != 1:
            raise ValueError(f"invalid component cache shape: {path}")
        rows.append(actions[0])
    return np.asarray(rows)


def _matrices(
    raw: np.ndarray,
    *,
    base_words,
    effective_words,
    expansions,
    means: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    effective = effective_word_actions(
        raw, base_words, effective_words, expansions
    )
    components, hamiltonian = centered_scalar_component_matrices(
        effective,
        effective_words,
        labels=("H_J1J2", "A1_resonance"),
        hamiltonian_label="H_J1J2",
        means=means,
        scales=scales,
    )
    centers = effective[:, effective_words.index(())]
    return components / centers[:, None], hamiltonian / centers[:, None]


def _ratio_summary(
    state: np.ndarray,
    action: np.ndarray,
    *,
    chain_count: int,
    n_sites: int,
) -> dict[str, object]:
    indices = np.arange(len(state), dtype=np.int64)

    def energy(selected: np.ndarray) -> float:
        numerator = np.sum(np.conj(state[selected]) * action[selected])
        denominator = np.sum(np.abs(state[selected]) ** 2)
        return float(np.real(numerator / denominator) / n_sites)

    estimate = energy(indices)
    leave_one_out = []
    if chain_count > 1:
        for chain in range(chain_count):
            keep = indices[indices % chain_count != chain]
            leave_one_out.append(energy(keep))
        center = float(np.mean(leave_one_out))
        standard_error = float(
            np.sqrt(
                (chain_count - 1.0)
                / chain_count
                * np.sum((np.asarray(leave_one_out) - center) ** 2)
            )
        )
    else:
        standard_error = 0.0
    probability = np.abs(state) ** 2
    normalized = probability / np.sum(probability)
    total_energy = estimate * n_sites
    variance_total = float(
        np.sum(np.abs(action - total_energy * state) ** 2)
        / np.sum(probability)
    )
    return {
        "energy_per_site": estimate,
        "jackknife_standard_error_per_site": standard_error,
        "variance_total": variance_total,
        "importance_ess_fraction": float(
            1.0 / np.sum(normalized**2) / len(normalized)
        ),
        "maximum_normalized_weight": float(np.max(normalized)),
        "leave_one_chain_out_energy_per_site": leave_one_out,
    }


def analyze(root: Path) -> dict[str, object]:
    source = json.loads((root / "results.json").read_text())
    if source.get("status") != "complete" or source.get("task") != "spin_g1_operator_vit":
        raise ValueError("G1 result must be complete")
    sealed = dict(source)
    recorded_hash = sealed.pop("result_sha256")
    if canonical_sha256(sealed) != recorded_hash:
        raise ValueError("G1 result hash does not match its payload")

    spec = source["config"]
    expansions, _, effective_words, base_words = _effective_contract(spec)
    counts = source["sample_identity"]
    train_raw = _cached_rows(root, "train", int(counts["train_count"]))
    audit_raw = _cached_rows(root, "audit", int(counts["audit_count"]))
    means = np.asarray(source["standardization"]["means"], dtype=np.float64)
    scales = np.asarray(source["standardization"]["scales"], dtype=np.float64)
    train_components, train_h = _matrices(
        train_raw,
        base_words=base_words,
        effective_words=effective_words,
        expansions=expansions,
        means=means,
        scales=scales,
    )
    audit_components, audit_h = _matrices(
        audit_raw,
        base_words=base_words,
        effective_words=effective_words,
        expansions=expansions,
        means=means,
        scales=scales,
    )

    coefficients = np.asarray(source["operator_coefficients"], dtype=np.float64)
    if coefficients.shape != (2, 2):
        raise ValueError("G1 operator coefficient shape changed")
    first, second = coefficients
    weights = np.concatenate((np.ones(1), first + second, np.outer(second, first).reshape(-1)))
    train_operator_state = train_components @ weights
    train_operator_action = train_h @ weights
    operator_state = audit_components @ weights
    operator_action = audit_h @ weights
    control_coefficients, _ = _generalized_ritz(
        train_components, train_h, (0, 1, 3)
    )
    train_control_state = train_components[:, (0, 1, 3)] @ control_coefficients
    train_control_action = train_h[:, (0, 1, 3)] @ control_coefficients
    control_state = audit_components[:, (0, 1, 3)] @ control_coefficients
    control_action = audit_h[:, (0, 1, 3)] @ control_coefficients

    chain_count = int(spec["sampling"]["chain_count"])
    train_operator = _ratio_summary(
        train_operator_state,
        train_operator_action,
        chain_count=chain_count,
        n_sites=36,
    )
    train_control = _ratio_summary(
        train_control_state,
        train_control_action,
        chain_count=chain_count,
        n_sites=36,
    )
    operator = _ratio_summary(
        operator_state, operator_action, chain_count=chain_count, n_sites=36
    )
    control = _ratio_summary(
        control_state, control_action, chain_count=chain_count, n_sites=36
    )
    paired = _paired_chain_difference(
        operator_state,
        operator_action,
        control_state,
        control_action,
        chain_count=chain_count,
    )
    exact = float(source["literature_targets"]["exact_energy_per_site"])
    operator_reference_consistent = (
        float(operator["energy_per_site"])
        + 2.0 * float(operator["jackknife_standard_error_per_site"])
        >= exact
    )
    control_reference_consistent = (
        float(control["energy_per_site"])
        + 2.0 * float(control["jackknife_standard_error_per_site"])
        >= exact
    )
    resolved_gain = (
        float(paired["difference_per_site"])
        + 2.0 * float(paired["chain_standard_error_per_site"])
        < 0.0
    )
    lower_variance = float(operator["variance_total"]) < float(
        control["variance_total"]
    )
    checkpoint_rows = []
    model_config = _model_config(spec)
    for checkpoint_path in sorted((root / "checkpoints").glob("step_*.pkl")):
        checkpoint = load_resumable_checkpoint(checkpoint_path)
        checkpoint_coefficients = np.asarray(
            su2_a1_operator_vit_coefficients(checkpoint["params"], model_config),
            dtype=np.float64,
        )
        checkpoint_first, checkpoint_second = checkpoint_coefficients
        checkpoint_weights = np.concatenate(
            (
                np.ones(1),
                checkpoint_first + checkpoint_second,
                np.outer(checkpoint_second, checkpoint_first).reshape(-1),
            )
        )
        checkpoint_train = _ratio_summary(
            train_components @ checkpoint_weights,
            train_h @ checkpoint_weights,
            chain_count=chain_count,
            n_sites=36,
        )
        checkpoint_audit = _ratio_summary(
            audit_components @ checkpoint_weights,
            audit_h @ checkpoint_weights,
            chain_count=chain_count,
            n_sites=36,
        )
        checkpoint_rows.append(
            {
                "completed_steps": int(checkpoint["completed_steps"]),
                "training_energy_per_site": checkpoint_train["energy_per_site"],
                "audit_energy_per_site": checkpoint_audit["energy_per_site"],
                "audit_jackknife_standard_error_per_site": checkpoint_audit[
                    "jackknife_standard_error_per_site"
                ],
                "audit_variance_total": checkpoint_audit["variance_total"],
                "audit_importance_ess_fraction": checkpoint_audit[
                    "importance_ess_fraction"
                ],
                "coefficients": checkpoint_coefficients.tolist(),
            }
        )
    exploratory_best = (
        min(checkpoint_rows, key=lambda row: float(row["audit_energy_per_site"]))
        if checkpoint_rows
        else None
    )
    payload = {
        "schema_version": 1,
        "task": "spin_g1_operator_vit_paired_reanalysis",
        "source_result_sha256": recorded_hash,
        "uncertainty_method": "delete_one_persistent_chain_jackknife_of_self_normalized_ratio_estimators",
        "training_operator_vit": train_operator,
        "training_two_lanczos_control": train_control,
        "operator_vit": operator,
        "two_lanczos_control": control,
        "audit_minus_training_energy_per_site": {
            "operator_vit": float(operator["energy_per_site"])
            - float(train_operator["energy_per_site"]),
            "two_lanczos_control": float(control["energy_per_site"])
            - float(train_control["energy_per_site"]),
        },
        "paired_operator_minus_two_lanczos": paired,
        "checkpoint_trajectory": checkpoint_rows,
        "exploratory_best_audit_checkpoint": exploratory_best,
        "checkpoint_selection_warning": (
            "The audit set was inspected after training. Its best checkpoint "
            "is diagnostic only and cannot be used as a variational claim or "
            "as a preregistered early-stopping selection."
        ),
        "literature_exact_energy_per_site": exact,
        "decision": {
            "operator_reference_consistent_at_2se": operator_reference_consistent,
            "control_reference_consistent_at_2se": control_reference_consistent,
            "operator_gain_resolved_at_2se": resolved_gain,
            "operator_variance_lower": lower_variance,
            "production_claim_licensed": (
                operator_reference_consistent
                and control_reference_consistent
                and resolved_gain
                and lower_variance
            ),
        },
    }
    payload["analysis_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    payload["analysis_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = analyze(args.result_root)
    atomic_json(args.output, payload)
    print(json.dumps(payload["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
