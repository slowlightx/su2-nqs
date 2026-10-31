"""Local-only controller comparison on frozen full-A1/H tangent caches.

This consumes a physical dictionary produced by ``projected_residual_probe``;
it neither queries H nor trains the CT state. Outputs are fixed candidate
directions for a later independent finite-amplitude/full-H check. All current
metrics are restricted to the cached parent Born support, not exact nodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.projected_capacity import (
    tangent_moments, budgeted_multiplet_actions, conditional_greedy_actions,
    restricted_natural_direction, heldout_direction_metrics,
)
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.models.capacity_attention import (
    CapacityControllerConfig, init_capacity_controller,
    predict_capacity_coefficients, capacity_quadratic_loss,
)
from su2_nqs.vmc.adam import _manual_adam_init, _manual_adam_update
from su2_nqs.vmc.checkpoint import save_checkpoint, load_checkpoint

SEEDS = (74121, 74122, 74123)
STEPS = 250
LEARNING_RATE = .01
RIDGE = .01
FS_RADIUS = .001
NATIVE_BUDGET = 14000
MAGNETIC_BUDGET = 10
MODES = ("linear", "bilinear", "pointwise", "pooled", "attention", "attention_scrambled")


def _screening_feature_means(panel: dict, screening: np.ndarray) -> np.ndarray:
    """Six declared q-by-kind averages of signed *other-edge* screening."""
    columns = []
    for q in (1, 3, 5):
        for kind in (0, 1):
            mask = ((panel["token_q"] == q)[None, :]
                    & (panel["token_kind"] == kind)[None, :]
                    & (panel["token_nodes"][:, None] != panel["token_nodes"][None, :]))
            columns.append(np.sum(np.where(mask, screening, 0.), axis=1)
                           /np.maximum(np.sum(mask, axis=1), 1))
    return np.stack(columns, axis=-1)


def prepare_panel(path: Path, *, direction_features: bool = False,
                  screening_context: bool = False) -> dict:
    """Read raw cached scores, never producer forces with another convention."""
    with np.load(path, allow_pickle=False) as saved:
        panel = {name: saved[name] for name in saved.files}
    panel["name"] = path.parent.name
    panel["path"] = str(path)
    metadata_path = path.parent/"registration.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if "feature_names" in metadata:
        panel["feature_names"] = np.asarray(metadata["feature_names"])
    # H changes without a parent change cannot be inferred from state-only
    # features. Coupling is known physical input, not a residual target.
    if "j2" in metadata:
        panel["token_features"] = np.column_stack((panel["token_features"],
            np.full(panel["token_features"].shape[0], float(metadata["j2"]))))
        panel["feature_names"] = np.concatenate((panel.get("feature_names", np.array([], str)),
                                                 np.asarray(["j2_over_j1"])))
    if direction_features:
        sidecar = path.parent/"raw_pair_direction_response.npz"
        with np.load(sidecar, allow_pickle=False) as saved:
            responses = saved["raw_pair_direction_response"]
        if responses.shape != (panel["token_features"].shape[0], 36) or not np.isfinite(responses).all():
            raise ValueError("Declared 36 scalar local-pair direction responses required")
        panel["token_features"] = np.column_stack((panel["token_features"], responses))
        panel["feature_names"] = np.concatenate((panel["feature_names"],
            np.asarray([f"delta_pair_{index}" for index in range(36)])))
        panel["direction_feature_source"] = dict(path=str(sidecar),
            sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest())
    if "pair_relations" in panel:
        panel["token_pair_relations"] = panel["pair_relations"]
    required = ("scores", "local_energies", "fit_mask", "token_features", "token_q",
                "token_cost", "token_nodes")
    for name in required:
        if name not in panel:
            raise ValueError(f"Missing cache field {name} in {path}")
    for split, mask in (("fit", panel["fit_mask"].astype(bool)),
                        ("audit", ~panel["fit_mask"].astype(bool))):
        if np.count_nonzero(mask) < 2:
            raise ValueError("Two disjoint nonempty fit/audit sets required")
        panel[f"{split}_gram"], panel[f"{split}_force"] = tangent_moments(
            panel["scores"][mask], panel["local_energies"][mask])
    if "chain_labels" in panel:
        fit = set(panel["chain_labels"][panel["fit_mask"].astype(bool)].tolist())
        audit = set(panel["chain_labels"][~panel["fit_mask"].astype(bool)].tolist())
        if fit & audit:
            raise ValueError("Fit/audit split leaks chains")
    count = panel["scores"].shape[1]
    if panel["token_features"].shape[0] != count:
        raise ValueError("One cheap token per actual full-A1 tangent required")
    if "token_pair_relations" not in panel:
        # Same edge only is a declared topology feature, NOT spatial geometry.
        nodes = panel["token_nodes"]
        panel["token_pair_relations"] = (nodes[:, None] == nodes[None, :])[..., None].astype(float)
        panel["relation_semantics"] = "same_edge_only; no spatial-geometry claim"
    else:
        panel["relation_semantics"] = "producer-fixed pair relations; inspect cache metadata"
    if screening_context:
        if not direction_features:
            raise ValueError("v4 screening case is the predeclared augmentation of v3 direction features")
        sidecar = path.parent/"screening_context.npz"
        with np.load(sidecar, allow_pickle=False) as saved:
            context = saved["screening_context"]
            scrambled = saved["screening_context_scrambled"]
        if (context.shape != (count, count) or scrambled.shape != context.shape
                or not np.isfinite(context).all() or not np.isfinite(scrambled).all()):
            raise ValueError("One finite signed pair-screening relation per action pair required")
        panel["screening_feature_start"] = panel["token_features"].shape[1]
        panel["token_features"] = np.column_stack((panel["token_features"], _screening_feature_means(panel, context)))
        panel["feature_names"] = np.concatenate((panel["feature_names"], np.asarray([
            f"screening_mean_q{q}_{kind}" for q in (1, 3, 5) for kind in ("delete", "rotate")])))
        panel["token_pair_relations"] = np.concatenate((panel["token_pair_relations"], context[..., None]), axis=-1)
        panel["screening_context"] = context
        panel["screening_context_scrambled"] = scrambled
        panel["screening_context_source"] = dict(path=str(sidecar),
            sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            semantics="R_ab=f_a^T [3 Cov_fit(Sz,Sz)]_AB f_b/(norm(f_a) norm(f_b)); raw f, projected FIT rows only")
    return panel


def _fit_controller(training: list[dict], mode: str, seed: int, output: Path, tag: str) -> tuple:
    """Use fit parents/moments only; cache scaling cannot inspect audit inputs."""
    features = np.stack([p["token_features"] for p in training])
    screening_scramble = mode == "attention_scrambled" and "screening_context" in training[0]
    if screening_scramble:
        for index, panel in enumerate(training):
            features[index, :, panel["screening_feature_start"]:] = _screening_feature_means(
                panel, panel["screening_context_scrambled"])
    mean, scale = features.mean(axis=(0, 1)), features.std(axis=(0, 1))
    active = scale > 1e-12
    scale = np.where(active, scale, 1.)
    features = np.where(active, (features-mean)/scale, 0.)
    relations = np.stack([p["token_pair_relations"] for p in training])
    if screening_scramble:
        for index, panel in enumerate(training):
            relations[index, :, :, -1] = panel["screening_context_scrambled"]
    mixing = "attention" if mode == "attention_scrambled" else mode
    # Pointwise width5 has 291 parameters versus genuine attention width4's
    # 293 for the registered 47 scalar / four relation features.
    width = 5 if mode == "pointwise" else 4
    config = CapacityControllerConfig(features.shape[-1], width=width,
        relation_dim=relations.shape[-1], mixing=mixing)
    params = init_capacity_controller(jax.random.PRNGKey(seed), config)
    rng = np.random.default_rng(seed)
    nodes = training[0]["token_nodes"]
    node_order = np.unique(nodes)
    permuted_nodes = rng.permutation(node_order)
    permutation = np.arange(len(nodes))
    for a, b in zip(node_order, permuted_nodes, strict=True):
        left, right = np.flatnonzero(nodes == a), np.flatnonzero(nodes == b)
        if len(left) != len(right):
            raise ValueError("geometry scramble requires matching action lists on each node")
        permutation[left] = right
    if mode == "attention_scrambled" and not screening_scramble:
        # Scramble spatial bond geometry only, not same-q / same-edge relations.
        relations[:, :, :, 2:] = relations[:, permutation][:, :, permutation, 2:]
    physical_scales = np.stack([np.sqrt(np.maximum(np.diag(p["fit_gram"]), 1e-24)) for p in training])
    gram = np.stack([p["fit_gram"] for p in training])/physical_scales[:, :, None]/physical_scales[:, None, :]
    force = np.stack([p["fit_force"] for p in training])/physical_scales
    force_scale = np.maximum(np.linalg.norm(force, axis=1)/np.sqrt(force.shape[1]), 1e-12)
    force /= force_scale[:, None]
    features, relations, gram, force = map(jnp.asarray, (features, relations, gram, force))

    def loss(p):
        values = jax.vmap(lambda x, r: predict_capacity_coefficients(p, x, config, relations=r))(features, relations)
        return jnp.mean(jax.vmap(lambda d, s, g: capacity_quadratic_loss(d, s, g, ridge=RIDGE))(values, gram, force))

    @jax.jit
    def step(p, state):
        value, gradient = jax.value_and_grad(loss)(p)
        p, state = _manual_adam_update(p, gradient, state, LEARNING_RATE)
        return p, state, value

    state = _manual_adam_init(params)
    losses = []
    start = time.perf_counter()
    for index in range(STEPS):
        params, state, value = step(params, state)
        if index in (0, STEPS//2-1, STEPS-1):
            losses.append(float(value))
    jax.block_until_ready(params)
    elapsed = time.perf_counter()-start
    checkpoint = output/f"{tag}_{mode}_{seed}.pkl"
    save_checkpoint(checkpoint, params, optimizer_state=state,
        metadata={"steps": STEPS, "seed": seed, "mean": mean.tolist(), "scale": scale.tolist(),
                  "active": active.tolist(), "mode": mode, "permutation": permutation.tolist(),
                  "force_convention": "Re Cov(O,E), NOT 2 Cov", "tangent_only": True})
    restored = load_checkpoint(checkpoint)
    # Check parameters AND optimizer replay by one additional uncommitted step.
    next_params, next_state, _ = step(params, state)
    restored_params, restored_state, _ = step(restored["params"], restored["optimizer_state"])
    discrepancy = max(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) for a, b in zip(
        jax.tree.leaves((next_params, next_state)), jax.tree.leaves((restored_params, restored_state)), strict=True))
    receipt = dict(mode=mode, seed=seed, steps=STEPS, losses=losses,
        parameters=sum(int(a.size) for a in jax.tree.leaves(params)), seconds=elapsed,
        checkpoint=str(checkpoint), optimizer_replay_maximum_error=discrepancy,
        scramble_semantics=("screening R plus its six pooled features; original geometry unchanged"
                            if screening_scramble else "geometry channels only (CT6 invariant)" if mode == "attention_scrambled" else None),
        inactive_feature_indices=np.flatnonzero(~active).tolist())
    print("controller", mode, seed, "loss", losses[-1], "seconds", elapsed, flush=True)
    return params, config, mean, scale, active, permutation, receipt


def compare_cached_panels(paths: list[Path], output: Path, *, fit_count: int = 6,
                          direction_features: bool = False, screening_context: bool = False) -> dict:
    """Freeze controllers/selection on fit data and evaluate each audit once."""
    output.mkdir(parents=True, exist_ok=True)
    panels = [prepare_panel(path, direction_features=direction_features,
                           screening_context=screening_context) for path in paths]
    if not 1 <= fit_count < len(panels):
        raise ValueError("fit_count must leave at least one entirely untouched parent")
    for index, panel in enumerate(panels):
        panel["is_heldout"] = index >= fit_count
    modes = MODES+("signed_graph",) if screening_context else MODES
    registration = dict(seeds=SEEDS, steps=STEPS, learning_rate=LEARNING_RATE, ridge=RIDGE,
        fit_fs_radius=FS_RADIUS, native_budget=NATIVE_BUDGET,
        magnetic_budget=MAGNETIC_BUDGET, modes=modes,
        force_convention="g = Re Cov(O,E_local); full energy slope = 2 g.dot(d)",
        metric_scope="parent-Born-support sample tangent; exact nodes not observed",
        target="physical quadratic d.S.d+2g.d; no synthetic teacher",
        finite_energy_or_variance_certified=False,
        actual_capacity_or_padded_cost_reduction_certified=False,
        cross_parent_scope="first fit_count older parents train; later complete parents untouched",
        parent_holdout_applies_to_controller_fit_only=True,
        target_fit_moments_used_for_final_direction=True,
        fit_count=fit_count,
        input_features="raw local scalars+known J2" + ("+36 cheap scalar direction responses" if direction_features else ""),
        direction_feature_sources=[p.get("direction_feature_source") for p in panels],
        screening_context_sources=[p.get("screening_context_source") for p in panels],
        screening_feature_means_order="q=(1,3,5), each delete/rotate; mean of signed R to other-edge actions",
        development_followup_after_v3_outcomes=screening_context,
        relation_semantics=[p["relation_semantics"] for p in panels],
        cache_paths=[str(p) for p in paths])
    atomic_json(output/"registration.json", registration)
    rows, directions = [], {}

    def evaluate(panel: dict, merit: np.ndarray, label: str, metadata: dict, *, conditional: bool = False):
        budget = NATIVE_BUDGET
        selector = conditional_greedy_actions if conditional else budgeted_multiplet_actions
        args = (panel["fit_gram"], panel["fit_force"], panel["token_cost"], panel["token_q"]) if conditional else (merit, panel["token_cost"], panel["token_q"])
        selected = selector(*args, native_budget=budget, magnetic_budget=MAGNETIC_BUDGET)
        direction = restricted_natural_direction(panel["fit_gram"], panel["fit_force"], selected, ridge=RIDGE, fs_radius=FS_RADIUS)
        fit = heldout_direction_metrics(direction, panel["fit_gram"], panel["fit_force"])
        audit = heldout_direction_metrics(direction, panel["audit_gram"], panel["audit_force"])
        audit_mask = ~panel["fit_mask"].astype(bool)
        chains = np.unique(panel["chain_labels"][audit_mask])
        jackknife = []
        for chain in chains:
            mask = audit_mask & (panel["chain_labels"] != chain)
            jack_gram, jack_force = tangent_moments(panel["scores"][mask], panel["local_energies"][mask])
            jackknife.append(heldout_direction_metrics(direction, jack_gram, jack_force)["energy_slope_per_fs_total"])
        jackknife = np.asarray(jackknife)
        jackknife_se = float(np.sqrt((len(chains)-1)/len(chains)*np.sum((jackknife-jackknife.mean())**2)))
        name = f"{panel['name']}__{label}"
        directions[name] = direction
        rows.append(dict(parent=panel["name"], arm=label, selected=selected.tolist(),
            controller_parent_heldout=panel["is_heldout"],
            target_fit_moments_used_for_final_direction=True,
            native_budget=budget, magnetic_budget=MAGNETIC_BUDGET,
            native_cost=int(np.sum(panel["token_cost"][selected])),
            magnetic_cost=int(np.sum(panel["token_q"][selected]+1)),
            fit=fit, audit=audit, audit_fit_fs_ratio=audit["fs_tangent"]/fit["fs_tangent"] if fit["fs_tangent"] else 0.,
            audit_chain_jackknife_slope_per_fs=jackknife.tolist(),
            audit_slope_per_fs_chain_jackknife_se=jackknife_se,
            **metadata))

    for panel in panels:
        diagonal = np.maximum(np.diag(panel["fit_gram"]), 1e-24)
        evaluate(panel, np.abs(panel["fit_force"])/np.sqrt(diagonal), "free_diagonal_force", {})
        evaluate(panel, np.zeros_like(diagonal), "free_conditional_greedy", {}, conditional=True)
        if "token_raw_weight" in panel:
            evaluate(panel, panel["token_raw_weight"], "raw_weight", {})
        else:
            # Feature column identity must be explicit, not inferred from labels.
            names = panel.get("feature_names", np.array([], dtype=str)).tolist()
            if "mode_weight" in names:
                evaluate(panel, panel["token_features"][:, names.index("mode_weight")], "raw_weight", {})
            else:
                raise ValueError("Required raw_weight control lacks explicit mode_weight feature provenance")

    # One shared controller trains on the old parents' FIT CHAINS only. Later
    # complete parents, including their cheap features, never fit or tune it.
    training = panels[:fit_count]
    diagonals = np.stack([np.sqrt(np.maximum(np.diag(p["fit_gram"]), 1e-24)) for p in training])
    metric_average = np.mean(np.stack([p["fit_gram"] for p in training])/diagonals[:, :, None]/diagonals[:, None, :], axis=0)
    forces = np.stack([p["fit_force"] for p in training])/diagonals
    forces /= np.maximum(np.linalg.norm(forces, axis=1)/np.sqrt(forces.shape[1]), 1e-12)[:, None]
    force_average = forces.mean(axis=0)
    shared_free = -np.linalg.pinv(metric_average+RIDGE*np.diag(np.diag(metric_average)), rcond=1e-10) @ force_average
    for panel in panels:
        evaluate(panel, abs(shared_free), "shared_free_per_token", dict(
            parameters=len(shared_free), train_parents=[p["name"] for p in training],
            whole_parent_held_out=panel["is_heldout"]))
    for mode in modes:
        for seed in SEEDS:
            fitted = _fit_controller(training, mode, seed, output, "shared")
            params, config, mean, scale, active, permutation, receipt = fitted
            for test_index, target in enumerate(panels):
                raw_features = target["token_features"].copy()
                relations = target["token_pair_relations"].copy()
                if mode == "attention_scrambled" and screening_context:
                    raw_features[:, target["screening_feature_start"]:] = _screening_feature_means(
                        target, target["screening_context_scrambled"])
                    relations[:, :, -1] = target["screening_context_scrambled"]
                elif mode == "attention_scrambled":
                    relations[:, :, 2:] = relations[permutation][:, permutation, 2:]
                features = np.where(active, (raw_features-mean)/scale, 0.)
                coefficients = np.asarray(predict_capacity_coefficients(params, jnp.asarray(features), config, relations=jnp.asarray(relations)))
                metadata = dict(controller=receipt, train_parents=[p["name"] for p in training],
                    whole_parent_held_out=test_index >= fit_count,
                    feature_ood_maximum_standardized=float(np.max(abs(features))))
                evaluate(target, abs(coefficients), f"{mode}_{seed}", metadata)
    np.savez_compressed(output/"directions.npz", **directions)
    result = dict(registration=registration, rows=rows,
        conclusion="Tangent prioritization only; independent finite amplitudes/H required for energy or variance claims.")
    atomic_json(output/"summary.json", result)
    return result


def summarize_controller_comparisons(summaries: dict[str, Path], output: Path) -> dict:
    """Compare ALL seeds on the SAME held-out chains, without selecting a winner.

    Jackknife SE of a seed-averaged difference averages the paired delete-chain
    values *before* computing SE. Seeds are not independent physical samples.
    Their spread is reported separately. None of these are finite energies.
    """
    differences, arms = [], []
    controls = ("linear", "pointwise", "pooled", "signed_graph", "attention_scrambled",
                "free_conditional_greedy", "shared_free_per_token", "raw_weight")

    def jackknife_se(values: np.ndarray) -> float:
        return float(np.sqrt((len(values)-1)/len(values)*np.sum((values-values.mean())**2)))

    for case, path in summaries.items():
        summary = json.loads(path.read_text())
        rows = summary["rows"]
        parents = sorted({r["parent"] for r in rows if r.get("controller_parent_heldout", False)})
        for parent in parents:
            lookup = {r["arm"]: r for r in rows if r["parent"] == parent}
            for mode in summary["registration"]["modes"]:
                selected = [lookup[f"{mode}_{seed}"] for seed in SEEDS]
                values = np.asarray([r["audit"]["energy_slope_per_fs_total"] for r in selected])
                jacks = np.asarray([r["audit_chain_jackknife_slope_per_fs"] for r in selected])
                arms.append(dict(case=case, parent=parent, mode=mode,
                    seed_average_slope_per_fs=float(values.mean()),
                    seed_average_paired_chain_se=jackknife_se(jacks.mean(axis=0)),
                    seed_range=values.tolist(),
                    audit_fit_fs_ratio=[r["audit_fit_fs_ratio"] for r in selected],
                    actual_native_cost=[r["native_cost"] for r in selected],
                    parameters=selected[0]["controller"]["parameters"]))
            for control in controls:
                if control not in lookup and f"{control}_{SEEDS[0]}" not in lookup:
                    continue
                seed_values, seed_jacks, records = [], [], []
                for seed in SEEDS:
                    attention = lookup[f"attention_{seed}"]
                    other = lookup.get(f"{control}_{seed}", lookup.get(control))
                    delta = (attention["audit"]["energy_slope_per_fs_total"]
                             -other["audit"]["energy_slope_per_fs_total"])
                    jack = (np.asarray(attention["audit_chain_jackknife_slope_per_fs"])
                            -np.asarray(other["audit_chain_jackknife_slope_per_fs"]))
                    seed_values.append(delta); seed_jacks.append(jack)
                    records.append(dict(seed=seed, attention_minus_control=delta,
                        paired_chain_jackknife_se=jackknife_se(jack),
                        attention_fs_ratio=attention["audit_fit_fs_ratio"],
                        control_fs_ratio=other["audit_fit_fs_ratio"],
                        native_cost_difference=attention["native_cost"]-other["native_cost"]))
                differences.append(dict(case=case, parent=parent, control=control,
                    per_seed=records, seed_average_difference=float(np.mean(seed_values)),
                    seed_average_paired_chain_se=jackknife_se(np.mean(seed_jacks, axis=0)),
                    seed_range=seed_values))
    result = dict(qualifications=[
        "Negative difference favors attention; comparison is sampled tangent slope per actual audit FS, not finite energy.",
        "Same budget caps, not necessarily identical realized native cost.",
        "All three seeds retained; primary seed remains74121 fixed before outcomes.",
        "V4 is an explicitly developmental physics-context followup after V3, not untouched architecture confirmation.",
        "V3 finite-H check is separate; its finite energies/variances are NOT assigned to V4.",
        "V2/V3 scramble is invariant CT6 geometry; V4 scramble is actual screening R plus consistently repooled features."],
        arms=arms, paired_attention_differences=differences)
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("caches", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-count", type=int, default=6)
    parser.add_argument("--direction-features", action="store_true")
    parser.add_argument("--screening-context", action="store_true")
    args = parser.parse_args()
    compare_cached_panels(args.caches, args.output, fit_count=args.fit_count,
                          direction_features=args.direction_features,
                          screening_context=args.screening_context)


if __name__ == "__main__":
    main()
