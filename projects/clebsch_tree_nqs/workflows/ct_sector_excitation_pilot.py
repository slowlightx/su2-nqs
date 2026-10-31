"""Bounded, opt-in CT6 full-W strict-spin/momentum optimization pilot.

This is not a root-only Ritz fit.  Every inherited nonroot native W, including
binary six-site maps, and every new target-root coefficient is trainable.
The positive proposal is FIXED by the independently audited singlet parent:
normalized raw-orbit/H-exchange bridge plus an analytic uniform m2=0 term.
Changing this proposal never changes the target wavefunction.  Independent
fit, finite-state calibration, optional H/JVP, and final full-H streams are
kept separate.  A cost-gated preflight is not an optimized excitation or gap.

The local default permits one update in600s; the GPU template permits two
updates in7200s. Neither template submits a job or grants execution authority.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from math import comb
from pathlib import Path
import signal
from time import perf_counter
from typing import Any, Callable

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.finite_candidate_calibration import select_finite_grid, bilateral_direction_moments
from su2_nqs.vmc.full_support_proposal import draw_full_support_sz0, full_support_sz0_density
from .triangle_iid_orbit_audit import paired_iid_moments

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "projects/clebsch_tree_nqs/results/ct_sector_excitation_pilot_20261028"
PARENTS = ROOT / "projects/clebsch_tree_nqs/results/ct_status_20261028/verified_next_parents"
TASK = "CT6_full_W_strict_sector_bounded_pilot_v1"
FRACTIONS = np.asarray((1., .5, .25, .125, 0.))
PARENT_IDENTITIES = {
    0.: ("ct6_j0", 466, "ternary_joint", "5fe4b4dfe1887718b5ba3754964eff3cfcb9d2bc895b8e30a034a74d9e8f10f5"),
    .125: ("ct6_j0125", 492, "binary_truncated", "c6fb9819715ba89f8da3a8fa60810e5dd6243f89a7b86096c17ef5636d054a04"),
}


def sector_spec(label: str) -> dict[str, Any]:
    """Two declared sectors; q and m2 are doubled integer labels."""
    if label == "q2_K":
        return dict(label=label, target_q=2, target_m2=0, momentum=[[1, 3], [1, 3]],
                    little_group_character="all_plus_one")
    if label == "q0_X":
        return dict(label=label, target_q=0, target_m2=0, momentum=[[-1, 6], [-1, 6]],
                    little_group_character="all_plus_one")
    raise ValueError("only doubled(q,m2)=(2,0) K or(0,0) X are registered")


def make_plan(j2: float, sector: str, *, profile: str = "local") -> dict[str, Any]:
    """Create an editable but bounded plan; numerical execution remains off."""
    if j2 not in PARENT_IDENTITIES or profile not in ("local", "gpu"):
        raise ValueError("J2=0/.125 and local/gpu profile required")
    tag, step, arity, parameter_hash = PARENT_IDENTITIES[j2]
    directory = PARENTS / tag
    offset = (0 if j2 == 0. else 100) + (0 if sector == "q2_K" else 10)
    return dict(task=TASK, profile=profile, j2=float(j2), sector=sector_spec(sector),
        model=dict(length=6, multiplicity_cap=28, maximum_q=10), arity=arity, binary_rank=8,
        parent=dict(directory=str(directory), global_step=step, parameter_sha256=parameter_hash,
            checkpoint_sha256=file_sha256(directory / "parent.pkl"),
            audit_sha256=file_sha256(directory / "parent_snapshot/audit.json")),
        initialization_seed=2610288001 + offset, initialization_attempts=4,
        initialization_norm_threshold=1e-24, master_seed=2610289001 + offset,
        steps=1 if profile == "local" else 2, fit_records=64 if profile == "local" else 128,
        calibration_records=64 if profile == "local" else 128,
        audit_records=64 if profile == "local" else 512, audit_groups=8 if profile == "local" else 32,
        guard_H_JVP_records=0 if profile == "local" else 4,
        projection_chunk=4, action_chunk=1, score_chunk=1, H_chunk=1,
        damping=.5, learning_rate=.03, target_FS=.001, maximum_FS=.005,
        tangent_FS_radius=.003, epsilon=.125, parameter_chunk_size=8192,
        fractions=FRACTIONS.tolist(), wall_cap_seconds=600 if profile == "local" else 7200,
        audit_reserve_fraction=.35, maximum_jacobian_bytes=8 * 1024**3,
        exact_invariants_passed=False, execution_authorized=False,
        automatic_continuation=False, no_job_submission=True,
        all_nonroot_W_and_target_root_trainable=True, sampler_is_frozen_ground_parent=True)


def validate_plan(plan: dict[str, Any], *, executing: bool = False) -> None:
    """Fail closed on changed sectors, unsupported layouts or enlarged budgets."""
    if plan.get("task") != TASK or plan.get("j2") not in PARENT_IDENTITIES:
        raise ValueError("registered CT6 excitation task/coupling required")
    profile = plan.get("profile")
    if profile not in ("local", "gpu") or plan.get("sector") != sector_spec(plan.get("sector", {}).get("label")):
        raise ValueError("exact registered spin/momentum sector required")
    tag, step, arity, digest = PARENT_IDENTITIES[plan["j2"]]
    parent = plan.get("parent", {})
    if (plan.get("model") != dict(length=6, multiplicity_cap=28, maximum_q=10)
            or plan.get("arity") != arity or plan.get("binary_rank") != 8
            or parent.get("global_step") != step or parent.get("parameter_sha256") != digest
            or any(len(parent.get(key, "")) != 64 for key in ("checkpoint_sha256", "audit_sha256"))
            or Path(parent.get("directory", "")).name != tag):
        raise ValueError("fixed audited CT6 global466/492 w28 parent required")
    cap = 600 if profile == "local" else 7200
    upper_records = 128 if profile == "local" else 4096
    for field in ("steps", "fit_records", "calibration_records", "audit_records", "audit_groups",
                  "projection_chunk", "action_chunk", "score_chunk", "H_chunk", "parameter_chunk_size"):
        value = plan.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("positive integer record/kernel settings required")
    if (plan["steps"] > 2 or max(plan[k] for k in ("fit_records", "calibration_records", "audit_records")) > upper_records
            or plan["fit_records"] < 4 or plan["calibration_records"] < 16
            or plan["calibration_records"] % 8 or plan["audit_groups"] < 4
            or plan["audit_records"] % plan["audit_groups"]
            or not 0 < plan.get("wall_cap_seconds", 0) <= cap
            or plan.get("guard_H_JVP_records") not in (0, 4, 8, 16)
            or not 0 < plan.get("audit_reserve_fraction", 0) < .75
            or plan.get("epsilon") != .125 or plan.get("fractions") != FRACTIONS.tolist()
            or plan.get("initialization_attempts") != 4
            or plan.get("initialization_norm_threshold") != 1e-24
            or not 0 < plan.get("target_FS", 0) < plan.get("maximum_FS", 0) < .1
            or not 0 < plan.get("damping", 0) <= 10 or not 0 < plan.get("learning_rate", 0) <= .1
            or not 0 < plan.get("tangent_FS_radius", 0) <= .01
            or not 0 < plan.get("maximum_jacobian_bytes", 0) <= 64 * 1024**3
            or plan.get("automatic_continuation") is not False
            or plan.get("no_job_submission") is not True
            or plan.get("all_nonroot_W_and_target_root_trainable") is not True
            or plan.get("sampler_is_frozen_ground_parent") is not True):
        raise ValueError("bounded records/steps, positive support and all-W protocol required")
    for field in ("initialization_seed", "master_seed"):
        if isinstance(plan.get(field), bool) or not isinstance(plan.get(field), int) or not 0 <= plan[field] < 2**32:
            raise ValueError("distinct uint32 initialization/master streams required")
    if plan["initialization_seed"] == plan["master_seed"]:
        raise ValueError("initialization must not share the optimization/audit RNG stream")
    if "restart" in plan:
        restart = plan["restart"]
        if (profile != "gpu" or set(restart) != {"directory", "checkpoint_sha256", "initial_sha256", "source_plan_sha256"}
                or not Path(restart["directory"]).resolve().is_relative_to(BASE.resolve())
                or any(len(restart.get(k, "")) != 64 for k in ("checkpoint_sha256", "initial_sha256", "source_plan_sha256"))):
            raise ValueError("sealed local checkpoint-to-GPU restart receipt required")
    if executing and (plan.get("exact_invariants_passed") is not True or plan.get("execution_authorized") is not True):
        raise PermissionError("root must separately approve ordered exact invariants and bounded execution")


@contextmanager
def deadline(seconds: float):
    """Foreground hard wall bound; never launches an unmonitored background run."""
    previous = signal.getsignal(signal.SIGALRM)
    timer = signal.getitimer(signal.ITIMER_REAL)
    started = perf_counter()
    def stop(*_: Any) -> None:
        raise TimeoutError("registered sector pilot wall cap reached")
    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, timer[0]) if timer[0] else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(1e-6, timer[0] - (perf_counter() - started)), timer[1])


def save_state(path: Path, state: tuple, *, completed_updates: int, metadata: dict[str, Any]) -> None:
    """Atomically preserve ALL trained coordinates, optimizer and next PCG64 state."""
    params, moments, rng = state
    save_checkpoint(path, params, optimizer_state=moments,
        metadata=dict(metadata, completed_updates=completed_updates, next_rng=deepcopy(rng),
            parameter_sha256=pytree_sha256(params), optimizer_sha256=pytree_sha256(moments),
            rng_sha256=canonical_sha256(rng)))


def restore_state(path: Path) -> tuple[tuple, dict[str, Any]]:
    """Require bit-identical parameters/moments/unused RNG on continuation."""
    if not jax.config.x64_enabled:
        raise RuntimeError("x64 must be enabled before loading a trusted checkpoint")
    saved = load_checkpoint(path)
    meta = saved["metadata"]
    state = (saved["params"], saved["optimizer_state"], meta["next_rng"])
    if (pytree_sha256(state[0]) != meta["parameter_sha256"]
            or pytree_sha256(state[1]) != meta["optimizer_sha256"]
            or canonical_sha256(state[2]) != meta["rng_sha256"]):
        raise ValueError("sector checkpoint parameter/optimizer/RNG seal mismatch")
    return state, meta


def next_seed(rng_state: dict[str, Any], role: str) -> tuple[int, dict[str, Any]]:
    """Advance a saved master state once, recording every disjoint role seed."""
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(rng_state["generator"])
    used = set(rng_state["used_seeds"]) | set(rng_state.get("excluded_seeds", []))
    while True:
        seed = int(rng.integers(0, 2**32, dtype=np.uint32))
        if seed not in used:
            break
    following = deepcopy(rng_state)
    following.update(generator=rng.bit_generator.state, used_seeds=[*rng_state["used_seeds"], seed],
                     roles=[*rng_state["roles"], dict(role=role, seed=seed)])
    return seed, following


def interpolate(parent: Any, proposed: Any, fraction: float) -> Any:
    """Actual parameter update, not a linear-wavefunction approximation."""
    return jax.tree.map(lambda old, new: old + fraction * (new - old), parent, proposed)


def family_norms(params: Any) -> dict[str, Any]:
    """Coordinate-norm diagnostics prove participation, not physical importance."""
    result = {}
    for label, values in (("root", params["root"]), *params["backbone"].items()):
        leaves = [np.asarray(x) for x in jax.tree.leaves(values)]
        result[label] = dict(L2=float(np.sqrt(sum(np.vdot(x, x).real for x in leaves))),
            nonzero_entries=sum(int(np.count_nonzero(x)) for x in leaves),
            coordinate_count=sum(x.size for x in leaves))
    return dict(families=result, coordinate_gauge_dependent_not_a_physical_capacity_measure=True)


def replay_reference(runtime: dict, params: Any, reference: dict, *, require_H: bool) -> dict:
    """Fixed complex phase/scale, same rows and Q; no phase alignment or floors."""
    if require_H and (reference.get("hpsi") is None or reference.get("proposal") is None):
        raise ValueError("a complete source full-H/Q replay reference is required before GPU execution")
    rows = np.asarray(reference["rows"], np.int32)
    if reference.get("hpsi") is not None:
        psi, hpsi = runtime["H"](params, rows)
        Q, _ = runtime["Q"](rows)
        observed = dict(psi=psi, hpsi=hpsi, proposal=Q)
    else:
        observed = dict(psi=runtime["amplitudes"](params, rows))
    amplitude_scale = float(np.max(abs(np.asarray(reference["psi"]))))
    if amplitude_scale <= 0:
        raise ValueError("frozen replay batch must include at least one nonzero target amplitude")
    errors = {}
    absolute_errors = {}
    for name, actual in observed.items():
        expected = np.asarray(reference[name]); actual = np.asarray(actual)
        if expected.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ArithmeticError("numerical checkpoint replay shape/finiteness failed")
        scale = float(np.max(abs(expected))) if name == "proposal" else amplitude_scale
        error = float(np.max(abs(actual - expected)))
        relative = error / scale if scale > 0 else (0. if error == 0 else float("inf"))
        tolerance = 2e-14 if name == "proposal" else 2e-11
        if relative > tolerance:
            raise ArithmeticError("fixed complex-scale checkpoint replay failed for " + name)
        errors[name] = relative
        absolute_errors[name] = error
    return dict(records=len(rows), relative_errors=errors, absolute_errors=absolute_errors,
                common_nonzero_amplitude_reference_scale=amplitude_scale, phase_aligned=False,
                full_H_replayed="hpsi" in observed, passed=True)


def fit_cost_estimate(plan: dict, *, amplitude_seconds: float, H_seconds: float,
                      jacobian_seconds: float, Q_seconds: float) -> dict[str, Any]:
    """Conservative warm-throughput estimate, not a promise about compilation."""
    values = np.asarray((amplitude_seconds, H_seconds, jacobian_seconds, Q_seconds), float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("finite nonnegative measured per-record costs required")
    audit = plan["audit_records"] * (2 * H_seconds + Q_seconds)
    update = (plan["fit_records"] * (H_seconds + jacobian_seconds + Q_seconds)
              + plan["calibration_records"] * (len(FRACTIONS) * amplitude_seconds + Q_seconds)
              + plan["guard_H_JVP_records"] * (4 * H_seconds + Q_seconds)
              + 4 * (H_seconds + Q_seconds))
    return dict(estimated_audit_seconds=1.5 * audit, estimated_one_update_seconds=1.5 * update,
                estimated_all_seconds=1.5 * (plan["steps"] * update + audit) + 30,
                safety_factor=1.5, compilation_peak_not_certified=True,
                sampling_and_serialization_must_fit_remaining_margin=True)


def checked_parent(plan: dict) -> dict:
    """Bind actual audited global466/492 parameters without mutating their files."""
    identity = plan["parent"]
    directory = Path(identity["directory"])
    if (file_sha256(directory / "parent.pkl") != identity["checkpoint_sha256"]
            or file_sha256(directory / "parent_snapshot/audit.json") != identity["audit_sha256"]):
        raise ValueError("frozen parent byte identity changed")
    saved = load_checkpoint(directory / "parent.pkl")
    audit = json.loads((directory / "parent_snapshot/audit.json").read_text())
    if (pytree_sha256(saved["params"]) != identity["parameter_sha256"]
            or audit.get("endpoint_params_sha256") != identity["parameter_sha256"]
            or saved["metadata"].get("absolute_completed_steps") != identity["global_step"]
            or saved.get("optimizer_state") is None):
        raise ValueError("parent audit/parameters/optimizer/global step mismatch")
    return saved


def make_runtime(plan: dict, saved: dict) -> dict[str, Any]:
    """Connect verified sector, complex-SR and unchanged physical-H components."""
    from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
    from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
    from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank
    from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
    from su2_nqs.models.triangular_sector_amplitude import (make_sector_prepare,
        make_prepared_sector_amplitude, make_prepared_sector_exchange_queries)
    from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action, make_orbit_exchange_bridge_products
    from .ct6_native_direction_calibration import normalized_raw_bank, draw_orbit_rows
    cfg = MixedClebschConfig(**plan["model"])
    bonds = tuple(triangular_j1j2_bonds(6, 6, j2=plan["j2"], periodic=True))
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    bank, normalization = normalized_raw_bank(saved["params"], dict(cfg=cfg,
        bank=lambda p: arity_clebsch_bank(p, cfg, plan["arity"], plan["binary_rank"])))
    probabilities = np.abs([value for _, _, value in bonds]); probabilities /= probabilities.sum()
    parent_products, product_note = make_orbit_exchange_bridge_products(
        lambda edges: make_feature_lut_exchange_queries(cfg, edges), images, bonds, probabilities,
        n_sites=36, projection_chunk=plan["projection_chunk"])
    parent_prepared = prepare_feature_lut_bank(bank, cfg)
    draw_context = dict(cfg=cfg, images=images, bonds=bonds)
    work = dict(ground_full_H_centers_requested=0, ground_full_H_centers_complete=0,
        target_full_H_centers_requested=0, target_full_H_centers_complete=0,
        target_amplitude_centers_requested=0, target_amplitude_centers_complete=0,
        new_IID_rows=0, full_W_complex_J_rows=0, target_full_H_JVP_centers=0)
    def draw(count: int, seed: int) -> tuple[np.ndarray, dict]:
        value = draw_full_support_sz0(lambda n, s: draw_orbit_rows([bank], draw_context, n, s, bridge=True),
            count, n_sites=36, seed=seed, epsilon=plan["epsilon"])
        work["new_IID_rows"] += count
        return value
    def Q(rows: np.ndarray) -> tuple[np.ndarray, tuple]:
        work["ground_full_H_centers_requested"] += len(rows)
        products = jax.block_until_ready(parent_products(parent_prepared, jnp.asarray(rows)))
        work["ground_full_H_centers_complete"] += len(rows)
        values = tuple(np.asarray(x) for x in products)
        density = full_support_sz0_density(values[2], values[3], n_sites=36, epsilon=plan["epsilon"])
        return density, values
    spec = plan["sector"]
    prepare = jax.jit(make_sector_prepare(cfg, arity=plan["arity"], binary_rank=plan["binary_rank"]))
    amplitude = make_prepared_sector_amplitude(cfg, spec["target_q"], spec["target_m2"],
        momentum=tuple(tuple(x) for x in spec["momentum"]), chunk=plan["projection_chunk"])
    amplitude_batch = jax.jit(lambda prepared, rows: jax.lax.map(
        lambda row: amplitude(prepared, row), rows, batch_size=plan["action_chunk"]))
    physical_action = make_amplitude_heisenberg_action(amplitude, bonds, n_sites=36)
    exchange = make_prepared_sector_exchange_queries(cfg, spec["target_q"], spec["target_m2"],
        tuple(tuple(x) for x in spec["momentum"]), tuple((a, b) for a, b, _ in bonds),
        chunk=plan["projection_chunk"])
    edges = jnp.asarray([(a, b) for a, b, _ in bonds], jnp.int32)
    couplings = jnp.asarray([v for _, _, v in bonds], jnp.float64)
    def cached_one(prepared: Any, row: jax.Array) -> jax.Array:
        psi, changed = exchange(prepared, row)
        unlike = row[edges[:, 0]] != row[edges[:, 1]]
        hp = jnp.sum(couplings * jnp.where(unlike, -.25, .25)) * psi
        hp = hp + jnp.sum(couplings * jnp.where(unlike, .5 * changed, 0.))
        return jnp.stack((psi, hp))
    # One-row H microbatches prevent the generic full-H wrapper from vmapping
    # a large center batch through every spatial image and bond at once.
    action = jax.jit(lambda prepared, rows: jax.lax.map(
        lambda row: cached_one(prepared, row), rows, batch_size=plan["H_chunk"]))
    generic_action = jax.jit(lambda prepared, rows: jax.lax.map(
        lambda row: jnp.stack(physical_action(prepared, row[None]))[:, 0], rows,
        batch_size=plan["H_chunk"]))
    def H(params: Any, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        work["target_full_H_centers_requested"] += len(rows)
        products = np.asarray(jax.block_until_ready(action(prepare(params), jnp.asarray(rows))))
        work["target_full_H_centers_complete"] += len(rows)
        return products[:, 0], products[:, 1]
    def amplitudes(params: Any, rows: np.ndarray) -> np.ndarray:
        work["target_amplitude_centers_requested"] += len(rows)
        values = np.asarray(jax.block_until_ready(amplitude_batch(prepare(params), jnp.asarray(rows))))
        work["target_amplitude_centers_complete"] += len(rows)
        return values
    def generic_H(params: Any, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        work["target_full_H_centers_requested"] += len(rows)
        products = np.asarray(jax.block_until_ready(generic_action(prepare(params), jnp.asarray(rows))))
        work["target_full_H_centers_complete"] += len(rows)
        return products[:, 0], products[:, 1]
    return dict(cfg=cfg, draw=draw, Q=Q, prepare=prepare, amplitude=amplitude,
        amplitudes=amplitudes, H=H, generic_H=generic_H, prepared_action=action, normalization=normalization,
        proposal_metadata=product_note, full_ground_A1_images=len(images), bonds=bonds, work=work)


def initialize_target(plan: dict, saved: dict, runtime: dict, probe_rows: np.ndarray,
                      probe_Q: np.ndarray) -> tuple[tuple, dict]:
    """Physical raw-parent descendant first; inherited ALL lower W unchanged.

    Only an exactly zero probe permits preregistered random-root fallback.
    An unresolved small nonzero probe fails closed, not a search over energies
    or over the largest random norm. The threshold never floors amplitudes.
    """
    from su2_nqs.models.triangular_sector_amplitude import pack_sector_parameters
    from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
    from su2_nqs.models.triangular_target_root import (target_root_layout,
        singlet_root_coefficients, collective_spin_root_coefficients)
    layout = target_root_layout(runtime["cfg"], plan["sector"]["target_q"])
    source_bank = arity_clebsch_bank(saved["params"], runtime["cfg"], plan["arity"], plan["binary_rank"])
    primary = (singlet_root_coefficients(source_bank, runtime["cfg"])
        if plan["sector"]["target_q"] == 0 else
        collective_spin_root_coefficients(source_bank, runtime["cfg"], plan["sector"]["target_q"]))
    attempts = []
    for attempt in range(plan["initialization_attempts"] + 1):
        seed = None if attempt == 0 else (plan["initialization_seed"] + attempt - 1) % 2**32
        if attempt == 0:
            root = tuple(jnp.asarray(x, jnp.float64) for x in primary)
        else:
            rng = np.random.default_rng(seed)
            root = tuple(jnp.asarray(rng.normal(size=(dl, dr)) / np.sqrt(dl * dr), jnp.float64)
                         for _, _, dl, dr in layout)
        params = pack_sector_parameters(saved["params"], root)
        phi = runtime["amplitudes"](params, probe_rows)
        norm = float(np.mean(abs(phi)**2 / probe_Q))
        attempts.append(dict(attempt=attempt, seed=seed, kind="physical_raw_parent_descendant" if attempt == 0 else "fixed_random_root_fallback",
                             sampled_projected_norm=norm,
                             exact_probe_nodes=int(np.sum(phi == 0))))
        if np.isfinite(phi).all() and np.isfinite(norm) and norm > plan["initialization_norm_threshold"]:
            old_moments = saved["optimizer_state"]
            zeros = jax.tree.map(jnp.zeros_like, root)
            # Preserve inherited nonroot optimizer history; the new sector root
            # has no old history. SR leaves these first/second moments intact.
            moments = (old_moments[0], pack_sector_parameters(old_moments[1], zeros),
                       pack_sector_parameters(old_moments[2], zeros))
            old_rng = saved["metadata"].get("rng", {})
            excluded = [(plan["initialization_seed"] + k) % 2**32 for k in range(4)]
            excluded += [plan["master_seed"], plan["initialization_seed"] ^ 0x72843E1B]
            if isinstance(old_rng, dict):
                excluded += old_rng.get("consumed_role_seeds", []) + old_rng.get("excluded_seeds", [])
            following = dict(generator=np.random.default_rng(plan["master_seed"]).bit_generator.state,
                             used_seeds=[], excluded_seeds=sorted(set(excluded)), roles=[])
            return (params, moments, following), dict(attempts=attempts, selected_attempt=attempt,
                root_layout=[list(x) for x in layout], seed_selection_uses_energy=False,
                inherited_nonroot_parameters_exact=True,
                native_parent_parameter_count=sum(x.size for x in jax.tree.leaves(saved["params"])),
                trainable_sector_parameter_count=sum(x.size for x in jax.tree.leaves(params)),
                root_parameter_count=sum(x.size for x in root), raw_A1_parent_not_used_as_K_or_X_seed=True)
        if not np.isfinite(phi).all() or not np.isfinite(norm) or not np.all(phi == 0):
            raise ArithmeticError("unresolved small/nonfinite projected norm: no random search or amplitude floor")
    raise ArithmeticError("physical root and four fixed fallback roots have zero probe support; no training")


def _timed(report: dict, label: str, call: Callable) -> Any:
    tick = perf_counter(); value = call(); jax.block_until_ready(value)
    report[label] = perf_counter() - tick
    return value


def preflight(plan: dict, runtime: dict, state: tuple, rows: np.ndarray, output: Path) -> dict:
    """Measure amplitudes, complete H and an ALL-W two-row complex Jacobian."""
    params = state[0]; timings = {}
    _timed(timings, "amplitude_compile_four", lambda: runtime["amplitudes"](params, rows[:4]))
    probe_phi = _timed(timings, "amplitude_warm_four", lambda: runtime["amplitudes"](params, rows[:4]))
    Hrows = rows[:4]
    if not np.any(abs(probe_phi) > 0):
        values = runtime["amplitudes"](params, rows)
        Hrows = rows[np.argsort(-abs(values))[:4]]
        if not np.any(abs(values) > 0):
            raise ArithmeticError("no nonzero fixed support center for numerical replay")
    _timed(timings, "Q_compile_four", lambda: runtime["Q"](rows[:4]))
    _timed(timings, "Q_warm_four", lambda: runtime["Q"](rows[:4]))
    # A regressed generic H compilation cannot consume the entire local cap.
    # The initial full-state checkpoint has already been saved before here.
    with deadline(120. if plan["profile"] == "local" else 600.):
        _timed(timings, "full_H_compile_four", lambda: runtime["H"](params, Hrows))
        psi, hpsi = _timed(timings, "full_H_warm_four", lambda: runtime["H"](params, Hrows))
    q, ground = runtime["Q"](Hrows)
    np.savez_compressed(output / "preflight_H_products.npz", rows=Hrows, proposal=q,
        psi=psi, hpsi=hpsi, ground_psi=ground[0], ground_hpsi=ground[1])
    with deadline(120. if plan["profile"] == "local" else 600.):
        oracle_p, oracle_h = _timed(timings, "generic_full_H_oracle_compile_one",
                                  lambda: runtime["generic_H"](params, Hrows[:1]))
    oracle_error = float(max(np.max(abs(oracle_p - psi[:1])), np.max(abs(oracle_h - hpsi[:1]))) /
                         max(np.max(abs(psi[:1])), np.max(abs(hpsi[:1])), np.finfo(float).tiny))
    if oracle_error > 2e-11 or not np.isfinite(oracle_error):
        raise ArithmeticError("cached target-sector complete H disagrees with generic independent H")
    theta, unravel = ravel_pytree(params)
    # Real coordinates; both real/imaginary amplitude derivatives enter.
    def reim(values: jax.Array, row: jax.Array) -> jax.Array:
        phi = runtime["amplitude"](runtime["prepare"](unravel(values)), row)
        return jnp.stack((jnp.real(phi), jnp.imag(phi)))
    derivative = jax.jit(jax.vmap(jax.jacrev(reim), in_axes=(None, 0)))
    with deadline(120. if plan["profile"] == "local" else 600.):
        jac = _timed(timings, "full_W_complex_J_compile_two", lambda: derivative(theta, jnp.asarray(rows[:2])))
        jac = _timed(timings, "full_W_complex_J_warm_two", lambda: derivative(theta, jnp.asarray(rows[:2])))
    runtime.get("work", {})["full_W_complex_J_rows"] = runtime.get("work", {}).get("full_W_complex_J_rows", 0) + 4
    if not np.isfinite(np.asarray(jac)).all():
        raise ArithmeticError("nonfinite ALL-W complex derivative")
    counts = dict(parameters=len(theta), estimated_retained_complex_J_bytes=2 * 8 * plan["fit_records"] * len(theta),
                  dual_Gram_dimension=2 * plan["fit_records"], actual_probe_J_shape=list(jac.shape),
                  probe_nonzero_real_parameter_columns=int(np.count_nonzero(np.any(np.asarray(jac) != 0, axis=(0, 1)))))
    counts["full_W_probe_derivatives_by_family"] = family_norms(unravel(jnp.asarray(
        np.sqrt(np.sum(abs(np.asarray(jac))**2, axis=(0, 1))))))
    estimate = fit_cost_estimate(plan, amplitude_seconds=timings["amplitude_warm_four"] / 4,
        H_seconds=timings["full_H_warm_four"] / 4, jacobian_seconds=timings["full_W_complex_J_warm_two"] / 2,
        Q_seconds=timings["Q_warm_four"] / 4)
    report = dict(status="measured_capabilities_only_not_optimized", timings=timings, **counts, **estimate,
        all_native_W_in_AD=True, root_only_Ritz=False, full_H_bond_labels=len(runtime["bonds"]),
        cached_full_H_generic_oracle_relative_error=oracle_error,
        physical_H_backend="coherent_sector_inverse_image_subtree_exchange_cache",
        executable_GPU_template_available=True)
    atomic_json(output / "preflight.json", report)
    return report


def write_gpu_restart_plan(source: Path) -> Path:
    """Seal a separately authorized GPU restart, with zero amplitude queries.

    The selected native state, trained moments and next RNG are inherited
    exactly; this creates neither an allocation nor permission to execute it.
    """
    original = json.loads((source / "plan.json").read_text())
    state, meta = restore_state(source / "checkpoint.pkl")
    del state
    plan = make_plan(original["j2"], original["sector"]["label"], profile="gpu")
    plan["steps"] = min(2, max(1, int(meta["completed_updates"]) + 1))
    plan["restart"] = dict(directory=str(source.resolve()),
        checkpoint_sha256=file_sha256(source / "checkpoint.pkl"),
        initial_sha256=file_sha256(source / "initial.pkl"), source_plan_sha256=canonical_sha256(original))
    validate_plan(plan)
    destination = source / "gpu_restart_plan.json"
    atomic_json(destination, plan)
    return destination


def transition(plan: dict, runtime: dict, state: tuple, output: Path, *, index: int) -> tuple[tuple, dict]:
    """One all-W SR proposal, fresh amplitude-only finite FS calibration/guard."""
    from su2_nqs.vmc.complex_importance_amplitude_sr import make_complex_importance_amplitude_sr
    output.mkdir()
    params, moments, rng = state
    fit_seed, rng = next_seed(rng, f"step{index}_fit")
    cal_seed, rng = next_seed(rng, f"step{index}_finite_geometry")
    rows, fit_draw = runtime["draw"](plan["fit_records"], fit_seed)
    Q, _ = runtime["Q"](rows)
    psi, Hpsi = runtime["H"](params, rows)
    np.savez_compressed(output / "fit_products.npz", rows=rows, proposal=Q, psi=psi, hpsi=Hpsi)
    update = make_complex_importance_amplitude_sr(lambda p, row: runtime["amplitude"](runtime["prepare"](p), row),
        params, damping=plan["damping"], score_chunk=plan["score_chunk"],
        parameter_chunk_size=plan["parameter_chunk_size"], sr_tangent_fs_radius=plan["tangent_FS_radius"],
        prepare=runtime["prepare"], prepared_amplitude=runtime["amplitude"])
    proposed, next_moments, info = jax.block_until_ready(update(params, moments, jnp.asarray(rows),
        jnp.asarray(Hpsi, jnp.complex128), jnp.asarray(Q, jnp.float64), plan["learning_rate"]))
    runtime.get("work", {})["full_W_complex_J_rows"] = runtime.get("work", {}).get("full_W_complex_J_rows", 0) + len(rows)
    report = dict(index=index, fit_seed=fit_seed, calibration_seed=cal_seed,
        SR={key: np.asarray(value).item() for key, value in info.items()},
        fit_draw_seed=fit_seed, fit_records=len(rows), all_W_trainable=True, root_only_Ritz=False)
    accepted = bool(info["accepted"])
    if accepted:
        cal_rows, _ = runtime["draw"](plan["calibration_records"], cal_seed)
        cal_Q, _ = runtime["Q"](cal_rows)
        candidates = [interpolate(params, proposed, float(f)) for f in FRACTIONS]
        phi = np.column_stack([runtime["amplitudes"](candidate, cal_rows) for candidate in candidates])
        selected, guard = select_finite_grid(phi, cal_Q, FRACTIONS, split=len(cal_rows)//2,
            groups=4, target=plan["target_FS"], maximum=plan["maximum_FS"])
        np.savez_compressed(output / "finite_geometry.npz", rows=cal_rows, proposal=cal_Q, phi=phi, fractions=FRACTIONS)
        accepted = bool(guard["accepted"]) and selected < len(FRACTIONS) - 1
        report.update(finite_geometry=guard, selected_fraction=float(FRACTIONS[selected]))
        chosen = candidates[selected] if accepted else params
    else:
        chosen = params
        report.update(selected_fraction=0., finite_geometry=dict(accepted=False, reason="invalid_SR_proposal"))
    if accepted and plan["guard_H_JVP_records"]:
        seed, rng = next_seed(rng, f"step{index}_independent_H_JVP")
        rows, _ = runtime["draw"](plan["guard_H_JVP_records"], seed)
        Q, _ = runtime["Q"](rows)
        delta = jax.tree.map(lambda new, old: new - old, chosen, params)
        def products(p):
            return runtime["prepared_action"](runtime["prepare"](p), jnp.asarray(rows))
        value, tangent = jax.block_until_ready(jax.jvp(products, (params,), (delta,)))
        runtime.get("work", {})["target_full_H_JVP_centers"] = runtime.get("work", {}).get("target_full_H_JVP_centers", 0) + len(rows)
        value, tangent = np.asarray(value), np.asarray(tangent)
        slope = bilateral_direction_moments(value[:, 0], value[:, 1], tangent[:, 0], tangent[:, 1], Q,
            groups=4, n_sites=36)
        np.savez_compressed(output / "independent_H_JVP.npz", rows=rows, proposal=Q,
            psi=value[:, 0], hpsi=value[:, 1], dpsi=tangent[:, 0], hdpsi=tangent[:, 1])
        report["independent_bilateral_slope"] = dict(slope, seed=seed,
            diagnostic_only_not_used_to_fit_or_select_update=True)
    report["accepted"] = accepted
    report["actual_parameter_change_by_family"] = family_norms(jax.tree.map(
        lambda new, old: new - old, chosen, params))
    next_state = (chosen, next_moments if accepted else moments, rng)
    atomic_json(output / "transition.json", report)
    return next_state, report


def paired_final_audit(plan: dict, runtime: dict, initial: Any, final: Any,
                       rng: dict, output: Path) -> tuple[dict, dict]:
    """Freeze endpoints then use NEW full-H rows, keeping nodes and H² tails."""
    output.mkdir()
    seed, following = next_seed(rng, "final_independent_full_H_audit")
    rows, _ = runtime["draw"](plan["audit_records"], seed)
    Q, ground = runtime["Q"](rows)
    products = [runtime["H"](p, rows) for p in (initial, final)]
    psi = np.column_stack([x[0] for x in products]); hp = np.column_stack([x[1] for x in products])
    labels = np.repeat(np.arange(plan["audit_groups"]), len(rows)//plan["audit_groups"])
    report = paired_iid_moments(psi, hp, Q, labels, n_sites=36, seed=seed ^ 0x5A17D0B1)
    ground_comparison = paired_iid_moments(np.column_stack((ground[0], psi[:, 1])),
        np.column_stack((ground[1], hp[:, 1])), Q, labels, n_sites=36, seed=seed ^ 0x34AE67C9)
    np.savez_compressed(output / "products.npz", rows=rows, proposal=Q, psi=psi, hpsi=hp,
                        ground_psi=ground[0], ground_hpsi=ground[1], groups=labels)
    receipt = dict(status="complete_exploratory_independent_audit", seed=seed, moments=report,
        final_minus_approximate_ground_reference=ground_comparison,
        initial_parameter_sha256=pytree_sha256(initial), final_parameter_sha256=pytree_sha256(final),
        independent_from_fit_and_finite_geometry=True, gap_not_certified=True,
        finite_size_approximate_ground_reference_not_exact_eigenenergy=True,
        records=len(rows), full_H_centers=3 * len(rows))
    atomic_json(output / "audit.json", receipt)
    return receipt, following


def run(plan: dict, output: Path, *, stage: str) -> dict[str, Any]:
    """Prepare, preflight or continue only within the registered cumulative cap."""
    validate_plan(plan, executing=True)
    if stage not in ("prepare", "preflight", "train", "resume"):
        raise ValueError("prepare/preflight/train/resume stage required")
    if not output.resolve().is_relative_to(BASE.resolve()):
        raise ValueError("new sector pilot artifacts must stay in the dedicated results directory")
    if stage == "prepare":
        if output.exists():
            raise FileExistsError("new immutable pilot directory required")
        output.mkdir(parents=True)
        atomic_json(output / "plan.json", plan)
    elif json.loads((output / "plan.json").read_text()) != plan:
        raise ValueError("registered plan changed after preparation")
    previous = json.loads((output / "progress.json").read_text()) if (output / "progress.json").exists() else {}
    already = float(previous.get("cumulative_wall_seconds", 0.))
    available = plan["wall_cap_seconds"] - already
    if available <= 0:
        raise TimeoutError("cumulative sector pilot budget exhausted")
    started = perf_counter()
    report = dict(previous, task=TASK, stage=stage, status="started", completed_updates=previous.get("completed_updates", 0))
    manifest = transitive_local_source_manifest([Path(__file__)], root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"))
    prior_work = deepcopy(previous.get("work", {})); runtime = {}
    def preserve() -> None:
        report["cumulative_wall_seconds"] = already + perf_counter() - started
        report["work"] = {key: prior_work.get(key, 0) + runtime.get("work", {}).get(key, 0)
                          for key in set(prior_work) | set(runtime.get("work", {}))}
        atomic_json(output / "progress.json", report)
    try:
        with deadline(available):
            saved = checked_parent(plan)
            runtime = make_runtime(plan, saved)
            if stage == "prepare":
                # Uniform initialization probes are independent of every later
                # role. This is a norm-only support check, not an energy audit.
                rng = np.random.default_rng(plan["initialization_seed"] ^ 0x72843E1B)
                template = np.r_[np.zeros(18, np.int32), np.ones(18, np.int32)]
                rows = np.asarray([rng.permutation(template) for _ in range(16)])
                Q = np.full(16, 1. / comb(36, 18))
                if "restart" in plan:
                    restart = plan["restart"]; source = Path(restart["directory"])
                    old_plan = json.loads((source / "plan.json").read_text())
                    if (canonical_sha256(old_plan) != restart["source_plan_sha256"]
                            or file_sha256(source / "checkpoint.pkl") != restart["checkpoint_sha256"]
                            or file_sha256(source / "initial.pkl") != restart["initial_sha256"]
                            or old_plan["sector"] != plan["sector"] or old_plan["parent"] != plan["parent"]):
                        raise ValueError("GPU restart changed the parent/sector/frozen local state")
                    state, old_meta = restore_state(source / "checkpoint.pkl")
                    initial, initial_meta = restore_state(source / "initial.pkl")
                    completed = int(old_meta["completed_updates"])
                    reference = old_meta.get("numerical_replay")
                    if reference is None and completed == 0 and (source / "preflight_H_products.npz").exists():
                        with np.load(source / "preflight_H_products.npz") as data:
                            reference = {key: data[key] for key in ("rows", "psi", "hpsi", "proposal")}
                    if reference is None:
                        raise ValueError("source has no complete psi/Hpsi/Q numerical reference; GPU restart not released")
                    report["cross_runtime_frozen_state_replay"] = replay_reference(runtime, state[0], reference, require_H=True)
                    initialization = dict(old_meta["initialization"], warm_restart=restart,
                        previous_optimizer_and_next_RNG_inherited=True)
                    metadata = dict(old_meta, plan_sha256=canonical_sha256(plan), source_manifest=manifest,
                        original_parent=plan["parent"], initialization=initialization, numerical_replay=reference)
                    initial_metadata = dict(initial_meta, plan_sha256=canonical_sha256(plan), source_manifest=manifest,
                        original_parent=plan["parent"], initialization=initialization)
                    save_state(output / "initial.pkl", initial, completed_updates=initial_meta["completed_updates"], metadata=initial_metadata)
                else:
                    state, initialization = initialize_target(plan, saved, runtime, rows, Q)
                    completed = 0
                    metadata = dict(plan_sha256=canonical_sha256(plan), source_manifest=manifest,
                        original_parent=plan["parent"], initialization=initialization)
                    save_state(output / "initial.pkl", state, completed_updates=0, metadata=metadata)
                save_state(output / "checkpoint.pkl", state, completed_updates=completed, metadata=metadata)
                np.savez(output / "initialization_probe.npz", rows=rows, proposal=Q,
                         psi=runtime["amplitudes"](state[0], rows))
                report.update(status="prepared_target_full_W_no_optimization", initialization=initialization,
                              completed_updates=completed, proposal=runtime["proposal_metadata"],
                              parent_normalization=runtime["normalization"])
            else:
                transaction = json.loads((output / "in_progress.json").read_text()) if (output / "in_progress.json").exists() else {}
                if transaction.get("status") == "started":
                    raise ValueError("incomplete transaction cannot resume in place; use the last atomic checkpoint in a new sealed restart directory")
                state, metadata = restore_state(output / "checkpoint.pkl")
                initial, _ = restore_state(output / "initial.pkl")
                if metadata["plan_sha256"] != canonical_sha256(plan) or metadata["source_manifest"] != manifest:
                    raise ValueError("checkpoint registered plan or numerical source identity changed")
                completed = int(metadata["completed_updates"])
                if metadata.get("numerical_replay") is not None:
                    report["same_runtime_checkpoint_replay"] = replay_reference(runtime, state[0], metadata["numerical_replay"], require_H=True)
                else:
                    with np.load(output / "initialization_probe.npz") as data:
                        reference = dict(rows=data["rows"][:4], psi=data["psi"][:4])
                    report["same_runtime_checkpoint_replay"] = replay_reference(runtime, state[0], reference, require_H=False)
                if (output / "final_audit/audit.json").exists():
                    report.update(status="already_frozen_and_independently_audited", completed_updates=completed)
                else:
                    probe = np.load(output / "initialization_probe.npz")["rows"]
                    if not (output / "preflight.json").exists():
                        cost = preflight(plan, runtime, state, probe, output)
                    else:
                        cost = json.loads((output / "preflight.json").read_text())
                    remaining = available - (perf_counter() - started)
                    enough = (cost["estimated_retained_complex_J_bytes"] <= plan["maximum_jacobian_bytes"]
                              and cost["estimated_all_seconds"] < remaining)
                    report.update(preflight=cost, cost_gate_passed=bool(enough))
                    if stage == "preflight" or not enough:
                        report["status"] = "capabilities_measured_no_optimized_state" if completed == 0 else "bounded_checkpoint_retained"
                        if plan["profile"] == "local":
                            report["executable_GPU_restart_plan"] = str(write_gpu_restart_plan(output))
                    else:
                        while completed < plan["steps"]:
                            remaining = available - (perf_counter() - started)
                            if remaining < cost["estimated_one_update_seconds"] + cost["estimated_audit_seconds"] + 15:
                                report["stopped_for_fixed_endpoint_audit_reserve"] = True
                                break
                            atomic_json(output / "in_progress.json", dict(status="started", role="transition",
                                checkpoint_sha256=file_sha256(output / "checkpoint.pkl"), index=completed+1,
                                resume_only_from_previous_complete_atomic_commit=True))
                            original_state = state
                            allocated_rng = state[2]
                            for role in (f"step{completed+1}_fit", f"step{completed+1}_finite_geometry",
                                         *((f"step{completed+1}_independent_H_JVP",) if plan["guard_H_JVP_records"] else ())):
                                _, allocated_rng = next_seed(allocated_rng, role)
                            # Reserve every potentially consumed role BEFORE any
                            # draw. Even an interrupted/rejected role cannot be
                            # presented as a future untouched audit stream.
                            save_state(output / "checkpoint.pkl", (state[0], state[1], allocated_rng),
                                completed_updates=completed, metadata=metadata)
                            atomic_json(output / "in_progress.json", dict(status="started", role="transition", index=completed+1,
                                checkpoint_sha256=file_sha256(output / "checkpoint.pkl"),
                                allocated_roles=allocated_rng["roles"][len(state[2]["roles"]):],
                                original_rng_sha256=canonical_sha256(state[2]),
                                reserved_next_rng_sha256=canonical_sha256(allocated_rng),
                                resume_only_from_previous_complete_atomic_commit=True))
                            state, transition_report = transition(plan, runtime, original_state,
                                output / f"step{completed+1}", index=completed+1)
                            state = (state[0], state[1], allocated_rng)
                            # Rejections still consume role RNG; no hidden retry.
                            completed += int(transition_report["accepted"])
                            replay_rows = probe[:4]
                            rp, rh = runtime["H"](state[0], replay_rows)
                            rq, _ = runtime["Q"](replay_rows)
                            metadata["numerical_replay"] = dict(rows=replay_rows, psi=rp, hpsi=rh, proposal=rq)
                            save_state(output / "checkpoint.pkl", state, completed_updates=completed, metadata=metadata)
                            atomic_json(output / "in_progress.json", dict(status="complete", role="transition", index=completed,
                                checkpoint_sha256=file_sha256(output / "checkpoint.pkl")))
                            report.update(completed_updates=completed, latest_transition=transition_report)
                            preserve()
                            if not transition_report["accepted"]:
                                break
                        # Endpoints and next audit stream are frozen atomically
                        # before querying any audit full-H product.
                        atomic_json(output / "endpoint_frozen.json", dict(parameter_sha256=pytree_sha256(state[0]),
                            completed_updates=completed, next_rng_sha256=canonical_sha256(state[2]),
                            accepted_steps_not_a_convergence_certificate=True))
                        atomic_json(output / "in_progress.json", dict(status="started", role="final_audit",
                            checkpoint_sha256=file_sha256(output / "checkpoint.pkl"),
                            resume_only_from_previous_complete_atomic_commit=True))
                        original_rng = state[2]
                        audit_seed, audit_rng = next_seed(original_rng, "final_independent_full_H_audit")
                        save_state(output / "checkpoint.pkl", (state[0], state[1], audit_rng),
                            completed_updates=completed, metadata=metadata)
                        atomic_json(output / "in_progress.json", dict(status="started", role="final_audit",
                            checkpoint_sha256=file_sha256(output / "checkpoint.pkl"),
                            allocated_seed=audit_seed, reserved_next_rng_sha256=canonical_sha256(audit_rng),
                            resume_only_from_previous_complete_atomic_commit=True))
                        audit, next_rng = paired_final_audit(plan, runtime, initial[0], state[0], original_rng, output / "final_audit")
                        state = (state[0], state[1], next_rng)
                        save_state(output / "checkpoint.pkl", state, completed_updates=completed, metadata=metadata)
                        atomic_json(output / "in_progress.json", dict(status="complete", role="final_audit",
                            checkpoint_sha256=file_sha256(output / "checkpoint.pkl"),
                            audit_sha256=file_sha256(output / "final_audit/audit.json")))
                        report.update(status="optimized_prefix_independently_audited" if completed else "no_accepted_update_independently_audited",
                                      completed_updates=completed, final_audit=audit)
            report["source_manifest_unchanged"] = all(file_sha256(ROOT / name) == digest for name, digest in manifest.items())
            if not report["source_manifest_unchanged"]:
                raise ValueError("numerical sources changed during the bounded run")
    except TimeoutError as error:
        report.update(status="wall_cost_gate_checkpoint_only_not_final_precision", error=str(error))
        if plan["profile"] == "local" and (output / "initial.pkl").exists() and (output / "checkpoint.pkl").exists():
            report["executable_GPU_restart_plan"] = str(write_gpu_restart_plan(output))
    except Exception as error:
        report.update(status="failed_closed_no_precision_claim", error=f"{type(error).__name__}: {error}")
        preserve()
        raise
    finally:
        preserve()
    atomic_json(output / "results.json", report)
    return report


def main() -> None:
    """Write a plan without queries, or execute separately authorized stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "prepare", "preflight", "train", "resume"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--j2", type=float, choices=(0., .125), default=.125)
    parser.add_argument("--sector", choices=("q2_K", "q0_X"), default="q2_K")
    parser.add_argument("--profile", choices=("local", "gpu"), default="local")
    args = parser.parse_args()
    if args.stage == "plan":
        if args.output.exists() or not args.output.resolve().is_relative_to(BASE.resolve()):
            raise ValueError("new dedicated result-plan path required")
        plan = make_plan(args.j2, args.sector, profile=args.profile)
        validate_plan(plan)
        atomic_json(args.output, plan)
        print(json.dumps(dict(status="plan_only_no_amplitude_or_H_query", plan=str(args.output))))
    else:
        if args.plan is None:
            parser.error("--plan is required for every numerical stage")
        print(json.dumps(run(json.loads(args.plan.read_text()), args.output, stage=args.stage)), flush=True)


if __name__ == "__main__":
    main()
