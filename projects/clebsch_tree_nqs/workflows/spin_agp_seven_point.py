"""Bounded seven-point, complete-A1 projected singlet AGP benchmark.

This is a non-neural computational-basis variational family.  It reuses the
established node-safe determinant, physical-H query, importance metric and
Hermitian-force implementations.  Each Hamiltonian resets optimization from
a sealed parent; no checkpoint is selected using the independent audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from typing import Any

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import (
    _pair_chain_comparison, _pair_dynamic_functions, _pair_padded_query_arguments,
    _pair_sine_fs, _pair_weighted_moments,
)
from su2_nqs.benchmarks.projected_singlet_agp import (
    ProjectedSingletAGP, square_gamma_a1_group, triangular_hnf_gamma_a1_group,
)
from su2_nqs.benchmarks.singlet_physics_readout import (
    importance_jackknife_observables, singlet_structure_factor_records,
    triangular_torus_structure_records,
)
from su2_nqs.benchmarks.spin_operator_information import three_bond_readout
from su2_nqs.benchmarks.tensor_network_readout import physical_spin_block_readout
from su2_nqs.frontier.provenance import (
    file_sha256, numerical_runtime_identity, transitive_local_source_manifest,
)
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange
from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries
from su2_nqs.vmc.regional_density import (
    conditional_regional_matrices, regional_density_summary, regional_query_plan,
)
from su2_nqs.vmc.sr import importance_amplitude_metric, importance_hermitian_force

ROOT = Path(__file__).resolve().parents[3]


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sealed_json(path: Path, payload: dict) -> dict:
    """Atomically write an independently replayable canonical JSON seal."""
    data = _plain(payload)
    data["result_sha256"] = canonical_sha256(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return data


def read_sealed(path: Path) -> dict:
    data = json.loads(path.read_text())
    body = dict(data)
    digest = body.pop("result_sha256")
    if canonical_sha256(body) != digest:
        raise ValueError(f"invalid result seal: {path}")
    return data


def lattice_contract(lattice: str, length: int, coupling: float) -> tuple:
    """Return complete physical Gamma.A1 maps and J1=1 PBC bonds.

    Sites are row-major and labels use physical down/up bits, site zero MSB.
    The triangle metric is the repository (1,0),(0,1),(1,1) convention.
    """
    if lattice not in {"triangular", "square"} or length < 2:
        raise ValueError("square/triangular torus of length >=2 required")
    if not np.isfinite(coupling) or coupling < 0:
        raise ValueError("finite nonnegative J2/J1 required")
    if lattice == "triangular":
        group = triangular_hnf_gamma_a1_group(((length, 0), (0, length)))
        bonds = triangular_j1j2_bonds(length, length, j1=1., j2=coupling, periodic=True)
    else:
        group = square_gamma_a1_group(length)
        bonds = square_j1j2_bonds(length, length, j1=1., j2=coupling, periodic=True)
    return group, bonds


def validate_spec(spec: dict) -> None:
    """Fail closed outside the registered seven finite 6x6 Hamiltonians."""
    expected = [("triangular", v) for v in (0., .05, .075, .125, .175)] + [
        ("square", 0.), ("square", .5)]
    if [(c["lattice"], c["j2_over_j1"]) for c in spec["cases"]] != expected:
        raise ValueError("the complete ordered seven-point registry is required")
    b = spec["budget"]
    for key in ("step_cap", "chains", "burn_sweeps", "audit_burn_sweeps",
                "training_blocks", "geometry_blocks", "audit_blocks",
                "decorrelation_sweeps", "H_batch_size", "tile_size", "line_search_cap"):
        if isinstance(b[key], bool) or not isinstance(b[key], int) or b[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if b["step_cap"] != 12 or b["milestones"] != [0, 6, 12] or b["chains"] < 4:
        raise ValueError("fixed 0/6/12 quality sequence and >=4 chains required")
    if b["training_blocks"] * b["chains"] % b["H_batch_size"]:
        raise ValueError("complete fixed H batches required")
    if b["training_wall_seconds"] > 2400 or b["total_wall_seconds"] > 3600:
        raise ValueError("per-point fixed GPU wall cap exceeded")
    if len(b["audit_seed_offsets"]) != 2:
        raise ValueError("two independent audit streams are required")
    offsets = set(range(b["step_cap"]))
    geometry = {b["geometry_seed_offset"] + i for i in range(b["step_cap"])}
    audit = set(b["audit_seed_offsets"])
    if offsets & geometry or offsets & audit or geometry & audit or len(audit) != 2:
        raise ValueError("independent seed schedules required")


class CaseRuntime:
    """One unchanged full real pairing family and complete physical projection."""
    def __init__(self, case: dict, budget: dict, parameters: np.ndarray,
                 *, amplitude_scale: float | None = None):
        self.case, self.budget = case, budget
        self.group, self.bonds = lattice_contract(case["lattice"], 6, case["j2_over_j1"])
        self.config = ProjectedSingletAGPConfig(36, self.group)
        if np.asarray(parameters).shape != (630,) or not np.isfinite(parameters).all():
            raise ValueError("finite 630-entry real symmetric pairing required")
        self.raw_fns = _pair_dynamic_functions(self.config, amplitude_scale=1., tile_size=budget["tile_size"])
        if amplitude_scale is None:
            probe = uniform_probe(36, 64, case["seed"] + 700)
            values = np.asarray(self.raw_fns["amplitude"](parameters, jnp.asarray(probe)))
            amplitude_scale = float(np.sqrt(np.mean(abs(values)**2)))
        if not np.isfinite(amplitude_scale) or amplitude_scale <= 1e-280:
            raise ValueError("warm start projection has unresolved/zero norm; no node floor")
        self.scale = amplitude_scale
        self.fns = _pair_dynamic_functions(self.config, amplitude_scale=self.scale,
                                          tile_size=budget["tile_size"])
        self.ledger = dict(H_centers=0, unique_H_queries=0, padded_H_queries=0,
                           sampler_calls=0, regional_queries=0)

    def amplitude(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.fns["amplitude"](jnp.asarray(p), jnp.asarray(x)))

    def sample(self, density, seed: int, blocks: int, *, audit: bool = False):
        self.ledger["sampler_calls"] += 1
        return sample_fixed_wavefunction_exchange(lambda x: np.sqrt(density(x)),
            n_sites=36, n_up=18, chain_count=self.budget["chains"],
            burn_sweeps=self.budget["audit_burn_sweeps"] if audit else self.budget["burn_sweeps"],
            measurement_blocks=blocks, decorrelation_sweeps=self.budget["decorrelation_sweeps"],
            seed=seed)

    def plans(self, x: np.ndarray) -> list:
        result = []
        for begin in range(0, len(x), self.budget["H_batch_size"]):
            plan = compile_physical_h_queries(x[begin:begin+self.budget["H_batch_size"]], self.bonds)
            result.append((begin, plan, _pair_padded_query_arguments(plan)))
        return result

    def action(self, p: np.ndarray, plans: list) -> tuple[np.ndarray, np.ndarray]:
        aa, bb = [], []
        for _, plan, args in plans:
            a, b = self.fns["action"](jnp.asarray(p), *args)
            aa.append(np.asarray(a)); bb.append(np.asarray(b))
            self.ledger["H_centers"] += len(plan.centers)
            self.ledger["unique_H_queries"] += len(plan.configurations)
            self.ledger["padded_H_queries"] += len(args[0])
        return np.concatenate(aa), np.concatenate(bb)

    def jacobian(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(self.fns["center_jacobian"](
            jnp.asarray(p), jnp.asarray(x[i:i+self.budget["tile_size"]])))
            for i in range(0, len(x), self.budget["tile_size"])])


def uniform_probe(n_sites: int, count: int, seed: int, *, n_up: int | None = None) -> np.ndarray:
    """Return deterministic independent fixed-m binary probes (not VMC)."""
    rng = np.random.default_rng(seed)
    bits = np.zeros((count, n_sites), np.int8)
    for row in bits:
        row[rng.choice(n_sites, n_sites//2 if n_up is None else n_up, replace=False)] = 1
    return bits


def singlet_and_a1_probe(runtime: CaseRuntime, p: np.ndarray, seed: int) -> dict:
    """Check S+ annihilation and physical A1 generators at N36 without ED.

    For a fixed m=+1 target, each up spin contributes the m=0 coefficient
    obtained by lowering that bit.  These identities supplement exhaustive
    exact-small S² tests; random probes do not certify the complete state.
    """
    targets = uniform_probe(36, 8, seed, n_up=19)
    source = []
    for row in targets:
        for site in np.flatnonzero(row):
            lowered = row.copy(); lowered[site] = 0; source.append(lowered)
    values = runtime.amplitude(p, np.asarray(source)).reshape(8, 19)
    singlet_error = float(np.max(abs(values.sum(1)) / np.maximum(np.sum(abs(values), axis=1), 1e-300)))
    probe = uniform_probe(36, 16, seed+1)
    a = runtime.amplitude(p, probe)
    defects = []
    for image in runtime.group[::max(1, len(runtime.group)//8)]:
        b = runtime.amplitude(p, probe[:, image])
        defects.append(float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-300)))
    return dict(sampled_Splus_relative_l1=singlet_error,
                sampled_A1_relative_norm=max(defects), tolerance=2e-10,
                exhaustive=False, exact_singlet_basis="symmetric spin-singlet pairing",
                complete_group_order=len(runtime.group))


def register(spec: dict, config_path: Path) -> Path:
    """Seal one isolated source/input snapshot before preflight or optimization."""
    validate_spec(spec)
    output = ROOT/spec["output"]
    sources = transitive_local_source_manifest((Path(__file__),), root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"))
    inputs = {item["path"]: item["sha256"] for item in spec["parents"].values()}
    for item in spec["parents"].values():
        inputs[item["source_completion"]] = item["source_completion_sha256"]
        inputs[item["source_registration"]] = item["source_registration_sha256"]
        completion = json.loads((ROOT/item["source_completion"]).read_text())
        if completion["outputs"].get(Path(item["path"]).name) != item["sha256"]:
            raise ValueError("checkpoint does not match its historical completion seal")
    for path, digest in inputs.items():
        if file_sha256(ROOT/path) != digest:
            raise ValueError(f"parent checkpoint changed: {path}")
    data = dict(config=spec, config_sha256=file_sha256(config_path), sources=sources,
        parent_checkpoints=inputs, state_family="630-entry real singlet AGP with complete physical A1",
        neural_model=False, ViT=False, autoregressive=False, normalization="unknown",
        parent_optimizer_inherited=False, selection="last completed fixed-budget SR step",
        new_many_body_ED=0, direct_fusion_training=False)
    if (output/"registration.json").exists():
        old = read_sealed(output/"registration.json")
        for key in ("config_sha256", "sources", "parent_checkpoints"):
            if old[key] != data[key]:
                raise ValueError("registered config/source/input changed")
        return output
    output.mkdir(parents=True, exist_ok=True)
    for path in sources:
        dest = output/"source_snapshot"/path
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/path, dest)
    data["registered_utc"] = datetime.now(timezone.utc).isoformat()
    sealed_json(output/"registration.json", data)
    return output


def preflight(spec: dict, output: Path) -> dict:
    """Measure actual complete projection/sampling/H/Jacobian cost before release."""
    report = output/"preflight.json"
    if report.exists():
        return read_sealed(report)
    rows = []
    for index in (0, 3, 5, 6):
        case = spec["cases"][index]
        p = np.asarray(load_checkpoint(ROOT/spec["parents"][case["parent"]]["path"])
                       ["params"]["pairing_real"], float)
        began = time.perf_counter(); rt = CaseRuntime(case, spec["budget"], p)
        x = uniform_probe(36, 8, case["seed"]+701)
        a = rt.amplitude(p, x)
        # Independent host signed-determinant reference, complete Reynolds sum.
        matrix = np.zeros((36, 36)); first, second = np.triu_indices(36, 1)
        matrix[first, second] = p; matrix[second, first] = p
        ref = ProjectedSingletAGP(matrix, rt.group).amplitudes(x) / rt.scale
        replay = float(np.max(abs(a-ref))/max(np.max(abs(ref)), 1e-300))
        sym = singlet_and_a1_probe(rt, p, case["seed"]+702)
        if replay > 2e-10 or max(sym["sampled_Splus_relative_l1"], sym["sampled_A1_relative_norm"]) > 2e-10:
            raise ValueError("independent AGP replay/SU2/A1 validation failed")
        density = lambda z: abs(rt.amplitude(p, z))**2 + spec["budget"]["positive_floor"]
        start = time.perf_counter()
        sample = rt.sample(density, case["seed"]+703, 4)
        sampling_seconds = time.perf_counter()-start
        plans = rt.plans(sample.configurations)
        start = time.perf_counter(); actions = rt.action(p, plans)
        full_h_compile_seconds = time.perf_counter()-start
        start = time.perf_counter(); rt.action(p, plans)
        full_h_warm_seconds = time.perf_counter()-start
        start = time.perf_counter(); jac = rt.jacobian(p, sample.configurations)
        jac_compile_seconds = time.perf_counter()-start
        start = time.perf_counter(); rt.jacobian(p, sample.configurations)
        jac_warm_seconds = time.perf_counter()-start
        q = density(sample.configurations)
        moments = _pair_weighted_moments(*actions, q, 36)
        weight = abs(actions[0])**2/q
        estimate = (full_h_warm_seconds*spec["budget"]["training_blocks"]/4
                    * (1+spec["budget"]["line_search_cap"])
                    + jac_warm_seconds*spec["budget"]["training_blocks"]/4
                    + sampling_seconds*2)
        rows.append(dict(case=case["name"], amplitude_scale=rt.scale, group_order=len(rt.group),
            bond_count=len(rt.bonds), independent_numpy_amplitude_relative_error=replay,
            symmetry_probe=sym, sampling_seconds=sampling_seconds,
            sampling_records=len(sample.configurations), sampling_acceptance=sample.acceptance_fraction,
            H_compile_seconds=full_h_compile_seconds, H_warm_seconds=full_h_warm_seconds,
            jacobian_compile_seconds=jac_compile_seconds, jacobian_warm_seconds=jac_warm_seconds,
            finite_jacobian=bool(np.isfinite(jac).all()),
            diagnostic_energy_per_site=float(moments[0]), diagnostic_variance_per_site=float(moments[1]),
            diagnostic_weight_ESS_fraction=float(weight.sum()**2/(weight@weight*len(weight))),
            conservative_estimated_warm_step_seconds=estimate, seconds=time.perf_counter()-began))
        print(json.dumps(rows[-1]), flush=True)
    passed = all(r["finite_jacobian"] and r["conservative_estimated_warm_step_seconds"] < 150
                 for r in rows)
    return sealed_json(report, dict(rows=rows, training_released=passed,
        gate="finite nonzero complete projection, independent host replay, S+/A1 probes, bounded actual cost",
        energy_is_diagnostic_only=True, equilibrium_certified=False, runtime=numerical_runtime_identity()))


def structure_records(bits: np.ndarray, lattice: str) -> dict:
    if lattice == "triangular":
        return triangular_torus_structure_records(bits, 6, 6, strict_singlet=True)
    records = singlet_structure_factor_records(bits, np.array([divmod(i, 6) for i in range(36)]),
        {"Gamma": (0., 0.), "Neel": (.5, .5), "stripe_x": (.5, 0.), "stripe_y": (0., .5)},
        strict_singlet=True)
    records["stripe_A1_average"] = (records["stripe_x"]+records["stripe_y"])/2
    return records


def run_case(spec: dict, output: Path, index: int) -> dict:
    """Optimize twelve bounded steps and audit frozen 0/6/12 checkpoints."""
    if not read_sealed(output/"preflight.json")["training_released"]:
        raise ValueError("actual GPU preflight did not release training")
    case, budget = spec["cases"][index], spec["budget"]
    folder = output/case["name"]
    if (folder/"results.json").exists():
        return read_sealed(folder/"results.json")
    folder.mkdir(exist_ok=True)
    parent = spec["parents"][case["parent"]]
    initial = np.asarray(load_checkpoint(ROOT/parent["path"])["params"]["pairing_real"], float)
    began = time.perf_counter(); previous_elapsed = 0.
    if (folder/"progress.pkl").exists():
        restored = load_checkpoint(folder/"progress.pkl")
        p = np.asarray(restored["params"]["pairing_real"])
        state = restored["optimizer_state"]
        history = state["history"]; previous_elapsed = state["elapsed_seconds"]
        rt = CaseRuntime(case, budget, initial, amplitude_scale=restored["metadata"]["amplitude_scale"])
        rt.ledger = state["ledger"]
    else:
        p = initial.copy(); history = []
        rt = CaseRuntime(case, budget, initial)
        sealed_json(folder/"registration.json", dict(case=case, parent=parent,
            suite_registration_sha256=read_sealed(output/"registration.json")["result_sha256"],
            group_order=len(rt.group), physical_bonds=rt.bonds,
            parent_lattice="triangular", parent_optimizer_inherited=False,
            initialization="preregistered nearest saved coupling; square uses cross-lattice projection",
            amplitude_scale=rt.scale, family_parameter_count=630,
            complete_A1=True, strict_total_q=0))
    elapsed = lambda: previous_elapsed + time.perf_counter()-began
    gauge_norm = np.linalg.norm(initial)
    def freeze(step: int, values: np.ndarray):
        path = folder/f"step_{step:03d}.pkl"
        if path.exists():
            return
        save_checkpoint(path, {"pairing_real": values.copy()}, metadata=dict(
            n_sites=36, j2=case["j2_over_j1"], lattice=case["lattice"],
            group_order=len(rt.group), amplitude_scale=rt.scale, completed_step=step,
            strict_total_q=0, neural_model=False, selected_using="fixed quality milestone"))
    def checkpoint():
        save_checkpoint(folder/"progress.pkl", {"pairing_real": p},
            optimizer_state=dict(history=history, elapsed_seconds=elapsed(), ledger=rt.ledger),
            metadata=dict(amplitude_scale=rt.scale, optimizer="fresh-sample Hermitian-center SR"))
        sealed_json(folder/"progress.json", dict(history=history, seconds=elapsed(), ledger=rt.ledger))
    freeze(0, initial)
    if not (folder/"frozen_endpoint.json").exists():
        for step in range(len(history), budget["step_cap"]):
            if elapsed() > budget["training_wall_seconds"]:
                break
            step_start = time.perf_counter(); old_p = p.copy()
            density = lambda x: abs(rt.amplitude(old_p, x))**2 + budget["positive_floor"]
            sample = rt.sample(density, case["seed"]+step, budget["training_blocks"])
            x = sample.configurations; q = density(x); plans = rt.plans(x)
            a, b = rt.action(p, plans)
            old = _pair_weighted_moments(a, b, q, 36)
            jac = rt.jacobian(p, x)
            metric = importance_amplitude_metric(a, jac, q)
            force = importance_hermitian_force(a, jac, b, q, n_sites=36)
            unit = p/np.linalg.norm(p); projector = np.eye(630)-np.outer(unit, unit)
            metric = projector@metric@projector; force = projector@force
            eig = np.linalg.eigvalsh(metric)
            damping = budget["relative_damping"]*np.trace(metric)/630+budget["absolute_damping"]
            delta = -budget["step_size_total_energy"]*36*np.linalg.solve(metric+damping*np.eye(630), force)
            delta = projector@delta
            tangent_fs = np.sqrt(max(0., delta@metric@delta/2))
            if tangent_fs > budget["maximum_sine_FS"]:
                delta *= budget["maximum_sine_FS"]/tangent_fs
            tangent_fs = np.sqrt(max(0., delta@metric@delta/2))
            cauchy = abs(float(force@delta))/max(2*np.sqrt(old[1]/36)*tangent_fs, 1e-300)
            if cauchy > 1.000001:
                raise ValueError("Hermitian force violates physical-metric Cauchy bound")
            cal = rt.sample(density, case["seed"]+budget["geometry_seed_offset"]+step,
                            budget["geometry_blocks"])
            cx = cal.configurations; cq = density(cx); ca = rt.amplitude(p, cx)
            fraction = 1.; accepted = False; attempts = []; new = old
            fs = geometry_fs = 0.
            for _ in range(budget["line_search_cap"]):
                candidate = p+fraction*delta; candidate *= gauge_norm/np.linalg.norm(candidate)
                fs = _pair_sine_fs(a, rt.amplitude(candidate, x), q)
                geometry_fs = _pair_sine_fs(ca, rt.amplitude(candidate, cx), cq)
                if max(fs, geometry_fs) > 1.02*budget["maximum_sine_FS"]:
                    attempts.append(dict(fraction=fraction, status="physical_radius"))
                    fraction *= .5; continue
                pair = rt.action(candidate, plans)
                candidate_moments = _pair_weighted_moments(*pair, q, 36)
                accepted = bool(candidate_moments[0] < old[0] and
                                candidate_moments[1] <= budget["maximum_variance_ratio"]*old[1])
                attempts.append(dict(fraction=fraction, energy=float(candidate_moments[0]),
                    variance=float(candidate_moments[1]), status="accepted" if accepted else "training_objective"))
                if accepted:
                    p = candidate; new = candidate_moments; break
                fraction *= .5
            history.append(dict(step=step+1, accepted=accepted, energy_before=float(old[0]),
                variance_before=float(old[1]), energy_after=float(new[0]), variance_after=float(new[1]),
                training_sine_FS=fs, geometry_sine_FS=geometry_fs, damping=float(damping),
                physical_metric_rank=int(np.count_nonzero(eig > max(eig[-1]*1e-8, 1e-12))),
                force_Cauchy_ratio=cauchy, training_acceptance=sample.acceptance_fraction,
                geometry_acceptance=cal.acceptance_fraction, attempts=attempts,
                step_seconds=time.perf_counter()-step_start, seconds=elapsed()))
            checkpoint()
            if step+1 in budget["milestones"]:
                freeze(step+1, p)
            print(json.dumps(dict(case=case["name"], **history[-1])), flush=True)
        freeze(len(history), p)
        states = {f"step_{s:03d}": file_sha256(folder/f"step_{s:03d}.pkl")
                  for s in sorted(set([0, len(history)]+[s for s in (6, 12) if s <= len(history)]))}
        sealed_json(folder/"frozen_endpoint.json", dict(checkpoints=states, completed_steps=len(history),
            audit_opened=False, seconds=elapsed(), endpoint_selection="last completed step"))
    frozen = read_sealed(folder/"frozen_endpoint.json")
    params = {}
    for label, digest in frozen["checkpoints"].items():
        path = folder/(label+".pkl")
        if file_sha256(path) != digest:
            raise ValueError("immutable quality checkpoint changed")
        params[label] = np.asarray(load_checkpoint(path)["params"]["pairing_real"])
    audits = []
    for audit_index, offset in enumerate(budget["audit_seed_offsets"]):
        path = folder/f"audit_{audit_index}.json"
        if path.exists():
            audits.append(read_sealed(path)); continue
        if elapsed() > budget["total_wall_seconds"]:
            break
        density = lambda x: sum(abs(rt.amplitude(v, x))**2 for v in params.values())+budget["positive_floor"]
        sample = rt.sample(density, case["seed"]+offset, budget["audit_blocks"], audit=True)
        x = sample.configurations; q = density(x); plans = rt.plans(x)
        arrays = {label: rt.action(values, plans) for label, values in params.items()}
        comparison = _pair_chain_comparison(arrays, q, budget["chains"], 36)
        groups = np.arange(len(x)) % budget["chains"]
        saved = dict(configurations=x, density=q)
        comparison["structure_factors"] = {}
        for label, (a, b) in arrays.items():
            saved[label+"_psi"] = a; saved[label+"_Hpsi"] = b
            comparison["structure_factors"][label] = importance_jackknife_observables(
                structure_records(x, case["lattice"]), a, q, groups)
        comparison["regions"] = {}
        for name, region in spec["regions"].items():
            plan = regional_query_plan(x, region["sites"])
            rq = density(plan.configurations)
            rt.ledger["regional_queries"] += len(plan.configurations)*(len(params)+1)
            comparison["regions"][name] = {}
            for label, values in params.items():
                summary = regional_density_summary(conditional_regional_matrices(
                    plan, rt.amplitude(values, plan.configurations), rq), groups)
                rho = summary["rho"]; jacks = summary["jackknife_rhos"]
                block = physical_spin_block_readout(rho)
                readout = dict(block=block, strict_multiplicity_spectrum_reported=False,
                               input_twirl=False, positivity_repair=False)
                p_deletes = {key: np.asarray([physical_spin_block_readout(r)["p_q"][key]
                                             for r in jacks]) for key in block["p_q"]}
                readout["p_q_chain_SE"] = {key: float(np.sqrt((budget["chains"]-1)
                    * np.mean((values-values.mean())**2))) for key, values in p_deletes.items()}
                if len(region["sites"]) == 6:
                    def observe(r):
                        row = three_bond_readout(r, region["left"])
                        return np.array([row["high_q3"], row["orbits"]["Y"]["connected"]])
                    central = observe(rho); deletes = np.asarray([observe(r) for r in jacks])
                    se = np.sqrt((budget["chains"]-1)*np.mean((deletes-deletes.mean(0))**2, axis=0))
                    readout["scalar_channels"] = {key: dict(value=float(value), chain_SE=float(err))
                        for key, value, err in zip(("high_q3", "Y_connected"), central, se)}
                comparison["regions"][name][label] = readout
                saved[name+"_"+label+"_rho"] = rho
                saved[name+"_"+label+"_jackknife_rhos"] = jacks
        np.savez_compressed(folder/f"audit_{audit_index}.npz", **saved)
        comparison.update(seed=case["seed"]+offset, sampling_acceptance=sample.acceptance_fraction,
            checkpoints=frozen["checkpoints"], raw_file_sha256=file_sha256(folder/f"audit_{audit_index}.npz"))
        audits.append(sealed_json(path, comparison))
        checkpoint()
        print(json.dumps(dict(case=case["name"], audit=audit_index, states=comparison["states"])), flush=True)
    # Pool independent audit chains using global ratios; retain every raw stream.
    pooled = None
    if audits:
        chunks = [np.load(folder/f"audit_{i}.npz") for i in range(len(audits))]
        def interleave(items):
            return np.concatenate([a.reshape(budget["audit_blocks"], budget["chains"])
                                   for a in items], axis=1).reshape(-1)
        arrays = {label: (interleave([c[label+"_psi"] for c in chunks]),
                          interleave([c[label+"_Hpsi"] for c in chunks])) for label in params}
        pooled = _pair_chain_comparison(arrays, interleave([c["density"] for c in chunks]),
                                        budget["chains"]*len(chunks), 36)
        for chunk in chunks:
            chunk.close()
    symmetry = {label: singlet_and_a1_probe(rt, values, case["seed"]+800+i)
                for i, (label, values) in enumerate(params.items())}
    if any(max(v["sampled_Splus_relative_l1"], v["sampled_A1_relative_norm"]) > 2e-10
           for v in symmetry.values()):
        raise ValueError("frozen quality checkpoint violates independent SU2/A1 probes")
    result = sealed_json(folder/"results.json", dict(case=case, history=history, audits=audits,
        pooled=pooled, frozen_checkpoints=frozen["checkpoints"], symmetry_probes=symmetry,
        completed_steps=len(history), accepted_steps=sum(h["accepted"] for h in history),
        seconds=elapsed(), ledger=rt.ledger, candidate_families=1, parameter_count=630,
        neural_model=False, ViT=False, new_many_body_ED=0, complete_A1=True,
        phase_claim=False, QSL_claim=False, convergence_certified=False, equilibrium_certified=False,
        complete_bounded_audit=len(audits)==2,
        interpretation="Same-family quality sequence; finite6x6PBC. Mean-field naming establishes no phase.",
        warnings=["Chain SE and importance ESS do not establish equilibrium or omitted-tail bias.",
                  "Regional p(q) and scalar channels are legal; no strict spectrum from noisy noninvariant RDM.",
                  "These finite PBC energies cannot be ranked against thermodynamic iPEPS energies."]))
    sealed_json(folder/"completion.json", dict(status="complete" if len(audits)==2 else "bounded_incomplete_audit",
        outputs={p.name: file_sha256(p) for p in folder.iterdir()
                 if p.is_file() and p.name != "completion.json"}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "case", "summary"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text()); output = register(spec, args.config)
    if args.mode == "preflight":
        preflight(spec, output)
    elif args.mode == "case":
        if args.index is None or not 0 <= args.index < 7:
            raise ValueError("one registered case index 0..6 required")
        run_case(spec, output, args.index)
    else:
        rows = []
        for case in spec["cases"]:
            path = output/case["name"]/"results.json"
            if path.exists():
                row = read_sealed(path)
                rows.append({key: row[key] for key in ("case", "completed_steps", "accepted_steps",
                    "pooled", "frozen_checkpoints", "seconds", "complete_bounded_audit")})
        sealed_json(output/"results.json", dict(rows=rows, requested_cases=7, completed_cases=len(rows),
            new_many_body_ED=0, neural_model=False, QSL_claim=False))


if __name__ == "__main__":
    main()
