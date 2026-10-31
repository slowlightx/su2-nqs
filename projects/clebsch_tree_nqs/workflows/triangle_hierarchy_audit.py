"""Bounded independent complete-A1 audit of a saved physical CT bank.

No optimization, parameter fitting, or state selection occurs here. The cost
preflight is not an energy estimate. Finite-burn Born estimates are explicitly
distinguished from certified equilibrium and from node-complete H^2 moments.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import json
from numbers import Integral
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.models.triangular_clebsch_gauge import triangular_bank_reduced_tree
from su2_nqs.models.triangular_clebsch_active_layout import repack_active_clebsch_bank
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.conditioned_patch_proposal import physical_leaf_diagonals, make_conditioned_patch_proposal
from su2_nqs.vmc.exchange_bridge_importance import make_exchange_bridge, importance_audit
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.projected_spin_linear import make_terminal_tensor_products
from su2_nqs.vmc.dual_sampled_sr import importance_overlap_diagnostics
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action
from .triangle_dictionary_vmc_local import make_record_sampler, summarize_born_audit
from .triangle_scale_cost import balanced_block_probe_states


def audit_cost_forecast(*, chains: int, records: int, burn: int, between: int,
                        amplitude_batch_seconds: float, H_batch_seconds: float) -> dict:
    """Conservative work forecast using actual same-chain-batch kernel timings."""
    for name, count in (("chains", chains), ("records", records), ("burn", burn), ("between", between)):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError(name+" must be a positive integer")
    costs = np.asarray([amplitude_batch_seconds, H_batch_seconds], float)
    if not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError("measured kernel costs must be finite and positive")
    proposal_batches = burn+records*between
    return dict(sample_records=int(chains*records), full_H_centers=int(chains*records),
        proposal_batches=int(proposal_batches),
        estimated_sampling_seconds=float((proposal_batches+1)*amplitude_batch_seconds),
        estimated_H_seconds=float(records*H_batch_seconds),
        conservative_seconds=float(1.5*((proposal_batches+1)*amplitude_batch_seconds+records*H_batch_seconds)),
        forecast_is_not_equilibrium_certificate=True)


def checked_active_widths(value, *, nodes: int) -> tuple[dict[int, int], ...]:
    """Read ordered width metadata without relying on dictionary insertion order."""
    def label(q):
        if isinstance(q, str):
            try:
                parsed = int(q)
            except ValueError as error:
                raise ValueError("integer node/doubled-q labels required") from error
            if str(parsed) != q:
                raise ValueError("canonical integer node/doubled-q labels required")
            return parsed
        if isinstance(q, bool) or not isinstance(q, Integral):
            raise ValueError("integer node/doubled-q labels required")
        return int(q)
    if isinstance(value, Mapping):
        indexed = {label(k): v for k, v in value.items()}
        if len(indexed) != nodes or set(indexed) != set(range(nodes)):
            raise ValueError("active-width node indices must cover the complete hierarchy")
        value = [indexed[i] for i in range(nodes)]
    if len(value) != nodes or any(not isinstance(row, Mapping) for row in value):
        raise ValueError("one doubled-q width mapping per node required")
    result = tuple({label(q): int(d) for q, d in row.items()} for row in value)
    if any(any(isinstance(d, bool) or not isinstance(d, Integral) or d < 1 for d in row.values())
           for row in value):
        raise ValueError("active multiplicities must be positive integers")
    return result


def finite_relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Relative replay discrepancy that cannot accidentally pass on NaN."""
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("finite matching nonempty replay arrays required")
    return float(np.max(abs(a-b))/max(float(np.max(abs(b))), 1e-300))


def compare_saved_A1_amplitudes(spec, cfg, images, prepared, amplitudes, output, result, timed):
    """Reused-pool, common-Q comparison; no H, fitting, or global-F claim."""
    samples_path = Path(spec['overlap_samples_path'])
    for field, path in [('overlap_samples_sha256', samples_path),
                         ('overlap_candidate_sha256', Path(spec['overlap_candidate_checkpoint']))]:
        expected = spec.get(field)
        if expected is not None and file_sha256(path) != expected:
            raise ValueError('fixed overlap input SHA mismatch: '+field)
    old_registration = json.loads((samples_path.parent/'registration.json').read_text())
    old_report = json.loads((samples_path.parent/'results.json').read_text())
    proposal_path = Path(spec['proposal_checkpoint'])
    if (old_registration['checkpoint_sha256'] != file_sha256(proposal_path)
            or old_report.get('status') != 'development_audit_complete'):
        raise ValueError('old sample-pool physical parent identity is not verified')
    with np.load(samples_path) as saved:
        rows = np.asarray(saved['rows'])
    if rows.ndim != 3 or rows.shape[-1] != cfg.length**2:
        raise ValueError('old audit rows must have time,chain,physical-site axes')
    chains = rows.shape[1]
    flat_rows = rows.reshape(-1, rows.shape[-1])
    arrays, identities = {}, {}

    def evaluate(label, callback, bank):
        batches = []
        for start in range(0, len(flat_rows), 8):
            batch = flat_rows[start:start+8]
            batches.append(np.asarray(timed(label+'_A1_'+str(start), callback, bank, batch)))
            result['actual_amplitude_centers'] += len(batch)
        value = np.concatenate(batches)
        if not np.isfinite(value).all():
            raise FloatingPointError('nonfinite complete-A1 overlap amplitudes')
        return value

    arrays['reference'] = evaluate('reference', amplitudes, prepared)
    for label, path in [('proposal', proposal_path),
                         ('candidate', Path(spec['overlap_candidate_checkpoint']))]:
        frozen = load_checkpoint(path)
        bank = frozen['params']; meta = frozen.get('metadata', {})
        widths = meta.get('active_top_widths', meta.get('embedding', {}).get('active_top_widths'))
        if widths is None:
            raise ValueError('comparison bank requires explicit active-width metadata')
        widths = checked_active_widths(widths, nodes=len(bank['top']))
        execution = repack_active_clebsch_bank(bank, cfg, widths)
        query = jax.jit(jax.vmap(make_feature_lut_projected_amplitude(
            cfg, images, chunk=4, active_top_widths=widths), in_axes=(None, 0)))
        identities[label] = dict(checkpoint=str(path), checkpoint_sha256=file_sha256(path),
                                 params_sha256=pytree_sha256(bank))
        arrays[label] = evaluate(label, query, prepare_feature_lut_bank(execution, cfg))
    # Do NOT treat the tau4 proposal as the tau8 reference Born distribution.
    # A common positive Q is required in both overlap norms and coherence.
    density = abs(arrays['proposal'])**2
    if not np.isfinite(density).all() or np.any(density <= 0):
        raise FloatingPointError('source Born pool contains a zero/nonfinite proposal density')
    measured = importance_overlap_diagnostics(arrays['reference'], arrays['candidate'], density)
    jackknife = []
    for chain in range(chains):
        mask = np.arange(len(flat_rows)) % chains != chain
        jackknife.append(importance_overlap_diagnostics(arrays['reference'][mask],
            arrays['candidate'][mask], density[mask])['fidelity_squared'])
    theta = np.asarray(jackknife)
    measured['fidelity_chain_jackknife_SE'] = float(np.sqrt((chains-1)/chains*np.sum((theta-theta.mean())**2)))
    np.savez_compressed(output/'overlap_amplitudes.npz', rows=rows,
        reference=arrays['reference'], candidate=arrays['candidate'],
        proposal=arrays['proposal'], proposal_density=density)
    atomic_json(output/'overlap_parent_identities.json', dict(identities,
        old_registration_sha256=file_sha256(samples_path.parent/'registration.json'),
        old_samples_sha256=file_sha256(samples_path), sample_pool_refitted_or_selected=False))
    return dict(status='reused_pool_A1_overlap_complete', overlap=measured,
        independent_new_chains=False, sample_pool_reused_for_development=True,
        old_pool_split_Rhat_not_repaired=True,
        old_pool_unsplit_Rhat=old_report['moments']['unsplit_energy_Rhat'],
        complete_support_certified=False, global_fidelity_certified=False,
        wavefunction_norm_computed=False, no_optimizer_or_candidate_selection=True,
        candidate_is_not_automatically_approved_for_training=True,
        energy_estimate=False, no_H_queries=True)


def independent_importance(bank, cfg, images, prepared, amplitudes, action, bonds,
                           spec, output, result, timed, started, seconds_cap) -> dict:
    """IID full-support audit, including exact nodes in the H^2 numerator.

    The fixed proposal uses RAW-tree physical leaf marginals, not the target
    coherent-A1 density. Correct importance weights evaluate that target
    separately. A disjoint norm-only pilot cannot contribute to energy moments.
    """
    count = spec.get("importance_samples", 128)
    pilot_count = spec.get("norm_pilot_samples", 64)
    groups = spec.get("independent_groups", 16)
    seed = spec.get("seed", 261023271)
    if (any(isinstance(x, bool) or not isinstance(x, Integral) for x in (count, pilot_count, groups))
            or groups < 4 or count < 16 or pilot_count < 16 or count % groups):
        raise ValueError("importance audit needs >=16 draws, >=4 equal independent groups")
    state = triangular_bank_reduced_tree(bank, cfg)
    features = tuple({q: np.asarray(v) for q, v in zip(cfg.local_q, row, strict=True)}
                     for row in prepared["local_features"])
    diagonals, raw_norm = physical_leaf_diagonals(state, features)
    patches = np.asarray([b.sites for b in triangular_schur_hierarchy(cfg.length).blocks], np.int32)
    log_base, draw_base, proposal = make_conditioned_patch_proposal(patches, diagonals, images,
        n_up=cfg.length**2//2, uniform_fraction=.05, image_batch_size=4)
    # A frozen SIX-edge bridge improves coverage without multiplying the
    # expensive proposal density by every H bond. The positive uniform floor,
    # not this subset, guarantees complete fixed-Sz support.
    indices = np.linspace(0, len(bonds)-1, min(6, len(bonds)), dtype=int)
    edges = tuple((bonds[i][0], bonds[i][1]) for i in indices)
    density, draw = make_exchange_bridge(log_base, draw_base, edges,
        np.full(len(edges), 1/len(edges)), n_sites=cfg.length**2, bridge_fraction=.5)
    proposal.update(bridge_fraction=.5, bridge_edges=edges, all_H_neighbors_pushed=False,
                    raw_norm_diagnostics=raw_norm)
    atomic_json(output/"importance_proposal.json", proposal)
    np.savez_compressed(output/"importance_proposal.npz", patches=patches,
                        physical_diagonals=diagonals, images=images)
    pilot_key, audit_key = jax.random.split(jax.random.PRNGKey(seed))
    pilot_rows, _ = timed("independent_norm_pilot_draw", lambda key: draw(key, pilot_count), pilot_key)
    pilot_logq = np.asarray(timed("norm_pilot_proposal_density", density, pilot_rows)[0])
    pilot_psi = np.asarray(timed("norm_pilot_A1_amplitudes", amplitudes, prepared, pilot_rows))
    result["actual_amplitude_centers"] += pilot_count
    weight = abs(pilot_psi)**2/np.exp(pilot_logq)
    ess = float(weight.sum()**2/np.sum(weight**2)) if np.any(weight) else 0.
    norm_pilot = dict(samples=pilot_count, norm_estimate=float(weight.mean()),
        norm_iid_se=float(weight.std(ddof=1)/np.sqrt(pilot_count)), effective_contributions=ess,
        max_sample_fraction=float(weight.max()/weight.sum()) if weight.sum() else 1.,
        energy_or_H_evaluations=0, disjoint_from_energy_audit=True,
        proposal_refitted_after_pilot=False)
    atomic_json(output/"norm_pilot.json", norm_pilot)
    np.savez_compressed(output/"norm_pilot_samples.npz", rows=np.asarray(pilot_rows),
                        logQ=pilot_logq, psi=pilot_psi)
    if ess < max(8., .05*pilot_count):
        return dict(status="importance_audit_not_started_low_norm_pilot_ESS", energy_estimate=False,
                    norm_pilot=norm_pilot, proposal=proposal)
    forecast = 1.5*count*result["timings"]["full_H_warm_center"]
    if forecast > seconds_cap-(perf_counter()-started):
        return dict(status="importance_audit_not_started_due_to_measured_budget", energy_estimate=False,
                    norm_pilot=norm_pilot, conservative_H_seconds=forecast, proposal=proposal)
    rows, bridge_used = timed("independent_importance_draw", lambda key: draw(key, count), audit_key)
    logq = np.asarray(timed("importance_density", density, rows)[0])
    psis = []; hpsis = []
    batch_size = spec.get("H_batch_size", 4)
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("positive H batch size required")
    for start in range(0, count, batch_size):
        batch = rows[start:start+batch_size]
        p, h = timed("independent_importance_full_H_"+str(start), action, prepared, batch)
        psis.extend(np.asarray(p)); hpsis.extend(np.asarray(h))
        result["actual_full_H_centers"] += len(batch)
        np.savez_compressed(output/"partial_importance_audit.npz",
            rows=np.asarray(rows[:len(psis)]), logQ=logq[:len(psis)],
            psi=np.asarray(psis), Hpsi=np.asarray(hpsis))
    psis, hpsis = np.asarray(psis), np.asarray(hpsis)
    independent_groups = np.repeat(np.arange(groups), count//groups)
    report = importance_audit(psis, hpsis, np.exp(logq), independent_groups,
        n_sites=cfg.length**2, bootstrap_replicates=300, seed=int(seed)+1)
    np.savez_compressed(output/"importance_audit_samples.npz", rows=np.asarray(rows), logQ=logq,
        psi=psis, Hpsi=hpsis, groups=independent_groups, bridge_used=np.asarray(bridge_used))
    return dict(status="independent_importance_development_audit_complete", energy_estimate=True,
        importance=report, norm_pilot=norm_pilot, proposal=proposal,
        variance_node_complete_in_expectation=True, empirical_tail_coverage_certified=False,
        publication_quality_gate_passed=False, proposal_is_target_density=False,
        physical_wavefunction_norm_computed=False, physical_norm_estimated_independently=True)


def run(spec: dict, output: Path, *, phase: str = "cost") -> dict:
    """Measure real kernels, then optionally evaluate fresh finite-burn chains.

    A completed cost-only preflight never passes an energy/variance gate. A
    sampling audit uses whole-chain uncertainties and records zero support,
    imaginary energies, initial basins, and all work without clipping.
    """
    jax.config.update("jax_enable_x64", True)
    if phase not in ("cost", "audit", "importance", "root-cost", "overlap"):
        raise ValueError("phase must be cost, audit, importance, root-cost or overlap")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("immutable output: choose a fresh directory")
    cfg = MixedClebschConfig(**spec["model"])
    checkpoint = Path(spec["checkpoint"])
    if spec.get("checkpoint_sha256") and file_sha256(checkpoint) != spec["checkpoint_sha256"]:
        raise ValueError("fixed checkpoint SHA mismatch")
    saved = load_checkpoint(checkpoint)
    bank = saved["params"]
    if not isinstance(bank, dict) or set(bank) != {"local", "top"}:
        raise ValueError("checkpoint must contain a physical local/top CT bank")
    metadata = saved.get("metadata", {})
    active = spec.get("active_top_widths", metadata.get("active_top_widths",
        metadata.get("embedding", {}).get("active_top_widths")))
    if active is None and "embedding" in metadata and "active_upper_width" in metadata["embedding"]:
        width = metadata["embedding"]["active_upper_width"]
        active = [{q: (1 if i == len(bank["top"])-1 else width)
                   for q in ((0,) if i == len(bank["top"])-1 else range(0, cfg.maximum_q+1, 2))}
                  for i in range(len(bank["top"]))]
    if active is not None:
        active = checked_active_widths(active, nodes=len(bank["top"]))
        execution_bank = repack_active_clebsch_bank(bank, cfg, active)
    else:
        execution_bank = bank
    j2 = float(spec["j2_over_j1"])
    if not np.isfinite(j2):
        raise ValueError("finite Hamiltonian coupling required")
    chains, records = spec.get("chains", 8), spec.get("records", 4)
    burn, between = spec.get("burn_proposals", 256), spec.get("proposals_between", 32)
    # Validate counts before allocating models or compiling kernels.
    audit_cost_forecast(chains=chains, records=records, burn=burn, between=between,
                        amplitude_batch_seconds=1., H_batch_seconds=1.)
    if chains < 4 or records < 2:
        raise ValueError("independent development audit needs >=4 chains and >=2 records")
    seed = spec.get("seed", 261023271)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("fresh seed must be a uint32 integer")
    seconds_cap = float(spec.get("seconds_cap", 1800.))
    if not np.isfinite(seconds_cap) or not 10 <= seconds_cap <= 7200:
        raise ValueError("bounded wall cap must be 10..7200 seconds")
    if spec.get("require_gpu", False) and jax.default_backend() != "gpu":
        raise RuntimeError("registered GPU preflight cannot fall back to CPU")
    output.mkdir(parents=True, exist_ok=True)
    registration = dict(spec=spec, phase=phase,
        created_utc=datetime.now(timezone.utc).isoformat(), workflow_sha256=file_sha256(Path(__file__)),
        checkpoint_sha256=file_sha256(checkpoint), params_sha256=pytree_sha256(bank),
        no_optimization=True, fresh_chains=phase in ('audit', 'importance'),
        reused_pool_only=phase == 'overlap', fixed_endpoint=True)
    atomic_json(output/"registration.json", registration)
    started = perf_counter()
    result = dict(phase=phase, status="started", energy_estimate=False,
        timings={}, actual_full_H_centers=0, actual_amplitude_centers=0,
        params_sha256=registration["params_sha256"], backend=jax.default_backend(),
        complete_A1=True, stochastic_space_projection=False)

    def timed(name, fn, *args):
        if perf_counter()-started >= seconds_cap:
            raise TimeoutError("fixed wall budget reached before "+name)
        atomic_json(output/"progress.json", dict(result, operation=name, elapsed_seconds=perf_counter()-started))
        tick = perf_counter(); value = fn(*args); jax.block_until_ready(value)
        result["timings"][name] = perf_counter()-tick
        return value

    try:
        images = np.asarray(triangular_hnf_gamma_a1_group(((cfg.length, 0), (0, cfg.length))), np.int32)
        bonds = triangular_j1j2_bonds(cfg.length, cfg.length, j2=j2, periodic=True)
        amplitude = jax.jit(make_feature_lut_projected_amplitude(cfg, images, chunk=4, active_top_widths=active))
        amplitudes = jax.jit(jax.vmap(amplitude, in_axes=(None, 0)))
        action = make_cached_projected_heisenberg_action(cfg, images, bonds,
            projection_chunk=4, active_top_widths=active)
        prepared = prepare_feature_lut_bank(execution_bank, cfg)
        # Probe rows are NOT Born samples and never used in reported moments.
        probe = jnp.asarray(balanced_block_probe_states(cfg, count=chains, seed=seed+1))
        timed("A1_amplitudes_first", amplitudes, prepared, probe)
        active_values = timed("A1_amplitudes_warm", amplitudes, prepared, probe)
        result["actual_amplitude_centers"] += 2*chains
        if active is not None:
            native_query = jax.jit(make_feature_lut_projected_amplitude(cfg, images, chunk=4))
            native_prepared = prepare_feature_lut_bank(bank, cfg)
            native = np.asarray(timed("native_A1_replay_center", native_query, native_prepared, probe[0]))
            result["actual_amplitude_centers"] += 1
            replay = float(abs(native-np.asarray(active_values[0]))/max(float(abs(native)), 1e-300))
            if replay > 2e-10:
                raise ArithmeticError("active fixed-bank complete-A1 amplitude differs from native bank")
            result.update(active_upper_fixed_bank_execution=True,
                active_top_widths=active, native_A1_amplitude_relative_error=replay,
                inactive_channel_opening_gradients_preserved=False,
                native_upper_entries=sum(np.asarray(v).size for x in bank["top"] for v in x),
                active_upper_entries=sum(np.asarray(v).size for x in execution_bank["top"] for v in x))
        if phase == 'overlap':
            result.update(compare_saved_A1_amplitudes(spec, cfg, images, prepared,
                amplitudes, output, result, timed))
            result['total_seconds'] = perf_counter()-started
            atomic_json(output/'results.json', result)
            return result
        first = timed("full_H_first_center", action, prepared, probe[:1])
        result["actual_full_H_centers"] += 1
        second = timed("full_H_warm_center", action, prepared, probe[:1])
        result["actual_full_H_centers"] += 1
        np.testing.assert_allclose(first, second, rtol=2e-12, atol=1e-100)
        psi, hp = (np.asarray(x) for x in second)
        amp0 = np.asarray(amplitudes(prepared, probe[:1]))
        result["actual_amplitude_centers"] += 1
        rel = float(np.max(abs(psi-amp0))/max(float(np.max(abs(amp0))), 1e-300))
        if rel > 2e-10 or not np.isfinite(psi).all() or not np.isfinite(hp).all():
            raise ArithmeticError("full-H kernel center amplitude does not replay complete A1")
        np.savez_compressed(output/"cost_probe.npz", rows=np.asarray(probe), psi=psi, Hpsi=hp)
        result.update(group_order=len(images), complete_H_amplitude_relative_error=rel,
            fixed_probe_is_energy_audit=False, physical_wavefunction_norm_computed=False)
        # Single-center H timings scaled conservatively; measured batch timing
        # is not claimed until the audit actually runs those batches.
        forecast = audit_cost_forecast(chains=chains, records=records, burn=burn, between=between,
            amplitude_batch_seconds=result["timings"]["A1_amplitudes_warm"],
            H_batch_seconds=chains*result["timings"]["full_H_warm_center"])
        result["forecast"] = forecast
        if phase == "root-cost":
            coordinates, replace, root_columns = make_terminal_tensor_products(action, prepared)
            phi, hphi = timed("full_A1_root_columns_first", root_columns, prepared, probe[:1])
            warm = timed("full_A1_root_columns_warm", root_columns, prepared, probe[:1])
            for before, after in zip((phi, hphi), warm, strict=True):
                np.testing.assert_allclose(before, after, rtol=2e-12, atol=1e-100)
            rebuilt = np.asarray([phi @ coordinates, hphi @ coordinates])
            reference = np.asarray(second)
            relative = finite_relative_error(rebuilt, reference)
            if relative > 2e-10:
                raise ArithmeticError("root columns do not replay existing complete-A1/H action")
            changed = coordinates+.001*jnp.cos(jnp.arange(coordinates.size))
            changed_values = timed("full_H_root_linearity_probe", action,
                replace(prepared, changed), probe[:1])
            result["actual_full_H_centers"] += 1
            reconstruction = np.asarray([phi @ changed, hphi @ changed])
            direct = np.asarray(changed_values)
            linear_error = finite_relative_error(reconstruction, direct)
            if linear_error > 2e-10:
                raise ArithmeticError("existing terminal tensor is not a homogeneous linear physical space")
            np.savez_compressed(output/"root_linear_columns.npz", rows=np.asarray(probe[:1]),
                coefficients=np.asarray(coordinates), Phi=np.asarray(phi), HPhi=np.asarray(hphi),
                direct_psi=np.asarray(second[0]), direct_Hpsi=np.asarray(second[1]),
                probe_coefficients=np.asarray(changed))
            result.update(status="root_linear_cost_complete", energy_estimate=False,
                root_coordinates=int(coordinates.size),
                root_full_H_basis_column_center_products=2*int(coordinates.size),
                root_basis_center_batches=2, root_scalar_replay_relative_error=relative,
                root_probe_linearity_relative_error=linear_error,
                no_new_variational_head_or_width=True, optimizer_updates=0,
                root_basis_calls_are_not_scalar_full_H_calls=True,
                root_work_scope="two vector-basis center calls plus separately counted scalar full-H probes",
                finite_bit_probes_are_not_energy_or_physical_distance_audits=True)
        elif phase == "cost":
            result["status"] = "cost_complete"
        elif phase == "importance":
            result.update(independent_importance(bank, cfg, images, prepared, amplitudes,
                action, bonds, spec, output, result, timed, started, seconds_cap))
        elif forecast["conservative_seconds"] > seconds_cap-(perf_counter()-started):
            result.update(status="audit_not_started_due_to_measured_budget", energy_estimate=False)
        else:
            key, initial_key = jax.random.split(jax.random.PRNGKey(seed))
            states = initialize_spin_exchange_chains(initial_key, n_chains=chains,
                n_sites=cfg.length**2, n_up=cfg.length**2//2)
            # Half of independent initializations use a distinct balanced-block
            # basin. This is a fixed initial distribution, not sample repair.
            states = states.at[chains//2:].set(probe[chains//2:])
            sampler = make_record_sampler(amplitude, records=records, burn_proposals=burn,
                proposals_between=between, exchange_edges=tuple((a, b) for a, b, _ in bonds))
            last, key, rows, info = timed("independent_sampling", sampler, prepared, key, states)
            # run_exchange_sweeps refreshes one center per chain, then evaluates
            # one proposal per chain/attempt (including null/lazy proposals).
            sampler_queries = chains*(burn+records*between+1)
            result["sampler_amplitude_center_evaluations"] = sampler_queries
            result["actual_amplitude_centers"] += sampler_queries
            result["sampler_diagnostics"] = jax.tree.map(lambda x: np.asarray(x).tolist(), info)
            result["amplitude_count_scope"] = "standalone center evaluations, excluding queries internal to full-H kernels"
            if int(info["terminal_zero_centers"]) != 0:
                raise FloatingPointError("finite burn left exact zero-probability centers")
            values = []
            for index, batch in enumerate(rows):
                p, h = timed("audit_full_H_record_"+str(index), action, prepared, batch)
                result["actual_full_H_centers"] += chains
                if np.any(np.asarray(p) == 0):
                    raise FloatingPointError("Born audit contains an exact node")
                values.append(np.asarray(h/p))
                np.savez_compressed(output/"partial_audit.npz", rows=np.asarray(rows[:index+1]),
                                    local_energy=np.asarray(values))
            energy = np.asarray(values)
            moments = summarize_born_audit(energy, np.asarray(rows), n_sites=cfg.length**2)
            save_checkpoint(output/"audit_checkpoint.pkl", bank, optimizer_state=None,
                metadata=dict(task="independent_fixed_hierarchy_audit", rng=np.asarray(key),
                    sampler_states=np.asarray(last), optimization_performed=False,
                    source_checkpoint_sha256=registration["checkpoint_sha256"]))
            np.savez_compressed(output/"audit_samples.npz", rows=np.asarray(rows), local_energy=energy)
            result.update(status="development_audit_complete", energy_estimate=True, moments=moments,
                variance_node_complete=False, strict_SU2_is_model_property_not_sampling_test=True,
                initial_chain_basins=["uniform_fixed_Sz"]*(chains//2)+["balanced_blocks"]*(chains-chains//2),
                finite_burn_equilibrium_certified=False, publication_quality_gate_passed=False)
    except TimeoutError as error:
        result.update(status="bounded_incomplete", error=str(error), energy_estimate=False)
    except Exception as error:
        result.update(status="failed_validation", error=repr(error), energy_estimate=False)
        atomic_json(output/"results.json", dict(result, total_seconds=perf_counter()-started))
        raise
    result["total_seconds"] = perf_counter()-started
    atomic_json(output/"results.json", result)
    return result


def paired_endpoint_moments(psi: np.ndarray, hpsi: np.ndarray, density: np.ndarray,
                            groups: np.ndarray, *, n_sites: int) -> dict:
    """Full-H ratio moments on common, possibly finite-burn MH records.

    Arrays have shape(samples,2). The positive guiding density need not be
    normalized; its common unknown constant cancels. Groups are independent
    chains, NOT independent records. No IID error or absolute norm is claimed.
    Exact target nodes remain in the H-squared numerator.
    """
    p, hp = np.asarray(psi, complex), np.asarray(hpsi, complex)
    d, g = np.asarray(density, float), np.asarray(groups)
    if (p.ndim != 2 or p.shape[1] != 2 or hp.shape != p.shape
            or d.shape != (len(p),) or g.shape != d.shape
            or isinstance(n_sites,bool) or not isinstance(n_sites,Integral) or n_sites < 1
            or not np.isfinite(p).all() or not np.isfinite(hp).all()
            or not np.isfinite(d).all() or np.any(d <= 0)):
        raise ValueError('finite matching two-endpoint products and positive guide density required')
    labels, counts = np.unique(g, return_counts=True)
    if len(labels) < 4 or len(set(counts)) != 1:
        raise ValueError('at least four equal-size independent chain groups required')
    def moments(ids):
        norm = np.sum(abs(p[ids])**2/d[ids, None], axis=0)
        if np.any(norm <= 0):
            raise FloatingPointError('zero sampled endpoint norm')
        energy = np.sum(p[ids].conj()*hp[ids]/d[ids, None], axis=0)/norm
        variance = np.sum(abs(hp[ids]-energy*p[ids])**2/d[ids, None], axis=0)/norm
        return np.stack((energy.real/n_sites, energy.imag/n_sites, variance/n_sites), axis=1)
    ids = np.arange(len(p)); point = moments(ids)
    leave = np.asarray([moments(ids[g != label]) for label in labels])
    se = np.sqrt((len(labels)-1)/len(labels)*np.sum((leave-leave.mean(0))**2, axis=0))
    delta = point[1]-point[0]; delta_leave = leave[:, 1]-leave[:, 0]
    delta_se = np.sqrt((len(labels)-1)/len(labels)*np.sum((delta_leave-delta_leave.mean(0))**2, axis=0))
    energy_total = point[:, 0]*n_sites+1j*point[:, 1]*n_sites
    weight = abs(p)**2/d[:, None]
    residual = abs(hp-energy_total*p)**2/d[:, None]
    return dict(metric_order=['energy_per_site','imaginary_energy_per_site','variance_per_site'],
        values=point.tolist(), whole_chain_jackknife_SE=se.tolist(),
        candidate_minus_control=delta.tolist(), paired_whole_chain_jackknife_SE=delta_se.tolist(),
        normalization_effective_contributions=(weight.sum(0)**2/np.sum(weight**2,axis=0)).tolist(),
        maximum_residual_chain_fraction=[float(max(residual[g==label, i].sum() for label in labels)
            /max(float(residual[:, i].sum()), np.finfo(float).tiny)) for i in range(2)],
        exact_node_records=np.sum(p==0,axis=0).tolist(),
        sample_records=len(p), independent_chain_groups=len(labels),
        guide_normalization_known=False, finite_burn_equilibrium_certified=False,
        ratio_estimates_finite_sample_biased=True, variance_uses_actual_full_H=True)


def make_pair_support_guide(amplitude: Callable) -> Callable:
    """Return sqrt(|A_old|²+|A_new|²), without a physical-probability floor.

    ``amplitude(bank,bits)`` remains the same complete coherent A1 amplitude.
    Hypot avoids premature underflow of its square; it adds no support or
    normalization and does not sample an incoherent spatial-image mixture.
    """
    def guide(banks, bits):
        return jnp.hypot(jnp.abs(amplitude(banks[0], bits)),
                         jnp.abs(amplitude(banks[1], bits)))
    return guide


def support_coverage_diagnostics(psi: np.ndarray, *, old_floor: float,
        window_records: int = 16, maximum_split_rhat: float = 1.1,
        maximum_drift_standard_errors: float = 2.,
        minimum_effective_fraction: float = .25,
        maximum_chain_fraction: float = .35) -> dict:
    """Zero-H operational checks on ALL chains; never certify equilibrium.

    Physical endpoint amplitudes have shape(time,chain,2). The fixed trailing
    window is split into equal time halves, not selected by observed quality.
    The old uniform floor is only a diagnostic reference in this NO-floor
    target. Norm effective contributions are not autocorrelation-corrected ESS.
    """
    from .triangle_hierarchy_root_refinement import finite_chain_rhat
    p = np.asarray(psi)
    if (p.ndim != 3 or p.shape[2] != 2 or p.shape[1] < 4 or p.shape[0] < 1 or not np.isfinite(p).all()
            or isinstance(window_records, bool) or not isinstance(window_records, Integral)
            or window_records < 8 or window_records % 2 or not np.isfinite(old_floor)
            or old_floor <= 0):
        raise ValueError('finite [time,chain>=4,2], positive reference floor and even window>=8 required')
    thresholds = np.asarray([maximum_split_rhat, maximum_drift_standard_errors,
                             minimum_effective_fraction, maximum_chain_fraction], float)
    if (not np.isfinite(thresholds).all() or maximum_split_rhat < 1
            or maximum_drift_standard_errors <= 0 or not 0 < minimum_effective_fraction <= 1
            or not 0 < maximum_chain_fraction <= 1):
        raise ValueError('finite operational coverage thresholds required')
    magnitude = np.hypot(abs(p[..., 0]), abs(p[..., 1]))
    if np.any(magnitude <= 0):
        raise FloatingPointError('a support-chain record has an exact common endpoint node')
    log_q = 2*np.log(magnitude)
    reference_log_ratio = log_q-np.log(old_floor)
    output = dict(record_count=int(p.shape[0]*p.shape[1]), chains=int(p.shape[1]),
        endpoint_count=2, records_per_chain=int(p.shape[0]),
        final_log_Q0_over_old_floor=reference_log_ratio[-1].tolist(),
        final_mean_log_Q0_over_old_floor=float(reference_log_ratio[-1].mean()),
        all_terminal_chains_above_old_floor=bool(np.all(reference_log_ratio[-1] > 0)),
        no_energy_or_H_readout=True, all_chains_retained=True,
        finite_burn_equilibrium_certified=False,
        effective_contributions_are_not_autocorrelation_corrected_ESS=True,
        old_floor_is_not_used_in_sampling=True,
        gate_scope='Necessary operational support/stability checks only; not proof of mixing or tail coverage')
    if len(p) < window_records:
        output.update(status='insufficient_fixed_window', operational_gate_passed=False)
        return output
    selected = p[-window_records:]
    mag = magnitude[-window_records:]
    responsibilities = (abs(selected)/mag[..., None])**2
    values = [log_q[-window_records:], responsibilities[..., 0], responsibilities[..., 1]]
    notes = []
    for x in values:
        rhat = finite_chain_rhat(x)
        constant = bool(np.ptp(x) <= 1e-12)
        half = len(x)//2
        per_chain = x[half:].mean(0)-x[:half].mean(0)
        se = float(np.std(per_chain, ddof=1)/np.sqrt(len(per_chain)))
        delta = float(per_chain.mean())
        compatible = bool(abs(delta) <= maximum_drift_standard_errors*se+1e-12)
        notes.append(dict(rhat=rhat, constant_observable=constant,
            paired_half_window_drift=delta, whole_chain_drift_SE=se,
            drift_pass=compatible,
            rhat_pass=bool(constant or (rhat['split'] is not None
                                      and rhat['split'] <= maximum_split_rhat))))
    weights = []
    for k in range(2):
        w = responsibilities[..., k]
        total = float(w.sum())
        eff = float(total**2/np.sum(w*w)) if total > 0 else 0.
        fraction = float(w.sum(axis=0).max()/total) if total > 0 else 1.
        weights.append(dict(mean_responsibility=float(w.mean()), effective_contributions=eff,
            effective_fraction=eff/w.size, maximum_chain_fraction=fraction,
            pass_spread=bool(eff/w.size >= minimum_effective_fraction and fraction <= maximum_chain_fraction)))
    checks = dict(left_old_floor_platform=output['all_terminal_chains_above_old_floor'],
        log_Q0_between_chain_and_time_stability=bool(notes[0]['rhat_pass'] and notes[0]['drift_pass']),
        endpoint_responsibility_stability=bool(all(row['rhat_pass'] and row['drift_pass'] for row in notes[1:])),
        endpoint_norm_contribution_spread=bool(all(row['pass_spread'] for row in weights)))
    output.update(status='fixed_window_evaluated', fixed_window_records=int(window_records),
        diagnostics=dict(zip(('log_Q0','old_responsibility','new_responsibility'),notes,strict=True)),
        endpoint_weight_diagnostics=weights, checks=checks,
        operational_gate_passed=bool(all(checks.values())))
    return output


def support_work_limit(spec: dict) -> int:
    """Keep the legacy CPU limit; larger budgets require a registered GPU route."""
    profile = spec.get('execution_profile', 'legacy_bounded')
    if profile not in ('legacy_bounded', 'registered_single_gpu'):
        raise ValueError('unknown support execution_profile')
    limit = spec.get('maximum_registered_proposals', 512)
    ceiling = 16384 if profile == 'registered_single_gpu' else 2048
    if (isinstance(limit, bool) or not isinstance(limit, Integral) or not 1 <= limit <= ceiling):
        raise ValueError('support proposal limit exceeds the explicitly registered execution profile')
    return int(limit)


def support_initial_rng_key(spec: dict) -> jax.Array:
    """Continue a saved legacy uint32[2] key, or retain the old explicit-seed API."""
    if 'initial_rng_key' in spec:
        raw = np.asarray(spec['initial_rng_key'])
        if (raw.shape != (2,) or not np.issubdtype(raw.dtype, np.integer)
                or np.issubdtype(raw.dtype, np.bool_) or np.any(raw < 0)
                or np.any(raw > np.iinfo(np.uint32).max)):
            raise ValueError('initial_rng_key must be two exact uint32 integers')
        return jnp.asarray(raw, dtype=jnp.uint32)
    seed = spec['seed']
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError('an explicit uint32 seed or saved initial_rng_key is required')
    return jax.random.PRNGKey(int(seed))


def support_projection_chunk(spec: dict, *, image_count: int = 1728) -> int:
    """Validate coherent-image execution batching, not a projector truncation.

    Only the forward sampler amplitude uses this option. Every image keeps its
    original weight; H has a separate optional batch size. Defaults preserve
    legacy CPU execution and reduction order.
    """
    chunk = spec.get('sampler_projection_chunk', 4)
    if (isinstance(chunk, bool) or not isinstance(chunk, Integral)
            or chunk < 1 or image_count % chunk):
        raise ValueError('sampler_projection_chunk must be a positive integer dividing all spatial images')
    return int(chunk)


def bridge_projection_chunk(spec: dict, *, image_count: int = 1728) -> int:
    """Validate full-H coherent image batching, without dropping any H edge."""
    chunk = spec.get('H_projection_chunk', 4)
    if (isinstance(chunk, bool) or not isinstance(chunk, Integral)
            or chunk < 1 or image_count % chunk):
        raise ValueError('H_projection_chunk must be a positive integer dividing all spatial images')
    return int(chunk)


def support_runtime_identity(spec: dict, *, cpu_reference: bool = False) -> dict:
    """Validate the accelerator without allocating jobs or changing physics."""
    import jaxlib
    devices = jax.local_devices()
    if cpu_reference:
        if any(device.platform != 'cpu' for device in devices):
            raise ValueError('CPU short-reference mode must execute on CPU')
    elif spec.get('execution_profile') == 'registered_single_gpu':
        if len(devices) != 1 or devices[0].platform != 'gpu':
            raise ValueError('registered_single_gpu requires exactly one visible GPU')
    if not jax.config.jax_enable_x64:
        raise ValueError('the frozen support route requires float64/complex128')
    return dict(jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
        backend=jax.default_backend(), x64_enabled=True,
        devices=[dict(platform=d.platform, kind=d.device_kind, id=d.id) for d in devices],
        cpu_reference_only=cpu_reference)


def run_pair_support_cost(spec: dict, output: Path) -> dict:
    """Same-state/key short replay and actual128-proposal GPU throughput only.

    CPU reference mode stops after two repeats of a TWO-proposal trace. GPU
    mode additionally repeats the actual128-proposal segment from the SAME
    initial state/key. None of these discarded cost traces advance production
    chains. There are no Hamiltonian queries or support-release decisions.
    """
    jax.config.update('jax_enable_x64', True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('immutable support-cost output already exists')
    cfg = MixedClebschConfig(**spec['model'])
    sampler_chunk = support_projection_chunk(spec)
    cpu_reference = spec.get('cpu_reference_only', False)
    if not isinstance(cpu_reference, bool) or cfg.length != 12 or spec.get('chains', 8) != 8:
        raise ValueError('bounded L12/all8-chain cost protocol required')
    limit = support_work_limit(spec)
    segments = spec.get('segments', 4)
    burn, records, between = (spec.get('segment_burn', 8), spec.get('segment_records', 8),
                             spec.get('segment_between', 15))
    if (any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1
            for v in (segments, burn, records, between)) or burn+records*between != 128
            or segments*128 > limit):
        raise ValueError('cost probe requires the registered actual128-proposal segment')
    cap = float(spec.get('cost_seconds_cap', 180. if cpu_reference else 900.))
    if not np.isfinite(cap) or not 30 <= cap <= (180 if cpu_reference else 1800):
        raise ValueError('CPU reference cap<=180s; GPU cost cap<=1800s required')
    runtime = support_runtime_identity(spec, cpu_reference=cpu_reference)
    if not cpu_reference and spec.get('execution_profile') != 'registered_single_gpu':
        raise ValueError('full segment cost mode requires registered_single_gpu')
    paths = [Path(spec[name]) for name in ('control_checkpoint', 'candidate_checkpoint')]
    for name, path in zip(('control', 'candidate'), paths, strict=True):
        if file_sha256(path) != spec[name+'_sha256']:
            raise ValueError('frozen '+name+' checkpoint hash mismatch')
    samples = Path(spec['initial_samples_path'])
    if file_sha256(samples) != spec['initial_samples_sha256']:
        raise ValueError('frozen support starting sample hash mismatch')
    key = support_initial_rng_key(spec)
    with np.load(samples, allow_pickle=False) as saved:
        initial = saved[spec.get('initial_states_key', 'terminal_states')].copy()
        initial_psi = saved[spec.get('initial_psi_key', 'terminal_psi')].copy()
        if 'initial_rng_key' in spec and ('rng' not in saved or not np.array_equal(saved['rng'], key)):
            raise ValueError('explicit initial_rng_key does not match the saved starting RNG')
    if (initial.shape != (8, 144) or not np.issubdtype(initial.dtype, np.integer)
            or not np.isin(initial, (0, 1)).all() or not np.all(initial.sum(1) == 72)
            or initial_psi.shape != (8, 2) or not np.isfinite(initial_psi).all()):
        raise ValueError('all8 physical source states and endpoint amplitudes required')
    reference = None
    if not cpu_reference:
        reference_path = Path(spec['cpu_reference_path'])
        if file_sha256(reference_path) != spec['cpu_reference_sha256']:
            raise ValueError('frozen CPU short-reference hash mismatch')
        with np.load(reference_path, allow_pickle=False) as saved:
            reference = {k: saved[k].copy() for k in saved.files}
        if (not np.array_equal(reference['initial_states'], initial)
                or not np.array_equal(reference['initial_rng_key'], key)):
            raise ValueError('CPU reference must use the same source states and RNG')
    active = checked_active_widths(spec['active_top_widths'], nodes=15)
    banks = [load_checkpoint(p)['params'] for p in paths]
    execution = [repack_active_clebsch_bank(b, cfg, active) for b in banks]
    output.mkdir(parents=True, exist_ok=True)
    registration = dict(spec=spec, workflow_sha256=file_sha256(Path(__file__)), runtime=runtime,
        params_sha256=[pytree_sha256(b) for b in banks], actual_H_queries=0,
        no_optimization=True, all_cost_traces_discarded=True, initial_rng_key=np.asarray(key).tolist(),
        identical_state_key_repeats=True, complete_A1=True, projection_chunk=sampler_chunk,
        sampler_projection_chunk=sampler_chunk, batching_changes_coherent_sum_order_only=True)
    atomic_json(output/'registration.json', registration)
    (output/'registered_workflow.py').write_bytes(Path(__file__).read_bytes())
    started = perf_counter()
    result = dict(status='started', actual_full_H_centers=0, energy_estimate=False,
        completed_amplitude_centers=0, timings={}, runtime=runtime, production_chain_advanced=False,
        sampler_projection_chunk=sampler_chunk)
    def timed(label, call, count):
        if perf_counter()-started >= cap:
            raise TimeoutError('support-cost wall cap before '+label)
        atomic_json(output/'progress.json', dict(result, operation=label, elapsed_seconds=perf_counter()-started))
        tick = perf_counter(); value = call(); jax.block_until_ready(value)
        result['timings'][label] = perf_counter()-tick
        result['completed_amplitude_centers'] += count
        return value
    try:
        images = np.asarray(triangular_hnf_gamma_a1_group(((12, 0), (0, 12))), np.int32)
        if len(images) != 1728:
            raise ArithmeticError('complete spatial group size changed')
        amplitude = jax.jit(make_feature_lut_projected_amplitude(cfg, images, chunk=sampler_chunk, active_top_widths=active))
        guide = make_pair_support_guide(amplitude)
        endpoints = jax.jit(jax.vmap(lambda data, x: jnp.stack(
            (amplitude(data[0], x), amplitude(data[1], x))), in_axes=(None, 0)))
        prepared = tuple(prepare_feature_lut_bank(b, cfg) for b in execution)
        states = jnp.asarray(initial)
        bonds = triangular_j1j2_bonds(12, 12, j2=float(spec['j2_over_j1']), periodic=True)
        options = dict(exchange_edges=tuple((i, j) for i, j, _ in bonds),
                       bond_proposal_probability=.75, lazy_probability=.05)
        observed = np.asarray(timed('endpoints_first', lambda: endpoints(prepared, states), 16))
        warm = np.asarray(timed('endpoints_warm', lambda: endpoints(prepared, states), 16))
        result['source_amplitude_relative_error'] = finite_relative_error(observed, initial_psi)
        if result['source_amplitude_relative_error'] > 2e-10 or finite_relative_error(warm, observed) > 2e-12:
            raise ArithmeticError('saved physical endpoint amplitudes failed replay')
        short = make_record_sampler(guide, records=1, burn_proposals=1, proposals_between=1, **options)
        first = timed('short_first', lambda: short(prepared, key, states), 48)
        replay = timed('short_warm', lambda: short(prepared, key, states), 48)
        for a, b in zip(first[:3], replay[:3], strict=True):
            if not np.array_equal(a, b):
                raise ArithmeticError('same-device fixed-key short trace did not replay exactly')
        if reference is not None:
            for field, actual in (('short_last', first[0]), ('short_rng', first[1]), ('short_rows', first[2])):
                if not np.array_equal(reference[field], actual):
                    raise ArithmeticError('CPU/GPU short trace differs: '+field+'; inspect acceptance roundoff before continuation')
            result['CPU_GPU_short_states_and_rng_exact'] = True
            result['CPU_GPU_endpoint_relative_error'] = finite_relative_error(observed, reference['endpoint_psi'])
            if result['CPU_GPU_endpoint_relative_error'] > 2e-10:
                raise ArithmeticError('CPU/GPU physical endpoint amplitudes differ')
        np.savez_compressed(output/'short_reference.npz', initial_states=initial,
            initial_rng_key=np.asarray(key), endpoint_psi=observed,
            short_last=np.asarray(first[0]), short_rng=np.asarray(first[1]), short_rows=np.asarray(first[2]))
        if cpu_reference:
            result['status'] = 'CPU_short_reference_complete_no_long_sampling'
        else:
            sampler = make_record_sampler(guide, records=records, burn_proposals=burn,
                proposals_between=between, **options)
            first = timed('segment_first', lambda: sampler(prepared, key, states), 16*129)
            replay = timed('segment_warm', lambda: sampler(prepared, key, states), 16*129)
            for a, b in zip(first[:3], replay[:3], strict=True):
                if not np.array_equal(a, b):
                    raise ArithmeticError('same-device full-segment states/RNG did not replay')
            tail = np.asarray(timed('terminal_endpoints_warm', lambda: endpoints(prepared, replay[0]), 16))
            np.savez_compressed(output/'segment_cost_trace.npz', rows=np.asarray(replay[2]),
                terminal_states=np.asarray(replay[0]), terminal_psi=tail, rng=np.asarray(replay[1]),
                initial_states=initial, initial_rng_key=np.asarray(key))
            forecast = 1.5*segments*(result['timings']['segment_warm']
                + records*max(result['timings']['endpoints_warm'], result['timings']['terminal_endpoints_warm']))+30.
            result.update(status='GPU_support_cost_complete', proposals_per_segment=128,
                warm_seconds_per_8chain_proposal=result['timings']['segment_warm']/128,
                registered_segment_count=int(segments), support_remaining_forecast_seconds=forecast,
                forecast_is_not_equilibrium_certificate=True,
                memory_stats=jax.local_devices()[0].memory_stats())
    except TimeoutError as error:
        result.update(status='bounded_incomplete', error=str(error))
    except Exception as error:
        result.update(status='failed_validation', error=repr(error))
        atomic_json(output/'results.json', dict(result, total_seconds=perf_counter()-started)); raise
    result['total_seconds'] = perf_counter()-started
    atomic_json(output/'results.json', result)
    return result


def run_pair_support_stage(spec: dict, output: Path) -> dict:
    """Bounded zero-H continuation of all frozen physical support chains.

    Every segment and the final endpoint are fixed before the run. Neither
    energy nor an unusually large intermediate probability selects records.
    Prefixes survive a time bound, but only the full fixed endpoint can pass
    the operational gate. H-bridge evaluation belongs to a separate phase.
    """
    started=perf_counter()
    jax.config.update('jax_enable_x64', True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('immutable support stage output already exists')
    cfg = MixedClebschConfig(**spec['model'])
    sampler_chunk = support_projection_chunk(spec)
    chains, segments = spec.get('chains',8), spec.get('segments',4)
    burn, records, between = (spec.get('segment_burn',8), spec.get('segment_records',8),
                               spec.get('segment_between',15))
    counts = (chains, segments, burn, records, between)
    maximum_proposals=support_work_limit(spec)
    if (cfg.length != 12 or any(isinstance(v,bool) or not isinstance(v,Integral) or v<1 for v in counts)
            or chains != 8 or records < 4
            or segments*(burn+records*between) > maximum_proposals):
        raise ValueError('bounded L12/all8-chain support stage within explicit registered limit required')
    cap = float(spec.get('seconds_cap',1200.))
    if not np.isfinite(cap) or not 60 <= cap <= 5400:
        raise ValueError('support stage wall bound must be60..5400 seconds')
    paths = [Path(spec[name]) for name in ('control_checkpoint','candidate_checkpoint')]
    for name,path in zip(('control','candidate'),paths,strict=True):
        if file_sha256(path) != spec[name+'_sha256']:
            raise ValueError('frozen '+name+' checkpoint hash mismatch')
    samples_path = Path(spec['initial_samples_path'])
    if file_sha256(samples_path) != spec['initial_samples_sha256']:
        raise ValueError('frozen support starting sample hash mismatch')
    active = checked_active_widths(spec['active_top_widths'],nodes=15)
    initial_key = support_initial_rng_key(spec)
    with np.load(samples_path) as saved:
        initial = np.asarray(saved[spec.get('initial_states_key','terminal_states')]).copy()
        initial_psi = np.asarray(saved[spec.get('initial_psi_key','terminal_psi')]).copy()
        if 'initial_rng_key' in spec and ('rng' not in saved or not np.array_equal(saved['rng'], initial_key)):
            raise ValueError('explicit initial_rng_key does not match saved starting RNG')
    if (initial.shape != (chains,144) or not np.issubdtype(initial.dtype,np.integer)
            or not np.isin(initial,(0,1)).all() or np.any(initial.sum(axis=1) != 72)
            or initial_psi.shape != (chains,2) or not np.isfinite(initial_psi).all()):
        raise ValueError('all8 matching frozen fixed-Sz states and endpoint amplitudes required')
    gate_spec = dict(spec.get('coverage_gate',{}))
    # Validate coverage thresholds before spending any numerical work.
    support_coverage_diagnostics(initial_psi[None,...],old_floor=1.,**gate_spec)
    runtime = support_runtime_identity(spec)
    banks = [load_checkpoint(path)['params'] for path in paths]
    execution = [repack_active_clebsch_bank(b,cfg,active) for b in banks]
    output.mkdir(parents=True,exist_ok=True)
    registration = dict(spec=spec,workflow_sha256=file_sha256(Path(__file__)),
        params_sha256=[pytree_sha256(b) for b in banks], complete_A1=True, runtime=runtime,
        sampler_projection_chunk=sampler_chunk, batching_changes_coherent_sum_order_only=True,
        initial_rng_key=np.asarray(initial_key).tolist(), continuing_saved_rng='initial_rng_key' in spec,
        target='Q0=abs(P_A1 old)**2+abs(P_A1 new)**2; NO uniform floor',
        actual_full_H_centers=0,no_optimization=True,all8_chains_retained=True,
        fixed_endpoint_no_probability_or_energy_selection=True,
        time_bound_checked_at_short_python_kernel_boundaries=True,
        support_records_are_development_not_untouched_energy_audit=True)
    # Freeze the exact executed workflow before numerical work. This artifact
    # complements input hashes rather than relying on a future working tree.
    (output/'registered_workflow.py').write_bytes(Path(__file__).read_bytes())
    if file_sha256(output/'registered_workflow.py') != registration['workflow_sha256']:
        raise ArithmeticError('support workflow changed during registration')
    atomic_json(output/'registration.json',registration)
    result=dict(status='started',actual_full_H_centers=0,energy_estimate=False,
        timings={},completed_amplitude_centers=0,segments=[],fixed_endpoint_reached=False,
        finite_burn_equilibrium_certified=False,operational_gate_passed=False,
        sampler_projection_chunk=sampler_chunk)
    def timed(name,call,centers):
        if perf_counter()-started >= cap:
            raise TimeoutError('support stage wall cap before '+name)
        atomic_json(output/'progress.json',dict(result,operation=name,
            elapsed_seconds=perf_counter()-started,in_flight_amplitude_centers_upper_bound=centers))
        tick=perf_counter(); value=call(); jax.block_until_ready(value)
        result['timings'][name]=perf_counter()-tick
        result['completed_amplitude_centers']+=centers
        return value
    try:
        from math import lgamma
        images=np.asarray(triangular_hnf_gamma_a1_group(((12,0),(0,12))),np.int32)
        if len(images)!=1728:raise ArithmeticError('complete spatial group size changed')
        old_floor=(.05/len(images))*np.exp(2*lgamma(73)-lgamma(145))
        amplitude=jax.jit(make_feature_lut_projected_amplitude(cfg,images,chunk=sampler_chunk,active_top_widths=active))
        guide=make_pair_support_guide(amplitude)
        def endpoints(data,bits):
            return jnp.stack((amplitude(data[0],bits),amplitude(data[1],bits)))
        endpoint_batch=jax.jit(jax.vmap(endpoints,in_axes=(None,0)))
        prepared=tuple(prepare_feature_lut_bank(b,cfg) for b in execution)
        bonds=triangular_j1j2_bonds(12,12,j2=float(spec['j2_over_j1']),periodic=True)
        options=dict(exchange_edges=tuple((i,j) for i,j,_ in bonds),
                     bond_proposal_probability=.75,lazy_probability=.05)
        sampler=make_record_sampler(guide,records=records,burn_proposals=burn,
                                    proposals_between=between,**options)
        short=make_record_sampler(guide,records=1,burn_proposals=1,proposals_between=1,**options)
        key=initial_key; states=jnp.asarray(initial)
        replay=np.asarray(timed('initial_endpoints_first',lambda:endpoint_batch(prepared,states),2*chains))
        warm=np.asarray(timed('initial_endpoints_warm',lambda:endpoint_batch(prepared,states),2*chains))
        if finite_relative_error(replay,initial_psi)>2e-10 or finite_relative_error(warm,replay)>2e-12:
            raise ArithmeticError('frozen support-start amplitudes failed numerical replay')
        for label,offset in (('short_first',1),('short_warm',2)):
            cost_key = (jax.random.fold_in(initial_key, offset) if 'initial_rng_key' in spec
                        else jax.random.PRNGKey(int(spec['seed'])+offset))
            timed(label,lambda cost_key=cost_key:short(prepared,cost_key,states),6*chains)
        proposals=burn+records*between
        per_segment=(proposals+1)/3*result['timings']['short_warm']+records*result['timings']['initial_endpoints_warm']
        result.update(old_floor_reference_only=old_floor,proposals_per_segment=proposals,
            preflight_forecast_seconds=1.5*segments*per_segment+30.,
            initial_mean_log_Q0_over_old_floor=float(np.mean(2*np.log(np.hypot(abs(replay[:,0]),abs(replay[:,1])))-np.log(old_floor))))
        if result['preflight_forecast_seconds']>cap-(perf_counter()-started):
            result['status']='cost_only_fixed_endpoint_not_affordable'
        else:
            all_rows=[]; all_psi=[]
            for index in range(segments):
                remaining=cap-(perf_counter()-started)
                measured=max(per_segment,max([r['segment_seconds'] for r in result['segments']],default=0.))
                if 1.2*measured>remaining:
                    result['status']='bounded_prefix_no_next_complete_segment';break
                tick=perf_counter()
                last,next_key,rows,info=timed('sampling_segment_'+str(index),
                    lambda:sampler(prepared,key,states),2*chains*(proposals+1))
                columns=[]
                for t,batch in enumerate(rows):
                    columns.append(np.asarray(timed('endpoints_segment_'+str(index)+'_'+str(t),
                        lambda batch=batch:endpoint_batch(prepared,batch),2*chains)))
                rows_np=np.asarray(rows); psi_np=np.asarray(columns)
                all_rows.extend(rows_np);all_psi.extend(psi_np)
                states,key=last,next_key
                diagnostics=support_coverage_diagnostics(np.asarray(all_psi),old_floor=old_floor,**gate_spec)
                result['segments'].append(dict(segment=index+1,cumulative_proposals=(index+1)*proposals,
                    segment_seconds=perf_counter()-tick,coverage=diagnostics,
                    sampler_diagnostics=jax.tree.map(lambda x:np.asarray(x).tolist(),info)))
                # Always persist the fixed entire prefix, never select best rows.
                np.savez_compressed(output/'support_prefix.npz',rows=np.asarray(all_rows),
                    psi=np.asarray(all_psi),terminal_states=np.asarray(states),terminal_psi=psi_np[-1],
                    rng=np.asarray(key),initial_states=initial,initial_psi=initial_psi,
                    completed_segments=index+1)
                atomic_json(output/'progress.json',dict(result,operation='segment_complete',
                    elapsed_seconds=perf_counter()-started))
            if len(result['segments'])==segments:
                result.update(status='fixed_endpoint_support_stage_complete',fixed_endpoint_reached=True,
                    coverage=result['segments'][-1]['coverage'],
                    operational_gate_passed=result['segments'][-1]['coverage']['operational_gate_passed'])
    except TimeoutError as error:
        result.update(status='bounded_incomplete',error=str(error))
    except Exception as error:
        result.update(status='failed_validation',error=repr(error))
        atomic_json(output/'results.json',dict(result,total_seconds=perf_counter()-started));raise
    result.update(total_seconds=perf_counter()-started,
        logical_raw_image_amplitude_count=1728*result['completed_amplitude_centers'])
    atomic_json(output/'results.json',result)
    return result


def run_paired_endpoint_audit(spec: dict, output: Path) -> dict:
    """Bounded fresh common-guide audit; no state fitting or endpoint selection.

    Endpoints must have the SAME declared active layout (zero-padding an old
    state is permitted after exact closure validation). Guide density is the
    sum of actual complete-A1 endpoint probabilities plus a positive uniform
    fixed-Sz floor. It is not a coherent wavefunction or an SU(2) projection.
    """
    from .triangle_hierarchy_root_refinement import finite_chain_rhat
    jax.config.update('jax_enable_x64', True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('immutable paired audit output already exists')
    cfg = MixedClebschConfig(**spec['model'])
    if cfg.length != 12:
        raise ValueError('this bounded transfer audit is registered for L=12')
    chains, records = spec.get('chains',8), spec.get('records',8)
    burn, between = spec.get('burn_proposals',128), spec.get('proposals_between',8)
    if any(isinstance(x,bool) or not isinstance(x,Integral) or x<4 for x in (chains,records,burn,between)):
        raise ValueError('integer chain/record/proposal counts >=4 required')
    cap = float(spec.get('seconds_cap',1800.))
    if not 60<=cap<=7200 or not np.isfinite(cap):
        raise ValueError('bounded paired wall cap must be 60..7200 seconds')
    paths = [Path(spec[name]) for name in ('control_checkpoint','candidate_checkpoint')]
    for name, path in zip(('control','candidate'),paths,strict=True):
        if file_sha256(path) != spec[name+'_sha256']:
            raise ValueError('frozen '+name+' checkpoint hash mismatch')
    active = checked_active_widths(spec['active_top_widths'],nodes=15)
    banks = [load_checkpoint(path)['params'] for path in paths]
    execution = [repack_active_clebsch_bank(b,cfg,active) for b in banks]
    output.mkdir(parents=True,exist_ok=True)
    registration = dict(spec=spec,workflow_sha256=file_sha256(Path(__file__)),
        params_sha256=[pytree_sha256(b) for b in banks],no_optimization=True,
        complete_A1=True,guide='abs(P_control)**2+abs(P_candidate)**2+(.05/G)*uniform_fixed_Sz',
        fresh_chains=True,endpoint_selection_after_audit=False,
        full_H_microbatch=1,no_unmeasured_larger_H_batch_compile=True,
        guide_density_reused_from_actual_H_center_amplitudes=True,
        time_bound_checked_at_python_kernel_boundaries=True,
        finite_burn_is_not_IID_or_equilibrium_certification=True)
    atomic_json(output/'registration.json',registration)
    started=perf_counter(); result=dict(status='started',energy_estimate=False,timings={},
        full_H_centers=0,amplitude_centers=0,complete_A1=True,publication_quality_gate_passed=False)
    def timed(name, fn, *args):
        if perf_counter()-started>=cap:
            raise TimeoutError('paired complete-A1 wall cap reached before '+name)
        atomic_json(output/'progress.json',dict(result,operation=name,elapsed_seconds=perf_counter()-started))
        tick=perf_counter(); answer=fn(*args);jax.block_until_ready(answer)
        result['timings'][name]=perf_counter()-tick
        return answer
    try:
        from math import lgamma
        images=np.asarray(triangular_hnf_gamma_a1_group(((12,0),(0,12))),np.int32)
        bonds=triangular_j1j2_bonds(12,12,j2=float(spec['j2_over_j1']),periodic=True)
        amplitude=jax.jit(make_feature_lut_projected_amplitude(cfg,images,chunk=4,active_top_widths=active))
        amp_batch=jax.jit(jax.vmap(amplitude,in_axes=(None,0)))
        action=make_cached_projected_heisenberg_action(cfg,images,bonds,projection_chunk=4,active_top_widths=active)
        prepared=tuple(prepare_feature_lut_bank(b,cfg) for b in execution)
        log_uniform=2*lgamma(73)-lgamma(145)
        floor=(.05/len(images))*np.exp(log_uniform)
        def guide(banks,bits):
            a=amplitude(banks[0],bits);b=amplitude(banks[1],bits)
            return jnp.sqrt(abs(a)**2+abs(b)**2+floor)
        guide_batch=jax.jit(jax.vmap(guide,in_axes=(None,0)))
        seed=int(spec.get('seed',2610237100));key=jax.random.PRNGKey(seed)
        key,draw=jax.random.split(key)
        states=initialize_spin_exchange_chains(draw,n_chains=chains,n_sites=144,n_up=72)
        if chains>=8:
            states=states.at[chains//2:].set(jnp.asarray(balanced_block_probe_states(cfg,count=chains-chains//2,seed=seed+1)))
        timed('guide_first',guide_batch,prepared,states)
        timed('guide_warm',guide_batch,prepared,states)
        result['amplitude_centers']+=4*chains
        probe=states[:1]
        for k in range(2):
            first=timed('H_first_'+str(k),action,prepared[k],probe);result['full_H_centers']+=1
            second=timed('H_warm_'+str(k),action,prepared[k],probe);result['full_H_centers']+=1
            np.testing.assert_allclose(first,second,rtol=2e-12,atol=1e-100)
            check=timed('psi_replay_'+str(k),amp_batch,prepared[k],probe);result['amplitude_centers']+=1
            if finite_relative_error(np.asarray(second[0]),np.asarray(check))>2e-10:
                raise ArithmeticError('complete H center amplitude does not replay')
        warm_H=sum(result['timings']['H_warm_'+str(k)] for k in range(2))
        forecast=1.5*((burn+records*between+1)*result['timings']['guide_warm']+chains*records*warm_H)
        result['remaining_cost_forecast_seconds']=forecast
        if forecast>cap-(perf_counter()-started):
            result['status']='cost_complete_audit_not_started_due_to_measured_budget'
        else:
            sampler=make_record_sampler(guide,records=records,burn_proposals=burn,
                proposals_between=between,exchange_edges=tuple((a,b) for a,b,_ in bonds))
            last,key,rows,info=timed('fresh_common_guide_sampling',sampler,prepared,key,states)
            result['amplitude_centers']+=2*chains*(burn+records*between+1)
            products=[]
            for t,batch in enumerate(rows):
                # Exactly the measured shape, no hidden batch8 H compilation.
                endpoint_products=[[],[]]
                for chain in range(chains):
                    for k in range(2):
                        p,h=timed('audit_H_'+str(t)+'_'+str(chain)+'_'+str(k),
                                  action,prepared[k],batch[chain:chain+1])
                        result['full_H_centers']+=1
                        endpoint_products[k].append((np.asarray(p)[0],np.asarray(h)[0]))
                columns=[(np.asarray([v[0] for v in values]),np.asarray([v[1] for v in values]))
                         for values in endpoint_products]
                products.append(columns)
                np.savez_compressed(output/'partial_products.npz',rows=np.asarray(rows[:t+1]),
                    psi=np.asarray([[c[0] for c in row] for row in products]).transpose(0,2,1),
                    Hpsi=np.asarray([[c[1] for c in row] for row in products]).transpose(0,2,1))
            p=np.asarray([[c[0] for c in row] for row in products]).transpose(0,2,1).reshape(-1,2)
            hp=np.asarray([[c[1] for c in row] for row in products]).transpose(0,2,1).reshape(-1,2)
            # Algebraically the same Q as the sampling guide; no redundant
            # large-shape amplitude kernel or loss of exact-node support.
            density=np.sum(abs(p)**2,axis=1)+floor
            groups=np.tile(np.arange(chains),records)
            moments=paired_endpoint_moments(p,hp,density,groups,n_sites=144)
            norm=abs(p)**2/density[:,None]
            result.update(status='paired_finite_burn_audit_complete',energy_estimate=True,moments=moments,
                norm_chain_diagnostics=[finite_chain_rhat(norm[:,k].reshape(records,chains)) for k in range(2)],
                guide_chain_diagnostics=finite_chain_rhat(np.log(density).reshape(records,chains)),
                sampler_diagnostics=jax.tree.map(lambda x:np.asarray(x).tolist(),info))
            np.savez_compressed(output/'untouched_products.npz',rows=np.asarray(rows),psi=p,Hpsi=hp,
                guide_density=density,groups=groups,last_states=np.asarray(last),rng=np.asarray(key))
    except TimeoutError as error:
        result.update(status='bounded_incomplete',error=str(error),energy_estimate=False)
    except Exception as error:
        result.update(status='failed_validation',error=repr(error),energy_estimate=False)
        atomic_json(output/'results.json',dict(result,total_seconds=perf_counter()-started))
        raise
    result['total_seconds']=perf_counter()-started
    atomic_json(output/'results.json',result)
    return result


def run_h_bridge_endpoint_audit(spec: dict, output: Path) -> dict:
    """Fresh paired records on Q0, then a fixed exchange bridge for H^2.

    Q0 is the sum of the two frozen complete-A1 probabilities, with no floor.
    Quality audits require an independently registered support stage whose
    necessary checks passed. An EXPLICIT separately registered coverage-only
    diagnostic may inspect an already completed failed stage, with fixed work
    and no benchmark/attention release. It never changes the original gate.
    Neither mode proves global equilibrium. Pushing each base record preserves its chain ID and
    unknown guide mass. Products and Qmix are recomputed at the pushed record,
    including target nodes; whole-chain errors are used, never IID errors.
    """
    from .triangle_hierarchy_root_refinement import finite_chain_rhat
    from su2_nqs.vmc.exchange_bridge_importance import make_exchange_record_bridge
    from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_bridge

    jax.config.update('jax_enable_x64', True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('immutable bridge audit output already exists')
    cfg = MixedClebschConfig(**spec['model'])
    sampler_chunk = support_projection_chunk(spec)
    h_chunk = bridge_projection_chunk(spec)
    chains, records = spec['chains'], spec['records']
    burn, between = spec['burn_proposals'], spec['proposals_between']
    if (cfg.length != 12 or chains != 8 or any(isinstance(x, bool) or not isinstance(x, Integral)
            or x < 4 for x in (records, burn, between))):
        raise ValueError('bounded L12/all-eight-chain bridge protocol required')
    cap = float(spec['seconds_cap'])
    if not np.isfinite(cap) or not 60 <= cap <= 7200:
        raise ValueError('kernel-boundary wall cap must be60..7200 seconds')
    purpose = spec.get('audit_purpose', 'independent_quality_audit')
    if purpose not in ('independent_quality_audit', 'coverage_diagnostic'):
        raise ValueError('unknown bridge audit_purpose')
    coverage_only = purpose == 'coverage_diagnostic'
    if coverage_only and (any(spec.get(name) is not True for name in
            ('separately_registered_diagnostic', 'no_benchmark', 'no_attention'))
            or (records, burn, between) != (32, 128, 128) or cap > 3600):
        raise ValueError('coverage diagnostic requires explicit separate registration, no benchmark/attention, '
                         '32 records,128 burn,128 interval and cap<=3600 seconds')
    if coverage_only:
        seed = spec.get('seed')
        if (isinstance(seed, bool) or not isinstance(seed, Integral)
                or not 0 <= seed <= np.iinfo(np.uint32).max):
            raise ValueError('coverage diagnostic requires an explicit fresh uint32 seed')
    paths = [Path(spec[name]) for name in ('control_checkpoint', 'candidate_checkpoint')]
    for name, path in zip(('control', 'candidate'), paths, strict=True):
        if file_sha256(path) != spec[name+'_sha256']:
            raise ValueError('frozen '+name+' checkpoint hash mismatch')
    support_path, diagnostic_path = map(Path, (spec['support_samples_path'], spec['support_results_path']))
    for name, path in (('support_samples', support_path), ('support_results', diagnostic_path)):
        if file_sha256(path) != spec[name+'_sha256']:
            raise ValueError('frozen support input hash mismatch')
    support_registration = Path(spec['support_registration_path'])
    if file_sha256(support_registration) != spec['support_registration_sha256']:
        raise ValueError('frozen support registration hash mismatch')
    parent_spec = json.loads(support_registration.read_text())['spec']
    for field in ('model', 'j2_over_j1', 'control_sha256', 'candidate_sha256', 'active_top_widths'):
        if parent_spec[field] != spec[field]:
            raise ValueError('support stage and H audit must refer to the same endpoints/layout/H')
    support_result = json.loads(diagnostic_path.read_text())
    support_passed = support_result.get('coverage', {}).get('operational_gate_passed', False) is True
    if (support_result.get('fixed_endpoint_reached', False) is not True
            or (not support_passed and not coverage_only)):
        raise ValueError('registered all-chain support checks must pass before H release')
    start = np.load(support_path)
    states = jnp.asarray(start['terminal_states'])
    if (states.shape != (chains, 144) or not np.issubdtype(np.asarray(states).dtype, np.integer)
            or not np.all((np.asarray(states) == 0) | (np.asarray(states) == 1))
            or not np.all(np.sum(np.asarray(states), axis=1) == 72)):
        raise ValueError('all eight fixed-Sz support endpoints required')
    terminal_psi = np.asarray(start['terminal_psi'])
    if terminal_psi.shape != (chains, 2) or not np.isfinite(terminal_psi).all():
        raise ValueError('all eight finite frozen endpoint amplitude pairs required')
    active = checked_active_widths(spec['active_top_widths'], nodes=15)
    runtime = support_runtime_identity(spec)
    banks = [load_checkpoint(path)['params'] for path in paths]
    execution = [repack_active_clebsch_bank(b, cfg, active) for b in banks]
    output.mkdir(parents=True, exist_ok=True)
    registration = dict(spec=spec, workflow_sha256=file_sha256(Path(__file__)),
        params_sha256=[pytree_sha256(b) for b in banks], no_optimization=True,
        audit_purpose=purpose, runtime=runtime,
        support_stage_operational_gate_passed=support_passed,
        diagnostic_only=coverage_only, benchmark_eligible=False, attention_release=False,
        original_support_gate_unchanged=True,
        complete_A1=True, guide='Q0=sum_k abs(Ppsi_k)**2; Qmix=(Q0+KQ0)/2',
        sampler_projection_chunk=sampler_chunk, H_projection_chunk=h_chunk,
        batching_changes_coherent_sum_order_only=True,
        fixed_bond_probabilities='abs(J_b)/sum_b abs(J_b)',
        floor=0., fresh_audit_records=True, all_support_chains_retained=True,
        endpoint_selection_after_audit=False, full_H_microbatch=1,
        density_evaluated_at_pushed_record=True, exact_target_nodes_retained=True,
        chain_ID_preserved_after_push=True, time_bound_at_kernel_boundaries=True,
        finite_burn_not_IID_or_equilibrium_certification=True)
    atomic_json(output/'registration.json', registration)
    (output/'registered_workflow.py').write_bytes(Path(__file__).read_bytes())
    started = perf_counter()
    result = dict(status='started', energy_estimate=False, full_H_centers=0,
        amplitude_centers=0, timings={}, complete_A1=True,
        support_stage_operational_gate_passed=support_passed, publication_quality_gate_passed=False,
        audit_purpose=purpose, diagnostic_only=coverage_only, runtime=runtime,
        benchmark_eligible=False, attention_release=False, original_support_gate_unchanged=True,
        energy_estimate_scope=('finite_chain_coverage_diagnostic_not_reference_energy' if coverage_only
                               else 'fresh_chain_ratios_subject_to_equilibration_and_tail_checks'),
        chain_jackknife_covers_unknown_equilibration_bias=False,
        sampler_projection_chunk=sampler_chunk, H_projection_chunk=h_chunk)

    def timed(name, fn, *args):
        if perf_counter()-started >= cap:
            raise TimeoutError('bridge wall cap before '+name)
        atomic_json(output/'progress.json', dict(result, operation=name, elapsed_seconds=perf_counter()-started))
        tick = perf_counter(); answer = fn(*args); jax.block_until_ready(answer)
        result['timings'][name] = perf_counter()-tick
        return answer

    try:
        images = np.asarray(triangular_hnf_gamma_a1_group(((12, 0), (0, 12))), np.int32)
        bonds = triangular_j1j2_bonds(12, 12, j2=float(spec['j2_over_j1']), periodic=True)
        probability = np.abs(np.asarray([value for _, _, value in bonds], float))
        probability /= probability.sum()
        amplitude = jax.jit(make_feature_lut_projected_amplitude(
            cfg, images, chunk=sampler_chunk, active_top_widths=active))
        guide = make_pair_support_guide(amplitude)
        guide_batch = jax.jit(jax.vmap(guide, in_axes=(None, 0)))
        action, metadata = make_cached_projected_heisenberg_bridge(
            cfg, images, bonds, probability, projection_chunk=h_chunk, active_top_widths=active)
        result['kernel_metadata'] = metadata
        prepared = tuple(prepare_feature_lut_bank(bank, cfg) for bank in execution)
        observed = timed('guide_first', guide_batch, prepared, states)
        expected = np.hypot(abs(start['terminal_psi'][:, 0]), abs(start['terminal_psi'][:, 1]))
        if finite_relative_error(np.asarray(observed), expected) > 2e-10:
            raise ArithmeticError('frozen support-terminal physical amplitudes did not replay')
        timed('guide_warm', guide_batch, prepared, states)
        result['amplitude_centers'] += 4*chains
        for k in range(2):
            cold = timed('H_first_'+str(k), action, prepared[k], states[:1])
            result['full_H_centers'] += 1
            warm = timed('H_warm_'+str(k), action, prepared[k], states[:1])
            result['full_H_centers'] += 1
            for a, b in zip(cold, warm, strict=True):
                np.testing.assert_allclose(a, b, rtol=2e-11, atol=1e-100)
        forecast = 1.4*((burn+records*between+1)*result['timings']['guide_warm']
            + chains*records*sum(result['timings']['H_warm_'+str(k)] for k in range(2)))
        result['remaining_cost_forecast_seconds'] = forecast
        if forecast > cap-(perf_counter()-started):
            result['status'] = 'cost_complete_audit_not_released_due_to_measured_budget'
        else:
            key = jax.random.PRNGKey(int(spec['seed']))
            sampler = make_record_sampler(guide, records=records, burn_proposals=burn,
                proposals_between=between, exchange_edges=tuple((a, b) for a, b, _ in bonds))
            last, key, base_rows, info = timed('fresh_Q0_records', sampler, prepared, key, states)
            result['amplitude_centers'] += 2*chains*(burn+records*between+1)
            push = make_exchange_record_bridge([(a, b) for a, b, _ in bonds], probability,
                n_sites=144, bridge_fraction=.5)
            key, push_key = jax.random.split(key)
            rows, used, labels = timed('fixed_exchange_record_push', push, push_key, base_rows.reshape(-1, 144))
            np.savez_compressed(output/'registered_audit_rows.npz', base_rows=np.asarray(base_rows),
                pushed_rows=np.asarray(rows), used_bridge=np.asarray(used), bond_labels=np.asarray(labels),
                last_states=np.asarray(last), rng=np.asarray(key))
            products = []
            for index in range(len(rows)):
                endpoint = []
                for k in range(2):
                    triple = timed('audit_H_'+str(index)+'_'+str(k), action, prepared[k], rows[index:index+1])
                    result['full_H_centers'] += 1
                    endpoint.append([np.asarray(value)[0] for value in triple])
                products.append(endpoint)
                np.savez_compressed(output/'partial_products.npz', products=np.asarray(products))
            data = np.asarray(products)
            psi, hp, push_density = data[:, :, 0], data[:, :, 1], data[:, :, 2].real.sum(axis=1)
            q0 = np.sum(abs(psi)**2, axis=1)
            density = .5*(q0+push_density)
            if not np.isfinite(density).all() or np.any(density <= 0):
                raise FloatingPointError('positive finite bridge density required at every record')
            groups = np.tile(np.arange(chains), records)
            moments = paired_endpoint_moments(psi, hp, density, groups, n_sites=144)
            # Record independent chain concentration and within-run drift for
            # BOTH norm and full second moment, without an IID ESS claim.
            norm = abs(psi)**2/density[:, None]
            second = abs(hp)**2/density[:, None]
            diagnostics = {}
            for name, values in (('norm', norm), ('H2', second)):
                diagnostics[name+'_split_rhat'] = [finite_chain_rhat(values[:, k].reshape(records, chains)) for k in range(2)]
                diagnostics[name+'_maximum_chain_fraction'] = [float(max(values[groups == label, k].sum()
                    for label in range(chains))/values[:, k].sum()) if values[:, k].sum() > 0 else None
                    for k in range(2)]
            result.update(status=('fresh_paired_H_bridge_coverage_diagnostic_complete' if coverage_only
                                  else 'fresh_paired_H_bridge_audit_complete'), energy_estimate=True,
                moments=moments, moment_diagnostics=diagnostics,
                observed_push_fraction=float(np.asarray(used).mean()),
                sampler_diagnostics=jax.tree.map(lambda x: np.asarray(x).tolist(), info))
            np.savez_compressed(output/'untouched_products.npz', rows=np.asarray(rows), psi=psi,
                Hpsi=hp, Q0=q0, Qpush=push_density, guide_density=density, groups=groups,
                base_rows=np.asarray(base_rows), used_bridge=np.asarray(used), bond_labels=np.asarray(labels),
                last_states=np.asarray(last), rng=np.asarray(key))
    except TimeoutError as error:
        result.update(status='bounded_incomplete', error=str(error), energy_estimate=False)
    except Exception as error:
        result.update(status='failed_validation', error=repr(error), energy_estimate=False)
        atomic_json(output/'results.json', dict(result, total_seconds=perf_counter()-started))
        raise
    result['total_seconds'] = perf_counter()-started
    atomic_json(output/'results.json', result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("cost", "audit", "importance", "root-cost", "overlap", "paired", "support", "support-cost", "bridge"), default="cost")
    args = parser.parse_args()
    if args.phase == 'support-cost':
        run_pair_support_cost(json.loads(args.plan.read_text()), args.output)
    elif args.phase == 'support':
        run_pair_support_stage(json.loads(args.plan.read_text()),args.output)
    elif args.phase == 'bridge':
        run_h_bridge_endpoint_audit(json.loads(args.plan.read_text()), args.output)
    elif args.phase == 'paired':
        run_paired_endpoint_audit(json.loads(args.plan.read_text()),args.output)
    else:
        run(json.loads(args.plan.read_text()), args.output, phase=args.phase)
