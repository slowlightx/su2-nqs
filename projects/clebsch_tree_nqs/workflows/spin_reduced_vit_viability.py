"""Bounded, process-isolated complete reduced-space ViT viability probe.

The sampled law is the *native* normalized singlet AR law.  A full spatial
projector changes that law and is deliberately not advertised as implemented.
No training, eigensolve, q cutoff, bond truncation, or old artifact mutation is
performed.  A stopped sparse-translation diagnostic is a failure to finish,
never a truncated implementation of a unitary or projector.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig, init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes, many_cell_schur_vit_parameter_count,
    sample_many_cell_schur_vit,
)
from su2_nqs.operators.tree_automorphism import compile_tree_automorphism
from su2_nqs.operators.tree_path_connectivity import compile_tree_heisenberg_graph_connectivity_plan
from su2_nqs.su2.fusion_tree import FusionTreeState
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout, validate_layout_state
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.many_cell_tree import (
    build_many_cell_tree_connectivity, compile_many_cell_device_hamiltonian,
    evaluate_many_cell_device_local_energy,
)

ROOT = Path(__file__).resolve().parents[3]
MODULE = "projects.clebsch_tree_nqs.workflows.spin_reduced_vit_viability"


def _plain(value: object) -> object:
    if isinstance(value, (np.generic, np.ndarray)):
        return _plain(value.tolist())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def model_config(settings: dict, *, small: bool = False) -> ManyCellSchurViTConfig:
    """Use complete balanced plaquette labels, doubled q, with no q cutoff."""
    size = 2 if small else settings["lx"]
    layout = rectangular_four_spin_layout(size, 2 if small else settings["ly"])
    return ManyCellSchurViTConfig(layout, embedding_dim=settings["embedding_dim"],
        head_count=settings["head_count"], depth=settings["depth"],
        direct_phase_initialization_scale=settings["direct_phase_initialization_scale"],
        conditional_mode=settings.get('conditional_mode','legacy'))


def measured(call):
    """Time the completed value, including compilation on a first invocation."""
    started = time.perf_counter()
    value = call()
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, value)
    return value, time.perf_counter()-started


def physical_reflection(rows: int, columns: int) -> tuple[int, ...]:
    """Involutive unitary site relabeling, 120-degree -> 60-degree metric."""
    return tuple(row*columns+(-column)%columns for row in range(rows) for column in range(columns))


def neural_bonds(physical_bonds, rows: int, columns: int):
    """Relabel a physical graph before the layout's cell-major remapping."""
    image=physical_reflection(rows,columns)
    return tuple((*sorted((image[first],image[second])),coupling) for first,second,coupling in physical_bonds)


def neural_image(physical_image, rows: int, columns: int):
    """Conjugate a spatial action as R g R, not an unreflected action."""
    image=physical_reflection(rows,columns)
    return tuple(image[physical_image[image[site]]] for site in range(rows*columns))


def support_geometry_error(config) -> float:
    """Compare native locality envelope with its actual physical supports."""
    from su2_nqs.models.many_cell_schur_vit import _support_torus_distances
    from su2_nqs.su2.tree_rotations import leaf_signature
    from su2_nqs.models.tree_gru import tree_decisions
    columns=config.layout.physical_cells[0][2]-config.layout.physical_cells[0][0]
    rows=config.n//columns
    reflection=physical_reflection(rows,columns)
    physical_order=tuple(reflection[site] for site in config.layout.site_order)
    supports=[tuple(physical_order[i] for i in leaf_signature(config.layout.tree.internal_nodes[d.node_index]))
        for d in tree_decisions(config.layout.tree)]
    def distance(first,second):
        r,c=divmod(first,columns)
        rr,cc=divmod(second,columns)
        return math.sqrt(min((rr-r+dr*rows)**2+(cc-c+dc*columns)**2-
            (rr-r+dr*rows)*(cc-c+dc*columns) for dr in (-1,0,1) for dc in (-1,0,1)))
    expected=np.asarray([[min(distance(a,b) for a in left for b in right) for right in supports] for left in supports])
    return float(np.max(np.abs(expected-_support_torus_distances(config))))


def probability_force(params, labels, local_energies, config, *, weights=None):
    """Both-sided energy force, not derivative of a fixed-sample local mean.

With exact Born weights this equals the exact Rayleigh gradient.  With native
Born samples it is the covariance estimator.  The H action is computed before
this score VJP; no derivative through a stopped empirical sampling law is used.
    """
    labels = jnp.asarray(labels)
    local = jnp.asarray(local_energies)
    probability = jnp.ones(len(labels))/len(labels) if weights is None else jnp.asarray(weights)
    probability = jax.lax.stop_gradient(probability)
    center = jnp.sum(probability*local)
    residual = jax.lax.stop_gradient(local-center)
    def loss(p):
        logs = many_cell_schur_log_amplitudes(p, labels, config)
        return 2*jnp.real(jnp.sum(probability*jnp.conj(residual)*logs))
    return jax.grad(loss)(params)


def permutation_swaps(image: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Ordered physical transpositions whose action maps old site to image."""
    if sorted(image) != list(range(len(image))):
        raise ValueError("image must be a site permutation")
    visited = set()
    swaps = []
    for first in range(len(image)):
        if first in visited:
            continue
        cycle = []
        site = first
        while site not in visited:
            visited.add(site)
            cycle.append(site)
            site = image[site]
        swaps.extend((cycle[0], other) for other in cycle[1:])
    return tuple(swaps)


def swap_column(config, row, physical_pair):
    """Exact singlet P_ij column from P_ij=2 S_i.S_j + I/2."""
    bonds = config.layout.remap_bonds(((physical_pair[0], physical_pair[1], 2.0),))
    plan = compile_tree_heisenberg_graph_connectivity_plan(config.layout.tree, bonds, target_q=0)
    states, values = plan.column(FusionTreeState(tuple(int(q) for q in row)))
    result = {state.labels: float(v) for state, v in zip(states, values, strict=True)}
    old = tuple(int(q) for q in row)
    result[old] = result.get(old, 0.0)+0.5
    return {key: value for key, value in result.items() if value != 0.0}


def _write_stage(output, stage, record):
    atomic_json(output/(stage+".json"), _plain(record))


def _small(settings, output):
    from su2_nqs.benchmarks.exact_diagonalization import heisenberg_sz_matrix, spin_squared_matrix
    from su2_nqs.su2.recoupling import tree_basis_matrix
    config = model_config(settings, small=True)
    params = init_many_cell_schur_vit(jax.random.PRNGKey(settings["seed"]), config)
    labels = np.asarray([s.labels for s in config.layout.tree.enumerate_states(0)], dtype=np.int32)
    u, _ = tree_basis_matrix(config.layout.tree, 0, 0)
    bonds = ((0,1,1.0),(1,3,0.7),(2,3,0.6),(0,2,1.1),(0,3,0.125))
    h_physical, _ = heisenberg_sz_matrix(4, bonds)
    s2, _ = spin_squared_matrix(4)
    h = np.asarray(u.conj().T@h_physical@u)
    gram = float(np.max(np.abs(u.conj().T@u-np.eye(2))))
    spin = float(np.max(np.abs(s2@u)))
    host = build_many_cell_tree_connectivity(labels, config, bonds)
    lookup = {tuple(row): i for i,row in enumerate(labels)}
    actual = np.zeros((2,2))
    for column in range(2):
        for slot in range(int(host.widths[column])):
            actual[lookup[tuple(host.connected_labels[column,slot])],column] = host.coefficients[column,slot]
    h_error = float(np.max(np.abs(h-actual)))
    logs = many_cell_schur_log_amplitudes(params, labels, config)
    psi = jnp.exp(logs)
    weights = jnp.abs(psi)**2
    local = jnp.asarray(h)@psi/psi
    force = probability_force(params, labels, local, config, weights=weights)
    def rayleigh(p):
        c = jnp.exp(many_cell_schur_log_amplitudes(p, labels, config))
        return jnp.real(jnp.vdot(c,jnp.asarray(h)@c)/jnp.vdot(c,c))
    exact_force = jax.grad(rayleigh)(params)
    gradient_error = max(float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        for a,b in zip(jax.tree_util.tree_leaves(force),jax.tree_util.tree_leaves(exact_force),strict=True))
    sample_fn = jax.jit(lambda p,k: sample_many_cell_schur_vit(p,k,4096,config))
    samples = np.asarray(sample_fn(params,jax.random.PRNGKey(settings["seed"]+1)))
    frequencies = np.asarray([np.mean(np.all(samples==row,axis=1)) for row in labels])
    frequency_error = float(np.max(np.abs(frequencies-np.asarray(weights))))
    large_small = ManyCellSchurViTConfig(rectangular_four_spin_layout(4,4),
        embedding_dim=settings["embedding_dim"],head_count=settings["head_count"],depth=settings["depth"])
    small_params = init_many_cell_schur_vit(jax.random.PRNGKey(5),large_small)
    labels16 = np.asarray([s.labels for s in large_small.layout.tree.enumerate_states(0)],dtype=np.int32)
    log16 = many_cell_schur_log_amplitudes(small_params,labels16,large_small)
    normalization16 = float(abs(jnp.sum(jnp.exp(2*jnp.real(log16)))-1))
    # An independent physical SWAP matrix, not another recoupling convention.
    swap = np.zeros((16,16))
    for index in range(16):
        bits = [int(x) for x in f"{index:04b}"]
        bits[0],bits[2] = bits[2],bits[0]
        swap[int(''.join(map(str,bits)),2),index] = 1
    physical_swap = np.asarray(u.conj().T@swap@u)
    reduced_swap = np.zeros((2,2))
    for column,row in enumerate(labels):
        for state,value in swap_column(config,row,(0,2)).items():
            reduced_swap[lookup[state],column] = value
    swap_error = float(np.max(np.abs(physical_swap-reduced_swap)))
    # This C2 subgroup already demonstrates that projected Born != native AR.
    projected = (np.eye(2)+physical_swap)@np.asarray(psi)/2
    projected_probability = abs(projected)**2/np.vdot(projected,projected).real
    changed_law = float(np.max(np.abs(projected_probability-np.asarray(weights))))
    result = dict(status="passed",n4_basis_gram_error=gram,n4_S2_error=spin,
        n4_complete_H_error=h_error,n4_both_sided_gradient_error=gradient_error,
        n4_normalization_error=float(abs(jnp.sum(weights)-1)),
        n4_sample_frequency_max_error=frequency_error,n16_singlet_dimension=len(labels16),
        n16_AR_normalization_error=normalization16,n4_physical_swap_error=swap_error,
        illustrative_C2_projected_Born_minus_native_max=changed_law,
        eigensolves=0,statement="Spatial coherent projection changes the Born law; native draws cannot be relabelled projected draws.")
    if max(gram,spin,h_error,gradient_error,normalization16,swap_error)>1e-10 or frequency_error>.05:
        raise AssertionError(result)
    _write_stage(output,"small",result)
    return result


def _native(settings, output):
    config = model_config(settings)
    params = init_many_cell_schur_vit(jax.random.PRNGKey(settings["seed"]),config)
    count = settings["samples"]
    draw = jax.jit(lambda p,k: sample_many_cell_schur_vit(p,k,count,config))
    key = jax.random.PRNGKey(settings["seed"]+2)
    labels, cold = measured(lambda: draw(params,key))
    key, next_key = jax.random.split(key)
    next_labels,warm = measured(lambda: draw(params,next_key))
    labels = np.asarray(labels)
    for row in labels:
        validate_layout_state(config.layout,row,target_q=0)
    amplitude = jax.jit(lambda p,x: many_cell_schur_log_amplitudes(p,x,config))
    logs,amp_cold = measured(lambda: amplitude(params,labels))
    _,amp_warm = measured(lambda: amplitude(params,labels))
    _,checkpoint_write=measured(lambda: save_checkpoint(output/"native_checkpoint.pkl",jax.tree_util.tree_map(np.asarray,params),
        optimizer_state={"step":0,"training":False},
        metadata={"sampler":"exact-normalized independent AR, no MCMC chain", "next_draw_key":np.asarray(next_key),
            "sample_count":count,"config":settings,"native_Born":True,"full_A1":False}))
    loaded,checkpoint_read = measured(lambda: load_checkpoint(output/"native_checkpoint.pkl"))
    replay,_ = measured(lambda: draw(loaded["params"],jnp.asarray(loaded["metadata"]["next_draw_key"])))
    replay_logs = amplitude(loaded["params"],labels)
    np.testing.assert_array_equal(replay,next_labels)
    error = float(np.max(np.abs(np.asarray(replay_logs)-np.asarray(logs))))
    np.savez_compressed(output/"native_samples.npz",labels=labels,log_amplitudes=np.asarray(logs),next_labels=np.asarray(next_labels))
    # A score VJP is independent of H feasibility. It is a cost/finite-gradient
    # probe, not an energy update or evidence of training convergence.
    def score(p):
        values=many_cell_schur_log_amplitudes(p,labels,config)
        direction=jnp.linspace(-1.,1.,count)
        return jnp.real(jnp.sum(direction*values))+jnp.imag(jnp.sum(direction[::-1]*values))
    score_vjp=jax.jit(jax.grad(score))
    derivative,score_cold=measured(lambda: score_vjp(params))
    _,score_warm=measured(lambda: score_vjp(params))
    assert all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(derivative))
    reflection=physical_reflection(settings["ly"],settings["lx"])
    result = dict(status="passed",n_sites=config.n,n_internal_labels=config.decision_count,
        exact_singlet_dimension=math.comb(36,18)//19,
        parameters=many_cell_schur_vit_parameter_count(params),sample_count=count,
        samples_cold_compile_and_execute_seconds=cold,samples_warm_seconds=warm,
        samples_warm_per_second=count/warm,amplitude_cold_seconds=amp_cold,amplitude_warm_seconds=amp_warm,
        checkpoint_next_sample_identical=True,checkpoint_amplitude_error=error,
        checkpoint_write_seconds=checkpoint_write,checkpoint_read_seconds=checkpoint_read,
        checkpoint_bytes=(output/"native_checkpoint.pkl").stat().st_size,
        all_parameter_score_VJP_cold_seconds=score_cold,all_parameter_score_VJP_warm_seconds=score_warm,
        physical_to_neural_site_map=reflection,
        model_leaf_to_actual_physical_site=tuple(reflection[site] for site in config.layout.site_order),
        physical_support_distance_max_error=support_geometry_error(config),
        params_sha256=pytree_sha256(params),complete_q_support=True,native_normalized_Born=True,
        exact_SU2=True,full_A1=False,spatially_projected_Born=False,training_steps=0)
    _write_stage(output,"native",result)
    return result


def _host(settings, output):
    config = model_config(settings)
    payload = load_checkpoint(output/"native_checkpoint.pkl")
    params = payload["params"]
    labels = np.load(output/"native_samples.npz")["labels"][:settings["hamiltonian_samples"]]
    bonds = triangular_j1j2_bonds(6,6,j2=settings["j2"],periodic=True)
    assert len(bonds)==216 and sum(j==1 for _,_,j in bonds)==108
    workload = build_many_cell_tree_connectivity(labels,config,neural_bonds(bonds,6,6))
    tile = settings["amplitude_tile_size"]
    amp = jax.jit(lambda p,x: many_cell_schur_log_amplitudes(p,x,config))
    _,amp_cold = measured(lambda: amp(params,np.tile(labels[0],(tile,1))))
    centers = np.asarray(amp(params,labels))
    local = []
    started = time.perf_counter()
    for row,width in enumerate(workload.widths):
        all_logs = []
        for offset in range(0,int(width),tile):
            batch = workload.connected_labels[row,offset:min(offset+tile,int(width))]
            padded = np.concatenate([batch,np.tile(labels[row],(tile-len(batch),1))])
            all_logs.extend(np.asarray(amp(params,padded))[:len(batch)])
        local.append(np.sum(workload.coefficients[row,:width]*np.exp(np.asarray(all_logs)-centers[row])))
    query_seconds = time.perf_counter()-started
    local = np.asarray(local)
    np.savez_compressed(output/"host_H.npz",labels=labels,local_energies=local,widths=workload.widths)
    force_fn = jax.jit(lambda p: probability_force(p,labels,local,config))
    gradient,grad_cold = measured(lambda: force_fn(params))
    _,grad_warm = measured(lambda: force_fn(params))
    leaves = jax.tree_util.tree_leaves(gradient)
    grad_norm = float(np.sqrt(sum(np.sum(np.asarray(x)**2) for x in leaves)))
    if not all(np.all(np.isfinite(x)) for x in leaves):
        raise FloatingPointError("non-finite full-parameter covariance force")
    reloaded = load_checkpoint(output/"native_checkpoint.pkl")
    replay_logs = np.asarray(amp(reloaded["params"],labels))
    amplitude_replay_error = float(np.max(np.abs(replay_logs-centers)))
    restored_connected=[]
    width=int(workload.widths[0])
    for offset in range(0,width,tile):
        batch=workload.connected_labels[0,offset:min(offset+tile,width)]
        padded=np.concatenate([batch,np.tile(labels[0],(tile-len(batch),1))])
        restored_connected.extend(np.asarray(amp(reloaded["params"],padded))[:len(batch)])
    restored_local=np.sum(workload.coefficients[0,:width]*np.exp(np.asarray(restored_connected)-replay_logs[0]))
    replay_error=float(abs(restored_local-local[0]))
    result = dict(status="passed",physical_bonds=len(bonds),NN_bonds=108,NNN_bonds=108,
        sample_count=len(labels),full_H_calls=len(labels),full_H_bond_calls=len(labels)*216,
        exact_host_connectivity_seconds=workload.connectivity_seconds,unique_widths=workload.widths,
        amplitude_tile=tile,amplitude_compile_seconds=amp_cold,
        all_connected_amplitudes_warm_seconds=query_seconds,amplitude_queries=int(sum(workload.widths)),
        local_energies=local,full_parameter_gradient_cold_seconds=grad_cold,
        full_parameter_gradient_warm_seconds=grad_warm,gradient_l2=grad_norm,
        all_parameter_leaf_count=len(leaves),gradient_parameters=sum(x.size for x in leaves),
        checkpoint_H_ratio_replay_error=replay_error,
        checkpoint_amplitude_replay_error=amplitude_replay_error,
        checkpoint_full_H_ratio_replay_amplitude_queries=width,
        energy_is_untrained_cost_diagnostic=True,ratio_truncation=False,
        gradient_convention="2 Re covariance(score*, E_local); full H forward plus all-parameter score VJP")
    _write_stage(output,"host",result)
    return result


def _device(settings, output):
    config = model_config(settings)
    params = load_checkpoint(output/"native_checkpoint.pkl")["params"]
    labels = np.load(output/"native_samples.npz")["labels"][:settings["hamiltonian_samples"]]
    bonds = triangular_j1j2_bonds(6,6,j2=settings["j2"],periodic=True)
    started = time.perf_counter()
    compiled = compile_many_cell_device_hamiltonian(config,neural_bonds(bonds,6,6),candidate_capacity=None,
        connected_capacity=settings["connected_capacity"],amplitude_tile_size=settings["amplitude_tile_size"])
    table_seconds = time.perf_counter()-started
    result = dict(status="started",physical_bonds=len(bonds),numeric_tables_seconds=table_seconds,
        raw_transition_slots_per_sample=compiled.raw_transition_slots_per_sample,
        connected_capacity=compiled.connected_capacity,bucket_capacities=compiled.bucket_capacities,
        bucket_bond_counts=compiled.bucket_bond_counts,candidate_capacity=None,zero_bond_or_q_truncation=True)
    _write_stage(output,"device",result)
    for count in (1,len(labels)):
        first,cold = measured(lambda: evaluate_many_cell_device_local_energy(compiled,params,labels[:count],fail_on_overflow=False))
        second,warm = measured(lambda: evaluate_many_cell_device_local_energy(compiled,params,labels[:count],fail_on_overflow=False))
        row = dict(cold_compile_execute_seconds=cold,warm_seconds=warm,
            unique_widths=second.unique_widths,overflow=second.overflow,
            local_energies=second.local_energies,full_H_calls=count,full_H_bond_calls=count*len(bonds))
        result[str(count)+"_samples"] = row
        _write_stage(output,"device",result)
        if np.any(second.overflow):
            row["first_failed_sample"]=int(np.flatnonzero(second.overflow)[0])
            row["overflow_bond_identification"]="aggregate kernel flag; individual bond overflow not exposed by current API"
            _write_stage(output,"device",result)
            raise RuntimeError("full-H kernel overflow: no truncated result used")
        if (output/"host_H.npz").exists():
            expected = np.load(output/"host_H.npz")["local_energies"][:count]
            row["host_device_max_error"] = float(np.max(np.abs(second.local_energies-expected)))
            np.testing.assert_allclose(second.local_energies,expected,rtol=1e-10,atol=1e-10)
        result[str(count)+"_samples"] = row
        _write_stage(output,"device",result)
    result["status"] = "passed"
    _write_stage(output,"device",result)
    return result


def _spatial(settings, output):
    config = model_config(settings)
    group = triangular_hnf_gamma_a1_group(((6,0),(0,6)))
    monomial = 0
    for image in group:
        try:
            compile_tree_automorphism(config.layout.tree,config.layout.site_order,neural_image(image,6,6))
            monomial += 1
        except ValueError:
            pass
    image = neural_image(tuple((i//6)*6+(i%6+1)%6 for i in range(36)),6,6)
    swaps = permutation_swaps(image)
    result = dict(status="diagnostic",group_size=len(group),tree_monomial_group_elements=monomial,
        full_A1_available=False,spatially_projected_Born_available=False,
        translation_transpositions=len(swaps),full_translation_completed=False,
        meaning="This factorization tests cost, not an impossibility theorem for all spatial projectors.",prefixes=[])
    _write_stage(output,"spatial",result)
    if not (output/"native_samples.npz").exists():
        result["stopped_reason"]="native sampling did not complete"
        _write_stage(output,"spatial",result)
        return result
    first = tuple(int(q) for q in np.load(output/"native_samples.npz")["labels"][0])
    vector = {first:1.0}
    calls = 0
    started = time.perf_counter()
    for index,pair in enumerate(swaps):
        if time.perf_counter()-started>settings["translation_seconds"]:
            result["stopped_reason"]="pre-registered translation wall budget"
            break
        new = {}
        complete = True
        for row,coefficient in vector.items():
            if calls>=settings["translation_maximum_column_calls"] or time.perf_counter()-started>settings["translation_seconds"]:
                result["stopped_reason"]="pre-registered exact column/time budget"
                complete=False
                break
            for state,value in swap_column(config,row,pair).items():
                new[state]=new.get(state,0.0)+coefficient*value
            calls+=1
            if len(new)>settings["translation_maximum_branches"]:
                result["stopped_reason"]="branch cap exceeded: prefix discarded, not truncated"
                complete=False
                break
        if not complete:
            result["unfinished_next_prefix_branch_lower_bound"]=len(new)
            break
        vector={row:value for row,value in new.items() if value!=0.0}
        norm=float(sum(value*value for value in vector.values()))
        result["prefixes"].append(dict(transposition=index+1,branches=len(vector),norm_error=abs(norm-1)))
        if abs(norm-1)>1e-9:
            raise AssertionError("spatial swap prefix not unitary")
        _write_stage(output,"spatial",result)
    else:
        result["full_translation_completed"]=True
    result.update(exact_column_calls=calls,elapsed_seconds=time.perf_counter()-started,
        coefficient_tolerance=1e-14,branch_pruning=False)
    _write_stage(output,"spatial",result)
    return result


STAGES={"small":_small,"native":_native,"host":_host,"device":_device,"spatial":_spatial}


def _run_child(stage, settings, output):
    started=time.perf_counter()
    try:
        result=STAGES[stage](settings,output)
    except BaseException as error:
        result=dict(status="failed",exception=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        _write_stage(output,stage+"_failure",result)
        raise
    _write_stage(output,stage+"_completion",dict(status=result["status"],wall_seconds=time.perf_counter()-started))


def _rss_bytes(pid: int) -> int:
    record=subprocess.run(["ps","-o","rss=","-p",str(pid)],capture_output=True,text=True,check=False)
    value=record.stdout.strip()
    if record.returncode!=0:
        raise RuntimeError("RSS measurement denied or process exited: "+record.stderr.strip())
    return int(value)*1024 if value else 0


def run(config_path: Path) -> dict:
    settings=json.loads(config_path.read_text())
    output=ROOT/settings["output"]
    output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),config_path,ROOT/"su2_nqs/models/many_cell_schur_vit.py",
        ROOT/"su2_nqs/su2/many_cell_schur.py",ROOT/"su2_nqs/operators/tree_path_connectivity.py",
        ROOT/"su2_nqs/operators/jax_tree_path_connectivity.py",ROOT/"su2_nqs/vmc/many_cell_tree.py",
        ROOT/"su2_nqs/vmc/checkpoint.py"]
    manifest=transitive_local_source_manifest([Path(__file__)],root=ROOT)
    for source in sources:
        manifest[str(source.relative_to(ROOT))]=file_sha256(source)
    snapshot=output/"source_snapshot"
    for name in manifest:
        source=ROOT/name
        saved=snapshot/name
        saved.parent.mkdir(parents=True,exist_ok=True)
        saved.write_bytes(source.read_bytes())
    atomic_json(output/"registration.json",dict(config=settings,
        sources=manifest,source_snapshot=str(snapshot.relative_to(ROOT)),
        runtime=numerical_runtime_identity(),training_steps=0,eigensolves=0,
        native_AR_not_full_A1=True,missing_full_A1_is_production_blocker=True))
    started=time.perf_counter()
    records={}
    env={**os.environ,"JAX_PLATFORM_NAME":"cpu","JAX_ENABLE_X64":"true","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2"}
    for stage in STAGES:
        if stage in ("host","device") and not (output/"native_completion.json").exists():
            records[stage]={"status":"not_run","reason":"native sampling prerequisite failed"}
            continue
        if stage!="small" and not (output/"small_completion.json").exists():
            records[stage]={"status":"not_run","reason":"ordered small oracles failed"}
            continue
        remaining=settings["total_seconds"]-(time.perf_counter()-started)
        if remaining<=0:
            records[stage]={"status":"not_run","reason":"registered total wall budget"}
            continue
        cap=min(settings["child_seconds"],remaining)
        stage_started=time.perf_counter()
        peak=0
        stop=None
        with (output/(stage+".log")).open("w") as stream:
            process=subprocess.Popen([sys.executable,"-m",MODULE,"--config",str(config_path),"--child",stage],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    peak=max(peak,_rss_bytes(process.pid))
                except RuntimeError as error:
                    if process.poll() is not None:
                        break
                    stop="RSS unmeasured: fail-closed; "+str(error)
                elapsed=time.perf_counter()-stage_started
                if peak>settings["rss_limit_GiB"]*1024**3:
                    stop="pre-registered RSS budget"
                elif elapsed>cap:
                    stop="pre-registered process wall budget"
                if stop:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                time.sleep(0.25)
        records[stage]=dict(status="completed" if process.returncode==0 else "failed",
            exit_code=process.returncode,peak_RSS_GiB=peak/1024**3,
            wall_seconds=time.perf_counter()-stage_started,stop_reason=stop,
            child_limit_seconds=cap)
        atomic_json(output/"progress.json",records)
        print(stage,json.dumps(records[stage]),flush=True)
    result=dict(stages=records,total_wall_seconds=time.perf_counter()-started,
        native_exact_SU2_available=records.get("native",{}).get("status")=="completed",
        complete_full_H_device_probe_passed=records.get("device",{}).get("status")=="completed",
        full_A1_available=False,projected_Born_sampling_available=False,
        N36_training_authorized=False,scientific_energy_claim=False,
        conclusion="Native complete SU2 AR viability and full-H cost are tested independently. Full A1 coefficient action and its Born sampler remain missing, so this is not a production SU2+A1 Spatial-ViT.")
    atomic_json(output/"results.json",result)
    atomic_json(output/"completion.json",dict(results_sha256=file_sha256(output/"results.json"),wall_seconds=result["total_wall_seconds"]))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--child",choices=tuple(STAGES))
    args=parser.parse_args()
    path=args.config.resolve()
    if args.child:
        settings=json.loads(path.read_text())
        _run_child(args.child,settings,ROOT/settings["output"])
    else:
        print(json.dumps(run(path),indent=2))


if __name__=="__main__":
    main()
