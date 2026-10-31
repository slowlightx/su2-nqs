"""Bounded AFM initialization and exact-small coherent A1 Born-law validation.

No energy optimization, eigensolve, head, q cutoff, or large carrier conversion
is performed. The N36 state remains native S0, NOT a repaired full-A1 state.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_reduced_vit_viability import (
    model_config,measured,probability_force,physical_reflection,
)
from projects.clebsch_tree_nqs.workflows.spin_reduced_vit_train import (
    _Hamiltonian,local_energy_statistics,
)
from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis,physical_permutation_index_table
from su2_nqs.frontier.artifacts import atomic_json,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest,numerical_runtime_identity
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.models.many_cell_schur_vit import (
    many_cell_schur_log_amplitudes,sample_many_cell_schur_vit,many_cell_schur_vit_parameter_count,
)
from su2_nqs.models.schur_initialization import afm_child_casimir_initialization
from su2_nqs.su2.reduced_physical_overlap import tree_overlaps_on_bits
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint
from su2_nqs.vmc.exact_reduced_projector import exact_projected_reduced_state

ROOT=Path(__file__).resolve().parents[3]
MODULE="projects.clebsch_tree_nqs.workflows.spin_reduced_vit_init"


def _q_readout(labels,config):
    decoded=config.layout.decode(np.asarray(labels))
    q=decoded.cell_q
    fractions={str(doubled):np.mean(q==doubled,axis=1) for doubled in (0,2,4)}
    return dict(cell_q={key:dict(probability=float(value.mean()),
        iid_stderr=float(value.std(ddof=1)/np.sqrt(len(value)))) for key,value in fractions.items()},
        nonroot_intercell_q_positive_probability=float(np.mean(decoded.intercell_q[:,:-1]>0)),
        root_q_identically_zero=bool(np.all(decoded.intercell_q[:,-1]==0)),
        cell_average_SA_squared=float(np.mean((q/2)*(q/2+1))),
        frequency_is_not_multiplicity_spectrum=True)


def run(config_path:Path)->dict:
    """Register once and enforce the fixed compute cap in an isolated process."""
    settings=json.loads(config_path.read_text())
    if not isinstance(settings["total_seconds"],int) or not 1<=settings["total_seconds"]<=1800:
        raise ValueError("initialization development cap must be at most 30 minutes")
    output=ROOT/settings["output"]
    output.mkdir(parents=True,exist_ok=False)
    parent=ROOT/settings["parent_checkpoint"]
    if file_sha256(parent)!=settings["parent_checkpoint_sha256"]:
        raise ValueError("frozen source checkpoint changed")
    sources=transitive_local_source_manifest([Path(__file__)],root=ROOT,
        package_prefixes=("su2_nqs","projects.clebsch_tree_nqs"))
    sources[str(config_path.relative_to(ROOT))]=file_sha256(config_path)
    for name in sources:
        saved=output/"source_snapshot"/name
        saved.parent.mkdir(parents=True,exist_ok=True)
        saved.write_bytes((ROOT/name).read_bytes())
    atomic_json(output/"registration.json",dict(config=settings,sources=sources,
        registered_UTC=datetime.now(timezone.utc).isoformat(),runtime=numerical_runtime_identity(),
        no_training=True, no_ED_eigensolve=True,full_A1_N36=False,
        hard_compute_cap_seconds=settings["total_seconds"]))
    started=time.perf_counter()
    env={**os.environ,"JAX_PLATFORM_NAME":"cpu","JAX_ENABLE_X64":"true",
        "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2"}
    with (output/"compute.log").open("w") as stream:
        process=subprocess.Popen([sys.executable,"-m",MODULE,"--config",str(config_path),"--child"],
            cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        try:
            exit_code=process.wait(timeout=settings["total_seconds"])
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
            exit_code=process.returncode
            atomic_json(output/"budget_stop.json",dict(status="hard_compute_cap",
                compute_seconds=time.perf_counter()-started))
    result_path=output/"results.json"
    atomic_json(output/"completion.json",dict(exit_code=exit_code,
        results_sha256=file_sha256(result_path) if result_path.exists() else None,
        compute_wall_seconds=time.perf_counter()-started,
        hard_compute_cap_seconds=settings["total_seconds"]))
    if exit_code:
        raise RuntimeError(f"bounded initialization failed ({exit_code}); see {output/'compute.log'}")
    return json.loads(result_path.read_text())


def _compute(settings:dict,output:Path)->dict:
    started=time.perf_counter()
    parent=ROOT/settings["parent_checkpoint"]
    if file_sha256(parent)!=settings["parent_checkpoint_sha256"]:
        raise ValueError("frozen source checkpoint changed after registration")
    loaded=load_checkpoint(parent)
    params=jax.tree_util.tree_map(jnp.asarray,loaded["params"])
    warm=afm_child_casimir_initialization(params,settings["inverse_temperature"])
    assert many_cell_schur_vit_parameter_count(warm)==settings["expected_parameters"]
    # First exhaustive normalization and genuine full-A1 law on the small torus.
    small_settings={**settings,"lx":4,"ly":4}
    small=model_config(small_settings)
    labels=np.asarray([row.labels for row in small.layout.tree.enumerate_states(0)],dtype=np.int32)
    logs=np.asarray(many_cell_schur_log_amplitudes(warm,labels,small))
    c=np.exp(logs)
    normalization_error=float(abs(np.sum(abs(c)**2)-1))
    if normalization_error>1e-10 or not np.all(np.isfinite(logs)) or not np.all(abs(c)>0):
        raise AssertionError("unchanged AR normalization/support failed")
    bits=np.asarray(fixed_m_spin_basis(16,0),dtype=np.int32)
    reflection=np.asarray(physical_reflection(4,4))
    order=reflection[np.asarray(small.layout.site_order)]
    (u,overlap_seconds)=measured(lambda:tree_overlaps_on_bits(small.layout.tree,labels,bits,
        physical_leaf_order=order,maximum_entries=settings["maximum_overlap_entries"]))
    group=triangular_hnf_gamma_a1_group(((4,0),(0,4)))
    probes=np.unique(np.linspace(0,len(labels)-1,32,dtype=np.int64))
    (projected,project_seconds)=measured(lambda:exact_projected_reduced_state(labels,c,u,bits,group,gram_probe_indices=probes))
    indices=physical_permutation_index_table(bits,group)
    psi=u@projected.coefficients
    invariance_error=float(max(np.max(abs(psi[row]-psi)) for row in indices))
    twice=exact_projected_reduced_state(labels,projected.coefficients,u,bits,group,gram_probe_indices=probes)
    idempotence_error=float(np.max(abs(twice.coefficients-projected.coefficients)))
    if max(invariance_error,idempotence_error)>1e-10:
        raise AssertionError("coherent full physical A1 projection failed")
    samples=np.asarray(projected.sample(jax.random.PRNGKey(settings["projected_sample_seed"]),settings["projected_samples"]))
    lookup={tuple(row):i for i,row in enumerate(labels)}
    frequency=np.bincount([lookup[tuple(row)] for row in samples],minlength=len(labels))/len(samples)
    frequency_error=float(np.max(abs(frequency-projected.probabilities)))
    frequency_threshold=float(6*np.sqrt(np.max(projected.probabilities*(1-projected.probabilities))/len(samples)))
    if frequency_error>frequency_threshold:
        raise AssertionError("exact projected Born sampling frequency check failed")
    small_result=dict(n_sites=16,singlet_dimension=len(labels),physical_Sz0_rows=len(bits),
        basis_bytes=u.nbytes,full_space_group_images=len(group),overlap_seconds=overlap_seconds,
        basis_check="all column norms and 32 deterministic Gram columns; full N8 CG oracle checked separately before launch",
        coherent_projection_seconds=project_seconds,normalization_error=normalization_error,
        minimum_native_probability=float(np.min(abs(c)**2)),projection_norm_squared=projected.projection_norm_squared,
        physical_projection_closure_error=projected.physical_projection_closure_error,
        physical_A1_invariance_error=invariance_error,idempotence_error=idempotence_error,
        native_minus_projected_probability_max=float(np.max(abs(abs(c)**2-projected.probabilities))),
        projected_iid_samples=len(samples),projected_frequency_error=frequency_error,
        projected_frequency_six_sigma_threshold=frequency_threshold,
        exact_projected_Born=True,native_draws_relabelled_projected=False)
    atomic_json(output/"exact_small_A1.json",small_result)
    np.savez_compressed(output/"exact_small_A1.npz",labels=labels,native_coefficients=c,
        projected_coefficients=projected.coefficients,projected_samples=samples)
    del u,indices,psi,bits
    # Development-only N36: independent audits, not fake paired SEs.
    config=model_config(settings)
    ledger=dict(full_H_calls=0,full_H_bond_calls=0,stage_H_calls={},kernel_compilations=[],exact_capacity_escalations=[])
    hamiltonian=_Hamiltonian(config,settings,ledger,started)
    force_fn=jax.jit(lambda p,x,e:probability_force(p,x,e,config))
    draw=jax.jit(lambda p,k:sample_many_cell_schur_vit(p,k,settings["samples"],config))
    records={}
    for name,p,seed in (("frozen_parent",params,settings["parent_audit_seed"]),("AFM_warm_parent",warm,settings["warm_audit_seed"])):
        samples,sample_seconds=measured(lambda:draw(p,jax.random.PRNGKey(seed)))
        local,hdiag=hamiltonian.evaluate(p,samples,stage=name)
        np.savez_compressed(output/(name+"_audit.npz"),labels=np.asarray(samples),local_energies=local)
        force,gradient_seconds=measured(lambda:force_fn(p,samples,jnp.asarray(local)))
        leaves=jax.tree_util.tree_leaves(force)
        if not all(np.isfinite(np.asarray(value)).all() for value in leaves):
            raise FloatingPointError("non-finite full-parameter phase-sensitive force")
        phase_leaves=[force[key] for key in ("direct_phase_bilinear","global_phase_head","global_phase_bias")]
        records[name]=dict(**local_energy_statistics(local,36),q_readout=_q_readout(samples,config),
            full_H=hdiag,sampling_seconds=sample_seconds,full_covariance_gradient_seconds=gradient_seconds,
            gradient_parameters=sum(value.size for value in leaves),gradient_l2=float(np.sqrt(sum(float(np.sum(np.asarray(value)**2)) for value in leaves))),
            phase_gradient_l2=float(np.sqrt(sum(float(np.sum(np.asarray(value)**2)) for value in phase_leaves))),
            params_sha256=pytree_sha256(p),exact_SU2=True)
        atomic_json(output/"progress.json",dict(records=records,ledger=ledger,wall_seconds=time.perf_counter()-started))
    checkpoint=output/"AFM_warm_checkpoint.pkl"
    save_checkpoint(checkpoint,jax.tree_util.tree_map(np.asarray,warm),
        optimizer_state=dict(step=0,training=False,initializer="finite beta child Casimir, existing parameters only"),
        metadata=dict(config=settings,parent_checkpoint_sha256=settings["parent_checkpoint_sha256"],
            parent_training_step=loaded["optimizer_state"]["step"],inverse_temperature=settings["inverse_temperature"],
            exact_SU2=True,full_A1=False,native_normalized_Born=True))
    restored=load_checkpoint(checkpoint)
    checkpoint_error=max(float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(
        jax.tree_util.tree_leaves(warm),jax.tree_util.tree_leaves(restored["params"]),strict=True))
    # Reuse the sealed N36 failure rather than spend time proving it again.
    prior=ROOT/settings["existing_N36_spatial_probe"]
    spatial=json.loads(prior.read_text())
    limitation=dict(full_A1_N36=False,exact_A1_N36_Born_available=False,
        new_N36_projected_amplitude_calls=0,prior_probe_path=str(prior.relative_to(ROOT)),
        prior_probe_sha256=file_sha256(prior),prior_probe=spatial,
        native_singlet_dimension=477638700,N36_Sz0_dimension=9075135300,
        exhaustive_U_complex128_estimated_bytes=16*477638700*9075135300,
        statement="Exact-small complete A1 sampler is implemented. Generic complete neural reduced coefficients still have no scalable exact physical decoder/group action at N36. Reference memory estimate is not a lower bound on all possible algorithms; old sparse-swap failure is reused without a repeat.")
    result=dict(status="completed",exact_small_A1=small_result,N36_native=records,N36_A1_limitation=limitation,
        warm_checkpoint_sha256=file_sha256(checkpoint),warm_checkpoint_bytes=checkpoint.stat().st_size,
        warm_checkpoint_parameter_error=checkpoint_error,expected_parameters=settings["expected_parameters"],
        architecture_changed=False,q_cutoff_added=False,training_steps=0,head_added=False,
        comparison_is_not_paired_Born_audit=True,ledger=ledger,wall_seconds=time.perf_counter()-started)
    atomic_json(output/"results.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--child",action="store_true")
    args=parser.parse_args()
    path=args.config.resolve()
    if args.child:
        jax.config.update("jax_enable_x64",True)
        settings=json.loads(path.read_text())
        try:
            _compute(settings,ROOT/settings["output"])
        except BaseException as error:
            atomic_json(ROOT/settings["output"]/"failure.json",dict(exception=type(error).__name__,
                message=str(error),traceback=traceback.format_exc()))
            raise
    else:
        print(json.dumps(run(path),indent=2))


if __name__=="__main__":
    main()
