"""Local CT6 target-spin development probe; no remote jobs or physical ED.

The actual saved nine/18-site subspaces are frozen. Their projected physical
Hamiltonian includes EVERY 6x6 PBC bond exactly once. Target-root Ritz values
are variational values within this restricted space, not optimized physical
excitation energies, entanglement levels, or a rotor-family identification.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
from math import factorial
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_hnf import C6, hnf_site_permutation
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_gauge import canonicalize_triangular_bank
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, make_mixed_clebsch_amplitude
from su2_nqs.models.triangular_target_root import (make_target_root_amplitude,
    singlet_root_coefficients, target_root_layout, collective_spin_root_coefficients)
from su2_nqs.operators.su2_block_transfer import BlockIsometry, project_block
from su2_nqs.operators.su2_sector_merge import SectorMergeOperator, project_sector_maps, reduced_block_from_projected
from su2_nqs.su2.triangle_blocks import triangle_block_tables
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from .triangle_hierarchy_initialization import source_channels, split_bond_ledger, _node_timeout
from .triangle_sector_transfer import extract_merge_map
from .triangle_transfer_initialization import physical_pair_operator


def run(output: Path, *, seconds_cap: float = 540., only: float = .125,
        amplitude_only: bool = False, collective_seed: bool = False) -> dict:
    """Run a deadline-bounded exact-symmetry/retained-space CT6 probe."""
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("refusing to overwrite sector evidence")
    started = perf_counter(); deadline = started+seconds_cap
    root = Path(__file__).resolve().parents[3]
    records = []
    atomic_json(output/"registration.json", dict(target_q=[0,2,4,6], target_m2="highest weight",
        seconds_cap=seconds_cap, coupling=only, local_only=True, training=False,
        amplitude_only=amplitude_only,
        root_seed="physical (S_A+)**S raw-parent descendant" if collective_seed else "q0 parent; other q fixed random reduced root",
        subtree_changes=False, coherent_A1=False, physical_variance_computed=False,
        workflow_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    for j2 in (only,):
        tag = "j0125" if j2 else "j0"
        path = root/"projects/clebsch_tree_nqs/results/ct_status_20261028/verified_next_parents"/f"ct6_{tag}"/"parent_snapshot/checkpoint.pkl"
        saved = load_checkpoint(path); cfg = MixedClebschConfig(6,28,10)
        mode = "binary_truncated" if "six" in saved["params"] else "ternary_joint"
        bank, gauge = canonicalize_triangular_bank(arity_clebsch_bank(saved["params"],cfg,mode,8),cfg)
        record = dict(j2=j2, parent=str(path.relative_to(root)),
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            global_step=saved["metadata"].get("absolute_completed_steps"), arity=mode,
            gauge=gauge, q_width=28, target_sectors={}, status="running",
            physical_excitation_spectrum=False, entanglement_spectrum=False,
            lattice_quantum_numbers="not resolved: raw frozen subspaces need not be invariant",
            full_A1_energy_known=False, full_physical_variance_known=False)
        records.append(record)
        def persist():
            atomic_json(output/"results.json", dict(records=records, seconds=perf_counter()-started))
        persist()
        rng = np.random.default_rng(2610283101)
        legacy = jax.jit(make_mixed_clebsch_amplitude(cfg))
        images = np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
        transforms = dict(translation=np.asarray(hnf_site_permutation(((6,0),(0,6)),translation=(1,0))),
            C6=np.asarray(hnf_site_permutation(((6,0),(0,6)),operation=C6)))
        for q in (0,2,4,6):
            tick=perf_counter()
            roots = (singlet_root_coefficients(bank,cfg) if q==0 else
                     collective_spin_root_coefficients(bank,cfg,q) if collective_seed else
                     tuple(rng.normal(size=(dl,dr))/np.sqrt(dl*dr) for _,_,dl,dr in target_root_layout(cfg,q)))
            query = jax.jit(jax.vmap(make_target_root_amplitude(cfg,q,q), in_axes=(None,None,0)))
            spins = rng.permutation(np.r_[np.ones(18+q//2,np.int32),np.zeros(18-q//2,np.int32)])
            rows=[spins]
            for i in np.flatnonzero(spins):
                for j in np.flatnonzero(1-spins):
                    row=spins.copy();row[i],row[j]=row[j],row[i];rows.append(row)
            amplitudes=np.asarray(query(bank,roots,np.asarray(rows)))
            casimir=(q*q/4+18-q*(q+2)/4)*amplitudes[0]+amplitudes[1:].sum()
            item=dict(target_q=q,spin=q//2,physical_Casimir_eigenvalue=q*(q+2)/4,
                root_reduced_dimension=sum(dl*dr for _,_,dl,dr in target_root_layout(cfg,q)),
                pointwise_S2_relative_residual=float(abs(casimir)/np.linalg.norm(amplitudes)),
                amplitude_rows=len(rows), first_query_seconds=perf_counter()-tick)
            if q==0:
                old=float(legacy(bank,spins)); item["q0_legacy_relative_replay_error"]=float(abs(amplitudes[0]-old)/max(abs(old),1e-200))
            if q:
                lower=jax.jit(jax.vmap(make_target_root_amplitude(cfg,q,q-2),in_axes=(None,None,0)))
                lowered=[]
                for site in np.flatnonzero(spins):
                    row=spins.copy();row[site]=0;lowered.append(row)
                vals=np.asarray(lower(bank,roots,np.asarray(lowered)))
                item["highest_weight_ladder_relative_residual"]=float(abs(vals.sum()-np.sqrt(q)*amplitudes[0])/
                    max(np.linalg.norm(vals),abs(amplitudes[0]),1e-200))
                if collective_seed:
                    hierarchy=triangular_schur_hierarchy(6)
                    left=hierarchy.coarse_supports[hierarchy.binary_merges[-1][0]]
                    physical_rows=[]
                    for subset in combinations([i for i in left if spins[i]],q//2):
                        value=spins.copy();value[list(subset)]=0;physical_rows.append(value)
                    observed=factorial(q//2)*np.asarray(jax.jit(jax.vmap(
                        make_mixed_clebsch_amplitude(cfg),in_axes=(None,0)))(bank,np.asarray(physical_rows))).sum()
                    item["physical_collective_operator_relative_replay_error"]=float(abs(observed-amplitudes[0])/
                        max(abs(observed),abs(amplitudes[0]),1e-200))
                    item["physical_operator_rows"]=len(physical_rows)
                    item["raw_parent_operator_not_projected_parent_operator"]=True
            item["exact_raw_root_norm_squared"]=float(sum(np.vdot(x,x).real for x in roots))
            tick=perf_counter()
            orbit=np.asarray(query(bank,roots,spins[images]))
            projected=orbit.mean()
            item["coherent_A1_amplitude_probe"]=dict(group_order=len(images),
                amplitude=float(projected),normalization_computed=False,projection_commutes_with_total_spin=True,
                covariance_relative_errors={name:float(abs(np.asarray(query(bank,roots,spins[image][images])).mean()-projected)/
                    max(np.linalg.norm(orbit)/np.sqrt(len(orbit)),1e-200)) for name,image in transforms.items()},
                seconds=perf_counter()-tick,
                energy_and_variance_not_computed=True)
            record["target_sectors"][str(q)]=item;persist()
        if amplitude_only:
            record["status"]="completed amplitude/Casimir/ladder/coherent-A1 probe; no H solve"
            persist();continue
        leaves, joins, bonds = split_bond_ledger(6,j2)
        record["bond_ledger"]=dict(total=len(bonds), leaf_counts=[len(x) for x in leaves],
            merge_counts=[len(x) for x in joins], every_PBC_bond_used_once=True)
        tables=triangle_block_tables()["nine"]
        magnetization=np.asarray([2*i.bit_count()-9 for i in range(512)])
        children=[]
        for i,inside in enumerate(leaves):
            tick=perf_counter()
            h=sum((g*physical_pair_operator(9,a,b) for a,b,g in inside),sparse.csr_matrix((512,512)))
            sectors={}
            for q,w in zip(cfg.local_q,bank["local"][i],strict=True):
                physical=tables[q]@np.asarray(w)
                sectors[q]=np.stack([physical*(magnetization==m)[:,None] for m in range(-q,q+1,2)],axis=-1)
            projected=project_block(BlockIsometry(9,sectors),h.toarray())
            children.append(reduced_block_from_projected(projected))
            record.setdefault("child_projection",[]).append(dict(node=i,seconds=perf_counter()-tick,diagnostics=projected.diagnostics));persist()
        tree=triangular_schur_hierarchy(6)
        for node,(left,right,_) in enumerate(tree.binary_merges[:-1]):
            if perf_counter()>deadline:raise TimeoutError("deadline before18-site projection")
            tick=perf_counter()
            qvalues=range(0,11,2)
            maps={q:extract_merge_map(bank,cfg,node,q,source_channels(children[left].q_widths,children[right].q_widths,q)) for q in qvalues}
            try:
                with _node_timeout(deadline-perf_counter()):
                    child=project_sector_maps(children[left],children[right],joins[node],maps=maps,
                        max_output_dimension=2048,memory_limit_bytes=1536*1024**2)
            except TimeoutError as error:
                record.update(status="bounded incomplete18-site operator propagation",error=str(error))
                save_checkpoint(output/f"{tag}_partial_children.pkl",dict(children=children,bank=bank),metadata=record)
                persist()
                return dict(records=records,seconds=perf_counter()-started)
            children.append(child)
            record.setdefault("child_projection",[]).append(dict(node=node+4,seconds=perf_counter()-tick,dimension=child.dimension));persist()
            save_checkpoint(output/f"{tag}_partial_children.pkl",dict(children=children,bank=bank),metadata=record)
        left,right=children[-2:]
        save_checkpoint(output/f"{tag}_frozen_children.pkl",dict(left=left,right=right,cross_bonds=joins[-1]),metadata=record)
        root0=singlet_root_coefficients(bank,cfg)
        c0=np.concatenate([np.asarray(x).ravel() for x in root0]);c0/=np.linalg.norm(c0)
        op0=SectorMergeOperator(left,right,joins[-1],0,memory_limit_bytes=1536*1024**2)
        record["raw_parent_energy_per_site"]=float(np.vdot(c0,op0.matvec(c0)).real)/36
        persist()
        for q in (0,2,4,6):
            if perf_counter()>deadline-5:
                record["target_sectors"][str(q)]["Ritz_status"]="not started within deadline";persist();continue
            tick=perf_counter()
            op=SectorMergeOperator(left,right,joins[-1],q,memory_limit_bytes=1536*1024**2)
            def action(v):
                if perf_counter()>deadline:raise TimeoutError("bounded sector solve deadline")
                return op.matvec(v)
            operator=LinearOperator((op.dimension,op.dimension),matvec=action,dtype=op.dtype)
            item=record["target_sectors"][str(q)]
            try:
                values,vectors=eigsh(operator,k=3,which="SA",tol=1e-7,maxiter=160,ncv=20,
                    v0=rng.normal(size=op.dimension))
                residuals=np.linalg.norm(np.column_stack([op.matvec(v) for v in vectors.T])-vectors*values,axis=0)
                item.update(Ritz_status="converged in frozen child space",Ritz_total_energies=values.tolist(),
                    Ritz_energies_per_site=(values/36).tolist(),effective_eigen_residuals=residuals.tolist(),
                    residual_is_not_physical_variance=True)
                np.savez_compressed(output/f"{tag}_q{q}_root_eigenvectors.npz",values=values,vectors=vectors,
                    channels=np.asarray(op.channels))
            except Exception as error:
                item.update(Ritz_status="bounded incomplete",error=repr(error))
            item.update(eigensolve_seconds=perf_counter()-tick,operator_diagnostics=op.diagnostics,
                matvec_count=op.matvec_count,matvec_seconds=op.matvec_seconds)
            persist()
        record["status"]="completed bounded probe";persist()
    return dict(records=records,seconds=perf_counter()-started)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seconds-cap",type=float,default=540.)
    parser.add_argument("--j2",type=float,choices=(0.,.125),default=.125)
    parser.add_argument("--amplitude-only",action="store_true")
    parser.add_argument("--collective-seed",action="store_true")
    args=parser.parse_args()
    run(args.output,seconds_cap=args.seconds_cap,only=args.j2,
        amplitude_only=args.amplitude_only,collective_seed=args.collective_seed)
