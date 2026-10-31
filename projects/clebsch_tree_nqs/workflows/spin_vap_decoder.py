"""Bounded saved-VAP compression and energy refinement of a physical CG decoder.

No teacher is added to the state; no spin projector is used in the decoder.
All physical magnetic structure comes from the existing local Schur/CG model.
Orbit sums are exact-small spatial projection, not a large-system sampler.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from su2_nqs.benchmarks.krylov_anchored_full_vit import (
    a1_orbit_reduction, a1_orbit_coordinates, a1_orbit_hamiltonian,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact, load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest, numerical_runtime_identity
from su2_nqs.models.cell_multiplicity_clebsch_tree import (
    SpinCellMultiplicityConfig, init_spin_cell_multiplicity_tree,
    spin_cell_multiplicity_tree_amplitudes, spin_cell_multiplicity_parameter_count,
)
from su2_nqs.vmc.exact_fs_adam import train_exact_fs_adam
from .spin import _prepare_case
from .spin_full_vit_vap import state_metrics

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/"projects/clebsch_tree_nqs/configs/validation/spin_vap_decoder.json"


def decoder_config(rows: int, columns: int, width: int) -> SpinCellMultiplicityConfig:
    """Local four-spin complete Schur cells; all q,m labels remain doubled."""
    if rows % 2 or columns % 2 or rows*columns < 8 or rows*columns > 16:
        raise ValueError("exact-support decoder test needs 8..16 sites and even sides")
    cells = tuple(tuple((r+dr)*columns+c+dc for dr,dc in ((0,0),(0,1),(1,0),(1,1)))
                  for r in range(0,rows,2) for c in range(0,columns,2))
    return SpinCellMultiplicityConfig("square_plaquette", cells, feature_dim=width,
                                     merge_mode="full_joint", target_q=0, target_m2=0)


def init_decoder(seed: int, config: SpinCellMultiplicityConfig) -> dict:
    """Two real exact singlets give complex amplitudes without unsafe dtype casts."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    return {name:init_spin_cell_multiplicity_tree(key,config)
            for name,key in zip(("real","imag"),keys,strict=True)}


def prepare_decoder(data: dict, config: SpinCellMultiplicityConfig) -> dict:
    """Exact-small physical-amplitude contraction and A1 coordinates; never P0."""
    if config.n_sites != data['n_sites']:
        raise ValueError("decoder and physical support differ")
    if 'gamma_a1' not in data['tables']:
        raise ValueError("this torus has no configured Gamma-A1 group")
    basis = jnp.asarray(data['basis'],dtype=jnp.int32)
    a1 = a1_orbit_reduction(data['tables']['gamma_a1'])
    h = a1_orbit_hamiltonian(data['h_rows'],data['h_columns'],data['h_values'],a1)

    def raw(params):
        return (spin_cell_multiplicity_tree_amplitudes(params['real'],basis,config)
                + 1j*spin_cell_multiplicity_tree_amplitudes(params['imag'],basis,config))

    def state_fn(params):
        coordinates = a1_orbit_coordinates(raw(params),a1)
        norm = jnp.linalg.norm(coordinates)
        return coordinates/jnp.where(norm > 1e-30,norm,jnp.nan)

    def physical(vector):
        return np.asarray(vector)[a1.orbit_id]/np.sqrt(a1.orbit_sizes[a1.orbit_id])

    return dict(state_fn=jax.jit(state_fn), raw_fn=jax.jit(raw),
                raw=dict(physical=physical), h=h, a1=a1)


def teacher_objective(teacher: np.ndarray) -> np.ndarray:
    """Hermitian rank-one infidelity loss on normalized exact-support states."""
    vector=np.asarray(teacher,dtype=np.complex128)
    if vector.ndim!=1 or not np.isfinite(vector).all() or np.linalg.norm(vector)<1e-12:
        raise ValueError("teacher must be a finite nonzero vector")
    vector=vector/np.linalg.norm(vector)
    return np.eye(len(vector))-np.outer(vector,vector.conj())


def decoder_metrics(params: dict, problem: dict, data: dict, teacher: np.ndarray,
                    *, cuts: bool=False) -> dict:
    """Full physical operators independently check the normalized decoder image."""
    vector=np.asarray(problem['state_fn'](params))
    record=state_metrics(vector,problem,data,cuts=cuts)
    record['teacher_infidelity']=float(max(0.,1-abs(np.vdot(teacher,vector))**2))
    return record


def validate_decoder(params: dict, problem: dict, data: dict, target: np.ndarray) -> dict:
    """Check raw singlet, A1 projection, operator action, tangents and gradients."""
    raw=np.asarray(problem['raw_fn'](params));raw=raw/np.linalg.norm(raw)
    vector=np.asarray(problem['state_fn'](params));psi=problem['raw']['physical'](vector)
    projected=np.mean(raw[data['tables']['gamma_a1']],axis=0)
    projected/=np.linalg.norm(projected)
    checks=dict(raw_singlet_residual=float(np.linalg.norm(data['spin_squared']@raw)),
        projected_singlet_residual=float(np.linalg.norm(data['spin_squared']@psi)),
        spatial_projection_replay_max=float(np.max(abs(psi-projected))),
        normalization_error=float(abs(np.vdot(psi,psi)-1)),
        H_action_replay_max=float(np.max(abs(data['hamiltonian']@psi-problem['raw']['physical'](problem['h']@vector)))))
    direction=jax.tree_util.tree_map(lambda p:jnp.ones_like(p)*.01,params)
    tangent=np.asarray(jax.jvp(problem['state_fn'],(params,),(direction,))[1])
    checks['tangent_singlet_residual']=float(np.linalg.norm(data['spin_squared']@problem['raw']['physical'](tangent)))
    for name,matrix in [('fit',teacher_objective(target)),('energy',problem['h'])]:
        def loss(p):
            v=problem['state_fn'](p)
            return jnp.vdot(v,jnp.asarray(matrix)@v).real
        derivative=float(jax.jvp(loss,(params,),(direction,))[1])
        shifted=[jax.tree_util.tree_map(lambda p,d:p+sign*1e-5*d,params,direction) for sign in (-1,1)]
        finite=float((loss(shifted[1])-loss(shifted[0]))/2e-5)
        checks[name+'_gradient_error']=abs(derivative-finite)
    if any(value>(2e-7 if name.endswith('gradient_error') else 2e-10)
           for name,value in checks.items()):
        raise RuntimeError(f"decoder invariant failed: {checks}")
    return checks


def tangent_rank(params: dict, state_fn, sector_dimension: int) -> dict:
    """Random-direction exact JVP rank lower bound, removing normalization/phase."""
    started=perf_counter()
    flat,unravel=ravel_pytree(params)
    n_directions=2*sector_dimension+4
    rng=np.random.default_rng(0)
    directions=rng.normal(size=(n_directions,len(flat)))
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    fn=jax.jit(lambda direction:jax.jvp(lambda x:state_fn(unravel(x)),(flat,),(direction,))[1])
    psi=np.asarray(state_fn(params))
    tangent=np.column_stack([np.asarray(fn(jnp.asarray(row))) for row in directions])
    tangent-=psi[:,None]*(psi.conj()@tangent)[None,:]
    singular=np.linalg.svd(np.concatenate((tangent.real,tangent.imag),axis=0),compute_uv=False)
    ranks={str(tol):int(np.count_nonzero(singular>tol*singular[0])) for tol in (1e-6,1e-8,1e-10)}
    return dict(directions=n_directions,physical_projective_dimension=2*sector_dimension-2,
                rank_lower_bounds=ranks,singular_values=singular.tolist(),wall_seconds=perf_counter()-started)


def registration(spec: dict) -> dict:
    """Seal immutable teachers, numerical sources and fixed budgets before fitting."""
    files={}
    for teacher in spec['teachers']:
        for name in ('results.json','states.npz'):
            path=ROOT/spec['teacher_root']/f'{teacher}_vap'/name
            files[str(path.relative_to(ROOT))]=file_sha256(path)
    path=ROOT/spec['teacher_root']/'registration.json'
    files[str(path.relative_to(ROOT))]=file_sha256(path)
    result=dict(config=spec,teacher_files=files,runtime=numerical_runtime_identity(),
                source_manifest=transitive_local_source_manifest((Path(__file__),),root=ROOT,
                    package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    result['registration_sha256']=canonical_sha256(result)
    return result


def run(config_path: Path, teacher_name: str, width: int, *, resume: bool=False) -> dict:
    """Fit then energy-refine one preregistered teacher/width; retain failures too."""
    spec=json.loads(config_path.read_text());output=ROOT/spec['output']
    reg=json.loads((output/'registration.json').read_text())
    if reg!=registration(spec):
        raise ValueError("sealed inputs, numerical sources or runtime changed")
    if teacher_name not in spec['teachers'] or width not in spec['widths']:
        raise ValueError("unregistered teacher/width")
    destination=output/f'{teacher_name}_d{width}'
    if (destination/'results.json').exists():
        raise FileExistsError(destination/'results.json')
    started=perf_counter()
    parent_folder=ROOT/spec['teacher_root']/f'{teacher_name}_vap'
    parent=json.loads((parent_folder/'results.json').read_text())
    if parent['physical_states_sha256']!=file_sha256(parent_folder/'states.npz'):
        raise ValueError("teacher state archive changed")
    data=_prepare_case(parent['case'],'fixed_m0')
    config=decoder_config(data['rows'],data['columns'],width)
    problem=prepare_decoder(data,config)
    with np.load(parent_folder/'states.npz') as d:
        if not np.array_equal(d['basis'],np.asarray(data['basis'])):
            raise ValueError("teacher basis order mismatch")
        teacher_physical=d['projected_terminal'].copy()
    teacher=np.array(a1_orbit_coordinates(teacher_physical,problem['a1']))
    teacher/=np.linalg.norm(teacher)
    teacher_checks=state_metrics(teacher,problem,data)
    if abs(teacher_checks['energy_per_site']-parent['final']['energy_per_site'])>1e-11:
        raise ValueError("teacher energy does not replay")
    seed=spec['decoder_seeds'][spec['teachers'].index(teacher_name)%2]
    initial_params=init_decoder(seed,config)
    checks=validate_decoder(initial_params,problem,data,teacher)
    initial=decoder_metrics(initial_params,problem,data,teacher)
    initial_rank=tangent_rank(initial_params,problem['state_fn'],parent['singlet_a1_dimension'])
    params=initial_params
    stages={};vectors={}
    trajectory=(json.loads((destination/'diagnostics.json').read_text())
                if resume and (destination/'diagnostics.json').exists() else [])
    for stage,matrix in [('fit',teacher_objective(teacher)),('energy',problem['h'])]:
        identity=dict(registration_sha256=reg['registration_sha256'],teacher=teacher_name,width=width,stage=stage)
        manifest=destination/f'{stage}_parameters.json'
        stage_file=destination/f'{stage}_result.json'
        if resume and stage_file.exists():
            params=load_parameter_artifact(manifest,initial_params,expected_identity=identity)
            stages[stage]=json.loads(stage_file.read_text())
        else:
            def callback(p,row):
                metrics=decoder_metrics(p,problem,data,teacher)
                trajectory.append(dict(stage=stage,step=row['step'],**metrics))
                atomic_json(destination/'diagnostics.json',trajectory)
                print(teacher_name,width,stage,row['step'],metrics['teacher_infidelity'],
                      metrics['energy_error_per_site'],flush=True)
            checkpoint=destination/f'{stage}_checkpoint.pkl'
            print(teacher_name,width,stage,'prepared',perf_counter()-started,flush=True)
            params,training=train_exact_fs_adam(params,state_fn=problem['state_fn'],hamiltonian=matrix,
                identity=identity,checkpoint=checkpoint,callback=callback,
                resume=resume and checkpoint.exists(),**spec[stage])
            metrics=decoder_metrics(params,problem,data,teacher,cuts=True)
            save_parameter_artifact(manifest,params,identity=identity,overwrite=resume)
            stages[stage]=dict(training=training,metrics=metrics,params_sha256=pytree_sha256(params),
                objective_meaning='teacher infidelity' if stage=='fit' else 'physical total energy')
            atomic_json(stage_file,stages[stage])
        if stages[stage]['metrics']['singlet_residual_norm']>2e-10 or stages[stage]['metrics']['a1_replay_max']>2e-12:
            raise RuntimeError('trained decoder violates its exact symmetry')
        vectors[stage]=problem['raw']['physical'](np.asarray(problem['state_fn'](params)))
        if stage=='fit':
            fit_rank=tangent_rank(params,problem['state_fn'],parent['singlet_a1_dimension'])
    loaded=load_parameter_artifact(destination/'energy_parameters.json',initial_params,
        expected_identity=dict(registration_sha256=reg['registration_sha256'],teacher=teacher_name,width=width,stage='energy'))
    checks['checkpoint_amplitude_replay_max']=float(np.max(abs(np.asarray(problem['state_fn'](loaded))-np.asarray(problem['state_fn'](params)))))
    state_path=destination/'states.npz';temporary=destination/'states.pending.npz'
    np.savez_compressed(temporary,basis=np.asarray(data['basis'],dtype=np.int8),teacher=teacher_physical,
        initial=problem['raw']['physical'](np.asarray(problem['state_fn'](initial_params))),**vectors)
    temporary.replace(state_path)
    result=dict(teacher=teacher_name,width=width,decoder_seed=seed,case=parent['case'],
        registration_sha256=reg['registration_sha256'],parameter_count=spin_cell_multiplicity_parameter_count(initial_params),
        teacher_parameter_count=parent['parameter_count'],initial=initial,teacher_metrics=teacher_checks,
        stages=stages,trajectory=trajectory,checks=checks,tangent_ranks=dict(initial=initial_rank,fit=fit_rank),
        physical_states_sha256=file_sha256(state_path),
        initial_params_sha256=pytree_sha256(initial_params),wall_seconds=perf_counter()-started,
        exact_SU2=True,exact_A1=True,spin_projection_in_model=False,teacher_added_to_state=False,
        new_ED=False,new_samples=0,N36_training=False,remote_submitted=False)
    result['result_sha256']=canonical_sha256(result)
    atomic_json(destination/'results.json',result)
    print(teacher_name,width,'DONE',result['stages']['energy']['metrics'],flush=True)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--teacher');parser.add_argument('--width',type=int)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    spec=json.loads(args.config.read_text())
    if args.register:
        path=ROOT/spec['output']/'registration.json'
        if path.exists(): raise FileExistsError(path)
        reg=registration(spec);atomic_json(path,reg);print(reg['registration_sha256'])
    else:
        run(args.config,args.teacher,args.width,resume=args.resume)


if __name__=='__main__': main()
