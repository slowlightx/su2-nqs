"""Bounded physical-basis training of the separate triangular ClebschTree.

Reuse complete-A1 amplitudes, full-H products, exchange MCMC, SR-Adam and
atomic optimizer/RNG checkpoints. No ViT, carrier or additional head.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.triangle_clebsch import local_filter_maps,local_spectrum_summary
from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import transitive_local_source_manifest,file_sha256
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig,mixed_clebsch_bank,
    local_c3_bases,make_mixed_clebsch_amplitude,make_mixed_projected_amplitude,mixed_parameter_count,
)
from su2_nqs.models.triangular_arity_clebsch import init_arity_pair,arity_clebsch_bank,binary_to_joint_bank
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.su2.triangle_blocks import triangle_block_tables
from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.vmc.checkpoint import save_checkpoint,load_checkpoint
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains,run_exchange_sweeps
from su2_nqs.vmc.physical_sr_adam import make_physical_sr_adam

ROOT=Path(__file__).resolve().parents[3]
jax.config.update('jax_enable_x64',True)


def load_sealed_cost_family(cost: Path,gate: dict,expected_hashes: dict[str,str]) -> dict:
    """Use the ACTUAL frozen arrays, not bitwise replay of GPU contractions.

    Initial random keys do not guarantee bit-identical derived GEMM tensors
    across devices/compiler choices. File hashes and the original passed
    physical-amplitude gate instead identify the exact serialized parents.
    """
    family={}
    for arm in ('binary_free','ternary_free','ternary_c3'):
        path=cost/(arm+'.pkl')
        if file_sha256(path)!=expected_hashes[arm]:raise ValueError('frozen cost checkpoint changed: '+arm)
        saved=load_checkpoint(path)
        if (saved['metadata']['registration_sha256']!=gate['registration_sha256']
                or mixed_parameter_count(saved['params'])!=gate['arms'][arm]['real_parameter_count']):
            raise ValueError('cost checkpoint registration/layout mismatch: '+arm)
        family[arm]=saved['params']
    return family


def filtered_initial_family(family: dict,config: MixedClebschConfig,*,j2: float,beta: float) -> tuple[dict,float]:
    """One local SU2/C3 scalar initialization, identical in all three arms.

    Re-expand the filtered C3 map in its independent orthonormal coordinates.
    This changes initialization only, not tensor capacity or the wavefunction
    definition. No spin sector is set to zero and beta is never optimized.
    """
    constrained=family['ternary_c3']; bank=mixed_clebsch_bank(constrained,config,'ternary_c3')
    filters=local_filter_maps(config.length,j2,beta); bases=local_c3_bases(config)
    nodes=[]; error=0.
    for weights,maps in zip(bank['local'],filters,strict=True):
        filtered=[maps[q]@np.asarray(w) for q,w in zip(config.local_q,weights,strict=True)]
        # A common scalar for the whole block does not change relative q weights.
        norm=np.sqrt(sum(np.sum(w*w) for w in filtered)); coefficients=[]
        for basis,w in zip(bases,filtered,strict=True):
            target=w/norm; c=np.einsum('pik,ik->p',basis,target)
            error=max(error,float(np.max(abs(np.einsum('p,pik->ik',c,basis)-target))))
            coefficients.append(jnp.asarray(c))
        nodes.append(tuple(coefficients))
    if error>2e-12:raise ArithmeticError('local filter left the C3 intertwiner space')
    constrained=dict(constrained,local=tuple(nodes))
    common=mixed_clebsch_bank(constrained,config,'ternary_c3')
    free=dict(family['ternary_free'],local=common['local'],top=constrained['top'])
    overlaps=triangle_block_tables()['binary_overlaps']
    binary=dict(family['binary_free'],top=constrained['top'],local=tuple(
        tuple(jnp.asarray(overlaps[q])@w for q,w in zip(config.local_q,node,strict=True))
        for node in common['local']))
    return dict(binary_free=binary,ternary_free=free,ternary_c3=constrained),error


def checked_trained_parent(spec: dict,request: dict) -> tuple[dict,dict,str]:
    """Read a sealed same-H endpoint without modifying its optimizer or RNG."""
    directory=Path(request['directory'])
    registration=json.loads((directory/'registration.json').read_text())
    seal=registration['sha256']
    if (seal!=request['registration_sha256'] or seal!=canonical_sha256({
            k:v for k,v in registration.items() if k!='sha256'})):
        raise ValueError('trained parent registration changed')
    parent_spec=registration['config']
    if (parent_spec['arm'] not in ('ternary_c3','ternary_free') or parent_spec['model']!=spec['model']
            or parent_spec['j2_over_j1']!=spec['j2_over_j1']):
        raise ValueError('trained parent must be same-model/H free or C3 endpoint')
    checkpoint=directory/'checkpoint.pkl'
    if file_sha256(checkpoint)!=request['checkpoint_sha256']:
        raise ValueError('trained parent checkpoint changed')
    saved=load_checkpoint(checkpoint); meta=saved['metadata']; p=saved['params']
    if (meta['registration_sha256']!=seal or meta['params_sha256']!=pytree_sha256(p)
            or meta['params_sha256']!=request['params_sha256']
            or meta['completed_steps']!=request['completed_steps']
            or meta['completed_steps']!=len(meta['history'])
            or int(saved['optimizer_state'][0])!=meta['optimizer_step']
            or pytree_sha256(saved['optimizer_state'])!=meta['optimizer_sha256']
            or pytree_sha256((jnp.asarray(meta['rng']),jnp.asarray(meta['sampler_states'])))!=meta['sampler_rng_sha256']):
        raise ValueError('trained parent state/optimizer/RNG identity mismatch')
    audit_path=directory/'terminal_audit.json'
    if file_sha256(audit_path)!=request['audit_sha256']:
        raise ValueError('trained parent audit file changed')
    audit=json.loads(audit_path.read_text())
    if audit['registration_sha256']!=seal or audit['params_sha256']!=meta['params_sha256']:
        raise ValueError('trained parent audit mismatch')
    return p,audit,parent_spec['arm']


def trained_parent_family(family: dict,config: MixedClebschConfig,spec: dict) -> tuple[dict,dict]:
    """Expand one C3 state, preserving its amplitude but not tangent/SR metric."""
    request=spec['trained_parent']
    p,audit,parent_mode=checked_trained_parent(spec,request)
    if parent_mode!='ternary_c3':
        raise ValueError('the three-arm historical comparison needs a C3 parent')
    common=mixed_clebsch_bank(p,config,'ternary_c3')
    free=dict(family['ternary_free'],local=common['local'],top=p['top'])
    free_bank=mixed_clebsch_bank(free,config,'ternary_free')
    if pytree_sha256(common)!=pytree_sha256(free_bank):
        raise ArithmeticError('same-parent free expansion changed the effective physical bank')
    overlaps=triangle_block_tables()['binary_overlaps']
    binary=dict(family['binary_free'],top=p['top'],local=tuple(
        tuple(jnp.asarray(overlaps[q])@w for q,w in zip(config.local_q,node,strict=True))
        for node in common['local']))
    note=dict(parent=request,parent_terminal_audit=audit,
        common_physical_bank_sha256=pytree_sha256(common),
        optimizer_reset=True,sampler_reset=True,rng_reset=True,
        amplitude_preserving=True,tangent_spaces_identical=False,
        interpretation='Same physical parent; C3 restriction versus free reduced-tensor directions, not identical SR geometry.')
    return dict(binary_free=binary,ternary_free=free,ternary_c3=p),note


def arity_parent_family(config: MixedClebschConfig,spec: dict) -> tuple[dict,dict]:
    """One compressed parent lifted into a true joint node and a binary TTN.

    BOTH children require new initial audits: truncating the parent's six-site
    edge generally changes its physical state. Never reuse the parent's audit
    as if it measured this common compressed initialization.
    """
    request=spec['arity_comparison']['parent']
    p,audit,parent_mode=checked_trained_parent(spec,request)
    bank=mixed_clebsch_bank(p,config,parent_mode)
    family,note=init_arity_pair(bank,config,spec['arity_comparison']['binary_rank'])
    counts={a:mixed_parameter_count(z) for a,z in family.items()}
    mismatch=abs(counts['binary_truncated']/counts['ternary_joint']-1.)
    if mismatch>spec['arity_comparison']['maximum_parameter_mismatch_fraction']:
        raise ValueError('binary/ternary independent parameter budgets are not matched')
    local_counts={a:mixed_parameter_count({k:v for k,v in z.items() if k!='top'})
                  for a,z in family.items()}
    local_mismatch=abs(local_counts['binary_truncated']/local_counts['ternary_joint']-1.)
    if local_mismatch>spec['arity_comparison'].get('maximum_local_parameter_mismatch_fraction',1.):
        raise ValueError('binary/ternary LOCAL parameter budgets are not matched')
    note.update(parent=request,parent_terminal_audit=audit,
        parent_mode=parent_mode,optimizer_reset=True,sampler_reset=True,rng_reset=True,
        fresh_initial_audit_required=True,physical_parent_changed_by_truncation=True,
        parameter_counts=counts,parameter_mismatch_fraction=mismatch,
        local_parameter_counts=local_counts,local_parameter_mismatch_fraction=local_mismatch)
    return family,note


def checked_arity_endpoint(config: MixedClebschConfig,spec: dict) -> tuple[dict,dict,dict]:
    """Restore a sealed finite-binary/joint endpoint, with explicit H policy.

    Same-H continuation preserves optimizer, RNG, chains and history. A
    same-H common-binary restart lifts ONE frozen physical state exactly
    into both representations and resets optimizer/RNG/chains together. A
    Hamiltonian transfer starts both arms from the same trained binary
    physical state, exactly lifted into the unrestricted joint bank, and
    resets optimizer/RNG/chains. Its source-H audit is not a target-H audit.
    """
    request=spec['arity_endpoint_parent'];directory=Path(request['directory'])
    registration=json.loads((directory/'registration.json').read_text())
    seal=registration['sha256']
    if (seal!=request['registration_sha256'] or seal!=canonical_sha256({
            k:v for k,v in registration.items() if k!='sha256'})):
        raise ValueError('arity endpoint registration changed')
    parent_spec=registration['config'];mode=parent_spec['arm']
    rank=spec['arity_comparison']['binary_rank']
    if (mode not in ('binary_truncated','ternary_joint') or parent_spec['model']!=spec['model']
            or parent_spec['arity_comparison']['binary_rank']!=rank):
        raise ValueError('arity endpoint model/rank mismatch')
    path=directory/'checkpoint.pkl'
    if file_sha256(path)!=request['checkpoint_sha256']:
        raise ValueError('arity endpoint checkpoint changed')
    saved=load_checkpoint(path);meta=saved['metadata'];p=saved['params']
    if (meta['registration_sha256']!=seal or meta['params_sha256']!=pytree_sha256(p)
            or meta['params_sha256']!=request['params_sha256']
            or meta['completed_steps']!=request['completed_steps']
            or meta['completed_steps']!=len(meta['history'])
            or int(saved['optimizer_state'][0])!=meta['optimizer_step']
            or pytree_sha256(saved['optimizer_state'])!=meta['optimizer_sha256']
            or pytree_sha256((jnp.asarray(meta['rng']),jnp.asarray(meta['sampler_states'])))!=meta['sampler_rng_sha256']):
        raise ValueError('arity endpoint state/optimizer/RNG identity mismatch')
    audit_path=directory/'terminal_audit.json'
    if file_sha256(audit_path)!=request['audit_sha256']:
        raise ValueError('arity endpoint audit changed')
    audit=json.loads(audit_path.read_text())
    if audit['registration_sha256']!=seal or audit['params_sha256']!=meta['params_sha256']:
        raise ValueError('arity endpoint audit identity mismatch')
    protocol=spec['endpoint_protocol'];source_j2=parent_spec['j2_over_j1'];target_j2=spec['j2_over_j1']
    selected=spec['arm'];reference_bank=arity_clebsch_bank(p,config,mode,rank)
    if protocol=='same_H_full_state_continuation':
        if source_j2!=target_j2 or selected!=mode:
            raise ValueError('full-state continuation requires the same H and representation')
        if spec['steps']!=meta['completed_steps']+spec['additional_steps']:
            raise ValueError('continuation endpoint must preserve its registered additional-step budget')
        family={selected:p};restore=saved
        note=dict(optimizer_reset=False,sampler_reset=False,rng_reset=False,
            fresh_initial_audit_required=False,parent_history_preserved=True)
    elif protocol=='new_H_common_binary_state':
        transfer=spec['hamiltonian_transfer']
        if (mode!='binary_truncated' or source_j2==target_j2
                or transfer!={'source_j2':source_j2,'target_j2':target_j2}
                or spec['steps']!=spec['additional_steps']):
            raise ValueError('new-H transfer requires a declared binary source and fresh step budget')
        family={'binary_truncated':p,'ternary_joint':binary_to_joint_bank(p,config,rank)}
        restore={};note=dict(optimizer_reset=True,sampler_reset=True,rng_reset=True,
            fresh_initial_audit_required=True,parent_history_preserved=False,
            source_H_audit_is_not_target_H_audit=True)
    elif protocol=='same_H_common_binary_state':
        if (mode!='binary_truncated' or source_j2!=target_j2
                or spec['steps']!=spec['additional_steps'] or 'hamiltonian_transfer' in spec):
            raise ValueError('same-H common restart requires a binary source, unchanged H and fresh step budget')
        family={'binary_truncated':p,'ternary_joint':binary_to_joint_bank(p,config,rank)}
        restore={};note=dict(optimizer_reset=True,sampler_reset=True,rng_reset=True,
            fresh_initial_audit_required=False,parent_history_preserved=False,
            same_physical_initial_state_between_arity_arms=True,
            source_H_audit_is_not_target_H_audit=False)
    else:
        raise ValueError('unknown arity endpoint protocol')
    note.update(parent=request,parent_terminal_audit=audit,endpoint_protocol=protocol,
        source_j2=source_j2,target_j2=target_j2,source_arm=mode,
        reference_bank_sha256=pytree_sha256(reference_bank),amplitude_preserving=True,
        same_initial_state_between_couplings=False)
    return family,note,dict(saved=restore,reference_bank=reference_bank)


def checked_target_H_cost_certificate(spec: dict) -> bool:
    """Reuse a SEALED independent target-H preflight, never a relabelled H.

    Numerical dependencies must still match the original passed cost gate.
    A fresh full-A1/S-plus/Sz-connectivity oracle is also run in ``train``
    before updates. This certificate only permits that initial preflight.
    """
    request=spec.get('target_H_cost_certificate')
    if request is None:return False
    directory=Path(request['directory'])
    reg=json.loads((directory/'registration.json').read_text())
    path=directory/'preflight.json'
    if (reg['sha256']!=request['registration_sha256']
            or reg['sha256']!=canonical_sha256({k:v for k,v in reg.items() if k!='sha256'})
            or file_sha256(path)!=request['preflight_sha256']):
        raise ValueError('target-H cost certificate seal changed')
    preflight=json.loads(path.read_text());config=reg['config']
    if (config['model']!=spec['model'] or config['j2_over_j1']!=spec['j2_over_j1']
            or preflight['registration_sha256']!=reg['sha256']
            or preflight['status']!='passed' or not preflight['target_H_independent_oracle']
            or not np.isfinite(preflight['validation']['target_H_oracle_relative_error'])
            or preflight['validation']['target_H_oracle_relative_error']>2e-11
            or request['directory']!=spec['arity_endpoint_parent']['directory']):
        raise ValueError('target-H cost certificate does not certify this frozen parent/model/H')
    return True


def checked_declared_H_transfer(spec: dict) -> bool:
    """Validate the ACTUAL sealed endpoint Hamiltonian for a warm start.

    The original cost gate establishes the contraction/model implementation,
    not a permanent source coupling. A later audited endpoint can have a
    different Hamiltonian from that original gate. Its own sealed source H
    must match the explicit transfer declaration; the target H still needs
    the independent full projected-amplitude/Sz-connectivity oracle below.
    Checkpoint, optimizer, RNG and audit identities are additionally checked
    by ``checked_arity_endpoint`` before any initialization is accepted.
    """
    if ('arity_endpoint_parent' not in spec
            or spec.get('endpoint_protocol') != 'new_H_common_binary_state'):
        return False
    request = spec['arity_endpoint_parent']
    registration = json.loads((Path(request['directory'])/'registration.json').read_text())
    seal = registration['sha256']
    if (seal != request['registration_sha256']
            or seal != canonical_sha256({k: v for k, v in registration.items() if k != 'sha256'})):
        raise ValueError('Hamiltonian transfer endpoint registration changed')
    parent = registration['config']
    source = parent['j2_over_j1']
    target = spec['j2_over_j1']
    if (parent['model'] != spec['model'] or parent['arm'] != 'binary_truncated'
            or source == target
            or spec.get('hamiltonian_transfer') != {'source_j2': source, 'target_j2': target}):
        raise ValueError('Hamiltonian transfer must declare the actual sealed source and target H')
    return True


def check_arity_pilot(output: Path,minimum_step: int) -> None:
    """Require a replayed bounded pilot without modifying any checkpoint.

    This checks update safety, not energy improvement from two noisy steps.
    """
    result=json.loads((output/'results.json').read_text())
    preflight=json.loads((output/'preflight.json').read_text())
    if (result['completed_steps']<minimum_step or preflight['status']!='passed'
            or not result['checkpoint_rng_sampler_replay_exact']
            or not result['checkpoint_optimizer_exact']):
        raise ValueError('bounded arity pilot or checkpoint replay did not pass')
    for row in result['history'][-2:]:
        if (not row['accepted'] or not np.isfinite(row['energy_before_update_per_site'])
                or abs(row['energy_before_update_per_site'])>3
                or row['FS_sine']>.025*(1.+1e-10)
                or row['ESS_fraction']<.95):
            raise ValueError('bounded arity pilot has an unsafe physical update')


def resolve_arm_spec(spec: dict,arm: str | None=None) -> dict:
    """Select an arm without duplicating common training/budget configuration.

    Only representation, independent terminal-audit seed and output path may
    differ. An override cannot silently change samples, optimizer or steps.
    """
    selected=dict(spec);overrides=selected.pop('arm_overrides',None)
    if overrides is None:
        if arm is not None:raise ValueError('this config has no registered arm overrides')
        return selected
    arm=selected['arm'] if arm is None else arm
    if arm not in overrides:raise ValueError('unregistered comparison arm')
    if set(overrides[arm])-{'arm','terminal_audit_seed','output'}:
        raise ValueError('arm override cannot change shared training conditions')
    if overrides[arm].get('arm')!=arm:raise ValueError('arm override identity mismatch')
    return dict(selected,**overrides[arm])


def train(spec: dict,output: Path,*,session_seconds: float=3300.,until_step: int | None=None) -> dict:
    """Train a preregistered finite budget and audit only independent chains."""
    if spec['require_gpu'] and jax.default_backend()!='gpu':raise RuntimeError('allocated GPU required')
    cost=Path(spec['cost_parent']); gate=json.loads((cost/'results.json').read_text())
    parent_reg=json.loads((cost/'registration.json').read_text())
    same_h=gate['config']['j2_over_j1']==spec['j2_over_j1']
    declared_transfer=checked_declared_H_transfer(spec)
    certified_target=checked_target_H_cost_certificate(spec)
    if gate['status']!='passed' or gate['config']['model']!=spec['model'] or not (same_h or declared_transfer or certified_target):
        raise ValueError('passed same-model, same-H cost gate required')
    if parent_reg['sha256']!=canonical_sha256({k:v for k,v in parent_reg.items() if k!='sha256'}):
        raise ValueError('cost registration changed')
    for name,digest in parent_reg['sources'].items():
        if file_sha256(ROOT/name)!=digest:raise ValueError('passed numerical source changed: '+name)
    cfg=MixedClebschConfig(**spec['model']); n=cfg.length**2; mode=spec['arm']
    family=load_sealed_cost_family(cost,gate,spec['cost_checkpoint_sha256'])
    initialization=None;filter_error=0.;endpoint={}
    if 'arity_comparison' in spec:
        if 'trained_parent' in spec:
            raise ValueError('choose one initialization protocol, not both')
        if 'arity_endpoint_parent' in spec:
            family,initialization,endpoint=checked_arity_endpoint(cfg,spec)
        else:
            family,initialization=arity_parent_family(cfg,spec)
        rank=spec['arity_comparison']['binary_rank']
        bank_fn=lambda z:arity_clebsch_bank(z,cfg,mode,rank)
    else:
        family,filter_error=filtered_initial_family(family,cfg,j2=spec['j2_over_j1'],beta=spec['initialization_beta'])
        if 'trained_parent' in spec:
            family,initialization=trained_parent_family(family,cfg,spec)
        bank_fn=lambda z:mixed_clebsch_bank(z,cfg,mode)
    p=family[mode]; moments=_manual_adam_init(p)
    images=np.asarray(triangular_hnf_gamma_a1_group(((cfg.length,0),(0,cfg.length))),np.int32)
    projected=make_mixed_projected_amplitude(cfg,images,chunk=spec['projection_chunk'])
    def amp(params,bits):return projected(bank_fn(params),bits)
    bonds=triangular_j1j2_bonds(cfg.length,cfg.length,j2=spec['j2_over_j1'],periodic=True)
    nearest=[(a,b) for a,b,_ in triangular_j1j2_bonds(cfg.length,cfg.length,j2=0.,periodic=True)]
    h=make_amplitude_heisenberg_action(projected,bonds,n_sites=n)
    @jax.jit
    def action(params,batch):
        bank=bank_fn(params)
        psi,hp=jax.lax.map(lambda rows:h(bank,rows),batch.reshape(-1,2,n))
        return psi.reshape(-1),hp.reshape(-1)
    def sample(params,key,states,proposals):
        bank=bank_fn(params)
        return run_exchange_sweeps(bank,key,states,amplitude_fn=projected,particle_kind='spin',
            sweeps=proposals,exchange_edges=nearest)
    sample=jax.jit(sample,static_argnames=('proposals',))
    update=make_physical_sr_adam(amp,p,damping=spec['damping'],score_chunk=2,
        maximum_fs_sine=spec['maximum_FS_sine'])
    # Include the cost entry's independent oracle dependencies as well: its
    # complete source manifest is checked before any training is allowed.
    sources=transitive_local_source_manifest((Path(__file__),ROOT/'su2_nqs/vmc/local_energy.py',
        ROOT/'projects/clebsch_tree_nqs/workflows/spin_triangular_decoder_cost.py'),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    identity=dict(config=spec,sources=sources,cost_registration_sha256=gate['registration_sha256'])
    seal=canonical_sha256(identity); output.mkdir(parents=True,exist_ok=True)
    registration=dict(identity,sha256=seal)
    if (output/'registration.json').exists():
        if json.loads((output/'registration.json').read_text())!=registration:raise ValueError('training registration changed')
    else:
        atomic_json(output/'registration.json',registration)
        atomic_json(output/'local_spectrum.json',local_spectrum_summary(cfg.length,spec['j2_over_j1']))
        if initialization is not None:atomic_json(output/'initialization.json',initialization)
    key,init_key=jax.random.split(jax.random.PRNGKey(spec['training_seed']))
    states=initialize_spin_exchange_chains(init_key,n_chains=spec['chains'],n_sites=n,n_up=n//2)
    history=[]; completed=0
    restored_parent=endpoint.get('saved',{})
    if restored_parent:
        meta=restored_parent['metadata'];moments=restored_parent['optimizer_state']
        key=jnp.asarray(meta['rng']);states=jnp.asarray(meta['sampler_states'])
        history=list(meta['history']);completed=meta['completed_steps']
    start_parent_step=completed
    if 'arity_comparison' in spec and not (output/'preflight.json').exists():
        # New parameterization, SAME established physical/H kernel. Check the
        # actual pair and time complete operations before spending a trajectory.
        rank=spec['arity_comparison']['binary_rank'];timing={};validation={}
        compiled_bank=jax.jit(bank_fn)
        def timed(name,fn,*args):
            start=perf_counter();value=jax.block_until_ready(fn(*args));cold=perf_counter()-start
            start=perf_counter();jax.block_until_ready(fn(*args));warm=perf_counter()-start
            timing[name]=dict(first_call_seconds=cold,warm_seconds=warm)
            return value
        b=timed('bank_generation',compiled_bank,p)
        if endpoint:
            b2=endpoint['reference_bank']
        else:
            other='ternary_joint' if mode=='binary_truncated' else 'binary_truncated'
            b2=arity_clebsch_bank(family[other],cfg,other,rank)
        validation['common_initial_bank_max_error']=max(float(jnp.max(abs(x-y)))
            for x,y in zip(jax.tree.leaves(b),jax.tree.leaves(b2),strict=True))
        x=states[0]
        evaluate=jax.jit(jax.vmap(projected,in_axes=(None,0)))
        values=timed('two_complete_A1_amplitudes',evaluate,b,states[:2])
        transformed=projected(b,x[images[37]])
        validation['A1_relative_error']=float(abs(values[0]-transformed)/jnp.maximum(abs(values[0]),1e-100))
        raw=make_mixed_clebsch_amplitude(cfg)
        bra=np.asarray(x).copy();bra[np.flatnonzero(bra==0)[0]]=1
        lowered=np.tile(bra,(n//2+1,1));lowered[np.arange(n//2+1),np.flatnonzero(bra)]=0
        ward=jax.jit(jax.vmap(raw,in_axes=(None,0)))(b,jnp.asarray(lowered))
        validation['S_plus_relative_error']=float(abs(ward.sum())/jnp.maximum(jnp.linalg.norm(ward),1e-100))
        a,hp=timed('two_complete_H_centers',action,p,states[:2])
        validation['H_finite']=bool(np.isfinite(np.asarray(hp)).all() and np.isfinite(np.asarray(a)).all())
        if endpoint:
            # Independently combine Sz connectivity with FULL projected
            # amplitudes. This is not ED and is especially required when H
            # changes: a source-H timing gate cannot certify the target H.
            graph=SzHeisenbergGraph(n,tuple(bonds),n_up=n//2)
            oracle=[];oracle_started=perf_counter()
            evaluate_rows=jax.jit(lambda rows:jax.lax.map(lambda y:projected(b,y),rows))
            for row in np.asarray(states[:2]):
                connected,elements=graph.get_conn(tuple(row))
                oracle.append(np.dot(elements,np.asarray(evaluate_rows(jnp.asarray(connected)))))
            oracle=np.asarray(oracle)
            validation['target_H_oracle_relative_error']=float(np.linalg.norm(np.asarray(hp)-oracle)
                /max(np.linalg.norm(oracle),1e-100))
            timing['independent_target_H_oracle_seconds']=perf_counter()-oracle_started
        loss=lambda z:jnp.log(jnp.abs(amp(z,x)))
        gradient=timed('one_complete_parameter_gradient',jax.jit(jax.grad(loss)),p)
        validation['gradient_finite']=all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(gradient))
        memory=[d.memory_stats() or {} for d in jax.local_devices()]
        passed=(max(validation[k] for k in ('common_initial_bank_max_error','A1_relative_error','S_plus_relative_error'))<=2e-11
                and validation.get('target_H_oracle_relative_error',0.)<=2e-11
                and validation['H_finite'] and validation['gradient_finite'])
        preflight=dict(status='passed' if passed else 'failed',validation=validation,timing=timing,registration_sha256=seal,
            group_order=len(images),physical_bonds=len(bonds),full_H_calls=6 if endpoint else 4,
            cost_gate_same_H=same_h,target_H_independent_oracle=bool(endpoint),
            peak_device_bytes=max(int(m.get('peak_bytes_in_use',0)) for m in memory),
            calibration_rows_are_not_Born_energy_samples=True)
        atomic_json(output/'preflight.json',preflight)
        if not passed:
            raise ArithmeticError('new finite-rank binary/joint ternary preflight failed')
    if 'arity_comparison' in spec:
        preflight=json.loads((output/'preflight.json').read_text())
        if preflight['status']!='passed' or preflight['registration_sha256']!=seal:
            raise ValueError('failed or mismatched arity preflight cannot authorize training')
    def audit(parameters: dict,seed: int,name: str) -> dict:
        if (output/(name+'.json')).exists():
            record=json.loads((output/(name+'.json')).read_text())
            if record['registration_sha256']!=seal or record['params_sha256']!=pytree_sha256(parameters):
                raise ValueError('frozen audit belongs to a different endpoint')
            return record
        ka,ki=jax.random.split(jax.random.PRNGKey(seed))
        xa=initialize_spin_exchange_chains(ki,n_chains=spec['audit_chains'],n_sites=n,n_up=n//2)
        xa,_,ka,_=sample(parameters,ka,xa,proposals=spec['audit_burn_proposals']); rows=[]
        samples=spec.get('initial_audit_samples',spec['audit_samples']) if name=='initial_audit' else spec['audit_samples']
        if samples % spec['audit_chains']:
            raise ValueError('audit samples must be a multiple of independent chains')
        for _ in range(samples//spec['audit_chains']):
            xa,_,ka,_=sample(parameters,ka,xa,proposals=spec['audit_spacing_proposals']);rows.append(xa)
        batch=jnp.stack(rows).reshape(-1,n);psi,hp=action(parameters,batch)
        e=np.asarray(hp/psi).reshape(-1,spec['audit_chains'])
        if not np.isfinite(e).all():raise FloatingPointError('nonfinite full-H audit ratios')
        mean=float(e.mean());var=float(e.var()/n)
        leave=np.asarray([np.delete(e,c,axis=1).var()/n for c in range(spec['audit_chains'])])
        record=dict(energy_per_site=mean/n,energy_chain_SE=float(e.mean(0).std(ddof=1)/np.sqrt(spec['audit_chains'])/n),
            variance_per_site=var,variance_chain_jackknife_SE=float(np.sqrt((len(leave)-1)/len(leave)*np.sum((leave-leave.mean())**2))),
            seed=seed,samples=e.size,chains=spec['audit_chains'],params_sha256=pytree_sha256(parameters),
            independent_from_training=True,mixing_certified=False,registration_sha256=seal)
        np.savez_compressed(output/(name+'.npz'),bits=np.asarray(batch),local_energies=e.reshape(-1),chain_labels=np.tile(np.arange(spec['audit_chains']),len(e)))
        atomic_json(output/(name+'.json'),record);print(name,json.dumps(record),flush=True);return record
    if (output/'checkpoint.pkl').exists():
        saved=load_checkpoint(output/'checkpoint.pkl');meta=saved['metadata']
        if meta['registration_sha256']!=seal or meta['params_sha256']!=pytree_sha256(saved['params']):raise ValueError('checkpoint identity mismatch')
        p=saved['params'];moments=saved['optimizer_state'];key=jnp.asarray(meta['rng']);states=jnp.asarray(meta['sampler_states'])
        history=meta['history'];completed=meta['completed_steps']
        if (completed!=len(history) or int(moments[0])!=meta['optimizer_step']
                or pytree_sha256(moments)!=meta['optimizer_sha256']
                or pytree_sha256((key,states))!=meta['sampler_rng_sha256']):
            raise ValueError('checkpoint optimizer/history/sampler mismatch')
    else:
        if initialization is None or initialization.get('fresh_initial_audit_required',False):
            audit(p,spec['initial_audit_seed'],'initial_audit')
        else:
            # Reuse the frozen, independent parent audit; do not relabel it
            # as a fresh child sample or spend the pilot budget auditing twice.
            atomic_json(output/'parent_audit.json',initialization['parent_terminal_audit'])
        if not restored_parent:
            states,_,key,_=sample(p,key,states,proposals=spec['burn_proposals'])
    stop=[False]
    handlers={s:signal.getsignal(s) for s in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT)}
    def request_stop(signum,frame):stop[0]=True
    for s in handlers:signal.signal(s,request_stop)
    def save():
        save_checkpoint(output/'checkpoint.pkl',p,optimizer_state=moments,metadata=dict(
            registration_sha256=seal,params_sha256=pytree_sha256(p),completed_steps=completed,
            optimizer_step=int(moments[0]),optimizer_sha256=pytree_sha256(moments),
            sampler_rng_sha256=pytree_sha256((key,states)),history=history,
            rng=np.asarray(key),sampler_states=np.asarray(states)))
    started=perf_counter();session_start_step=completed;save()
    try:
        limit=spec['steps'] if until_step is None else min(until_step,spec['steps'])
        while completed<limit and perf_counter()-started<session_seconds and not stop[0]:
            tick=perf_counter();rows=[]
            for _ in range(spec['samples']//spec['chains']):
                states,_,key,_=sample(p,key,states,proposals=spec['spacing_proposals']);rows.append(states)
            batch=jnp.stack(rows).reshape(-1,n);psi,hp=action(p,batch);e=hp/psi
            if not np.isfinite(np.asarray(e)).all():raise FloatingPointError('nonfinite training ratios')
            p,moments,info=update(p,moments,batch,e,spec['learning_rate'],warm=True)
            jax.block_until_ready(p);completed+=1
            history.append(dict(step=completed,energy_before_update_per_site=float(e.mean())/n,
                variance_before_update_per_site=float(e.var())/n,full_H_calls=len(batch),seconds=perf_counter()-tick,
                energy_min_per_site=float(e.min())/n,energy_max_per_site=float(e.max())/n,
                top_one_percent_residual_weight=float(jnp.sort(abs(e-e.mean())**2)[-max(1,len(e)//100):].sum()/jnp.maximum(jnp.sum(abs(e-e.mean())**2),1e-300)),
                **{k:np.asarray(v).item() for k,v in info.items()}))
            save();atomic_json(output/'progress.json',dict(completed_steps=completed,last=history[-1]))
            print(json.dumps(history[-1]),flush=True)
        result=dict(status='fixed_steps_complete' if completed==spec['steps'] else 'safe_pause',
            completed_steps=completed,history=history,parameter_count=mixed_parameter_count(p),
            filter_reconstruction_error=filter_error,registration_sha256=seal,training_moments_not_audit=True,
            start_parent_step=start_parent_step,session_start_step=session_start_step,
            newly_completed_session_steps=completed-session_start_step,
            session_training_seconds=perf_counter()-started)
        saved=load_checkpoint(output/'checkpoint.pkl');meta=saved['metadata']
        actual=sample(p,key,states,proposals=2)
        replay=sample(saved['params'],jnp.asarray(meta['rng']),jnp.asarray(meta['sampler_states']),proposals=2)
        result['checkpoint_rng_sampler_replay_exact']=bool(np.array_equal(np.asarray(actual[0]),np.asarray(replay[0])) and np.array_equal(np.asarray(actual[2]),np.asarray(replay[2])))
        result['checkpoint_optimizer_exact']=pytree_sha256(saved['optimizer_state'])==pytree_sha256(moments)
        if not result['checkpoint_rng_sampler_replay_exact'] or not result['checkpoint_optimizer_exact']:
            raise ArithmeticError('trained checkpoint replay failed')
        if completed==spec['steps']:
            result['terminal_audit']=audit(p,spec['terminal_audit_seed'],'terminal_audit')
        atomic_json(output/'results.json',result);return result
    finally:
        for s,handler in handlers.items():signal.signal(s,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--session-seconds',type=float,default=2400.);parser.add_argument('--until-step',type=int)
    parser.add_argument('--arm',choices=('binary_truncated','ternary_joint'))
    parser.add_argument('--require-pilot-step',type=int)
    args=parser.parse_args();spec=resolve_arm_spec(json.loads(args.config.read_text()),args.arm)
    if args.require_pilot_step is not None:check_arity_pilot(Path(spec['output']),args.require_pilot_step)
    result=train(spec,Path(spec['output']),session_seconds=args.session_seconds,until_step=args.until_step)
    print(result['status'],result['completed_steps'],flush=True)
