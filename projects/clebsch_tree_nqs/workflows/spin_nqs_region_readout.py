"""Bounded N36 physics readouts of immutable carrier and saved attention states.

Reuse the actual three-determinant A1 carrier, scalar-gate queries, Krylov
coefficients and full-H executor. No optimization, new head, ED, projection
shortcut for the dressed state, or DMRG prerequisite. Completed phases are
immutable and interrupted per-query work is checkpointed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.a1_carrier_queries import compile_a1_overlapping_carrier_queries
from su2_nqs.benchmarks.multicontext_resonance_queries import multicontext_overlap_filter_rows
from su2_nqs.benchmarks.operator_word_jax import evaluate_scalar_word_action_plan_numpy
from su2_nqs.benchmarks.spin_power_bridge import build_bounded_spin_power_plan
from su2_nqs.benchmarks.spin_operator_information import joint_spin_operator_split, spin_dot_matrix
from su2_nqs.benchmarks.spin_partition_readout import spin_partition_readout, _spin_projectors, _joint_projectors
from su2_nqs.benchmarks.tensor_network_readout import physical_spin_block_readout
from su2_nqs.benchmarks.triangular_induced_parton import build_n36_pi_zero_flux_orbit, projected_pi_zero_flux_a1_amplitudes_jax
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.vmc.carrier_reanchoring import polynomial_carrier_products
from su2_nqs.vmc.cutset_training import paired_chain_audit
from su2_nqs.vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange
from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries
from su2_nqs.vmc.regional_density import (regional_query_plan, conditional_regional_matrices,
    regional_density_summary, regional_observable_summary)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_nqs_region_readout.json'
NAMES = ('projected_VMC', 'two_Lanczos', 'attention_initial', 'attention_final')
jax.config.update('jax_enable_x64', True)


class SavedStates:
    """Fixed physical amplitudes, no training or state selection from readouts."""
    def __init__(self, spec: dict, output: Path):
        self.spec, self.output = spec, output
        self.start = perf_counter(); self.queries = 0
        orbit = build_n36_pi_zero_flux_orbit()
        self.raw = jax.jit(lambda x:projected_pi_zero_flux_a1_amplitudes_jax(x, orbit=orbit))
        self.scale = json.loads((ROOT/spec['units_source']).read_text())['cscale']
        self.poly = np.asarray(json.loads((ROOT/spec['K2_source']).read_text())['polynomial_coefficients'])
        source = ROOT/spec['attention_source']
        template = dict(parameters=jnp.zeros(730), generator_table=jnp.zeros((6,6,10)), coefficients=jnp.zeros(1176))
        tables = [load_parameter_artifact(source/name, template)['generator_table'] for name in ('initial0.json','attention0_endpoint.json')]
        self.weights = [np.asarray(jax.jit(multicontext_overlap_filter_rows)(t,t,jnp.arange(4096))) for t in tables]
        self.group = triangular_hnf_gamma_a1_group(((6,0),(0,6)))
        self.bonds = triangular_hnf_j1j2_bonds(((6,0),(0,6)), j2=.125)
        old = json.loads((ROOT/spec['cells_source']).read_text())
        self.cells = old['config']['physical_cells']

    def check(self):
        if perf_counter()-self.start > self.spec['phase_wall_seconds_cap']:
            raise TimeoutError('bounded phase expired; numerical checkpoints retained')

    def carrier(self, x):
        self.check(); batch = self.spec['amplitude_batch']
        parts = []
        for off in range(0,len(x),batch):
            v=x[off:off+batch]; padded=np.pad(v,((0,batch-len(v)),(0,0)),mode='edge')
            parts.append(np.asarray(self.raw(jnp.asarray(padded)))[:len(v)]/self.scale)
            self.queries += len(v)
        return np.concatenate(parts)

    def attention(self, x):
        """All432 dressed images, both coherent gates; only C arguments reused."""
        plan = compile_a1_overlapping_carrier_queries(x, self.group, cells=self.cells, carrier_sector='Gamma-A1')
        c = self.carrier(plan.configurations)
        rows = plan.unique_local_rows[plan.local_row_indices]
        values = c[plan.connection_indices]
        return np.column_stack([np.mean(np.sum(w[rows]*values,axis=-1),axis=1) for w in self.weights])

    def powers(self, x, maximum_power: int):
        plan, work = build_bounded_spin_power_plan(x,self.bonds,maximum_power=maximum_power,
            maximum_expansion_entries=2_000_000,maximum_terminal_queries=524288)
        values = evaluate_scalar_word_action_plan_numpy(plan,self.carrier,amplitude_batch_size=self.spec['amplitude_batch'])
        return values, work

    def amplitudes(self, x):
        """Query all four states, with node-safe powers through H2."""
        p,_=self.powers(x,2)
        return np.column_stack((p[:,0],p@self.poly,self.attention(x)))


def register(config: Path) -> dict:
    spec=json.loads(config.read_text()); out=ROOT/spec['output']
    if out.exists(): raise FileExistsError('registration is immutable')
    if (spec['chains'],spec['records'],spec['phase_wall_seconds_cap']) != (16,8,1800):
        raise ValueError('fixed bounded128-record readout pilot required')
    paths=[config,ROOT/spec['units_source'],ROOT/spec['K2_source'],ROOT/spec['cells_source']]
    for name in ('initial0','attention0_endpoint'):
        paths += [ROOT/spec['attention_source']/(name+ext) for ext in ('.json','.npz')]
    sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    out.mkdir(parents=True)
    for p in sources:
        dst=out/'source_snapshots'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dst)
    record=dict(config=spec,inputs={str(p.relative_to(ROOT)):file_sha256(p) for p in paths},sources=sources,
        physical_problem='triangular6x6 PBC, J1=1,J2=.125,216 bonds,432 Gamma-A1 images',
        state_names=list(NAMES), neural_description='existing730-parameter two-environment scalar attention, NOT a new/full foundation ViT',
        uncertainty_unit='whole persistent chain, not IID records', new_training=0,new_ED=0,phase_claim=False)
    record['registration_sha256']=canonical_sha256(record);atomic_json(out/'registration.json',record)
    return record


def verified(config: Path):
    spec=json.loads(config.read_text());out=ROOT/spec['output'];reg=json.loads((out/'registration.json').read_text())
    if reg['config'] != spec: raise ValueError('protocol changed')
    for p,h in {**reg['inputs'],**reg['sources']}.items():
        if file_sha256(ROOT/p)!=h: raise ValueError('sealed numerical input/source changed: '+p)
    return spec,out,reg


def sample(spec, out, model):
    if (out/'samples.npz').exists(): raise FileExistsError('no additional samples in this pilot')
    start=perf_counter(); density=lambda x: np.sqrt(abs(model.carrier(x))**2+spec['uniform_floor'])
    samples=sample_fixed_wavefunction_exchange(density,n_sites=36,n_up=18,chain_count=spec['chains'],
        burn_sweeps=spec['burn'],measurement_blocks=spec['records'],decorrelation_sweeps=spec['decorrelation'],seed=spec['seed'])
    states=samples.configurations; c=model.carrier(states)
    np.savez_compressed(out/'samples.npz',states=states,q=abs(c)**2+spec['uniform_floor'],
        groups=np.tile(np.arange(spec['chains']),spec['records']))
    atomic_json(out/'sampling.json',dict(seconds=perf_counter()-start,acceptance=samples.acceptance_fraction,
        burn_acceptance=samples.burn_acceptance_fraction,seed=spec['seed'],records=len(states),independent_chains=spec['chains'],
        target_Q='|C/cscale|^2 + fixed uniform floor; NO exchange bridge',
        sample_sha256=file_sha256(out/'samples.npz'),equilibrium_certified=False,
        new_full_H_calls=0,carrier_amplitude_queries=model.queries))


def products(spec,out,model):
    """Checkpoint per two centers, reusing carrier queries for all attention states."""
    dest=out/'products';dest.mkdir(exist_ok=True)
    with np.load(out/'samples.npz') as z: x,q,g=z['states'],z['q'],z['groups']
    parts=[];work=[];start=perf_counter()
    for i in range(0,len(x),2):
        p=dest/f'{i:03d}.npz';meta=p.with_suffix('.json')
        if p.exists():
            with np.load(p) as z:parts.append((z['psi'],z['hpsi']))
            work.append(json.loads(meta.read_text()));continue
        tick=perf_counter();model.check()
        # H3 union exceeds per-plan bound for two centers: power-checkpoint each.
        rows=[model.powers(v[None],3)[0][0] for v in x[i:i+2]]
        powers=np.stack(rows); k,hk=polynomial_carrier_products(powers,model.poly)
        hp=compile_physical_h_queries(x[i:i+2],model.bonds)
        attention=np.concatenate([model.attention(hp.configurations[j:j+2]) for j in range(0,len(hp.configurations),2)])
        ps=np.column_stack((powers[:,0],k,attention[hp.center_indices]))
        ha=np.sum(np.where(hp.valid[:,:,None],attention[hp.connected_indices]*hp.elements[:,:,None],0),axis=1)
        hs=np.column_stack((powers[:,1],hk,ha))
        np.savez_compressed(p,psi=ps,hpsi=hs)
        record=dict(centers=len(ps),seconds=perf_counter()-tick,carrier_queries_cumulative=model.queries,
            algebraic_H_applications=3*len(ps),attention_full_H_centers=2*len(ps))
        atomic_json(meta,record);parts.append((ps,hs));work.append(record)
        print('full H',i+len(ps),'/',len(x),round(perf_counter()-start,1),'s',flush=True)
    psi=np.concatenate([v[0] for v in parts]); hpsi=np.concatenate([v[1] for v in parts])
    np.savez_compressed(out/'products.npz',psi=psi,hpsi=hpsi)
    audit=paired_chain_audit(psi,hpsi,q,g,36,spec['bootstrap_replicates'],spec['bootstrap_seed'],NAMES)
    # Strict-SU2 states allow isotropic3<Sz Sz>, but not for a generic Sz ViT.
    coords=np.array([divmod(i,6) for i in range(36)])
    momenta=dict(K=(1/3,1/3),minus_K=(-1/3,-1/3),M1=(.5,0),M2=(0,.5),M3=(.5,.5))
    structure={}
    for name,k in momenta.items():
        diagonal=3*abs((x-.5)@np.exp(2j*np.pi*(coords@np.array(k))))**2/36
        rows={}
        for a,arm in enumerate(NAMES):
            w=abs(psi[:,a])**2/q;value=np.dot(w,diagonal)/w.sum()
            jack=np.array([np.dot(w[g!=c],diagonal[g!=c])/w[g!=c].sum() for c in np.unique(g)])
            rows[arm]=dict(value=float(value),chain_jackknife_se=float(np.sqrt((len(jack)-1)*np.mean((jack-jack.mean())**2))))
        structure[name]=rows
    atomic_json(out/'energy_correlations.json',dict(audit=audit,structure_factors=structure,work=work,
        energy_variance_full_H=True,isotropy_used_only_for_proved_exact_singlets=True,
        finite_sample_ratios_biased=True,equilibrium_certified=False,seconds=perf_counter()-start))


def region(spec,out,model,cut):
    """Same independent samples for compact4 then compact6/winding6, no twirl."""
    if (out/(cut+'.json')).exists():raise FileExistsError('regional readout is immutable')
    sites=spec['regions'][cut]['sites'];left=spec['regions'][cut]['left']
    with np.load(out/'samples.npz') as z:x,g=z['states'],z['groups']
    plan=regional_query_plan(x,sites);dest=out/cut;dest.mkdir(exist_ok=True)
    start=perf_counter();values=[]
    for i in range(0,len(plan.configurations),2):
        path=dest/f'{i:05d}.npz'
        if path.exists():
            with np.load(path) as z: values.append(z['amplitudes'])
        else:
            amps=model.amplitudes(plan.configurations[i:i+2]);np.savez_compressed(path,amplitudes=amps);values.append(amps)
        if (i+2)%64==0:print(cut,'queries',i+2,'/',len(plan.configurations),round(perf_counter()-start,1),'s',flush=True)
    values=np.concatenate(values);Q=abs(values[:,0])**2+spec['uniform_floor']
    results={};arrays={};summaries={}
    for a,arm in enumerate(NAMES):
        per_record=conditional_regional_matrices(plan,values[:,a],Q)
        summary=regional_density_summary(per_record,g);rho=summary['rho']
        summaries[arm]=summary
        block=physical_spin_block_readout(rho);joint=spin_partition_readout(rho,left)
        weights={str(q):regional_observable_summary(summary,p) for q,p in _spin_projectors(len(sites))}
        joint_uncertainty={str(labels):regional_observable_summary(summary,p) for labels,p in _joint_projectors(len(sites),tuple(left))}
        high={}
        if len(sites)==6:
            for q in (1,3):
                split=joint_spin_operator_split(6,left,(q,q,0))
                high[str(q)]={k:regional_observable_summary(summary,getattr(split,k)) for k in ('full','low','high')}
        pairs=[(i,j) for i in range(len(sites)) for j in range(i+1,len(sites))]
        ops={pair:spin_dot_matrix(len(sites),*pair) for pair in pairs}
        spins={str(pair):regional_observable_summary(summary,op) for pair,op in ops.items()}
        dimers=[]
        for i,pair in enumerate(pairs):
            for other in pairs[i+1:]:
                if not set(pair).isdisjoint(other):continue
                op=ops[pair]@ops[other]
                def measure(r):return float((np.trace(r@op)-np.trace(r@ops[pair])*np.trace(r@ops[other])).real)
                value=measure(rho);jack=np.array([measure(r) for r in summary['jackknife_rhos']])
                dimers.append(dict(local_bonds=[list(pair),list(other)],value=value,
                    chain_jackknife_se=float(np.sqrt((len(jack)-1)*np.mean((jack-jack.mean())**2)))))
        arrays[arm+'_rho']=rho;arrays[arm+'_jackknife_rhos']=summary['jackknife_rhos']
        results[arm]=dict(spin_weights=weights,joint_spin=joint,joint_weights_uncertainty=joint_uncertainty,higher_order=high,spin_pairs=spins,connected_dimers=dimers,
            sampling=dict(records=summary['records'],independent_chains=summary['independent_chains'],
                weight_ess_fraction=summary['weight_ess_fraction'],maximum_weight_fraction=summary['maximum_weight_fraction']),
            raw_rdm_diagnostics={k:block[k] for k in ('su2_factorization_frobenius','strict_su2_eligible','trace_error','hermiticity_frobenius','minimum_eigenvalue')},
            strict_multiplicity_spectrum_reported=False,
            warning='No SU2 twirl: finite-sample RDM need not factorize even for an exact singlet. Do not interpret sampled eigenvalues as strict multiplicity levels.')
    np.savez_compressed(out/(cut+'.npz'),**arrays,unique_configurations=plan.configurations,amplitudes=values,Q=Q)
    paired={}
    for old,new in (('projected_VMC','two_Lanczos'),('attention_initial','attention_final'),('two_Lanczos','attention_final')):
        ops={str(q):p for q,p in _spin_projectors(len(sites))}
        if len(sites)==6:
            ops.update({f'high_q{q}':joint_spin_operator_split(6,left,(q,q,0)).high for q in (1,3)})
        contrast={}
        for name,op in ops.items():
            delta=summaries[new]['rho']-summaries[old]['rho']
            j=summaries[new]['jackknife_rhos']-summaries[old]['jackknife_rhos']
            values=np.einsum('cab,ba->c',j,op).real
            contrast[name]=dict(value=float(np.trace(delta@op).real),
                chain_jackknife_se=float(np.sqrt((len(values)-1)*np.mean((values-values.mean())**2))))
        paired[new+'-minus-'+old]=contrast
    atomic_json(out/(cut+'.json'),dict(sites=sites,left=left,states=results,paired=paired,seconds=perf_counter()-start,
        unique_regional_queries=len(plan.configurations),carrier_queries=model.queries,new_training=0,new_ED=0,
        paired_samples=True,phase_claim=False,equilibrium_certified=False,
        estimator='conditional coherent psi_A psi_A† / sum_A Q, normalized by estimated trace'))


def connected_three_bond_replay(config: Path = DEFAULT) -> dict:
    """Read saved N36 RDM/jackknife arrays, never query a state or sample.

    Nonlinear cumulants are recomputed for every delete-WHOLE-chain RDM.
    Paired differences use matched deletes, not independent error quadrature.
    Original seals remain unchanged; the child analysis is separately sealed.
    """
    from su2_nqs.benchmarks.spin_operator_information import three_bond_readout
    config = config.resolve(); spec=json.loads(config.read_text()); parent=ROOT/spec['output']
    completion=json.loads((parent/'completion.json').read_text())
    body=dict(completion);seal=body.pop('completion_sha256')
    if canonical_sha256(body)!=seal:raise ValueError('original pilot completion seal failed')
    for path,digest in completion['outputs'].items():
        if file_sha256(parent/path)!=digest:raise ValueError('original pilot output changed')
    out=parent/'connected_three_bond_replay'
    if out.exists():raise FileExistsError('connected replay is immutable')
    inputs=[config,parent/'completion.json']+[parent/(cut+ext)
        for cut in ('compact6','winding6') for ext in ('.json','.npz')]
    sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    out.mkdir(parents=True)
    for p in sources:
        dst=out/'source_snapshots'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dst)
    atomic_json(out/'registration.json',dict(
        inputs={str(p.relative_to(ROOT)):file_sha256(p) for p in inputs},sources=sources,
        protocol='ALL X/Y raw products and connected third cumulants on BOTH saved six-spin regions. Recompute nonlinear diagnostics on every matched whole-chain delete. No state/readout selection, no new amplitudes, H calls or MC samples.',
        frozen_state_names=NAMES,phase_claim=False))
    start=perf_counter();regions={};keys=('X_raw','X_connected','Y_raw','Y_connected','high_q1','high_q3')

    def values(rho,left):
        row=three_bond_readout(rho,left)
        return np.array([row['orbits'][name][kind] for name in ('X','Y')
                         for kind in ('raw','connected')]+[row['high_q1'],row['high_q3']])

    def uncertainty(central,jack):
        centered=jack-jack.mean(axis=0)
        covariance=(len(jack)-1)/len(jack)*(centered.T@centered)
        return dict(observables={k:dict(value=float(v),chain_jackknife_se=float(np.sqrt(max(0.,covariance[i,i]))))
                                 for i,(k,v) in enumerate(zip(keys,central))},
                    covariance=covariance.tolist(),keys=keys)

    for cut in ('compact6','winding6'):
        old=json.loads((parent/(cut+'.json')).read_text());left=old['left'];centers={};deletes={}
        with np.load(parent/(cut+'.npz')) as z:
            for name in NAMES:
                centers[name]=values(z[name+'_rho'],left)
                deletes[name]=np.array([values(r,left) for r in z[name+'_jackknife_rhos']])
        if any(j.shape!=(16,6) for j in deletes.values()):raise ValueError('saved16-chain common deletes required')
        maximum_replay=0.
        for name in NAMES:
            for i,q in ((4,'1'),(5,'3')):
                maximum_replay=max(maximum_replay,abs(centers[name][i]-old['states'][name]['higher_order'][q]['high']['value']))
        if maximum_replay>3e-12:raise ValueError('raw projector identity replay failed')
        pairs={}
        for before,after in (('projected_VMC','two_Lanczos'),('attention_initial','attention_final'),('two_Lanczos','attention_final')):
            pairs[after+'-minus-'+before]=uncertainty(centers[after]-centers[before],deletes[after]-deletes[before])
        regions[cut]=dict(sites=old['sites'],left=left,
            states={n:uncertainty(centers[n],deletes[n]) for n in NAMES},paired=pairs,
            maximum_raw_projector_replay_error=maximum_replay,
            raw_RDMs_not_twirled=True,strict_multiplicity_spectrum_reported=False)
    result=dict(regions=regions,seconds=perf_counter()-start,
        new_MC_samples=0,new_state_queries=0,new_full_H_calls=0,new_training=0,new_ED=0,
        independent_chains=16,records=128,equilibrium_certified=False,phase_claim=False,
        multiple_observables_reported_without_post_selection=True,
        warning='Connected cumulants differ from the linear six-body projector remainder. Chain errors do not certify equilibrium, unseen tails or candidate quality; this is a post-hoc physical diagnostic on known pilot data, not a new untouched validation.')
    atomic_json(out/'results.json',result)
    atomic_json(out/'completion.json',dict(status='complete',outputs={p.name:file_sha256(p) for p in out.iterdir() if p.suffix=='.json' and p.name!='completion.json'}))
    print(json.dumps(dict(seconds=result['seconds'],paired={c:r['paired'] for c,r in regions.items()})),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--phase',choices=('register','sample','products','compact4','compact6','winding6','connected-three-bond-replay'),required=True)
    args=parser.parse_args()
    if args.phase=='register':register(args.config);return
    if args.phase=='connected-three-bond-replay':connected_three_bond_replay(args.config);return
    spec,out,_=verified(args.config);model=SavedStates(spec,out)
    if args.phase=='sample':sample(spec,out,model)
    elif args.phase=='products':products(spec,out,model)
    else:
        if args.phase!='compact4' and not (out/'compact4.json').exists():raise ValueError('compact4 calibration precedes six-spin readouts')
        region(spec,out,model,args.phase)


if __name__=='__main__':main()
