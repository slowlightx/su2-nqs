"""Once-only bridge of frozen two-Lanczos into current physical amplitudes.

No training, new samples or coefficient fitting. Old chains are KNOWN data.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.operator_word_jax import evaluate_scalar_word_action_plan_numpy
from su2_nqs.benchmarks.spin_power_bridge import build_bounded_spin_power_plan, standardized_krylov_polynomial
from su2_nqs.benchmarks.triangular_induced_parton import build_n36_pi_zero_flux_orbit, projected_pi_zero_flux_a1_amplitudes_jax
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.vmc.carrier_reanchoring import polynomial_carrier_products
from su2_nqs.vmc.cutset_training import paired_chain_audit, weighted_state_distance
from .spin_n36_su2_scalar_dressing import _sealed_common_data, _baseline_from_cache

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_krylov_carrier_bridge.json'


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def raw_h_powers(raw: np.ndarray, words: tuple, coupling: float) -> np.ndarray:
    """Combine existing ORDERED shell products to powers of total H."""
    positions = {word:i for i,word in enumerate(words)}
    labels = ('J1_shell','J2_shell')
    weights = (1.,coupling)
    powers = []
    for depth in range(4):
        power = np.zeros(len(raw),complex)
        for indices in product(range(2),repeat=depth):
            word = tuple(labels[i] for i in indices)
            power += float(np.prod([weights[i] for i in indices]))*raw[:,positions[word]]
        powers.append(power)
    return np.column_stack(powers)


def verify(reg: dict) -> None:
    for path,digest in {**reg['sources'],**reg['inputs']}.items():
        if file_sha256(ROOT/path) != digest:
            raise ValueError('sealed source/input changed: '+path)


def prepare(spec: dict, output: Path) -> None:
    """Seal before any new physical H products; no silent resumption/refit."""
    if (spec['preflight_cache_rows'] != [0,1] or spec['maximum_power'] != 3
            or spec['j2_over_j1'] != .125 or spec['maximum_terminal_queries'] != 524288
            or spec['amplitude_batch_size'] != 128 or spec['bootstrap_replicates'] != 512):
        raise ValueError('fixed two-center protocol required')
    candidate,audit,tx,tr,ax,ar = _sealed_common_data(spec)
    baseline = _baseline_from_cache(candidate,audit,tr,ar,require_sealed_reproduction=True)
    poly,conversion = standardized_krylov_polynomial(
        candidate['frozen_two_lanczos']['coefficients'], candidate['standardization']['means'],
        candidate['standardization']['scales'],np.array([1.,.125,0.]))
    c2 = np.asarray(read(ROOT/spec['retained_filter'])['coefficients'],float)
    words = tuple(tuple(w) for w in candidate['words'])
    powers = [raw_h_powers(raw,words,.125) for raw in (tr,ar)]
    for prefix,p,center in zip(('train','audit'),powers,(baseline['train_carrier'],baseline['audit_carrier'])):
        k,hk = polynomial_carrier_products(p,poly)
        reference = baseline[prefix+'_state']*center
        hreference = baseline[prefix+'_action']*center
        for got,expected in ((k,reference),(hk,hreference)):
            defect = float(np.max(abs(got-expected))/max(float(np.max(abs(expected))),1e-300))
            if defect > 2e-10:
                raise ValueError('algebraic conversion did not preserve frozen physical products')
            conversion['maximum_cache_product_relative_defect'] = max(
                conversion.get('maximum_cache_product_relative_defect',0.),defect)
    inputs = [CONFIG,ROOT/spec['sealed_development_result'],ROOT/spec['sealed_audit_result'],
              ROOT/audit['config']['development_artifact'],ROOT/spec['retained_filter']]
    for result,prefix in ((ROOT/spec['sealed_development_result'],'develop'),
                          (ROOT/spec['sealed_audit_result'],'audit')):
        inputs += sorted((result.parent/'component_cache').glob(prefix+'_*.npz'))
    reg = dict(config=spec,runtime=numerical_runtime_identity(), polynomial_coefficients=poly.tolist(),
        retained_C2_coefficients=c2.tolist(),conversion=conversion,
        inputs={str(p.relative_to(ROOT)):file_sha256(p) for p in inputs},
        sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    reg['registration_sha256'] = canonical_sha256(reg)
    output.mkdir(exist_ok=False)
    atomic_json(output/'registration.json',reg)
    np.savez_compressed(output/'known.npz',development_states=tx,audit_states=ax,
                        development_powers=powers[0],audit_powers=powers[1])
    atomic_json(output/'prepared.json',dict(known_sha256=file_sha256(output/'known.npz'),
        registration_sha256=reg['registration_sha256'],known_data_not_untouched=True,
        parameter_updates=0,new_samples=0,new_H_calls=0))
    print('sealed frozen K2/C2 bridge: two known centers, no refit or training',flush=True)


def run(spec: dict, reg: dict, output: Path) -> None:
    if (output/'results.json').exists():
        raise ValueError('once-only bridge complete; no repeat or automatic training')
    if file_sha256(output/'known.npz') != read(output/'prepared.json')['known_sha256']:
        raise ValueError('known products changed')
    start = perf_counter()
    def budget_check():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform=='darwin' else 1024**2)
        if perf_counter()-start>spec['wall_limit_seconds'] or rss>spec['peak_rss_limit_gib']:
            raise RuntimeError('resource bound exceeded; preserve checkpoints, no continuation')
        return rss
    poly,c2 = np.asarray(reg['polynomial_coefficients']),np.asarray(reg['retained_C2_coefficients'])
    with np.load(output/'known.npz',allow_pickle=False) as stored:
        data={k:stored[k] for k in stored.files}
    summaries = {}
    for prefix in ('development','audit'):
        powers = data[prefix+'_powers']
        c,hc = polynomial_carrier_products(powers,c2)
        k,hk = polynomial_carrier_products(powers,poly)
        q = abs(powers[:,0])**2
        if np.any(q<=0):
            raise ValueError('old C0-Born records unexpectedly contain exact proposal nodes')
        summaries[prefix] = paired_chain_audit(np.column_stack((c,k)),np.column_stack((hc,hk)),
            q,np.arange(len(q))%64,36,spec['bootstrap_replicates'],spec['bootstrap_seed'],('C2','frozen_K2'))
        summaries[prefix]['physical_distance'] = weighted_state_distance(c,k,q)
        summaries[prefix]['known_data_not_untouched'] = True
    atomic_json(output/'known_comparison.json',summaries)
    orbit = build_n36_pi_zero_flux_orbit()
    amplitude = jax.jit(lambda x:projected_pi_zero_flux_a1_amplitudes_jax(x,orbit=orbit))
    batch = spec['amplitude_batch_size']
    amplitude(jnp.asarray(np.tile(data['audit_states'][0],(batch,1)))).block_until_ready()
    def evaluate(x):
        budget_check()
        padded = np.pad(x,((0,batch-len(x)),(0,0)),mode='edge')
        return np.asarray(amplitude(jnp.asarray(padded)))[:len(x)]
    bonds=triangular_j1j2_bonds(6,6,j2=.125,periodic=True)
    reports=[]
    for index in spec['preflight_cache_rows']:
        checkpoint = output/f'center{index}.json'
        if checkpoint.exists():
            reports.append(read(checkpoint)); continue
        t0 = perf_counter()
        plan,work=build_bounded_spin_power_plan(data['audit_states'][index:index+1],bonds,
            maximum_power=3,maximum_expansion_entries=spec['maximum_expansion_entries'],
            maximum_terminal_queries=spec['maximum_terminal_queries'])
        t1=perf_counter()
        values=evaluate_scalar_word_action_plan_numpy(plan,evaluate,amplitude_batch_size=batch)
        expected=data['audit_powers'][index:index+1]
        relative=(abs(values-expected)/np.maximum(abs(expected),1e-300))[0]
        # Check each power separately, not against a large H3 scale.
        if float(np.max(relative)) > spec['replay_relative_tolerance']:
            atomic_json(output/'replay_failure.json',dict(center=index,relative_defects=relative.tolist()))
            raise ValueError('physical power replay failed; preserve evidence, no auto reseal')
        k,hk=polynomial_carrier_products(values,poly)
        report=dict(center=index,work=work,terminal_evaluation_seconds=perf_counter()-t1,
            wall_seconds=perf_counter()-t0,relative_power_defects=relative.tolist(),
            physical_K2=[float(k[0].real),float(k[0].imag)],
            physical_HK2=[float(hk[0].real),float(hk[0].imag)],
            amplitude_queries_through_H2=int(work['terminals_per_center_and_power'][0][2]),
            full_H_equivalent_applications=3,amplitude_batch_size=batch)
        np.savez_compressed(output/f'center{index}.npz',state=data['audit_states'][index],powers=values)
        atomic_json(checkpoint,report); reports.append(report)
        print('center',index,'terminals',work['unique_terminal_queries'],'seconds',round(report['wall_seconds'],3),flush=True)
    rss=budget_check()
    seconds=perf_counter()-start
    if seconds>spec['wall_limit_seconds'] or rss>spec['peak_rss_limit_gib']:
        raise RuntimeError('resource bound exceeded; checkpoints retained, no continuation')
    stronger=all(v['paired']['frozen_K2-minus-C2']['energy_per_site']['value']<0
                 and v['paired']['frozen_K2-minus-C2']['variance_per_site']['value']<=0
                 for v in summaries.values())
    result=dict(registration_sha256=reg['registration_sha256'],known_comparison=summaries,
        preflight=reports,wall_seconds=seconds,peak_rss_gib=rss,
        maximum_power_relative_defect=max(max(r['relative_power_defects']) for r in reports),
        strong_K2_better_on_both_known_streams=stronger,
        new_samples=0,parameter_updates=0,new_full_H_equivalent_applications=6,
        GPU_hours=0,ED_performed=False,neural_variational_win=False,
        no_independent_new_energy_validation=True,
        decision='Retain frozen K2 as strong physical-amplitude reference. Do not repeat old gradient/step/head repairs; separate any additive neural test from baseline filter gain. Actual H3 cost, known-data status and tail uncertainty must be kept in the next budget. No automatic training/GPU license.')
    result['result_sha256']=canonical_sha256(result)
    atomic_json(output/'results.json',result)
    print(json.dumps({k:result[k] for k in ('wall_seconds','peak_rss_gib','maximum_power_relative_defect','strong_K2_better_on_both_known_streams')},indent=2),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=('prepare','run'),required=True)
    args=parser.parse_args(); jax.config.update('jax_enable_x64',True)
    spec=read(CONFIG); output=ROOT/spec['output']
    if args.phase=='prepare': prepare(spec,output)
    else:
        reg=read(output/'registration.json')
        if reg['config']!=spec: raise ValueError('config changed')
        body=dict(reg); body.pop('registration_sha256')
        if canonical_sha256(body)!=reg['registration_sha256']: raise ValueError('registration hash changed')
        verify(reg); run(spec,reg,output)


if __name__=='__main__': main()
