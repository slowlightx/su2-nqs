"""Bounded NQS-internal block-spin coupling calibration, independent of DMRG."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import shutil
import signal
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.spin_partition_readout import spin_partition_readout, spin_partition_distance
from su2_nqs.benchmarks.readout_quality import regional_readout_distance
from projects.clebsch_tree_nqs.workflows.spin_tensor_network_reference import (
    ROOT, _json, file_sha256, transitive_local_source_manifest,
)

DEFAULT = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_partition_coupling_readout.json'


def run(config: Path = DEFAULT) -> dict:
    """Read sealed saved RDMs and their quality histories; never create new states."""
    spec = json.loads(config.read_text())
    output = ROOT/spec['output']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('derived numerical artifacts are immutable')
    candidate, trusted = [ROOT/spec[k] for k in ('candidate_archive', 'trusted_archive')]
    screen, previous = [json.loads((d/'results.json').read_text()) for d in (candidate, trusted)]
    bundle_path = ROOT/screen['state_bundle']['path']
    if file_sha256(bundle_path) != screen['state_bundle']['sha256']:
        raise ValueError('candidate state bundle changed')
    if file_sha256(trusted/'readout.npz') != previous['readout_sha256']:
        raise ValueError('trusted state bundle changed')
    inputs = {str(p.relative_to(ROOT)): file_sha256(p) for p in
              (config, candidate/'results.json', bundle_path, trusted/'results.json', trusted/'readout.npz')}
    sources = transitive_local_source_manifest((Path(__file__),), root=ROOT,
                  package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    output.mkdir(parents=True)
    for path in sources:
        dst = output/'source_snapshots'/path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/path, dst)
    _json(output/'registration.json', dict(config=spec, inputs=inputs, sources=sources,
        physical_problem=screen['physical_problem'], new_ED=0, new_states=0, DMRG_required=False))
    start = perf_counter()
    old_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('bounded regional readout')))
    signal.alarm(spec['wall_seconds_cap'])
    try:
        records, rhos = {}, {}
        with np.load(bundle_path, allow_pickle=False) as arrays:
            for name, source in screen['states'].items():
                records[name] = dict(quality=source['quality'], provenance=source['provenance'], regions={})
                for cut, left in spec['cuts'].items():
                    rho = arrays[f'{name}_{cut}_rho']
                    records[name]['regions'][cut] = spin_partition_readout(rho, left)
                    rhos[name, cut] = rho
        # Coupling controls reuse saved high-quality states, not new eigensolves.
        with np.load(trusted/'readout.npz', allow_pickle=False) as arrays:
            for name in ('triangular_j0', 'triangular_j0.075', 'triangular_j0.125', 'triangular_j0.175',
                         'dimer_internal', 'dimer_four_crossing', 'multiplicity_bell_q0',
                         'collective_two_sublattice', 'collective_three_sublattice'):
                source = previous['states'][name]
                records[name] = dict(quality=source.get('quality'), provenance=source['provenance'], regions={})
                for cut, left in spec['cuts'].items():
                    rho = arrays[f'{name}_{cut}_rho']
                    records[name]['regions'][cut] = spin_partition_readout(rho, left)
                    rhos[name, cut] = rho
        def compare(a, b):
            result = {}
            for cut in spec['cuts']:
                d = spin_partition_distance(records[a]['regions'][cut], records[b]['regions'][cut])
                physical = regional_readout_distance(rhos[a, cut], rhos[b, cut])['physical_trace_distance']
                d.update(physical_trace_distance=physical, joint_measurement_contractivity_error=max(0., d['joint_total_variation']-physical))
                result[cut] = d
            return result
        couplings = [dict(j2_a=a, j2_b=b, metrics=compare(f'triangular_j{a:g}', f'triangular_j{b:g}'))
                     for a, b in ((0., .075), (.075, .125), (.125, .175), (0., .125))]
        families = list(screen['primary_quality_flows'])
        comparisons = []
        for a, b in itertools.combinations(families, 2):
            final = compare(a+'_k4', b+'_k4')
            own_a, own_b = compare(a+'_k4', a+'_k2'), compare(b+'_k4', b+'_k2')
            comparisons.append(dict(family_a=a, family_b=b, regions={cut: dict(
                **final[cut], refinement_a=own_a[cut]['joint_total_variation'],
                refinement_b=own_b[cut]['joint_total_variation'],
                refinement_sum=own_a[cut]['joint_total_variation']+own_b[cut]['joint_total_variation'],
                exceeds_each_refinement_sum=final[cut]['joint_total_variation'] >
                    own_a[cut]['joint_total_variation']+own_b[cut]['joint_total_variation']+1e-12,
            ) for cut in spec['cuts']}, warning='Observable-specific empirical changes are not certified state errors.'))
        vap_pair = compare('triangular_vap_2026092901_terminal', 'triangular_vap_2026092902_terminal')
        vap_refinement = [compare(f'triangular_vap_{s}_parent', f'triangular_vap_{s}_terminal')
                          for s in (2026092901, 2026092902)]
        errors = {name: compare(name, 'saved_reference') for name in screen['states']}
        diagnostics = [r['diagnostics'] for state in records.values() for r in state['regions'].values()]
        result = dict(config=spec, states=records, coupling_calibration=couplings,
            primary_quality_flows=screen['primary_quality_flows'], primary_pair_comparisons=comparisons,
            errors_to_same_H_reference=errors, VAP_pair=dict(final=vap_pair, each_refinement=vap_refinement),
            validation=dict(maximum_total_spin_marginal_error=max(r['total_spin_marginal_error'] for r in diagnostics),
                maximum_probability_closure_error=max(r['probability_closure_error'] for r in diagnostics),
                maximum_joint_contractivity_error=max(m['joint_measurement_contractivity_error'] for v in errors.values() for m in v.values())),
            seconds=perf_counter()-start, new_ED=0, new_training=0, new_samples=0, new_DMRG=0,
            new_full_H_calls=0, phase_claim=False, size_evolution_released=False,
            disposition='Block-spin recoupling calibration and NQS quality dependence, not evidence for a distinct thermodynamic liquid.')
        _json(output/'results.json', result)
        print(json.dumps(dict(seconds=result['seconds'], validation=result['validation'],
              resolved_joint_pairs=sum(any(r['exceeds_each_refinement_sum'] for r in p['regions'].values()) for p in comparisons))), flush=True)
        return result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    run(parser.parse_args().config)
