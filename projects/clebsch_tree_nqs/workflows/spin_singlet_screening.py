"""Separate six-spin screening occupations from invariant singlet coherence.

Read existing sealed off-diagonal Born RDM caches only. This workflow never
optimizes, samples, calls H, repairs symmetry, or assigns a QSL phase.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.spin_partition_readout import _three_plus_three_singlet_columns
from su2_nqs.benchmarks.spin_operator_information import joint_spin_operator_split
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.chain_quadratic import whole_chain_squared_mean
from .spin_spatial_vit_quality import read_sealed


def screening_chain_readout(matrices: np.ndarray, chain_ids: np.ndarray,
                           left_sites: tuple[int, ...]) -> dict:
    """Measure (q_L,q_R,q_T)=(1,1,0)/(3,3,0), q=2S, in one cut.

    Born conditional matrices must be unit trace and Hermitian. The four
    doublet-pair singlets and one quartet-pair singlet are exact regional
    singlets; their operators remain legal when the full state is not SU2.
    Quadratic numerators use independent chains; occupation ratios remain
    finite-sample biased. No strict entanglement spectrum is reported.
    """
    a = np.asarray(matrices, np.complex128)
    groups = np.asarray(chain_ids)
    if (a.ndim != 3 or a.shape[1:] != (64, 64)
            or groups.shape != (len(a),) or not np.isfinite(a).all()
            or np.max(abs(np.trace(a, axis1=1, axis2=2)-1)) > 1e-10
            or np.max(abs(a-a.conj().transpose(0, 2, 1))) > 1e-10
            or len(left_sites) != 3 or len(set(left_sites)) != 3
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 or x > 5 for x in left_sites)):
        raise ValueError('unit-trace Hermitian six-site Born matrices and three distinct physical sites required')
    c = _three_plus_three_singlet_columns(left_sites)
    blocks = np.einsum('ia,nij,jb->nab', c.conj(), a, c, optimize=True)
    # This call validates independent-chain grouping before any inference.
    coherence = whole_chain_squared_mean(blocks[:, :4, 4], groups)
    purity = whole_chain_squared_mean(blocks[:, :4, :4], groups)
    labels = np.unique(groups)
    linear = np.column_stack((np.trace(blocks[:, :4, :4], axis1=1, axis2=2).real,
                              blocks[:, 4, 4].real))
    means = np.stack([linear[groups == g].mean(0) for g in labels])
    point = means.mean(0)
    deletes = (means.sum(0)-means)/(len(means)-1)
    def occupation(row: np.ndarray) -> np.ndarray:
        p11, p33 = row
        if p11+p33 <= 1e-12:
            raise ValueError('regional singlet weight absent; conditional ratio undefined')
        return np.array([p11, p33, p11+p33, p33/(p11+p33)])
    values = occupation(point)
    jack = np.stack([occupation(row) for row in deletes])
    centered = jack-jack.mean(0)
    covariance = (len(jack)-1)/len(jack)*centered.T@centered
    names = ('p11_singlet', 'p33_singlet', 'p_total_singlet', 'quartet_fraction_within_regional_singlet')
    split = joint_spin_operator_split(6,left_sites,(3,3,0))
    components = np.column_stack([np.einsum('nij,ji->n',a,op).real
                                  for op in (split.full,split.low,split.high)])
    component_means = np.stack([components[groups==g].mean(0) for g in labels])
    component_point = component_means.mean(0)
    component_covariance = np.cov(component_means,rowvar=False,ddof=1)/len(labels)
    if max(abs(component_point[0]-point[1]),abs(component_point[0]-sum(component_point[1:])))>1e-11:
        raise RuntimeError('q33 screening/projector low-order closure failed')
    return dict(left_sites=list(left_sites),
        occupations={name:dict(value=float(value),whole_chain_jackknife_SE=float(np.sqrt(covariance[i,i])))
                     for i,(name,value) in enumerate(zip(names,values))},
        occupation_covariance=covariance.tolist(), occupation_order=list(names),
        cross_screening_coherence_squared=coherence,
        doublet_block_purity_numerator=purity,
        q33_linear_decomposition={name:dict(value=float(value),whole_chain_jackknife_SE=float(np.sqrt(component_covariance[i,i])))
                                  for i,(name,value) in enumerate(zip(('full','low','high'),component_point))},
        q33_component_covariance=component_covariance.tolist(),
        q33_operator_split_diagnostics=split.diagnostics,
        low_order_definition='Hilbert-Schmidt span of I, ALL pair spin products and symmetrized two-pair products; high is signed, not a probability or independence from every conventional observable.',
        conditional_ratio_bias_removed=False, strict_entanglement_spectrum_reported=False,
        phase_claim=False, mixing_certified=False,
        interpretation='Occupation and coherent switching between doublet/quartet screening, not an exclusive QSL fingerprint.')


def analyze(config_path: Path, output_path: Path | None = None) -> dict:
    """Apply a fixed set of cuts to every available registered quality cache."""
    spec = json.loads(config_path.read_text())
    output = Path(spec['output']) if output_path is None else output_path
    if output.exists():
        raise FileExistsError('use a new analysis output; preserve previous evidence')
    records, missing = [], []
    for case in spec['inputs']:
        directory = Path(case['directory'])
        if not all((directory/name).exists() for name in ('registration.json','results.json','physics.npz')):
            missing.append(case)
            continue
        reg = read_sealed(directory/'registration.json')
        result = read_sealed(directory/'results.json')
        digest = file_sha256(directory/'physics.npz')
        if (digest != result['raw_sha256'] or result['registration_sha256'] != reg['result_sha256']
                or result['params_sha256'] != reg['params_sha256']):
            raise ValueError('cache, state or registration identity changed')
        with np.load(directory/'physics.npz', allow_pickle=False) as archive:
            ids = archive['chain_ids']
            for cut, settings in reg['config']['regions'].items():
                if len(settings['sites']) != 6:
                    continue
                partitions = {'rows':tuple(settings['left'])}
                if settings.get('compact_triangular_2x3',False):
                    partitions['triangles'] = (0,3,4)
                for partition,left in partitions.items():
                    records.append(dict(label=case['label'], j2=reg['config']['problem']['j2'],
                        step=result['step'], cut=cut, partition=partition, sites=settings['sites'],
                        raw_sha256=digest, params_sha256=result['params_sha256'],
                        readout=screening_chain_readout(archive[cut+'__per_record_matrices'],ids,left)))
    report = dict(records=records,missing=missing,config=spec,
        actual_output=str(output), exploratory_analysis=True,
        analysis_sources_sha256={str(p):file_sha256(p) for p in (Path(__file__),Path('su2_nqs/vmc/chain_quadratic.py'),Path('su2_nqs/benchmarks/spin_operator_information.py'),Path('su2_nqs/benchmarks/spin_partition_readout.py'))},
        new_samples=0,new_training=0,new_H_calls=0,new_amplitude_queries=0,
        phase_claim=False,strict_entanglement_spectrum=False,
        warning='Independent-chain statistics, not an equilibration or ground-state error certificate; compare own quality drift before physical phase interpretation.')
    atomic_json(output,report)
    print(json.dumps(dict(records=len(records),missing=len(missing),output=str(output))))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    analyze(args.config,args.output)
