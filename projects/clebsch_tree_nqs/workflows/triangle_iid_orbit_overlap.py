"""Zero-query, paired overlap postprocessing of a frozen full-A1 IID audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256


def paired_projected_overlap(psi: np.ndarray, density: np.ndarray,
                             groups: np.ndarray, *, seed: int) -> dict:
    """Estimate complete-A1 endpoint overlap from common IID importance rows.

    Raw cross/norm integrals are unbiased. Their normalized overlap and FS
    distance are finite-sample ratios, not exact state overlaps. Group errors
    retain the common-row correlation of both endpoints and their norms.
    """
    p, q, g = np.asarray(psi, complex), np.asarray(density, float), np.asarray(groups)
    if (p.ndim != 2 or p.shape[1] != 2 or q.shape != (len(p),) or g.shape != q.shape
            or not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(q <= 0)):
        raise ValueError('finite paired amplitudes and positive proposal density required')
    labels, counts = np.unique(g, return_counts=True)
    if len(labels) < 4 or len(set(counts)) != 1:
        raise ValueError('at least four equal-size independent groups required')
    norms = abs(p)**2/q[:, None]
    cross = p[:, 0].conj()*p[:, 1]/q

    def estimate(ids):
        z = norms[ids].mean(0)
        if np.any(z <= 0):
            raise FloatingPointError('sampled endpoint norm is zero')
        c = cross[ids].mean()
        overlap = c/np.sqrt(np.prod(z))
        magnitude = abs(overlap)
        if not np.isfinite(overlap) or magnitude > 1+2e-12:
            raise ArithmeticError('common-row Gram matrix violates Cauchy-Schwarz')
        return np.asarray([c.real, c.imag, overlap.real, overlap.imag,
                           magnitude, np.arccos(min(1., magnitude))])

    ids = np.arange(len(p))
    point = estimate(ids)
    leave = np.asarray([estimate(ids[g != label]) for label in labels])
    jackknife = np.sqrt((len(labels)-1)/len(labels)*np.sum((leave-leave.mean(0))**2, axis=0))
    rng = np.random.default_rng(seed)
    by_group = [ids[g == label] for label in labels]
    bootstrap = np.asarray([estimate(np.concatenate([by_group[i]
        for i in rng.integers(len(labels), size=len(labels))])) for _ in range(512)])
    return dict(metric_order=['cross_integral_real', 'cross_integral_imaginary',
        'normalized_overlap_real', 'normalized_overlap_imaginary',
        'normalized_overlap_magnitude', 'endpoint_FS_distance_radians'],
        values=point.tolist(), independent_block_jackknife_SE=jackknife.tolist(),
        joint_block_bootstrap_95=np.quantile(bootstrap, [.025, .975], axis=0).T.tolist(),
        projected_norm_estimates=norms.mean(0).tolist(), records=len(p), independent_groups=len(labels),
        cross_convention='mean(conj(Phi_old)*Phi_new/Q)',
        complete_A1_endpoint_overlap_not_raw_or_parent_distance=True,
        normalized_overlap_and_FS_have_finite_sample_bias=True,
        no_additional_amplitude_or_H_queries=True, optimization_updates=0)


def run(audit: Path) -> dict:
    """Write an independent immutable postprocessing artifact, leaving run files intact."""
    target = audit/'overlap_postprocessing.json'
    if target.exists():
        raise FileExistsError('overlap artifact already exists')
    result = json.loads((audit/'results.json').read_text())
    if result['status'] != 'complete_fresh_IID_full_A1_energy_variance_audit':
        raise ValueError('completed untouched audit products required')
    product = audit/'untouched_products.npz'
    data = np.load(product)
    seed = 2610269022
    overlap = paired_projected_overlap(data['psi'], data['proposal_density'], data['groups'], seed=seed)
    report = dict(overlap=overlap, bootstrap_seed=seed,
        endpoint_checkpoint_sha256=result['registration']['checkpoint_sha256'],
        audit_registration_sha256=file_sha256(audit/'registration.json'),
        untouched_products_sha256=file_sha256(product),
        postprocessor_source_sha256=file_sha256(Path(__file__)),
        postprocessing_does_not_change_frozen_audit=True)
    atomic_json(target, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    run(parser.parse_args().audit)
