"""Quadratic readouts from independent whole-chain vector estimates."""
from __future__ import annotations

import numpy as np


def whole_chain_squared_mean(values: np.ndarray, chain_ids: np.ndarray) -> dict:
    """Estimate ||E[x]||² without the diagonal norm-noise bias.

    Records in one chain may be correlated. Distinct chain means must be
    independent, identically distributed and equally weighted. The ordered
    distinct-chain U statistic is signed; negative estimates are retained.
    Delete-whole-chain jackknife SE is diagnostic, not an equilibration or
    rare-event certificate. No normalization ratio is taken here.
    """
    x = np.asarray(values, np.complex128)
    ids = np.asarray(chain_ids)
    if (x.ndim < 2 or not len(x) or ids.shape != (len(x),)
            or ids.dtype.kind not in 'iu' or not np.isfinite(x).all()):
        raise ValueError('finite vector records and integer whole-chain IDs required')
    labels, counts = np.unique(ids, return_counts=True)
    if len(labels) < 3 or np.any(counts != counts[0]):
        raise ValueError('at least three equal-size independent whole chains required')
    means = np.stack([x[ids == label].mean(axis=0) for label in labels])
    rows = means.reshape(len(means), -1)
    total = rows.sum(axis=0)
    sum_norm = np.sum(abs(rows)**2)
    size = len(rows)
    numerator = np.vdot(total, total).real-sum_norm
    estimate = float(numerator/(size*(size-1)))
    delete = np.array([(np.vdot(total-row, total-row).real
                       -sum_norm+np.vdot(row, row).real)/((size-1)*(size-2))
                       for row in rows])
    se = float(np.sqrt((size-1)*np.mean((delete-delete.mean())**2)))
    return dict(value=estimate, whole_chain_jackknife_SE=se,
        plug_in_value=float(np.vdot(total/size, total/size).real),
        independent_chains=size, records_per_chain=int(counts[0]),
        negative_estimate_retained=estimate < 0,
        normalization_ratio_bias_present=False, mixing_certified=False)
