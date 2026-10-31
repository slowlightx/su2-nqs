"""Forensic replay comparison: discrete trajectories exact, floats diagnosed.

This does not change or repair a trajectory. Array shapes/dtypes must match;
integer/boolean outputs (including chains, RNG and acceptance) are bit exact.
Finite floating outputs use a declared *relative* vector tolerance, with no
absolute floor that could hide changes in tiny wavefunction amplitudes.
"""
from __future__ import annotations

import jax
import numpy as np


def compare_replay(expected: object, actual: object, *, tolerance: float = 2e-11,
                   diagnostic_prefixes: tuple[str, ...] = ()) -> dict:
    """Return per-leaf diagnostics; never replace actual outputs by expected."""
    if not np.isfinite(tolerance) or not 0 < tolerance <= 2e-11:
        raise ValueError('replay tolerance must be positive and at most 2e-11')
    first, shape = jax.tree_util.tree_flatten_with_path(expected)
    second, other_shape = jax.tree_util.tree_flatten_with_path(actual)
    if shape != other_shape:
        return dict(passed=False, bitwise_exact=False, reason='pytree structure differs', leaves=[])
    leaves = []
    for (path, x), (_, y) in zip(first, second, strict=True):
        a, b = np.asarray(x), np.asarray(y)
        same_type = a.shape == b.shape and a.dtype == b.dtype
        exact = same_type and a.tobytes() == b.tobytes()
        discrete = a.dtype.kind in 'biu'
        finite = same_type and bool(np.isfinite(a).all() and np.isfinite(b).all())
        error = None
        absolute_error = None
        # Dimensionless scalar diagnostics such as a ~1e-14 linear residual
        # need a natural-unit absolute test. Physical amplitudes, energies
        # and parameter arrays NEVER receive this exception.
        scalar_diagnostic = (a.ndim == 0 and any(jax.tree_util.keystr(path).startswith(prefix)
                                                for prefix in diagnostic_prefixes))
        if finite and not discrete:
            scale = max(float(np.max(np.abs(a), initial=0.)),
                        float(np.max(np.abs(b), initial=0.)))
            # Divide before squaring to avoid underflow for tiny amplitudes.
            error = (0. if scale == 0 else float(np.linalg.norm((a-b).ravel()/scale)/
                     max(np.linalg.norm(a.ravel()/scale), np.linalg.norm(b.ravel()/scale))))
            absolute_error = float(np.max(np.abs(a-b), initial=0.))
        passed = exact if discrete else finite and error is not None and (
            error <= tolerance or scalar_diagnostic and absolute_error <= tolerance)
        leaves.append(dict(path=jax.tree_util.keystr(path), shape=list(a.shape), dtype=str(a.dtype),
                           discrete=discrete, bitwise_exact=exact, finite=finite,
                           relative_vector_error=error, absolute_error=absolute_error,
                           natural_unit_scalar_diagnostic=scalar_diagnostic, passed=bool(passed)))
    return dict(protocol='exact_discrete_relative_float_v1', tolerance=tolerance,
                passed=all(row['passed'] for row in leaves),
                bitwise_exact=all(row['bitwise_exact'] for row in leaves),
                leaves=leaves)
