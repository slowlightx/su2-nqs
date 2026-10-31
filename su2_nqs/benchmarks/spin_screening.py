"""Angular-momentum interpretation of a highest-weight screening texture.

These are bounds derived from existing observables, NOT a new independent
entanglement witness or a QSL-specific indicator. Every Hilbert label is a
doubled integer. The input vector must have fixed m, with its minimal-spin
purity checked by the PHYSICAL Casimir (not a retained-space projection).
"""
from __future__ import annotations

from math import isfinite, sqrt
from numbers import Integral

from ..su2.quantum_numbers import coupling_range, validate_m, validate_q


def block_spin_z(q_left: int, q_right: int, q_total: int, m2: int) -> float:
    """Right-block Sz in a normalized coupled |q_total,m2> vector.

    Independent of internal multiplicities. Coherences between different
    block spins do not enter this expectation: Sz_right commutes with both
    block Casimirs. q=2S and m2=2M; the returned observable is NOT a label.
    """
    ql, qr, qt = (validate_q(q) for q in (q_left, q_right, q_total))
    m = validate_m(qt, m2)
    if qt not in coupling_range(ql, qr):
        raise ValueError('block spins cannot couple to q_total')
    if qt == 0:
        return 0.
    c = qt*(qt+2)
    return (m/2)*(c+qr*(qr+2)-ql*(ql+2))/(2*c)


def screening_weight_bound(*, n_left: int, n_right: int, q_total: int, m2: int,
                           right_spin_z: float, casimir: float,
                           casimir_absolute_error: float = 1e-9) -> dict:
    """Bound weight in block-spin channels antipolarized against total M.

    Only highest-weight m2=+/-q_total>0 is supported. The physical Casimir
    bounds larger-spin contamination; a trace-distance/operator-norm bound
    propagates it to the polarization. Result is conditional on the declared
    input precision, not a rigorously certified floating-point interval.
    """
    for name, n in (('n_left', n_left), ('n_right', n_right)):
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
            raise ValueError(name+' must count a positive integer number of spin halves')
    q = validate_q(q_total); m = validate_m(q, m2)
    if q == 0 or abs(m) != q:
        raise ValueError('nonzero highest-weight magnetic label required')
    if q > n_left+n_right or (n_left+n_right-q) % 2:
        raise ValueError('unreachable region spin')
    if (not all(isfinite(v) for v in (right_spin_z, casimir, casimir_absolute_error))
            or casimir_absolute_error <= 0 or abs(right_spin_z) > n_right/2+casimir_absolute_error):
        raise ValueError('finite physical polarization and positive precision required')
    target = q*(q+2)/4
    if casimir < target-casimir_absolute_error:
        raise ValueError('Casimir below the minimum spin allowed by m2')
    channels = [dict(q_left=ql, q_right=qr,
        aligned_right_spin_z=block_spin_z(ql, qr, q, q))
        for ql in range(n_left % 2, n_left+1, 2)
        for qr in range(n_right % 2, n_right+1, 2)
        if q in coupling_range(ql, qr)]
    minimum = min(row['aligned_right_spin_z'] for row in channels)
    delta = min(1., (max(0., casimir-target)+casimir_absolute_error)/(q+2))
    # ||Sz_right||=n_right/2, trace distance <=2sqrt(delta).
    error = n_right*sqrt(delta)
    aligned = right_spin_z*(1 if m > 0 else -1)
    if aligned < minimum-error-casimir_absolute_error:
        raise ValueError('polarization inconsistent with its certified spin content')
    nominal = max(0., -aligned)/(-minimum) if minimum < 0 else 0.
    conditional = max(0., -aligned-error)/(-minimum) if minimum < 0 else 0.
    return dict(q_total=q, m2=m, n_left=int(n_left), n_right=int(n_right),
        right_spin_z=right_spin_z, minimum_right_spin_z_in_target_sector=minimum,
        possible_channels=channels,
        negative_channels=[row for row in channels if row['aligned_right_spin_z'] < 0],
        nominal_negative_channel_weight_lower_bound=min(1., nominal),
        precision_adjusted_negative_channel_weight_lower_bound=min(1., (1-delta)*conditional),
        nominal_higher_spin_weight_upper_bound=delta, polarization_projection_error_bound=error,
        casimir_absolute_error=casimir_absolute_error,
        conditional_on_input_precision=True, independent_new_observable=False,
        qsl_specific=False)
