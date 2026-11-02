"""Conditional finite entanglement-spectrum tests for a positive SO(3) rotor.

Body-axis anisotropy cannot arbitrarily separate the lowest S=2 branch from
the lowest S=1 branch of ``H_rotor=sum_a A_a J_a**2`` with all ``A_a>=0``.
The variational bound is ``E_2,min <= 3 E_1,min``, even for a general triaxial
inertia. Applying it to a measured magnetic-charge spectrum requires the
spin assignment and the scalar ground branch to be independently checked.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any, Sequence

import numpy as np


def integer_spin_generators(q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return angular-momentum matrices for doubled integer spin ``q=2S``.

    SO(3) rotor sectors have even nonnegative q. The basis is the physical
    doubled magnetic ordering ``m=-q,-q+2,...,q``; no half-integer Hilbert
    labels are stored. Matrix entries can of course be real/complex floats.
    """
    if isinstance(q, bool) or not isinstance(q, Integral) or q < 0 or q % 2:
        raise ValueError('an even nonnegative doubled spin q is required')
    q=int(q)
    m=np.arange(-q,q+1,2,dtype=np.int64)
    plus=np.zeros((q+1,q+1),dtype=np.complex128)
    for column,magnetic in enumerate(m[:-1]):
        plus[column+1,column]=.5*np.sqrt((q-int(magnetic))*(q+int(magnetic)+2))
    minus=plus.conj().T
    return .5*(plus+minus), (plus-minus)/(2j), np.diag(m/2).astype(np.complex128)


def quadratic_rotor_levels(coefficients: Sequence[float], q: int) -> np.ndarray:
    """Actual body-axis eigenlevels for positive inverse inertia coefficients."""
    values=np.asarray(coefficients,dtype=np.float64)
    if values.shape!=(3,) or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('three finite nonnegative inverse inertia coefficients required')
    matrices=integer_spin_generators(q)
    hamiltonian=sum(a*(j @ j) for a,j in zip(values,matrices))
    return np.linalg.eigvalsh(hamiltonian)


def positive_rotor_spin2_bound(coefficients: Sequence[float]) -> dict[str, Any]:
    """Evaluate and independently document ``E2_min <= 3 E1_min``.

    Order ``a<=b<=c``. In the Cartesian spin-one representation the three
    eigenvalues are ``b+c,a+c,a+b``, hence ``E1_min=a+b``. In spin two, the
    trial state with ``m_body=0`` along the c axis obeys ``<J_c^2>=0`` and
    ``<J_a^2>=<J_b^2>=3``. Its energy is ``3(a+b)``. Rayleigh's principle
    gives the bound; no isotropic-inertia assumption is used.
    """
    values=np.asarray(coefficients,dtype=np.float64)
    if values.shape!=(3,) or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('three finite nonnegative inverse inertia coefficients required')
    ordered=np.sort(values)
    e1=quadratic_rotor_levels(values,2)
    e2=quadratic_rotor_levels(values,4)
    trial=np.zeros(5,dtype=np.complex128);trial[2]=1.
    axes=integer_spin_generators(4)
    trial_expectations=[float(np.vdot(trial,j @ j @ trial).real) for j in axes]
    trial_energy=float(ordered @ trial_expectations)
    analytic_e1=float(ordered[:2].sum())
    return dict(coefficients=values.tolist(),spin1_levels=e1.tolist(),spin2_levels=e2.tolist(),
        analytic_spin1_minimum=analytic_e1, spin1_minimum=float(e1[0]),
        spin2_minimum=float(e2[0]), bound=3*analytic_e1,
        variational_trial_expectations=trial_expectations,
        variational_trial_energy=trial_energy, bound_slack=float(3*analytic_e1-e2[0]),
        spin1_formula_error=float(abs(e1[0]-analytic_e1)),
        trial_formula_error=float(abs(trial_energy-3*analytic_e1)),
        hypothesis='complete positive quadratic SO(3) rotor, spin-zero energy chosen as zero')


def conditional_magnetic_rotor_test(
    magnetic_sector_energies: dict[str, Sequence[float | None]],
) -> dict[str, Any]:
    """Compute the rotor-bound margin without assigning unverified q labels.

    Input uses physical doubled magnetic m=0,2,4. The numbers become a
    spin-one/spin-two test ONLY if the lowest m=0 branch is spin zero, the
    lowest m=2 branch is spin one, and the proposed spectrum is the complete
    positive quadratic rotor family. This routine does not certify any of
    those premises. In a SU(2) rotor, m=4 includes the lowest S=2 multiplet;
    therefore its minimum must lie below ``3*(xi_m2_min-xi_m0_min)``.
    """
    minima={}
    for charge in ('0','2','4'):
        values=np.asarray([x for x in magnetic_sector_energies[charge] if x is not None],dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('finite spectra for physical m=0,2,4 required')
        minima[charge]=float(values.min())
    d1=minima['2']-minima['0']
    d2=minima['4']-minima['0']
    return dict(magnetic_sector_minima=minima,magnetic_m1_gap=d1,magnetic_m2_gap=d2,
        positive_rotor_bound_if_premises_hold=3*d1,
        excess_above_bound_if_premises_hold=d2-3*d1,
        ratio_if_premises_hold=d2/d1 if d1>0 else None,
        positive_rotor_template_rejected_if_premises_hold=bool(d2>3*d1),
        premises_verified=False,q_labels_assigned=False,bulk_magnetic_order_excluded=False,
        qsl_certified=False,
        warning='A conditional finite-region quadratic-rotor template check, not a bulk phase certificate; full spin assignment and rotation covariance remain necessary.')
