"""Finite-size fits with free exponents and explicit nonuniversal amplitudes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Fit:
    """Fit value and descriptive residual; residual is not statistical error."""

    value: float
    intercept: float
    rms: float
    points: int

    def to_dict(self) -> dict:
        return asdict(self)


def power_fit(x: np.ndarray, y: np.ndarray) -> Fit:
    """Fit y=A*x^p using log least squares, returning p without fixing it."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 3 or np.any(x <= 0) or np.any(y <= 0) or not np.all(np.isfinite(x + y)):
        raise ValueError("at least three positive finite x,y observations required")
    design = np.stack((np.log(x), np.ones(len(x))), axis=1)
    slope, intercept = np.linalg.lstsq(design, np.log(y), rcond=None)[0]
    return Fit(float(slope), float(intercept), float(np.sqrt(np.mean((design @ [slope, intercept] - np.log(y)) ** 2))), len(x))


def fit_eta(n: int, correlations: np.ndarray, r_min: int = 4) -> Fit:
    """Fit C(r)=A*[n/pi*sin(pi*r/n)]^-eta, r_min<=r<=n/2."""
    r = np.arange(r_min, n // 2 + 1)
    if len(correlations) != n or len(r) < 3:
        raise ValueError("correlation vector must have length n and fit at least three distances")
    fit = power_fit(n / np.pi * np.sin(np.pi * r / n), np.asarray(correlations)[r])
    return Fit(-fit.value, fit.intercept, fit.rms, fit.points)


def fit_c(n: int, ells: np.ndarray, entropies: np.ndarray, order: float = 2) -> Fit:
    """PBC entropy slope c/6*(1+1/order), including order=1."""
    ells, entropies = np.asarray(ells), np.asarray(entropies, dtype=float)
    if len(ells) < 3 or entropies.shape != ells.shape or np.any(ells <= 0) or np.any(ells >= n) or not np.isfinite(order) or order <= 0 or not np.all(np.isfinite(entropies)):
        raise ValueError("invalid interval/entropy observations or Renyi order")
    x = np.log(n / np.pi * np.sin(np.pi * ells / n))
    design = np.stack((x, np.ones(len(x))), axis=1)
    slope, intercept = np.linalg.lstsq(design, entropies, rcond=None)[0]
    return Fit(float(slope * 6 / (1 + 1 / order)), float(intercept),
               float(np.sqrt(np.mean((design @ [slope, intercept] - entropies) ** 2))), len(x))


def collapse_fit(sizes: np.ndarray, fields: np.ndarray, chi_total: np.ndarray,
                 degree: int = 6, correction: bool = True,
                 chi_se: np.ndarray | None = None) -> dict:
    """Fit hc,nu with fixed physical-error weights and a free scaling curve.

    chi=L^(2/nu)*[f((h-hc)L^(1/nu))+g(x)/L]. The linear curve coefficients
    are eliminated in physical measurement units. Reweighting residuals by
    L^(-2/nu) while varying nu would suppress noise at smaller nu.
    Supplied positive SE are fixed; without them, fixed per-size observed
    means balance fractional errors. Descriptive residuals are not confidence
    intervals. No exponent is fixed to its Ising value.
    """
    from scipy.optimize import least_squares

    sizes,fields,chi = (np.asarray(a,dtype=float) for a in (sizes,fields,chi_total))
    if (sizes.ndim != 1 or fields.shape != sizes.shape or chi.shape != sizes.shape
        or len(np.unique(sizes)) < 3 or len(sizes) < 2*(degree+1)+4
        or np.any(chi<=0) or np.any(sizes<=0) or np.ptp(fields)<=0
        or not np.all(np.isfinite(sizes+fields+chi))):
        raise ValueError("need positive finite data from >=3 sizes and a resolved critical window")
    size_means = np.array([np.mean(chi[sizes==n]) for n in sizes])
    weights = size_means if chi_se is None else np.asarray(chi_se,dtype=float)
    if weights.shape != chi.shape or not np.all(np.isfinite(weights)) or np.any(weights<=0):
        raise ValueError("chi SE must be a positive finite vector matching observations")

    def project(parameters):
        hc,nu = parameters
        x = (fields-hc)*sizes**(1/nu)
        x_scale = max(1.,float(np.max(abs(x))))
        design = np.polynomial.chebyshev.chebvander(x/x_scale,degree)
        if correction:
            design = np.concatenate((design,design/sizes[:,None]),axis=1)
        physical_design = design*sizes[:,None]**(2/nu)
        coefficients = np.linalg.lstsq(physical_design/weights[:,None],chi/weights,rcond=None)[0]
        predicted = physical_design@coefficients
        return (chi-predicted)/weights,coefficients,predicted,x_scale

    solution = least_squares(lambda parameters:project(parameters)[0],
        [float(np.median(fields)),1.],bounds=([float(fields.min()),.7],[float(fields.max()),1.3]),
        xtol=1e-11,ftol=1e-11,gtol=1e-11)
    residual,coefficients,predicted,x_scale = project(solution.x)
    return {"hc":float(solution.x[0]),"nu":float(solution.x[1]),
        "relative_rms":float(np.sqrt(np.mean(((chi-predicted)/size_means)**2))),
        "objective_rms":float(np.sqrt(np.mean(residual**2))),
        "objective_normalization":"fixed_physical_standard_errors" if chi_se is not None else "fixed_per_size_observed_means",
        "success":bool(solution.success),"at_bound":bool(np.any(solution.active_mask)),
        "degree":degree,"correction_L_inverse":correction,"points":len(chi),"x_scale":x_scale,
        "scaling_coefficients":coefficients[:degree+1].tolist(),
        "correction_coefficients":coefficients[degree+1:].tolist() if correction else []}


def phase0_gate(hc: float | None, nu: float | None, eta: float | None,
                observable_checks_passed: bool, sizes: list[int], seeds: int) -> dict:
    """Fail closed: small-size/single-seed smoke runs cannot establish Phase 0."""
    checks = {"hc": hc is not None and np.isfinite(hc) and abs(hc - 1) <= 0.02,
              "nu": nu is not None and np.isfinite(nu) and abs(nu - 1) <= 0.05,
              "eta": eta is not None and np.isfinite(eta) and abs(eta - 0.25) <= 0.03,
              "independent_observable_accuracy": bool(observable_checks_passed),
              "size_coverage": len(set(sizes)) >= 3 and min(sizes, default=0) >= 16 and max(sizes, default=0) >= 32,
              "multiple_seeds": seeds >= 3}
    return {"passed": bool(all(checks.values())), "checks": checks,
            "missing": [name for name, passed in checks.items() if not passed]}
