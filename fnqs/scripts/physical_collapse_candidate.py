"""Staged collapse fitter using fixed measurement units; no training imports."""

from __future__ import annotations

import numpy as np


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
