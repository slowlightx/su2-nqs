"""Independent synthetic check of parameter-dependent residual normalization."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from fnqs.artifacts import source_hashes,write_json
from fnqs.scaling import collapse_fit


def physical_fit(n: np.ndarray,h: np.ndarray,chi: np.ndarray,sigma: np.ndarray) -> dict:
    """Variable projection in fixed physical error units, with a free curve."""
    def project(parameters):
        hc,nu=parameters
        x=(h-hc)*n**(1/nu)
        v=np.polynomial.chebyshev.chebvander(x/max(1.,np.max(abs(x))),4)
        design=np.concatenate((v,v/n[:,None]),axis=1)*n[:,None]**(2/nu)/sigma[:,None]
        y=chi/sigma
        coefficients=np.linalg.lstsq(design,y,rcond=None)[0]
        return y-design@coefficients
    fit=least_squares(project,[float(np.median(h)),1.],bounds=([h.min(),.7],[h.max(),1.3]),
                      xtol=1e-11,ftol=1e-11,gtol=1e-11)
    return {"hc":float(fit.x[0]),"nu":float(fit.x[1]),"at_bound":bool(np.any(fit.active_mask)),
            "rms_in_fixed_error_units":float(np.sqrt(np.mean(project(fit.x)**2)))}


def main() -> None:
    true_nu,true_hc=.9,1.07
    sizes=np.array([16,24,32,48,64])
    n=np.repeat(sizes,41).astype(float)
    x=np.tile(np.linspace(-1,1,41),len(sizes))
    h=true_hc+x/n**(1/true_nu)
    # Exact degree-four scaling curve and degree-two 1/L correction. Neither
    # fitting algorithm is told these coefficients or the true hc/nu.
    truth=.03*n**(2/true_nu)*(1-.25*x*x+.05*x**4+(.4-.1*x*x)/n)
    sigma=.01*truth
    rows=[]
    for seed in (17,29,43):
        chi=truth+np.random.default_rng(seed).normal(size=len(truth))*sigma
        old=collapse_fit(n,h,chi,degree=4,correction=True)
        fixed=physical_fit(n,h,chi,sigma)
        rows.append({"seed":seed,"old":{"hc":old["hc"],"nu":old["nu"],"at_bound":old["at_bound"]},
                     "fixed_physical_error_units":fixed})
    output={"kind":"independent_scaling_noise_probe","source_sha256":source_hashes(),
            "launcher_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "true_nu":true_nu,"true_hc":true_hc,"relative_noise_sigma":.01,"rows":rows,
            "neural_result":False,"changes_running_experiments":False}
    write_json(Path("fnqs/results/scaling_noise_probe_v6.json"),output)
    print(rows)


if __name__ == "__main__":main()
