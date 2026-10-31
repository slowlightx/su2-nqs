"""Full-support importance moments for a local SU(2)-scalar tangent probe.

At a strict A1 carrier, <H psi|P_A1 O psi>=<H psi|O psi>. This permits a
projected first derivative without a Reynolds sum, but NOT a projected
tangent metric, finite-step variational energy, or dressed-state variance.
"""

import numpy as np


def importance_tangent_moments(psi: np.ndarray, hpsi: np.ndarray, actions: np.ndarray, proposal_squared: np.ndarray) -> dict:
    """Self-normalized moments under a positive, unnormalized proposal.

    psi and hpsi may vanish; no division by physical amplitudes occurs.
    Full support is essential: carrier nodes can contribute to variance and
    the energy derivative. Finite-sample self-normalization is biased.
    """
    psi,hpsi,actions,q=map(np.asarray,(psi,hpsi,actions,proposal_squared))
    if psi.ndim!=1 or hpsi.shape!=psi.shape or actions.ndim!=2 or actions.shape[0]!=len(psi) or q.shape!=psi.shape or np.any(q<=0):
        raise ValueError("incompatible shapes or non-positive proposal")
    if not all(np.isfinite(v).all() for v in (psi,hpsi,actions,q)):
        raise ValueError("finite arrays required")
    weights=np.abs(psi)**2/q
    norm=np.sum(weights)
    if norm<=0:
        raise ValueError("carrier has zero sampled norm")
    energy=float(np.sum(psi.conj()*hpsi/q).real/norm)
    variance=float(np.sum(np.abs(hpsi-energy*psi)**2/q)/norm)
    means=np.sum(psi.conj()[:,None]*actions/q[:,None],axis=0)/norm
    gradients=2*np.sum((hpsi-energy*psi).conj()[:,None]*actions/q[:,None],axis=0).real/norm
    return {"energy_total":energy,"variance_total":variance,"operator_means_real":means.real.tolist(),
            "energy_derivative_total":gradients.tolist(),"operator_mean_imaginary_max":float(np.max(np.abs(means.imag))),
            "importance_ess_fraction":float(norm**2/(len(q)*np.sum(weights**2)))}
