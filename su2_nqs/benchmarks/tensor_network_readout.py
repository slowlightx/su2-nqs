"""Optional tensor-network references in the repository's physical convention.

No TeNPy import is required for regional measurements. Spin labels are doubled
integers; physical bit 0 is down and the first region site is most significant.
Broken-symmetry candidates retain p(q), but are never silently SU(2)-twirled.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any, Sequence

import numpy as np

from .regional_su2_readout import regional_schur_columns, su2_multiplicity_readout


def physical_spin_block_readout(rho: np.ndarray, *, tolerance: float = 1e-9) -> dict:
    """Measure p(q) for any physical regional state and test SU2 factorization.

    The block traces are physical observables even when SU2 is broken. Only
    invariant inputs receive the singlet multiplicity/entropy interpretation.
    Neither twirling nor positivity clipping is performed.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or not np.isfinite(rho).all():
        raise ValueError('rho must be finite and square')
    d = rho.shape[0]
    if d < 2 or d & (d-1) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('rho requires a spin-half power-of-two dimension and positive tolerance')
    trace_error = float(abs(np.trace(rho)-1))
    hermiticity = float(np.linalg.norm(rho-rho.conj().T))
    minimum = float(np.linalg.eigvalsh(rho).min())
    if max(trace_error,hermiticity,-minimum) > tolerance:
        raise ValueError('rho is not normalized, Hermitian and positive')
    u,labels = regional_schur_columns(d.bit_length()-1)
    x = u.conj().T @ rho @ u
    invariant = np.zeros_like(x)
    weights = {}
    for q in sorted({label[0] for label in labels}):
        blocks = [[i for i,label in enumerate(labels) if label[:2]==(q,m)]
                  for m in range(-q,q+1,2)]
        reduced = sum(x[np.ix_(ids,ids)] for ids in blocks)
        weights[str(q)] = float(np.trace(reduced).real)
        for ids in blocks:
            invariant[np.ix_(ids,ids)] = reduced/(q+1)
    defect = float(np.linalg.norm(x-invariant))
    result = dict(p_q=weights,spin_squared_mean=sum(p*int(q)*(int(q)+2)/4 for q,p in weights.items()),
                  su2_factorization_frobenius=defect,strict_su2_eligible=defect<=tolerance,
                  trace_error=trace_error,hermiticity_frobenius=hermiticity,minimum_eigenvalue=minimum,
                  warning='p(q) is measurable without SU2; multiplicity entropy needs invariant rho. No twirl is applied.')
    if defect<=tolerance:
        result['invariant_readout'] = su2_multiplicity_readout(rho,tolerance=tolerance)
    return result


def tenpy_region_density(psi: Any, sites: Sequence[int]) -> np.ndarray:
    """Read a finite MPS region in the REQUESTED site order and down/up basis.

    TeNPy sorts the sites internally and may sort physical states by charge.
    Both orderings are explicitly converted, not guessed from state labels.
    """
    requested = tuple(sites)
    if (not requested or len(requested)>8 or
        any(isinstance(i,bool) or not isinstance(i,Integral) for i in requested) or
        len(set(requested))!=len(requested) or min(requested)<0 or max(requested)>=psi.L):
        raise ValueError('region requires 1..8 unique valid integer MPS sites')
    ordered = sorted(requested)
    rho = psi.get_rho_segment(ordered)
    n = len(ordered)
    labels = [f'p{i}' for i in range(n)]+[f'p{i}*' for i in range(n)]
    tensor = rho.transpose(labels).to_ndarray()
    for axis,site_index in enumerate(ordered):
        sz = psi.sites[site_index].get_op('Sz').to_ndarray()
        permutation = np.argsort(np.diag(sz).real)
        if (sz.shape!=(2,2) or np.linalg.norm(sz-np.diag(np.diag(sz)))>1e-14 or
            not np.allclose(np.diag(sz).real[permutation],[-.5,.5],atol=1e-14)):
            raise ValueError('site is not a spin-half Sz eigenbasis')
        tensor = np.take(np.take(tensor,permutation,axis=axis),permutation,axis=n+axis)
    axes = [ordered.index(i) for i in requested]
    tensor = tensor.transpose(axes+[n+i for i in axes])
    return np.asarray(tensor.reshape(2**n,2**n),dtype=np.complex128)


def tenpy_graph_model(n_sites: int, bonds: Sequence[tuple[int,int,float]], *, conserve: str | None = 'Sz') -> Any:
    """Construct a finite-MPS Heisenberg MPO from explicit PHYSICAL bonds.

    The MPS has open virtual ends; physical boundary conditions reside ONLY
    in the supplied graph. This avoids changing the triangular metric to the
    default TeNPy lattice convention. No dense many-body operator is formed.
    """
    if isinstance(n_sites,bool) or not isinstance(n_sites,Integral) or n_sites<2:
        raise ValueError('n_sites must be an integer >=2')
    checked=[]
    for i,j,weight in bonds:
        if (any(isinstance(k,bool) or not isinstance(k,Integral) for k in (i,j)) or
            not 0<=i<j<n_sites or not np.isfinite(weight)):
            raise ValueError('bonds require ordered integer sites and finite real weights')
        checked.append((int(i),int(j),float(weight)))
    from tenpy.models.model import CouplingMPOModel
    from tenpy.models.lattice import Chain
    from tenpy.networks.site import SpinHalfSite
    class GraphHeisenberg(CouplingMPOModel):
        def init_lattice(self, params):
            params.touch('n_sites','bonds','conserve')
            return Chain(n_sites,SpinHalfSite(conserve=conserve),bc='open',bc_MPS='finite')
        def init_terms(self, params):
            for i,j,w in checked:
                self.add_coupling_term(w,i,j,'Sz','Sz')
                self.add_coupling_term(w/2,i,j,'Sp','Sm',plus_hc=True)
    return GraphHeisenberg(dict(n_sites=int(n_sites),bonds=checked,conserve=conserve))


__all__ = ['physical_spin_block_readout','tenpy_region_density','tenpy_graph_model']
