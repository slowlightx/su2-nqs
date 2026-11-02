"""Exact-small operator innovations after reoptimizing a complete anchor space.

An independent direction need not lower energy. In particular, removing the
Hamiltonian tangent makes its first derivative at the old Ritz state zero.
Its usefulness must be tested after the Hamiltonian step, against a second
Hamiltonian step at equal variational dimension. No ED reference enters this
construction, and all operations are covariant under a unitary basis change.
"""

from __future__ import annotations

import numpy as np

from .spectral_error_tomography import lowest_subspace_state, orthonormal_column_space


def _remove(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    residual=np.array(vector,dtype=np.complex128,copy=True)
    for _ in range(2):
        residual-=basis@(basis.conj().T@residual)
    return residual


def _capture(direction: np.ndarray, gradient: np.ndarray) -> float:
    denominator=float(np.vdot(direction,direction).real*np.vdot(gradient,gradient).real)
    return float(abs(np.vdot(direction,gradient))**2/denominator) if denominator>1e-28 else 0.


def operator_innovation(hamiltonian: np.ndarray, anchor: np.ndarray,
                        operator: np.ndarray, *, tolerance: float=1e-10) -> dict:
    """Reoptimize the anchor, then remove it and its first full-H update.

    Inputs are matrices in one orthonormal physical sector (for example
    exact S=0,Gamma-A1). The operator must already represent the complete
    physical operator; compressing factors separately is not equivalent.
    Returned arrays include all trial bases/states, not a selected winner.
    ``innovation`` is (I-P_span(B,H psi_B)) O psi_B, with psi_B the current
    Ritz state, not the original weak carrier used to construct B.
    """
    h=np.asarray(hamiltonian,dtype=np.complex128)
    op=np.asarray(operator,dtype=np.complex128)
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or op.shape!=h.shape:
        raise ValueError('matching square Hamiltonian/operator required')
    if not all(np.isfinite(x).all() and np.allclose(x,x.conj().T,atol=2e-12,rtol=0) for x in (h,op)):
        raise ValueError('finite Hermitian Hamiltonian/operator required')
    b,rank,_=orthonormal_column_space(anchor,relative_tolerance=tolerance)
    if b.shape[0]!=len(h):raise ValueError('anchor dimension mismatch')
    psi,_=lowest_subspace_state(b,h,relative_tolerance=tolerance)
    direction=op@psi
    residual=_remove(direction,b)
    bh,hrank,_=orthonormal_column_space(np.column_stack((b,h@psi)),relative_tolerance=tolerance)
    psi_h,_=lowest_subspace_state(bh,h,relative_tolerance=tolerance)
    innovation=_remove(direction,bh)
    bhh,hhrank,_=orthonormal_column_space(np.column_stack((bh,h@psi_h)),relative_tolerance=tolerance)
    psi_hh,_=lowest_subspace_state(bhh,h,relative_tolerance=tolerance)
    direct,drank=lowest_subspace_state(np.column_stack((b,residual)),h,relative_tolerance=tolerance)
    conditional,crank=lowest_subspace_state(np.column_stack((bh,innovation)),h,relative_tolerance=tolerance)
    gradient=h@psi-np.vdot(psi,h@psi)*psi
    gradient_h=h@psi_h-np.vdot(psi_h,h@psi_h)*psi_h
    norm=float(np.vdot(direction,direction).real)
    rnorm=float(np.vdot(residual,residual).real)
    inorm=float(np.vdot(innovation,innovation).real)
    diagnostics=dict(
        anchor_rank=rank,h_once_rank=hrank,h_twice_rank=hhrank,direct_rank=drank,conditional_rank=crank,
        outside_anchor_fraction=rnorm/norm if norm else 0.,
        outside_anchor_and_h_fraction=inorm/norm if norm else 0.,
        surviving_residual_fraction=inorm/rnorm if rnorm>1e-28 else 0.,
        original_gradient_capture_fraction=_capture(residual,gradient),
        next_gradient_capture_fraction=_capture(innovation,gradient_h),
        innovation_first_derivative_total=float(2*np.vdot(innovation,gradient).real),
        innovation_orthogonality_max=float(np.max(abs(bh.conj().T@innovation))),
        ritz_stationarity_max=float(np.max(abs(b.conj().T@gradient))),
    )
    return dict(anchor=b,carrier=psi,direction=direction,anchor_residual=residual,
                h_basis=bh,h_once=psi_h,h_twice_basis=bhh,h_twice=psi_hh,
                innovation=innovation,direct=direct,conditional=conditional,diagnostics=diagnostics)
