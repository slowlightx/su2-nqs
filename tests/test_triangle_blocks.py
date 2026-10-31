"""Independent physical bit-permutation/Casimir checks of cached local data."""
import numpy as np
import pytest

from su2_nqs.su2.triangle_blocks import (
    triangle_block_tables, nine_site_output, real_c3_synthesis_basis,
    real_multiplicity_frame,
)


def test_complete_nine_site_columns_have_exact_Casimir_and_all_magnetic_partners():
    tables = triangle_block_tables()
    n = 9
    bits = (np.arange(512)[:,None] >> np.arange(8,-1,-1)) & 1
    m = (2*bits-1).sum(1)
    columns=[];casimirs=[]
    for q,u in tables['nine'].items():
        for magnetic in range(-q,q+1,2):
            columns.extend((u*(m==magnetic)[:,None]).T)
            casimirs.extend([q*(q+2)/4]*u.shape[1])
    u=np.column_stack(columns)
    np.testing.assert_allclose(u.T@u,np.eye(512),atol=7e-15,rtol=0)
    action=n*(4-n)/4*u
    # S^2 = sum_(i<j) P_ij + N(4-N)/4, independent of the CG implementation.
    for i in range(n):
        for j in range(i):
            swapped=bits.copy();swapped[:,[i,j]]=bits[:,[j,i]]
            packed=swapped@(1<<np.arange(8,-1,-1))
            action+=u[packed]
    np.testing.assert_allclose(action,u*np.asarray(casimirs),atol=3e-14,rtol=0)


@pytest.mark.parametrize('q',[1,3,5,7,9])
def test_local_C3_representation_agrees_with_actual_nine_site_permutation(q):
    d=triangle_block_tables();u=d['nine'][q]
    bits=(np.arange(512)[:,None]>>np.arange(8,-1,-1))&1
    image=[3*((p+1)%3)+(i+1)%3 for p in range(3) for i in range(3)]
    permuted=np.empty_like(bits);permuted[:,image]=bits
    packed=permuted@(1<<np.arange(8,-1,-1))
    np.testing.assert_allclose(u[np.argsort(packed)],u@d['rotations'][q],atol=2e-15,rtol=0)
    np.testing.assert_allclose(d['binary_overlaps'][q].T@d['binary_overlaps'][q],
                               np.eye(u.shape[1]),atol=3e-15,rtol=0)


@pytest.mark.parametrize('cap',[2,9,16])
def test_independent_real_C3_maps_support_paper_width_without_dense_image_SVD(cap):
    d=triangle_block_tables()
    for q,r in d['rotations'].items():
        target=nine_site_output(q,cap)
        basis=real_c3_synthesis_basis(r,target)
        flat=basis.reshape(len(basis),-1)
        np.testing.assert_allclose(flat@flat.T,np.eye(len(basis)),atol=3e-15,rtol=0)
        s=real_multiplicity_frame(target.time_reversal_multiplicity())
        rout=(s.conj().T@target.rotation()@s).real
        np.testing.assert_allclose(np.einsum('ij,pjk->pik',r,basis),
                                   np.einsum('pij,jk->pik',basis,rout),atol=2e-15,rtol=0)


def test_invalid_realification_and_partial_irrep_requests_reject():
    with pytest.raises(ValueError):
        real_multiplicity_frame(np.array([[1.,1.],[0.,1.]]))
    for q,cap in ((0,16),(1,True),(3,1)):
        with pytest.raises(ValueError):nine_site_output(q,cap)
