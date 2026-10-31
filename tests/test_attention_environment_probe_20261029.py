"""Exact invariants for the isolated N10 environment probe, not CT production."""
import numpy as np
import jax.numpy as jnp

from projects.clebsch_tree_nqs.workflows.attention_environment_probe_20261029 import (
    physical_reference, candidate_frames, protected_projector, init_params, predict, make_dataset, PAIRS,
)


def test_exact_singlet_reference_and_physical_pair_operator():
    u,ops,slices=physical_reference()
    np.testing.assert_allclose(u.T@u,np.eye(42),atol=2e-14)
    raised=np.zeros_like(u)
    for state in range(1024):
        for site in range(10):
            bit=1 << (9-site)
            if not state & bit: raised[state|bit]+=u[state]
    np.testing.assert_allclose(raised,0,atol=2e-14)
    np.testing.assert_allclose(ops,ops.transpose(0,2,1),atol=2e-14)
    # Sum of all pair interactions is (S²-3N/4)/2 in the singlet sector.
    np.testing.assert_allclose(ops.sum(0),-3.75*np.eye(42),atol=2e-14)
    assert [(q,s.stop-s.start) for q,s in slices.items()]==[(1,25),(3,16),(5,1)]
    # Independent computational bit-flip action, before reducing back to 42D.
    rng=np.random.default_rng(11); bonds=rng.normal(size=45); v=rng.normal(size=42)
    psi=u@v; acted=np.zeros(1024)
    for old,amplitude in enumerate(psi):
        for bond,(i,j) in zip(bonds,PAIRS,strict=True):
            bit_i=1 << (9-i); bit_j=1 << (9-j)
            if bool(old & bit_i)==bool(old & bit_j): acted[old]+=.25*bond*amplitude
            else:
                acted[old]-=.25*bond*amplitude
                acted[old^bit_i^bit_j]+=.5*bond*amplitude
    np.testing.assert_allclose(acted,u@np.einsum('p,pij,j->i',bonds,ops,v),atol=3e-14)


def test_psd_kernel_covariance_and_protected_old_physical_space():
    _,ops,slices=physical_reference(); frames=candidate_frames(ops,slices)
    rng=np.random.default_rng(12)
    for q,f in frames.items():
        raw=rng.normal(size=(3,3)); a=raw@raw.T
        p=protected_projector(q,f,a)
        np.testing.assert_allclose(p@p,p,atol=2e-14)
        np.testing.assert_allclose(p@f[:,0],f[:,0],atol=2e-14)
        np.testing.assert_allclose((np.eye(len(f))-f@f.T)@p,0,atol=2e-14)
        gauge=np.linalg.qr(rng.normal(size=(len(f),len(f)))+1j*rng.normal(size=(len(f),len(f))))[0]
        np.testing.assert_allclose(protected_projector(q,gauge@f,a),gauge@p@gauge.conj().T,atol=2e-14)
    x=jnp.asarray(rng.normal(size=(2,2,5,16)))
    raw=rng.normal(size=(2,2,5,3,3)); gram=np.einsum('...ik,...jk->...ij',raw,raw)
    p=init_params(13,16)
    gauge=np.linalg.qr(rng.normal(size=(3,3)))[0]
    transformed=np.einsum('ai,...ab,bj->...ij',gauge,gram,gauge)
    for mode in ('attention','pointwise'):
        original=np.asarray(predict(p,x,jnp.asarray(gram),mode))
        changed=np.asarray(predict(p,x,jnp.asarray(transformed),mode))
        np.testing.assert_allclose(changed,np.einsum('ai,...ab,bj->...ij',gauge,original,gauge),atol=2e-14)
        assert np.linalg.eigvalsh(original).min()>-2e-14


def test_paired_physical_environments_and_holdout_are_separate(tmp_path):
    data,diagnostics=make_dataset(tmp_path)
    pairs=data['couplings'].reshape(-1,2,45)
    local_or_cross=np.array([i < 5 for i,j in PAIRS])
    np.testing.assert_array_equal(pairs[:,0,local_or_cross],pairs[:,1,local_or_cross])
    assert np.max(abs(pairs[:,0,~local_or_cross]-pairs[:,1,~local_or_cross]))>.1
    assert set(data['cut'][data['split']=='train'])=={0,1}
    assert set(data['cut'][data['split']=='test_cut'])=={2}
    assert set(data['pair'][data['split']=='train']).isdisjoint(data['pair'][data['split']=='test_environment'])
    assert diagnostics['maximum_eigen_residual']<2e-13
