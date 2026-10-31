"""Independent finite-population exact oracles for the isolated CT replay."""
from itertools import combinations
import numpy as np

from projects.clebsch_tree_nqs.explorations.ct_attention_control_variate_20261030 import corrected_force, cheap_features
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds


def test_exact_exhaustive_residual_correction_is_unbiased_and_gauge_covariant():
    rng=np.random.default_rng(2610301); o=rng.normal(size=(12,4)); o-=o.mean(0)
    e=rng.normal(size=12); f=rng.normal(size=12); groups=np.repeat(np.arange(6),2)
    reference=2
    # Condition on the known reference chain, enumerate every allowed correction subset.
    estimates=[corrected_force(o,e,f,groups,reference,np.array(ids))
               for ids in combinations([0,1,3,4,5],2)]
    expected=(o*e[:,None]).mean(0)
    np.testing.assert_allclose(np.mean(estimates,0),expected,atol=3e-15)
    rotation=np.linalg.qr(rng.normal(size=(4,4)))[0]
    got=corrected_force(o@rotation,e,f,groups,reference,np.array([0,4]))
    np.testing.assert_allclose(got,corrected_force(o,e,f,groups,reference,np.array([0,4]))@rotation,atol=3e-15)
    # Energy shifts change neither the covariance nor any single corrected estimate.
    np.testing.assert_allclose(corrected_force(o,e+24,f,groups,reference,np.array([0,4])),
                               corrected_force(o,e,f,groups,reference,np.array([0,4])),atol=3e-15)


def test_perfect_surrogate_gives_exact_force_for_every_query_subset():
    rng=np.random.default_rng(2610302); o=rng.normal(size=(10,3)); o-=o.mean(0)
    e=rng.normal(size=10); groups=np.repeat(np.arange(5),2)
    reference=1; prediction=e-e[groups==reference].mean()
    for ids in combinations([0,2,3,4],2):
        np.testing.assert_allclose(corrected_force(o,e,prediction,groups,reference,np.array(ids)),
                                   (o*e[:,None]).mean(0),atol=3e-15)


def test_feature_law_has_no_H_label_input_and_common_J2_coordinates():
    bits=np.random.default_rng(2610303).integers(2,size=(7,36))
    data=dict(bits=bits,token_features=np.zeros((24,46)))
    for coupling in (0.,.125):
        tokens,diagonal=cheap_features(data,coupling)
        assert tokens.shape==(7,4,63) and diagonal.shape==(7,216)
        expected=sum(j*(bits[:,i]-.5)*(bits[:,k]-.5)
                     for i,k,j in triangular_j1j2_bonds(6,6,j2=coupling,periodic=True))
        np.testing.assert_allclose(diagonal.sum(1),expected,atol=3e-15)


def test_unqueried_H_labels_cannot_change_corrected_estimate():
    rng=np.random.default_rng(2610304); o=rng.normal(size=(12,4)); o-=o.mean(0)
    e=rng.normal(size=12); f=rng.normal(size=12); groups=np.repeat(np.arange(6),2)
    changed=e.copy(); changed[~np.isin(groups,[0,1,4])]+=100*rng.normal(size=6)
    np.testing.assert_allclose(corrected_force(o,e,f,groups,0,np.asarray([1,4])),
                               corrected_force(o,changed,f,groups,0,np.asarray([1,4])),atol=3e-15)
