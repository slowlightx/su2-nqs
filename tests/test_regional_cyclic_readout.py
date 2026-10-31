"""Cyclic characters checked against analytic three-/four-spin counting."""
import numpy as np
import pytest

from su2_nqs.benchmarks.regional_su2_readout import (
    regional_schur_columns, su2_cyclic_entanglement_readout,
)


def test_three_spin_cyclic_characters_separate_symmetric_and_doublet_multiplicities():
    r=su2_cyclic_entanglement_readout(np.eye(8)/8,(1,2,0),order=3)
    for q,k,p in [(1,0,0),(1,1,.25),(1,2,.25),(3,0,.5),(3,1,0),(3,2,0)]:
        assert r['sectors'][str(q)][str(k)]['p']==pytest.approx(p,abs=2e-15)


def test_four_spin_cycle_matches_independent_fixed_m_counts():
    r=su2_cyclic_entanglement_readout(np.eye(16)/16,(1,2,3,0),order=4)
    expected={0:[1/16,0,1/16,0],2:[0,3/16,3/16,3/16],4:[5/16,0,0,0]}
    for q,values in expected.items():
        np.testing.assert_allclose([r['sectors'][str(q)][str(k)]['p'] for k in range(4)],values,atol=3e-15)


def test_su2_invariance_does_not_license_spatial_projection_repair():
    u,labels=regional_schur_columns(4)
    v=u[:,labels.index((0,0,0))]; rho=np.outer(v,v.conj())
    with pytest.raises(ValueError,match='no twirl'):
        su2_cyclic_entanglement_readout(rho,(1,2,3,0),order=4)
    with pytest.raises(ValueError,match='specified cyclic order'):
        su2_cyclic_entanglement_readout(np.eye(16)/16,(1,2,3,0),order=3)
    with pytest.raises(ValueError):
        su2_cyclic_entanglement_readout(np.eye(16)/16,(1,2,3,0.0),order=4)
