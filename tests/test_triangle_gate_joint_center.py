"""A light factor-onlyview must reuse the testedfixedwidthcenteralgebra."""
import numpy as np

from projects.clebsch_tree_nqs.workflows.triangle_gate_joint_center import _FactorView
from su2_nqs.operators.su2_sector_merge import ReducedBlockData
from su2_nqs.operators.su2_three_block_center import ThreeBlockSingletOperator


def test_factor_only_view_restores_complex_center_without_H_preparation():
    child=ReducedBlockData(3,{1:2},{1:np.zeros((2,2))})
    environment=ReducedBlockData(6,{0:2,2:2},{q:np.zeros((2,2)) for q in (0,2)})
    channels=tuple((1,a,1,b,q,e) for q in (0,2)
                   for a in range(2) for b in range(2) for e in range(2))
    indices={q:np.array([k for k,c in enumerate(channels) if c[4]==q]) for q in (0,2)}
    view=_FactorView((child,child,environment),channels,len(channels),indices)
    rng=np.random.default_rng(2710614)
    center=rng.normal(size=len(channels))+1j*rng.normal(size=len(channels))
    center/=np.linalg.norm(center)
    maps,root,proof=view.factor(center,widths={0:2,2:2})
    actual=ThreeBlockSingletOperator.encode(view,maps,root)
    np.testing.assert_allclose(actual,center,atol=8e-16)
    for w in maps.values():np.testing.assert_allclose(w.conj().T@w,np.eye(2),atol=8e-16)
    assert proof['no_singular_value_truncation']
    assert not hasattr(view,'pairs')
