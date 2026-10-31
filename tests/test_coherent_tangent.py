"""Two-sided slopes on complete and incomplete quadrature, nodes and gauge."""
import numpy as np
import pytest

from su2_nqs.vmc.coherent_tangent import (coherent_tangent_diagnostics,
    CoherentTangentPolicy,cross_chain_tangent_gate)
from su2_nqs.vmc.carrier_protection import physical_moments


@pytest.mark.parametrize('complete',(True,False))
def test_two_sided_slopes_match_independent_Rayleigh_and_variance_finite_difference(complete):
    h=np.array([[0,1,.7j],[1,2,.2],[-.7j,.2,3]],complex)
    p=np.array([1,0,.3j]);d=np.array([.2j,-.13,.11+.09j]);hp=h@p;hd=h@d
    # Keep H and Hdelta from full H, but intentionally omit one outer row.
    keep=np.arange(3) if complete else np.array([0,1]);q=np.array([.2,1.9,3.])[keep]
    scale=np.sqrt(q);p,hp,d,hd=[v[keep]*scale for v in (p,hp,d,hd)]
    result=coherent_tangent_diagnostics(p,hp,d,hd,q,3)
    eps=1e-6
    plus=physical_moments(p+eps*d,hp+eps*hd,q,3)
    minus=physical_moments(p-eps*d,hp-eps*hd,q,3)
    assert result['energy_slope_per_site']==pytest.approx((plus['energy_per_site']-minus['energy_per_site'])/(2*eps),abs=2e-10)
    assert result['variance_slope_per_site']==pytest.approx((plus['variance_per_site']-minus['variance_per_site'])/(2*eps),abs=2e-10)
    assert result['centered_orthogonality_scaled']<1e-14
    if complete:
        assert result['hermiticity_quadrature_defect_per_site']<1e-14
    else:
        assert result['hermiticity_quadrature_defect_per_site']>1e-3
        assert abs(result['energy_slope_per_site']-result['one_sided_energy_slope_per_site'])>1e-3


def test_tangent_gauge_and_Q_normalizer_invariance_and_node_opening():
    h=np.array([[0.,1.],[1.,100.]])
    p=np.array([1.,0]);d=np.array([0.,-.1]);hp=h@p;hd=h@d;q=np.ones(2)
    r=coherent_tangent_diagnostics(p,hp,d,hd,q,2)
    assert r['exact_parent_node_records']==1 and r['tangent_at_parent_nodes_norm']>0
    for shift in (2.+3j,-.8+.4j):
        other=coherent_tangent_diagnostics(p,hp,d+shift*p,hd+shift*hp,q*7.3,2)
        for key in ('energy_slope_per_site','variance_slope_per_site','tangent_fs_speed',
                    'forward_coherence_per_site','reverse_coherence_per_site'):
            assert other[key]==pytest.approx(r[key],abs=1e-14)


def test_cross_chain_policy_rejects_one_sided_descent_or_geometry_inflation_without_flipping():
    good=dict(forward_coherence_per_site=-.1,reverse_coherence_per_site=-.2,
              energy_slope_per_site=-.3,tangent_fs_speed=1.)
    policy=CoherentTangentPolicy()
    assert cross_chain_tangent_gate([good,{**good,'tangent_fs_speed':1.9}],policy)['eligible']
    bad={**good,'reverse_coherence_per_site':.02,'energy_slope_per_site':-.08}
    r=cross_chain_tangent_gate([good,bad],policy)
    assert not r['eligible'] and r['checks']['total_descent_each_split']
    assert 'reverse_descent_each_split' in r['rejected_by']
    r=cross_chain_tangent_gate([good,{**good,'tangent_fs_speed':2.1}],policy)
    assert not r['eligible'] and r['rejected_by']==['cross_chain_geometry']
    with pytest.raises(ValueError):cross_chain_tangent_gate([good],policy)
    with pytest.raises(ValueError):CoherentTangentPolicy(maximum_speed_ratio=.5)
    with pytest.raises(ValueError):coherent_tangent_diagnostics([0],[0],[0],[0],[1],1)
    with pytest.raises(ValueError):coherent_tangent_diagnostics([1],[0],[0],[0],[0],1)
